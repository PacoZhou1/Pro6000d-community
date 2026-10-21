#!/usr/bin/env python3
"""Offline exact NVTX -> CUDA API -> GPU kernel attribution for eager vLLM.

The input is an existing Nsight Systems SQLite export with CUDA and NVTX
tables. GEMM_ATTRIB NVTX range labels come from attribution_patch.py. No
kernel is assigned by name or timestamp overlap. Normal CUDA Graph replay
kernel-time shares must be reported separately from this eager diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import pathlib
import re
import sqlite3
from bisect import bisect_left
from collections import defaultdict


LABEL = re.compile(
    r"^GEMM_ATTRIB:(?P<kind>[^:]+):(?P<prefix>.*):M(?P<M>\d+):N(?P<N>\d+):K(?P<K>\d+)$"
)
STAGE = re.compile(r"^(?:execute_context|generation)", re.I)
# A generic CUTLASS token also appears in FlashInfer GDN delta-rule kernels;
# require an explicit GEMM/matmul or cuBLAS name to count a BF16 GEMM.
GEMM = re.compile(r"gemm|cublas|matmul|\bmm\b", re.I)
BF16 = re.compile(r"bf16|bfloat16", re.I)
FP8 = re.compile(r"fp8|float8", re.I)
PID_MASK = ~0xFFFFFF


def tables(db):
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def rows(db, table, fields):
    present = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
    cols = [field for field in fields if field in present]
    if not cols:
        return []
    query = "SELECT " + ",".join(f'"{col}"' for col in cols) + f' FROM "{table}"'
    return [dict(zip(cols, values)) for values in db.execute(query)]


def span(row):
    return isinstance(row.get("start"), int) and isinstance(row.get("end"), int) and row["end"] >= row["start"]


def pid_of_tid(tid):
    return tid & PID_MASK if isinstance(tid, int) else None


def resolve_name(row, strings):
    return next((strings[row[key]] for key in ("demangledName", "shortName", "mangledName")
                 if row.get(key) in strings), "<unresolved kernel name>")


def shortest_matches(ranges, points):
    """Yield (point payload, shortest enclosing range payload) by time sweep.

    ``ranges`` are (start, end, payload) and ``points`` are (time, payload).
    Endpoints are inclusive. A heap by end retires expired intervals, while a
    second heap by (width, start, end) selects the shortest *live* interval.
    Each range enters/leaves once: O((R + Q) log R), not one range scan per
    point. Equal-width intervals choose earliest start deterministically.
    """
    ordered = sorted(ranges, key=lambda row: (row[0], row[1]))
    by_end = []
    by_width = []
    live = set()
    position = 0
    for when, payload in sorted(points, key=lambda row: row[0]):
        while position < len(ordered) and ordered[position][0] <= when:
            start, end, _ = ordered[position]
            serial = position
            heapq.heappush(by_end, (end, serial))
            heapq.heappush(by_width, (end - start, start, end, serial))
            live.add(serial)
            position += 1
        while by_end and by_end[0][0] < when:
            _, serial = heapq.heappop(by_end)
            live.discard(serial)
        while by_width and by_width[0][3] not in live:
            heapq.heappop(by_width)
        match = ordered[by_width[0][3]][2] if by_width else None
        yield payload, match


def covering_matches(ranges, spans):
    """Yield shortest range fully covering each (start, end) query.

    Sort queries by start. Insert ranges once their start is reached; a suffix
    minimum over compressed range ends selects only those whose end also
    covers the query end. O((R + Q) log R), including crossing/nested ranges.
    """
    ordered = sorted(ranges, key=lambda row: (row[0], row[1]))
    ends = sorted({end for _, end, _ in ordered})
    size = 1
    while size < len(ends):
        size *= 2
    tree = [None] * (2 * size)

    def best(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return a if a[:4] <= b[:4] else b

    def add(serial):
        start, end, payload = ordered[serial]
        index = size + bisect_left(ends, end)
        candidate = (end - start, start, end, serial, payload)
        tree[index] = best(tree[index], candidate)
        index //= 2
        while index:
            tree[index] = best(tree[2 * index], tree[2 * index + 1])
            index //= 2

    def suffix_min(first):
        left, right = size + first, size + len(ends)
        result = None
        while left < right:
            if left & 1:
                result = best(result, tree[left])
                left += 1
            if right & 1:
                right -= 1
                result = best(result, tree[right])
            left //= 2
            right //= 2
        return result

    position = 0
    for query_start, query_end, query_payload in sorted(spans, key=lambda row: row[0]):
        while position < len(ordered) and ordered[position][0] <= query_start:
            add(position)
            position += 1
        first = bisect_left(ends, query_end)
        hit = suffix_min(first) if first < len(ends) else None
        yield query_payload, hit[4] if hit else None


def summarize(records, duration_key="duration_ns", limit=0):
    result = []
    for name, values in records.items():
        count, total = values
        result.append({"name": name, "count": count, duration_key: total,
                       "mean_duration_ns": total / count})
    result.sort(key=lambda item: (-item[duration_key], item["name"]))
    return result[:limit] if limit else result


def analyze(db, label, selected_pids=None, top=20):
    names = tables(db)
    required = {"CUPTI_ACTIVITY_KIND_KERNEL", "NVTX_EVENTS", "StringIds"}
    if not required.issubset(names):
        raise ValueError("SQLite must contain CUPTI kernels, NVTX_EVENTS, and StringIds")
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    processes = rows(db, "PROCESSES", ("globalPid", "pid", "name")) if "PROCESSES" in names else []
    allowed = None
    if selected_pids:
        allowed = {row["globalPid"] for row in processes if row.get("pid") in selected_pids}
        if not allowed:
            raise ValueError("selected OS PID absent from PROCESSES")

    kernel_fields = ("start", "end", "globalPid", "deviceId", "contextId", "streamId",
                     "correlationId", "demangledName", "shortName", "mangledName")
    kernels = []
    for row in rows(db, "CUPTI_ACTIVITY_KIND_KERNEL", kernel_fields):
        if not span(row) or allowed is not None and row.get("globalPid") not in allowed:
            continue
        row["name"] = resolve_name(row, strings)
        kernels.append(row)

    stage_by_tid = defaultdict(list)
    attribution_by_tid = defaultdict(list)
    attribution = []
    malformed_labels = []
    for row in rows(db, "NVTX_EVENTS", ("eventId", "start", "end", "globalTid",
                                       "text", "textId", "eventType")):
        if not span(row) or allowed is not None and pid_of_tid(row.get("globalTid")) not in allowed:
            continue
        text = row.get("text") or strings.get(row.get("textId"), "")
        tid = row.get("globalTid")
        if STAGE.match(text):
            stage_by_tid[tid].append((row["start"], row["end"], text))
        if not text.startswith("GEMM_ATTRIB:"):
            continue
        match = LABEL.match(text)
        if not match:
            malformed_labels.append({"eventId": row.get("eventId"), "text": text})
            continue
        item = {"eventId": row.get("eventId"), "start": row["start"],
                "end": row["end"], "globalTid": tid, "text": text,
                "kind": match["kind"], "module_prefix": match["prefix"],
                "M": int(match["M"]), "N": int(match["N"]), "K": int(match["K"]),
                "api_record_count": 0, "kernel_count": 0, "kernel_sum_duration_ns": 0}
        attribution_by_tid[tid].append((item["start"], item["end"], len(attribution)))
        attribution.append(item)
    for item in attribution:
        item["enclosing_stage"] = None
    for tid, gemm_ranges in attribution_by_tid.items():
        spans = [(start, end, index) for start, end, index in gemm_ranges]
        for index, stage_text in covering_matches(stage_by_tid.get(tid, []), spans):
            attribution[index]["enclosing_stage"] = stage_text

    api_tables = [table for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
                  if table in names]
    correlation_to_ranges = defaultdict(set)
    api_record_count = 0
    api_records_in_attribution_ranges = 0
    api_keys = set()
    api_keys_in_ranges = set()
    for table in api_tables:
        api_by_tid = defaultdict(list)
        for row in rows(db, table, ("start", "end", "globalTid", "correlationId", "nameId")):
            if not span(row) or row.get("correlationId") is None:
                continue
            tid = row.get("globalTid")
            global_pid = pid_of_tid(tid)
            if allowed is not None and global_pid not in allowed:
                continue
            api_record_count += 1
            api_keys.add((global_pid, row["correlationId"]))
            api_by_tid[tid].append(row)
        for tid, events in api_by_tid.items():
            ranges = attribution_by_tid.get(tid, [])
            if not ranges:
                continue
            for row, index in shortest_matches(ranges, ((event["start"], event) for event in events)):
                if index is None:
                    continue
                api_records_in_attribution_ranges += 1
                attribution[index]["api_record_count"] += 1
                key = (pid_of_tid(tid), row["correlationId"])
                api_keys_in_ranges.add(key)
                correlation_to_ranges[key].add(index)

    ambiguous = {key for key, ids in correlation_to_ranges.items() if len(ids) > 1}
    groups = {}
    assigned_count = assigned_sum = 0
    bf16_name_count = bf16_name_sum = 0
    bf16_assigned_count = bf16_assigned_sum = 0
    bf16_remaining_names = defaultdict(lambda: [0, 0])
    for kernel in kernels:
        duration = kernel["end"] - kernel["start"]
        name = kernel["name"]
        is_bf16_gemm = bool(GEMM.search(name) and BF16.search(name) and not FP8.search(name))
        if is_bf16_gemm:
            bf16_name_count += 1
            bf16_name_sum += duration
        matches = correlation_to_ranges.get((kernel.get("globalPid"), kernel.get("correlationId")), set())
        if len(matches) != 1:
            if is_bf16_gemm:
                stats = bf16_remaining_names[name]
                stats[0] += 1
                stats[1] += duration
            continue
        index = next(iter(matches))
        record = attribution[index]
        record["kernel_count"] += 1
        record["kernel_sum_duration_ns"] += duration
        assigned_count += 1
        assigned_sum += duration
        if is_bf16_gemm:
            bf16_assigned_count += 1
            bf16_assigned_sum += duration
        key = (record["kind"], record["module_prefix"], record["M"], record["N"],
               record["K"], record["enclosing_stage"])
        if key not in groups:
            groups[key] = {"kind": record["kind"], "module_prefix": record["module_prefix"],
                           "M": record["M"], "N": record["N"], "K": record["K"],
                           "enclosing_stage": record["enclosing_stage"],
                           "nvtx_range_count": 0, "cuda_api_record_count": 0,
                           "kernel_count": 0, "kernel_sum_duration_ns": 0,
                           "bf16_named_gemm_kernel_count": 0,
                           "bf16_named_gemm_sum_duration_ns": 0,
                           "raw_names": defaultdict(lambda: [0, 0])}
        group = groups[key]
        group["kernel_count"] += 1
        group["kernel_sum_duration_ns"] += duration
        if is_bf16_gemm:
            group["bf16_named_gemm_kernel_count"] += 1
            group["bf16_named_gemm_sum_duration_ns"] += duration
        stats = group["raw_names"][name]
        stats[0] += 1
        stats[1] += duration
    for record in attribution:
        key = (record["kind"], record["module_prefix"], record["M"], record["N"],
               record["K"], record["enclosing_stage"])
        if key not in groups:
            groups[key] = {"kind": record["kind"], "module_prefix": record["module_prefix"],
                           "M": record["M"], "N": record["N"], "K": record["K"],
                           "enclosing_stage": record["enclosing_stage"],
                           "nvtx_range_count": 0, "cuda_api_record_count": 0,
                           "kernel_count": 0, "kernel_sum_duration_ns": 0,
                           "bf16_named_gemm_kernel_count": 0,
                           "bf16_named_gemm_sum_duration_ns": 0,
                           "raw_names": defaultdict(lambda: [0, 0])}
        groups[key]["nvtx_range_count"] += 1
        groups[key]["cuda_api_record_count"] += record["api_record_count"]
    total_sum = sum(row["end"] - row["start"] for row in kernels)
    group_rows = list(groups.values())
    for group in group_rows:
        group["raw_kernel_ranking"] = summarize(group.pop("raw_names"), limit=top)
        group["share_of_all_gpu_kernel_sum"] = (
            group["kernel_sum_duration_ns"] / total_sum if total_sum else None
        )
    group_rows.sort(key=lambda item: (-item["kernel_sum_duration_ns"], item["module_prefix"], item["M"]))
    return {
        "label": label,
        "scope": "eager diagnostic attribution; not normal CUDA Graph performance",
        "unit": "ns",
        "available_processes": processes,
        "selected_os_pids": selected_pids,
        "selected_kernel_global_pids": sorted({row.get("globalPid") for row in kernels if row.get("globalPid") is not None}),
        "cuda_api_tables": api_tables,
        "cuda_api_record_count_with_correlation": api_record_count,
        "cuda_api_unique_process_correlation_key_count": len(api_keys),
        "cuda_api_record_count_inside_gemm_nvtx": api_records_in_attribution_ranges,
        "cuda_api_unique_process_correlation_key_count_inside_gemm_nvtx": len(api_keys_in_ranges),
        "gemm_nvtx_range_count": len(attribution),
        "malformed_gemm_labels": malformed_labels,
        "ambiguous_correlation_key_count": len(ambiguous),
        "gpu_kernel_count": len(kernels),
        "gpu_kernel_sum_duration_ns": total_sum,
        "attributed_kernel_count": assigned_count,
        "attributed_kernel_sum_duration_ns": assigned_sum,
        "attributed_share_of_all_gpu_kernel_sum": assigned_sum / total_sum if total_sum else None,
        "unattributed_kernel_count": len(kernels) - assigned_count,
        "unattributed_kernel_sum_duration_ns": total_sum - assigned_sum,
        "bf16_named_gemm": {
            "name_rule": "explicit GEMM/cuBLAS/matmul and BF16/bfloat16, excluding FP8/float8; generic CUTLASS name alone is insufficient",
            "all_kernel_count": bf16_name_count,
            "all_kernel_sum_duration_ns": bf16_name_sum,
            "all_share_of_gpu_kernel_sum": bf16_name_sum / total_sum if total_sum else None,
            "attributed_kernel_count": bf16_assigned_count,
            "attributed_kernel_sum_duration_ns": bf16_assigned_sum,
            "unattributed_kernel_count": bf16_name_count - bf16_assigned_count,
            "unattributed_kernel_sum_duration_ns": bf16_name_sum - bf16_assigned_sum,
            "unattributed_raw_kernel_ranking": summarize(bf16_remaining_names, limit=top),
        },
        "groups": group_rows,
        "method": "same globalTid NVTX range contains CUDA runtime/driver API start; exact (globalPid,correlationId) joins API to GPU kernel; shortest enclosing GEMM_ATTRIB range wins; stage only when same-tid stage fully covers GEMM NVTX start and end",
        "limits": "Kernel time is eager diagnostic GPU duration sum, not normal-mode latency or module contribution; unmatched and ambiguous correlation IDs remain unassigned. BF16/FP8 classification is inferred from raw kernel name, not target/draft ownership.",
    }


def self_test():
    """Arithmetic/attribution fixtures only; no measured model or GPU data."""
    db = sqlite3.connect(":memory:")
    for ddl in (
        "CREATE TABLE StringIds(id INTEGER,value TEXT)",
        "CREATE TABLE PROCESSES(globalPid INTEGER,pid INTEGER,name TEXT)",
        "CREATE TABLE NVTX_EVENTS(eventId INTEGER,start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER,eventType INTEGER)",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER)",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER)",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER,shortName INTEGER,mangledName INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER)",
    ):
        db.execute(ddl)
    pid1, pid2 = 0x1000048000000, 0x1000049000000
    tid1, tid2 = pid1 + 72, pid2 + 88
    db.executemany("INSERT INTO StringIds VALUES (?,?)", [
        (1, "void cutlass::Kernel2<gemm_bf16>()"), (2, "cudaLaunchKernel")])
    db.executemany("INSERT INTO PROCESSES VALUES (?,?,?)", [
        (pid1, 72, "synthetic-a"), (pid2, 88, "synthetic-b")])
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?,?,?)", [
        (1, 0, 1000, tid1, "execute_context_0(0)_generation_0(0)", None, 59),
        (2, 100, 800, tid1, "generation_inner", None, 59),
        (3, 150, 700, tid1, "GEMM_ATTRIB:unquantized_linear:outer:M1:N10:K4", None, 59),
        (4, 200, 400, tid1, "GEMM_ATTRIB:unquantized_embedding_gemm:inner:M1:N20:K4", None, 59),
        (5, 100, 500, tid2, "GEMM_ATTRIB:unquantized_linear:other_pid:M2:N30:K4", None, 59),
    ])
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)", [
        (250, 260, tid1, 7, 2), (600, 610, tid1, 8, 2),
        (251, 261, tid1, 9, 2), (601, 611, tid1, 9, 2),
    ])
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (?,?,?,?,?)",
               (150, 160, tid2, 7, 2))
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (260, 360, pid1, 7, 1, None, None, 0, 0, 0),
        (610, 660, pid1, 8, 1, None, None, 0, 0, 0),
        (170, 250, pid2, 7, 1, None, None, 0, 0, 0),
        (700, 720, pid1, 9, 1, None, None, 0, 0, 0),
    ])
    result = analyze(db, "synthetic-self-test")
    selected = analyze(db, "synthetic-selected-pid", selected_pids=[72])
    db.close()
    assert result["attributed_kernel_count"] == 3
    assert result["attributed_kernel_sum_duration_ns"] == 230
    assert result["unattributed_kernel_count"] == 1
    assert result["unattributed_kernel_sum_duration_ns"] == 20
    assert result["ambiguous_correlation_key_count"] == 1
    assert result["bf16_named_gemm"]["all_kernel_count"] == 4
    by_prefix = {group["module_prefix"]: group for group in result["groups"]}
    assert by_prefix["inner"]["kernel_sum_duration_ns"] == 100
    assert by_prefix["inner"]["enclosing_stage"] == "generation_inner"
    assert by_prefix["outer"]["kernel_sum_duration_ns"] == 50
    assert by_prefix["outer"]["enclosing_stage"] == "generation_inner"
    assert by_prefix["other_pid"]["kernel_sum_duration_ns"] == 80
    assert selected["attributed_kernel_count"] == 2
    assert all(group["module_prefix"] != "other_pid" for group in selected["groups"])
    # Dense nested intervals and a sliding expiry window exercise the sweep
    # without creating a 10,000 x 10,000 scan. These are algorithm fixtures.
    n = 10000
    nested = [(i, 2 * n - i, i) for i in range(n)]
    assert all(hit == n - 1 for _, hit in shortest_matches(
        nested, ((n, query) for query in range(n))))
    sliding = [(i, i + 7, i) for i in range(n)]
    assert all(hit == query for query, hit in shortest_matches(
        sliding, ((query + 7, query) for query in range(n))))
    crossing = [(0, 1000, "full_outer"), (100, 300, "start_only_inner")]
    matches = dict(covering_matches(crossing, [
        (150, 700, "long_gemm"), (150, 250, "short_gemm"),
        (1100, 1200, "outside")]))
    assert matches == {"long_gemm": "full_outer",
                       "short_gemm": "start_only_inner", "outside": None}
    assert all(hit == n - 1 for _, hit in covering_matches(
        nested, ((n, n + 1, query) for query in range(n))))
    return {"self_test": "passed", "cases": [
        "nested_GEMM_and_stage_ranges", "same_correlation_id_different_pids",
        "ambiguous_correlation_id_same_pid", "selected_pid_isolation",
        "dense_nested_intervals_10000",
        "sliding_expiry_intervals_10000", "full_span_stage_cover",
        "dense_full_span_cover_10000"], "synthetic_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=pathlib.Path, nargs="?")
    parser.add_argument("--label")
    parser.add_argument("--pid", type=int, action="append", help="OS PID; repeat if needed")
    parser.add_argument("--top", type=int, default=20, help="Raw-name limit per group; 0 means all")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic parser/interval fixtures only")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if args.sqlite is None or not args.label:
        parser.error("sqlite and --label are required unless --self-test is used")
    if not args.sqlite.is_file():
        parser.error("SQLite file missing")
    if args.top < 0:
        parser.error("--top must be nonnegative")
    db = sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite integrity_check failed: {integrity}")
        result = analyze(db, args.label, args.pid, args.top)
    finally:
        db.close()
    result["source_sqlite"] = str(args.sqlite)
    result["source_sqlite_integrity_check"] = integrity
    digest = hashlib.sha256()
    with args.sqlite.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    result["source_sqlite_sha256"] = digest.hexdigest()
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
