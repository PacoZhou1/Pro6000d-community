"""Analyze an exported Nsight Systems SQLite file offline.

All CUPTI timestamps/durations are nanoseconds. A request label is supplied by
the caller; this script never infers prefill/decode or draft/target from names.
NVTX stage attribution requires an exact CUDA API correlation ID whose launch
starts within a same-thread NVTX range. Name-based kernel groups are hypotheses.

Example: python3 analyze_nsys.py raw/capture.sqlite --label mtp4-decode \
         --pid 1234 --output mtp4-decode-analysis.json
"""

import argparse
import heapq
import json
import pathlib
import re
import sqlite3
from collections import Counter, defaultdict


NAME_RULES = (
    ("norm+quant", "fused RMS norm and FP8 activation quantization", r"fused.?add.?rms.?norm.*(?:fp8.?quant|per.?token.?group.?quant)"),
    ("norm/elementwise", "fused RMS norm (may contain copies)", r"fused.?add.?rms.?norm"),
    ("activation/quant", "activation or per-token/per-block FP8 quantization", r"per.?token.?group.?(?:fp8.?)?quant|per.?token.?group.?quant.?8bit|silu.?and.?mul.?per.?block.?quant"),
    ("GDN convolution", "causal conv1d / post-conv kernel name", r"causal.?conv1d|post.?conv"),
    ("GDN", "gdn/gated-delta/delta-rule/mamba/linear-attn", r"gdn|gated.?delta|delta.?net|delta.?rule|mamba|linear.?attn|recurrent"),
    ("attention", "attention/flash-attn/XQA/FMHA/kernel_mha", r"attention|flash.?attn|\bxqa\b|\bfmha\b|paged.?attn|kernel_mha"),
    ("sampling", "sampling/top-k/argmax", r"sampl|top.?k|top.?p|multinomial|argmax"),
    ("state-copy", "copy/scatter/gather/cache-update", r"copy|scatter|gather|index.?select|cache.?update|state.?update|swap"),
    ("GEMM/linear", "GEMM/matmul/CUTLASS/cuBLAS/linear", r"gemm|matmul|cutlass|cublas|(?:^|[_:.])mm(?:[_:.0-9]|$)|linear"),
    ("norm/elementwise", "norm/activation/pointwise/softmax", r"norm|elementwise|pointwise|softmax|silu|gelu|swiglu|activation|fused.?add"),
)
RULES = [(category, text, re.compile(pattern, re.I)) for category, text, pattern in NAME_RULES]
STAGE_PREFIX = re.compile(r"^(execute_context|generation)", re.I)


def tables(connection):
    return {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def columns(connection, name):
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{name}")')}


def rows(connection, name, wanted):
    present = columns(connection, name)
    names = [field for field in wanted if field in present]
    if not names:
        return []
    query = "SELECT " + ", ".join('"' + field + '"' for field in names) + f' FROM "{name}"'
    return [dict(zip(names, item)) for item in connection.execute(query)]


def valid_span(row):
    return isinstance(row.get("start"), int) and isinstance(row.get("end"), int) and row["end"] >= row["start"]


def union_ns(events):
    spans = sorted((row["start"], row["end"]) for row in events)
    if not spans:
        return 0
    first, last = spans[0]
    total = 0
    for start, end in spans[1:]:
        if start > last:
            total += last - first
            first, last = start, end
        else:
            last = max(last, end)
    return total + last - first


def category_for(name):
    for category, rule, pattern in RULES:
        if pattern.search(name):
            return category, rule
    return "other", "no name rule matched"


def ranking(events, categorized=False):
    totals = defaultdict(lambda: [0, 0])
    for row in events:
        item = totals[row["name"]]
        item[0] += 1
        item[1] += row["end"] - row["start"]
    result = []
    for name, (count, total) in totals.items():
        item = {"name": name, "count": count, "sum_duration_ns": total,
                "mean_duration_ns": total / count}
        if categorized:
            group, rule = category_for(name)
            item.update({"category": group, "category_rule": rule,
                         "category_inferred_from_name": True})
        result.append(item)
    return sorted(result, key=lambda item: (-item["sum_duration_ns"], item["name"]))


def process_for_tid(global_tid):
    # Nsight's serialized globalTid uses its low 24 bits for the thread ID;
    # tool_smoke.sqlite confirms the remaining bits equal PROCESSES.globalPid.
    return (global_tid & ~0xFFFFFF) if isinstance(global_tid, int) else None


def attach_stages(stages, runtime, kernels):
    """Map GPU kernels via same-thread NVTX -> CUDA API -> correlation ID."""
    stages_by_tid = defaultdict(list)
    for index, stage in enumerate(stages):
        stages_by_tid[stage["globalTid"]].append((stage["start"], stage["end"], index))
    runtime_by_tid = defaultdict(list)
    for row in runtime:
        if row.get("correlationId") is not None:
            runtime_by_tid[row.get("globalTid")].append(row)
    correlation_stages = defaultdict(set)
    for tid, api_events in runtime_by_tid.items():
        ranges = sorted(stages_by_tid.get(tid, []))
        if not ranges:
            continue
        active = []  # shortest enclosing range first; nested ranges choose innermost.
        position = 0
        for api in sorted(api_events, key=lambda row: row["start"]):
            when = api["start"]
            while position < len(ranges) and ranges[position][0] <= when:
                start, end, index = ranges[position]
                heapq.heappush(active, (end - start, end, start, index))
                position += 1
            while active and active[0][1] < when:
                heapq.heappop(active)
            if active:
                index = active[0][3]
                key = (process_for_tid(tid), api["correlationId"])
                correlation_stages[key].add(index)
                stages[index]["matched_cuda_api_count"] += 1
    ambiguous = {key for key, matches in correlation_stages.items() if len(matches) != 1}
    matched = 0
    for kernel in kernels:
        key = (kernel.get("globalPid"), kernel.get("correlationId"))
        choices = correlation_stages.get(key, set())
        if len(choices) == 1:
            index = next(iter(choices))
            kernel["nvtx_stage_index"] = index
            stages[index]["matched_kernel_count"] += 1
            stages[index]["matched_kernel_sum_ns"] += kernel["end"] - kernel["start"]
            matched += 1
    return {"method": "same globalTid; CUDA API start inside NVTX range; matching (globalPid, correlationId) on GPU kernel; shortest enclosing NVTX range chosen",
            "matched_kernel_count": matched,
            "unmatched_kernel_count": len(kernels) - matched,
            "ambiguous_correlation_keys": len(ambiguous),
            "caveat": "CUDA Graph replay or asynchronous launches may lack usable API correlation. Unmatched kernels are not assigned by timestamp overlap."}


def analyze(connection, label, pid=None, global_pid=None, device=None, context=None, top=0):
    names = tables(connection)
    required = {"CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
    if not required.issubset(names):
        raise ValueError("SQLite lacks CUPTI kernel or StringIds table")
    strings = dict(connection.execute("SELECT id, value FROM StringIds"))
    processes = rows(connection, "PROCESSES", ("globalPid", "pid", "name")) if "PROCESSES" in names else []
    selected_global_pids = None
    if pid is not None or global_pid is not None:
        selected_global_pids = {row["globalPid"] for row in processes if row.get("pid") in (pid or [])}
        selected_global_pids.update(global_pid or [])
        if not selected_global_pids:
            raise ValueError("selected PID absent; inspect available_processes")
    def selected(row, gpu=False):
        if selected_global_pids is not None and row.get("globalPid") not in selected_global_pids:
            return False
        if gpu and device is not None and row.get("deviceId") not in device:
            return False
        if gpu and context is not None and row.get("contextId") not in context:
            return False
        return True
    kernel_fields = ("start", "end", "deviceId", "contextId", "streamId", "correlationId",
                     "globalPid", "demangledName", "shortName", "mangledName", "gridX", "gridY",
                     "gridZ", "blockX", "blockY", "blockZ")
    kernels = []
    for row in rows(connection, "CUPTI_ACTIVITY_KIND_KERNEL", kernel_fields):
        if not valid_span(row) or not selected(row, gpu=True):
            continue
        row["name"] = next((strings[value] for field in ("demangledName", "shortName", "mangledName")
                            if (value := row.get(field)) in strings),
                           "<unresolved kernel name>")
        kernels.append(row)
    memcpy = []
    for table in sorted(name for name in names if name.startswith("CUPTI_ACTIVITY_KIND_MEMCPY")):
        for row in rows(connection, table, ("start", "end", "deviceId", "contextId", "streamId",
                                            "correlationId", "globalPid", "bytes", "copyKind", "srcKind", "dstKind")):
            if valid_span(row) and selected(row, gpu=True):
                row["name"] = table + ":copyKind=" + str(row.get("copyKind"))
                row["source_table"] = table
                memcpy.append(row)
    memset = []
    if "CUPTI_ACTIVITY_KIND_MEMSET" in names:
        for row in rows(connection, "CUPTI_ACTIVITY_KIND_MEMSET", ("start", "end", "deviceId",
                      "contextId", "streamId", "correlationId", "globalPid", "bytes", "value")):
            if valid_span(row) and selected(row, gpu=True):
                row["name"] = "GPU memset"
                memset.append(row)
    runtime = []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in names:
        for row in rows(connection, "CUPTI_ACTIVITY_KIND_RUNTIME", ("start", "end", "globalTid",
                      "correlationId", "nameId")):
            row["globalPid"] = process_for_tid(row.get("globalTid"))
            if valid_span(row) and selected(row):
                row["name"] = strings.get(row.get("nameId"), "<unresolved CUDA API name>")
                runtime.append(row)
    stages = []
    nvtx_total = 0
    if "NVTX_EVENTS" in names:
        for row in rows(connection, "NVTX_EVENTS", ("eventId", "start", "end", "globalTid",
                      "endGlobalTid", "text", "textId", "eventType")):
            nvtx_total += 1
            row["globalPid"] = process_for_tid(row.get("globalTid"))
            text = row.get("text") or strings.get(row.get("textId"), "")
            if valid_span(row) and selected(row) and STAGE_PREFIX.match(text):
                row["text"] = text
                row["matched_cuda_api_count"] = 0
                row["matched_kernel_count"] = 0
                row["matched_kernel_sum_ns"] = 0
                stages.append(row)
    correlation = attach_stages(stages, runtime, kernels)
    kernel_sum = sum(row["end"] - row["start"] for row in kernels)
    raw_ranking = ranking(kernels, categorized=True)
    category_totals = defaultdict(lambda: [0, 0])
    for row in kernels:
        category, _ = category_for(row["name"])
        category_totals[category][0] += 1
        category_totals[category][1] += row["end"] - row["start"]
    categories = [{"category": category, "count": values[0], "sum_duration_ns": values[1],
                   "share_of_all_selected_kernel_sum": values[1] / kernel_sum if kernel_sum else None,
                   "inferred_from_name": True}
                  for category, values in category_totals.items()]
    categories.sort(key=lambda row: (-row["sum_duration_ns"], row["category"]))
    stage_groups = defaultdict(lambda: [0, 0, 0, 0])
    for stage in stages:
        group = stage_groups[stage["text"]]
        group[0] += 1
        group[1] += stage["end"] - stage["start"]
        group[2] += stage["matched_kernel_count"]
        group[3] += stage["matched_kernel_sum_ns"]
    stage_summary = [{"text": text, "range_count": values[0], "cpu_nvtx_range_sum_ns": values[1],
                      "api_correlated_kernel_count": values[2],
                      "api_correlated_kernel_sum_ns": values[3],
                      "share_of_all_selected_kernel_sum": values[3] / kernel_sum if kernel_sum else None}
                     for text, values in stage_groups.items()]
    stage_summary.sort(key=lambda row: (-row["api_correlated_kernel_sum_ns"], row["text"]))
    analysis = rows(connection, "ANALYSIS_DETAILS", ("duration", "startTime", "stopTime")) if "ANALYSIS_DETAILS" in names else []
    gpu_info = rows(connection, "TARGET_INFO_GPU", ("id", "name", "chipName", "computeMajor", "computeMinor")) if "TARGET_INFO_GPU" in names else []
    all_activity = kernels + memcpy + memset
    activity_span = {"first_start_ns": min(row["start"] for row in all_activity),
                     "last_end_ns": max(row["end"] for row in all_activity),
                     "span_ns": max(row["end"] for row in all_activity) - min(row["start"] for row in all_activity)} if all_activity else None
    return {
        "label": label, "unit": "nanoseconds",
        "metadata": {"schema_tables_available": sorted(names), "available_processes": processes,
                     "available_gpu_devices": gpu_info, "analysis_details": analysis},
        "selection": {"os_pids": pid, "global_pids": global_pid, "device_ids": device,
                      "context_ids": context, "selected_kernel_processes": sorted({row.get("globalPid") for row in kernels if row.get("globalPid") is not None})},
        "gpu_activity_event_span": activity_span,
        "gpu_kernel_count": len(kernels),
        "gpu_kernel_sum_duration_ns": kernel_sum,
        "gpu_kernel_busy_union_ns": union_ns(kernels),
        "gpu_memcpy_count": len(memcpy),
        "gpu_memcpy_sum_duration_ns": sum(row["end"] - row["start"] for row in memcpy),
        "gpu_memcpy_bytes_sum": sum(row.get("bytes") or 0 for row in memcpy),
        "gpu_memset_count": len(memset),
        "gpu_memset_sum_duration_ns": sum(row["end"] - row["start"] for row in memset),
        "gpu_activity_busy_union_ns": union_ns(all_activity),
        "duration_denominators": {"kernel_category_share": "all selected GPU kernel duration sum (not wall time)",
                                  "nvtx_stage_kernel_share": "all selected GPU kernel duration sum; only API-correlated kernels enter numerator",
                                  "busy_union": "merged intervals, not duration sum; concurrent streams overlap",
                                  "analysis_duration": "capture duration from ANALYSIS_DETAILS, not HTTP request latency"},
        "kernel_category_basis": "Raw kernel names matched by ordered regex; inferred category, not confirmed model module or draft/target attribution.",
        "kernel_categories": categories,
        "raw_kernel_ranking": raw_ranking[:top] if top else raw_ranking,
        "raw_kernel_name_count": len(raw_ranking),
        "unrecognized_kernel_names": [row for row in raw_ranking if row["category"] == "other"],
        "gpu_memcpy_ranking": ranking(memcpy),
        "cuda_runtime_api_ranking": ranking(runtime),
        "cuda_runtime_api_count": len(runtime),
        "nvtx_stage_evidence": {"table_present": "NVTX_EVENTS" in names,
                                "all_nvtx_event_count": nvtx_total,
                                "matching_range_count": len(stages),
                                "name_filter": "^(execute_context|generation), case-insensitive",
                                "ranges": stages,
                                "summary_by_exact_text": stage_summary,
                                "gpu_kernel_api_correlation": correlation,
                                "interpretation": "NVTX labels are CPU ranges. API correlation can associate launched GPU kernels; these labels alone do not identify prefill/decode or draft/target."},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=pathlib.Path, help="Local exported .sqlite file")
    parser.add_argument("--label", required=True, help="Explicit request/capture label")
    parser.add_argument("--pid", type=int, action="append", help="OS PID; repeat to select several")
    parser.add_argument("--global-pid", type=int, action="append", help="Nsight serialized globalPid")
    parser.add_argument("--device", type=int, action="append", help="GPU deviceId; repeat if needed")
    parser.add_argument("--context", type=int, action="append", help="CUDA contextId; repeat if needed")
    parser.add_argument("--top", type=int, default=0, help="Raw kernel ranking limit; 0 means all")
    parser.add_argument("--output", type=pathlib.Path, help="Save analysis JSON")
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be nonnegative")
    if not args.sqlite.is_file():
        parser.error("SQLite file does not exist")
    connection = sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        report = analyze(connection, args.label, args.pid, args.global_pid,
                         args.device, args.context, args.top)
    finally:
        connection.close()
    report["source_sqlite"] = str(args.sqlite)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
