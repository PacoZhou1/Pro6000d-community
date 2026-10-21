#!/usr/bin/env python3
"""Validate and summarize paired CUTLASS/B12X inference and GEMM evidence.

Reads local JSONL only. Writes ab-summary.json only when both arms, all nine
matched formal requests, the quality/open checks, and the microprobe exist.
Small n=3 ranges are descriptive; output hashes are disclosed, not assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


CASES = ("prefill", "decode", "thinking")
ARMS = ("cutlass-control", "b12x-candidate")
EXPECTED = {(case, index) for case in CASES for index in (1, 2, 3)}
EMBED_BUSY = ("vllm:num_requests_running", "vllm:num_requests_waiting")
CACHE_HITS = ("vllm:prefix_cache_hits_total", "vllm:external_prefix_cache_hits_total",
              "vllm:prompt_tokens_cached_total")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_jsonl(path: Path) -> list[dict]:
    require(path.is_file(), f"missing input: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        require(isinstance(row, dict), f"{path}:{line_no}: expected object")
        rows.append(row)
    return rows


def stat(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    require(all(math.isfinite(x) for x in values), "non-finite timing")
    return {"n": len(values), "median": median(values), "min": min(values),
            "max": max(values)}


def response_hash(row: dict, kind: str) -> str | None:
    """Use saved hashes or hash the actual saved message; absence is unknown."""
    saved = row.get(kind + "_sha256")
    if saved is not None:
        return saved
    message = row.get("message")
    if not isinstance(message, dict):
        return None
    value = (message.get("content") if kind == "content" else
             message.get("reasoning") or message.get("reasoning_content")) or ""
    return hashlib.sha256(value.encode()).hexdigest()


def same_response(a: dict, b: dict, kind: str) -> bool | None:
    left, right = response_hash(a, kind), response_hash(b, kind)
    return None if left is None or right is None else left == right


def check_performance(rows: list[dict], arm: str, *, allow_embedding_activity=False) -> dict[tuple[str, int], dict]:
    formal = {}
    warmups = 0
    for row in rows:
        key = (row.get("case"), row.get("index"))
        label = f"{arm}:{key}"
        require(row.get("tag") == arm, f"{label}: wrong tag")
        require(row.get("valid") is True, f"{label}: invalid request")
        require(row.get("profiled") is False and row.get("timing_scope") ==
                "no profiler installed", f"{label}: profiler contaminated timing")
        require(isinstance(row.get("payload_sha256"), str), f"{label}: missing payload hash")
        require(row.get("finish_reason") == "length" and row.get("stream_done") is True,
                f"{label}: incomplete stream")
        delta = row.get("metrics_delta")
        require(isinstance(delta, dict), f"{label}: missing metrics_delta")
        for metric in CACHE_HITS:
            require(delta.get(metric) == 0, f"{label}: cache hit {metric}={delta.get(metric)}")
        require(delta.get("vllm:num_preemptions_total") == 0,
                f"{label}: preemption")
        before, after = row.get("embedding_before"), row.get("embedding_after")
        require(isinstance(before, dict) and isinstance(after, dict),
                f"{label}: missing embedding counters")
        if not allow_embedding_activity:
            require(before == after, f"{label}: embedding counters changed")
        for snapshot_name, snapshot in (("before", before), ("after", after)):
            for metric in EMBED_BUSY:
                require(snapshot.get(metric) == 0,
                        f"{label}: embedding {snapshot_name} busy: {metric}")
        usage = row.get("usage")
        require(isinstance(usage, dict) and usage.get("completion_tokens", 0) > 0,
                f"{label}: missing token usage")
        require(row.get("first_text_s") is not None and row.get("total_s") is not None,
                f"{label}: missing timing")
        if row.get("warmup") is True:
            require(key[1] == 0 and key[0] in CASES, f"{label}: unexpected warmup")
            warmups += 1
            continue
        require(row.get("warmup") is False, f"{label}: missing warmup flag")
        require(key in EXPECTED, f"{label}: unexpected formal request")
        require(key not in formal, f"{label}: duplicate request")
        if key[0] != "prefill":
            tps = row.get("decode_estimate_tps")
            require(isinstance(tps, (int, float)) and math.isfinite(tps) and tps > 0,
                    f"{label}: invalid decode estimate")
        formal[key] = row
    require(set(formal) == EXPECTED, f"{arm}: expected nine formal requests, got {sorted(formal)}")
    require(warmups == len(CASES), f"{arm}: expected three warmups, got {warmups}")
    return formal


def performance_stats(rows: dict[tuple[str, int], dict]) -> dict:
    result = {}
    for case in CASES:
        group = [rows[(case, i)] for i in (1, 2, 3) if (case, i) in rows]
        result[case] = {
            "n": len(group),
            "ttft_s": stat([float(r["first_text_s"]) for r in group]),
            "total_s": stat([float(r["total_s"]) for r in group]),
            # A one-token prefill response has no decode interval.
            "decode_estimate_tps": (None if case == "prefill" else stat(
                [float(r["decode_estimate_tps"]) for r in group])),
            "prompt_tokens": stat([float(r["usage"]["prompt_tokens"]) for r in group]),
            "completion_tokens": stat([float(r["usage"]["completion_tokens"]) for r in group]),
        }
    return result


def check_checks(rows: list[dict], arm: str, kind: str) -> dict:
    expected = 5 if kind == "quality" else 4
    require(len(rows) == expected, f"{arm}:{kind}: expected {expected} rows, got {len(rows)}")
    keyed = {}
    for row in rows:
        require(row.get("tag") == arm and row.get("passed") is True,
                f"{arm}:{kind}:{row.get('case')}: failed or wrong tag")
        key = (row.get("case"), row.get("repeat", 1))
        require(key not in keyed, f"{arm}:{kind}:{key}: duplicate")
        keyed[key] = row
    return keyed


def check_gemm(rows: list[dict]) -> dict:
    checks = [r for r in rows if "backend" in r]
    require(len(checks) == 12, f"microprobe: expected 12 backend checks, got {len(checks)}")
    keyed = {}
    for row in checks:
        key = (row.get("K"), row.get("N"), row.get("M"), row.get("backend"))
        require(key not in keyed and row.get("pass") is True,
                f"microprobe:{key}: duplicate or numeric failure")
        score = row.get("metrics", {})
        require(score.get("finite") is True and score.get("all_output_finite") is True,
                f"microprobe:{key}: non-finite output")
        keyed[key] = row
    shapes = ((5120, 34816), (17408, 5120))
    expected = {(k, n, m, backend) for k, n in shapes for m in (1, 5, 16)
                for backend in ("cutlass", "b12x")}
    require(set(keyed) == expected, "microprobe: unexpected/missing shape")
    result = []
    for k, n in shapes:
        for m in (1, 5, 16):
            c, b = keyed[(k, n, m, "cutlass")], keyed[(k, n, m, "b12x")]
            c_us, b_us = float(c["latency_us"]), float(b["latency_us"])
            require(math.isfinite(c_us) and math.isfinite(b_us) and c_us > 0 and b_us > 0,
                    f"microprobe:{k},{n},{m}: invalid latency")
            result.append({"K": k, "N": n, "M": m,
                           "cutlass_us": c_us, "b12x_us": b_us,
                           "b12x_vs_cutlass_pct": (b_us / c_us - 1) * 100,
                           "cutlass_nrmse": c["metrics"]["nrmse"],
                           "b12x_nrmse": b["metrics"]["nrmse"],
                           "cutlass_cosine": c["metrics"]["cosine"],
                           "b12x_cosine": b["metrics"]["cosine"],
                           "cutlass_max_abs": c["metrics"]["max_abs"],
                           "b12x_max_abs": b["metrics"]["max_abs"]})
    return {"checks_passed": len(checks), "checks_expected": 12,
            "scope": "synthetic quantized operands; CUDA graph GEMM only; no model quality claim",
            "rows": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    base = args.dir
    perf = {arm: check_performance(read_jsonl(base / f"{arm}-performance.jsonl"), arm)
            for arm in ARMS}
    matched = []
    for case in CASES:
        for index in (1, 2, 3):
            a, b = perf[ARMS[0]][(case, index)], perf[ARMS[1]][(case, index)]
            same = a["payload_sha256"] == b["payload_sha256"]
            matched.append({"case": case, "index": index, "payload_hash_match": same,
                            "content_hash_match": a.get("content_sha256") == b.get("content_sha256"),
                            "reasoning_hash_match": a.get("reasoning_sha256") == b.get("reasoning_sha256")})
    require(all(x["payload_hash_match"] for x in matched),
            "formal A/B payload hashes differ; cannot compare performance")
    checks = {arm: {kind: check_checks(read_jsonl(base / f"{arm}-{kind}.jsonl"), arm, kind)
                    for kind in ("quality", "open")} for arm in ARMS}
    check_pairs = {}
    for kind in ("quality", "open"):
        keys = set(checks[ARMS[0]][kind])
        require(keys == set(checks[ARMS[1]][kind]), f"{kind}: case/repeat mismatch")
        check_pairs[kind] = [{"case": key[0], "repeat": key[1],
                              "content_hash_match": same_response(checks[ARMS[0]][kind][key],
                                                                  checks[ARMS[1]][kind][key], "content"),
                              "reasoning_hash_match": same_response(checks[ARMS[0]][kind][key],
                                                                    checks[ARMS[1]][kind][key], "reasoning")}
                             for key in sorted(keys)]
    micro = check_gemm(read_jsonl(base / "gemm-probe.jsonl"))
    summary = {
        "status": "comparable_descriptive_ab",
        "controls": {"formal_requests_per_arm": 9, "warmups_excluded_per_arm": 3,
                     "valid_no_profiler_no_cache_no_embedding_activity_no_busy": True,
                     "formal_payload_hash_matches": sum(x["payload_hash_match"] for x in matched),
                     "formal_payload_hash_expected": 9,
                     "formal_content_hash_matches": sum(x["content_hash_match"] for x in matched),
                     "formal_reasoning_hash_matches": sum(x["reasoning_hash_match"] for x in matched)},
        "performance": {arm: performance_stats(perf[arm]) for arm in ARMS},
        "formal_pair_hashes": matched,
        "synthetic_checks": {arm: {kind: {"passed": len(checks[arm][kind]),
                                            "expected": 5 if kind == "quality" else 4}
                                   for kind in ("quality", "open")} for arm in ARMS},
        "synthetic_check_pair_hashes": check_pairs,
        "microbenchmark": micro,
        "limits": ["n=3 per performance case; ranges are descriptive, not statistical significance",
                   "decode tps estimated from SSE text time and usage tokens",
                   "different output hashes are reported and do not alone prove a quality regression",
                   "synthetic checks and microbenchmarks do not establish MoBase end-to-end quality"],
    }
    repeat_path = base / "cutlass-repeat-performance.jsonl"
    if repeat_path.exists():
        repeat = check_performance(read_jsonl(repeat_path), "cutlass-repeat",
                                   allow_embedding_activity=True)
        excluded = []
        clean = {}
        for key, row in repeat.items():
            before, after = row['embedding_before'], row['embedding_after']
            if before != after:
                excluded.append({'case': key[0], 'index': key[1],
                                 'reason': 'embedding request activity; excluded only from clean repeat comparison',
                                 'embedding_metric_deltas': {k: after[k] - v for k, v in before.items()
                                                            if after[k] != v}})
            else:
                clean[key] = row
        pairs = []
        for key in sorted(EXPECTED):
            first, again = perf[ARMS[0]][key], repeat[key]
            require(first["payload_sha256"] == again["payload_sha256"],
                    f"control repeat payload differs: {key}")
            pairs.append({"case": key[0], "index": key[1], "payload_hash_match": True,
                          "content_hash_match": same_response(first, again, "content"),
                          "reasoning_hash_match": same_response(first, again, "reasoning")})
        summary["control_repeat"] = {
            "formal_requests": 9, "warmups_excluded": 3,
            "clean_formal_requests": len(clean), "excluded_requests": excluded,
            "performance_all_including_flagged": performance_stats(repeat),
            "clean_performance": performance_stats(clean),
            "matched_clean_initial_performance": performance_stats({k: perf[ARMS[0]][k] for k in clean}),
            "paired_hashes": pairs,
            "scope": "same CUTLASS container restarted; empty prefix cache; performance only"}
    target = base / "ab-summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
