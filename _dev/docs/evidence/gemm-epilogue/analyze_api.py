#!/usr/bin/env python3
"""Offline A1/B/A2 analysis of the fixed FP8 experiment JSONL files.

No network, credentials, server controls, or synthetic replacement data. Input
arms are a1, b, a2 under --root. profile_client.py writes one warmup (index 0)
and 12 prefill / 6 decode measured requests per arm.
"""

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path

ARMS = ("a1", "b", "a2")
CASES = {"prefill": 12, "decode": 6}
FILES = ("prefill", "decode", "quality", "open-output")
QUALITY_CASES = frozenset({
    "json_arithmetic", "history_correction", "tool_result_reading",
    "long_retrieval_cold", "long_retrieval_repeated",
})
OPEN_OUTPUT_CASES = frozenset({
    ("open_chinese_operations_summary", 1),
    ("open_chinese_operations_summary", 2),
    ("open_thinking_arithmetic", 1),
    ("open_thinking_arithmetic", 2),
})
EMBEDDING_REQUIRED_COUNTERS = frozenset({
    "vllm:request_success_total", "vllm:prompt_tokens_total",
    "vllm:request_prompt_tokens_count",
})
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
EXTRA_METRICS = (
    "vllm:request_prefill_time_seconds_sum",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def load_jsonl(path):
    if not path.is_file():
        raise ValueError(f"missing input: {path}")
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: JSON object expected")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def stat(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def get_delta(row, key):
    return row.get("metrics_delta", {}).get(key)


def embedding_activity(row):
    before, after = row.get("embedding_before"), row.get("embedding_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {"known": False, "changed": None, "counter_deltas": None,
                "reason": "embedding counter snapshot missing"}
    if set(before) != set(after) or not EMBEDDING_REQUIRED_COUNTERS <= set(before):
        return {"known": False, "changed": None, "counter_deltas": None,
                "reason": "embedding counter keys missing or differ between snapshots"}
    if any(not isinstance(before[name], (int, float)) or
           not isinstance(after[name], (int, float)) or
           not math.isfinite(before[name]) or not math.isfinite(after[name])
           for name in before):
        return {"known": False, "changed": None, "counter_deltas": None,
                "reason": "embedding counter snapshot contains non-finite/non-numeric values"}
    changed = {name: after[name] - before[name] for name in before
               if before[name] != after[name]}
    return {"known": True, "changed": bool(changed),
            "counter_deltas": dict(sorted(changed.items())), "reason": None}


def validate_request(row, arm, case, expected_index, warmup):
    ident = f"{arm}/{case}/{expected_index}"
    if row.get("case") != case or row.get("index") != expected_index or row.get("warmup") is not warmup:
        raise ValueError(f"{ident}: unexpected case/index/warmup")
    if row.get("valid") is not True or row.get("profiled") is not False:
        raise ValueError(f"{ident}: request invalid or profiled")
    if row.get("stream_done") is not True or row.get("finish_reason") != "length":
        raise ValueError(f"{ident}: incomplete fixed-length request")
    usage = row.get("usage") or {}
    delta = row.get("metrics_delta") or {}
    if not isinstance(row.get("payload_sha256"), str) or len(row["payload_sha256"]) != 64:
        raise ValueError(f"{ident}: missing payload hash")
    # profile_client only labels a request valid when these counters match.
    # Check them independently so a busy/extra target request cannot hide.
    counts = ("vllm:request_success_total", "vllm:request_prompt_tokens_count",
              "vllm:request_generation_tokens_count")
    if any(delta.get(name) != 1 for name in counts):
        raise ValueError(f"{ident}: target service window has extra/missing requests")
    if delta.get("vllm:prompt_tokens_total") != usage.get("prompt_tokens") or \
       delta.get("vllm:generation_tokens_total") != usage.get("completion_tokens"):
        raise ValueError(f"{ident}: target token counters do not match request usage")
    if not isinstance(row.get("first_text_s"), (int, float)) or row["first_text_s"] <= 0:
        raise ValueError(f"{ident}: missing first-text time")
    if not warmup and case == "decode" and not isinstance(row.get("decode_estimate_tps"), (int, float)):
        raise ValueError(f"{ident}: missing decode estimate")


def request_sample(row):
    usage = row["usage"]
    return {
        "index": row["index"], "at_utc": row.get("at_utc"),
        "payload_sha256": row["payload_sha256"], "valid": row["valid"],
        "first_text_s": row["first_text_s"], "first_content_s": row.get("first_content_s"),
        "last_text_s": row.get("last_text_s"), "total_s": row.get("total_s"),
        "decode_estimate_tps": row.get("decode_estimate_tps"),
        "finish_reason": row.get("finish_reason"), "usage": usage,
        "content_sha256": row.get("content_sha256"),
        "reasoning_sha256": row.get("reasoning_sha256"),
        "metric_deltas": {name: get_delta(row, name) for name in EXTRA_METRICS},
        "request_success_delta": get_delta(row, "vllm:request_success_total"),
        "embedding_activity": embedding_activity(row),
    }


def summarize_case(rows):
    def metric(key):
        return stat(get_delta(row, key) for row in rows)
    stats = {"n": len(rows), "first_text_s": stat(row.get("first_text_s") for row in rows),
             "total_s": stat(row.get("total_s") for row in rows),
             "decode_estimate_tps": stat(row.get("decode_estimate_tps") for row in rows),
             "request_prefill_time_seconds_sum": metric("vllm:request_prefill_time_seconds_sum"),
             "prefix_cache_hits_total": metric("vllm:prefix_cache_hits_total"),
             "prefix_cache_queries_total": metric("vllm:prefix_cache_queries_total"),
             "spec_decode_num_drafts_total": metric("vllm:spec_decode_num_drafts_total"),
             "spec_decode_num_draft_tokens_total": metric("vllm:spec_decode_num_draft_tokens_total"),
             "spec_decode_num_accepted_tokens_total": metric("vllm:spec_decode_num_accepted_tokens_total"),
             "accepted_per_draft": stat(
                 get_delta(row, "vllm:spec_decode_num_accepted_tokens_total") /
                 get_delta(row, "vllm:spec_decode_num_drafts_total")
                 if get_delta(row, "vllm:spec_decode_num_drafts_total") else None
                 for row in rows),
             "embedding_active_indices": [row["index"] for row in rows
                                           if embedding_activity(row)["changed"] is True],
             "embedding_unknown_indices": [row["index"] for row in rows
                                            if not embedding_activity(row)["known"]]}
    return stats


def bootstrap_median(values, seed=20261014, draws=10000):
    if not values:
        return None
    rng = random.Random(seed)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(draws))
    return {"method": "resample paired request indices with replacement; median paired percent reduction",
            "draws": draws, "seed": seed,
            "percentile_2_5": medians[int(0.025 * (draws - 1))],
            "percentile_97_5": medians[int(0.975 * (draws - 1))],
            "interpretation": "descriptive prompt-index variability only; no independent repeated time windows or causal confidence interval"}


def paired_timing(raw, case):
    pairs = []
    for index in range(1, CASES[case] + 1):
        a1, b, a2 = (raw[arm][case][index] for arm in ARMS)
        if len({a1["payload_sha256"], b["payload_sha256"], a2["payload_sha256"]}) != 1:
            raise ValueError(f"{case}/{index}: payload hash differs between A1/B/A2")
        baseline = (a1["first_text_s"] + a2["first_text_s"]) / 2
        reduction = 100 * (baseline - b["first_text_s"]) / baseline
        quiet = all(embedding_activity(row)["known"] and
                    not embedding_activity(row)["changed"] for row in (a1, b, a2))
        pairs.append({"index": index, "payload_sha256": a1["payload_sha256"],
                      "a1_first_text_s": a1["first_text_s"], "b_first_text_s": b["first_text_s"],
                      "a2_first_text_s": a2["first_text_s"],
                      "baseline_a1_a2_mean_s": baseline, "b_ttft_reduction_pct": reduction,
                      "embedding_quiet_all_arms": quiet})
    all_changes = [p["b_ttft_reduction_pct"] for p in pairs]
    quiet_changes = [p["b_ttft_reduction_pct"] for p in pairs if p["embedding_quiet_all_arms"]]
    return {"formula": "100 * (((A1_TTFT + A2_TTFT)/2) - B_TTFT) / ((A1_TTFT + A2_TTFT)/2)",
            "pairs": pairs, "all_pairs_reduction_pct": stat(all_changes),
            "all_pairs_bootstrap_descriptive_pct": bootstrap_median(all_changes),
            "embedding_quiet_pairs_reduction_pct": stat(quiet_changes),
            "embedding_quiet_pairs_bootstrap_descriptive_pct": bootstrap_median(quiet_changes),
            "excluded_from_embedding_quiet_comparison": [
                {"index": p["index"], "reason": "embedding activity or unavailable counters"}
                for p in pairs if not p["embedding_quiet_all_arms"]]}


def hash_comparison(raw, case, left, right):
    def valid(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None

    pairs = []
    for index in range(1, CASES[case] + 1):
        a, b = raw[left][case][index], raw[right][case][index]
        pair = {"index": index, "payload_sha256": a["payload_sha256"]}
        for field in ("content", "reasoning"):
            lhs, rhs = a.get(field + "_sha256"), b.get(field + "_sha256")
            hashes_valid = valid(lhs) and valid(rhs)
            raw_equal = hashes_valid and lhs.lower() == rhs.lower()
            nonempty_equal = raw_equal and lhs.lower() != EMPTY_SHA256
            pair[field] = {"left_hash": lhs, "right_hash": rhs,
                           "both_hashes_valid": hashes_valid,
                           "raw_hash_equal": raw_equal,
                           "meaningful_nonempty_equal": nonempty_equal,
                           "both_empty": raw_equal and lhs.lower() == EMPTY_SHA256}
        pairs.append(pair)
    return {"left": left, "right": right, "pairs": pairs,
            "content": {"raw_hash_equal_count": sum(p["content"]["raw_hash_equal"] for p in pairs),
                        "meaningful_nonempty_equal_count": sum(p["content"]["meaningful_nonempty_equal"] for p in pairs),
                        "both_empty_count": sum(p["content"]["both_empty"] for p in pairs),
                        "invalid_pair_count": sum(not p["content"]["both_hashes_valid"] for p in pairs)},
            "reasoning": {"raw_hash_equal_count": sum(p["reasoning"]["raw_hash_equal"] for p in pairs),
                          "meaningful_nonempty_equal_count": sum(p["reasoning"]["meaningful_nonempty_equal"] for p in pairs),
                          "both_empty_count": sum(p["reasoning"]["both_empty"] for p in pairs),
                          "invalid_pair_count": sum(not p["reasoning"]["both_hashes_valid"] for p in pairs)},
            "n": len(pairs)}


def quality_summary(rows, kind):
    cases = []
    for row in rows:
        entry = {"case": row.get("case"), "repeat": row.get("repeat"),
                 "passed": row.get("passed") is True,
                 "finish_reason": row.get("finish_reason"),
                 "payload_sha256": row.get("payload_sha256"),
                 "content_sha256": row.get("content_sha256"),
                 "reasoning_sha256": row.get("reasoning_sha256")}
        if kind == "open-output":
            content = row.get("content") or (row.get("message") or {}).get("content") or ""
            facts = row.get("fact_checks") or {}
            entry.update(final_body=content, final_body_nonempty=bool(content.strip()),
                         fact_checks=facts,
                         normal_stop_with_answer=bool(row.get("passed") is True and
                                                       row.get("finish_reason") == "stop" and
                                                       content.strip() and
                                                       all(facts.values())))
        cases.append(entry)
    return {"n": len(rows), "passed": sum(x["passed"] for x in cases),
            "all_passed": all(x["passed"] for x in cases),
            "finish_reason_counts": dict(Counter(x["finish_reason"] for x in cases)),
            "cases": cases,
            "scope": "fixed synthetic API checks; normal stop/final body is not broad business-quality evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "api-comparison.json"
    source_hashes, raw, summary = {}, {}, {}
    for arm in ARMS:
        source_hashes[arm], raw[arm], summary[arm] = {}, {}, {}
        for name in FILES:
            path = root / arm / (name + ".jsonl")
            source_hashes[arm][name] = {"path": str(path),
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            rows = load_jsonl(path)
            if name in CASES:
                expected_n = CASES[name]
                by_index = {}
                for row in rows:
                    index = row.get("index")
                    if not isinstance(index, int) or index in by_index:
                        raise ValueError(f"{arm}/{name}: duplicate or missing index")
                    by_index[index] = row
                if set(by_index) != set(range(expected_n + 1)):
                    raise ValueError(f"{arm}/{name}: expected warmup index 0 plus indices 1..{expected_n}")
                for index in range(expected_n + 1):
                    validate_request(by_index[index], arm, name, index, index == 0)
                raw[arm][name] = by_index
                measured = [by_index[i] for i in range(1, expected_n + 1)]
                summary[arm][name] = {"statistics": summarize_case(measured),
                                       "samples": [request_sample(r) for r in measured],
                                       "excluded_from_performance": [
                                           {"index": 0, "reason": "profile_client warmup",
                                            "sample": request_sample(by_index[0])}]}
            else:
                if name == "quality" and (len(rows) != 5 or
                                          {r.get("case") for r in rows} != QUALITY_CASES or
                                          any(r.get("repeat") is not None for r in rows)):
                    raise ValueError(f"{arm}/quality: expected exactly the five fixed cases")
                if name == "open-output" and (len(rows) != 4 or
                                              {(r.get("case"), r.get("repeat")) for r in rows} != OPEN_OUTPUT_CASES):
                    raise ValueError(f"{arm}/open-output: expected both fixed cases, repeats 1 and 2")
                raw[arm][name] = rows
                summary[arm][name] = quality_summary(rows, name)
                if not summary[arm][name]["all_passed"]:
                    raise ValueError(f"{arm}/{name}: synthetic quality failure")
                if name == "open-output" and not all(x["normal_stop_with_answer"] for x in summary[arm][name]["cases"]):
                    raise ValueError(f"{arm}/open-output: no normal stop or final answer")
    # Historical quality probes store case names/results but no request hashes.
    # Missing hashes are unavailable evidence, never an equality check.
    probe_payload_validation = {}
    for kind in ("quality", "open-output"):
        keys = []
        hash_counts = {}
        for arm in ARMS:
            hashes = [r.get("payload_sha256") for r in raw[arm][kind]]
            present = [h for h in hashes if h is not None]
            if any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
                   for h in present):
                raise ValueError(f"{arm}/{kind}: invalid payload hash")
            hash_counts[arm] = len(present)
            if kind == "open-output" and len(present) != len(hashes):
                raise ValueError(f"{arm}/{kind}: missing payload hash")
            arm_keys = [(r.get("case"), r.get("repeat"), r.get("payload_sha256"))
                        for r in raw[arm][kind]]
            if len(arm_keys) != len(set(arm_keys)):
                raise ValueError(f"{arm}/{kind}: duplicate test")
            keys.append(sorted(arm_keys, key=str))
        if not keys[0] == keys[1] == keys[2]:
            raise ValueError(f"{kind}: test cases/payload hashes differ across arms")
        complete = all(hash_counts[arm] == len(raw[arm][kind]) for arm in ARMS)
        probe_payload_validation[kind] = {
            "same_named_cases": True, "payload_hash_count_by_arm": hash_counts,
            "payload_hashes_equal": True if complete else None,
            "status": "verified_hashes" if complete else "payload_hash_unavailable",
        }
    for case in CASES:
        for index in range(CASES[case] + 1):
            hashes = {raw[arm][case][index]["payload_sha256"] for arm in ARMS}
            if len(hashes) != 1:
                raise ValueError(f"{case}/{index}: payload hashes differ across arms")
    result = {
        "scope": "offline C1 fixed synthetic API A1/B/A2; no profiler; no independent repeated time windows",
        "sources": source_hashes, "validation": {"all_measured_requests_valid": True,
            "one_target_service_request_per_metrics_window": True,
            "all_index_payload_hashes_equal": True, "quality_all_passed": True,
            "index_payload_hash_scope": "prefill/decode, including excluded warmups",
            "probe_payloads": probe_payload_validation,
            "open_output_all_normal_stop_with_final_body": True},
        "arms": summary,
        "paired_ttft": {case: paired_timing(raw, case) for case in CASES},
        "decode_output_hashes": {
            "a1_vs_a2": hash_comparison(raw, "decode", "a1", "a2"),
            "a1_vs_b": hash_comparison(raw, "decode", "a1", "b"),
            "b_vs_a2": hash_comparison(raw, "decode", "b", "a2")},
        "prefill_one_token_output_hashes": {
            "a1_vs_a2": hash_comparison(raw, "prefill", "a1", "a2"),
            "a1_vs_b": hash_comparison(raw, "prefill", "a1", "b")},
        "limitations": [
            "Historical quality rows lack payload hashes; only the five named cases and their results are checked, not captured request-byte equality.",
            "Hash differences among A1/B/A2 prevent attributing output or acceptance changes solely to the candidate kernel.",
            "Bootstrap resamples paired prompt indices only; it is descriptive and does not estimate time-window or restart variability.",
            "Streaming decode tps uses usage completion tokens and first/last text time, not SSE chunk count.",
            "Embedding counters cover completed requests between before/after snapshots; 0→0 running gauges do not prove no transient activity between samples.",
            "An equal SHA256 of empty reasoning/content is recorded as raw equality, not evidence of matching nonempty text.",
            "Synthetic quality and normally stopped final body do not establish MoBase/business task quality."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "prefill_pairs": len(result["paired_ttft"]["prefill"]["pairs"]),
                      "decode_pairs": len(result["paired_ttft"]["decode"]["pairs"])}))


if __name__ == "__main__":
    main()
