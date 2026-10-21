"""Recompute controlled experiment summaries; n=3 medians, no tail claims."""
import json
from pathlib import Path
import statistics

root = Path(__file__).parent
result = {}
all_rows = {}
for path in sorted(root.glob("paired-*.jsonl")):
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    all_rows[path.name] = rows
    groups = {}
    invalid = []
    for r in rows:
        if r["label"].startswith("warmup_") or r["label"] == "thinking_observation":
            continue
        if not r.get("complete_fixed_output") or not r.get("single_request_metrics_window"):
            invalid.append(r["label"])
            continue
        groups.setdefault(r["label"].rsplit("_r", 1)[0], []).append(r)
    out = {"total_rows": len(rows), "invalid_formal_rows": invalid, "groups": {}}
    for label, values in groups.items():
        def median(key):
            xs = [r[key] for r in values if isinstance(r.get(key), (float, int))]
            return statistics.median(xs) if xs else None
        accepted, lengths, overlaps = [], [], []
        for r in values:
            m = r["metrics_delta"]
            drafts = m.get("vllm:spec_decode_num_drafts_total", 0)
            tokens = m.get("vllm:spec_decode_num_draft_tokens_total", 0)
            good = m.get("vllm:spec_decode_num_accepted_tokens_total", 0)
            if tokens:
                accepted.append(good / tokens)
            if drafts:
                lengths.append(1 + good / drafts)
            b, a = r.get("embedding_before", {}), r.get("embedding_after", {})
            if not b or not a or "unavailable" in b or "unavailable" in a:
                overlaps.append({"label": r["label"], "status": "unknown"})
            else:
                changes = {k: a.get(k, v) - v for k, v in b.items()
                           if k.endswith("_total") and a.get(k) != v}
                busy = any(x.get("vllm:num_requests_running", 0) or
                           x.get("vllm:num_requests_waiting", 0) for x in [a, b])
                if changes or busy:
                    overlaps.append({"label": r["label"], "status": "activity_observed", "delta": changes})
        out["groups"][label] = {
            "n": len(values), "input_tokens": [r["usage"]["prompt_tokens"] for r in values],
            "output_tokens": [r["usage"]["completion_tokens"] for r in values],
            "ttft_median_s": median("first_text_s"),
            "decode_estimate_median_tps": median("client_decode_tps_estimate"),
            "e2e_output_median_tps": median("end_to_end_output_tps"),
            "total_median_s": median("total_s"),
            "acceptance_median": statistics.median(accepted) if accepted else None,
            "accepted_plus_bonus_per_step_median": statistics.median(lengths) if lengths else None,
            "prefix_cache_hit_tokens": [r["metrics_delta"].get("vllm:prefix_cache_hits_total") for r in values],
            "embedding_overlap_or_unknown": overlaps,
        }
    result[path.name] = out

baseline = {r["label"]: r for r in all_rows.get("paired-mtp2-bf16.jsonl", [])}
for name, rows in all_rows.items():
    shared = [r for r in rows if r["label"] in baseline]
    result[name]["matched_payloads_vs_mtp2_bf16"] = {
        "compared": len(shared),
        "identical": sum(r["payload_sha256"] == baseline[r["label"]]["payload_sha256"] for r in shared),
        "identical_output_hashes": sum(
            r.get("content_sha256") == baseline[r["label"]].get("content_sha256") and
            r.get("reasoning_sha256") == baseline[r["label"]].get("reasoning_sha256") for r in shared)}
(root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
for name, data in result.items():
    print(name, "rows", data["total_rows"], "invalid", data["invalid_formal_rows"])
    for label, g in data["groups"].items():
        print(label, "n", g["n"], "decode", g["decode_estimate_median_tps"], "ttft", g["ttft_median_s"],
              "accepted+bonus", g["accepted_plus_bonus_per_step_median"])
