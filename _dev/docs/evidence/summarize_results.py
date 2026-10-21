"""Recompute medians from raw baseline JSONL; no percentile claims for n=3."""
import json
from pathlib import Path
import statistics

root = Path(__file__).parent
summary = {}
for path in sorted(list(root.glob("baseline-*.jsonl")) + list(root.glob("paired-*.jsonl"))):
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups = {}
    for row in samples:
        if row["label"] in ("warmup_excluded", "thinking_observation"):
            continue
        if not row.get("single_request_metrics_window"):
            continue
        if row.get("harness_version") == 2 and not row.get("complete_fixed_output"):
            continue
        label = row["label"].rsplit("_r", 1)[0]
        groups.setdefault(label, []).append(row)
    data = {}
    for label, rows in groups.items():
        def med(key):
            values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            return statistics.median(values) if values else None
        prefill, accepts, lengths = [], [], []
        for row in rows:
            delta = row["metrics_delta"]
            elapsed = delta.get("vllm:request_prefill_time_seconds_sum", 0)
            hits = delta.get("vllm:prefix_cache_hits_total", 0)
            if elapsed > 0:
                prefill.append((row["usage"]["prompt_tokens"] - hits) / elapsed)
            drafted = delta.get("vllm:spec_decode_num_draft_tokens_total", 0)
            steps = delta.get("vllm:spec_decode_num_drafts_total", 0)
            accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0)
            if drafted:
                accepts.append(accepted / drafted)
            if steps:
                lengths.append(1 + accepted / steps)
        emb_observed = all("embedding_before" in r and "unavailable" not in r["embedding_before"]
                           and bool(r["embedding_before"]) and "unavailable" not in r["embedding_after"] for r in rows)
        emb_idle = emb_observed and all(
            all(r["embedding_after"].get(k) == v for k, v in r["embedding_before"].items() if k.endswith("_total"))
            and r["embedding_before"].get("vllm:num_requests_running") == 0
            and r["embedding_after"].get("vllm:num_requests_running") == 0 for r in rows)
        data[label] = {
            "n_valid": len(rows), "input_tokens_range": [min(r["usage"]["prompt_tokens"] for r in rows), max(r["usage"]["prompt_tokens"] for r in rows)],
            "output_tokens": sorted(set(r["usage"]["completion_tokens"] for r in rows)),
            "ttft_text_median_s": med("first_text_s"), "decode_estimate_median_tps": med("client_decode_tps_estimate"),
            "total_median_s": med("total_s"),
            "serving_prefill_median_tps": statistics.median(prefill) if prefill else None,
            "mtp_acceptance_median": statistics.median(accepts) if accepts else None,
            "mtp_mean_length_median": statistics.median(lengths) if lengths else None,
            "prefix_hit_tokens": [r["metrics_delta"].get("vllm:prefix_cache_hits_total") for r in rows],
            "embedding_metrics_observed": emb_observed, "embedding_counters_unchanged_and_running_zero": emb_idle}
    summary[path.name] = {"total_rows": len(samples), "groups": data}
out = root / "summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
