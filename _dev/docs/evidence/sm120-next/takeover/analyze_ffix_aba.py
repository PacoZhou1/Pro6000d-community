#!/usr/bin/env python3
"""CPU-only strict raw-request analysis of the fresh null-MD-fix A/B/A.

This checks one bracketed API experiment, not broad quality or kernel-only
causality. No service calls, torch imports or GPU operations are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

ARMS = ("a1", "b", "a2")
EXPECTED_SOURCE = {
    "sitecustomize.py": "854043107312ee708f4107e35362d6096ce270cc33d4a50293d1f2026df1bbcc",
    "f_kernels.py": "1b51a6d934e214190fcb586b217d1f890d096a699fdc06a83c97f0f2d61c8dd6",
    "v2_sitecustomize.py": "8cdace709c1c9787c07ea5c013bdefdc1cc2c7262aec1f9be8064a2250753b63",
    "v2/libfp8_silu_quant.so": "087b7caa92c4742ece3875b6dd1421e25360b834ce4106f7fa166f90bc933b3f",
}
HASH = re.compile(r"[0-9a-f]{64}\Z")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def metric(row, key):
    value = row["metrics_delta"]["vllm:" + key]
    require(isinstance(value, (int, float)) and math.isfinite(value), f"bad metric {key}")
    return value


def quiet(row):
    before, after = row["embedding_before"], row["embedding_after"]
    for key in ("request_success_total", "prompt_tokens_total", "generation_tokens_total"):
        k = "vllm:" + key
        require(k in before and k in after, f"missing embedding metric {key}")
        if before[k] != after[k]:
            return False
    for key in ("num_requests_running", "num_requests_waiting"):
        if before.get("vllm:" + key) != 0 or after.get("vllm:" + key) != 0:
            return False
    return True


def summarize(values):
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "raw": values}


def analyze(root):
    hashes = {}

    def load(path, lines=False):
        hashes[str(path.relative_to(root))] = digest(path)
        return ([json.loads(l) for l in path.read_text().splitlines() if l.strip()]
                if lines else json.loads(path.read_text()))

    manifest = load(root / "source-manifest.json")
    require(manifest == EXPECTED_SOURCE, "runtime source manifest changed")
    controller = load(root / "status.json")
    require(controller["state"] == "completed"
            and [r["arm"] for r in controller["arms"]] == list(ARMS)
            and all(r["exit_code"] == 0 for r in controller["arms"]), "controller incomplete")
    runtime, env, status, services, logs, raw = {}, {}, {}, {}, {}, {}
    for arm in ARMS:
        p = root / arm
        runtime[arm] = load(p / "runtime.json")
        require(runtime[arm]["State"]["Running"] is True
                and runtime[arm]["State"]["OOMKilled"] is False, f"runtime not healthy {arm}")
        env[arm] = dict(x.split("=", 1) for x in runtime[arm]["env"])
        require(env[arm].get("PRO6000D_F_ENABLE") == ("norm,conv" if arm == "b" else ""),
                f"wrong F_ENABLE {arm}")
        require(env[arm].get("PRO6000D_F_V2") == "1"
                and env[arm].get("PRO6000D_FP8_FUSION_ENABLE") == "1", f"missing V2 {arm}")
        require(env[arm].get("PRO6000D_F_KERNELS_SHA256") == EXPECTED_SOURCE["f_kernels.py"]
                and env[arm].get("PRO6000D_FP8_FUSION_LIBRARY_SHA256")
                == EXPECTED_SOURCE["v2/libfp8_silu_quant.so"], f"runtime library SHA {arm}")
        require(all(env[arm].get(k, "0") == "0" for k in
                    ("PRO6000D_F_VERIFY", "PRO6000D_F_VERIFY_CONV")), f"diagnostic enabled {arm}")
        require(not env[arm].get("PRO6000D_F_DIAGNOSTIC_SHA256"), f"diagnostic helper {arm}")
        status[arm] = load(p / "status.json")
        require({r["kind"] for r in status[arm]} == {"prefill", "decode", "quality", "open-output"}
                and len(status[arm]) == 4 and all(r["exit_code"] == 0 for r in status[arm]),
                f"API subprocess status {arm}")
        services[arm] = {phase: load(p / ("services-" + phase + ".json"))
                         for phase in ("before", "after")}
        for phase, s in services[arm].items():
            require(s["passed"] is True and len(s["services"]) == 4, f"service check {arm}/{phase}")
            for row in s["services"]:
                require(row["passed"] is True and row["running"] == (row["name"] == "vllm-embedding"),
                        f"service policy {arm}/{phase}")
                if row["name"] == "vllm-embedding":
                    require(row["health"] == 200 and row["functional"]["status"] == 200
                            and row["functional"]["dimensions"] == 1024
                            and row["functional"]["finite"] is True, f"embedding function {arm}/{phase}")
        hashes[f"{arm}/server.log"] = digest(p / "server.log")
        logs[arm] = (p / "server.log").read_text()
        raw[arm] = {kind: load(p / (kind + ".jsonl"), True)
                    for kind in ("prefill", "decode", "quality", "open-output")}
    for arm in ARMS[1:]:
        require(runtime[arm]["Image"] == runtime["a1"]["Image"]
                and runtime[arm]["cmd"] == runtime["a1"]["cmd"], f"image/cmd changed {arm}")
        require({k: v for k, v in env[arm].items() if k != "PRO6000D_F_ENABLE"}
                == {k: v for k, v in env["a1"].items() if k != "PRO6000D_F_ENABLE"},
                f"unexpected runtime environment difference {arm}")
        for dest in ("/model", "/instrument"):
            mounts = [[m for m in runtime[a]["Mounts"] if m["Destination"] == dest] for a in ("a1", arm)]
            require(all(len(m) == 1 and m[0]["RW"] is False for m in mounts)
                    and mounts[0][0]["Source"] == mounts[1][0]["Source"], f"mount changed {dest}/{arm}")
    identities = {}
    for arm in ARMS:
        for phase in ("before", "after"):
            for service in services[arm][phase]["services"]:
                identity = tuple(service[k] for k in ("id", "image", "started_at", "restart_count"))
                name = service["name"]
                require(identities.setdefault(name, identity) == identity,
                        f"original service identity/start/restarts changed {arm}/{phase}/{name}")

    result = {"schema": "ffix-fresh-aba-v1", "scope": "single fresh A/B/A; API synthetic workload",
              "source_manifest": manifest, "image": runtime["a1"]["Image"],
              "runtime_environment_only_difference": "PRO6000D_F_ENABLE: empty / norm,conv / empty",
              "timing_definition": "TTFT=client first_text_s; decode=(completion_tokens-1)/(last_text_s-first_text_s)",
              "paired_definition": "per request index: baseline=(A1+A2)/2; positive reduction=(1-B/baseline)*100",
              "limitations": ["single sequential bracket, no randomized repetitions or causal root-cause isolation",
                              "quality input payload hashes unavailable; same named cases passing is not captured input identity",
                              "empty reasoning hash is not model quality evidence",
                              "matching API content/draft counters does not prove whole-model bit identity",
                              "no VERIFY/clones/synchronizing reference in this timing run",
                              "fixed warmup output contract bug is not established as the only cause of historical B divergence"],
              "cases": {}}
    formal = {a: {} for a in ARMS}
    for kind, count, completion in (("prefill", 12, 1), ("decode", 6, 1024)):
        for arm in ARMS:
            rows = raw[arm][kind]
            require(len(rows) == count + 1 and sum(r["warmup"] is True for r in rows) == 1,
                    f"warmup/formal count {arm}/{kind}")
            require(rows[0]["warmup"] is True and rows[0]["index"] == 0, "warmup not first")
            formal[arm][kind] = sorted([r for r in rows if not r["warmup"]], key=lambda r: r["index"])
            require([r["index"] for r in formal[arm][kind]] == list(range(1, count + 1)), "indices")
            for row in rows:
                require(row["case"] == kind, "request case mismatch")
                require(row["valid"] is True and row["stream_done"] is True
                        and row["finish_reason"] == "length" and row["profiled"] is False, "request validity")
                require(all(valid_hash(row.get(k)) for k in ("payload_sha256", "content_sha256", "reasoning_sha256")),
                        f"missing request/output hash {arm}/{kind}")
                require(metric(row, "request_success_total") == 1
                        and metric(row, "prompt_tokens_total") == row["usage"]["prompt_tokens"]
                        and metric(row, "generation_tokens_total") == row["usage"]["completion_tokens"], "request accounting")
                require(all(metric(row, k) == 0 for k in ("prefix_cache_hits_total", "external_prefix_cache_hits_total",
                                                          "prompt_tokens_cached_total", "num_preemptions_total")), "cache/preemption")
                require(row["usage"]["total_tokens"] == row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"],
                        "token total")
                if not row["warmup"]:
                    require(row["usage"]["completion_tokens"] == completion, "formal completion length")
                require(math.isfinite(row["first_text_s"]) and row["first_text_s"] > 0, "TTFT")
                if kind == "decode":
                    expected = (row["usage"]["completion_tokens"] - 1) / (row["last_text_s"] - row["first_text_s"])
                    require(math.isclose(row["decode_estimate_tps"], expected, rel_tol=1e-12), "decode TPS definition")
                    require(metric(row, "spec_decode_num_draft_tokens_total") == 4 * metric(row, "spec_decode_num_drafts_total")
                            and metric(row, "spec_decode_num_accepted_tokens_total")
                            == metric(row, "spec_decode_num_accepted_tokens_per_pos_total"), "MTP accounting")
        pairs = []
        for trio in zip(*(formal[a][kind] for a in ARMS)):
            require(len({r["payload_sha256"] for r in trio}) == 1
                    and len({r["usage"]["prompt_tokens"] for r in trio}) == 1, "payload/token mismatch")
            a1, b, a2 = trio
            baseline = (a1["first_text_s"] + a2["first_text_s"]) / 2
            row = {"index": b["index"], "payload_sha256": b["payload_sha256"],
                   "prompt_tokens": b["usage"]["prompt_tokens"],
                   "ttft_s": {a: r["first_text_s"] for a, r in zip(ARMS, trio)},
                   "ttft_reduction_pct": (1 - b["first_text_s"] / baseline) * 100,
                   "baseline_drift_pct": (a2["first_text_s"] / a1["first_text_s"] - 1) * 100,
                   "embedding_quiet_all_arms": all(quiet(r) for r in trio),
                   "content_hashes": {a: r["content_sha256"] for a, r in zip(ARMS, trio)},
                   "all_content_equal": len({r["content_sha256"] for r in trio}) == 1,
                   "reasoning_tokens": {a: r["usage"].get("completion_tokens_details", {}).get("reasoning_tokens")
                                        for a, r in zip(ARMS, trio)},
                   "drafts_accepted": {a: [metric(r, "spec_decode_num_drafts_total"),
                                           metric(r, "spec_decode_num_accepted_tokens_total")]
                                       for a, r in zip(ARMS, trio)}}
            if kind == "decode":
                tps = {a: r["decode_estimate_tps"] for a, r in zip(ARMS, trio)}
                row.update(decode_estimate_tps=tps, tps_gain_pct=(tps["b"] / ((tps["a1"] + tps["a2"]) / 2) - 1) * 100)
            pairs.append(row)
        pairs_quiet = [p for p in pairs if p["embedding_quiet_all_arms"]]
        result["cases"][kind] = {
            "per_arm_ttft_s": {a: summarize([r["first_text_s"] for r in formal[a][kind]]) for a in ARMS},
            "paired_ttft_reduction_pct": summarize([p["ttft_reduction_pct"] for p in pairs]),
            "baseline_ttft_drift_pct": summarize([p["baseline_drift_pct"] for p in pairs]),
            "embedding_quiet_paired_ttft_reduction_pct": summarize([p["ttft_reduction_pct"] for p in pairs_quiet]),
            "embedding_quiet_definition": "all three requests have unchanged embedding request/prompt/generation counters and zero running/waiting at both snapshots; no proof of activity between scrapes",
            "content_equal_all_arms": sum(p["all_content_equal"] for p in pairs),
            "warmup_payload_and_content_all_equal": all(len({raw[a][kind][0][k] for a in ARMS}) == 1
                                                        for k in ("payload_sha256", "content_sha256")),
            "pairs": pairs}
        if kind == "decode":
            result["cases"][kind].update(
                per_arm_tps={a: summarize([r["decode_estimate_tps"] for r in formal[a][kind]]) for a in ARMS},
                paired_tps_gain_pct=summarize([p["tps_gain_pct"] for p in pairs]),
                embedding_quiet_paired_tps_gain_pct=summarize([p["tps_gain_pct"] for p in pairs_quiet]),
                drafts_accepted_equal_all_arms=sum(len({tuple(v) for v in p["drafts_accepted"].values()}) == 1 for p in pairs))

    result["quality"] = {a: {"passed": sum(r.get("passed") is True for r in raw[a]["quality"]),
                              "n": len(raw[a]["quality"]), "cases": [r["case"] for r in raw[a]["quality"]],
                              "payload_hashes_available": all(valid_hash(r.get("payload_sha256")) for r in raw[a]["quality"])}
                         for a in ARMS}
    for arm in ARMS:
        require(result["quality"][arm]["n"] == 5 and result["quality"][arm]["passed"] == 5, "quality failure/count")
        require(result["quality"][arm]["cases"] == result["quality"]["a1"]["cases"], "quality case names changed")
    result["quality"]["cross_arm_payload_identity"] = None
    result["quality"]["payload_identity_status"] = "unavailable; do not equate missing hashes"
    opened = {}
    for arm in ARMS:
        rows = raw[arm]["open-output"]
        require(len(rows) == 4 and all(r["passed"] is True and r["finish_reason"] == "stop" for r in rows), "open-output")
        opened[arm] = {(r["case"], r["repeat"]): r for r in rows}
        require(len(opened[arm]) == 4, "duplicate open-output key")
        for r in rows:
            require(all(valid_hash(r.get(k)) for k in ("payload_sha256", "content_sha256", "reasoning_sha256")), "open hash missing")
            for field in ("content", "reasoning"):
                require(hashlib.sha256(r[field].encode()).hexdigest() == r[field + "_sha256"], "open output hash invalid")
            # Writer hashes the canonical parsed request, not HTTP wire bytes.
            require(hashlib.sha256(json.dumps(r["payload"], sort_keys=True, ensure_ascii=False,
                                              separators=(",", ":")).encode()).hexdigest()
                    == r["payload_sha256"], "open payload canonical hash invalid")
    require(all(opened[a].keys() == opened["a1"].keys() for a in ARMS), "open-output pairing")
    result["open_output"] = {"n_per_arm": 4, "all_passed": True,
                             "same_payload_content_reasoning": sum(all(len({opened[a][key][f] for a in ARMS}) == 1
                                                                       for f in ("payload_sha256", "content_sha256", "reasoning_sha256"))
                                                                   for key in opened["a1"])}
    result["runtime_log_observations"] = {
        a: {"kv_memory_gib": re.findall(r"Available KV cache memory: ([\d.]+) GiB", logs[a]),
            "kv_tokens": re.findall(r"GPU KV cache size: ([\d,]+) tokens", logs[a]),
            "graph_pool_gib": re.findall(r"Graph capturing finished.*?took ([\d.]+) GiB", logs[a]),
            "new_custom_op_mentioned": "vllm::pro6000d_gdn_core_norm_quant" in logs[a],
            "f1_fused_marker": "P3 gdn core+norm+quant fused path used" in logs[a],
            "f2_fused_marker": "P3 conv_post_fused kernel launched" in logs[a],
            "verification_log_present": "P3 VERIFY" in logs[a],
            "prewarm_marker_present": "P3 kernels pre-compiled during warmup" in logs[a]}
        for a in ARMS}
    for a in ARMS:
        observation = result["runtime_log_observations"][a]
        require(not observation["verification_log_present"], "verification present in timing log")
        require(observation["kv_memory_gib"] == ["30.75"] and observation["kv_tokens"] == ["848,709"]
                and observation["graph_pool_gib"] == ["0.84", "0.25"], f"KV/graph budget changed {a}")
        require(observation["f1_fused_marker"] == observation["f2_fused_marker"] == (a == "b"),
                f"fused markers inconsistent {a}")
    request_windows = []
    for arm in ARMS:
        times = [datetime.fromisoformat(r["at_utc"]) for k in ("prefill", "decode") for r in raw[arm][k]]
        request_windows.append((min(times), max(times)))
    require(all(a[1] < b[0] for a, b in zip(request_windows, request_windows[1:])), "ABA request windows overlap/order")
    result["services_before_after_all_passed"] = True
    result["service_identities"] = {name: dict(zip(("id", "image", "started_at", "restart_count"), value))
                                    for name, value in identities.items()}
    monitor = load(root / "embedding-monitor.jsonl", True)
    require(len(monitor) >= 2, "monitor missing")
    expected = identities["vllm-embedding"]
    require(all(r["running"] is True and r["health"] == 200
                and (r["id"], r["started_at"], r["restart_count"]) == (expected[0], expected[2], expected[3])
                for r in monitor), "embedding monitor failed/identity changed")
    times = [datetime.fromisoformat(r["at_utc"]) for r in monitor]
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    require(all(g > 0 for g in gaps), "monitor timestamps")
    result["embedding_monitor"] = {"samples": len(monitor), "all_health_200": True,
                                   "identity_start_restarts_unchanged": True,
                                   "max_sample_gap_s": max(gaps), "first_utc": monitor[0]["at_utc"],
                                   "last_utc": monitor[-1]["at_utc"]}
    result["controller"] = controller
    result["raw_sha256"] = hashes
    result["analyzer_sha256"] = digest(Path(__file__))
    result["validated"] = True
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"validated": True, "output": str(args.output), "summary": {
        k: {f: v for f, v in row.items() if f != "pairs"} for k, row in result["cases"].items()}}, allow_nan=False))


if __name__ == "__main__":
    main()
