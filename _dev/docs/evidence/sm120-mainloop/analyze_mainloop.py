#!/usr/bin/env python3
"""CPU-only strict audit of mainloop_probe JSONL, reporting every selected arm."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import mainloop_probe as current_contract

from mainloop_probe import (BASELINE_SHA, DIAGNOSTICS, FALLBACK_M, FUSED_ARMS,
                           NORMAL_M, OFFICIAL_ARMS, PIPELINE_SHA, SCHEDULER_SHA,
                           SPECIAL, QUANT_SHA, VARIANTS, metadata_check, parse_arm)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_ok(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def checks_pass(checks, required, context, m):
    require(isinstance(checks, list) and len(checks) == len(required), f"{context}: comparison count mismatch")
    labels = [c.get("label") for c in checks]
    require(len(set(labels)) == len(labels) and set(labels) == required, f"{context}: comparison labels mismatch")
    require(all(c.get("pass") is True and c.get("mismatches") == 0
                and isinstance(c.get("elements"), int) and c["elements"] > 0 for c in checks),
            f"{context}: exact full-bit comparison failed")
    for check in checks:
        label = check["label"]
        width = (5120 if label.endswith("downstream_BF16") else
                 34816 if label.endswith("BF16") else
                 17408 if label.endswith("FP8") else 136)
        require(check["elements"] == m * width, f"{context}: comparison covers a partial output")


def output_kinds(arm):
    if arm in OFFICIAL_ARMS:
        return ("BF16", "FP8", "scales")
    if arm.startswith("paired_"):
        return ("paired_BF16",)
    return ("FP8", "scales")


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def analyze(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(all(len(r) == 1 and next(iter(r)) in {"metadata", "numerics", "timing", "profiled_once", "summary"}
                for r in rows), "unknown JSONL record or multi-key record")
    metas = [r["metadata"] for r in rows if "metadata" in r]
    summaries = [r["summary"] for r in rows if "summary" in r]
    require(len(metas) == len(summaries) == 1 and "metadata" in rows[0] and "summary" in rows[-1],
            "missing/duplicate metadata or final summary")
    meta, summary = metas[0], summaries[0]
    # Preserve the frozen v2 diagnostic contract, but never admit its timings.
    labels = set(meta.get("libraries_sha256", {}))
    contract = current_contract
    if labels & {"warp24x2", "warp24n64"}:
        require(meta.get("mode") == "gate", "v2 libraries are diagnostic gate only; timing/NCU rejected")
        contract = current_contract.pinned_import(
            Path(__file__).with_name("mainloop_probe_v2.py"),
            "2382be414ee4eca24030627ba26c3a8262603fb6ae0c73d64645ba05f88d135c",
            "frozen_mainloop_v2_contract")
    FUSED_ARMS, DIAGNOSTICS = contract.FUSED_ARMS, contract.DIAGNOSTICS
    VARIANTS, metadata_check, parse_arm = contract.VARIANTS, contract.metadata_check, contract.parse_arm
    require(meta.get("schema") == "sm120-mainloop-v1" and summary.get("pass") is True,
            "schema or successful summary missing")
    mode = meta.get("mode")
    require(mode in ("gate", "timing", "ncu") and summary.get("mode") == mode, "mode mismatch")
    arms = meta.get("arms", [])
    require(arms and len(set(arms)) == len(arms) and all(a in (*OFFICIAL_ARMS, *FUSED_ARMS, *DIAGNOSTICS) for a in arms),
            "invalid or duplicate selected arms")
    if mode != "ncu":
        require(set(OFFICIAL_ARMS) <= set(arms) and not set(arms) & set(DIAGNOSTICS),
                "all official complete-chain controls required")
    deps = meta.get("dependencies_sha256", {})
    require(set(deps) == {"scheduler_probe", "old_probe", "baseline", "quant", "probe"}
            and all(hash_ok(h) for h in deps.values()), "dependency/probe hashes incomplete")
    require(deps["scheduler_probe"] == SCHEDULER_SHA and deps["old_probe"] == PIPELINE_SHA
            and deps["baseline"] == BASELINE_SHA and deps["quant"] == QUANT_SHA, "fixed dependency lineage mismatch")
    require(hash_ok(meta.get("official_ops_sha256")) and meta.get("packed_roundtrip_pass") is True
            and set(meta.get("weight_sha256", {})) == {"canonical", "packed", "down"}
            and all(hash_ok(h) for h in meta["weight_sha256"].values()), "input/weight provenance incomplete")
    libraries = meta.get("libraries_sha256", {})
    loaded = meta.get("loaded_libraries", [])
    require(libraries and set(libraries) <= set(VARIANTS) and all(hash_ok(h) for h in libraries.values()),
            "library binary hashes incomplete")
    require(len(set(loaded)) == len(loaded) and set(loaded) <= set(libraries)
            and set(meta.get("library_metadata", {})) == set(loaded), "loaded library metadata mismatch")
    for label in loaded:
        metadata_check(label, meta["library_metadata"][label])
    needed = {parse_arm(a)[0] for a in arms if a not in OFFICIAL_ARMS}
    if mode != "ncu" or (not needed and meta.get("ncu_selected_arm") != "official_vllm"):
        needed.add("control42")
    require(set(loaded) == needed and meta.get("legacy_libraries_loaded") is False,
            "unexpected library loaded or selected library missing")
    if any(a in ("official_h8", "official_m1") for a in arms):
        require(meta.get("official_control_library") in loaded, "official control library missing")
    require(meta.get("minimum_free_bytes", 0) >= 8 * (1 << 30)
            and meta.get("repeats", 0) >= 31, "memory/pair guard weakened")
    require(meta.get("input_seed_rule") == "seed+100+M" and meta.get("changed_seed_rule") == "seed+9000+M",
            "seed rules changed")
    shapes = meta.get("timed_shapes", [])
    require(len(set(shapes)) == len(shapes) and all(m in NORMAL_M for m in shapes), "invalid timed shapes")
    expected_settings = {a: ([0, 8] if a in ("official_vllm", "official_h8") else
                             [1, 1] if a == "official_m1" else list(parse_arm(a)[2:])) for a in arms}
    require(meta.get("settings") == expected_settings, "raster/swizzle manifest mismatch")
    if mode == "ncu":
        require(len(shapes) == 1 and meta.get("ncu_selected_arm") in arms
                and set(arms) == {"official_vllm", meta["ncu_selected_arm"]}, "NCU selected-arm scope mismatch")
        expected_cases = [(shapes[0], "normal")]
        graph_shapes = shapes
    else:
        expected_cases = [(m, "normal") for m in (*FALLBACK_M, *NORMAL_M)] + [(257, d) for d in SPECIAL]
        graph_shapes = sorted(NORMAL_M)
    require(meta.get("cases") == [{"M": m, "distribution": d} for m, d in expected_cases]
            and meta.get("graph_shapes") == graph_shapes, "required shape/distribution/Graph coverage omitted")
    numeric = [r["numerics"] for r in rows if "numerics" in r]
    timings = [r["timing"] for r in rows if "timing" in r]
    windows = [r["profiled_once"] for r in rows if "profiled_once" in r]
    require([(r.get("M"), r.get("distribution")) for r in numeric] == expected_cases
            and summary.get("numeric_cases") == len(expected_cases), "missing/duplicate numerical case")
    gates = {}
    for gate in numeric:
        m, distribution = gate["M"], gate["distribution"]
        expected_arms = arms if m > 256 else ["official_vllm"]
        require(gate.get("pass") is True and gate.get("arms") == expected_arms
                and gate.get("fallback_only") is (m <= 256)
                and hash_ok(gate.get("input_sha256"))
                and gate.get("input_seed") == meta["input_seed"] + 100 + m, "numerical gate identity mismatch")
        labels = {f"{a}_{kind}" for a in expected_arms for kind in output_kinds(a)}
        labels |= {a + "_mainloop_paired_BF16" for a in expected_arms if a in FUSED_ARMS}
        if distribution == "normal":
            labels |= {a + "_downstream_BF16" for a in expected_arms if a not in DIAGNOSTICS}
        checks_pass(gate.get("comparisons"), labels, f"M{m}/{distribution}", m)
        graphs = gate.get("graph_replay", [])
        wants_graph = distribution == "normal" and m in graph_shapes
        require([g.get("arm") for g in graphs] == (expected_arms if wants_graph else []), "Graph arm coverage mismatch")
        for graph in graphs:
            require(all(graph.get(k) is True for k in ("pass", "input_bytes_changed", "input_scales_changed", "output_changed"))
                    and hash_ok(graph.get("changed_input_sha256"))
                    and graph["changed_input_sha256"] != gate["input_sha256"], "changed-input replay witness incomplete")
            checks_pass(graph.get("comparisons"), {f"{graph['arm']}_replay_{k}" for k in output_kinds(graph["arm"])}, "Graph", m)
        require(len({g["changed_input_sha256"] for g in graphs}) <= 1,
                "Graph arms replayed different changed input bytes/scales")
        gates[(m, distribution)] = gate
    require(summary.get("timed_shapes") == shapes and summary.get("ncu") is (mode == "ncu")
            and summary.get("timings_emitted") is (mode == "timing"), "summary mode/shape mismatch")
    result = {"scope": "synthetic operator evidence; no model/API gain or all-FP8-input proof",
              "metadata": meta, "numeric_cases": len(numeric), "coverage_pass": True,
              "selection_scope": "all explicitly selected arms; no winner-only filtering"}
    if mode == "gate":
        require(not timings and not windows and not shapes, "gate-only run contains timing/profile records")
        return result
    if mode == "ncu":
        require(not timings and len(windows) == 1, "NCU must contain exactly one window and no timing")
        w, m, arm = windows[0], shapes[0], meta["ncu_selected_arm"]
        require(w.get("M") == m and w.get("arm") == arm and w.get("call_count") == 1
                and w.get("nvtx") == "SM120_MAINLOOP:" + arm
                and w.get("input_seed") == meta["input_seed"] + 100 + m
                and w.get("input_sha256") == gates[(m, "normal")]["input_sha256"]
                and w.get("complete_pipeline") is (arm not in DIAGNOSTICS)
                and w.get("scope") == ("paired_BF16_GEMM_only" if arm in DIAGNOSTICS else "complete_chain"),
                "single NCU call/input/scope witness mismatch")
        result["profiled_once"] = w
        result["scope"] = "one NCU window witness; no unprofiled performance result"
        return result
    require(not windows and [t.get("M") for t in timings] == shapes, "missing/duplicate timed shape")
    # No timing is admissible before all required numeric gates have been emitted.
    last_gate = max(i for i, r in enumerate(rows) if "numerics" in r)
    require(all(i > last_gate for i, r in enumerate(rows) if "timing" in r), "timing preceded complete numerical gate")
    result["shapes"] = {}
    for row in timings:
        m = row["M"]
        require(row.get("pass") is True and row.get("full_pipeline") is True and row.get("arms") == arms
                and row.get("input_seed") == meta["input_seed"] + 100 + m
                and row.get("input_sha256") == gates[(m, "normal")]["input_sha256"]
                and row.get("free_bytes_before_capture", 0) >= meta["minimum_free_bytes"], "timed input/scope/memory mismatch")
        checks_pass(row.get("timing_capture_checks"), {f"{a}_timing_capture_{k}" for a in arms for k in output_kinds(a)}, "timing capture", m)
        by_mode = {}
        for cache_mode in ("warm", "cache_disturbed"):
            data = row[cache_mode]
            values, orders = data["values_us"], data["orders"]
            n = len(orders)
            require(n == meta["repeats"] and n >= 31 and data.get("pairs") == n
                    and set(values) == set(arms) and all(len(v) == n for v in values.values()), "raw paired timings incomplete")
            require(all(len(order) == len(arms) and set(order) == set(arms) for order in orders), "arm order missing/duplicate")
            require(all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for series in values.values() for v in series),
                    "nonfinite/nonpositive event time")
            arm_results = {}
            for arm in arms:
                series = values[arm]
                require(math.isclose(data["median_us"][arm], statistics.median(series), rel_tol=1e-12), "stored median differs from raw samples")
                comparisons = {}
                for baseline in OFFICIAL_ARMS:
                    ratios = [a / b for a, b in zip(values[baseline], series)]
                    comparisons[baseline] = {"baseline_over_arm_paired_median": statistics.median(ratios),
                                             "p10": percentile(ratios, .1), "p90": percentile(ratios, .9)}
                arm_results[arm] = {"median_us": statistics.median(series),
                                    "p10_us": percentile(series, .1), "p90_us": percentile(series, .9),
                                    "paired_ratios": comparisons}
            # Same scheduler old/new warp comparison, when both explicit arms were measured.
            layout_pairs = {}
            for arm in arms:
                if arm not in OFFICIAL_ARMS and parse_arm(arm)[0] != "control42":
                    variant = parse_arm(arm)[0]
                    control = arm.replace(variant + "_", "control42_", 1)
                    if control in values:
                        layout_pairs[arm] = statistics.median(a / b for a, b in zip(values[control], values[arm]))
            by_mode[cache_mode] = {"pairs": n, "arms": arm_results,
                                   "control42_over_candidate_same_scheduler_paired_median": layout_pairs}
        result["shapes"][str(m)] = by_mode
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = analyze(args.jsonl)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
