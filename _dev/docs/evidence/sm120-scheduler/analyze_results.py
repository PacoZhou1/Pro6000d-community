#!/usr/bin/env python3
"""CPU-only audit of scheduler_probe.py JSONL; never extrapolates to API speed."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ARMS = ("official", "legacy_v6") + tuple(
    f"v6_{raster}_s{swizzle}"
    for raster in ("heuristic", "along_m", "along_n")
    for swizzle in (1, 8)
)


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    diagnostic_metadata = [r["metadata"] for r in records if "metadata" in r and
                           r["metadata"].get("inverse_cost_separate") is True]
    if diagnostic_metadata and any(r.get("summary", {}).get("ncu") is True for r in records):
        numeric = [r["numerics"] for r in records if "numerics" in r]
        profiled = [r["profiled_once"] for r in records if "profiled_once" in r]
        if (len(numeric) != 1 or numeric[0].get("pass") is not True or
            len(profiled) != 1 or profiled[0].get("call_count") != 1):
            raise ValueError("diagnostic NCU numerical gate or single launch missing")
        return {"scope": "ordinary BF16 diagnostic NCU witness only; no timing contrast",
                "metadata": diagnostic_metadata[0], "profiled_once": profiled[0]}
    if any("diagnostic_timing" in r for r in records):
        return diagnostic_main(records)
    metadata = [r["metadata"] for r in records if "metadata" in r]
    numerics = [r["numerics"] for r in records if "numerics" in r]
    timings = [r["timing"] for r in records if "timing" in r]
    summaries = [r["summary"] for r in records if "summary" in r]
    if len(metadata) != 1 or len(summaries) != 1 or summaries[0].get("pass") is not True:
        raise ValueError("missing metadata or successful final summary")
    if not numerics or any(r.get("pass") is not True for r in numerics):
        raise ValueError("numerical gates missing or failed")
    if summaries[0].get("numeric_cases") != len(numerics):
        raise ValueError("numerical case count mismatch")
    expected_hashes = {"old_probe", "legacy_adapter", "baseline", "quant", "legacy_v6", "scheduler", "probe"}
    if set(metadata[0].get("sha256", {})) != expected_hashes or any(
        len(value) != 64 for value in metadata[0]["sha256"].values()
    ):
        raise ValueError("source and binary SHA manifest incomplete")
    if summaries[0].get("ncu"):
        profiled = [r["profiled_once"] for r in records if "profiled_once" in r]
        if len(numerics) != 1 or len(profiled) != 1 or timings:
            raise ValueError("NCU requires one selected numeric gate and one profile window")
        gate, window, meta = numerics[0], profiled[0], metadata[0]
        selected = window.get("arm")
        shape = window.get("M")
        if (selected not in ARMS or meta.get("timed_shapes") != [shape]
            or gate.get("M") != shape or gate.get("distribution") != "normal"
            or gate.get("input_seed") != meta.get("input_seed", -100) + 100 + shape
            or not gate.get("input_sha256")
            or gate.get("input_sha256") != window.get("input_sha256")
            or gate.get("input_seed") != window.get("input_seed")
            or window.get("call_count") != 1
            or window.get("complete_pipeline") is not True
            or window.get("nvtx") != f"SM120_SCHEDULER:{selected}"
            or summaries[0].get("numeric_cases") != 1):
            raise ValueError("NCU selected arm, input, or single-launch scope mismatch")
        required = {"official", selected}
        graph = gate.get("graph_replay", [])
        witnessed = {r.get("arm") for r in graph
                     if r.get("pass") is True and r.get("input_changed") is True
                     and r.get("output_changed") is True}
        if not required <= witnessed:
            raise ValueError("NCU selected arm lacks changed-input graph witness")
        labels = {c.get("label") for c in gate.get("comparisons", [])
                  if c.get("pass") is True}
        required_labels = {"baseline_vs_fixed_vllm_BF16",
                           "baseline_vs_fixed_vllm_FP8",
                           "baseline_vs_fixed_vllm_scales"}
        if selected == "legacy_v6":
            required_labels |= {"legacy_v6_FP8", "legacy_v6_scales"}
        elif selected != "official":
            required_labels |= {f"{selected}_FP8", f"{selected}_scales",
                                f"{selected}_downstream_BF16"}
        if not required_labels <= labels:
            raise ValueError("NCU selected arm numerical comparisons incomplete")
        if "numerical_arms" in meta:
            expected = tuple(arm for arm in ARMS if arm in required)
            if (tuple(meta["numerical_arms"]) != expected
                or set(meta.get("excluded_arms", [])) != set(ARMS) - required
                or set(gate.get("numerical_arms", [])) != required
                or set(gate.get("excluded_arms", [])) != set(ARMS) - required
                or witnessed != required
                or meta.get("ncu_selected_arm") != selected
                or meta.get("changed_input_graph_shapes") != [shape]):
                raise ValueError("NCU arm inclusion/exclusion manifest mismatch")
        return {"scope": "NCU window only; no performance contrast", "metadata": metadata[0],
                "numeric_cases": len(numerics), "profiled_once": window,
                "numerical_witness_arms": sorted(witnessed), "summary": summaries[0]}
    wanted_shapes = set(metadata[0]["timed_shapes"])
    if len(timings) != len(wanted_shapes) or {r["M"] for r in timings} != wanted_shapes:
        raise ValueError("missing or duplicate timed shape")
    result = {"scope": "synthetic same-input graph-replay microbenchmark, not model/API speed",
              "metadata": metadata[0], "numeric_cases": len(numerics), "shapes": {}}
    for row in timings:
        shape = str(row["M"])
        if row.get("full_pipeline") is not True or row.get("pass") is not True:
            raise ValueError(f"M={shape}: invalid timing scope")
        expected_seed = metadata[0]["input_seed"] + 100 + row["M"]
        matching_gates = [r for r in numerics if r["M"] == row["M"] and r["distribution"] == "normal"]
        if (len(matching_gates) != 1 or row.get("input_seed") != expected_seed or
            matching_gates[0].get("input_seed") != expected_seed or
            not row.get("input_sha256") or
            row["input_sha256"] != matching_gates[0].get("input_sha256")):
            raise ValueError(f"M={shape}: timed input seed/content differs from numerical gate")
        graph_records = matching_gates[0].get("graph_replay", [])
        if {r.get("arm") for r in graph_records if r.get("pass") is True and r.get("output_changed") is True} != set(ARMS):
            raise ValueError(f"M={shape}: changed-input graph witness missing for an arm")
        checks = row.get("timing_capture_checks", [])
        if len(checks) != len(ARMS) * 2 or any(c.get("pass") is not True for c in checks):
            raise ValueError(f"M={shape}: timing graph fresh-output check incomplete")
        shape_result = {}
        for mode in ("warm", "cache_disturbed"):
            payload = row[mode]
            values = payload["values_us"]
            orders = payload["orders"]
            if set(values) != set(ARMS):
                raise ValueError(f"M={shape} {mode}: missing arm")
            n = len(orders)
            if n < 31 or any(len(v) != n for v in values.values()):
                raise ValueError(f"M={shape} {mode}: fewer than 31 matched pairs")
            if any(set(order) != set(ARMS) or len(order) != len(ARMS) for order in orders):
                raise ValueError(f"M={shape} {mode}: arm order incomplete")
            if any(not math.isfinite(x) or x <= 0 for series in values.values() for x in series):
                raise ValueError(f"M={shape} {mode}: invalid event time")
            arms = {}
            for arm, series in values.items():
                ratios = [b / c for b, c in zip(values["official"], series)]
                arms[arm] = {
                    "median_us": statistics.median(series),
                    "p10_us": percentile(series, 0.1), "p90_us": percentile(series, 0.9),
                    "official_over_arm_paired_median": statistics.median(ratios),
                    "official_over_arm_paired_p10": percentile(ratios, 0.1),
                    "official_over_arm_paired_p90": percentile(ratios, 0.9),
                }
            shape_result[mode] = {"pairs": n, "arms": arms,
                                  "cache_scope": "owned cache disturbance, not fixed cold L2" if mode == "cache_disturbed" else "graph replay without explicit flush"}
        result["shapes"][shape] = shape_result
    return result


def diagnostic_main(records: list[dict]) -> dict:
    metadata = [r["metadata"] for r in records if "metadata" in r]
    numerics = [r["numerics"] for r in records if "numerics" in r]
    timings = [r["diagnostic_timing"] for r in records if "diagnostic_timing" in r]
    summaries = [r["summary"] for r in records if "summary" in r]
    if (len(metadata) != 1 or len(numerics) != 1 or len(timings) != 1 or
        len(summaries) != 1 or summaries[0].get("pass") is not True or
        numerics[0].get("pass") is not True):
        raise ValueError("diagnostic metadata, exact numerical gate or timing missing")
    meta, row = metadata[0], timings[0]
    if (row["M"] != meta["M"] or row["raster"] != meta["raster"] or
        row["swizzle"] != meta["swizzle"] or
        row["input_sha256"] != meta["input_sha256"] or
        row["input_seed"] != meta["seed"] + 100 + meta["M"]):
        raise ValueError("diagnostic shape/control/input mismatch")
    if len(row.get("capture_checks", [])) != 6 or any(
        c.get("pass") is not True for c in row["capture_checks"]
    ):
        raise ValueError("diagnostic timing graph fresh-output check incomplete")
    expected = {"old_pipeline", "new_official_gemm", "new_official_pipeline",
                "paired_gemm", "inverse_only", "quant_only", "paired_pipeline"}
    if {g.get("arm") for g in numerics[0].get("changed_input_graph", [])
        if g.get("pass") is True and g.get("output_changed") is True} != {
            "old_pipeline", "new_official_pipeline", "paired_pipeline"}:
        raise ValueError("diagnostic changed-input graph witness incomplete")
    output = {"scope": "ordinary BF16 epilogue diagnostic; pipeline arms are not fused",
              "metadata": meta, "modes": {}}
    for mode in ("warm", "cache_disturbed"):
        payload = row[mode]
        values, orders = payload["values_us"], payload["orders"]
        if set(values) != expected or len(orders) < 31 or any(len(v) != len(orders) for v in values.values()):
            raise ValueError(f"diagnostic {mode}: incomplete paired records")
        if any(set(order) != expected or len(order) != len(expected) for order in orders):
            raise ValueError(f"diagnostic {mode}: incomplete order")
        if any(not math.isfinite(v) or v <= 0 for series in values.values() for v in series):
            raise ValueError(f"diagnostic {mode}: invalid event time")
        output["modes"][mode] = {
            "pairs": len(orders),
            "median_us": {name: statistics.median(series) for name, series in values.items()},
            "paired_old_over_new_official_pipeline_median": statistics.median(
                a / b for a, b in zip(values["old_pipeline"], values["new_official_pipeline"])),
            "paired_old_over_paired_pipeline_median": statistics.median(
                a / b for a, b in zip(values["old_pipeline"], values["paired_pipeline"])),
            "no_subtractive_attribution": True,
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = main(args.jsonl)
    rendered = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
