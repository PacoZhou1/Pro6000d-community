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
        return {"scope": "NCU window only; no performance contrast", "metadata": metadata[0],
                "numeric_cases": len(numerics), "summary": summaries[0]}
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
