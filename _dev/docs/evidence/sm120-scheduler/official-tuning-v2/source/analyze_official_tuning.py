#!/usr/bin/env python3
"""CPU-only strict audit of official_tuning_probe.py JSONL."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from analyze_results import percentile

ARMS = {"old_h8", "new_h8", "new_along_m_s1"}


def main(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metadata = [r["metadata"] for r in records if "metadata" in r]
    numerics = [r["numerics"] for r in records if "numerics" in r]
    timings = [r["timing"] for r in records if "timing" in r]
    summaries = [r["summary"] for r in records if "summary" in r]
    if (len(metadata) != 1 or len(summaries) != 1 or
        summaries[0].get("pass") is not True or
        summaries[0].get("timings_emitted") is not True):
        raise ValueError("missing successful metadata/summary")
    meta = metadata[0]
    shapes = meta.get("shapes", [])
    if (not shapes or len(shapes) != len(set(shapes)) or
        set(meta.get("arms", [])) != ARMS or
        len(numerics) != len(shapes) or len(timings) != len(shapes) or
        summaries[0].get("numeric_cases") != len(shapes) or
        summaries[0].get("timed_shapes") != shapes or
        {r.get("M") for r in numerics} != set(shapes) or
        {r.get("M") for r in timings} != set(shapes)):
        raise ValueError("shape/arm coverage incomplete")
    hashes = meta.get("sha256", {})
    if set(hashes) != {"old_probe", "baseline", "quant", "scheduler", "probe"} or any(
        not isinstance(x, str) or len(x) != 64 for x in hashes.values()
    ):
        raise ValueError("source and binary hashes missing")
    output = {"scope": "same-process synthetic complete-chain contrast, not model/API speed",
              "metadata": meta, "shapes": {}}
    for m in shapes:
        gate = next(r for r in numerics if r["M"] == m)
        row = next(r for r in timings if r["M"] == m)
        seed = meta["seed"] + 100 + m
        if (gate.get("pass") is not True or row.get("pass") is not True or
            row.get("full_pipeline") is not True or
            gate.get("input_seed") != seed or row.get("input_seed") != seed or
            not gate.get("input_sha256") or
            gate["input_sha256"] != row.get("input_sha256")):
            raise ValueError(f"M={m}: numerical/timed input or scope mismatch")
        labels = {c.get("label") for c in gate.get("comparisons", []) if c.get("pass") is True}
        expected_labels = {f"{arm}_{kind}" for arm in ARMS
                           for kind in ("BF16", "FP8", "scales")}
        if labels != expected_labels:
            raise ValueError(f"M={m}: full-bit BF16/FP8/scale witness incomplete")
        graph = gate.get("changed_input_graph", [])
        if {g.get("arm") for g in graph if g.get("pass") is True and
            g.get("input_changed") is True and g.get("output_changed") is True} != ARMS:
            raise ValueError(f"M={m}: changed-input graph witness incomplete")
        capture = row.get("capture_checks", [])
        if {c.get("label") for c in capture if c.get("pass") is True} != {
            f"{arm}_capture_{kind}" for arm in ARMS
            for kind in ("BF16", "FP8", "scales")
        }:
            raise ValueError(f"M={m}: fresh-output timing capture check incomplete")
        modes = {}
        for mode in ("warm", "cache_disturbed"):
            payload = row.get(mode, {})
            values, orders = payload.get("values_us", {}), payload.get("orders", [])
            n = len(orders)
            if (n < 31 or payload.get("pairs") != n or set(values) != ARMS or
                any(len(series) != n for series in values.values()) or
                any(set(order) != ARMS or len(order) != len(ARMS) for order in orders) or
                any(not math.isfinite(x) or x <= 0 for series in values.values() for x in series)):
                raise ValueError(f"M={m} {mode}: incomplete matched timing")
            modes[mode] = {"pairs": n,
                           "arms": {arm: {"median_us": statistics.median(series),
                                          "p10_us": percentile(series, 0.1),
                                          "p90_us": percentile(series, 0.9),
                                          "old_over_arm_paired_median": statistics.median(
                                              a / b for a, b in zip(values["old_h8"], series)),
                                          "new_h8_over_arm_paired_median": statistics.median(
                                              a / b for a, b in zip(values["new_h8"], series))}
                                    for arm, series in values.items()},
                           "cache_scope": "owned disturbance, not fixed cold L2" if mode == "cache_disturbed" else "warm graph replay"}
        output["shapes"][str(m)] = {"input_sha256": gate["input_sha256"],
                                     "input_seed": seed, "modes": modes}
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = main(args.jsonl)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
