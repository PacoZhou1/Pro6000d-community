#!/usr/bin/env python3
"""Sum full pipeline NCU bytes using explicit, reviewed kernel-ID arm maps.

Export with ncu --import REPORT.ncu-rep --csv --page raw --print-units base.
Pass each complete profiler-window kernel ID set using --baseline-ids and
--candidate-ids. The operator must obtain those sets from NVTX/window evidence;
this utility never guesses an arm from a kernel name. Duplicate metric rows,
missing DRAM read/write metrics, overlapping arms or unknown units fail closed.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

READ = "dram__bytes_op_read.sum"
WRITE = "dram__bytes_op_write.sum"
DURATION = "gpu__time_duration.sum"
L2 = "lts__t_bytes.sum"


def ids(value):
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("nonempty, unique comma-separated IDs required")
    return result


def read_csv(path):
    lines = path.read_text().splitlines()
    # NCU may prepend warning/banner lines to a raw CSV export.
    begin = next((index for index, line in enumerate(lines)
                  if line.startswith('"ID",') or line.startswith("ID,")), None)
    if begin is None:
        raise ValueError("CSV header ID not found")
    parsed = list(csv.DictReader(io.StringIO("\n".join(lines[begin:]))))
    if parsed and "Metric Name" in parsed[0]:
        return parsed
    # NCU 2026.3 exports a wide raw table: one units row, one row per kernel.
    # Normalize measured cells without inferring units or inventing metrics.
    if not parsed or parsed[0].get("ID") != "":
        raise ValueError("wide CSV must contain an explicit units row")
    units, kernels = parsed[0], parsed[1:]
    normalized = []
    for kernel in kernels:
        if not kernel.get("ID"):
            raise ValueError("wide CSV kernel ID missing")
        for metric in (READ, WRITE, DURATION, L2):
            if metric in kernel:
                normalized.append({
                    "ID": kernel["ID"], "Kernel Name": kernel["Kernel Name"],
                    "Metric Name": metric, "Metric Unit": units[metric],
                    "Metric Value": kernel[metric],
                })
    return normalized


def metric_value(row, metric):
    expected_units = {READ: {"byte", "bytes"}, WRITE: {"byte", "bytes"},
                      L2: {"byte", "bytes"}, DURATION: {"nsecond", "ns"}}
    if row["Metric Unit"] not in expected_units[metric]:
        raise ValueError(f"ID={row['ID']} metric={metric}: require base units; got {row['Metric Unit']}")
    # NCU comma-separated thousands remain inside quoted CSV cells.
    return float(row["Metric Value"].replace(",", ""))


def summarize(rows, selected):
    collected = {key: {} for key in selected}
    for row in rows:
        key = row.get("ID")
        metric = row.get("Metric Name")
        if key not in collected or metric not in (READ, WRITE, DURATION, L2):
            continue
        if metric in collected[key]:
            raise ValueError(f"duplicate metric ID={key}, metric={metric}; choose one complete-pipeline sample")
        collected[key][metric] = {"value": metric_value(row, metric), "kernel": row.get("Kernel Name", "")}
    for key, values in collected.items():
        if not all(metric in values for metric in (READ, WRITE)):
            raise ValueError(f"kernel ID={key}: missing DRAM read/write metrics")
    total = {metric: sum(values[metric]["value"] for values in collected.values())
             for metric in (READ, WRITE, DURATION, L2)
             if all(metric in values for values in collected.values())}
    return {"kernel_ids": selected, "kernel_count": len(selected),
            "kernels": collected, "sum_metrics_base_units": total,
            "dram_total_bytes": total[READ] + total[WRITE]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--baseline-ids", required=True, type=ids)
    parser.add_argument("--candidate-ids", required=True, type=ids)
    parser.add_argument("--window-evidence", required=True, help="NVTX/window evidence file or explanation proving arm membership")
    args = parser.parse_args()
    if set(args.baseline_ids) & set(args.candidate_ids):
        parser.error("baseline and candidate kernel IDs overlap")
    rows = read_csv(args.csv)
    baseline = summarize(rows, args.baseline_ids)
    candidate = summarize(rows, args.candidate_ids)
    ratio = {}
    for metric in (READ, WRITE, DURATION, L2):
        b = baseline["sum_metrics_base_units"].get(metric)
        c = candidate["sum_metrics_base_units"].get(metric)
        ratio[metric] = None if b is None or c is None or b == 0 else c / b
    b = baseline["dram_total_bytes"]
    ratio["dram_total_bytes"] = None if b == 0 else candidate["dram_total_bytes"] / b
    result = {"csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
              "window_evidence": args.window_evidence,
              "denominator": "one complete GEMM→SwiGLU/blockquant pipeline per arm",
              "baseline": baseline, "candidate": candidate,
              "candidate_over_baseline_ratios": ratio,
              "duration_unit": "nsecond when gpu__time_duration.sum is available",
              "scope": "synthetic pipeline NCU metrics; profiler replay policy recorded externally"}
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
