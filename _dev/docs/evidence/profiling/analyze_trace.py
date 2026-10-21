"""Summarize one or more PyTorch Chrome traces without mixing request labels.

Chrome trace timestamps and durations are microseconds. Kernel categories are
name-based hypotheses, not confirmed model modules. GPU kernel duration sums
can exceed both the GPU busy union and trace span when streams overlap.
"""

import argparse
import gzip
import json
import math
import pathlib
import re
from collections import Counter, defaultdict


# Ordered, deliberately conservative name rules. Every match remains inferred.
CATEGORY_RULES = (
    ("GDN", "gdn/gated-delta/mamba/linear-attn name",
     re.compile(r"gdn|gated.?delta|delta.?net|mamba|linear.?attn|recurrent", re.I)),
    ("attention", "attention/flash-attn/XQA/FMHA name",
     re.compile(r"attention|flash.?attn|\bxqa\b|\bfmha\b|paged.?attn", re.I)),
    ("sampling", "sampling/top-k/argmax name",
     re.compile(r"sampl|top.?k|top.?p|multinomial|argmax", re.I)),
    ("state-copy", "copy/scatter/gather/cache-update name",
     re.compile(r"copy|scatter|gather|index.?select|cache.?update|state.?update|swap", re.I)),
    ("GEMM/linear", "GEMM/matmul/CUTLASS/cuBLAS/linear name",
     re.compile(r"gemm|matmul|cutlass|cublas|(?:^|[_:.])mm(?:[_:.0-9]|$)|linear", re.I)),
    ("norm/elementwise", "norm/activation/pointwise/softmax name",
     re.compile(r"norm|elementwise|pointwise|softmax|silu|gelu|swiglu|activation|fused.?add", re.I)),
)
CPU_CUDA_API = re.compile(r"^cuda(?:Launch|GraphLaunch|DeviceSynchronize|StreamSynchronize|"
                          r"EventSynchronize|Memcpy|Memset|Malloc|Free)", re.I)


def event_kind(event):
    """Use trace category/device evidence, never a kernel-looking CPU name."""
    if event.get("ph") != "X":
        return None
    name = str(event.get("name", ""))
    categories = set(re.split(r"[,;\s]+", str(event.get("cat", "")).lower()))
    if categories & {"cuda_runtime", "cuda_driver", "gpu_runtime", "runtime"}:
        return "cpu_runtime"
    if categories & {"gpu_memcpy", "gpu_memset", "cuda_memcpy", "cuda_memset", "memcpy", "memset"}:
        return "gpu_memcpy"
    if CPU_CUDA_API.search(name):
        return "cpu_runtime"
    if categories & {"cpu_op", "cpu_operator", "operator"}:
        return "cpu_op"
    if categories & {"kernel", "gpu_kernel", "cuda_kernel"}:
        return "gpu_kernel"
    return "other"


def kernel_category(name):
    for category, rule, pattern in CATEGORY_RULES:
        if pattern.search(name):
            return category, rule
    return "other", "no name rule matched"


def interval_union(events):
    intervals = sorted((event["ts"], event["ts"] + event["dur"]) for event in events)
    if not intervals:
        return 0.0
    start, end = intervals[0]
    total = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def valid_complete(event):
    if event.get("ph") != "X":
        return False
    ts, dur = event.get("ts"), event.get("dur")
    return (isinstance(ts, (int, float)) and not isinstance(ts, bool) and
            isinstance(dur, (int, float)) and not isinstance(dur, bool) and
            math.isfinite(ts) and math.isfinite(dur) and dur >= 0)


def ranking(events, with_category=False):
    grouped = defaultdict(lambda: [0, 0.0])
    for event in events:
        entry = grouped[str(event.get("name", "<unnamed>"))]
        entry[0] += 1
        entry[1] += event["dur"]
    rows = []
    for name, (count, total) in grouped.items():
        row = {"name": name, "count": count, "total_duration_us": total,
               "mean_duration_us": total / count}
        if with_category:
            category, rule = kernel_category(name)
            row.update({"category": category, "category_rule": rule,
                        "category_inferred_from_name": True})
        rows.append(row)
    return sorted(rows, key=lambda row: (-row["total_duration_us"], row["name"]))


def _device(event):
    args = event.get("args") or {}
    if not isinstance(args, dict):
        return None
    for key in ("device", "device_index", "Device Id", "deviceId"):
        if key in args:
            return args[key]
    return None


def analyze_trace_data(data, label, selected_pids=None, top=0):
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    if not isinstance(events, list):
        raise ValueError("traceEvents must be a list")
    metadata_events = [event for event in events if isinstance(event, dict)
                       and event.get("ph") == "M"]
    complete = [event for event in events if isinstance(event, dict) and valid_complete(event)]
    pid_counts = defaultdict(Counter)
    for event in complete:
        pid_counts[str(event.get("pid", "<missing>"))][event_kind(event)] += 1
    for event in metadata_events:
        pid_counts.setdefault(str(event.get("pid", "<missing>")), Counter())
    available_pids = [{"pid": pid, "event_counts": dict(counts)}
                      for pid, counts in sorted(pid_counts.items())]
    chosen = None if selected_pids is None else {str(pid) for pid in selected_pids}
    if chosen is not None and not chosen.issubset(pid_counts):
        raise ValueError("selected PID absent from trace: " +
                         ", ".join(sorted(chosen - set(pid_counts))))
    selected = [event for event in complete if chosen is None or
                str(event.get("pid", "<missing>")) in chosen]
    by_kind = defaultdict(list)
    for event in selected:
        by_kind[event_kind(event)].append(event)
    kernels = by_kind["gpu_kernel"]
    copies = by_kind["gpu_memcpy"]
    kernel_sum = sum(event["dur"] for event in kernels)
    categories = defaultdict(lambda: {"count": 0, "sum_duration_us": 0.0,
                                      "rule": None, "inferred_from_name": True})
    for event in kernels:
        category, rule = kernel_category(str(event.get("name", "<unnamed>")))
        categories[category]["count"] += 1
        categories[category]["sum_duration_us"] += event["dur"]
        categories[category]["rule"] = rule if categories[category]["rule"] in (None, rule) else "multiple name rules"
    category_rows = [{"category": key, **value,
                      "share_of_kernel_duration_sum": value["sum_duration_us"] / kernel_sum if kernel_sum else None}
                     for key, value in categories.items()]
    category_rows.sort(key=lambda row: (-row["sum_duration_us"], row["category"]))
    raw_kernels = ranking(kernels, with_category=True)
    if top:
        shown_kernels = raw_kernels[:top]
    else:
        shown_kernels = raw_kernels
    span = (min(event["ts"] for event in selected),
            max(event["ts"] + event["dur"] for event in selected)) if selected else None
    gpu_span = (min(event["ts"] for event in kernels + copies),
                max(event["ts"] + event["dur"] for event in kernels + copies)) if kernels or copies else None
    trace_metadata = {}
    if isinstance(data, dict):
        trace_metadata = {key: data[key] for key in ("schemaVersion", "displayTimeUnit", "deviceProperties")
                          if key in data}
    return {
        "label": label,
        "time_unit": "microseconds (Chrome trace ts/dur convention)",
        "trace_metadata": trace_metadata,
        "metadata_events": [event for event in metadata_events if chosen is None or
                            str(event.get("pid", "<missing>")) in chosen],
        "available_pids": available_pids,
        "selected_pids": sorted(chosen) if chosen is not None else None,
        "available_gpu_devices_in_selected_events": sorted({str(device) for event in kernels + copies
                                                           if (device := _device(event)) is not None}),
        "event_counts": {kind: len(by_kind[kind]) for kind in
                         ("gpu_kernel", "gpu_memcpy", "cpu_op", "cpu_runtime", "other")},
        "ignored_non_complete_or_invalid_events": len(events) - len(complete) - len(metadata_events),
        "trace_window": None if span is None else {
            "first_ts_us": span[0], "last_end_ts_us": span[1], "span_us": span[1] - span[0],
            "meaning": "Span of selected complete events; not request latency unless capture boundaries match the request."},
        "gpu_activity_span": None if gpu_span is None else {
            "first_ts_us": gpu_span[0], "last_end_ts_us": gpu_span[1],
            "span_us": gpu_span[1] - gpu_span[0]},
        "gpu_kernel_sum_duration_us": kernel_sum,
        "gpu_kernel_busy_union_us": interval_union(kernels),
        "gpu_memcpy_sum_duration_us": sum(event["dur"] for event in copies),
        "gpu_activity_busy_union_us": interval_union(kernels + copies),
        "cpu_op_sum_duration_us": sum(event["dur"] for event in by_kind["cpu_op"]),
        "cpu_runtime_sum_duration_us": sum(event["dur"] for event in by_kind["cpu_runtime"]),
        "duration_interpretation": "Summed durations can overlap across streams and CPU threads. GPU busy union merges overlapping GPU kernel intervals; GPU activity union also includes GPU memcpy/memset. Neither sum is wall time.",
        "kernel_category_basis": "Inferred from raw kernel names by ordered regex rules; verify model-module attribution with profiler correlation and dispatch/shape evidence.",
        "kernel_categories": category_rows,
        "raw_kernel_ranking": shown_kernels,
        "raw_kernel_ranking_total_names": len(raw_kernels),
        "unrecognized_kernel_names": [row for row in raw_kernels if row["category"] == "other"],
        "gpu_memcpy_ranking": ranking(copies),
        "cpu_op_ranking": ranking(by_kind["cpu_op"]),
        "cpu_runtime_ranking": ranking(by_kind["cpu_runtime"]),
    }


def load_trace(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=pathlib.Path, help="Chrome trace JSON or JSON.gz")
    parser.add_argument("--label", action="append", required=True,
                        help="One explicit request label per trace, in input order")
    parser.add_argument("--pid", action="append", help="Select a trace PID; repeat to select several")
    parser.add_argument("--pid-for", action="append", metavar="LABEL:PID",
                        help="Select a PID only for a named trace; repeat as needed")
    parser.add_argument("--top", type=int, default=0, help="Limit raw kernel ranking; 0 means all")
    parser.add_argument("--output", type=pathlib.Path, help="Save the JSON report")
    args = parser.parse_args()
    if len(args.label) != len(args.traces) or len(set(args.label)) != len(args.label):
        parser.error("provide one unique --label for each trace")
    if args.top < 0:
        parser.error("--top must be nonnegative")
    label_pids = defaultdict(list)
    for item in args.pid_for or []:
        label, sep, pid = item.rpartition(":")
        if not sep or label not in args.label or not pid:
            parser.error("--pid-for must be LABEL:PID for a supplied --label")
        label_pids[label].append(pid)
    analyses = []
    for path, label in zip(args.traces, args.label):
        pids = list(args.pid or []) + label_pids[label]
        analysis = analyze_trace_data(load_trace(path), label, pids or None, args.top)
        analysis["source_trace"] = str(path)
        analyses.append(analysis)
    report = {"analyses": analyses, "cross_trace_aggregate": None,
              "note": "Each labelled request trace is summarized independently; prefill and decode are never combined."}
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
