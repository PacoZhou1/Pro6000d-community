#!/usr/bin/env python3
"""Validate saved scheduler NCU jobs and normalize wide CSV counters, CPU only.

Membership comes from the frozen command and one explicit NVTX/profile event;
kernel names only corroborate that contract. Failed/missing jobs remain in the
inventory with no usable counters. Profiler durations are not API speedups, and
total DRAM counters do not identify B or any other allocation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

READ = "dram__bytes_op_read.sum"
WRITE = "dram__bytes_op_write.sum"
DURATION = "gpu__time_duration.sum"
L2 = "lts__t_bytes.sum"
TENSOR = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
CORE = {READ, WRITE, DURATION, L2, TENSOR, "smsp__inst_executed.sum",
        "smsp__inst_executed_pipe_tensor.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"}
RESOURCE_CORE = {"launch__registers_per_thread", "launch__shared_mem_per_block",
                 "launch__shared_mem_per_block_dynamic", "launch__shared_mem_per_block_driver",
                 "launch__block_size"}
SHA_FLAGS = {"baseline": "--baseline-sha256", "quant": "--quant-sha256",
             "scheduler": "--scheduler-sha256", "old_probe": "--old-probe-sha256",
             "legacy_v6": "--legacy-sha256", "legacy_adapter": "--legacy-adapter-sha256"}
BYTE_FACTORS = {"byte": 1, "bytes": 1, "Kbyte": 1e3, "Mbyte": 1e6,
                "Gbyte": 1e9, "Tbyte": 1e12}
TIME_FACTORS = {"ns": 1, "nsecond": 1, "us": 1e3, "usecond": 1e3,
                "ms": 1e6, "msecond": 1e6, "s": 1e9, "second": 1e9}
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def flag(argv: list[str], name: str, default=None):
    positions = [i for i, item in enumerate(argv) if item == name]
    require(len(positions) <= 1, f"duplicate flag {name}")
    if not positions:
        return default
    i = positions[0]
    require(i + 1 < len(argv), f"missing flag value {name}")
    return argv[i + 1]


def events(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        if not line.lstrip().startswith("{"):
            continue  # NCU banners/progress are not probe JSONL.
        obj = json.loads(line)
        require(isinstance(obj, dict) and len(obj) == 1, "malformed probe JSON event")
        key, value = next(iter(obj.items()))
        require(isinstance(value, dict), f"malformed {key} event")
        out.setdefault(key, []).append(value)
    return out


def one(data, key):
    require(len(data.get(key, [])) == 1, f"require one {key} event")
    return data[key][0]


def contract(argv: list[str]) -> dict:
    arm, target = flag(argv, "--ncu-arm"), flag(argv, "--ncu-target")
    require(bool(arm) != bool(target), "require one --ncu-arm or --ncu-target")
    name = arm or target
    if arm == "official" or target == "old_pipeline":
        roles, scope = ["old_official_gemm", "v2_quant"], "complete_pipeline"
    elif arm == "legacy_v6":
        roles, scope = ["legacy_fused_gemm", "pack"], "complete_pipeline"
    elif arm and re.fullmatch(r"v6_(heuristic|along_m|along_n)_s(1|8)", arm):
        roles, scope = ["new_fused_gemm", "pack"], "complete_pipeline"
    elif target == "paired_gemm":
        roles, scope = ["paired_bf16_gemm"], "gemm_only_diagnostic"
    elif target == "new_official_gemm":
        roles, scope = ["new_official_gemm"], "gemm_only_diagnostic"
    elif target == "new_official_pipeline":
        roles, scope = ["new_official_gemm", "v2_quant"], "complete_pipeline"
    else:
        raise ValueError(f"unsupported capture contract: {name}")
    m = flag(argv, "--m") if target else flag(argv, "--timed-m")
    require(m is not None and re.fullmatch(r"[0-9]+", m) is not None, "capture needs exactly one explicit M")
    return {"target": name, "kind": "scheduler" if arm else "epilogue",
            "M": int(m), "expected_kernel_count": len(roles), "kernel_roles": roles,
            "scope": scope, "nvtx": "SM120_SCHEDULER:" + name}


def check_comparisons(checks, context):
    require(isinstance(checks, list) and bool(checks), f"missing {context} comparisons")
    for item in checks:
        require(item.get("pass") is True and item.get("mismatches") == 0 and
                isinstance(item.get("elements"), int) and item["elements"] > 0,
                f"failed/non-full-bit {context} comparison")


def validate_probe(job: Path, status: dict, argv: list[str], expected: dict) -> dict:
    require(isinstance(argv, list) and all(isinstance(x, str) for x in argv), "invalid command.json")
    require(status["command"][-len(argv):] == argv, "saved command differs from actual controller command")
    cmd = status["command"]
    require(flag(cmd, "--profile-from-start") == "off", "profiling must start explicitly")
    require("--nvtx" in cmd and flag(cmd, "--nvtx-include") == "regex:SM120_SCHEDULER:.*/",
            "unexpected NVTX include filter")
    require(flag(cmd, "--clock-control") == "none", "clock policy differs from declared jobs")
    cache = flag(cmd, "--cache-control")
    require(cache in ("none", "all"), "missing/unsupported cache policy")
    source_hashes = status.get("source_sha256", {})
    require(bool(source_hashes), "missing source snapshot manifest")
    for name, digest in source_hashes.items():
        require(Path(name).name == name and SHA_PATTERN.fullmatch(digest) is not None,
                "invalid source manifest entry")
        require(sha(job / "source" / name) == digest, f"source snapshot SHA mismatch: {name}")
    data = events(job / "stdout.log")
    metadata, window, summary = (one(data, k) for k in ("metadata", "profiled_once", "summary"))
    require(not data.get("timing") and not data.get("diagnostic_timing"), "capture contains timing samples")
    require(summary.get("pass") is True and summary.get("ncu") is True and
            summary.get("timings_emitted") is False, "failed/missing final NCU summary")
    require(window.get("call_count") == 1 and window.get("nvtx") == expected["nvtx"] and
            window.get("M") == expected["M"], "capture is not one declared NVTX call")
    require((window.get("arm") or window.get("target")) == expected["target"], "profiled target mismatch")
    if expected["kind"] == "scheduler":
        require(window.get("complete_pipeline") is True, "scheduler window is not complete pipeline")
    numeric = data.get("numerics", [])
    require(bool(numeric) and all(x.get("pass") is True for x in numeric), "numeric gate failed or missing")
    graphs = []
    for item in numeric:
        if expected["kind"] == "scheduler":
            require(item.get("M") == expected["M"], "numeric shape mismatch")
        check_comparisons(item.get("comparisons"), "numeric")
        replay = item.get("graph_replay", item.get("changed_input_graph", []))
        require(bool(replay), "changed-input Graph gate missing")
        for g in replay:
            require(g.get("pass") is True and g.get("output_changed") is True,
                    "Graph replay/output-change gate failed")
            if "input_changed" in g:
                require(g["input_changed"] is True, "Graph input did not change")
            check_comparisons(g.get("comparisons"), "Graph")
        graphs += replay
    graph_target = {"paired_gemm": "paired_pipeline", "new_official_gemm": "new_official_pipeline",
                    "new_official_pipeline": "new_official_pipeline", "old_pipeline": "old_pipeline"}.get(
                        expected["target"], expected["target"])
    require(graph_target in {g.get("arm") for g in graphs}, "selected arm has no changed-input Graph gate")
    if expected["kind"] == "scheduler":
        require(metadata.get("timed_shapes") == [expected["M"]] and
                metadata.get("numeric_shapes") == [expected["M"]] and
                metadata.get("changed_input_graph_shapes") == [expected["M"]], "effective shape gates differ")
    else:
        require(metadata.get("M") == expected["M"] and metadata.get("N") == 34816 and
                metadata.get("K") == 5120, "epilogue metadata shape mismatch")
        require(metadata.get("raster") == flag(argv, "--raster", "heuristic") and
                metadata.get("swizzle") == int(flag(argv, "--swizzle", "8")), "epilogue scheduler metadata mismatch")
    hashes = metadata.get("sha256", {})
    probe_name = Path(next(x for x in argv if x.endswith("_probe.py"))).name
    require(hashes.get("probe") == source_hashes.get(probe_name), "executed probe SHA differs from snapshot")
    for key in ("old_probe", "baseline", "quant", "scheduler"):
        wanted = flag(argv, SHA_FLAGS[key])
        require(wanted is not None and SHA_PATTERN.fullmatch(wanted) is not None and
                hashes.get(key) == wanted, f"missing/mismatched library/source SHA: {key}")
    for key in ("legacy_v6", "legacy_adapter"):
        wanted = flag(argv, SHA_FLAGS[key])
        if wanted is not None:
            require(hashes.get(key) == wanted, f"SHA mismatch: {key}")
    library_path = flag(argv, "--scheduler-library")
    build_manifest = job.parent / Path(library_path).parent.name / "build-manifest.json"
    require(build_manifest.exists(), "local compiled scheduler manifest missing")
    build = json.loads(build_manifest.read_text())
    require(build.get("status") == "compiled" and build.get("compile_exit_code") == 0 and
            build.get("library_sha256") == hashes["scheduler"], "scheduler library does not match compiled manifest")
    require(metadata.get("scheduler_abi") == 1 and
            metadata.get("scheduler_metadata", {}).get("abi_version") == 1, "scheduler ABI metadata mismatch")
    input_sha = window.get("input_sha256", metadata.get("input_sha256"))
    require(isinstance(input_sha, str) and SHA_PATTERN.fullmatch(input_sha) is not None,
            "capture input hash unavailable")
    return {"metadata": metadata, "profiled_once": window, "numeric_cases": len(numeric),
            "graph_arms": [g["arm"] for g in graphs], "input_sha256": input_sha,
            "library_sha256": hashes, "source_sha256": source_hashes,
            "build_manifest_sha256": sha(build_manifest), "cache_control": cache,
            "clock_control": flag(cmd, "--clock-control"), "controller_sha256": status.get("controller_sha256"),
            "nvtx_membership_evidence": "one profiled_once event + explicit profiler/NVTX filter + frozen source + exact kernel count"}


def normalize(value: str, unit: str) -> dict:
    require(value != "", "empty measured metric")
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        return {"value": value, "unit": unit, "numeric": False}
    require(math.isfinite(number), "non-finite measured metric")
    numerator, slash, denominator = unit.partition("/")
    if numerator in BYTE_FACTORS:
        factor, base = BYTE_FACTORS[numerator], "byte"
    elif unit in TIME_FACTORS:
        factor, base = TIME_FACTORS[unit], "ns"
    else:
        factor, base = 1, numerator
    return {"value": number, "unit": unit, "base_value": number * factor,
            "base_unit": base + (slash + denominator if slash else ""), "numeric": True}


def role_matches(role: str, kernel: str) -> bool:
    # NCU exports some entries with full namespaces and others as shortened
    # template names. The frozen command/window establishes arm membership.
    compact = re.sub(r"(?:\b[A-Za-z_]\w*::)+", "", kernel).replace(" ", "")
    n128 = "tuple<C<128>,C<128>,C<128>>" in compact
    n256 = "tuple<C<64>,C<256>,C<128>>" in compact
    if role == "v2_quant":
        return "silu_quant" in kernel
    if role == "pack":
        return "pack_gateup_fp8" in kernel
    if role == "old_official_gemm":
        return "pro6000d_scheduler_v1::" not in kernel and "bfloat16_t" in kernel and n128
    if role == "new_official_gemm":
        return "vllm::" not in kernel and "bfloat16_t" in kernel and n128
    if role == "paired_bf16_gemm":
        return "vllm::" not in kernel and "bfloat16_t" in kernel and n256
    if role == "new_fused_gemm":
        return "vllm::" not in kernel and "SchedulerPairedGateUpFp8V1" in kernel and n256
    return "pro6000d_scheduler_v1::" not in kernel and "PairedGateUpFp8" in kernel and n256


def parse_csv(path: Path, selected: list[str], expected: dict) -> list[dict]:
    lines = path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(('"ID",', 'ID,'))), None)
    require(start is not None, "NCU CSV ID header missing")
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    require(len(reader.fieldnames) == len(set(reader.fieldnames)), "duplicate CSV headers")
    rows = list(reader)
    require(rows and rows[0].get("ID") == "", "wide CSV explicit units row missing")
    units, raw = rows[0], rows[1:]
    require(len(raw) == expected["expected_kernel_count"], "unexpected number of captured kernels")
    require([r.get("ID") for r in raw] == [str(i) for i in range(len(raw))], "missing/duplicate/out-of-order kernel IDs")
    require(CORE <= set(selected), "essential requested counter missing from metrics manifest")
    require(all(k in units for k in selected), "requested metric missing from CSV header")
    result = []
    for row, role in zip(raw, expected["kernel_roles"]):
        require(None not in row and all(v is not None for v in row.values()), "malformed CSV row width")
        kernel = row.get("Kernel Name", "")
        require(role_matches(role, kernel), f"kernel ID {row['ID']} does not corroborate role {role}")
        vals = {}
        for name in row:
            if name in selected or name.startswith(("launch__", "device__attribute_")):
                if row[name] == "":
                    require(name not in selected, f"blank requested metric {name}")
                    continue
                vals[name] = normalize(row[name], units[name])
        for name in selected:
            require(vals[name]["numeric"] is True, f"unavailable/non-numeric requested metric {name}")
        require(RESOURCE_CORE <= vals.keys() and all(vals[k]["numeric"] for k in RESOURCE_CORE),
                "essential LaunchStats resource missing or non-numeric")
        for name, base in ((READ, "byte"), (WRITE, "byte"), (L2, "byte"), (DURATION, "ns")):
            require(vals[name]["base_unit"] == base, f"unknown required counter unit: {name}")
        def val(name): return vals[name]["base_value"]
        require(val(DURATION) > 0 and all(val(k) >= 0 for k in (READ, WRITE, L2)),
                "invalid duration/byte counters")
        result.append({"id": row["ID"], "role": role, "kernel": kernel,
                       "block": row.get("Block Size"), "grid": row.get("Grid Size"),
                       "duration_ms": val(DURATION) / 1e6,
                       "dram_read_bytes": val(READ), "dram_write_bytes": val(WRITE),
                       "dram_total_bytes": val(READ) + val(WRITE), "l2_request_bytes": val(L2),
                       "tensor_active": vals[TENSOR],
                       "instructions": {k: v for k, v in vals.items() if k.startswith("smsp__inst_executed")},
                       "shared_wavefronts_and_bank_conflicts": {k: v for k, v in vals.items() if
                           k.startswith(("l1tex__data_pipe_lsu_wavefronts_mem_shared_", "l1tex__data_bank_conflicts_"))},
                       "stalls": {k: v for k, v in vals.items() if k.startswith("smsp__warp_issue_stalled_")},
                       "local_sectors": {k: v for k, v in vals.items() if k.startswith("l1tex__t_sectors_pipe_lsu_mem_local_")},
                       "resources": {k: v for k, v in vals.items() if k.startswith(("launch__", "device__attribute_"))},
                       "metrics": vals})
    return result


def analyze_job(root: Path, label: str) -> dict:
    job = root / label
    out = {"job": label, "usable_counters": False, "errors": [], "evidence_sha256": {}}
    try:
        require(re.fullmatch(r"[a-z0-9][a-z0-9-]*", label) is not None, "unsafe job label")
        status = json.loads((job / "status.json").read_text())
        out["status"] = {k: status.get(k) for k in (
            "mode", "exit_code", "export_exit_code", "timeout", "export_timeout", "elapsed_s",
            "owned_container_present_after", "export_container_present_after", "service_after_passed", "embedding_monitor")}
        if status.get("exit_code") not in (None, 0):
            out["capture_failure_root_cause"] = "not_known"
        for name in ("status.json", "command.json", "stdout.log", "stderr.log", "metrics.json"):
            out["evidence_sha256"][name] = sha(job / name)
        argv = json.loads((job / "command.json").read_text())
        expected = contract(argv)
        out["contract"] = expected
        require(status.get("mode") == "ncu" and status.get("exit_code") == 0 and
                status.get("export_exit_code") == 0 and not status.get("timeout") and
                not status.get("export_timeout"), "capture/export failed or incomplete")
        require(status.get("owned_container_present_after") is False and
                status.get("export_container_present_after") is False, "owned capture/export container remains")
        require(status.get("service_after_passed") is True and
                status.get("embedding_monitor", {}).get("passed") is True, "protected-service checks failed")
        evidence = validate_probe(job, status, argv, expected)
        reports = list(job.glob("capture.ncu-rep*"))
        require(len(reports) == 1, "missing/ambiguous captured NCU report")
        out["evidence_sha256"][reports[0].name] = sha(reports[0])
        selected = json.loads((job / "metrics.json").read_text()).get("metrics")
        require(isinstance(selected, list) and len(selected) == len(set(selected)), "invalid metrics manifest")
        require(flag(status["command"], "--metrics") == ",".join(selected), "metrics selection differs from capture command")
        raw = job / "raw.csv"
        out["evidence_sha256"]["raw.csv"] = sha(raw)
        kernels = parse_csv(raw, selected, expected)
        totals = {k: sum(x[k] for x in kernels) for k in (
            "duration_ms", "dram_read_bytes", "dram_write_bytes", "dram_total_bytes", "l2_request_bytes")}
        out.update({"usable_counters": True, "validation": evidence, "kernels": kernels,
                    "sample_totals": totals, "total_scope": expected["scope"]})
    except (ValueError, KeyError, TypeError, OSError, StopIteration) as exc:
        out["errors"].append(f"{type(exc).__name__}: {exc}")
    return out


def comparisons(jobs: list[dict]) -> list[dict]:
    baseline = [x for x in jobs if x["usable_counters"] and
                x["contract"]["kernel_roles"] == ["old_official_gemm", "v2_quant"]]
    result = []
    for b in baseline:
        for c in jobs:
            if c is b or not c["usable_counters"] or c["contract"]["scope"] != "complete_pipeline":
                continue
            reason = []
            if b["contract"]["M"] != c["contract"]["M"]: reason.append("M differs")
            for key in ("input_sha256", "cache_control", "clock_control"):
                if b["validation"][key] != c["validation"][key]: reason.append(key + " differs")
            row = {"baseline": b["job"], "candidate": c["job"], "comparable_capture_contract": not reason,
                   "incomparable_reasons": reason, "scope": "profiler capture ratio; not API speedup or buffer attribution"}
            if not reason:
                row["candidate_over_baseline"] = {k: c["sample_totals"][k] / v if v else None
                                                   for k, v in b["sample_totals"].items()}
            result.append(row)
    return result


def main() -> int:
    root_default = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=root_default)
    p.add_argument("--jobs", type=Path, help="JSON list of job labels; defaults to root/ncu-jobs.json")
    p.add_argument("--output", type=Path, help="write JSON; omitted prints JSON only")
    p.add_argument("--allow-incomplete", action="store_true", help="inventory partial/failed jobs without nonzero exit; never marks them usable")
    a = p.parse_args()
    labels = json.loads((a.jobs or a.root / "ncu-jobs.json").read_text())
    require(isinstance(labels, list) and bool(labels) and all(isinstance(x, str) for x in labels) and
            len(labels) == len(set(labels)), "jobs must be a nonempty unique label list")
    jobs = [analyze_job(a.root, x) for x in labels]
    result = {"analyzer_sha256": sha(Path(__file__)), "jobs_manifest_sha256": sha(a.jobs or a.root / "ncu-jobs.json"),
              "all_jobs_usable": all(x["usable_counters"] for x in jobs),
              "usable_count": sum(x["usable_counters"] for x in jobs), "job_count": len(jobs),
              "jobs": jobs, "complete_pipeline_capture_comparisons": comparisons(jobs),
              "limits": ["NCU kernel replay durations are profiler measurements, not unprofiled API/model performance.",
                         "DRAM/L2 totals are kernel counters and do not attribute bytes to B, scales, output or any allocation.",
                         "Stall percentages describe active-warps; they are not additive fractions of wall/API time.",
                         "Cache-control none and uncontrolled clocks do not establish identical cache/thermal state.",
                         "Normalized byte values retain the precision of the exported CSV cells; they are not unrounded hardware-counter integers.",
                         "One-window samples provide no independent variability or causal cache proof."]}
    rendered = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if a.output:
        a.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if result["all_jobs_usable"] or a.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
