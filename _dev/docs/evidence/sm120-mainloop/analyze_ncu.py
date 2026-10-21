#!/usr/bin/env python3
"""Strict CPU-only normalization of five saved SM120 mainloop NCU captures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / "sm120-scheduler" / "analyze_ncu.py"
HELPER_SHA = "b2b7cce06958bb343024b7dfec01a448a45791a1bb358aadeda16d62a459585e"
STRICT = HERE / "analyze_mainloop.py"
JOBS = {
    "ncu-official-v1": ("official_vllm", ("old_official_gemm", "v2_quant")),
    "ncu-control-fused-v1": ("control42_along_m_s1", ("control_fused_gemm", "pack")),
    "ncu-control-paired-v1": ("paired_control42_along_m_s1", ("control_paired_bf16_gemm",)),
    "ncu-candidate-fused-v1": ("warp24n64_v3_along_m_s1", ("candidate_fused_gemm", "pack")),
    "ncu-candidate-paired-v1": ("paired_warp24n64_v3_along_m_s1", ("candidate_paired_bf16_gemm",)),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec and spec.loader, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kernel_matches(role: str, name: str) -> bool:
    compact = re.sub(r"(?:\b[A-Za-z_]\w*::)+", "", name).replace(" ", "")
    n128 = "tuple<C<128>,C<128>,C<128>>" in compact
    n256 = "tuple<C<64>,C<256>,C<128>>" in compact
    if role == "old_official_gemm":
        return "bfloat16_t" in name and n128 and "pro6000d_mainloop_" not in name
    if role == "v2_quant":
        return "silu_quant" in name
    if role == "pack":
        return "mainloop_pack_gateup_fp8" in name
    if role.startswith("control_"):
        # NCU often elides the private namespace for BF16 template aliases;
        # the frozen command, NVTX witness and one-kernel contract establish
        # its arm membership. The shape/dtype only corroborate that evidence.
        return (n256 and "bfloat16_t" in name if "paired_bf16" in role else
                "pro6000d_mainloop_control42::" in name and n256 and "MainloopPairedGateUpFp8_control42" in name)
    if role.startswith("candidate_"):
        return (n256 and "bfloat16_t" in name if "paired_bf16" in role else
                "pro6000d_mainloop_warp24n64_v3::" in name and n256 and "MainloopPairedGateUpFp8_warp24n64_v3" in name)
    return False


def parse_csv(path: Path, metrics: list[str], roles: tuple[str, ...], arm: str, helper) -> list[dict]:
    lines = path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(('"ID",', 'ID,'))), None)
    require(start is not None, "NCU CSV ID header missing")
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    require(len(reader.fieldnames) == len(set(reader.fieldnames)), "duplicate CSV headers")
    rows = list(reader)
    require(rows and rows[0].get("ID") == "", "wide CSV units row missing")
    units, raw = rows[0], rows[1:]
    require(len(raw) == len(roles), "captured kernel count differs from exact contract")
    require([r.get("ID") for r in raw] == [str(i) for i in range(len(roles))], "missing/out-of-order kernel IDs")
    require(helper.CORE <= set(metrics) and len(metrics) == len(set(metrics)), "required/unique metrics missing")
    require(all(k in units for k in metrics), "requested metric absent from CSV")
    result = []
    for row, role in zip(raw, roles):
        require(None not in row and all(v is not None for v in row.values()), "malformed CSV width")
        require(any("SM120_MAINLOOP:" + arm in (value or "") for key, value in row.items()
                    if "Range" in key or "Msg" in key), "CSV kernel is outside declared NVTX range")
        name = row.get("Kernel Name", "")
        require(kernel_matches(role, name), f"kernel {row.get('ID')} does not corroborate {role}: {name}")
        vals = {}
        for metric in row:
            if metric in metrics or metric.startswith(("launch__", "device__attribute_")):
                if row[metric] == "":
                    require(metric not in metrics, f"blank requested metric {metric}")
                    continue
                vals[metric] = helper.normalize(row[metric], units[metric])
        for metric in metrics:
            require(vals[metric]["numeric"] is True, f"unavailable/non-numeric metric {metric}")
        require(helper.RESOURCE_CORE <= vals.keys() and
                all(vals[k]["numeric"] for k in helper.RESOURCE_CORE), "essential launch resource missing")
        for metric, unit in ((helper.READ, "byte"), (helper.WRITE, "byte"),
                             (helper.L2, "byte"), (helper.DURATION, "ns")):
            require(vals[metric]["base_unit"] == unit, f"unknown unit for {metric}")
        def val(k): return vals[k]["base_value"]
        require(val(helper.DURATION) > 0 and all(val(k) >= 0 for k in (helper.READ, helper.WRITE, helper.L2)),
                "invalid duration/traffic")
        result.append({
            "id": row["ID"], "role": role, "kernel": name,
            "block": row.get("Block Size"), "grid": row.get("Grid Size"),
            "duration_ms": val(helper.DURATION) / 1e6,
            "dram_read_bytes": val(helper.READ), "dram_write_bytes": val(helper.WRITE),
            "dram_total_bytes": val(helper.READ) + val(helper.WRITE),
            "l2_request_bytes": val(helper.L2), "tensor_active": vals[helper.TENSOR],
            "shared_wavefronts_and_bank_conflicts": {k: v for k, v in vals.items() if
                k.startswith(("l1tex__data_pipe_lsu_wavefronts_mem_shared_", "l1tex__data_bank_conflicts_"))},
            "instructions": {k: v for k, v in vals.items() if k.startswith("smsp__inst_executed")},
            "stalls": {k: v for k, v in vals.items() if k.startswith("smsp__warp_issue_stalled_")},
            "local_sectors": {k: v for k, v in vals.items() if k.startswith("l1tex__t_sectors_pipe_lsu_mem_local_")},
            "resources": {k: v for k, v in vals.items() if k.startswith(("launch__", "device__attribute_"))},
            "metrics": vals,
        })
    return result


def analyze_job(root: Path, label: str, helper, strict) -> dict:
    job = root / label
    arm, roles = JOBS[label]
    item = {"job": label, "arm": arm, "scope": "paired_BF16_GEMM_only" if "paired" in label else "complete_chain",
            "usable": False, "errors": [], "evidence_sha256": {}}
    try:
        for filename in ("status.json", "command.json", "stdout.log", "stderr.log", "metrics.json", "raw.csv"):
            item["evidence_sha256"][filename] = sha(job / filename)
        status = json.loads((job / "status.json").read_text())
        argv = json.loads((job / "command.json").read_text())
        command = status["command"]
        require(command[-len(argv):] == argv, "saved command differs from controller execution")
        require(status.get("mode") == "ncu" and status.get("exit_code") == 0 and
                status.get("export_exit_code") == 0 and not status.get("timeout") and
                not status.get("export_timeout"), "NCU capture/export incomplete")
        require(status.get("owned_container_present_after") is False and
                status.get("export_container_present_after") is False, "owned container not cleaned")
        require(status.get("service_after_passed") is True and
                status.get("embedding_monitor", {}).get("passed") is True, "service guard failed")
        require(helper.flag(command, "--profile-from-start") == "off" and "--nvtx" in command and
                helper.flag(command, "--nvtx-include") == "regex:SM120_MAINLOOP:.*/", "NCU window filter mismatch")
        require(helper.flag(command, "--clock-control") == "none" and
                helper.flag(command, "--cache-control") == "none", "NCU cache/clock scope differs")
        source = status.get("source_sha256", {})
        require(source, "missing frozen source snapshot")
        for name, digest in source.items():
            require(Path(name).name == name and re.fullmatch(r"[0-9a-f]{64}", digest) and
                    sha(job / "source" / name) == digest, f"source SHA mismatch {name}")
        # Strictly reuse this mission's independent numeric/Graph/one-window
        # validator; NCU stdout may contain profiler progress before JSONL.
        lines = [line for line in (job / "stdout.log").read_text().splitlines() if line.lstrip().startswith("{")]
        require(lines, "no probe JSON events")
        extracted = "\n".join(lines) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=True) as tmp:
            tmp.write(extracted)
            tmp.flush()
            gate = strict.analyze(Path(tmp.name))
        meta, window = gate["metadata"], gate["profiled_once"]
        require(meta.get("mode") == "ncu" and window.get("arm") == arm and
                meta.get("ncu_selected_arm") == arm and window.get("M") == 16160 and
                window.get("nvtx") == "SM120_MAINLOOP:" + arm and window.get("call_count") == 1,
                "one M16160 selected-arm witness mismatch")
        require(meta["dependencies_sha256"]["probe"] == source.get("mainloop_probe.py"),
                "executed probe source differs from frozen snapshot")
        selected = json.loads((job / "metrics.json").read_text()).get("metrics")
        require(isinstance(selected, list) and helper.flag(command, "--metrics") == ",".join(selected),
                "NCU metrics command/manifest mismatch")
        kernels = parse_csv(job / "raw.csv", selected, roles, arm, helper)
        reports = list(job.glob("capture.ncu-rep*"))
        require(len(reports) == 1, "missing/ambiguous NCU report")
        item["evidence_sha256"][reports[0].name] = sha(reports[0])
        require(window["input_sha256"] == meta.get("profiled_input_sha256", window["input_sha256"]),
                "profiled input mismatch")
        library_label = "warp24n64_v3" if "candidate" in label else "control42"
        if label != "ncu-official-v1":
            manifest = root / ("build-" + ("warp24n64-v3" if library_label == "warp24n64_v3" else "control42-v2")) / "build-manifest.json"
            build = json.loads(manifest.read_text())
            require(build.get("status") == "compiled" and build.get("compile_exit_code") == 0 and
                    build.get("library_sha256") == meta["libraries_sha256"][library_label],
                    "sampled library differs from compiled manifest")
            item["evidence_sha256"]["build-manifest.json"] = sha(manifest)
        totals = {k: sum(x[k] for x in kernels) for k in
                  ("duration_ms", "dram_read_bytes", "dram_write_bytes", "dram_total_bytes", "l2_request_bytes")}
        item.update({"usable": True, "status": {k: status.get(k) for k in
                     ("exit_code", "export_exit_code", "timeout", "export_timeout", "embedding_monitor")},
                     "validated_gate": {"numeric_cases": gate["numeric_cases"], "coverage_pass": gate["coverage_pass"],
                         "profiled_once": window, "extracted_jsonl_sha256": hashlib.sha256(extracted.encode()).hexdigest()},
                     "input_sha256": window["input_sha256"], "weight_sha256": meta["weight_sha256"],
                     "library_sha256": meta["libraries_sha256"], "kernel_count": len(kernels),
                     "kernels": kernels, "sample_totals": totals,
                     "cache_control": helper.flag(command, "--cache-control"),
                     "clock_control": helper.flag(command, "--clock-control")})
    except (ValueError, KeyError, TypeError, OSError, AssertionError, json.JSONDecodeError) as exc:
        item["errors"].append(f"{type(exc).__name__}: {exc}")
    return item


def ratio(candidate, baseline):
    return candidate / baseline if baseline else None


def comparisons(jobs: list[dict]) -> dict:
    by = {j["job"]: j for j in jobs if j["usable"]}
    if len(by) != len(JOBS):
        return {"valid": False, "reason": "not all five capture contracts validated"}
    common = {(j["input_sha256"], json.dumps(j["weight_sha256"], sort_keys=True),
               j["cache_control"], j["clock_control"]) for j in by.values()}
    require(len(common) == 1, "five NCU jobs do not share input, weights, cache/clock settings")
    result = {"valid": True, "same_input_weights_and_capture_settings": True,
              "scope": "one NCU replay per arm; profiler counters, not API/model latency"}
    for title, old, new in (("fused_candidate_over_control", "ncu-control-fused-v1", "ncu-candidate-fused-v1"),
                            ("paired_candidate_over_control", "ncu-control-paired-v1", "ncu-candidate-paired-v1")):
        b, c = by[old], by[new]
        result[title] = {k: ratio(c["sample_totals"][k], b["sample_totals"][k]) for k in b["sample_totals"]}
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=HERE)
    p.add_argument("--output", type=Path, default=HERE / "ncu-analysis.json")
    a = p.parse_args()
    require(sha(HELPER) == HELPER_SHA, "frozen scheduler NCU helper SHA mismatch")
    helper = load_module(HELPER, "frozen_scheduler_ncu_helper")
    sys.path.insert(0, str(HERE))
    strict = load_module(STRICT, "strict_mainloop_gate")
    jobs = []
    for label in JOBS:
        # Keep validation in one place and retain failed jobs in inventory.
        result = analyze_job(a.root, label, helper, strict)
        result["strict_analyzer_sha256"] = sha(STRICT)
        jobs.append(result)
    comparison = comparisons(jobs)
    report = {"analyzer_sha256": sha(Path(__file__)), "frozen_helper_sha256": HELPER_SHA,
              "strict_analyzer_sha256": sha(STRICT), "all_jobs_usable": all(j["usable"] for j in jobs),
              "job_count": len(jobs), "usable_count": sum(j["usable"] for j in jobs),
              "jobs": jobs, "capture_comparisons": comparison,
              "limits": ["NCU replays and profiler durations are not unprofiled API/model timings.",
                         "DRAM/L2 totals do not identify B-weight bytes or any allocation.",
                         "Stall percentages are not additive fractions of wall time.",
                         "Cache-control none and clock-control none do not ensure identical cache or clock state.",
                         "Paired BF16 is a GEMM-only diagnostic; it is not a full pipeline.",
                         "Single NCU capture per arm gives no variance estimate."]}
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return 0 if report["all_jobs_usable"] and comparison["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
