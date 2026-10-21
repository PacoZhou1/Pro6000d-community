#!/usr/bin/env python3
"""Pure-CPU SM120 tile resource arithmetic from saved, fixed NCU CSVs.

This reads existing evidence only. No CUDA, SSH, model or service is loaded.
Run from anywhere: python3 resource_model.py [--output resource-model.json]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = HERE.parent / "gemm-epilogue"
TARGETED = HERE.parent / "gemm-targeted"
K, N, NK = 5120, 34816, 128
SHAPES = (2741, 16160, 32768)
KERNELS = {
    "official_m128n128_coop": {"tile_m": 128, "tile_n": 128, "mma_threads": 256,
                                "csv": E / "ncu-coop-v6-m2741/raw.csv", "row": 0},
    "v3_m64n256_pingpong": {"tile_m": 64, "tile_n": 256, "mma_threads": 128,
                              "csv": E / "ncu-v3-m2741/raw.csv", "row": 2},
    "v4_m32n256_pingpong": {"tile_m": 32, "tile_n": 256, "mma_threads": 128,
                              "csv": E / "ncu-m32-v4-m2741/raw.csv", "row": 2},
    "v6_m64n256_coop": {"tile_m": 64, "tile_n": 256, "mma_threads": 256,
                          "csv": E / "ncu-coop-v6-m2741/raw.csv", "row": 2},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_number(row: dict, key: str) -> int | float:
    value = row.get(key, "")
    if not value:
        raise ValueError(f"missing NCU field {key}")
    return float(value) if "." in value or "e" in value.lower() else int(value)


def ncu_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return [r for r in csv.DictReader(stream) if r.get("Kernel Name")]


def selected_wide_ncu(path: Path) -> dict:
    """Preserve NCU's displayed units; never treat a scaled GB as bytes."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    units = rows[0]
    records = [r for r in rows if r.get("Kernel Name")]
    if len(records) != 4:
        raise ValueError(f"expected GEMM, quant, GEMM, pack in {path}: {len(records)}")
    columns = (
        "gpu__time_duration.sum", "dram__bytes_op_read.sum",
        "dram__bytes_op_write.sum", "lts__t_bytes.sum",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__inst_executed.sum", "smsp__inst_executed_pipe_alu.sum",
        "smsp__inst_executed_pipe_fma.sum",
        "smsp__inst_executed_pipe_lsu.sum",
        "smsp__inst_executed_pipe_tensor.sum",
        "smsp__inst_executed_pipe_xu.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_eligible.avg.per_cycle_active",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
        "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
        "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    )
    selected = {}
    for label, row in zip(("official_gemm", "official_quant", "v6_gemm", "v6_pack"), records):
        selected[label] = {
            k: {"value": as_number(row, k), "ncu_display_unit": units[k]}
            for k in columns
        }
    return {"source_csv_sha256": sha256(path), "kernels": selected}


def main(output: Path) -> None:
    v6_rows = ncu_rows(E / "ncu-coop-v6-m2741/raw.csv")
    hardware_row = v6_rows[0]
    hkeys = {
        "sm_count": "device__attribute_multiprocessor_count",
        "registers_per_sm": "device__attribute_max_registers_per_multiprocessor",
        "shared_bytes_per_sm": "device__attribute_max_shared_memory_per_multiprocessor",
        "shared_bytes_per_block_optin": "device__attribute_max_shared_memory_per_block_optin",
        "threads_per_sm": "device__attribute_max_threads_per_multiprocessor",
        "warps_per_sm": "device__attribute_max_warps_per_multiprocessor",
        "max_blocks_per_sm": "device__attribute_max_blocks_per_multiprocessor",
        "l2_bytes": "device__attribute_l2_cache_size",
        "cluster_launch_supported": "device__attribute_cluster_launch",
    }
    hardware = {name: int(as_number(hardware_row, column)) for name, column in hkeys.items()}
    values = {}
    for name, config in KERNELS.items():
        path = config["csv"]
        row = ncu_rows(path)[config["row"]]
        tm, tn = config["tile_m"], config["tile_n"]
        block = int(as_number(row, "launch__block_size"))
        regs = int(as_number(row, "launch__registers_per_thread_allocated"))
        shared = int(as_number(row, "launch__shared_mem_per_block"))
        reg_ctas = hardware["registers_per_sm"] // (regs * block)
        smem_ctas = hardware["shared_bytes_per_sm"] // shared
        thread_ctas = hardware["threads_per_sm"] // block
        warp_ctas = hardware["warps_per_sm"] // (block // 32)
        stage_ab = (tm + tn) * NK  # both operands are FP8, one byte/element
        data = {
            "tile_mnk": [tm, tn, NK],
            "mma_threads_per_cta": config["mma_threads"],
            "mma_warps_per_cta": config["mma_threads"] // 32,
            "threads_per_cta_total": block,
            "output_elements_per_cta": tm * tn,
            "fp32_accumulator_elements_per_mma_thread": tm * tn // config["mma_threads"],
            "fp32_accumulator_bytes_per_mma_warp": tm * tn * 4 // (config["mma_threads"] // 32),
            "registers_allocated_per_thread": regs,
            "registers_nominal_per_cta": regs * block,
            "shared_bytes_per_cta_measured": shared,
            "shared_bytes_dynamic_measured": int(as_number(row, "launch__shared_mem_per_block_dynamic")),
            "shared_bytes_driver_measured": int(as_number(row, "launch__shared_mem_per_block_driver")),
            "shared_bytes_static_measured": int(as_number(row, "launch__shared_mem_per_block_static")),
            "fp8_a_plus_b_bytes_per_k128_stage": stage_ab,
            "two_ab_stages_bytes": 2 * stage_ab,
            "shared_bytes_remaining_after_two_raw_ab_stages": shared - 2 * stage_ab,
            "resident_cta_bounds": {"registers": reg_ctas, "shared": smem_ctas,
                                    "threads": thread_ctas, "warps": warp_ctas,
                                    "architectural_blocks": hardware["max_blocks_per_sm"]},
            "resident_cta_bound_min": min(reg_ctas, smem_ctas, thread_ctas,
                                          warp_ctas, hardware["max_blocks_per_sm"]),
            "ncu_m2741": {"csv_sha256": sha256(path),
                          "grid_size": int(as_number(row, "launch__grid_size")),
                          "waves_per_sm": float(as_number(row, "launch__waves_per_multiprocessor")),
                          "local_load_sectors": int(as_number(row, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum")),
                          "local_store_sectors": int(as_number(row, "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum")),
                          "l2_request_bytes": int(as_number(row, "lts__t_bytes.sum")),
                          "dram_read_bytes": int(as_number(row, "dram__bytes_op_read.sum")),
                          "dram_write_bytes": int(as_number(row, "dram__bytes_op_write.sum")),
                          "kernel_duration_ns": int(as_number(row, "gpu__time_duration.sum"))},
            "shapes": {},
        }
        for m in SHAPES:
            cta_m, cta_n = math.ceil(m / tm), N // tn
            ctas = cta_m * cta_n
            # Historical Sm90 arithmetic retained only to identify the old error.
            # Actual SM120 CLC uses unrounded tiles and direct divisor 8.
            swizzle = 8 if min(cta_m, cta_n) >= 6 else (4 if min(cta_m, cta_n) >= 3 else (2 if min(cta_m, cta_n) >= 2 else 1))
            raster_m = math.ceil(cta_m / swizzle) * swizzle
            raster_n = math.ceil(cta_n / swizzle) * swizzle
            raster = "AlongM" if raster_n > raster_m else "AlongN"
            clc_raster = "AlongM" if cta_n > cta_m else "AlongN"
            grid = [cta_m, cta_n, 1] if clc_raster == "AlongM" else [cta_n, cta_m, 1]
            a_bytes = ctas * tm * K
            b_bytes = ctas * tn * K
            # Scale loads are logical requests. Cache-line traffic and
            # CUTLASS scale-copy implementation can differ from this lower
            # element-count model.
            a_scale_bytes = ctas * tm * (K // NK) * 4
            b_scale_bytes = ctas * (tn // NK) * (K // NK) * 4
            data["shapes"][str(m)] = {
                "cta_m": cta_m, "cta_n": cta_n, "total_ctas": ctas,
                "theoretical_waves_at_one_cta_per_sm": ctas / hardware["sm_count"],
                "logical_fp8_a_bytes_loaded_across_ctas": a_bytes,
                "logical_fp8_b_bytes_loaded_across_ctas": b_bytes,
                "logical_fp8_ab_bytes_loaded_across_ctas": a_bytes + b_bytes,
                "logical_a_scale_bytes_loaded_across_ctas": a_scale_bytes,
                "logical_b_scale_bytes_loaded_across_ctas": b_scale_bytes,
                "a_duplication_across_n_ctas": cta_n,
                "b_duplication_across_m_ctas": cta_m,
                "legacy_sm90_swizzle_size": swizzle,
                "legacy_sm90_rounded_mn_ctas": [raster_m, raster_n],
                "legacy_sm90_raster_heuristic": raster,
                "legacy_sm90_minor_dimension": "N" if raster == "AlongM" else "M",
                "legacy_sm90_valid_for_actual_sm120_launch": False,
                "clc_scheduler": "PersistentTileSchedulerSm100",
                "clc_raster_heuristic": clc_raster,
                "clc_swizzle_divisor": 8,
                "clc_physical_grid": grid,
                "clc_remapped_interior_xy": [grid[0] // 8 * 8, grid[1] // 8 * 8],
                "clc_residual_columns_rows": [grid[0] % 8, grid[1] % 8],
            }
        if data["shapes"]["2741"]["total_ctas"] != data["ncu_m2741"]["grid_size"]:
            raise AssertionError(f"CTA model disagrees with NCU grid for {name}")
        values[name] = data

    baseline_tensor = ncu_rows(TARGETED / "counters/ncu-fp8-m16160/metrics.csv")[0]
    tensor = {key: float(as_number(baseline_tensor, key)) for key in (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed")}
    result = {"scope": "fixed SM120 FP8 gate_up; pure CPU arithmetic on saved NCU records",
              "units": "bytes unless a key says sectors, ns, waves or count",
              "hardware_from_ncu": hardware,
              "unique_fp8_weight_bytes": N * K,
              "unique_fp8_a_bytes_by_m": {str(m): m * K for m in SHAPES},
              "baseline_m16160_throughput_percentages": tensor,
              "extended_ncu": {
                  str(m): selected_wide_ncu(HERE / f"ncu-m{m}/raw.csv")
                  for m in (2741, 16160)
              },
              "m16160_repeat_ncu": selected_wide_ncu(HERE / "ncu-m16160-repeat/raw.csv"),
              "m16160_cache_all_status": {
                  "source_sha256": sha256(HERE / "ncu-m16160-cache-all/status.json"),
                  "exit_code": json.loads((HERE / "ncu-m16160-cache-all/status.json").read_text())["exit_code"],
                  "usable_numeric_or_counters": False,
              },
              "kernels": values,
              "scheduler_source": "../sm120-scheduler/scheduler-source/source-manifest.json",
              "caveats": [
                  "legacy_sm90 fields are superseded historical assumptions, not actual launch geometry",
                  "clc coordinate mapping is not measured hardware execution order; see scheduler-shape-notes.md",
                  "logical CTA operand bytes are not L2 request bytes or DRAM bytes",
                  "NCU reports static shared 0, driver-accounted shared 1024 and separate dynamic shared; launch shared total is used",
                  "FP32 accumulator elements are a lower bound on live register pressure",
                  "NCU waves are grid/SM for these one-CTA-per-SM launches, not observed execution waves",
                  "tensor active percentage is pipe busy-cycle ratio, not achieved peak FLOPS",
              ]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "resource-model.json")
    args = parser.parse_args()
    main(args.output)
