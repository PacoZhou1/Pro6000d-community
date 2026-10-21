#!/usr/bin/env python3
"""Build a narrow no-GPU SM120 down-projection tile experiment.

Variant 0 is the exact fixed vLLM dispatch. Variant 1 changes only M<=16,
K=17408,N=5120 to swap-AB 64x32x128 with the original 128x1x128 scale
granularity. Compilation establishes template support, not runtime support
or correctness. Use the separate numerical probe before timing or service use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import targeted_kernel as original


CONFIG_ANCHOR = "template <typename Gemm>\nvoid cutlass_gemm_caller_blockwise"
CONFIG = """template <typename OutType>
struct sm120_blockwise_fp8_config_swapab_down64 {
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  // Actual quantization block remains 128x1x128 after swap-AB.
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true>;
};

"""
DRIVER_ANCHOR = "variant == 1 && m > 256 && k == 5120 && n == 34816"
DRIVER_CONFIG = "sm120_blockwise_fp8_config_pingpong<Out>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cutlass-include", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    args = parser.parse_args()
    source = original.checked_source(args.source)
    if not (args.cutlass_include / "cutlass" / "cutlass.h").is_file():
        parser.error("CUTLASS include path is invalid")
    util_include = args.cutlass_include.parent / "tools" / "util" / "include"
    if not (util_include / "cutlass" / "util" / "packed_stride.hpp").is_file():
        parser.error("matching CUTLASS util include path is unavailable")
    adapted = original.adapted_dispatch(source)
    if adapted.count(CONFIG_ANCHOR) != 1:
        raise ValueError("dispatch config anchor is not unique")
    adapted = adapted.replace(CONFIG_ANCHOR, CONFIG + CONFIG_ANCHOR, 1)
    driver = original.DRIVER
    if driver.count(DRIVER_ANCHOR) != 1 or driver.count(DRIVER_CONFIG) != 1:
        raise ValueError("driver routing anchor is not unique")
    driver = driver.replace(DRIVER_ANCHOR,
                            "variant == 1 && m <= 16 && k == 17408 && n == 5120", 1)
    driver = driver.replace(DRIVER_CONFIG,
                            "sm120_blockwise_fp8_config_swapab_down64<Out>", 1)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    (args.build_dir / "standalone_adapter.cuh").write_text(original.ADAPTER)
    (args.build_dir / "standalone_dispatch.cuh").write_text(adapted)
    (args.build_dir / "targeted_driver.cu").write_text(driver)
    output = args.build_dir / "libdown_fp8.so"
    command = [str(args.nvcc), "-std=c++17", "-O3", "--expt-relaxed-constexpr",
               "--expt-extended-lambda", "-arch=sm_120a", "-Xcompiler=-fPIC",
               "-shared", "-I", str(args.build_dir), "-I", str(args.cutlass_include),
               "-I", str(util_include), str(args.build_dir / "targeted_driver.cu"),
               "-o", str(output), "-lcudart"]
    subprocess.run(command, check=True)
    print(json.dumps({"library": str(output),
                      "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "source_sha256": original.PINNED_SHA256,
                      "cuda_arch": "sm_120a",
                      "variant_0": "fixed vLLM 0.31.0 dispatch",
                      "variant_1": "only M<=16,K17408,N5120 swapAB tile64x32x128",
                      "scale_granularity": "128x1x128 after swapAB; unchanged",
                      "gpu_executed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
