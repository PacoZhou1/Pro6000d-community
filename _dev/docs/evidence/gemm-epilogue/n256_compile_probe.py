#!/usr/bin/env python3
"""Compile-only feasibility probe for a 128x256x128 SM120 block-FP8 GEMM.

This is NOT a fused epilogue and must not be used as an inference candidate.
It instantiates the fixed vLLM 0.31 CUTLASS mainloop with N=256 and an
ordinary LinearCombination epilogue to check whether one CTA can own a
packed gate/up 128-column pair. GPU execution, numerical checks and a custom
epilogue remain separate work.

Usage (inside the fixed image, no GPU allocation required)::

  python n256_compile_probe.py \
    --source /path/to/scaled_mm_blockwise_sm120_fp8_dispatch.cuh \
    --cutlass-include /path/to/cutlass/include \
    --build-dir /path/to/owned/build-n256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemm-targeted"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import targeted_kernel  # noqa: E402
import build_candidate as candidate_builder  # noqa: E402


DRIVER = r'''#include "standalone_dispatch.cuh"
#include <cstring>
#include <exception>

namespace { thread_local char error_message[512] = {0}; }
extern "C" const char* n256_last_error() { return error_message; }

// Packed B is [N,K] contiguous, with adjacent gate/up 128-row groups.
// This stage writes [M,N] in the selected D dtype to test CTA/mainloop only.
extern "C" int n256_block_fp8(
    void* output, const void* a, const void* packed_b, const void* as,
    const void* packed_bs, int m, int n, int k, int swizzle,
    void* workspace, size_t workspace_bytes, unsigned long long stream) {
  error_message[0] = 0;
  try {
    if (!output || !a || !packed_b || !as || !packed_bs || !workspace ||
        m <= 256 || n != 34816 || k != 5120 || (swizzle != 1 && swizzle != 8))
      throw std::runtime_error("unsupported argument");
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("cudaGetDevice failed");
    vllm::c3x::workspace = workspace;
    vllm::c3x::workspace_bytes = workspace_bytes;
    vllm::c3x::stream = reinterpret_cast<cudaStream_t>(stream);
    vllm::TensorView o{output, m, n, device};
    vllm::TensorView av{const_cast<void*>(a), m, k, device};
    vllm::TensorView bv{const_cast<void*>(packed_b), k, n, device};
    vllm::TensorView asv{const_cast<void*>(as), m, k / 128, device};
    vllm::TensorView bsv{const_cast<void*>(packed_bs), k / 128, n / 128, device};
    using Gemm = typename vllm::sm120_blockwise_fp8_config_paired_n256<cutlass::bfloat16_t>::Gemm;
    vllm::cutlass_gemm_caller_blockwise<Gemm>(o, av, bv, asv, bsv, swizzle);
    return 0;
  } catch (const std::exception& ex) {
    std::strncpy(error_message, ex.what(), sizeof(error_message) - 1);
    error_message[sizeof(error_message) - 1] = 0;
    return 1;
  }
}
'''


def make_dispatch(original: str, tile_m: int, epi_m: int | None,
                  cooperative_experimental: bool = False) -> str:
    if tile_m not in (64, 128):
        raise ValueError("tile_m must be 64 or 128")
    if cooperative_experimental and tile_m != 64:
        raise ValueError("experimental Cooperative probe requires M64")
    adapted = targeted_kernel.adapted_dispatch(original)
    if epi_m is not None:
        # Add an optional epilogue tile argument to the copied fixed template;
        # all original configurations retain EpilogueTileAuto.
        old = "  bool swap_ab_ = false>\nstruct cutlass_3x_gemm_fp8_blockwise {"
        new = ("  bool swap_ab_ = false,\n"
               "  class EpilogueTile_ = cutlass::epilogue::collective::EpilogueTileAuto>\n"
               "struct cutlass_3x_gemm_fp8_blockwise {")
        assert adapted.count(old) == 1
        adapted = adapted.replace(old, new, 1)
        old = "      cutlass::epilogue::collective::EpilogueTileAuto,\n"
        assert adapted.count(old) == 1
        adapted = adapted.replace(old, "      EpilogueTile_,\n", 1)
    anchor = "template <typename OutType>\nstruct sm120_blockwise_fp8_config_pingpong {"
    assert adapted.count(anchor) == 1
    paired = """template <typename OutType>
struct sm120_blockwise_fp8_config_paired_n256 {
  using KernelSchedule = _KERNEL_SCHEDULE;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_TILE_M, _256, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule_EXT>;
};

"""
    extension = (f", false, Shape<_{epi_m}, _256>" if epi_m is not None else "")
    paired = paired.replace("_TILE_M", f"_{tile_m}")
    schedule = ("cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120"
                if cooperative_experimental else
                "cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120"
                if tile_m == 64 else "cutlass::gemm::collective::KernelScheduleAuto")
    paired = paired.replace("_KERNEL_SCHEDULE", schedule)
    paired = paired.replace("KernelSchedule_EXT", "KernelSchedule" + extension)
    return adapted.replace(anchor, paired + anchor, 1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--cutlass-include", required=True, type=Path)
    p.add_argument("--build-dir", required=True, type=Path)
    p.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    p.add_argument("--tile-m", type=int, choices=(64, 128), default=128,
                   help="CTA M tile; 128 is the original feasibility test")
    p.add_argument("--epi-m", type=int, choices=(16, 32, 64),
                   help="explicit epilogue tile M; N is fixed 256")
    p.add_argument("--output-fp8", action="store_true",
                   help="instantiate an FP8 D store to model candidate shared memory")
    p.add_argument("--cooperative-experimental", action="store_true",
                   help="M64N256 ordinary D-store diagnostic with pinned M>=128 shadow assert")
    a = p.parse_args()
    if a.cooperative_experimental and a.tile_m != 64:
        p.error("Cooperative diagnostic requires --tile-m 64")
    original = targeted_kernel.checked_source(a.source)
    if not (a.cutlass_include / "cutlass" / "cutlass.h").exists():
        p.error("--cutlass-include must point to CUTLASS include root")
    util = a.cutlass_include.parent / "tools" / "util" / "include"
    if not (util / "cutlass" / "util" / "packed_stride.hpp").exists():
        p.error("CUTLASS tools/util/include missing")
    a.build_dir.mkdir(parents=True, exist_ok=True)
    coop_patch_sha256 = None
    mainloop_patch_sha256 = None
    if a.cooperative_experimental:
        shadow = a.build_dir / "include" / candidate_builder.COOPERATIVE_REL
        shadow.parent.mkdir(parents=True, exist_ok=True)
        shadow.write_text(candidate_builder.cooperative_m64_compile_probe_patch(a.cutlass_include))
        coop_patch_sha256 = hashlib.sha256(shadow.read_bytes()).hexdigest()
        mainloop_shadow = a.build_dir / "include" / candidate_builder.MAINLOOP_REL
        mainloop_shadow.parent.mkdir(parents=True, exist_ok=True)
        mainloop_shadow.write_text(candidate_builder.mainloop_scale_order_patch(a.cutlass_include))
        mainloop_patch_sha256 = hashlib.sha256(mainloop_shadow.read_bytes()).hexdigest()
    (a.build_dir / "standalone_adapter.cuh").write_text(targeted_kernel.ADAPTER)
    (a.build_dir / "standalone_dispatch.cuh").write_text(
        make_dispatch(original, a.tile_m, a.epi_m, a.cooperative_experimental))
    driver = DRIVER.replace("<cutlass::bfloat16_t>::Gemm",
                            "<cutlass::float_e4m3_t>::Gemm") if a.output_fp8 else DRIVER
    (a.build_dir / "n256_driver.cu").write_text(driver)
    epi_label = "auto" if a.epi_m is None else f"m{a.epi_m}n256"
    d_label = "fp8" if a.output_fp8 else "bf16"
    output = a.build_dir / f"libn256_m{a.tile_m}_epi{epi_label}_{d_label}_probe.so"
    cmd = [str(a.nvcc), "-std=c++17", "-O3", "--expt-relaxed-constexpr",
           "--expt-extended-lambda", "-arch=sm_120a", "-Xcompiler=-fPIC",
           "-shared", "-I", str(a.build_dir / "include"), "-I", str(a.build_dir), "-I", str(a.cutlass_include),
           "-I", str(util), str(a.build_dir / "n256_driver.cu"),
           "-o", str(output), "-lcudart"]
    subprocess.run(cmd, check=True)
    print(json.dumps({"probe": "N256 mainloop/default LinearCombination epilogue only; not fused",
                      "library": str(output),
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "source_sha256": targeted_kernel.PINNED_SHA256,
                      "tile_m": a.tile_m,
                      "epi_m": a.epi_m,
                      "schedule": "Cooperative-experimental" if a.cooperative_experimental else "original",
                      "cooperative_shadow_sha256": coop_patch_sha256,
                      "mainloop_scale_order_shadow_sha256": mainloop_patch_sha256,
                      "output_dtype": d_label,
                      "cutlass_version": (a.cutlass_include / "cutlass" / "version.h").read_text().splitlines()[0]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
