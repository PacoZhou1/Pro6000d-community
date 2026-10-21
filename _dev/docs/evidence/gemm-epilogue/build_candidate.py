#!/usr/bin/env python3
"""Build the isolated SM120 paired gate/up FP8 epilogue experiment.

This script only compiles; it never initializes CUDA or runs a GPU kernel.
It verifies the exact fixed vLLM dispatch and CUTLASS visitor sources, emits
modified headers into --build-dir, then invokes nvcc.  The original headers
and installed packages remain untouched.

The library ABI is version 2 and intentionally returns an error for unsupported
shapes.  It is experimental until full-byte numerical, CUDA Graph, and model
checks pass.  CUTLASS native SF128 scales are packed into the model's
column-major [M,136] layout by a second, lightweight CUDA kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemm-targeted"))
import targeted_kernel  # noqa: E402


VISITOR_REL = Path("cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp")
VISITOR_SHA256 = "1a0aa252dfd0883862dcf94a7afb1af0d2aad7d002cda3041eb28e4861b36266"
MAINLOOP_REL = Path("cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp")
MAINLOOP_SHA256 = "88bfdb527044d520c98fead4f0d1d59c051950642722b5344ad656e679287bd8"
COOPERATIVE_REL = Path("cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp")
COOPERATIVE_SHA256 = "d5eb1277d56571ab2dd7b906f857d0bcfca227e2ec4ffcef4440937e9466b777"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one exact CUTLASS anchor, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def visitor_patch(cutlass_include: Path, cooperative_sf128_two_warps: bool = False) -> str:
    original = (cutlass_include / VISITOR_REL).read_bytes()
    actual = hashlib.sha256(original).hexdigest()
    if actual != VISITOR_SHA256:
        raise ValueError(f"fixed CUTLASS visitor SHA mismatch: {actual}")
    text = original.decode("utf-8")
    boundary = text.index("struct Sm120BlockScaleFactorColStore {")
    row, rest = text[:boundary], text[boundary:]
    row = replace_once(
        row,
        "static_assert(SFVecSize == 16 || SFVecSize == 32 || SFVecSize == 64, \"SF vector size must be either 16, 32 or 64.\");",
        "static_assert(SFVecSize == 16 || SFVecSize == 32 || SFVecSize == 64 || SFVecSize == 128, \"SF vector size must be 16, 32, 64, or 128.\");",
    )
    # In the Pingpong/non-cooperative epilogue, warps 0+2 and 1+3 each own
    # a different row. A 128-column group uses those same two warps with
    # eight fragments per thread, not four warps (which crosses rows). The
    # cooperative variant would use four warps and four fragments per thread.
    sf128_warps = ("2" if cooperative_sf128_two_warps else
                   "(NumCollaboratingWarpGroups == 1 ? 2 : 4)")
    row = replace_once(
        row,
        "constexpr int WarpsPerSF = SFVecSize / 16;",
        "constexpr int WarpsPerSF = SFVecSize == 128 ? "
        f"{sf128_warps} : SFVecSize / 16;",
    )
    row = replace_once(
        row,
        "ElementCompute norm_constant = mNormConst(_0{},_0{},l);",
        "ElementCompute norm_constant = ElementCompute(1.0f);",
    )
    row = replace_once(
        row,
        "ElementCompute pvscale = mul(amax, norm_constant_scaled_down);",
        "ElementCompute pvscale = fmaxf(amax / 448.0f, 1.0f / (448.0f * 512.0f));",
    )
    old = """        ElementCompute acc_scale = mul(norm_constant, qpvscale_rcp);
        acc_scale = cutlass::minimum_with_nan_propagation<ElementCompute>{}(acc_scale, cutlass::platform::numeric_limits<ElementCompute>::max());"""
    row = replace_once(row, old, "        ElementCompute acc_scale = qpvscale;")
    row = replace_once(
        row,
        "visit_results(0, row, acc_frag_start_for_sf + i)[acc_frag_row    ] = mul(acc_frag[acc_frag_row], acc_scale);",
        "visit_results(0, row, acc_frag_start_for_sf + i)[acc_frag_row    ] = fmaxf(-448.0f, fminf(acc_frag[acc_frag_row] / acc_scale, 448.0f));",
    )
    row = replace_once(
        row,
        "visit_results(0, row, acc_frag_start_for_sf + i)[acc_frag_row + 1] = mul(acc_frag[acc_frag_row + 1], acc_scale);",
        "visit_results(0, row, acc_frag_start_for_sf + i)[acc_frag_row + 1] = fmaxf(-448.0f, fminf(acc_frag[acc_frag_row + 1] / acc_scale, 448.0f));",
    )
    # The official 128-group kernel's fmaxf reduction ignores a lone NaN.
    row = replace_once(
        row,
        "cutlass::maximum_absolute_value_reduction<ElementCompute, true> amax_op;",
        "auto amax_op = [] (ElementCompute lhs, ElementCompute rhs) { return fmaxf(lhs, fabsf(rhs)); };",
    )
    return row + rest


def mainloop_scale_order_patch(cutlass_include: Path) -> str:
    """Match the official N128 scale multiply grouping for N256 only.

    The fixed CUTLASS mainloop folds A*B before multiplying the MMA result
    when ScaleNsPerTile==1 (the official N128 tile). Its generic N256 path
    instead evaluates (tmp*A)*B. This shadow header changes only the generic
    both>1 branch to tmp*(A*B); it leaves tile, schedule, and K order intact.
    """
    original = (cutlass_include / MAINLOOP_REL).read_bytes()
    actual = hashlib.sha256(original).hexdigest()
    if actual != MAINLOOP_SHA256:
        raise ValueError(f"fixed CUTLASS mainloop SHA mismatch: {actual}")
    text = original.decode("utf-8")
    old = """      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          accum(i) += tmp_accum(i) * tCrScaleAViewAsC(i) * tCrScaleBViewAsC(i);
          tmp_accum(i) = 0;
        }
      }"""
    new = """      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          // Official N128 path folds A*B before applying it to the MMA result.
          const ElementSF scale_ab = tCrScaleAViewAsC(i) * tCrScaleBViewAsC(i);
          accum(i) += tmp_accum(i) * scale_ab;
          tmp_accum(i) = 0;
        }
      }"""
    return replace_once(text, old, new)


def cooperative_m64_compile_probe_patch(cutlass_include: Path) -> str:
    """Remove one CUTLASS support-domain assert for an isolated compile probe.

    This is not evidence that M64 Cooperative is correct or supported. Every
    generated binary remains experimental until full-bit and Graph tests.
    """
    original = (cutlass_include / COOPERATIVE_REL).read_bytes()
    actual = hashlib.sha256(original).hexdigest()
    if actual != COOPERATIVE_SHA256:
        raise ValueError(f"fixed CUTLASS Cooperative SHA mismatch: {actual}")
    old = """    static_assert(size<0>(TileShape{}) >= 128,
        "Cooperative kernel requires Tile Size to be greater than or equal to 128 along the M-dimension.");"""
    new = """    // Experimental M64N256 compile probe only. The original CUTLASS
    // supported-domain assertion is restored in every non-experimental build.
    static_assert(size<0>(TileShape{}) >= 64,
        "Experimental paired Cooperative requires Tile Size M >= 64.");"""
    return replace_once(original.decode("utf-8"), old, new)


def dispatch_patch(original: str, tile_m: int = 64,
                   cooperative_experimental: bool = False) -> str:
    if tile_m not in (32, 64):
        raise ValueError("paired tile M must be 32 or 64")
    if cooperative_experimental and tile_m != 64:
        raise ValueError("experimental Cooperative probe requires M64")
    adapted = targeted_kernel.adapted_dispatch(original)
    adapted = replace_once(
        adapted, '#include "cutlass_gemm_caller.cuh"',
        '#include "paired_gateup_visitor.cuh"',
    ) if '#include "cutlass_gemm_caller.cuh"' in adapted else adapted
    # adapted_dispatch stripped that include; insert after the last CUTLASS include.
    adapted = replace_once(
        adapted,
        '#include "cutlass/epilogue/collective/collective_builder.hpp"',
        '#include "cutlass/epilogue/collective/collective_builder.hpp"\n#include "paired_gateup_visitor.cuh"',
    )
    adapted = replace_once(
        adapted,
        "  bool swap_ab_ = false>\nstruct cutlass_3x_gemm_fp8_blockwise {",
        "  bool swap_ab_ = false,\n"
        "  class FusionOperation_ = void,\n"
        "  class EpilogueTile_ = cutlass::epilogue::collective::EpilogueTileAuto>\n"
        "struct cutlass_3x_gemm_fp8_blockwise {",
    )
    adapted = replace_once(
        adapted,
        "  static constexpr bool swap_ab = swap_ab_;",
        "  static constexpr bool swap_ab = swap_ab_;\n"
        "  static constexpr bool is_paired = !cute::is_void_v<FusionOperation_>;",
    )
    adapted = replace_once(
        adapted,
        "  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;",
        "  using DefaultOperation = cute::conditional_t<is_paired, FusionOperation_,\n"
        "      cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>>;",
    )
    adapted = replace_once(
        adapted,
        "      cutlass::epilogue::collective::EpilogueTileAuto,\n",
        "      EpilogueTile_,\n",
    )
    anchor = "template <typename OutType>\nstruct sm120_blockwise_fp8_config_pingpong {"
    paired = """template <typename OutType>
struct sm120_blockwise_fp8_config_paired_quant {
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_TILE_M, _256, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, false,
      cutlass::epilogue::fusion::PairedGateUpFp8, Shape<_TILE_M, _256>>;
};

"""
    paired = paired.replace("_TILE_M", f"_{tile_m}")
    if cooperative_experimental:
        paired = paired.replace(
            "cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120",
            "cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120")
    adapted = replace_once(adapted, anchor, paired + anchor)
    adapted = replace_once(
        adapted,
        "int max_swizzle_size) {\n  static constexpr bool swap_ab = Gemm::swap_ab;",
        "int max_swizzle_size, void* native_scales = nullptr) {\n"
        "  static constexpr bool swap_ab = Gemm::swap_ab;",
    )
    adapted = replace_once(
        adapted,
        "  typename GemmKernel::EpilogueArguments epilogue_args{\n      {}, c_ptr, c_stride, c_ptr, c_stride};",
        "  typename GemmKernel::EpilogueArguments epilogue_args{\n"
        "      {}, c_ptr, c_stride, c_ptr, c_stride};\n"
        "  if constexpr (Gemm::is_paired) {\n"
        "    epilogue_args.thread = { {}, {static_cast<float*>(native_scales), nullptr, {}} };\n"
        "  }",
    )
    return adapted


DRIVER = r'''#include "standalone_dispatch.cuh"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <exception>

namespace { thread_local char error_message[512] = {0}; }

// The identity visitor intentionally produces a different numerical result.
// Give it a distinct ABI so the model runtime adapter rejects it.
#ifdef PRO6000D_DIAGNOSTIC_IDENTITY_VISITOR
extern "C" int gateup_quant_abi_version() { return 1002; }
#else
extern "C" int gateup_quant_abi_version() { return 2; }
#endif
extern "C" const char* gateup_quant_last_error() { return error_message; }

extern "C" size_t gateup_quant_temp_scale_bytes(int m) {
  if (m <= 256 || m > 32768) return 0;
  using S = cutlass::detail::Sm1xxBlockScaledOutputConfig<128>;
  auto layout = S::tile_atom_to_shape_SFD(cute::make_shape(m, 34816, 5120, 1));
  return size_t(cute::cosize(layout)) * sizeof(float);
}

// Pure host query: forms the same fixed-shape CUTLASS arguments as launch,
// but c3x::query_only returns before can_implement/op.run or any CUDA API.
// A zero return means unsupported inputs or a query failure.
extern "C" size_t gateup_quant_workspace_bytes(int m, int swizzle) {
  if (m <= 256 || m > 32768 || (swizzle != 1 && swizzle != 8)) return 0;
  constexpr uintptr_t fake = 0x10000;
  vllm::TensorView out{reinterpret_cast<void*>(fake), m, 34816, 0};
  vllm::TensorView a{reinterpret_cast<void*>(fake), m, 5120, 0};
  vllm::TensorView b{reinterpret_cast<void*>(fake), 5120, 34816, 0};
  vllm::TensorView as{reinterpret_cast<void*>(fake), m, 40, 0};
  vllm::TensorView bs{reinterpret_cast<void*>(fake), 40, 272, 0};
  using Gemm = typename vllm::sm120_blockwise_fp8_config_paired_quant<cutlass::float_e4m3_t>::Gemm;
  vllm::c3x::query_only = true;
  vllm::c3x::required_workspace_bytes = 0;
  try {
    vllm::cutlass_gemm_caller_blockwise<Gemm>(
        out, a, b, as, bs, swizzle, reinterpret_cast<void*>(fake));
    const size_t needed = vllm::c3x::required_workspace_bytes;
    vllm::c3x::query_only = false;
    return needed == 0 ? 1 : needed;
  } catch (...) {
    vllm::c3x::query_only = false;
    return 0;
  }
}

__global__ void pack_gateup_fp8(
    cutlass::float_e4m3_t* __restrict__ out,
    float* __restrict__ out_scale,
    const cutlass::float_e4m3_t* __restrict__ paired,
    const float* __restrict__ native_scale, int m) {
  constexpr int H = 17408;
  constexpr int PackedN = 34816;
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= int64_t(m) * H) return;
  const int row = int(idx / H);
  const int col = int(idx % H);
  const int group = col / 128;
  const int within = col % 128;
  out[idx] = paired[int64_t(row) * PackedN + group * 256 + within];
  if (within == 0) {
    using S = cutlass::detail::Sm1xxBlockScaledOutputConfig<128>;
    auto layout = S::tile_atom_to_shape_SFD(cute::make_shape(m, PackedN, 5120, 1));
    const auto native_offset = layout(cute::make_coord(row, group * 256, 0));
    out_scale[int64_t(group) * m + row] = native_scale[native_offset];
  }
}

#ifdef PRO6000D_DIAGNOSTIC_IDENTITY_VISITOR
// Read both native N128 scale groups in the exact paired column order.
// out is a caller-owned FP32 [M,272] tensor with stride (1,M).
__global__ void diagnostic_copy_native_scales(
    float* __restrict__ out, const float* __restrict__ native_scale, int m) {
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= int64_t(m) * 272) return;
  const int row = int(idx % m);
  const int group = int(idx / m);
  using S = cutlass::detail::Sm1xxBlockScaledOutputConfig<128>;
  auto layout = S::tile_atom_to_shape_SFD(cute::make_shape(m, 34816, 5120, 1));
  out[idx] = native_scale[layout(cute::make_coord(row, group * 128, 0))];
}

extern "C" int gateup_quant_diagnostic_copy_scales(
    void* out, const void* native_scale, int m, unsigned long long stream_address) {
  error_message[0] = 0;
  if (!out || !native_scale || m <= 256 || m > 32768) {
    std::strncpy(error_message, "unsupported diagnostic copy arguments", sizeof(error_message) - 1);
    return 1;
  }
  const int64_t total = int64_t(m) * 272;
  diagnostic_copy_native_scales<<<int((total + 255) / 256), 256, 0,
                                  reinterpret_cast<cudaStream_t>(stream_address)>>>(
      static_cast<float*>(out), static_cast<const float*>(native_scale), m);
  auto err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    std::strncpy(error_message, cudaGetErrorString(err), sizeof(error_message) - 1);
    return 1;
  }
  return 0;
}
#endif

extern "C" int gateup_quant_fp8(
    void* out_fp8, void* out_scales, void* temp_fp8, void* temp_scales,
    const void* A, const void* Bpacked, const void* As, const void* Bspacked,
    int M, int swizzle, void* workspace, size_t workspace_bytes,
    unsigned long long stream_address) {
  error_message[0] = 0;
  try {
    if (!out_fp8 || !out_scales || !temp_fp8 || !temp_scales ||
        !A || !Bpacked || !As || !Bspacked || !workspace ||
        M <= 256 || M > 32768 || (swizzle != 1 && swizzle != 8))
      throw std::runtime_error("unsupported pointer, shape, or swizzle");
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("cudaGetDevice failed");
    vllm::c3x::workspace = workspace;
    vllm::c3x::workspace_bytes = workspace_bytes;
    vllm::c3x::stream = reinterpret_cast<cudaStream_t>(stream_address);
    vllm::TensorView out{temp_fp8, M, 34816, device};
    vllm::TensorView a{const_cast<void*>(A), M, 5120, device};
    vllm::TensorView b{const_cast<void*>(Bpacked), 5120, 34816, device};
    vllm::TensorView as{const_cast<void*>(As), M, 40, device};
    vllm::TensorView bs{const_cast<void*>(Bspacked), 40, 272, device};
    using Gemm = typename vllm::sm120_blockwise_fp8_config_paired_quant<cutlass::float_e4m3_t>::Gemm;
    vllm::cutlass_gemm_caller_blockwise<Gemm>(out, a, b, as, bs, swizzle, temp_scales);
    constexpr int H = 17408;
    const int64_t total = int64_t(M) * H;
    const int blocks = int((total + 255) / 256);
    pack_gateup_fp8<<<blocks, 256, 0, vllm::c3x::stream>>>(
        static_cast<cutlass::float_e4m3_t*>(out_fp8),
        static_cast<float*>(out_scales),
        static_cast<const cutlass::float_e4m3_t*>(temp_fp8),
        static_cast<const float*>(temp_scales), M);
    auto err = cudaPeekAtLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return 0;
  } catch (const std::exception& ex) {
    std::strncpy(error_message, ex.what(), sizeof(error_message) - 1);
    error_message[sizeof(error_message) - 1] = 0;
    return 1;
  }
}
'''


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--cutlass-include", required=True, type=Path)
    p.add_argument("--build-dir", required=True, type=Path)
    p.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    p.add_argument("--match-official-scale-order", action="store_true",
                   help="shadow-patch only the N256 A*B grouping (v3 experiment)")
    p.add_argument("--tile-m", type=int, choices=(32, 64), default=64,
                   help="paired CTA/epilogue M extent; M32 is a resource-pressure probe")
    p.add_argument("--cooperative-experimental", action="store_true",
                   help="compile-only first: shadow-relax pinned CUTLASS M>=128 assert for M64N256")
    p.add_argument("--diagnostic-identity-visitor", action="store_true",
                   help="diagnostic only: BF16-round and quantize both paired halves without SwiGLU; ABI=1002")
    p.add_argument("--cooperative-sf128-two-warps", action="store_true",
                   help="isolated v6: use warp^4 pair for each SF128 group in M64 Cooperative")
    a = p.parse_args()
    if a.tile_m == 32 and not a.match_official_scale_order:
        p.error("M32 resource probe requires --match-official-scale-order")
    if a.cooperative_experimental and (a.tile_m != 64 or not a.match_official_scale_order):
        p.error("Cooperative probe requires --tile-m 64 --match-official-scale-order")
    if a.cooperative_sf128_two_warps and not a.cooperative_experimental:
        p.error("Cooperative SF128 two-warp probe requires --cooperative-experimental")
    original = targeted_kernel.checked_source(a.source)
    util = a.cutlass_include.parent / "tools" / "util" / "include"
    if not (util / "cutlass" / "util" / "packed_stride.hpp").is_file():
        p.error("CUTLASS tools/util/include missing")
    a.build_dir.mkdir(parents=True, exist_ok=True)
    generated_include = a.build_dir / "include" / VISITOR_REL
    generated_include.parent.mkdir(parents=True, exist_ok=True)
    generated_include.write_text(visitor_patch(a.cutlass_include, a.cooperative_sf128_two_warps))
    mainloop_patch_sha256 = None
    if a.match_official_scale_order:
        mainloop_include = a.build_dir / "include" / MAINLOOP_REL
        mainloop_include.parent.mkdir(parents=True, exist_ok=True)
        mainloop_include.write_text(mainloop_scale_order_patch(a.cutlass_include))
        mainloop_patch_sha256 = hashlib.sha256(mainloop_include.read_bytes()).hexdigest()
    cooperative_patch_sha256 = None
    if a.cooperative_experimental:
        cooperative_include = a.build_dir / "include" / COOPERATIVE_REL
        cooperative_include.parent.mkdir(parents=True, exist_ok=True)
        cooperative_include.write_text(cooperative_m64_compile_probe_patch(a.cutlass_include))
        cooperative_patch_sha256 = hashlib.sha256(cooperative_include.read_bytes()).hexdigest()
    adapter = targeted_kernel.ADAPTER.replace("inline void* workspace", "inline thread_local void* workspace")
    adapter = adapter.replace("inline size_t workspace_bytes", "inline thread_local size_t workspace_bytes")
    adapter = adapter.replace("inline cudaStream_t stream", "inline thread_local cudaStream_t stream")
    adapter = replace_once(
        adapter,
        "inline thread_local cudaStream_t stream = nullptr;",
        "inline thread_local cudaStream_t stream = nullptr;\n"
        "inline thread_local bool query_only = false;\n"
        "inline thread_local size_t required_workspace_bytes = 0;",
    )
    adapter = replace_once(
        adapter,
        "  auto status = op.can_implement(args);",
        "  required_workspace_bytes = op.get_workspace_size(args);\n"
        "  if (query_only) return;\n"
        "  auto status = op.can_implement(args);",
    )
    (a.build_dir / "standalone_adapter.cuh").write_text(adapter)
    (a.build_dir / "standalone_dispatch.cuh").write_text(
        dispatch_patch(original, a.tile_m, a.cooperative_experimental))
    (a.build_dir / "paired_gateup_visitor.cuh").write_bytes(
        (Path(__file__).parent / "paired_gateup_visitor.cuh").read_bytes())
    (a.build_dir / "gateup_quant_driver.cu").write_text(DRIVER)
    output = a.build_dir / "libgateup_quant_fp8.so"
    cmd = [str(a.nvcc), "-std=c++17", "-O3", "--expt-relaxed-constexpr",
           "--expt-extended-lambda", "-arch=sm_120a", "-Xcompiler=-fPIC",
           "-shared", "-I", str(a.build_dir / "include"), "-I", str(a.build_dir),
           "-I", str(a.cutlass_include), "-I", str(util),
           str(a.build_dir / "gateup_quant_driver.cu"), "-o", str(output), "-lcudart"]
    if a.diagnostic_identity_visitor:
        cmd.insert(1, "-DPRO6000D_DIAGNOSTIC_IDENTITY_VISITOR=1")
    subprocess.run(cmd, check=True)
    print(json.dumps({"abi_version": 1002 if a.diagnostic_identity_visitor else 2,
                      "diagnostic_identity_visitor": a.diagnostic_identity_visitor,
                      "cooperative_sf128_two_warps": a.cooperative_sf128_two_warps,
                      "library": str(output),
                      "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "vllm_dispatch_sha256": targeted_kernel.PINNED_SHA256,
                      "cutlass_visitor_sha256": VISITOR_SHA256,
                      "cutlass_mainloop_sha256": MAINLOOP_SHA256 if a.match_official_scale_order else None,
                      "cutlass_mainloop_patch_sha256": mainloop_patch_sha256,
                      "cutlass_cooperative_sha256": COOPERATIVE_SHA256 if a.cooperative_experimental else None,
                      "cutlass_cooperative_patch_sha256": cooperative_patch_sha256,
                      "scale_order": "A*B before MMA multiply" if a.match_official_scale_order else "(MMA*A)*B",
                      "cuda_arch": "sm_120a", "tile": f"{a.tile_m}x256x128",
                      "epilogue_tile": f"{a.tile_m}x256",
                      "schedule": "Cooperative-experimental" if a.cooperative_experimental else "Pingpong",
                      "output": "FP8 paired temp + pack, native scale temp + column-major pack"},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
