#!/usr/bin/env python3
"""Emit a one-shape SM120 CUTLASS block-FP8 dispatch experiment patch.

Offline only: reads one header, verifies the exact vLLM 0.31.0 source hash,
and prints a unified diff. It never loads CUDA or changes a checkout.

Example, from a checkout of vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6:

  python targeted_kernel.py \
    csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/\
scaled_mm_blockwise_sm120_fp8_dispatch.cuh > targeted-prefill.patch

The experimental dispatch reuses the already-instantiated 64x128x128
Pingpong configuration only for M>256, K=5120, N=34816. All other shapes
retain their original CUTLASS path. This is a test candidate, not a known
speedup. Rebuild the CUDA extension, then verify kernel names, FP32-reference
numerics and unprofiled inference with identical requests before considering
it for deployment. Split-K and BF16 lm_head are deferred pending shape and
module attribution; neither can safely be inferred from a kernel name.

For a lightweight standalone build (no GPU needed), use --build-dir and
--cutlass-include. The resulting library exports targeted_block_fp8 and
targeted_last_error. targeted_block_fp8 accepts output,A,B,As,Bs pointers;
M,N,K; variant (0 fixed, 1 candidate); swizzle (1 or 8); preallocated CUDA
workspace pointer and byte count; current CUDA stream pointer. It returns
zero on successful launch. This C ABI is deliberately single-threaded and
does not allocate on the GPU, so a caller can capture it in a CUDA graph.
The fixed .cu selects swizzle=8 when K*N bytes exceed device L2; pass the
same value for both variants. This is an isolated CUTLASS measurement path,
not a drop-in replacement for vLLM's Python operator.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PINNED_SHA256 = "8a9ee5dc8edb96bc4619883a6248eb15211fd37db2d3fc02de25c5b02ee5af41"
RELATIVE = (
    "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/"
    "scaled_mm_blockwise_sm120_fp8_dispatch.cuh"
)
OLD = """    // M > 256: use default 128x128x128 config with Cooperative (Auto) schedule
    using Gemm = typename sm120_blockwise_fp8_config_default<OutType>::Gemm;
"""
NEW = """    // Experimental Qwen3.6-27B gate/up prefill path. Reuse the same
    // Pingpong kernel already selected for 65 <= M <= 256; only dispatch
    // changes. Keep the original default for every other model shape.
    if (a.size(1) == 5120 && b.size(1) == 34816) {
      using Gemm = typename sm120_blockwise_fp8_config_pingpong<OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          out, a, b, a_scales, b_scales, max_swizzle_size);
    }
    // M > 256: use default 128x128x128 config with Cooperative (Auto) schedule
    using Gemm = typename sm120_blockwise_fp8_config_default<OutType>::Gemm;
"""

ADAPTER = r'''#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace vllm {
struct TensorView {
  void* ptr;
  int64_t d0, d1;
  int device_id;
  int64_t size(int i) const { return i == 0 ? d0 : d1; }
  void* data_ptr() const { return ptr; }
  int device() const { return device_id; }
};

// Exact architecture guard shape from fixed vLLM cutlass_extensions/common.hpp.
template <typename Kernel>
struct enable_sm120_family : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
#if (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
    Kernel::operator()(std::forward<Args>(args)...);
#else
    asm("trap;");
#endif
#endif
  }
};

namespace c3x {
// The caller is adapted from fixed vLLM cutlass_gemm_caller.cuh. Workspace
// comes from the Python-owned CUDA tensor so graph capture allocates nothing.
inline void* workspace = nullptr;
inline size_t workspace_bytes = 0;
inline cudaStream_t stream = nullptr;

template <typename GemmKernel>
void cutlass_gemm_caller(
    int device, cute::Shape<int, int, int, int> prob_shape,
    typename GemmKernel::MainloopArguments mainloop_args,
    typename GemmKernel::EpilogueArguments epilogue_args,
    typename GemmKernel::TileSchedulerArguments scheduler = {}) {
  (void)device;
  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape, mainloop_args, epilogue_args,
                                      hw_info, scheduler};
  using Op = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  Op op;
  auto status = op.can_implement(args);
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(status));
  if (op.get_workspace_size(args) > workspace_bytes)
    throw std::runtime_error("workspace too small");
  status = op.run(args, workspace, stream);
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(status));
}
} // namespace c3x
} // namespace vllm
'''

DRIVER = r'''#include "standalone_dispatch.cuh"
#include <cstring>
#include <exception>

namespace {
thread_local char error_message[512] = {0};
}

extern "C" const char* targeted_last_error() { return error_message; }

// Inputs use the same layout as the vLLM Python adapter: A [M,K] contiguous,
// B storage [N,K] contiguous (passed as a transposed [K,N] view), A scales
// [M,K/128] column-major, B scales [N/128,K/128] contiguous. Output [M,N].
// variant=0: exact fixed vLLM M-only dispatch; variant=1: only large-M
// K5120/N34816 uses the existing Pingpong config. Both use the SAME swizzle.
extern "C" int targeted_block_fp8(
    void* output, const void* a, const void* b, const void* as,
    const void* bs, int m, int n, int k, int variant, int swizzle,
    void* workspace, size_t workspace_bytes, unsigned long long stream) {
  error_message[0] = 0;
  try {
    if (!output || !a || !b || !as || !bs || !workspace ||
        m <= 0 || n <= 0 || k <= 0 || (variant != 0 && variant != 1) ||
        (swizzle != 1 && swizzle != 8))
      throw std::runtime_error("invalid argument");
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("cudaGetDevice failed");
    vllm::c3x::workspace = workspace;
    vllm::c3x::workspace_bytes = workspace_bytes;
    vllm::c3x::stream = reinterpret_cast<cudaStream_t>(stream);
    vllm::TensorView o{output, m, n, device};
    vllm::TensorView av{const_cast<void*>(a), m, k, device};
    vllm::TensorView bv{const_cast<void*>(b), k, n, device};
    vllm::TensorView asv{const_cast<void*>(as), m, k / 128, device};
    vllm::TensorView bsv{const_cast<void*>(bs), k / 128, n / 128, device};
    using Out = cutlass::bfloat16_t;
    if (variant == 1 && m > 256 && k == 5120 && n == 34816) {
      using Gemm = typename vllm::sm120_blockwise_fp8_config_pingpong<Out>::Gemm;
      vllm::cutlass_gemm_caller_blockwise<Gemm>(
          o, av, bv, asv, bsv, swizzle);
    } else {
      vllm::cutlass_gemm_blockwise_sm120_fp8_dispatch<Out>(
          o, av, bv, asv, bsv, swizzle);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::strncpy(error_message, ex.what(), sizeof(error_message) - 1);
    error_message[sizeof(error_message) - 1] = 0;
    return 1;
  }
}
'''


def checked_source(path: Path) -> str:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != PINNED_SHA256:
        raise ValueError(f"source SHA256 {actual} != fixed vLLM 0.31.0 {PINNED_SHA256}")
    return raw.decode("utf-8")


def adapted_dispatch(source: str) -> str:
    # Strip the vLLM/PyTorch tensor wrapper. The CUTLASS kernel templates,
    # tile shapes, scale layout and original dispatch body remain untouched.
    replacements = {
        '#include <torch/headeronly/util/shim_utils.h>': '',
        '#include "cuda_utils.h"': '#include "standalone_adapter.cuh"',
        '#include "cutlass_gemm_caller.cuh"': '',
        'torch::stable::Tensor': 'TensorView',
    }
    for old, new in replacements.items():
        if old not in source:
            raise ValueError(f"missing expected header fragment: {old}")
        source = source.replace(old, new)
    return source


def build(source: Path, build_dir: Path, cutlass: Path, nvcc: Path) -> None:
    original = checked_source(source)
    if not (cutlass / "cutlass" / "cutlass.h").is_file():
        raise ValueError(f"not a CUTLASS include directory: {cutlass}")
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "standalone_adapter.cuh").write_text(ADAPTER)
    (build_dir / "standalone_dispatch.cuh").write_text(adapted_dispatch(original))
    (build_dir / "targeted_driver.cu").write_text(DRIVER)
    output = build_dir / "libtargeted_fp8.so"
    util_include = cutlass.parent / "tools" / "util" / "include"
    if not (util_include / "cutlass" / "util" / "packed_stride.hpp").is_file():
        raise ValueError(f"CUTLASS tools/util/include missing: {util_include}")
    cmd = [str(nvcc), "-std=c++17", "-O3", "--expt-relaxed-constexpr",
           "--expt-extended-lambda", "-arch=sm_120a", "-Xcompiler=-fPIC",
           "-shared", "-I", str(build_dir), "-I", str(cutlass),
           "-I", str(util_include),
           str(build_dir / "targeted_driver.cu"), "-o", str(output), "-lcudart"]
    subprocess.run(cmd, check=True)
    print(json.dumps({"library": str(output), "source_sha256": PINNED_SHA256,
                      "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "cuda_arch": "sm_120a", "variant_0": "fixed vLLM dispatch",
                      "variant_1": "M>256,K5120,N34816 uses 64x128x128 pingpong",
                      "swizzle": "caller supplies identical 1 or 8 for both variants"}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help=f"exact fixed source: {RELATIVE}")
    parser.add_argument("--build-dir", type=Path,
                        help="emit and compile isolated SM120 shared library here")
    parser.add_argument("--cutlass-include", type=Path,
                        help="CUTLASS include root; required with --build-dir")
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    args = parser.parse_args()
    if args.build_dir:
        if not args.cutlass_include:
            parser.error("--cutlass-include is required with --build-dir")
        build(args.source, args.build_dir, args.cutlass_include, args.nvcc)
        return 0
    before = checked_source(args.source)
    if before.count(OLD) != 1:
        parser.error("dispatch anchor is not unique; refusing to emit patch")
    after = before.replace(OLD, NEW, 1)
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile="a/" + RELATIVE,
        tofile="b/" + RELATIVE,
    )
    sys.stdout.writelines(diff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
