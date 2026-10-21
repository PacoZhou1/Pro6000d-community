#!/usr/bin/env python3
"""Numerical gate and isolated timing for targeted_kernel.py's CUDA library.

Run only after the GPU has been reserved for this experiment. This script
never changes a vLLM service. It calls the builder's two variants on the SAME
synthetic FP8 operands and scales. First it requires variant 0 to reproduce
the fixed vLLM CUTLASS operator bit for bit, then checks variant 1 against
variant 0 and sampled FP32 dequantized-FP8 references. Only passing shapes
are timed. Synthetic operator accuracy is not model-quality validation.

Normal mode: CUDA graph replay without cache flush ("warm") and CUDA events
with an owned 2x-L2 flush buffer before each launch ("cold approximation").
One deterministic shuffle of variant/mode order is recorded for each shape.
The full K5120,N34816 weight is ~170 MiB, exceeding this GPU's L2: "warm"
means repeated replay, not that the entire operand resides in L2.

With --ncu-one-shape, require one --m value. After correctness checks, invoke
cudaProfilerStart, launch exactly one GEMM selected by --ncu-variant
(official, 0, or 1), synchronize, then invoke cudaProfilerStop; use with
Nsight Compute --profile-from-start off. A variant-1 launch for M<=256 uses
the unchanged fallback and is labeled accordingly. No timing loop runs in
this mode. Other correctness kernels execute before the profile window.
This is a synthetic microkernel probe, not an inference-service patch.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib.metadata
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)


GROUP = 128
FP8_MAX = 448.0
DEFAULT_M = (1, 5, 16, 2741, 2786, 16160)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cudart() -> ctypes.CDLL:
    name = ctypes.util.find_library("cudart") or "libcudart.so"
    return ctypes.CDLL(name)


def cuda_error(runtime: ctypes.CDLL, code: int, call: str) -> None:
    if code:
        runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
        runtime.cudaGetErrorString.restype = ctypes.c_char_p
        message = runtime.cudaGetErrorString(code).decode(errors="replace")
        raise RuntimeError(f"{call}: CUDA {code}: {message}")


def l2_bytes(runtime: ctypes.CDLL, device: int) -> int:
    # cudaDevAttrL2CacheSize is 38 in cuda_runtime_api.h. The fixed vLLM .cu
    # uses cudaDeviceProp::l2CacheSize for its 1-versus-8 swizzle gate.
    answer = ctypes.c_int()
    runtime.cudaDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int),
                                               ctypes.c_int, ctypes.c_int]
    runtime.cudaDeviceGetAttribute.restype = ctypes.c_int
    cuda_error(runtime, runtime.cudaDeviceGetAttribute(
        ctypes.byref(answer), 38, device), "cudaDeviceGetAttribute(L2CacheSize)")
    if answer.value <= 0:
        raise RuntimeError("CUDA reported nonpositive L2 cache size")
    return answer.value


class Standalone:
    def __init__(self, library: Path, workspace: torch.Tensor, swizzle: int):
        self.lib = ctypes.CDLL(str(library.resolve()))
        self.fn = self.lib.targeted_block_fp8
        self.fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong]
        self.fn.restype = ctypes.c_int
        self.lib.targeted_last_error.argtypes = []
        self.lib.targeted_last_error.restype = ctypes.c_char_p
        self.workspace = workspace
        self.swizzle = swizzle

    def __call__(self, output: torch.Tensor, a: torch.Tensor,
                 weight: torch.Tensor, a_scale: torch.Tensor,
                 weight_scale: torch.Tensor, variant: int) -> None:
        m, k = a.shape
        n = weight.shape[0]
        stream = torch.cuda.current_stream().cuda_stream
        code = self.fn(output.data_ptr(), a.data_ptr(), weight.data_ptr(),
                       a_scale.data_ptr(), weight_scale.data_ptr(),
                       m, n, k, variant, self.swizzle,
                       self.workspace.data_ptr(), self.workspace.numel(), stream)
        if code:
            raise RuntimeError(f"standalone variant {variant}: " +
                               self.lib.targeted_last_error().decode(errors="replace"))


def make_weight(k: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    source = torch.randn((n, k), device="cuda", dtype=torch.float32) / math.sqrt(k)
    blocks = source.view(n // GROUP, GROUP, k // GROUP, GROUP)
    scales = blocks.abs().amax(dim=(1, 3)).div(FP8_MAX).clamp_min(1e-7).contiguous()
    weight = ((blocks / scales[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
              .to(torch.float8_e4m3fn).reshape(n, k).contiguous())
    del source, blocks
    return weight, scales


def make_activation(m: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    source = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    a, scales = per_token_group_quant_fp8(
        source, group_size=GROUP, dtype=torch.float8_e4m3fn,
        column_major_scales=True, use_ue8m0=False)
    del source
    assert a.shape == (m, k) and scales.shape == (m, k // GROUP)
    return a.contiguous(), scales


def sample_coordinates(m: int, n: int) -> tuple[list[int], list[int]]:
    rows = sorted({0, m // 2, m - 1})
    starts = sorted({0, (n // 2 // GROUP) * GROUP, n - GROUP})
    cols = [i for start in starts for i in range(start, start + GROUP)]
    return rows, cols


def fp32_reference(a: torch.Tensor, a_scale: torch.Tensor,
                   weight: torch.Tensor, weight_scale: torch.Tensor,
                   rows: list[int], cols: list[int]) -> torch.Tensor:
    # Only 1-3 token rows and first/middle/last 128-column blocks; all K.
    k = a.shape[1]
    row_index = torch.as_tensor(rows, device="cuda", dtype=torch.long)
    col_index = torch.as_tensor(cols, device="cuda", dtype=torch.long)
    # Index byte views so this probe does not depend on Float8 advanced-index
    # kernels in the fixed torch build. FP8 E4M3FN and uint8 are both 1 byte;
    # reinterpreting the selected bytes preserves every quantized value.
    av8 = a.view(torch.uint8).index_select(0, row_index).view(torch.float8_e4m3fn)
    av = av8.float().view(len(rows), k // GROUP, GROUP)
    av = (av * a_scale[rows].float().contiguous()[:, :, None]).reshape(len(rows), k)
    wv8 = weight.view(torch.uint8).index_select(0, col_index).view(torch.float8_e4m3fn)
    wv = wv8.float().view(len(cols), k // GROUP, GROUP)
    block_rows = torch.as_tensor([col // GROUP for col in cols], device="cuda",
                                 dtype=torch.long)
    ws = weight_scale[block_rows].float()
    wv = (wv * ws[:, :, None]).reshape(len(cols), k)
    return av @ wv.T


def score(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    x, y = actual.float(), expected.float()
    finite = bool(torch.isfinite(x).all().item())
    if not finite:
        return {"finite": False, "nrmse": None, "cosine": None,
                "max_abs": None}
    diff = x - y
    denom = y.square().mean().sqrt().clamp_min(1e-12)
    return {"finite": True,
            "nrmse": float((diff.square().mean().sqrt() / denom).item()),
            "cosine": float(torch.nn.functional.cosine_similarity(
                x.reshape(1, -1), y.reshape(1, -1), dim=1)[0].item()),
            "max_abs": float(diff.abs().max().item())}


def numerical_gate(call: Standalone, a: torch.Tensor, a_scale: torch.Tensor,
                   weight: torch.Tensor, weight_scale: torch.Tensor,
                   baseline: torch.Tensor, candidate: torch.Tensor,
                   max_nrmse: float, min_cosine: float) -> dict:
    m, k = a.shape
    n = weight.shape[0]
    call(baseline, a, weight, a_scale, weight_scale, 0)
    call(candidate, a, weight, a_scale, weight_scale, 1)
    official = ops.cutlass_scaled_mm(
        a, weight.T, out_dtype=torch.bfloat16, scale_a=a_scale,
        scale_b=weight_scale.T)
    torch.cuda.synchronize()
    base_finite = bool(torch.isfinite(baseline).all().item())
    cand_finite = bool(torch.isfinite(candidate).all().item())
    official_finite = bool(torch.isfinite(official).all().item())
    # This exact gate catches a bad standalone adapter/stride/scheduler before
    # any performance result is used as evidence about vLLM's implementation.
    base_exact = bool(torch.equal(baseline, official))
    rows, cols = sample_coordinates(m, n)
    ref = fp32_reference(a, a_scale, weight, weight_scale, rows, cols)
    base_sample = baseline[rows][:, cols]
    cand_sample = candidate[rows][:, cols]
    official_sample = official[rows][:, cols]
    base_vs_official = score(base_sample, official_sample)
    cand_vs_base = score(cand_sample, base_sample)
    base_vs_fp32 = score(base_sample, ref)
    cand_vs_fp32 = score(cand_sample, ref)
    fallback_shape = not (m > 256 and k == 5120 and n == 34816)
    fallback_exact = bool(torch.equal(candidate, baseline)) if fallback_shape else None
    passed = (base_finite and cand_finite and official_finite and base_exact and
              all(s["finite"] and s["nrmse"] <= max_nrmse and
                  s["cosine"] >= min_cosine for s in (base_vs_fp32, cand_vs_fp32)) and
              cand_vs_base["finite"] and cand_vs_base["nrmse"] <= max_nrmse and
              cand_vs_base["cosine"] >= min_cosine and
              (fallback_exact is not False))
    result = {"M": m, "K": k, "N": n,
              "experimental_candidate_path_active": not fallback_shape,
              "sample_rows": rows,
              "sample_columns": [cols[0], cols[GROUP], cols[2 * GROUP]],
              "full_output_finite": {"official": official_finite,
                                     "standalone_baseline": base_finite,
                                     "standalone_candidate": cand_finite},
              "baseline_exactly_equals_fixed_vllm_op": base_exact,
              "fallback_candidate_exactly_equals_baseline": fallback_exact,
              "baseline_vs_official_sample": base_vs_official,
              "candidate_vs_baseline_sample": cand_vs_base,
              "baseline_vs_fp32_quantized_sample": base_vs_fp32,
              "candidate_vs_fp32_quantized_sample": cand_vs_fp32,
              "pass": passed}
    del official
    return result


def time_cold(fn, flush: torch.Tensor, repetitions: int) -> dict:
    times = []
    for _ in range(repetitions):
        # This buffer belongs to the probe and is >L2. Eviction is an
        # approximation, not a hardware assertion that every cache line is cold.
        flush.zero_()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end) * 1000.0)
    return {"n": len(times), "median_us": statistics.median(times),
            "min_us": min(times), "max_us": max(times), "values_us": times}


def time_warm(fn, rep_ms: int) -> dict:
    from triton.testing import do_bench_cudagraph
    # This captures many identical launches and measures ten replays. The
    # CUTLASS kernel and workspace are already initialized and preallocated.
    values = do_bench_cudagraph(fn, rep=rep_ms, return_mode="all")
    micros = [float(x) * 1000.0 for x in values]
    return {"n": len(micros), "median_us": statistics.median(micros),
            "min_us": min(micros), "max_us": max(micros), "values_us": micros}


def profile_one(runtime: ctypes.CDLL, fn) -> None:
    runtime.cudaProfilerStart.argtypes = []
    runtime.cudaProfilerStart.restype = ctypes.c_int
    runtime.cudaProfilerStop.argtypes = []
    runtime.cudaProfilerStop.restype = ctypes.c_int
    cuda_error(runtime, runtime.cudaProfilerStart(), "cudaProfilerStart")
    try:
        fn()  # exactly one targeted GEMM launch inside the profiler window
        torch.cuda.synchronize()
    finally:
        cuda_error(runtime, runtime.cudaProfilerStop(), "cudaProfilerStop")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--m", type=int, nargs="+", default=list(DEFAULT_M))
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--n", type=int, default=34816)
    parser.add_argument("--workspace-mib", type=int, default=64)
    parser.add_argument("--cold-repetitions", type=int, default=15)
    parser.add_argument("--warm-rep-ms", type=int, default=20)
    parser.add_argument("--order-seed", type=int, default=20261014)
    parser.add_argument("--max-nrmse", type=float, default=0.02)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--ncu-one-shape", action="store_true")
    parser.add_argument("--ncu-variant", choices=("official", "0", "1"), default="1",
                        help="single GEMM to profile after numeric gates")
    args = parser.parse_args()
    if not args.library.is_file():
        parser.error("--library must name a built libtargeted_fp8.so")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        parser.error("requires the reserved SM120 CUDA GPU")
    if args.k <= 0 or args.n < 3 * GROUP or args.k % GROUP or args.n % GROUP:
        parser.error("K must be a positive multiple of 128; N needs at least three 128-column blocks")
    if not args.m or any(m <= 0 for m in args.m):
        parser.error("M values must be positive")
    if args.ncu_one_shape and len(args.m) != 1:
        parser.error("--ncu-one-shape requires exactly one --m value")
    if args.workspace_mib < 1 or args.cold_repetitions < 1 or args.warm_rep_ms < 1:
        parser.error("workspace and timing lengths must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    runtime = cudart()
    l2 = l2_bytes(runtime, torch.cuda.current_device())
    swizzle = 8 if args.k * args.n > l2 else 1
    workspace = torch.empty(args.workspace_mib * 1024 * 1024,
                            device="cuda", dtype=torch.uint8)
    call = Standalone(args.library, workspace, swizzle)
    flush = None if args.ncu_one_shape else torch.empty(
        max(2 * l2, 64 * 1024 * 1024), device="cuda", dtype=torch.uint8)
    print(json.dumps({"metadata": {"library": str(args.library),
                                   "device": torch.cuda.get_device_name(),
                                   "torch": torch.__version__, "vllm": package_version("vllm"),
                                   "l2_bytes": l2, "weight_bytes": args.k * args.n,
                                   "swizzle": swizzle,
                                   "official": "fixed vLLM ops.cutlass_scaled_mm",
                                   "variant_0": "standalone fixed dispatch, required bit-equal to official",
                                   "variant_1": "experimental Pingpong only for M>256 K5120/N34816; fixed fallback elsewhere",
                                   "order_seed": args.order_seed,
                                   "reference": "FP32 dequantized FP8, selected token rows and first/middle/last N128 blocks",
                                   "cold_limit": "owned >L2 flush is approximate eviction",
                                   "warm_limit": "graph replay reuses operands but entire weight may exceed L2",
                                   "validation_scope": "synthetic standalone operator; no service integration or model quality claim",
                                   "timestamp_unix": time.time()}}), flush=True)
    rng = random.Random(args.order_seed)
    weight, weight_scale = make_weight(args.k, args.n, 20261014)
    for m in args.m:
        a, a_scale = make_activation(m, args.k, 1000 + m)
        baseline = torch.empty((m, args.n), device="cuda", dtype=torch.bfloat16)
        candidate = torch.empty_like(baseline)
        result = numerical_gate(call, a, a_scale, weight, weight_scale,
                                baseline, candidate, args.max_nrmse, args.min_cosine)
        print(json.dumps({"numerics": result}, allow_nan=False), flush=True)
        if not result["pass"]:
            raise RuntimeError(f"M={m}: numeric gate failed; refusing performance result")
        def run_base():
            call(baseline, a, weight, a_scale, weight_scale, 0)
        def run_candidate():
            call(candidate, a, weight, a_scale, weight_scale, 1)
        def run_official():
            ops.cutlass_scaled_mm(a, weight.T, out_dtype=torch.bfloat16,
                                  scale_a=a_scale, scale_b=weight_scale.T)
        # Warm both kernels before randomizing their measurement sequence.
        for _ in range(10):
            run_base()
            run_candidate()
            if args.ncu_one_shape and args.ncu_variant == "official":
                run_official()
        torch.cuda.synchronize()
        if args.ncu_one_shape:
            selected = {"official": run_official, "0": run_base,
                        "1": run_candidate}[args.ncu_variant]
            profile_one(runtime, selected)
            print(json.dumps({"profiled_once": {"M": m, "K": args.k, "N": args.n,
                                                 "variant": args.ncu_variant,
                                                 "experimental_candidate_path_active":
                                                 (args.ncu_variant == "1" and m > 256
                                                  and args.k == 5120 and args.n == 34816)}}),
                  flush=True)
            continue
        tasks = [(v, mode) for v in (0, 1) for mode in ("cold", "warm")]
        rng.shuffle(tasks)
        print(json.dumps({"measurement_order": {"M": m, "order": tasks}}), flush=True)
        for variant, mode in tasks:
            fn = run_base if variant == 0 else run_candidate
            timing = (time_cold(fn, flush, args.cold_repetitions)
                      if mode == "cold" else time_warm(fn, args.warm_rep_ms))
            print(json.dumps({"timing": {"M": m, "K": args.k, "N": args.n,
                                         "variant": variant, "mode": mode,
                                         **timing}}, allow_nan=False), flush=True)
        del a, a_scale, baseline, candidate
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
