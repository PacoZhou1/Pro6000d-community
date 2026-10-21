#!/usr/bin/env python3
"""Isolated N-split test for the fixed vLLM 0.31.0 SM120 FP8 block GEMM.

Run only when the GPU is reserved for this experiment. Synthetic FP8 inputs
are generated once per shape; the full and split paths share the exact A, B,
As and Bs values. Splitting changes only B/Bs views at a 128-column boundary.
The full-output torch.equal gate against the original official op must pass
before any timings are emitted. This is operator validation, not model quality.

The standalone baseline variant 0 was separately gated against the fixed
official operator in standalone_fp8_probe.py. This probe gates it again per
shape. The standalone split runs are compared at fixed swizzle 8 and 1 to
separate the N-split effect from vLLM's L2-dependent swizzle heuristic.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from vllm import _custom_ops as ops

from standalone_fp8_probe import (
    GROUP, Standalone, cudart, l2_bytes, make_activation, make_weight,
    package_version, time_cold, time_warm,
)


K = 5120
N = 34816
HALF = N // 2
DEFAULT_M = (1, 5, 16, 2741, 2786, 16160)


def official(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor,
             b_scale: torch.Tensor) -> torch.Tensor:
    return ops.cutlass_scaled_mm(a, b.T, out_dtype=torch.bfloat16,
                                 scale_a=a_scale, scale_b=b_scale.T)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--m", nargs="+", type=int, default=list(DEFAULT_M))
    parser.add_argument("--workspace-mib", type=int, default=64)
    parser.add_argument("--cold-repetitions", type=int, default=12)
    parser.add_argument("--warm-rep-ms", type=int, default=20)
    parser.add_argument("--order-seed", type=int, default=20261015)
    args = parser.parse_args()
    if not args.library.is_file():
        parser.error("--library must name the tested libtargeted_fp8.so")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        parser.error("requires a reserved SM120 GPU")
    if not args.m or any(m <= 0 for m in args.m):
        parser.error("M values must be positive")
    if args.workspace_mib < 1 or args.cold_repetitions < 1 or args.warm_rep_ms < 1:
        parser.error("workspace and repetition counts must be positive")
    assert HALF % GROUP == 0 and K % GROUP == 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    l2 = l2_bytes(cudart(), torch.cuda.current_device())
    full_swizzle = 8 if K * N > l2 else 1
    half_swizzle = 8 if K * HALF > l2 else 1
    workspace = torch.empty(args.workspace_mib * 1024 * 1024,
                            device="cuda", dtype=torch.uint8)
    standalone = {s: Standalone(args.library, workspace, s) for s in (1, 8)}
    flush = torch.empty(max(2 * l2, 64 * 1024 * 1024),
                        device="cuda", dtype=torch.uint8)
    rng = random.Random(args.order_seed)
    print(json.dumps({"metadata": {
        "library": str(args.library), "torch": torch.__version__,
        "vllm": package_version("vllm"), "device": torch.cuda.get_device_name(),
        "K": K, "N": N, "N_half": HALF,
        "weight_bytes": K * N, "half_weight_bytes": K * HALF,
        "l2_bytes": l2, "official_full_swizzle_expected": full_swizzle,
        "official_half_swizzle_expected": half_swizzle,
        "order_seed": args.order_seed, "timestamp_unix": time.time(),
        "validation_scope": "synthetic FP8 operator; not model quality",
        "gemm_only_scope": "two sequential GEMM launches; excludes output concatenation",
        "end_to_end_scope": "two sequential GEMM launches plus torch.cat returning a new contiguous output",
        "cold_limit": "owned >L2 buffer zeroed before each timed sequence; approximate eviction only",
        "warm_limit": "CUDA graph replay; cache residency of all operands is not guaranteed",
        "comparison_limit": "sequences differ in output allocations/traffic; short samples are not inference throughput",
    }}), flush=True)

    weight, b_scale = make_weight(K, N, 20261014)
    assert weight.is_contiguous() and b_scale.is_contiguous()
    b_parts = (weight[:HALF], weight[HALF:])
    bs_parts = (b_scale[:HALF // GROUP], b_scale[HALF // GROUP:])
    assert all(x.is_contiguous() for x in (*b_parts, *bs_parts))
    assert b_parts[0].data_ptr() == weight.data_ptr()
    assert bs_parts[0].data_ptr() == b_scale.data_ptr()

    for m in args.m:
        a, a_scale = make_activation(m, K, 1000 + m)
        full = torch.empty((m, N), device="cuda", dtype=torch.bfloat16)
        halves = tuple(torch.empty((m, HALF), device="cuda", dtype=torch.bfloat16)
                       for _ in range(2))
        # The same contiguous half buffers are reused by the standalone paths.
        # Each path is checked before another overwrites them.
        def full_standalone() -> None:
            standalone[full_swizzle](full, a, weight, a_scale, b_scale, 0)

        def split_standalone(swizzle: int) -> None:
            caller = standalone[swizzle]
            for i in range(2):
                caller(halves[i], a, b_parts[i], a_scale, bs_parts[i], 0)

        def standalone_split_e2e(swizzle: int) -> torch.Tensor:
            split_standalone(swizzle)
            return torch.cat(halves, dim=1)

        # This exactly reproduces the runtime patch's two op calls and cat.
        def official_split_e2e() -> torch.Tensor:
            left = official(a, b_parts[0], a_scale, bs_parts[0])
            right = official(a, b_parts[1], a_scale, bs_parts[1])
            return torch.cat((left, right), dim=1)

        original = official(a, weight, a_scale, b_scale)
        full_standalone()
        full_exact = bool(torch.equal(full, original))
        finite_original = bool(torch.isfinite(original).all().item())
        checks: dict[str, bool] = {
            "full_standalone_exact_official": full_exact,
            "official_full_finite": finite_original,
        }
        for name, runner in (
            ("official_auto", official_split_e2e),
            ("standalone_swizzle8", lambda: standalone_split_e2e(8)),
            ("standalone_swizzle1", lambda: standalone_split_e2e(1)),
        ):
            result = runner()
            torch.cuda.synchronize()
            checks[f"{name}_exact_official"] = bool(torch.equal(result, original))
            checks[f"{name}_finite"] = bool(torch.isfinite(result).all().item())
        print(json.dumps({"numerics": {"M": m, "K": K, "N": N,
                                       "half_n": HALF, "checks": checks,
                                       "pass": all(checks.values())}}), flush=True)
        if not all(checks.values()):
            raise RuntimeError(f"M={m}: full-output exactness failed; refusing timing")
        del original, result

        # Official auto GEMM-only returns newly allocated halves but excludes
        # cat. Standalone GEMM-only writes into preallocated halves. These are
        # distinct measurement scopes and are explicitly named in results.
        def official_split_gemm_only() -> None:
            official(a, b_parts[0], a_scale, bs_parts[0])
            official(a, b_parts[1], a_scale, bs_parts[1])

        runners = {
            "original_official": lambda: official(a, weight, a_scale, b_scale),
            "original_standalone_swizzle8": full_standalone,
            "split_official_auto_gemm_only": official_split_gemm_only,
            "split_official_auto_e2e": official_split_e2e,
            "split_standalone_swizzle8_gemm_only": lambda: split_standalone(8),
            "split_standalone_swizzle8_e2e": lambda: standalone_split_e2e(8),
            "split_standalone_swizzle1_gemm_only": lambda: split_standalone(1),
            "split_standalone_swizzle1_e2e": lambda: standalone_split_e2e(1),
        }
        for runner in runners.values():
            runner()
        torch.cuda.synchronize()
        order = [(name, mode) for name in runners for mode in ("warm", "cold")]
        rng.shuffle(order)
        print(json.dumps({"measurement_order": {"M": m, "order": order}}), flush=True)
        for name, mode in order:
            result = (time_warm(runners[name], args.warm_rep_ms)
                      if mode == "warm"
                      else time_cold(runners[name], flush, args.cold_repetitions))
            print(json.dumps({"timing": {"M": m, "K": K, "N": N,
                                         "path": name, "mode": mode,
                                         **result}}, allow_nan=False), flush=True)
        del a, a_scale, full, halves
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
