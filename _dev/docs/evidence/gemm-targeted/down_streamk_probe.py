#!/usr/bin/env python3
"""Numerical and timing gate for the isolated down-projection Stream-K build.

Only run with the GPU reserved by the profiling coordinator. Synthetic FP8
operands prove single-operator numerical behavior, not model quality. The
fixed-tile Stream-K candidate can change reduction order, so exact equality
is required for variant 0 versus vLLM but not assumed for variant 1.
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
    Standalone, cudart, fp32_reference, l2_bytes, make_activation, make_weight,
    package_version, profile_one, sample_coordinates, score, time_cold,
    time_warm,
)


K = 17408
N = 5120
DEFAULT_M = (1, 5, 16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--m", nargs="+", type=int, default=list(DEFAULT_M))
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--cold-repetitions", type=int, default=20)
    parser.add_argument("--warm-rep-ms", type=int, default=30)
    parser.add_argument("--order-seed", type=int, default=20261015)
    parser.add_argument("--max-nrmse", type=float, default=0.02)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--ncu-one-shape", action="store_true")
    parser.add_argument("--ncu-variant", choices=("official", "0", "1"), default="1")
    args = parser.parse_args()
    if not args.library.is_file():
        parser.error("--library must name built libdown_streamk_fp8.so")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        parser.error("requires reserved SM120 GPU")
    if not args.m or any(m < 1 for m in args.m):
        parser.error("M must be positive")
    if args.ncu_one_shape and len(args.m) != 1:
        parser.error("--ncu-one-shape requires one M")
    if args.workspace_mib < 1 or args.cold_repetitions < 1 or args.warm_rep_ms < 1:
        parser.error("workspace and timing lengths must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    runtime = cudart()
    l2 = l2_bytes(runtime, torch.cuda.current_device())
    swizzle = 8 if K * N > l2 else 1
    workspace = torch.empty(args.workspace_mib * 1024 * 1024,
                            device="cuda", dtype=torch.uint8)
    call = Standalone(args.library, workspace, swizzle)
    flush = None if args.ncu_one_shape else torch.empty(
        max(2 * l2, 64 * 1024 * 1024), device="cuda", dtype=torch.uint8)
    print(json.dumps({"metadata": {
        "library": str(args.library), "M": args.m, "K": K, "N": N,
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "vllm": package_version("vllm"), "l2_bytes": l2,
        "weight_bytes": K * N, "swizzle": swizzle,
        "variant_0": "fixed vLLM CUTLASS dispatch",
        "variant_1": "M<=16 down swapAB 128x32x128 Stream-K; otherwise fixed fallback",
        "reference": "FP32 dequantized FP8, sampled first/middle/last 128-column blocks and token rows",
        "max_nrmse": args.max_nrmse, "min_cosine": args.min_cosine,
        "order_seed": args.order_seed, "timestamp_unix": time.time(),
        "cold_limit": "owned >L2 flush approximates cold cache only",
        "warm_limit": "CUDA graph replay of fixed operands; not full-model throughput",
        "validation_scope": "synthetic operator, not model quality",
    }}), flush=True)
    rng = random.Random(args.order_seed)
    weight, b_scale = make_weight(K, N, 20261015)
    for m in args.m:
        a, a_scale = make_activation(m, K, 1000 + m)
        fixed = torch.empty((m, N), device="cuda", dtype=torch.bfloat16)
        candidate = torch.empty_like(fixed)
        call(fixed, a, weight, a_scale, b_scale, 0)
        call(candidate, a, weight, a_scale, b_scale, 1)
        official = ops.cutlass_scaled_mm(a, weight.T, out_dtype=torch.bfloat16,
                                         scale_a=a_scale, scale_b=b_scale.T)
        torch.cuda.synchronize()
        rows, cols = sample_coordinates(m, N)
        reference = fp32_reference(a, a_scale, weight, b_scale, rows, cols)
        fixed_ref = score(fixed[rows][:, cols], reference)
        candidate_ref = score(candidate[rows][:, cols], reference)
        candidate_fixed = score(candidate[rows][:, cols], fixed[rows][:, cols])
        candidate_fixed_full = score(candidate, fixed)
        finite = {"official": bool(torch.isfinite(official).all().item()),
                  "fixed": bool(torch.isfinite(fixed).all().item()),
                  "candidate": bool(torch.isfinite(candidate).all().item())}
        exact_fixed = bool(torch.equal(fixed, official))
        exact_candidate = bool(torch.equal(candidate, fixed))
        active = m <= 16
        passed = (all(finite.values()) and exact_fixed and
                  all(s["finite"] and s["nrmse"] <= args.max_nrmse and
                      s["cosine"] >= args.min_cosine
                      for s in (fixed_ref, candidate_ref)) and
                  candidate_fixed["finite"] and
                  candidate_fixed["nrmse"] <= args.max_nrmse and
                  candidate_fixed["cosine"] >= args.min_cosine and
                  candidate_fixed_full["finite"] and
                  candidate_fixed_full["nrmse"] <= args.max_nrmse and
                  candidate_fixed_full["cosine"] >= args.min_cosine and
                  (active or exact_candidate))
        print(json.dumps({"numerics": {
            "M": m, "K": K, "N": N, "candidate_path_active": active,
            "full_output_finite": finite,
            "baseline_exactly_equals_fixed_vllm_op": exact_fixed,
            "candidate_exactly_equals_baseline": exact_candidate,
            "sample_rows": rows,
            "sample_column_blocks": [cols[0], cols[128], cols[256]],
            "baseline_vs_fp32": fixed_ref, "candidate_vs_fp32": candidate_ref,
            "candidate_vs_baseline_sample": candidate_fixed,
            "candidate_vs_baseline_full": candidate_fixed_full,
            "pass": passed,
        }}, allow_nan=False), flush=True)
        if not passed:
            raise RuntimeError(f"M={m}: numeric gate failed; refusing timings")
        del official, reference

        def run_fixed() -> None:
            call(fixed, a, weight, a_scale, b_scale, 0)

        def run_candidate() -> None:
            call(candidate, a, weight, a_scale, b_scale, 1)

        def run_official() -> None:
            ops.cutlass_scaled_mm(a, weight.T, out_dtype=torch.bfloat16,
                                  scale_a=a_scale, scale_b=b_scale.T)

        for _ in range(10):
            run_fixed()
            run_candidate()
        torch.cuda.synchronize()
        if args.ncu_one_shape:
            selected = {"official": run_official, "0": run_fixed,
                        "1": run_candidate}[args.ncu_variant]
            profile_one(runtime, selected)
            print(json.dumps({"profiled_once": {"M": m, "K": K, "N": N,
                                                 "variant": args.ncu_variant,
                                                 "candidate_path_active": active}}), flush=True)
            continue
        order = [(variant, mode) for variant in (0, 1)
                 for mode in ("cold", "warm")]
        rng.shuffle(order)
        print(json.dumps({"measurement_order": {"M": m, "order": order}}), flush=True)
        for variant, mode in order:
            fn = run_fixed if variant == 0 else run_candidate
            result = (time_cold(fn, flush, args.cold_repetitions)
                      if mode == "cold" else time_warm(fn, args.warm_rep_ms))
            print(json.dumps({"timing": {"M": m, "K": K, "N": N,
                                         "variant": variant, "mode": mode,
                                         **result}}, allow_nan=False), flush=True)
        del a, a_scale, fixed, candidate
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
