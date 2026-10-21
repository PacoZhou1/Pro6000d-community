#!/usr/bin/env python3
"""Isolated BF16 epilogue control: GEMM, inverse permutation and full chain.

One M/raster/swizzle arm per process avoids confusing the main eight-arm
fused comparison or holding six BF16 intermediates at once. No GPU runs here;
the parent schedules execution after compilation.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import traceback
from pathlib import Path

from scheduler_probe import (K, N, H, GROUP, RASTERS, SchedulerLibrary,
                             check_sha, cuda_versions, emit, import_file, input_fingerprint,
                             measure_graphs, profile_once, sha)


def inverse_into(torch, canonical, paired):
    # [M, gate||up] <- [M, gate128,up128,gate128,up128,...].
    m = paired.shape[0]
    canonical.view(m, 2, H // GROUP, GROUP).copy_(
        paired.view(m, H // GROUP, 2, GROUP).transpose(1, 2))


def run(args):
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("reserved SM120 GPU required")
    torch.backends.cuda.matmul.allow_tf32 = False
    old = import_file(args.old_probe, "fixed_pipeline_probe_for_epilogue")
    hashes = {"old_probe": check_sha(args.old_probe, args.old_probe_sha256),
              "baseline": check_sha(args.baseline_library, args.baseline_sha256),
              "quant": check_sha(args.quant_library, args.quant_sha256),
              "scheduler": check_sha(args.scheduler_library, args.scheduler_sha256),
              "probe": sha(Path(__file__))}
    scheduler = SchedulerLibrary(torch, args.scheduler_library)
    runtime = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
    l2 = old.l2_size(runtime, torch.cuda.current_device())
    b, bs = old.make_weight(torch, N, K, args.seed)
    pb, pbs = old.packed_weights(torch, b, bs)
    a, asc = old.make_activation(torch, per_token_group_quant_fp8, args.m,
                                 args.seed + 100 + args.m)
    input_hash = input_fingerprint(torch, a, asc)
    r = RASTERS[args.raster]
    so = scheduler.prepare("official_bf16", args.m, r, args.swizzle)
    sp = scheduler.prepare("paired_bf16", args.m, r, args.swizzle)
    xoff = torch.empty((args.m, N), dtype=torch.bfloat16, device="cuda")
    xpair = torch.empty_like(xoff)
    xcanon = torch.empty_like(xoff)
    qoff, soff = old.output_buffers(torch, args.m)
    qpair, spair = old.output_buffers(torch, args.m)
    base = old.Baseline(torch, args.baseline_library, args.quant_library,
                        args.workspace_mib << 20, 8)
    xb = torch.empty_like(xoff)
    qb, sb = old.output_buffers(torch, args.m)

    def old_chain():
        base.gemm(xb, a, asc, b, bs)
        base.quant(xb, qb, sb)
    def official_gemm(): scheduler.call(so, xoff, None, a, asc, b, bs)
    def paired_gemm(): scheduler.call(sp, xpair, None, a, asc, pb, pbs)
    def inverse(): inverse_into(torch, xcanon, xpair)
    def official_chain():
        official_gemm()
        base.quant(xoff, qoff, soff)
    def paired_chain():
        paired_gemm()
        inverse()
        base.quant(xcanon, qpair, spair)
    def quant_only():
        base.quant(xcanon, qpair, spair)

    emit("metadata", {"scope": "synthetic BF16 epilogue/scheduler diagnostic; no fused pipeline speed claim",
                      "M": args.m, "K": K, "N": N, "raster": args.raster,
                      "swizzle": args.swizzle, "seed": args.seed,
                      "activation_actual_seed": args.seed + 100 + args.m,
                      "input_sha256": input_hash, "sha256": hashes,
                      "scheduler_abi": 1, "scheduler_metadata": scheduler.metadata,
                      "l2_bytes": l2, "device": torch.cuda.get_device_name(),
                      "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
                      "cuda_versions": cuda_versions(runtime),
                      "inverse_cost_separate": True,
                      "full_paired_chain": "paired BF16 GEMM + inverse copy + V2 quant (3 kernels)",
                      "full_official_chain": "new official BF16 GEMM + V2 quant (2 kernels)"})

    old_chain()
    official_chain()
    paired_chain()
    xr = ops.cutlass_scaled_mm(a, b.T, out_dtype=torch.bfloat16,
                               scale_a=asc, scale_b=bs.T)
    qr, sr = old.output_buffers(torch, args.m)
    torch.ops._C.silu_and_mul_per_block_quant(qr, xr, sr, GROUP, None, True)
    torch.cuda.synchronize()
    checks = [old.bit_comparison(torch, xb, xr, "old_BF16_vs_vllm", torch.int16),
              old.bit_comparison(torch, xoff, xr, "new_official_BF16", torch.int16),
              old.bit_comparison(torch, xcanon, xr, "new_paired_inverse_BF16", torch.int16),
              old.bit_comparison(torch, qb, qr, "old_FP8_vs_vllm", torch.uint8),
              old.bit_comparison(torch, sb, sr, "old_scale_vs_vllm", torch.int32),
              old.bit_comparison(torch, qoff, qr, "new_official_FP8", torch.uint8),
              old.bit_comparison(torch, soff, sr, "new_official_scale", torch.int32),
              old.bit_comparison(torch, qpair, qr, "new_paired_FP8", torch.uint8),
              old.bit_comparison(torch, spair, sr, "new_paired_scale", torch.int32)]
    if not all(c["pass"] for c in checks):
        emit("numerics", {"pass": False, "comparisons": checks})
        emit("summary", {"pass": False, "timings_emitted": False})
        return 1

    # Capture each chain on unchanged buffers, then mutate actual FP8 input
    # and scales; replay must update both BF16 and quantized output.
    graphs = {"old_pipeline": old.capture(torch, old_chain),
              "new_official_pipeline": old.capture(torch, official_chain),
              "paired_pipeline": old.capture(torch, paired_chain)}
    prior = {"old_pipeline": qb.view(torch.uint8).clone(),
             "new_official_pipeline": qoff.view(torch.uint8).clone(),
             "paired_pipeline": qpair.view(torch.uint8).clone()}
    original_a = a.view(torch.uint8).clone()
    original_asc = asc.contiguous().view(torch.uint8).clone()
    fresh_a, fresh_asc = old.make_activation(torch, per_token_group_quant_fp8,
                                             args.m, args.seed + 9000 + args.m)
    a.copy_(fresh_a)
    asc.copy_(fresh_asc)
    for graph in graphs.values(): graph.replay()
    fresh_x = ops.cutlass_scaled_mm(a, b.T, out_dtype=torch.bfloat16,
                                    scale_a=asc, scale_b=bs.T)
    fresh_q, fresh_s = old.output_buffers(torch, args.m)
    torch.ops._C.silu_and_mul_per_block_quant(fresh_q, fresh_x, fresh_s, GROUP, None, True)
    torch.cuda.synchronize()
    graph_checks = []
    for name, q, sc in (("old_pipeline", qb, sb),
                        ("new_official_pipeline", qoff, soff),
                        ("paired_pipeline", qpair, spair)):
        changed = not bool(torch.equal(prior[name], q.view(torch.uint8)))
        results = [old.bit_comparison(torch, q, fresh_q, f"{name}_replay_FP8", torch.uint8),
                   old.bit_comparison(torch, sc, fresh_s, f"{name}_replay_scales", torch.int32)]
        graph_checks.append({"arm": name, "output_changed": changed,
                             "pass": changed and all(c["pass"] for c in results),
                             "comparisons": results})
    passed = all(x["pass"] for x in graph_checks)
    emit("numerics", {"pass": passed, "comparisons": checks,
                      "changed_input_graph": graph_checks})
    if not passed:
        emit("summary", {"pass": False, "timings_emitted": False})
        return 1
    a.copy_(original_a.view(torch.float8_e4m3fn))
    asc.copy_(original_asc.view(torch.float32).reshape(args.m, K // GROUP))
    old_chain()
    official_chain()
    paired_chain()
    del graphs, prior, fresh_x, fresh_q, fresh_s, fresh_a, fresh_asc, original_a, original_asc

    if args.ncu_target:
        targets = {"old_pipeline": old_chain, "new_official_gemm": official_gemm,
                   "new_official_pipeline": official_chain,
                   "paired_gemm": paired_gemm, "paired_pipeline": paired_chain}
        for _ in range(3): targets[args.ncu_target]()
        torch.cuda.synchronize()
        profile_once(torch, runtime, args.ncu_target, targets[args.ncu_target])
        emit("profiled_once", {"M": args.m, "target": args.ncu_target,
                               "nvtx": f"SM120_SCHEDULER:{args.ncu_target}",
                               "call_count": 1, "inverse_in_target": args.ncu_target == "paired_pipeline"})
        emit("summary", {"pass": True, "ncu": True, "timings_emitted": False})
        return 0

    # All graph outputs and workspaces are already allocated. Inverse-only
    # reads xpair generated above; it never invokes GEMM in its timed window.
    flush = torch.empty(max(2 * l2 + 1, 256 << 20), dtype=torch.uint8, device="cuda")
    calls = {"old_pipeline": old_chain, "new_official_gemm": official_gemm,
             "new_official_pipeline": official_chain,
             "paired_gemm": paired_gemm, "inverse_only": inverse,
             "quant_only": quant_only,
             "paired_pipeline": paired_chain}
    timed = {name: old.capture(torch, call) for name, call in calls.items()}
    for graph in timed.values(): graph.replay()
    torch.cuda.synchronize()
    capture_checks = [old.bit_comparison(torch, q, qr, f"{name}_capture_FP8", torch.uint8)
                      for name, q in (("old", qb), ("new_official", qoff), ("paired", qpair))]
    capture_checks += [old.bit_comparison(torch, sc, sr, f"{name}_capture_scale", torch.int32)
                       for name, sc in (("old", sb), ("new_official", soff), ("paired", spair))]
    if not all(c["pass"] for c in capture_checks):
        raise RuntimeError("diagnostic timing graph output failed fresh fixed-vLLM reference")
    times = measure_graphs(torch, timed, flush, args.order_seed, args.repeats)
    emit("diagnostic_timing", {"M": args.m, "raster": args.raster,
                               "swizzle": args.swizzle, "input_sha256": input_hash,
                               "input_seed": args.seed + 100 + args.m,
                               "pipeline_names": ["old_pipeline", "new_official_pipeline", "paired_pipeline"],
                               "gemm_only_names": ["new_official_gemm", "paired_gemm"],
                               "inverse_only_name": "inverse_only",
                               "quant_only_name": "quant_only",
                               "quant_only_scope": "precomputed canonical BF16 input; excludes GEMM and inverse",
                               "capture_checks": capture_checks,
                               "full_pipeline_names_not_interchangeable": True,
                               **times})
    emit("summary", {"pass": True, "ncu": False, "timings_emitted": True})
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-probe", type=Path, required=True)
    p.add_argument("--old-probe-sha256", default="")
    p.add_argument("--baseline-library", type=Path, required=True)
    p.add_argument("--baseline-sha256", default="")
    p.add_argument("--quant-library", type=Path, required=True)
    p.add_argument("--quant-sha256", default="")
    p.add_argument("--scheduler-library", type=Path, required=True)
    p.add_argument("--scheduler-sha256", default="")
    p.add_argument("--m", type=int, choices=(257, 2741, 2786, 16160, 32768), default=16160)
    p.add_argument("--raster", choices=tuple(RASTERS), default="heuristic")
    p.add_argument("--swizzle", type=int, choices=(1, 8), default=8)
    p.add_argument("--workspace-mib", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--order-seed", type=int, default=20261015)
    p.add_argument("--repeats", type=int, default=31)
    p.add_argument("--ncu-target", choices=("old_pipeline", "new_official_gemm",
                                            "new_official_pipeline", "paired_gemm", "paired_pipeline"))
    args = p.parse_args()
    if args.workspace_mib < 1 or args.repeats < 31:
        p.error("workspace must be positive and repeats >=31")
    try:
        return run(args)
    except Exception as exc:
        traceback.print_exc()
        emit("summary", {"pass": False, "timings_valid": False,
                         "error_type": type(exc).__name__, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
