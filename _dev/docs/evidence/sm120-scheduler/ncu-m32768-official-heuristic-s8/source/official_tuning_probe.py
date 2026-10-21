#!/usr/bin/env python3
"""Isolated same-input official M128N128 scheduler tuning probe (GPU by root).

Three complete chains only: fixed old H8, new official H8, and new official
AlongM/s1. No legacy v6 binary is loaded or imported. Every M is released
before the next to avoid retaining large graph/output buffers.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import traceback
from pathlib import Path

from scheduler_probe import (GROUP, H, K, N, RASTERS, SchedulerLibrary,
                             check_sha, cuda_versions, emit, import_file,
                             input_fingerprint, measure_graphs, sha)

ARMS = ("old_h8", "new_h8", "new_along_m_s1")
SHAPES = (2741, 2786, 16160, 32768)


def run(args):
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("reserved SM120 GPU required")
    torch.backends.cuda.matmul.allow_tf32 = False
    old = import_file(args.old_probe, "fixed_pipeline_for_official_tuning")
    hashes = {
        "old_probe": check_sha(args.old_probe, args.old_probe_sha256),
        "baseline": check_sha(args.baseline_library, args.baseline_sha256),
        "quant": check_sha(args.quant_library, args.quant_sha256),
        "scheduler": check_sha(args.scheduler_library, args.scheduler_sha256),
        "probe": sha(Path(__file__)),
    }
    scheduler = SchedulerLibrary(torch, args.scheduler_library)
    runtime = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
    l2 = old.l2_size(runtime, torch.cuda.current_device())
    b, bs = old.make_weight(torch, N, K, args.seed)
    torch.cuda.synchronize()
    emit("metadata", {
        "scope": "synthetic official M128N128 GEMM + unchanged V2 quant; no model/API claim",
        "shapes": list(args.shapes), "arms": list(ARMS), "seed": args.seed,
        "order_seed": args.order_seed, "repeats": args.repeats,
        "sha256": hashes, "official_ops_sha256": sha(Path(ops.__file__)),
        "scheduler_abi": 1, "scheduler_metadata": scheduler.metadata,
        "raster_swizzle": {"old_h8": [0, 8], "new_h8": [0, 8],
                           "new_along_m_s1": [RASTERS["along_m"], 1]},
        "input_seed_rule": "seed+100+M", "device": torch.cuda.get_device_name(),
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cuda_versions": cuda_versions(runtime), "l2_bytes": l2,
        "cache_disturbed_scope": "owned >2x L2 flush outside event; not fixed cold L2",
    })

    for m in args.shapes:
        a, asc = old.make_activation(torch, per_token_group_quant_fp8,
                                     m, args.seed + 100 + m)
        input_hash = input_fingerprint(torch, a, asc)
        base = old.Baseline(torch, args.baseline_library, args.quant_library,
                            args.workspace_mib << 20, 8)
        states = {"new_h8": scheduler.prepare("official_bf16", m, 0, 8),
                  "new_along_m_s1": scheduler.prepare("official_bf16", m, 1, 1)}
        xs = {arm: torch.empty((m, N), dtype=torch.bfloat16, device="cuda") for arm in ARMS}
        qs = {}
        ss = {}
        for arm in ARMS:
            qs[arm], ss[arm] = old.output_buffers(torch, m)

        def make_call(arm):
            x, q, sc = xs[arm], qs[arm], ss[arm]
            if arm == "old_h8":
                def call():
                    base.gemm(x, a, asc, b, bs)
                    base.quant(x, q, sc)
            else:
                state = states[arm]
                def call():
                    scheduler.call(state, x, None, a, asc, b, bs)
                    base.quant(x, q, sc)
            return call

        calls = {arm: make_call(arm) for arm in ARMS}
        for call in calls.values():
            call()
        xr = ops.cutlass_scaled_mm(a, b.T, out_dtype=torch.bfloat16,
                                   scale_a=asc, scale_b=bs.T)
        qr, sr = old.output_buffers(torch, m)
        torch.ops._C.silu_and_mul_per_block_quant(qr, xr, sr, GROUP, None, True)
        torch.cuda.synchronize()
        checks = []
        for arm in ARMS:
            checks.extend((
                old.bit_comparison(torch, xs[arm], xr, f"{arm}_BF16", torch.int16),
                old.bit_comparison(torch, qs[arm], qr, f"{arm}_FP8", torch.uint8),
                old.bit_comparison(torch, ss[arm], sr, f"{arm}_scales", torch.int32),
            ))
        if not all(c["pass"] for c in checks):
            emit("numerics", {"M": m, "pass": False, "input_seed": args.seed + 100 + m,
                              "input_sha256": input_hash, "comparisons": checks})
            emit("summary", {"pass": False, "timings_emitted": False,
                             "reason": f"M={m} full-bit numerical gate failed"})
            return 1

        graphs = {arm: old.capture(torch, call) for arm, call in calls.items()}
        old_a = a.view(torch.uint8).clone()
        old_asc = asc.contiguous().view(torch.uint8).clone()
        prior = {arm: qs[arm].view(torch.uint8).clone() for arm in ARMS}
        changed_a, changed_asc = old.make_activation(
            torch, per_token_group_quant_fp8, m, args.seed + 9000 + m)
        a.copy_(changed_a)
        asc.copy_(changed_asc)
        input_changed = not bool(torch.equal(old_a, a.view(torch.uint8)))
        for graph in graphs.values():
            graph.replay()
        changed_x = ops.cutlass_scaled_mm(a, b.T, out_dtype=torch.bfloat16,
                                          scale_a=asc, scale_b=bs.T)
        changed_q, changed_s = old.output_buffers(torch, m)
        torch.ops._C.silu_and_mul_per_block_quant(
            changed_q, changed_x, changed_s, GROUP, None, True)
        torch.cuda.synchronize()
        replay_checks = []
        for arm in ARMS:
            comparisons = [
                old.bit_comparison(torch, xs[arm], changed_x, f"{arm}_replay_BF16", torch.int16),
                old.bit_comparison(torch, qs[arm], changed_q, f"{arm}_replay_FP8", torch.uint8),
                old.bit_comparison(torch, ss[arm], changed_s, f"{arm}_replay_scales", torch.int32),
            ]
            output_changed = not bool(torch.equal(prior[arm], qs[arm].view(torch.uint8)))
            replay_checks.append({"arm": arm, "input_changed": input_changed,
                                  "output_changed": output_changed,
                                  "pass": input_changed and output_changed
                                          and all(c["pass"] for c in comparisons),
                                  "comparisons": comparisons})
        passed = all(c["pass"] for c in replay_checks)
        emit("numerics", {"M": m, "pass": passed, "input_seed": args.seed + 100 + m,
                          "input_sha256": input_hash, "comparisons": checks,
                          "changed_input_graph": replay_checks})
        if not passed:
            emit("summary", {"pass": False, "timings_emitted": False,
                             "reason": f"M={m} changed-input graph gate failed"})
            return 1
        a.copy_(old_a.view(torch.float8_e4m3fn))
        asc.copy_(old_asc.view(torch.float32).reshape(m, K // GROUP))
        for call in calls.values():
            call()
        del graphs, graph, changed_a, changed_asc, changed_x, changed_q, changed_s, old_a, old_asc, prior

        flush = torch.empty(max(2 * l2 + 1, 256 << 20), dtype=torch.uint8,
                            device="cuda")
        timed_graphs = {arm: old.capture(torch, call) for arm, call in calls.items()}
        for graph in timed_graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        capture_checks = []
        for arm in ARMS:
            capture_checks.extend((
                old.bit_comparison(torch, xs[arm], xr, f"{arm}_capture_BF16", torch.int16),
                old.bit_comparison(torch, qs[arm], qr, f"{arm}_capture_FP8", torch.uint8),
                old.bit_comparison(torch, ss[arm], sr, f"{arm}_capture_scales", torch.int32),
            ))
        if not all(c["pass"] for c in capture_checks):
            raise RuntimeError(f"M={m} timing graph failed fresh official reference")
        timing = measure_graphs(torch, timed_graphs, flush, args.order_seed + m,
                                args.repeats)
        emit("timing", {"M": m, "pass": True, "full_pipeline": True,
                        "input_seed": args.seed + 100 + m, "input_sha256": input_hash,
                        "capture_checks": capture_checks, **timing})
        del (a, asc, base, states, xs, qs, ss, calls, call, graph, xr, qr, sr, timed_graphs,
             capture_checks, flush, timing)
        torch.cuda.empty_cache()
    emit("summary", {"pass": True, "numeric_cases": len(args.shapes),
                     "timed_shapes": list(args.shapes), "timings_emitted": True})
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-probe", type=Path, required=True)
    p.add_argument("--old-probe-sha256", required=True)
    p.add_argument("--baseline-library", type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--quant-library", type=Path, required=True)
    p.add_argument("--quant-sha256", required=True)
    p.add_argument("--scheduler-library", type=Path, required=True)
    p.add_argument("--scheduler-sha256", required=True)
    p.add_argument("--shapes", nargs="+", type=int, default=SHAPES)
    p.add_argument("--workspace-mib", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--order-seed", type=int, default=20261015)
    p.add_argument("--repeats", type=int, default=31)
    args = p.parse_args()
    if (args.repeats < 31 or args.workspace_mib < 1 or
        not args.shapes or len(args.shapes) != len(set(args.shapes)) or
        any(m not in SHAPES for m in args.shapes) or
        any(len(x) != 64 for x in (args.old_probe_sha256, args.baseline_sha256,
                                  args.quant_sha256, args.scheduler_sha256))):
        p.error("repeats >=31, unique listed shapes and all SHA256 values are required")
    try:
        return run(args)
    except Exception as exc:
        traceback.print_exc()
        emit("summary", {"pass": False, "timings_emitted": False,
                         "error_type": type(exc).__name__, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
