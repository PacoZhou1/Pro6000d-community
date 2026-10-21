#!/usr/bin/env python3
"""Isolated SM120 block-FP8 MLP GEMM correctness and latency probe.

Run only after the profiling service is stopped and GPU memory is available.
This is an operator experiment on synthetic tensors, not a model-quality test.

The two call sites below reproduce the fixed vLLM 0.31.0 implementations of
CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm and
B12xFp8BlockScaledMMKernel.apply_block_scaled_mm. Quantization and reference
work are outside the timed region. B12X is optional for --backend cutlass.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import time

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)


SHAPES = ((5120, 34816), (17408, 5120))  # (K, N), Qwen3.6-27B dense MLP
TOKEN_COUNTS = (1, 5, 16)
GROUP = 128
FP8_MAX = 448.0  # torch.float8_e4m3fn finite maximum


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def prepare_weight(k: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one [N,K] e4m3 weight and [N/128,K/128] FP32 scales."""
    torch.manual_seed(seed)
    # Normalized random weights avoid a huge arbitrary output range. This
    # distribution is synthetic; no checkpoint values are read or stored.
    source = torch.randn((n, k), device="cuda", dtype=torch.float32) / math.sqrt(k)
    blocks = source.view(n // GROUP, GROUP, k // GROUP, GROUP)
    scale = blocks.abs().amax(dim=(1, 3)).div(FP8_MAX).clamp_min(1e-7).contiguous()
    qweight = (
        (blocks / scale[:, None, :, None])
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .reshape(n, k)
        .contiguous()
    )
    del source, blocks
    return qweight, scale


def prepare_activation(m: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    # Matches the fixed CUTLASS kernel's dynamic (1,128) quantization and
    # column-major scale layout. B12X receives the same values, contiguous.
    # QuantFP8.forward_cuda delegates to this exact helper for dynamic
    # per-token-group scales. Calling it directly avoids CustomOp's unrelated
    # requirement for a serving VllmConfig in a standalone probe.
    a, a_scale = per_token_group_quant_fp8(
        x,
        group_size=GROUP,
        dtype=torch.float8_e4m3fn,
        use_ue8m0=False,
        column_major_scales=True,
    )
    assert a.dtype == torch.float8_e4m3fn and a.shape == (m, k)
    assert a_scale.shape == (m, k // GROUP)
    return a.contiguous(), a_scale


def dequant_reference(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    columns: int,
) -> torch.Tensor:
    """FP32 matmul of *quantized* operands for the first N columns only.

    It measures GEMM arithmetic/output error, not FP8 quantization error
    relative to the original BF16 activation or unquantized random weights.
    """
    m, k = a.shape
    a32 = a.float().view(m, k // GROUP, GROUP)
    a32 = (a32 * a_scale.float().contiguous()[:, :, None]).reshape(m, k)
    w32 = weight[:columns].float().view(columns, k // GROUP, GROUP)
    scales = weight_scale[: columns // GROUP].float()
    w32 = (
        w32.view(columns // GROUP, GROUP, k // GROUP, GROUP)
        * scales[:, None, :, None]
    ).reshape(columns, k)
    return a32 @ w32.T


def metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    x, y = output.float(), reference.float()
    finite = bool(torch.isfinite(x).all().item())
    if not finite:
        return {"finite": False, "nrmse": math.inf, "cosine": float("nan"),
                "max_abs": math.inf, "max_abs_over_ref_rms": math.inf}
    diff = x - y
    ref_rms = y.square().mean().sqrt().clamp_min(1e-12)
    nrmse = diff.square().mean().sqrt() / ref_rms
    cosine = torch.nn.functional.cosine_similarity(
        x.reshape(1, -1), y.reshape(1, -1), dim=1
    )[0]
    max_abs = diff.abs().max()
    return {
        "finite": finite,
        "nrmse": float(nrmse.item()),
        "cosine": float(cosine.item()),
        "max_abs": float(max_abs.item()),
        "max_abs_over_ref_rms": float((max_abs / ref_rms).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cutlass", "b12x", "both"), default="both")
    parser.add_argument("--ref-cols", type=int, default=512,
                        help="First N columns for FP32 reference; multiple of 128")
    parser.add_argument("--warmup-calls", type=int, default=10)
    parser.add_argument("--measure-ms", type=int, default=20,
                        help="Approximate time represented by one captured graph")
    parser.add_argument("--nrmse-limit", type=float, default=0.02)
    parser.add_argument("--cosine-floor", type=float, default=0.999)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        parser.error("requires one free CUDA SM120 GPU")
    if args.ref_cols <= 0 or args.ref_cols % GROUP:
        parser.error("--ref-cols must be a positive multiple of 128")
    if args.ref_cols > min(n for _, n in SHAPES):
        parser.error("--ref-cols exceeds the smaller output width")
    if args.warmup_calls < 1 or args.measure_ms < 1:
        parser.error("warmup calls and measure milliseconds must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # CUTLASS_BLOCK_FP8_SUPPORTED is resolved from the installed vLLM build.
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
        CUTLASS_BLOCK_FP8_SUPPORTED,
    )
    if args.backend in ("cutlass", "both") and not CUTLASS_BLOCK_FP8_SUPPORTED:
        parser.error("this vLLM build does not support CUTLASS block FP8")

    blockscaled = None
    if args.backend in ("b12x", "both"):
        try:
            from b12x.gemm import blockscaled as blockscaled_module
            blockscaled = blockscaled_module
        except ImportError as exc:
            parser.error(f"B12X unavailable: {exc}")
        if not blockscaled.is_supported():
            parser.error("B12X blockscaled.is_supported() returned false")

    # This helper captures one GEMM into a CUDA graph and times graph replays,
    # excluding Python dispatch and compile/warmup. Same timing method for both.
    from triton.testing import do_bench_cudagraph

    meta = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "vllm": version("vllm"),
        "b12x": version("b12x"),
        "triton": version("triton"),
        "python": platform.python_version(),
        "out_dtype": "bfloat16",
        "fp8_dtype": "float8_e4m3fn",
        "activation_scale_shape": "[M,K/128] (1,128)",
        "weight_scale_shape": "[N/128,K/128] (128,128)",
        "reference": f"FP32 GEMM on dequantized FP8 operands, first {args.ref_cols} outputs",
        "timing": "Triton do_bench_cudagraph, GEMM only; excludes quantization and reference",
        "timing_limit": "repeated same weights may benefit from L2; not whole-model throughput",
        "reference_tf32": False,
        "warmup_calls": args.warmup_calls,
        "measure_ms": args.measure_ms,
        "quality_scope": "synthetic operator correctness only; no model weights or model-quality claim",
        "thresholds": {"nrmse_limit": args.nrmse_limit,
                       "cosine_floor": args.cosine_floor},
        "timestamp_unix": time.time(),
    }
    print(json.dumps({"metadata": meta}, ensure_ascii=False), flush=True)
    failed = False
    for shape_id, (k, n) in enumerate(SHAPES):
        weight, ws = prepare_weight(k, n, seed=20261014 + shape_id)
        for m in TOKEN_COUNTS:
            a, asc = prepare_activation(m, k, seed=1000 + shape_id * 100 + m)
            asa = asc.contiguous()  # B12X's row-major view, same numeric scales
            reference = dequant_reference(a, asc, weight, ws, args.ref_cols)
            functions = {}
            if args.backend in ("cutlass", "both"):
                # Exact tensor order and transposes in the fixed vLLM 0.31.0
                # CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm.
                functions["cutlass"] = lambda: ops.cutlass_scaled_mm(
                    a, weight.T, out_dtype=torch.bfloat16,
                    scale_a=asc, scale_b=ws.T,
                )
            if args.backend in ("b12x", "both"):
                # Exact order in B12xFp8BlockScaledMMKernel's helper.
                functions["b12x"] = lambda: blockscaled.mm_block_fp8(
                    a, asa, weight, ws, out_dtype=torch.bfloat16,
                )
            outputs = {}
            for name, fn in functions.items():
                out = fn()
                torch.cuda.synchronize()
                scores = metrics(out[:, : args.ref_cols], reference)
                scores["all_output_finite"] = bool(torch.isfinite(out).all().item())
                passed = bool(scores["all_output_finite"] and scores["finite"] and scores["nrmse"] <= args.nrmse_limit
                              and scores["cosine"] >= args.cosine_floor)
                failed |= not passed
                # Output remains alive for the A/B comparison; warmup and graph
                # capture happen only after correctness has been checked.
                outputs[name] = out
                for _ in range(args.warmup_calls):
                    fn()
                torch.cuda.synchronize()
                latency_ms = do_bench_cudagraph(
                    fn, rep=args.measure_ms, return_mode="median"
                )
                print(json.dumps({"K": k, "N": n, "M": m, "backend": name,
                                  "latency_us": latency_ms * 1000.0,
                                  "ref_cols": args.ref_cols, "metrics": scores,
                                  "pass": passed}, ensure_ascii=False), flush=True)
            if len(outputs) == 2:
                pair = metrics(outputs["b12x"][:, : args.ref_cols],
                               outputs["cutlass"][:, : args.ref_cols].float())
                print(json.dumps({"K": k, "N": n, "M": m,
                                  "comparison": "b12x_vs_cutlass_first_ref_cols",
                                  "metrics": pair}, ensure_ascii=False), flush=True)
            del a, asc, asa, reference, functions, outputs
        del weight, ws
        torch.cuda.empty_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
