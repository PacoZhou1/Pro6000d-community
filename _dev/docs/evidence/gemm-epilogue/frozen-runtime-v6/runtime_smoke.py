#!/usr/bin/env python3
"""Isolated synthetic CUDA smoke for functional op and exact-registry fallback.

Run only after the kernel's independent numeric gate. This program does not
start/stop services or use network. It needs the fixed image, V2 runtime and
explicit validated candidate SHA environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, default=257)
    args = parser.parse_args()
    if os.environ.get("PRO6000D_EPILOGUE_MODE") != "candidate":
        raise RuntimeError("runtime smoke requires candidate opt-in mode")
    library_sha = os.environ.get("PRO6000D_EPILOGUE_LIBRARY_SHA256", "")
    if library_sha != os.environ.get("PRO6000D_EPILOGUE_VALIDATED_LIBRARY_SHA256"):
        raise RuntimeError("runtime smoke requires validated library SHA256")

    import torch
    import vllm.compilation.passes.fusion.act_quant_fusion  # noqa: F401

    from candidate_abi import CandidateABI
    from candidate_op import last_dispatch_for_eager_smoke, register
    from packing import lookup_packed, prepare_model

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("runtime smoke requires SM120")
    swizzle = int(os.environ.get("PRO6000D_EPILOGUE_SWIZZLE", "8"))
    abi = CandidateABI(os.environ["PRO6000D_EPILOGUE_LIBRARY"], library_sha)
    register(abi, swizzle)
    torch.manual_seed(23062026)
    device = "cuda"

    class CutlassFp8BlockScaledMMKernel:
        pass

    class Fp8LinearMethod:
        def __init__(self):
            self.fp8_linear = CutlassFp8BlockScaledMMKernel()

    class MergedColumnParallelLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = Fp8LinearMethod()
            self.weight = torch.nn.Parameter(
                torch.randn((34816, 5120), device=device).to(torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.register_buffer("weight_scale_inv", (
                0.01 + torch.rand((272, 40), device=device)
            ).float())

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.language_model.layers[0].mlp = torch.nn.Module()
            self.language_model.layers[0].mlp.gate_up_proj = MergedColumnParallelLinear()

    model = Root()
    evidence_dir = str(args.output.parent / "startup-pack-smoke")
    prepare_model(model, evidence_dir)
    w = model.language_model.layers[0].mlp.gate_up_proj.weight
    ws = model.language_model.layers[0].mlp.gate_up_proj.weight_scale_inv
    if lookup_packed(w, ws) is None:
        raise RuntimeError("startup registry lookup failed")

    def input_for(m):
        a = torch.randn((m, 5120), device=device).to(torch.float8_e4m3fn)
        a_scale = (0.01 + torch.rand((40, m), device=device)).float().t()
        return a, a_scale

    def reference(a, a_scale, b, b_scale):
        m = a.shape[0]
        d = torch.empty((m, 34816), dtype=torch.bfloat16, device=device)
        q = torch.empty((m, 17408), dtype=torch.float8_e4m3fn, device=device)
        s = torch.empty((136, m), dtype=torch.float32, device=device).t()
        torch.ops._C.cutlass_scaled_mm(d, a, b.t(), a_scale, b_scale.t(), None)
        torch.ops.pro6000d_fp8.silu_and_mul_per_block_quant(
            q, d, s, 128, None, True
        )
        return q, s

    cases = []
    for label, m, b, b_scale, expected in (
        ("candidate", args.m, w, ws, "candidate_abi_v2"),
        ("unregistered_fallback", args.m, w.clone(), ws.clone(),
         "canonical_v2_fallback"),
        ("small_m_fallback", 5, w, ws, "canonical_v2_fallback"),
    ):
        a, a_scale = input_for(m)
        q_ref, s_ref = reference(a, a_scale, b, b_scale)
        q, s = torch.ops.pro6000d_gateup.fused(a, b, a_scale, b_scale)
        torch.cuda.synchronize()
        dispatch = last_dispatch_for_eager_smoke()
        result = {
            "case": label, "M": m, "dispatch": dispatch,
            "dispatch_expected": expected,
            "out_equal": bool(torch.equal(q, q_ref)),
            "scale_equal": bool(torch.equal(s, s_ref)),
            "out_shape": list(q.shape), "scale_shape": list(s.shape),
            "scale_stride": list(s.stride()),
        }
        cases.append(result)
    report = {
        "scope": "synthetic isolated direct-op smoke; no model/API benchmark",
        "library_sha256": library_sha,
        "abi_version": 2,
        "cases": cases,
        "passed": all(
            x["dispatch"] == x["dispatch_expected"]
            and x["out_equal"] and x["scale_equal"]
            and x["scale_stride"] == [1, x["M"]]
            for x in cases
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
