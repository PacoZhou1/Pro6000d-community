#!/usr/bin/env python3
"""GPU smoke for isolated official-M16160 runtime; root schedules execution.

Requires this directory first on PYTHONPATH, enabled V2 sitecustomize, and
PRO6000D_OFFICIAL_MODE=candidate. Does not start or stop any service.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def capture(torch, call):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        call()
    torch.cuda.synchronize()
    return graph


def bits_equal(torch, actual, expected, integer_dtype):
    """Compare FP8 bytes or FP32 scale bits, including NaN and signed zero."""
    return bool(torch.equal(actual.contiguous().view(integer_dtype),
                            expected.contiguous().view(integer_dtype)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PRO6000D_OFFICIAL_MODE") != "candidate":
        raise RuntimeError("official-tuning runtime smoke requires candidate mode")
    library_sha = os.environ.get("PRO6000D_OFFICIAL_LIBRARY_SHA256", "")
    if not library_sha or library_sha != os.environ.get(
        "PRO6000D_OFFICIAL_VALIDATED_LIBRARY_SHA256", ""
    ):
        raise RuntimeError("official-tuning smoke requires validated library SHA256")

    import torch
    import vllm.compilation.passes.fusion.act_quant_fusion  # noqa: F401
    from candidate_abi import CandidateABI
    from candidate_op import last_dispatch_for_eager_smoke, register
    from packing import lookup_packed, prepare_model

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("runtime smoke requires SM120")
    abi = CandidateABI(os.environ["PRO6000D_OFFICIAL_LIBRARY"], library_sha)
    register(abi)
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
    prepare_model(model, str(args.output.parent / "startup-pack-smoke"))
    w = model.language_model.layers[0].mlp.gate_up_proj.weight
    ws = model.language_model.layers[0].mlp.gate_up_proj.weight_scale_inv
    entry = lookup_packed(w, ws)
    if entry is None or entry["packed_weight"].data_ptr() == w.data_ptr():
        raise RuntimeError("startup paired-weight registry failed")

    def input_for(m, seed):
        torch.manual_seed(seed)
        a = torch.randn((m, 5120), device=device).to(torch.float8_e4m3fn)
        asc = (0.01 + torch.rand((40, m), device=device)).float().t()
        return a, asc

    def reference(a, asc, b, b_scale):
        m = a.shape[0]
        d = torch.empty((m, 34816), dtype=torch.bfloat16, device=device)
        q = torch.empty((m, 17408), dtype=torch.float8_e4m3fn, device=device)
        sc = torch.empty((136, m), dtype=torch.float32, device=device).t()
        torch.ops._C.cutlass_scaled_mm(d, a, b.t(), asc, b_scale.t(), None)
        torch.ops.pro6000d_fp8.silu_and_mul_per_block_quant(q, d, sc, 128, None, True)
        return q, sc

    cases = []
    for label, m, b, b_scale, expected in (
        ("target", 16160, w, ws, "official_along_m_s1_m16160"),
        ("small_fallback", 5, w, ws, "canonical_v2_fallback"),
        ("other_prefill_fallback", 2741, w, ws, "canonical_v2_fallback"),
        ("unregistered_target_fallback", 16160, w.clone(), ws.clone(),
         "canonical_v2_fallback"),
    ):
        a, asc = input_for(m, 20261014 + m)
        qref, sref = reference(a, asc, b, b_scale)
        q, sc = torch.ops.pro6000d_official_gateup.tuned(a, b, asc, b_scale)
        torch.cuda.synchronize()
        dispatch = last_dispatch_for_eager_smoke()
        result = {"case": label, "M": m, "dispatch": dispatch,
                  "dispatch_expected": expected,
                  "out_bits_equal": bits_equal(torch, q, qref, torch.uint8),
                  "scale_bits_equal": bits_equal(torch, sc, sref, torch.int32),
                  "out_shape": list(q.shape), "scale_shape": list(sc.shape),
                  "scale_stride": list(sc.stride())}
        cases.append(result)
        del a, asc, qref, sref, q, sc
        torch.cuda.empty_cache()

    # Changed-input graph proof for the actual target branch. The Python
    # dispatcher is called at capture time; replay must update the captured
    # output buffers from new A/As and agree with a fresh official reference.
    a, asc = input_for(16160, 888801)
    output = {}
    def run_target():
        output["q"], output["sc"] = torch.ops.pro6000d_official_gateup.tuned(
            a, w, asc, ws)
    graph = capture(torch, run_target)
    before = output["q"].view(torch.uint8).clone()
    changed_a, changed_asc = input_for(16160, 888802)
    input_changed = not bool(torch.equal(a.view(torch.uint8), changed_a.view(torch.uint8)))
    a.copy_(changed_a)
    asc.copy_(changed_asc)
    graph.replay()
    fresh_q, fresh_s = reference(a, asc, w, ws)
    torch.cuda.synchronize()
    graph_result = {
        "M": 16160, "capture_dispatch": last_dispatch_for_eager_smoke(),
        "input_changed": input_changed,
        "output_changed": not bool(torch.equal(before, output["q"].view(torch.uint8))),
        "out_bits_equal_fresh": bits_equal(torch, output["q"], fresh_q, torch.uint8),
        "scale_bits_equal_fresh": bits_equal(torch, output["sc"], fresh_s, torch.int32),
    }
    graph_result["pass"] = (
        graph_result["capture_dispatch"] == "official_along_m_s1_m16160"
        and graph_result["input_changed"] and graph_result["output_changed"]
        and graph_result["out_bits_equal_fresh"] and graph_result["scale_bits_equal_fresh"]
    )
    report = {
        "scope": "synthetic isolated runtime direct-op and changed-input graph; no model/API claim",
        "library_sha256": library_sha, "abi_version": 1,
        "raster": "AlongM", "swizzle": 1, "target_M": 16160,
        "startup_registered": True, "cases": cases, "changed_input_graph": graph_result,
        "passed": all(x["dispatch"] == x["dispatch_expected"]
                      and x["out_bits_equal"] and x["scale_bits_equal"]
                      and x["out_shape"] == [x["M"], 17408]
                      and x["scale_shape"] == [x["M"], 136]
                      and x["scale_stride"] == [1, x["M"]] for x in cases)
                  and graph_result["pass"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
