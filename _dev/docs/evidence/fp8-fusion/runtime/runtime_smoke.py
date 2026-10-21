#!/usr/bin/env python3
"""Isolated GPU smoke for the opt-in FP8 activation-quant replacement.

Run only after micro_probe's exact-byte gate passes. This verifies the
mutating custom op under auto_functionalized, then compiles non-target M=5
before candidate M=256 to catch a shape-guard miss. Legal excluded layouts,
dtypes and shapes are checked against the official op with a temporary Python
call counter. CUDA Graph capture/replay is validated by micro_probe.py; this
smoke alone does not establish whole-model quality or replay execution.
"""

import argparse
import hashlib
import importlib
import json
import os
import runpy
import sys
import traceback
from pathlib import Path
from unittest.mock import patch


K = 17408
G = K // 128


def run(args):
    target = "vllm.compilation.passes.fusion.act_quant_fusion"
    if target in sys.modules:
        raise RuntimeError("fusion pass imported before sitecustomize")
    os.environ["PRO6000D_FP8_FUSION_ENABLE"] = "1"
    os.environ["PRO6000D_FP8_FUSION_LIBRARY"] = str(args.library.resolve())
    os.environ["PRO6000D_FP8_FUSION_LIBRARY_SHA256"] = hashlib.sha256(
        args.library.read_bytes()).hexdigest()
    hook_present = any(type(item).__name__ == "_FusionImportHook"
                       for item in sys.meta_path)
    if not hook_present:
        runpy.run_path(str(args.instrument_dir / "sitecustomize.py"))

    import torch
    from torch._higher_order_ops.auto_functionalize import auto_functionalized
    from vllm import _custom_ops as _ops  # noqa: F401

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("reserved SM120 GPU required")
    fusion = importlib.import_module(target)
    op = fusion.FUSED_OPS[fusion.kFp8Dynamic128Sym]
    if "pro6000d_fp8.silu_and_mul_per_block_quant" not in str(op):
        raise RuntimeError("FP8 fusion import hook did not install custom op")

    def make_input(m, seed):
        torch.manual_seed(seed)
        return torch.empty((m, 2 * K), dtype=torch.bfloat16,
                           device="cuda").normal_(0.0, 0.5)

    def allocate(m):
        q = torch.empty((m, K), dtype=torch.float8_e4m3fn, device="cuda")
        s = torch.empty((G, m), dtype=torch.float32, device="cuda").t()
        return q, s

    def original(x):
        q, s = allocate(x.shape[0])
        torch.ops._C.silu_and_mul_per_block_quant(q, x, s, 128, None, True)
        return q, s

    def forward(x):
        q, s = allocate(x.shape[0])
        value = auto_functionalized(
            op, out=q, input=x, scales=s, group_size=128,
            scale_ub=None, is_scale_transposed=True)
        return value[1], value[2]

    def compare(actual, expected):
        q0, s0 = actual
        q1, s1 = expected
        return {"fp8_byte_mismatches": int((q0.view(torch.uint8) !=
                                             q1.view(torch.uint8)).sum().item()),
                "scale_bit_mismatches": int((s0.contiguous().view(torch.int32) !=
                                             s1.contiguous().view(torch.int32)).sum().item())}

    inputs = {5: make_input(5, 5), 256: make_input(256, 256)}
    records = {"library_sha256": os.environ[
                   "PRO6000D_FP8_FUSION_LIBRARY_SHA256"], "checks": {}}
    for m in (5, 256):
        actual = forward(inputs[m])
        expected = original(inputs[m])
        mismatch = compare(actual, expected)
        records["checks"][f"auto_functionalized_m{m}"] = {
            **mismatch, "candidate_expected": m == 256,
            "pass": all(value == 0 for value in mismatch.values())}

    # These are all valid arguments for the fixed _C operator but fail one
    # candidate guard each. Patch only the Python op packet while making the
    # direct custom-op call, then restore it before the official reference.
    # A fallback case must call the official packet exactly once; target M256
    # must not call it. This is independent of coincidentally equal output.
    def fallback_case(name, m=256, hidden=K, dtype=torch.bfloat16,
                      group=128, transposed=True, with_scale_ub=False,
                      official_calls_expected=1):
        torch.manual_seed(3000 + len(records["checks"]))
        x = torch.empty((m, 2 * hidden), dtype=dtype,
                        device="cuda").normal_(0.0, 0.5)
        q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device="cuda")
        group_count = hidden // group
        if transposed:
            s = torch.empty((group_count, m), dtype=torch.float32,
                            device="cuda").t()
            assert s.stride() == (1, m)
        else:
            s = torch.empty((m, group_count), dtype=torch.float32,
                            device="cuda")
            assert s.is_contiguous()
        scale_ub = (torch.tensor([1.0], dtype=torch.float32, device="cuda")
                    if with_scale_ub else None)

        official_packet = torch.ops._C.silu_and_mul_per_block_quant
        seen = []

        def counted_official(*call_args, **call_kwargs):
            seen.append(True)
            return official_packet(*call_args, **call_kwargs)

        with patch.object(torch.ops._C, "silu_and_mul_per_block_quant",
                          counted_official):
            op(q, x, s, group, scale_ub, transposed)
        q_ref = torch.empty_like(q)
        if transposed:
            s_ref = torch.empty((group_count, m), dtype=torch.float32,
                                device="cuda").t()
        else:
            s_ref = torch.empty_like(s)
        official_packet(q_ref, x, s_ref, group, scale_ub, transposed)
        torch.cuda.synchronize()
        mismatch = compare((q, s), (q_ref, s_ref))
        calls_ok = len(seen) == official_calls_expected
        records["checks"][f"guard_{name}"] = {
            "M": m, "hidden": hidden, "input_dtype": str(dtype),
            "group_size": group, "scale_stride": list(s.stride()),
            "is_scale_transposed": transposed,
            "scale_ub_present": with_scale_ub,
            "observed_official_calls_inside_custom_op": len(seen),
            "expected_official_calls_inside_custom_op": official_calls_expected,
            **mismatch,
            "pass": calls_ok and all(value == 0 for value in mismatch.values()),
        }

    fallback_case("target_m256", official_calls_expected=0)
    fallback_case("small_m5", m=5)
    fallback_case("fp16_input", dtype=torch.float16)
    fallback_case("hidden_4096", hidden=4096)
    fallback_case("row_major_scales", transposed=False)
    fallback_case("group64", group=64)
    fallback_case("scale_ub", with_scale_ub=True)

    graph_targets = []

    def inspect_backend(gm, _example_inputs):
        graph_targets.append(str(gm.graph))
        return gm.forward

    inspected = torch.compile(forward, backend=inspect_backend,
                              fullgraph=True, dynamic=True)
    for m in (5, 256):
        actual = inspected(inputs[m])
        mismatch = compare(actual, original(inputs[m]))
        records["checks"][f"inspected_compile_m{m}"] = {
            **mismatch, "pass": all(value == 0 for value in mismatch.values())}
    graph_counts = [graph.count("pro6000d_fp8.silu_and_mul_per_block_quant")
                    for graph in graph_targets]
    records["checks"]["opaque_op_in_every_captured_graph"] = {
        "compile_order_M": [5, 256], "captured_graph_count": len(graph_counts),
        "custom_op_mentions_per_graph": graph_counts,
        "pass": bool(graph_counts) and all(count >= 1 for count in graph_counts),
    }

    compiled = torch.compile(forward, fullgraph=True, dynamic=True)
    for m in (5, 256):
        actual = compiled(inputs[m])
        mismatch = compare(actual, original(inputs[m]))
        records["checks"][f"inductor_compile_m{m}"] = {
            **mismatch, "pass": all(value == 0 for value in mismatch.values())}
    records["pass"] = all(item["pass"] for item in records["checks"].values())
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--instrument-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        traceback.print_exc()
        result = {"pass": False, "error_type": type(exc).__name__,
                  "error": str(exc)}
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
