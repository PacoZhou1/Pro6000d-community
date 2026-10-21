#!/usr/bin/env python3
"""Root-run CUDA poison check of the extracted F1 null-metadata output contract.

The original packed warmup is deliberately a no-state/no-CUDA stub. The fixed
official source is checked to contain only prewarm+return for null metadata.
This does not load a model, run real packed warmup, register the opaque op,
exercise its decorator/Graph integration or establish model E2E correctness.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA = "ce01159d779ea193f7a86235aae1a147258b64db9c172f7a6772bf51c3b3380c"
FIXED_SHA = "854043107312ee708f4107e35362d6096ce270cc33d4a50293d1f2026df1bbcc"
OFFICIAL_SHA = "3bd2f805f0f4b821e6902ef18da8ba66ad42a7e5a633b76abe96892532293d9c"
METHOD = "pro6000d_gdn_core_norm_quant"
SHAPES, NH, HD = (1, 256, 32768), 48, 128


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def unique_function(tree, name):
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(matches) == 1, f"expected one function named {name}")
    return matches[0]


def null_branch(function, name):
    matches = []
    for node in ast.walk(function):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == name
                and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Is)
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value is None):
            matches.append(node)
    require(len(matches) == 1, f"expected one {name} is None branch")
    return matches[0]


def source_contract(original, fixed, official):
    for path, expected in ((original, ORIGINAL_SHA), (fixed, FIXED_SHA), (official, OFFICIAL_SHA)):
        require(sha(path) == expected, f"source SHA mismatch: {path}")
    old = unique_function(ast.parse(original.read_text()), METHOD)
    new = unique_function(ast.parse(fixed.read_text()), METHOD)
    old_branch, new_branch = null_branch(old, "md"), null_branch(new, "md")
    require(len(old_branch.body) == 2 and len(new_branch.body) == 3, "unexpected null-MD branch bodies")
    stripped = copy.deepcopy(new)
    addition = null_branch(stripped, "md").body.pop(1)
    expected_addition = ast.parse(
        "torch.ops._C.per_token_group_fp8_quant(core_attn_out.flatten(-2), q, s, "
        "128, 1e-10, fp8_min, fp8_max, False, True, False)"
    ).body[0]
    require(ast.dump(addition) == ast.dump(expected_addition), "patch is not the expected official quant call")
    # All original guards/branches/math are retained; remove only the added call.
    require(ast.dump(old) == ast.dump(stripped), "original/fixed functions differ outside null-MD quant")
    packed = unique_function(ast.parse(official.read_text()), "_forward_core_fused_norm_packed")
    branch = null_branch(packed, "attn_metadata")
    require(len(branch.body) == 2 and isinstance(branch.body[0], ast.Expr)
            and isinstance(branch.body[0].value, ast.Call)
            and isinstance(branch.body[0].value.func, ast.Attribute)
            and branch.body[0].value.func.attr == "_warmup_prefill_kernels"
            and isinstance(branch.body[1], ast.Return) and branch.body[1].value is None,
            "official null-MD branch no longer only prewarms and returns")
    return old, new


def compile_body(function, namespace, filename):
    function = copy.deepcopy(function)
    # Bypass framework decorators/annotation evaluation, not runtime guards.
    function.decorator_list = []
    function.returns = None
    for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
        arg.annotation = None
    for arg in (function.args.vararg, function.args.kwarg):
        if arg is not None:
            arg.annotation = None
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(tree, filename, "exec"), namespace)
    return namespace[function.name]


def integer_comparison(torch, actual, expected, integer_dtype):
    x = actual.contiguous().view(integer_dtype)
    y = expected.contiguous().view(integer_dtype)
    different = x != y
    count = int(different.sum().item())
    result = {"elements": actual.numel(), "mismatches": count, "pass": count == 0}
    if count:
        index = int(different.reshape(-1).to(torch.int8).argmax().item())
        result["first_mismatch"] = {"flat_element_index": index,
                                     "actual_bits": int(x.reshape(-1)[index].item()),
                                     "reference_bits": int(y.reshape(-1)[index].item())}
    return result


def nonfinite_count(torch, tensor):
    # Bound temporary FP32 storage when checking the M32768 FP8 output.
    return sum(int((~torch.isfinite(tensor[i:i + 256].float())).sum().item())
               for i in range(0, tensor.shape[0], 256))


def run(args, old_ast, new_ast):
    import torch
    from vllm import _custom_ops as _ops  # noqa: F401: registers the actual CUDA op
    from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max

    require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
            "this fixed-image probe requires the reserved SM120 device")
    fp8_min, fp8_max = get_fp8_min_max()
    require(float(fp8_min) == -448 and float(fp8_max) == 448, "FP8 quantization bounds changed")
    records = []

    def one(m):
        core = torch.zeros((m, NH, HD), dtype=torch.bfloat16, device="cuda")
        # The stub does not read projections; zero-element placeholders avoid
        # pretending to test projections or real warmup with synthetic weights.
        mixed = torch.empty((0,), dtype=torch.bfloat16, device="cuda")
        ba = torch.empty_like(mixed)
        qr = torch.empty((m, NH * HD), dtype=torch.float8_e4m3fn, device="cuda")
        sr = torch.empty((NH, m), dtype=torch.float32, device="cuda").t()
        torch.ops._C.per_token_group_fp8_quant(core.flatten(-2), qr, sr, 128,
                                             1e-10, fp8_min, fp8_max, False, True, False)
        torch.cuda.synchronize()

        class StubAttention:
            prefix = "null_metadata_probe"

            def __init__(self):
                self.calls = 0

            def _forward_core_fused_norm_packed(self, *, mixed_qkvz, ba, core_attn_out):
                require(mixed_qkvz is mixed and ba is ba_input and core_attn_out is core,
                        "unexpected inputs to no-state packed warmup stub")
                self.calls += 1
                # Intentionally no state, tensor writes, prewarm kernels or CUDA.

        ba_input = ba
        stub = StubAttention()
        context = types.SimpleNamespace(no_compile_layers={stub.prefix: stub}, attn_metadata=None)
        module = types.SimpleNamespace(_resolve_layer_name=lambda name: name,
                                       get_forward_context=lambda: context)
        base_namespace = {"module": module, "torch": torch,
                          "fp8_min": fp8_min, "fp8_max": fp8_max}
        old = compile_body(old_ast, dict(base_namespace), str(args.original_site))
        new = compile_body(new_ast, dict(base_namespace), str(args.fixed_site))
        qo, so = torch.empty_like(qr), torch.empty((NH, m), dtype=torch.float32, device="cuda").t()
        qn, sn = torch.empty_like(qr), torch.empty((NH, m), dtype=torch.float32, device="cuda").t()
        # E4M3FN 0x7f is NaN; scale fill's exact FP32 bit pattern is captured.
        for q, s in ((qo, so), (qn, sn)):
            q.view(torch.uint8).fill_(0x7f)
            s.fill_(float("nan"))
        scale_poison_bits = int(so.view(torch.int32)[0, 0].item())
        old(mixed, ba, core, qo, so, stub.prefix)
        new(mixed, ba, core, qn, sn, stub.prefix)
        torch.cuda.synchronize()
        old_unchanged_q = bool((qo.view(torch.uint8) == 0x7f).all().item())
        old_unchanged_s = bool((so.view(torch.int32) == scale_poison_bits).all().item())
        old_q = integer_comparison(torch, qo, qr, torch.uint8)
        old_s = integer_comparison(torch, so, sr, torch.int32)
        new_q = integer_comparison(torch, qn, qr, torch.uint8)
        new_s = integer_comparison(torch, sn, sr, torch.int32)
        q_nonfinite, s_nonfinite = nonfinite_count(torch, qn), nonfinite_count(torch, sn)
        core_still_zero = bool((core == 0).all().item())
        stride_ok = so.stride() == sn.stride() == sr.stride() == (1, m)
        passed = (stub.calls == 2 and old_unchanged_q and old_unchanged_s
                  and old_q["mismatches"] == qo.numel() and old_s["mismatches"] == so.numel()
                  and new_q["pass"] and new_s["pass"] and q_nonfinite == s_nonfinite == 0
                  and core_still_zero and stride_ok)
        return {"M": m, "NH": NH, "HD": HD, "pass": passed,
                "packed_warmup_stub_calls": stub.calls,
                "old_poison_unchanged": {"q": old_unchanged_q, "s": old_unchanged_s},
                "old_vs_official": {"FP8_bytes": old_q, "scale_bits": old_s},
                "fixed_vs_official": {"FP8_bytes": new_q, "scale_bits": new_s},
                "fixed_nonfinite": {"q": q_nonfinite, "s": s_nonfinite},
                "scale_stride": list(sn.stride()), "core_still_zero": core_still_zero,
                "reference_zero_q": bool((qr.view(torch.uint8) == 0).all().item()),
                "reference_scale_first_bits": int(sr.view(torch.int32)[0, 0].item())}

    for m in SHAPES:
        row = one(m)
        records.append(row)
        print(json.dumps({"null_metadata_case": row}, sort_keys=True, allow_nan=False), flush=True)
    return {"schema": "null-metadata-output-contract-v1", "passed": all(r["pass"] for r in records),
            "scope": "extracted customop null-MD q/s output contract only; not model E2E, real warmup or Graph proof",
            "packed_warmup": "explicit no-state/no-CUDA stub; official branch AST is prewarm+return",
            "projection_inputs": "unused zero-element CUDA placeholders passed only to the stub",
            "guards_preserved": True, "framework_decorators_exercised": False,
            "model_loaded": False, "device": torch.cuda.get_device_name(),
            "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "vllm": __import__("vllm").__version__, "shapes": list(SHAPES), "NH": NH,
            "source_sha256": {"original_site": sha(args.original_site), "fixed_site": sha(args.fixed_site),
                              "official_source": sha(args.official_source), "probe": sha(Path(__file__))},
            "results": records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-site", type=Path, default=HERE / "f-received/instr/sitecustomize.py")
    p.add_argument("--fixed-site", type=Path, default=HERE / "f-mdnone-fix/sitecustomize.py")
    p.add_argument("--official-source", type=Path, default=HERE / "source/qwen_gdn_linear_attn.py")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ast-only", action="store_true", help="CPU source/guard validation only; no torch import")
    args = p.parse_args()
    try:
        old_ast, new_ast = source_contract(args.original_site, args.fixed_site, args.official_source)
        if args.ast_only:
            result = {"schema": "null-metadata-output-contract-v1", "ast_passed": True,
                      "gpu_tested": False, "scope": "CPU source/guard validation only",
                      "guards_preserved": True, "official_null_metadata_branch": "prewarm+return",
                      "source_sha256": {"original_site": sha(args.original_site),
                                        "fixed_site": sha(args.fixed_site),
                                        "official_source": sha(args.official_source),
                                        "probe": sha(Path(__file__))}}
        else:
            result = run(args, old_ast, new_ast)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"output": str(args.output), "summary": {k: v for k, v in result.items() if k != "results"}},
                         sort_keys=True, allow_nan=False), flush=True)
        return 0 if result.get("passed", result.get("ast_passed", False)) else 1
    except Exception as exc:
        traceback.print_exc()
        result = {"schema": "null-metadata-output-contract-v1", "passed": False,
                  "error_type": type(exc).__name__, "error": str(exc),
                  "gpu_execution_requested": not args.ast_only, "valid_gpu_result": False,
                  "scope": "probe failure; no positive numerical result"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
