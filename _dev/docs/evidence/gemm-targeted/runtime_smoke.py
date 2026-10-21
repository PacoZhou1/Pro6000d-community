#!/usr/bin/env python3
"""GPU smoke test for the opt-in down Stream-K sitecustomize experiment.

Run only in a reserved GPU window, with the fixed vLLM 0.31.0 image and the
separately built libdown_streamk_fp8.so mounted. The sitecustomize file is
loaded with runpy before importing the CUTLASS linear module. This script
does not start or modify a serving process. It emits one final JSON object
with per-check evidence and exits nonzero on any failure.

Example:
  python3 runtime_smoke.py \
    --library /build/down-streamk/libdown_streamk_fp8.so \
    --instrument-dir /instrument/runtime_down_streamk

The standalone_fp8_probe.py generator must be importable beside this file,
or in --instrument-dir (the caller may mount a directory containing both).
"""

import argparse
import importlib
import json
import os
import runpy
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


K = 17408
N = 5120
SCALE_K = K // 128
SCALE_N = N // 128


def _score_ok(result, max_nrmse=0.02, min_cosine=0.999):
    return (result["finite"] and result["nrmse"] <= max_nrmse and
            result["cosine"] >= min_cosine)


def run(args):
    site_path = args.instrument_dir / "sitecustomize.py"
    if not site_path.is_file():
        raise FileNotFoundError(f"sitecustomize.py missing in {args.instrument_dir}")
    if not args.library.is_file():
        raise FileNotFoundError(f"library missing: {args.library}")
    # Deliberately execute before importing the class being patched.
    target = "vllm.model_executor.kernels.linear.scaled_mm.cutlass"
    if target in sys.modules:
        raise RuntimeError("CUTLASS module was imported before instrumentation")
    os.environ["PRO6000D_DOWN_STREAMK_ENABLE"] = "1"
    os.environ["PRO6000D_DOWN_STREAMK_LIBRARY"] = str(args.library.resolve())
    # An outer PYTHONPATH may already have auto-loaded this sitecustomize at
    # interpreter startup. Avoid installing a duplicate hook in that case.
    hook_present = any(type(item).__name__ == "_CutlassImportHook" and
                       type(item).__module__ == "sitecustomize"
                       for item in sys.meta_path)
    if not hook_present:
        runpy.run_path(str(site_path))

    import torch
    from vllm import _custom_ops as ops

    if str(args.instrument_dir) not in sys.path:
        sys.path.insert(0, str(args.instrument_dir))
    try:
        from standalone_fp8_probe import make_activation, make_weight, score
    except ModuleNotFoundError:
        # Typical layout: runtime_smoke.py and standalone_fp8_probe.py share
        # a parent, while sitecustomize.py lives in runtime_down_streamk/.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from standalone_fp8_probe import make_activation, make_weight, score

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("smoke test requires the reserved SM120 GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    cutlass = importlib.import_module(target)
    kernel_class = cutlass.CutlassFp8BlockScaledMMKernel
    if kernel_class.apply_block_scaled_mm.__name__ != "_patched":
        raise RuntimeError("sitecustomize did not patch the fixed CUTLASS class")
    kernel = SimpleNamespace(config=SimpleNamespace(out_dtype=torch.bfloat16))

    def patched(a, b, a_scale, b_scale):
        return kernel_class.apply_block_scaled_mm(
            kernel, a, b, a_scale, b_scale)

    def official(a, b, a_scale, b_scale):
        return ops.cutlass_scaled_mm(
            a, b.T, out_dtype=torch.bfloat16,
            scale_a=a_scale, scale_b=b_scale.T)

    weight, weight_scale = make_weight(K, N, 20261015)
    assert weight.shape == (N, K) and weight_scale.shape == (SCALE_N, SCALE_K)
    activations = {m: make_activation(m, K, 1000 + m) for m in (1, 5, 17)}
    results = {"library": str(args.library.resolve()),
               "instrument_dir": str(args.instrument_dir.resolve()),
               "torch": torch.__version__,
               "device": torch.cuda.get_device_name(), "checks": {}}

    # The experimental path has a different reduction order, so require
    # finite full output and a bounded full-output difference from official.
    a5, as5 = activations[5]
    eager5 = patched(a5, weight, as5, weight_scale)
    original5 = official(a5, weight, as5, weight_scale)
    torch.cuda.synchronize()
    m5_score = score(eager5, original5)
    results["checks"]["eager_m5"] = {
        "M": 5, "candidate_vs_official_full": m5_score,
        "pass": _score_ok(m5_score),
    }
    for m in (1, 17):
        a, a_scale = activations[m]
        actual = patched(a, weight, a_scale, weight_scale)
        expected = official(a, weight, a_scale, weight_scale)
        torch.cuda.synchronize()
        exact = bool(torch.equal(actual, expected))
        results["checks"][f"eager_m{m}_fallback"] = {
            "M": m, "bit_exact_official": exact, "pass": exact,
        }

    # A custom op with a fake impl should remain opaque to Dynamo. Both
    # branches must compile as full graphs; M=17 must retain exact fallback.
    def forward(a, b, a_scale, b_scale):
        return patched(a, b, a_scale, b_scale)

    # A numerical check alone cannot detect the old M==5 outer guard miss:
    # that guard compiled away and the official op still produced correct
    # output. Capture Dynamo's FX graph with an inspection backend and require
    # the opaque op node. The revised runtime keeps this node for every down
    # M, with M==5 versus official fallback selected inside its CUDA impl.
    captured_graph_targets = []

    def inspect_backend(graph_module, _example_inputs):
        captured_graph_targets.append([
            str(node.target) for node in graph_module.graph.nodes
            if node.op == "call_function"])
        return graph_module.forward

    inspected = torch.compile(forward, backend=inspect_backend,
                              fullgraph=True, dynamic=True)
    # Compile a non-candidate M first. Guard v1 missed in normal serving
    # because the prefill-range trace preceded M=5 decode/capture.
    a17, as17 = activations[17]
    inspected(a17, weight, as17, weight_scale)
    inspected(a5, weight, as5, weight_scale)
    opaque_counts = [sum("pro6000d_exp.down_streamk" in target
                         for target in targets)
                     for targets in captured_graph_targets]
    results["checks"]["dynamo_opaque_op_present"] = {
        "compile_order_M": [17, 5], "captured_graph_count": len(opaque_counts),
        "custom_op_node_counts_per_graph": opaque_counts,
        "pass": bool(opaque_counts) and all(count == 1 for count in opaque_counts),
    }

    compiled = torch.compile(forward, fullgraph=True, dynamic=True)
    for m in (17, 5):
        a, a_scale = activations[m]
        actual = compiled(a, weight, a_scale, weight_scale)
        expected = official(a, weight, a_scale, weight_scale)
        torch.cuda.synchronize()
        if m == 5:
            difference = score(actual, expected)
            passed = _score_ok(difference)
            detail = {"candidate_vs_official_full": difference}
        else:
            exact = bool(torch.equal(actual, expected))
            passed = exact
            detail = {"bit_exact_official": exact}
        results["checks"][f"torch_compile_dynamic_m{m}"] = {
            "M": m, "fullgraph": True, "dynamic": True,
            **detail, "pass": passed,
        }

    # Capture one M=5 custom op. Mutate both FP8 activation bytes and scales
    # between replays; a stale capture output must fail the second comparison.
    a5_changed, as5_changed = make_activation(5, K, 2005)
    static_a = a5.clone()
    static_as = torch.empty_strided((5, SCALE_K), (1, 5),
                                   dtype=as5.dtype, device=as5.device)
    static_as.copy_(as5)
    assert static_as.stride() == (1, 5)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = patched(static_a, weight, static_as, weight_scale)
    graph.replay()
    torch.cuda.synchronize()
    first = graph_output.clone()
    first_expected = official(a5, weight, as5, weight_scale)
    first_score = score(first, first_expected)
    static_a.view(torch.uint8).copy_(a5_changed.view(torch.uint8))
    static_as.copy_(as5_changed)
    graph.replay()
    torch.cuda.synchronize()
    second = graph_output.clone()
    second_expected = official(a5_changed, weight, as5_changed, weight_scale)
    second_score = score(second, second_expected)
    changed = not bool(torch.equal(first, second))
    results["checks"]["cuda_graph_replay_new_input"] = {
        "capture_M": 5, "replay_outputs_differ": changed,
        "first_vs_official_full": first_score,
        "second_vs_official_full": second_score,
        "pass": changed and _score_ok(first_score) and _score_ok(second_score),
    }

    # Different streams receive distinct outputs and per-call scratch. Wait
    # for the default-stream-generated inputs before either worker stream uses
    # them; then synchronize both before comparing with official results.
    a5_other, as5_other = make_activation(5, K, 3005)
    current = torch.cuda.current_stream()
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    stream_a.wait_stream(current)
    stream_b.wait_stream(current)
    with torch.cuda.stream(stream_a):
        out_a = patched(a5, weight, as5, weight_scale)
    with torch.cuda.stream(stream_b):
        out_b = patched(a5_other, weight, as5_other, weight_scale)
    stream_a.synchronize()
    stream_b.synchronize()
    expected_a = official(a5, weight, as5, weight_scale)
    expected_b = official(a5_other, weight, as5_other, weight_scale)
    torch.cuda.synchronize()
    score_a = score(out_a, expected_a)
    score_b = score(out_b, expected_b)
    streams_differ = not bool(torch.equal(out_a, out_b))
    results["checks"]["two_streams_independent"] = {
        "outputs_differ": streams_differ,
        "stream_a_vs_official_full": score_a,
        "stream_b_vs_official_full": score_b,
        "pass": streams_differ and _score_ok(score_a) and _score_ok(score_b),
    }
    results["pass"] = all(item["pass"] for item in results["checks"].values())
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--instrument-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        traceback.print_exc()
        result = {"pass": False, "error_type": type(exc).__name__,
                  "error": str(exc), "checks": {}}
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
