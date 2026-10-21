#!/usr/bin/env python3
"""CPU-only check for the inserted op's Inductor schema/metadata contract.

Run in the fixed image with epilogue mode unset. This does not load vLLM,
weights, candidate libraries, or CUDA. It tests the specific stale-metadata
failure and a four-Tensor custom op through torch.compile/Inductor.
"""

from __future__ import annotations

import json
import types


def main() -> None:
    import torch
    from torch.fx.operator_schemas import normalize_function

    from fx_rewrite import _inserted_meta

    library = torch.library.Library("pro6000d_mainloop_gateup_cpucheck", "DEF")
    library.define(
        "fused(Tensor A, Tensor B, Tensor A_scale, Tensor B_scale) "
        "-> (Tensor, Tensor)"
    )

    def cpu(a, b, a_scale, b_scale):
        return (a + b + a_scale, a - b + b_scale)

    def meta(a, b, a_scale, b_scale):
        return (torch.empty_like(a), torch.empty_like(a))

    library.impl("fused", cpu, "CPU")
    library.impl("fused", meta, "Meta")
    op = torch.ops.pro6000d_mainloop_gateup_cpucheck.fused.default
    t = torch.ones((2, 3))
    old_six_args = (t, t, t, t, t, t)
    assert normalize_function(op, old_six_args, {}) is None
    assert normalize_function(op, (t, t, t, t), {}) is not None

    old_quant = types.SimpleNamespace(meta={
        "val": (t, t, t),
        "eager_input_vals": (old_six_args, {}),
        "nn_module_stack": {"test": ("test.module", "Test")},
        "stack_trace": "selfcheck",
    })
    inserted = _inserted_meta(old_quant)
    assert "eager_input_vals" not in inserted
    assert set(inserted) == {"val", "nn_module_stack", "stack_trace"}

    def workload(a, b, a_scale, b_scale):
        out, scales = op(a, b, a_scale, b_scale)
        return out + scales

    expected = workload(t, t * 2, t * 3, t * 4)
    compiled = torch.compile(workload, backend="inductor", fullgraph=True)
    actual = compiled(t, t * 2, t * 3, t * 4)
    assert torch.equal(actual, expected)
    print(json.dumps({
        "passed": True,
        "scope": "fixed-image CPU schema/metadata and isolated Inductor custom op",
        "torch_version": torch.__version__,
        "stale_six_arg_normalization": "rejected",
        "four_arg_normalization": "accepted",
        "stale_eager_input_vals_copied": False,
        "compiled_equal": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
