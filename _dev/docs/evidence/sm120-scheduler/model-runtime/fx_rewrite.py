"""Official scheduler experiment: exact FX chain to an opaque M guard.

Never installed in baseline/probe. The matcher is structural and the opaque
dispatcher authorizes packed weights by exact canonical storage identity.
"""

from __future__ import annotations

import json
import operator
import os
from pathlib import Path

from fx_matcher import find_gateup_matches


def _inserted_meta(source: object) -> dict:
    """Copy provenance/output facts, never input facts for a different op.

    Inductor uses ``eager_input_vals`` to constrain a fallback op's inputs.
    The old auto_functionalized quant node has a different schema and input
    count from our functional op; inheriting that field makes normalization
    fail before lowering.
    """
    return {
        key: source.meta[key]
        for key in ("val", "tensor_meta", "nn_module_stack", "stack_trace")
        if key in source.meta
    }


class GateUpRewrite:
    def __init__(self, evidence_dir: str):
        self.evidence_dir = Path(evidence_dir)
        self.seq = 0

    def __call__(self, graph: object) -> None:
        import torch
        from vllm.compilation.passes.inductor_pass import get_pass_context

        self.seq += 1
        compile_range = get_pass_context().compile_range
        matches = find_gateup_matches(graph)
        rewritten = []
        static_small = []
        for match in matches:
            a_meta = match.a.meta.get("val", match.a.meta.get("tensor_meta"))
            if a_meta is None:
                raise RuntimeError("strict gate/up match lost A tensor metadata")
            m_text = str(a_meta.shape[0])
            if (
                compile_range.end <= 256
                or (m_text.isdecimal() and int(m_text) <= 256)
            ):
                static_small.append(match.gemm.name)
                continue
            # Use existing A/canonical B/As/Bs graph inputs. The new functional
            # op returns the two downstream tensors, so BF16 D is unnecessary.
            with graph.inserting_before(match.quant):
                fused = graph.call_function(
                    torch.ops.pro6000d_official_gateup.tuned.default,
                    args=(match.a, match.canonical_b,
                          match.a_scale, match.canonical_b_scale),
                )
                q_out = graph.call_function(operator.getitem, args=(fused, 0))
                q_scale = graph.call_function(operator.getitem, args=(fused, 1))
            # Post-grad passes run after fake propagation. Preserve exact
            # output FakeTensor metadata for Inductor's following lowering.
            old_out_meta = _inserted_meta(match.quant_out_item)
            old_scale_meta = _inserted_meta(match.quant_scale_item)
            if "val" not in old_out_meta or "val" not in old_scale_meta:
                raise RuntimeError("gate/up rewrite requires fake output values")
            fused.meta = _inserted_meta(match.quant)
            fused.meta.pop("tensor_meta", None)
            fused.meta["val"] = (old_out_meta["val"], old_scale_meta["val"])
            q_out.meta = old_out_meta
            q_scale.meta = old_scale_meta
            match.quant_out_item.replace_all_uses_with(q_out)
            match.quant_scale_item.replace_all_uses_with(q_scale)
            # The second BF16 D user reads only dim0; A has exactly the same
            # row count. Rebind it before erasing D and its old GEMM.
            for shape_user in match.shape_users:
                shape_user.replace_input_with(match.bf16, match.a)
            graph.erase_node(match.quant_out_item)
            graph.erase_node(match.quant_scale_item)
            graph.erase_node(match.quant)
            graph.erase_node(match.bf16)
            graph.erase_node(match.gemm)
            rewritten.append({
                "old_gemm": match.gemm.name,
                "new_op": fused.name,
                "canonical_weight_node": match.canonical_b.name,
                "canonical_scale_node": match.canonical_b_scale.name,
                "m_symbol": m_text,
            })
        if rewritten:
            graph.eliminate_dead_code()
            graph.lint()
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "pid": os.getpid(), "graph_number": self.seq,
            "compile_range": str(compile_range),
            "match_count": len(matches), "rewrite_count": len(rewritten),
            "static_small_fallbacks": static_small,
            "rewritten": rewritten,
            "scope": "FX replacements only; candidate launches require runtime registry hit",
        }
        (self.evidence_dir / f"fx-rewrite-{os.getpid()}-{self.seq}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def install_rewrite(manager: object, evidence_dir: str, library_sha256: str) -> None:
    from vllm.compilation.passes.inductor_pass import CallableInductorPass

    callback = GateUpRewrite(evidence_dir)
    manager.add(CallableInductorPass(
        callback, uuid="pro6000d-official-m16160-alongm-s1-v1:" + library_sha256
    ))
