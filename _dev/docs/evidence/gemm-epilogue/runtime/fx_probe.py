"""Read-only post-activation-fusion FX proof pass for fixed vLLM 0.31.

The pass intentionally does not rewrite the graph. It records dependencies,
tensor metadata and known gate/up signatures. Candidate authorization is a
separate runtime check against the startup weight registry.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _target(node: object) -> str:
    target = getattr(node, "target", None)
    return str(target) if target is not None else ""


def _op_signature(node: object) -> str:
    """Include the wrapped op of auto_functionalized without tensor values."""
    target = _target(node)
    if "auto_functionalized" in target and getattr(node, "args", ()):
        target += "<" + str(node.args[0]) + ">"
    return target


def _safe_arg(value: object) -> object:
    """Describe FX arguments without tensor contents or source paths."""
    from torch.fx import Node

    # Torch OpOverload also has .name and .op; .name is a bound method.
    # Only real FX nodes may supply a node-name field.
    if isinstance(value, Node):
        return {"node": value.name}
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value).__name__ in ("SymInt", "SymFloat", "SymBool"):
        return {"symbolic": str(value)}
    if isinstance(value, (tuple, list)):
        return [_safe_arg(x) for x in value]
    if hasattr(value, "_schema"):
        return {"operator": str(value)}
    return {"type": type(value).__name__}


def _meta(node: object) -> dict:
    meta = getattr(node, "meta", {})
    value = meta.get("val", meta.get("tensor_meta"))
    result: dict = {}
    if value is not None:
        for key in ("shape", "stride", "dtype"):
            attr = getattr(value, key, None)
            if attr is None:
                continue
            if key == "stride" and callable(attr):
                attr = attr()
            # SymInt is not JSON serializable. str() keeps the symbolic
            # expression intact; int() would specialize the dynamic M here.
            result[key] = (
                [str(dim) for dim in attr]
                if key in ("shape", "stride")
                else str(attr)
            )
    stack = meta.get("nn_module_stack")
    if stack:
        # The keys are runtime module identities; tuple values in fixed torch
        # are (qualified module name, type). Preserve both for an exact join
        # with startup-pack module_object_id, without source/tensor contents.
        result["module_stack"] = [
            {
                "id": str(key),
                "name": str(value[0]) if isinstance(value, tuple) and value else "",
                "type": (
                    getattr(value[1], "__name__", str(value[1]))
                    if isinstance(value, tuple) and len(value) > 1 else ""
                ),
            }
            for key, value in stack.items()
        ]
    return result


def _walk_to_quant(node: object, depth: int = 6) -> list[list[str]]:
    """Follow only transparent getitem/view nodes; keep complete paths."""
    frontier = [(node, [node.name])]
    found = []
    for _ in range(depth):
        following = []
        for current, path in frontier:
            for user in current.users:
                text = _op_signature(user)
                new_path = path + [user.name]
                if "silu_and_mul_per_block_quant" in text:
                    found.append(new_path)
                elif any(s in text for s in ("getitem", "view", "reshape", "alias")):
                    following.append((user, new_path))
        frontier = following
    return found


class GraphProof:
    def __init__(self, evidence_dir: str):
        self.evidence_dir = Path(evidence_dir)
        self.seq = 0

    def __call__(self, graph: object) -> None:
        self.seq += 1
        nodes = list(graph.nodes)
        try:
            from fx_matcher import find_gateup_matches

            structural_matches = [
                {"gemm": match.gemm.name, "bf16": match.bf16.name,
                 "quant": match.quant.name,
                 "canonical_b": match.canonical_b.name,
                 "canonical_b_scale": match.canonical_b_scale.name}
                for match in find_gateup_matches(graph)
            ]
            matcher_error = None
        except Exception as exc:  # diagnostic pass must not abort model startup
            structural_matches = []
            matcher_error = type(exc).__name__
        rows = []
        for node in nodes:
            rows.append({
                "name": node.name,
                "op": node.op,
                "target": _op_signature(node),
                "args": _safe_arg(node.args),
                "kwargs": {str(k): _safe_arg(v) for k, v in node.kwargs.items()},
                "inputs": [x.name for x in node.all_input_nodes],
                "users": [x.name for x in node.users],
                "meta": _meta(node),
            })
        gemms = []
        for node in nodes:
            if "cutlass_scaled_mm" not in _op_signature(node):
                continue
            paths = _walk_to_quant(node)
            gemms.append({
                "node": node.name,
                "direct_users": [u.name for u in node.users],
                "quant_paths": paths,
                "single_direct_user": len(node.users) == 1,
                "input_nodes": [x.name for x in node.all_input_nodes],
                "meta": _meta(node),
            })
        payload = {
            "pid": os.getpid(), "graph_number": self.seq,
            "pass_position": "after_vllm_activation_quant_fusion",
            "node_count": len(nodes), "nodes": rows, "gemms": gemms,
            "strict_structural_matches": structural_matches,
            "strict_matcher_error_type": matcher_error,
        }
        try:
            from vllm.compilation.passes.inductor_pass import get_pass_context

            payload["compile_range"] = str(get_pass_context().compile_range)
        except Exception:
            payload["compile_range"] = None
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        (self.evidence_dir / f"fx-proof-{os.getpid()}-{self.seq}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def install_probe(manager: object, evidence_dir: str) -> None:
    """Append via the fixed official PostGradPassManager.add() API."""
    from vllm.compilation.passes.inductor_pass import CallableInductorPass

    proof = GraphProof(evidence_dir)
    manager.add(CallableInductorPass(proof, uuid="pro6000d-gateup-fx-proof-v1"))
