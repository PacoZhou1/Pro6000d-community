"""Fail-closed structural matcher for the observed fixed-vLLM post-grad FX path.

This file only identifies graph nodes; it does not rewrite or launch. Runtime
authorization additionally requires lookup_packed(canonical_B, canonical_Bs).
Observed against E/fx-proof-v2/fx-proof-516-2.json. The saved proof predates
args/kwargs capture, so the next isolated model run must establish exact FX
operand forms before a rewrite can count as validated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateUpMatch:
    gemm: object
    bf16: object
    quant: object
    a: object
    canonical_b: object
    a_scale: object
    canonical_b_scale: object
    quant_out: object
    quant_scale: object
    quant_out_item: object
    quant_scale_item: object
    shape_users: tuple[object, ...]


def _signature(node: object) -> str:
    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    if "auto_functionalized" in target and args:
        target += "<" + str(args[0]) + ">"
    return target


def _tensor_meta(node: object) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    meta = getattr(node, "meta", {})
    value = meta.get("val", meta.get("tensor_meta"))
    if value is None or not hasattr(value, "shape") or not hasattr(value, "stride"):
        return None
    stride = value.stride() if callable(value.stride) else value.stride
    return str(value.dtype), tuple(map(str, value.shape)), tuple(map(str, stride))


def _item_of(node: object, parent: object, index: int) -> bool:
    args = getattr(node, "args", ())
    return (
        "getitem" in _signature(node)
        and len(args) == 2
        and args[0] is parent
        and type(args[1]) is int
        and args[1] == index
    )


def _unpermute(node: object, dtype: str, shape: tuple[str, str],
               stride: tuple[str, str]) -> object | None:
    args = getattr(node, "args", ())
    try:
        dimensions = tuple(args[1]) if len(args) == 2 else ()
    except TypeError:
        return None
    if (
        _signature(node) != "aten.permute.default"
        or len(args) != 2
        or dimensions != (1, 0)
        or not hasattr(args[0], "meta")
        or _tensor_meta(args[0]) != (dtype, shape, stride)
    ):
        return None
    return args[0]


def find_gateup_matches(graph: object) -> list[GateUpMatch]:
    """Return only exact, single-data-consumer paths; no shape-only identity."""
    found = []
    for gemm in graph.nodes:
        if _signature(gemm) != "auto_functionalized<_C.cutlass_scaled_mm.default>":
            continue
        inputs = list(gemm.all_input_nodes)
        if len(inputs) != 5 or len(gemm.users) != 1:
            continue
        out_alloc, a, b_transposed, a_scale, bs_transposed = inputs
        a_meta, as_meta = _tensor_meta(a), _tensor_meta(a_scale)
        out_meta = _tensor_meta(out_alloc)
        if a_meta is None or as_meta is None or out_meta is None:
            continue
        m = a_meta[1][0]
        if (
            a_meta != ("torch.float8_e4m3fn", (m, "5120"), ("5120", "1"))
            or as_meta != ("torch.float32", (m, "40"), ("1", m))
            or out_meta != ("torch.bfloat16", (m, "34816"), ("34816", "1"))
        ):
            continue
        b = _unpermute(
            b_transposed, "torch.float8_e4m3fn", ("34816", "5120"),
            ("5120", "1")
        )
        bs = _unpermute(
            bs_transposed, "torch.float32", ("272", "40"), ("40", "1")
        )
        if b is None or bs is None:
            continue
        bf16 = next(iter(gemm.users))
        if not _item_of(bf16, gemm, 1) or _tensor_meta(bf16) != out_meta:
            continue
        shape_users = [
            user for user in bf16.users
            if _signature(user) == "aten.sym_size.int"
        ]
        if any(
            len(getattr(user, "args", ())) != 2
            or user.args[0] is not bf16
            or user.args[1] != 0
            for user in shape_users
        ):
            continue
        data_users = [
            user for user in bf16.users
            if _signature(user) != "aten.sym_size.int"
        ]
        if len(data_users) != 1:
            continue
        quant = data_users[0]
        if (
            _signature(quant)
            != "auto_functionalized<pro6000d_fp8.silu_and_mul_per_block_quant.default>"
        ):
            continue
        kwargs = getattr(quant, "kwargs", {})
        if (
            kwargs.get("input") is not bf16
            or kwargs.get("group_size") != 128
            or kwargs.get("scale_ub", "missing") is not None
            or kwargs.get("is_scale_transposed") is not True
        ):
            continue
        q_out = kwargs.get("out")
        q_scale = kwargs.get("scales")
        if (
            q_out is None or q_scale is None
            or _tensor_meta(q_out)
            != ("torch.float8_e4m3fn", (m, "17408"), ("17408", "1"))
            or _tensor_meta(q_scale)
            != ("torch.float32", (m, "136"), ("1", m))
        ):
            continue
        output_items = list(quant.users)
        q_out_items = [user for user in output_items if _item_of(user, quant, 1)]
        q_scale_items = [user for user in output_items if _item_of(user, quant, 2)]
        if len(output_items) != 2 or len(q_out_items) != 1 or len(q_scale_items) != 1:
            continue
        found.append(GateUpMatch(
            gemm, bf16, quant, a, b, a_scale, bs, q_out, q_scale,
            q_out_items[0], q_scale_items[0], tuple(shape_users)
        ))
    return found
