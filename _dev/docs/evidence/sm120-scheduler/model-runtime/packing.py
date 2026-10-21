"""Startup-only canonical+paired gate/up weight preparation for fixed FP8 model.

This module is inert unless sitecustomize enables it. It keeps canonical
parameters intact, so all unoptimized calls retain the proven vLLM path.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


_BUFFER_W = "_pro6000d_gateup_paired_weight"
_BUFFER_S = "_pro6000d_gateup_paired_scale"
_EVENTS: list[dict] = []
_RUN_ID = uuid4().hex[:12]
_WEIGHT_REGISTRY: dict[tuple[str, int, int], dict] = {}


def _registry_key(weight: object, scale: object) -> tuple[str, int, int]:
    return str(weight.device), weight.data_ptr(), scale.data_ptr()


def lookup_packed(weight: object, scale: object) -> dict | None:
    """Authorize by exact live canonical weight+scale storage, never shape.

    The registry holds strong tensor references, so allocator pointer reuse
    cannot silently map a new model to an old packed pair. Shape/stride and
    storage-base checks reject aliases and transposed views. This is called
    only with canonical [N,K] B and [N/128,K/128] Bs tensors.
    """
    entry = _WEIGHT_REGISTRY.get(_registry_key(weight, scale))
    if entry is None:
        return None
    original_weight = entry["canonical_weight"]
    original_scale = entry["canonical_scale"]
    for current, original in ((weight, original_weight), (scale, original_scale)):
        if (
            current.device != original.device
            or current.dtype != original.dtype
            or tuple(current.shape) != tuple(original.shape)
            or tuple(current.stride()) != tuple(original.stride())
            or current.storage_offset() != original.storage_offset()
            or current.untyped_storage().data_ptr()
            != original.untyped_storage().data_ptr()
        ):
            return None
    return entry


def _target(name: str, module: object) -> bool:
    return (
        name.endswith(".mlp.gate_up_proj")
        and type(module).__name__ == "MergedColumnParallelLinear"
        and type(getattr(module, "quant_method", None)).__name__ == "Fp8LinearMethod"
        and type(getattr(getattr(module, "quant_method", None), "fp8_linear", None)).__name__
        == "CutlassFp8BlockScaledMMKernel"
    )


def prepare_model(model: object, evidence_dir: str) -> None:
    """Run after vLLM's weight post-processing, before warmup/compilation.

    Intentional hard failure on target shape/layout mismatch: a baseline arm
    missing its packed duplicate would not have comparable memory pressure.
    """
    import torch

    report = {
        "pid": os.getpid(),
        "run_id": _RUN_ID,
        "event_index": len(_EVENTS) + 1,
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "event": "gateup_startup_pack",
        "model_class": type(model).__name__,
        "model_object_id": hex(id(model)),
        "modules": [],
    }
    for name, module in model.named_modules():
        if not _target(name, module):
            continue
        w = module.weight
        s = module.weight_scale_inv
        expected = (
            w.device.type == "cuda"
            and w.dtype == torch.float8_e4m3fn
            and tuple(w.shape) == (34816, 5120)
            and tuple(w.stride()) == (5120, 1)
            and s.device == w.device
            and s.dtype == torch.float32
            and tuple(s.shape) == (272, 40)
            and tuple(s.stride()) == (40, 1)
        )
        if not expected:
            raise RuntimeError(f"gate/up startup layout mismatch at {name}")
        if hasattr(module, _BUFFER_W) or hasattr(module, _BUFFER_S):
            raise RuntimeError(f"gate/up duplicate registration at {name}")
        # 136 gate/up pairs, with each 128 output rows kept together.
        rows = torch.arange(34816, dtype=torch.long, device=w.device).reshape(272, 128)
        group_order = torch.stack(
            (torch.arange(136, device=w.device),
             torch.arange(136, 272, device=w.device)), dim=1
        ).reshape(-1)
        packed_w = w.index_select(0, rows.index_select(0, group_order).reshape(-1)).contiguous()
        packed_s = s.index_select(0, group_order).contiguous()
        module.register_buffer(_BUFFER_W, packed_w, persistent=False)
        module.register_buffer(_BUFFER_S, packed_s, persistent=False)
        key = _registry_key(w, s)
        if key in _WEIGHT_REGISTRY:
            raise RuntimeError(f"canonical weight/scale registry collision at {name}")
        _WEIGHT_REGISTRY[key] = {
            "name": name,
            "module_class": type(module).__name__,
            "module_object_id": str(id(module)),
            "canonical_weight": w,
            "canonical_scale": s,
            "packed_weight": packed_w,
            "packed_scale": packed_s,
        }
        report["modules"].append({
            "name": name,
            "module_object_id": str(id(module)),
            "canonical_weight_data_ptr": hex(w.data_ptr()),
            "canonical_scale_data_ptr": hex(s.data_ptr()),
            "packed_weight_data_ptr": hex(packed_w.data_ptr()),
            "packed_scale_data_ptr": hex(packed_s.data_ptr()),
            "weight_bytes": packed_w.numel() * packed_w.element_size(),
            "scale_bytes": packed_s.numel() * packed_s.element_size(),
            "weight_shape": list(packed_w.shape),
            "scale_shape": list(packed_s.shape),
        })
    report["count"] = len(report["modules"])
    report["total_bytes"] = sum(
        x["weight_bytes"] + x["scale_bytes"] for x in report["modules"]
    )
    if report["count"] not in (1, 64, 65):
        raise RuntimeError(f"unexpected gate/up module count: {report['count']}")
    path = Path(evidence_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"startup-pack-{os.getpid()}-{_RUN_ID}-{report['event_index']}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _EVENTS.append(report)
    summary = {
        "pid": os.getpid(),
        "run_id": _RUN_ID,
        "event": "gateup_startup_pack_summary",
        "events": [
            {"event_index": x["event_index"], "model_class": x["model_class"],
             "count": x["count"], "total_bytes": x["total_bytes"]}
            for x in _EVENTS
        ],
        "cumulative_count": sum(x["count"] for x in _EVENTS),
        "cumulative_bytes": sum(x["total_bytes"] for x in _EVENTS),
    }
    (path / f"startup-pack-{os.getpid()}-{_RUN_ID}-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
