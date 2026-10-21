"""Opt-in functional PyTorch bridge for official BF16 scheduler ABI1.

The graph pass supplies *canonical* B/Bs. Only an exact startup-registry hit
at M=16160 uses AlongM/s1. Other calls execute canonical GEMM + existing V2
activation quantization and are explicitly reported as fallback dispatches.
"""

from __future__ import annotations

import logging
import os

from packing import lookup_packed


_LIBRARY = None
_ABI = None
_LAST_DISPATCH = None
_LOGGED = set()
_DISPATCH_EVIDENCE = os.environ.get("PRO6000D_OFFICIAL_DISPATCH_EVIDENCE") == "1"
_EVIDENCE_SEEN = set()
if _DISPATCH_EVIDENCE:
    from threading import Lock

    _EVIDENCE_LOCK = Lock()


def _memory_snapshot(device):
    import torch

    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _write_dispatch_evidence(a, b, b_scale, entry, m, dispatch,
                             before, after, temporary_buffers, capturing):
    """Best-effort first Python dispatch record; never a graph-replay count."""
    import hashlib
    import json
    from pathlib import Path

    module = entry["name"] if entry else "unregistered_or_small_m"
    key = (module, int(m), dispatch)
    with _EVIDENCE_LOCK:
        if key in _EVIDENCE_SEEN:
            return
        directory = Path(os.environ["PRO6000D_OFFICIAL_EVIDENCE_DIR"])
        if not directory.is_absolute():
            raise RuntimeError("dispatch evidence directory must be absolute")
        record = {
            "event": "first_python_dispatch_or_capture_for_module_m_branch",
            "scope": "Python dispatch/capture only; not CUDA Graph replay or request counts",
            "pid": os.getpid(),
            "module_name": module,
            "module_object_id": entry["module_object_id"] if entry else None,
            "module_class": entry["module_class"] if entry else None,
            "M": int(m),
            "dispatch": dispatch,
            "cuda_graph_capturing": bool(capturing),
            "device": str(a.device),
            "canonical_weight_data_ptr": hex(b.data_ptr()),
            "canonical_scale_data_ptr": hex(b_scale.data_ptr()),
            "packed_weight_data_ptr": (
                hex(entry["packed_weight"].data_ptr()) if entry else None
            ),
            "packed_scale_data_ptr": (
                hex(entry["packed_scale"].data_ptr()) if entry else None
            ),
            "registered_identity_match": entry is not None,
            "output_buffer_bytes": {
                "fp8_output": int(m) * 17408,
                "fp32_scales": int(m) * 136 * 4,
            },
            "temporary_buffer_bytes": temporary_buffers,
            "memory_before": before,
            "memory_after": after,
            "memory_delta_after_minus_before": {
                k: after[k] - before[k] for k in before
            },
            "memory_scope": (
                "CUDA allocator process/device snapshot; peak is cumulative since "
                "last reset, not per-op peak; no device synchronization"
            ),
        }
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"dispatch-{os.getpid()}-{digest}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        _EVIDENCE_SEEN.add(key)


def _validate(a, b, a_scale, b_scale) -> int:
    import torch

    if a.ndim != 2 or a.shape[1] != 5120:
        raise RuntimeError("gate/up A shape mismatch")
    m = a.shape[0]
    if not 0 < m <= 32768:
        raise RuntimeError("gate/up M unsupported")
    if (
        a.device.type != "cuda" or a.dtype != torch.float8_e4m3fn
        or tuple(a.stride()) != (5120, 1)
        or b.device != a.device or b.dtype != torch.float8_e4m3fn
        or tuple(b.shape) != (34816, 5120)
        or tuple(b.stride()) != (5120, 1)
        or a_scale.device != a.device or a_scale.dtype != torch.float32
        or tuple(a_scale.shape) != (m, 40)
        or tuple(a_scale.stride()) != (1, m)
        or b_scale.device != a.device or b_scale.dtype != torch.float32
        or tuple(b_scale.shape) != (272, 40)
        or tuple(b_scale.stride()) != (40, 1)
    ):
        raise RuntimeError("gate/up tensor layout mismatch")
    return m


def _allocate_outputs(a, m):
    import torch

    out = torch.empty((m, 17408), dtype=torch.float8_e4m3fn, device=a.device)
    scales = torch.empty((136, m), dtype=torch.float32, device=a.device).t()
    return out, scales


def _dispatch(a, b, a_scale, b_scale):
    import torch

    global _LAST_DISPATCH
    m = _validate(a, b, a_scale, b_scale)
    before = _memory_snapshot(a.device) if _DISPATCH_EVIDENCE else None
    out, scales = _allocate_outputs(a, m)
    entry = lookup_packed(b, b_scale) if m == 16160 else None
    evidence_entry = entry
    if _DISPATCH_EVIDENCE and evidence_entry is None:
        evidence_entry = lookup_packed(b, b_scale)
    candidate = entry is not None and _ABI is not None
    if candidate:
        with torch.cuda.device(a.device):
            candidate = torch.cuda.get_device_capability(a.device) == (12, 0)
    d = torch.empty((m, 34816), dtype=torch.bfloat16, device=a.device)
    if not candidate:
        # Existing canonical GEMM + V2 quant; no new scheduler launch.
        torch.ops._C.cutlass_scaled_mm(
            d, a, b.t(), a_scale, b_scale.t(), None
        )
        _LAST_DISPATCH = "canonical_v2_fallback"
        temporary_buffers = {"bf16_intermediate": d.numel() * d.element_size()}
    else:
        workspace_bytes = _ABI.workspace_bytes(m)
        workspace = torch.empty((workspace_bytes,), dtype=torch.uint8,
                                device=a.device)
        with torch.cuda.device(a.device):
            stream = torch.cuda.current_stream(a.device)
            _ABI.launch(
                d.data_ptr(), a.data_ptr(), b.data_ptr(), a_scale.data_ptr(),
                b_scale.data_ptr(), m, workspace.data_ptr(), workspace_bytes,
                stream.cuda_stream,
            )
            for tensor in (d, workspace, out, scales, a, a_scale, b, b_scale):
                tensor.record_stream(stream)
        _LAST_DISPATCH = "official_along_m_s1_m16160"
        temporary_buffers = {
            "bf16_intermediate": d.numel() * d.element_size(),
            "cutlass_workspace": workspace.numel() * workspace.element_size(),
        }
    torch.ops.pro6000d_fp8.silu_and_mul_per_block_quant(
        out, d, scales, 128, None, True
    )
    # Python dispatch happens at capture time, not on CUDA Graph replay.
    # These are startup/capture diagnostics, never request launch counts.
    if _DISPATCH_EVIDENCE:
        try:
            capturing = torch.cuda.is_current_stream_capturing()
            _write_dispatch_evidence(
                a, b, b_scale, evidence_entry, m, _LAST_DISPATCH,
                before, _memory_snapshot(a.device), temporary_buffers,
                capturing,
            )
            if _LAST_DISPATCH not in _LOGGED:
                logging.getLogger("pro6000d_gateup").warning(
                    "gate/up dispatcher: %s M=%d (dispatch/capture only)",
                    _LAST_DISPATCH, m,
                )
                _LOGGED.add(_LAST_DISPATCH)
        except Exception as exc:
            logging.getLogger("pro6000d_gateup").warning(
                "gate/up dispatch evidence failed: %s", type(exc).__name__
            )
    return out, scales


def _fake(a, b, a_scale, b_scale):
    import torch

    m = a.shape[0]
    out = torch.empty((m, 17408), dtype=torch.float8_e4m3fn, device=a.device)
    scales = torch.empty((136, m), dtype=torch.float32, device=a.device).t()
    return out, scales


def register(abi) -> None:
    """Register exactly once after validated-library startup gating."""
    import torch

    global _LIBRARY, _ABI
    if _LIBRARY is not None:
        if _ABI is not abi:
            raise RuntimeError("gate/up op registered with different ABI")
        return
    library = torch.library.Library("pro6000d_official_gateup", "DEF")
    library.define(
        "tuned(Tensor A, Tensor B, Tensor A_scale, Tensor B_scale) "
        "-> (Tensor, Tensor)"
    )
    library.impl("tuned", _dispatch, "CUDA")
    library.impl("tuned", _fake, "Meta")
    _ABI, _LIBRARY = abi, library


def last_dispatch_for_eager_smoke() -> str | None:
    return _LAST_DISPATCH
