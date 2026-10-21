"""Synchronous, fail-closed F2 in-situ bit audit. Diagnostic use only.

No tensor contents are serialized. Each eligible call checks six complete
outputs/state tensors and records only sizes/counts and first mismatch bits.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from threading import Lock

_LOCK = Lock()
_CALLS = 0
_FAILURES = 0
_LOG = logging.getLogger("pro6000d_f_fusion_diagnostic")


def _record(row):
    directory = Path(os.environ.get("PRO6000D_F_DIAG_DIR", "/profiles"))
    if not directory.is_absolute():
        raise RuntimeError("PRO6000D_F_DIAG_DIR must be absolute")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"f2-diagnostic-{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    _LOG.warning("P3 VERIFY F2 %s", json.dumps(row, sort_keys=True, allow_nan=False))


def _compare(torch, actual, expected, label):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"tensor": label, "pass": False, "reason": "shape_or_dtype_mismatch",
                "actual_shape": list(actual.shape), "reference_shape": list(expected.shape),
                "actual_dtype": str(actual.dtype), "reference_dtype": str(expected.dtype)}
    integer = {torch.bfloat16: torch.int16, torch.float16: torch.int16,
               torch.float32: torch.int32, torch.float8_e4m3fn: torch.uint8}.get(actual.dtype)
    if integer is None:
        raise RuntimeError(f"unsupported diagnostic dtype: {actual.dtype}")
    x = actual.contiguous().view(integer).reshape(-1)
    y = expected.contiguous().view(integer).reshape(-1)
    different = x != y
    count = int(different.sum().item())
    record = {"tensor": label, "dtype": str(actual.dtype), "shape": list(actual.shape),
              "elements": actual.numel(), "mismatches": count, "pass": count == 0}
    if count:
        index = int(different.to(torch.int8).argmax().item())
        record["first_mismatch"] = {"flat_element_index": index,
                                     "actual_bits": int(x[index].item()),
                                     "reference_bits": int(y[index].item())}
    return record


def verify_f2(*, torch, original_conv, original_post, candidate, x, weight,
              bias, conv_kw, a, b, A_log, dt_bias, num_k_heads, head_k_dim,
              head_v_dim, metadata=None):
    """Same real x and starting state; only the candidate updates real state.

    The official reference gets a cloned full conv_state. Clone preserves the
    real layout where supported, and every state element is checked afterward.
    This covers inactive slots too. Host synchronization is intentional.
    """
    global _CALLS, _FAILURES
    with _LOCK:
        _CALLS += 1
        call_id = _CALLS
    state = conv_kw["conv_states"]
    L = x.shape[1]
    base = {"schema": "f2-in-situ-v1", "pid": os.getpid(), "call": call_id,
            "scope": "synchronous numerical diagnostic, not latency/throughput",
            "L": int(L), "x_shape": list(x.shape), "x_stride": list(x.stride()),
            "state_shape": list(state.shape), "state_stride": list(state.stride()),
            "state_data_ptr": hex(state.data_ptr()),
            "num_prefills": getattr(metadata, "num_prefills", None),
            "num_decodes": getattr(metadata, "num_decodes", None),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if torch.cuda.is_current_stream_capturing():
        _record({**base, "pass": False, "error": "capture_not_supported_no_silent_skip"})
        raise RuntimeError("F2 synchronous diagnostic encountered CUDA capture; use an explicit eager diagnostic run or implement a separate replay audit")
    reference_state = state.clone(memory_format=torch.preserve_format)
    reference_kw = dict(conv_kw)
    reference_kw["conv_states"] = reference_state
    # Input x is read by the official conv API. Only its conv_states argument
    # is mutable; the clone prevents reference execution changing the real state.
    conv_reference = original_conv(x, weight, bias, **reference_kw).transpose(0, 1)
    expected = original_post(conv_reference, a, b, A_log, dt_bias,
                             num_k_heads, head_k_dim, head_v_dim,
                             apply_l2norm=True, output_g_exp=False)
    actual = candidate(x.transpose(0, 1), weight, state,
                       conv_kw["cache_indices"], conv_kw["has_initial_state"],
                       a, b, A_log, dt_bias, num_k_heads, head_k_dim, head_v_dim)
    torch.cuda.current_stream(x.device).synchronize()
    if len(actual) != 5 or len(expected) != 5:
        _record({**base, "pass": False, "error": "unexpected_output_count"})
        raise RuntimeError("F2 diagnostic expected q/k/v/g/beta")
    checks = [_compare(torch, value, reference, label)
              for label, value, reference in zip(("q", "k", "v", "g", "beta"), actual, expected)]
    checks.append(_compare(torch, state, reference_state, "conv_state"))
    passed = all(c["pass"] for c in checks)
    with _LOCK:
        _FAILURES += not passed
        failures = _FAILURES
    _record({**base, "pass": passed, "cumulative_failed_calls": failures,
             "comparisons": checks})
    if not passed:
        raise RuntimeError(f"F2 in-situ bit mismatch on call {call_id}; diagnostic evidence saved")
    return actual
