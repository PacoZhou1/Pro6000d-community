"""Opt-in vLLM 0.31.0 GEMM attribution for an isolated eager profiler run.

Runtime installation without changing installed vLLM files:

  mkdir -p /tmp/gemm-attribution-site
  cp attribution_patch.py /tmp/gemm-attribution-site/sitecustomize.py
  GEMM_ATTRIBUTION_ENABLE=1 \
  GEMM_ATTRIBUTION_OUTPUT_DIR=/tmp/gemm-attribution-output \
  PYTHONPATH=/tmp/gemm-attribution-site:$PYTHONPATH ... vllm serve ... --enforce-eager

The same PYTHONPATH/environment must reach the engine worker process. Inspect
per-PID JSONL for ``patch_installed`` and ``gemm_signature`` rows before using
the trace. Nsight must include CUDA and NVTX traces. The ``GEMM_ATTRIB:`` NVTX
range encloses the actual CUTLASS FP8 GEMM or an unquantized GEMM dispatcher;
associate GPU kernels via CUDA API correlation ID and same-thread NVTX range,
not by wall-time overlap or kernel name alone. With torch.compile/CUDA Graph
replay, Python hooks may execute only during warmup/capture and are NOT evidence
that every replayed kernel came from that Python call. Use --enforce-eager for
an exact kernel-to-module attribution run; retain separate normal-mode runs
for performance conclusions.

This file reads tensor metadata only. It never calls .item(), transfers tensor
data to CPU, synchronizes CUDA, modifies weights, or logs prompts/credentials.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import json
import os
import re
import sys
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any


ENABLED = os.environ.get("GEMM_ATTRIBUTION_ENABLE") == "1"
OUTPUT_DIR = os.environ.get("GEMM_ATTRIBUTION_OUTPUT_DIR", "")
_installed = False
_seen: set[str] = set()
_fp8_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "gemm_attribution_fp8_context", default=None
)
_targets = {
    "vllm.model_executor.layers.quantization.fp8": "fp8",
    "vllm.model_executor.layers.linear": "linear",
    "vllm.model_executor.layers.vocab_parallel_embedding": "embedding",
    "vllm.model_executor.layers.logits_processor": "logits",
    "vllm.model_executor.kernels.linear.scaled_mm.cutlass": "cutlass",
}


def _emit(record: dict[str, Any], *, once: bool = False) -> None:
    if not OUTPUT_DIR:
        return
    record = {"schema": 1, "pid": os.getpid(), **record}
    if once:
        key = json.dumps(record, sort_keys=True, separators=(",", ":"))
        if key in _seen or len(_seen) >= 20000:
            return
        _seen.add(key)
    record["time_unix_ns"] = time.time_ns()
    destination = Path(OUTPUT_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"gemm-attribution.{os.getpid()}.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    finally:
        os.close(fd)


def _tensor(tensor: Any) -> dict[str, Any] | None:
    # Attribute access is metadata-only. Do not inspect values or repr(tensor).
    if tensor is None or not hasattr(tensor, "shape") or not hasattr(tensor, "stride"):
        return None
    return {
        "shape": [int(v) for v in tensor.shape],
        "stride": [int(v) for v in tensor.stride()],
        "dtype": str(tensor.dtype),
        "device_type": tensor.device.type,
    }


def _dimensions(x: Any, weight: Any) -> dict[str, int | None]:
    if x is None or weight is None or x.ndim < 1 or weight.ndim != 2:
        return {"M": None, "N": None, "K": None}
    k = int(x.shape[-1])
    m = int(x.numel() // k) if k else 0
    # Fixed vLLM linear and block-FP8 layer weights are [N,K]. Preserve
    # physical shape separately so this inference is independently checkable.
    return {"M": m, "N": int(weight.shape[0]), "K": k}


def _layer_details(layer: Any, x: Any, kind: str, bias: Any = None) -> dict[str, Any]:
    weight = getattr(layer, "weight", None)
    details = {
        "kind": kind,
        "module_prefix": str(getattr(layer, "prefix", "<missing>")),
        "module_class": type(layer).__name__,
        "quant_method_class": type(getattr(layer, "quant_method", None)).__name__,
        "input": _tensor(x),
        "weight": _tensor(weight),
        "bias": _tensor(bias),
        "weight_layout_assumption": "[N,K] from fixed vLLM 0.31.0 layer implementation",
        **_dimensions(x, weight),
    }
    for name in ("weight_scale", "weight_scale_inv", "input_scale"):
        value = getattr(layer, name, None)
        if value is not None:
            details[name] = _tensor(value)
    method = getattr(layer, "quant_method", None)
    backend = getattr(method, "fp8_linear", None)
    if backend is not None:
        details["fp8_backend_class"] = type(backend).__name__
    return details


def _compiling() -> bool:
    # During Dynamo tracing, skip all instrumentation so logging/NVTX cannot
    # force graph breaks. This is why normal CUDA Graph replay is not covered.
    import torch

    return bool(getattr(torch.compiler, "is_compiling", lambda: False)())


def _nvtx_label(details: dict[str, Any]) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.:-]", "_", details.get("module_prefix", "unknown"))
    label = (
        f"GEMM_ATTRIB:{details['kind']}:{prefix}:"
        f"M{details.get('M')}:N{details.get('N')}:K{details.get('K')}"
    )
    return label[:240]


def _invoke(original, instance, args, kwargs, details, *, nvtx: bool):
    if not nvtx:
        result = original(instance, *args, **kwargs)
    else:
        import torch

        torch.cuda.nvtx.range_push(_nvtx_label(details))
        try:
            result = original(instance, *args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()
    _emit({"event": "gemm_signature", **details, "output": _tensor(result)}, once=True)
    return result


def _patch_apply(cls, kind: str, *, nvtx: bool) -> None:
    original = cls.apply
    if getattr(original, "_gemm_attribution_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(self, layer, x, bias=None):
        if _compiling():
            return original(self, layer, x, bias)
        details = _layer_details(layer, x, kind, bias)
        args = (layer, x, bias)
        if kind == "fp8_linear":
            token = _fp8_context.set(details)
            try:
                return _invoke(original, self, args, {}, details, nvtx=False)
            finally:
                _fp8_context.reset(token)
        return _invoke(original, self, args, {}, details, nvtx=nvtx)

    wrapped._gemm_attribution_wrapped = True
    cls.apply = wrapped
    _emit({"event": "patch_installed", "class": cls.__module__ + "." + cls.__name__,
           "method": "apply"}, once=True)


def _patch_cutlass(cls) -> None:
    original = cls.apply_block_scaled_mm
    if getattr(original, "_gemm_attribution_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(self, A, B, As, Bs):
        if _compiling():
            return original(self, A, B, As, Bs)
        parent = _fp8_context.get()
        details = {
            "kind": "fp8_block_cutlass_gemm",
            "module_prefix": parent["module_prefix"] if parent else "<unlinked>",
            "parent_module_class": parent["module_class"] if parent else None,
            "M": int(A.shape[0]), "N": int(B.shape[0]), "K": int(A.shape[1]),
            "A": _tensor(A), "B": _tensor(B), "A_scale": _tensor(As),
            "B_scale": _tensor(Bs),
            "B_layout": "[N,K]; fixed CUTLASS path passes B.T to cutlass_scaled_mm",
        }
        return _invoke(original, self, (A, B, As, Bs), {}, details, nvtx=True)

    wrapped._gemm_attribution_wrapped = True
    cls.apply_block_scaled_mm = wrapped
    _emit({"event": "patch_installed", "class": cls.__module__ + "." + cls.__name__,
           "method": "apply_block_scaled_mm"}, once=True)


def _patch_logits(cls) -> None:
    original = cls._apply_head
    if getattr(original, "_gemm_attribution_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(self, lm_head, hidden_states, embedding_bias):
        if _compiling():
            return original(self, lm_head, hidden_states, embedding_bias)
        # The ordinary path is covered by UnquantizedEmbedding/LinearMethod.
        # The fixed source bypasses those methods when head_dtype differs.
        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
            return original(self, lm_head, hidden_states, embedding_bias)
        details = _layer_details(lm_head, hidden_states, "lm_head_dtype_override",
                                 embedding_bias)
        details["head_dtype"] = str(self.head_dtype)
        return _invoke(original, self, (lm_head, hidden_states, embedding_bias),
                       {}, details, nvtx=True)

    wrapped._gemm_attribution_wrapped = True
    cls._apply_head = wrapped
    _emit({"event": "patch_installed", "class": cls.__module__ + "." + cls.__name__,
           "method": "_apply_head"}, once=True)


def _patch_module(module, tag: str) -> None:
    if tag == "fp8":
        _patch_apply(module.Fp8LinearMethod, "fp8_linear", nvtx=False)
    elif tag == "linear":
        _patch_apply(module.UnquantizedLinearMethod, "unquantized_linear", nvtx=True)
    elif tag == "embedding":
        _patch_apply(module.UnquantizedEmbeddingMethod, "unquantized_embedding_gemm", nvtx=True)
    elif tag == "logits":
        _patch_logits(module.LogitsProcessor)
    elif tag == "cutlass":
        _patch_cutlass(module.CutlassFp8BlockScaledMMKernel)


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, loader, tag: str):
        self.loader = loader
        self.tag = tag

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        _patch_module(module, self.tag)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        tag = _targets.get(fullname)
        if tag is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _PatchLoader(spec.loader, tag)
        return spec


def install() -> None:
    """Install delayed, process-local hooks before importing vLLM modules."""
    global _installed
    if _installed or not ENABLED:
        return
    if not OUTPUT_DIR:
        raise RuntimeError("GEMM_ATTRIBUTION_OUTPUT_DIR must be set")
    _installed = True
    sys.meta_path.insert(0, _PatchFinder())
    for name, tag in _targets.items():
        if name in sys.modules:
            _patch_module(sys.modules[name], tag)


if __name__ == "sitecustomize":
    install()
