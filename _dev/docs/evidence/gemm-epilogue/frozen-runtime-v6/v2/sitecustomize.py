"""Opt-in vLLM 0.31.0 SM120 FP8 activation-quant experiment.

Mount this directory read-only first on PYTHONPATH and set:
  PRO6000D_FP8_FUSION_ENABLE=1
  PRO6000D_FP8_FUSION_LIBRARY=/absolute/path/libfp8_silu_quant.so
  PRO6000D_FP8_FUSION_LIBRARY_SHA256=<build manifest library_sha256>

This import hook changes only the fixed activation-quant fusion pass's FP8
dynamic-128 replacement op. Its replacement is opaque to Dynamo: runtime M
and layout selection happen inside the CUDA implementation, so tracing a
large or small M cannot eliminate the experimental entry. Non-target calls
execute the original _C op with the same arguments and output allocations.
Do not use in a normal service before isolated numeric and graph smoke gates.
"""

import ctypes
import hashlib
import importlib.abc
import importlib.machinery
import logging
import os
import runpy
import sys
from pathlib import Path


_TARGET = "vllm.compilation.passes.fusion.act_quant_fusion"
_LIBRARY = None
_CDLL = None
_INSTALLED = False
_LOGGED = set()

# Preserve the base image's ordinary sitecustomize behavior.
_BASE_SITE = Path("/usr/lib/python3.12/sitecustomize.py")
if _BASE_SITE.is_file() and _BASE_SITE.resolve() != Path(__file__).resolve():
    runpy.run_path(str(_BASE_SITE))


def _register_and_patch(module):
    global _LIBRARY, _CDLL, _INSTALLED
    if _INSTALLED:
        return

    import torch

    library_path = Path(os.environ["PRO6000D_FP8_FUSION_LIBRARY"])
    expected_hash = os.environ["PRO6000D_FP8_FUSION_LIBRARY_SHA256"]
    if not library_path.is_absolute() or not library_path.is_file():
        raise RuntimeError("FP8 fusion library must be an absolute mounted file")
    actual_hash = hashlib.sha256(library_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError("FP8 fusion library SHA256 mismatch")
    cdll = ctypes.CDLL(str(library_path))
    launch = cdll.pro6000d_fp8_silu_quant_launch
    launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [
        ctypes.c_uint64]
    launch.restype = ctypes.c_int
    cdll.pro6000d_fp8_silu_quant_last_error.restype = ctypes.c_char_p

    def dispatch(out: torch.Tensor, input: torch.Tensor,
                 scales: torch.Tensor, group_size: int,
                 scale_ub: torch.Tensor | None,
                 is_scale_transposed: bool) -> None:
        # All data-dependent and dynamic-M decisions stay behind this opaque
        # op. A distinct output/scales allocation is provided by the fixed
        # fusion replacement and owned by each invocation/CUDA graph.
        m = input.shape[0]
        candidate = (
            input.device.type == "cuda" and input.ndim == 2 and
            input.shape[1] == 34816 and input.dtype == torch.bfloat16 and
            input.is_contiguous() and m >= 256 and
            out.shape == (m, 17408) and out.dtype == torch.float8_e4m3fn and
            out.device == input.device and out.is_contiguous() and
            scales.shape == (m, 136) and scales.dtype == torch.float32 and
            scales.device == input.device and scales.stride() == (1, m) and
            group_size == 128 and scale_ub is None and is_scale_transposed
        )
        if candidate:
            with torch.cuda.device(input.device):
                candidate = torch.cuda.get_device_capability(input.device) == (12, 0)
        if not candidate:
            torch.ops._C.silu_and_mul_per_block_quant(
                out, input, scales, group_size, scale_ub, is_scale_transposed)
            return

        with torch.cuda.device(input.device):
            stream = torch.cuda.current_stream(input.device)
            phase = ("capture" if torch.cuda.is_current_stream_capturing()
                     else "eager")
            status = launch(out.data_ptr(), scales.data_ptr(), input.data_ptr(),
                            m, 17408, scales.stride(0), scales.stride(1),
                            stream.cuda_stream)
            if status:
                message = cdll.pro6000d_fp8_silu_quant_last_error()
                raise RuntimeError(f"FP8 fusion launch failed: {status} {message!r}")
            for tensor in (input, out, scales):
                tensor.record_stream(stream)
            if phase not in _LOGGED:
                logging.getLogger("pro6000d_fp8_fusion").warning(
                    "FP8 fusion candidate launched: phase=%s M=%d N=17408",
                    phase, m)
                _LOGGED.add(phase)

    def fake(out: torch.Tensor, input: torch.Tensor,
             scales: torch.Tensor, group_size: int,
             scale_ub: torch.Tensor | None,
             is_scale_transposed: bool) -> None:
        return None

    op_library = torch.library.Library("pro6000d_fp8", "DEF")
    op_library.define(
        "silu_and_mul_per_block_quant(Tensor(a!) out, Tensor input, "
        "Tensor(b!) scales, int group_size, Tensor? scale_ub, "
        "bool is_scale_transposed) -> ()")
    op_library.impl("silu_and_mul_per_block_quant", dispatch, "CUDA")
    op_library.impl("silu_and_mul_per_block_quant", fake, "Meta")
    module.FUSED_OPS[module.kFp8Dynamic128Sym] = (
        torch.ops.pro6000d_fp8.silu_and_mul_per_block_quant.default)
    _LIBRARY = op_library
    _CDLL = cdll
    _INSTALLED = True


class _AfterImportLoader(importlib.abc.Loader):
    def __init__(self, delegate):
        self._delegate = delegate

    def create_module(self, spec):
        create = getattr(self._delegate, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self._delegate.exec_module(module)
        _register_and_patch(module)


class _FusionImportHook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _AfterImportLoader(spec.loader)
        return spec


if os.environ.get("PRO6000D_FP8_FUSION_ENABLE") == "1":
    loaded = sys.modules.get(_TARGET)
    if loaded is not None and hasattr(loaded, "FUSED_OPS"):
        _register_and_patch(loaded)
    else:
        sys.meta_path.insert(0, _FusionImportHook())
