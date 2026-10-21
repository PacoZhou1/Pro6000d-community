"""Opt-in, reversible vLLM 0.31.0 down-projection Stream-K experiment.

Mount this directory read-only and prepend it to PYTHONPATH in an experimental
vLLM container. Set PRO6000D_DOWN_STREAMK_ENABLE=1 and
PRO6000D_DOWN_STREAMK_LIBRARY to the absolute path of the separately mounted
libdown_streamk_fp8.so. With the enable flag absent, this file does nothing.
It changes no installed package or image. Use only after the standalone
numeric gate; this integration still needs model-level validation.

The import hook patches only CutlassFp8BlockScaledMMKernel once it loads.
Only A=[5,17408], B=[5120,17408], As=[5,136], Bs=[40,136], BF16-output
FP8-E4M3 calls use the experimental operator. All other calls use the exact
original method. In particular, varying M falls back without reaching the
experimental library. The custom op has a fake implementation for Dynamo and
CUDA graph capture. Each invocation owns a fresh output and scratch tensor;
there is no process-global GPU workspace shared among streams or captures.
"""

import ctypes
import hashlib
import importlib.abc
import importlib.machinery
import logging
import os
import runpy
import sys
import threading
from pathlib import Path


_TARGET = "vllm.model_executor.kernels.linear.scaled_mm.cutlass"
_SHA256 = "ecad960defeeffa1d6fb1faae4998650a07fd2404fe8acf569f5dc6f159529bd"
_LIBRARY = None  # Keep torch.library.Library alive for the process lifetime.
_CUDA_LIBRARY = None
_INSTALLED = False
_LAUNCH_LOCK = threading.Lock()
_LOGGED_PHASES = set()

# The fixed image has a small Debian sitecustomize that installs apport's
# exception hook. Python loads only one sitecustomize, so retain that behavior
# when this experimental directory is prepended to PYTHONPATH.
_BASE_SITE = Path("/usr/lib/python3.12/sitecustomize.py")
if _BASE_SITE.is_file() and _BASE_SITE.resolve() != Path(__file__).resolve():
    runpy.run_path(str(_BASE_SITE))


def _register_and_patch(module):
    global _LIBRARY, _CUDA_LIBRARY, _INSTALLED
    if _INSTALLED:
        return

    import torch
    from vllm.utils.torch_utils import direct_register_custom_op

    library_path = Path(os.environ["PRO6000D_DOWN_STREAMK_LIBRARY"])
    if not library_path.is_absolute() or not library_path.is_file():
        raise RuntimeError("down Stream-K experiment library must be an absolute mounted file")
    actual_hash = hashlib.sha256(library_path.read_bytes()).hexdigest()
    if actual_hash != _SHA256:
        raise RuntimeError(f"down Stream-K library SHA256 mismatch: {actual_hash}")
    cuda_library = ctypes.CDLL(str(library_path))
    fn = cuda_library.targeted_block_fp8
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong
    ]
    fn.restype = ctypes.c_int
    cuda_library.targeted_last_error.argtypes = []
    cuda_library.targeted_last_error.restype = ctypes.c_char_p
    workspace_mib = int(os.environ.get("PRO6000D_DOWN_STREAMK_WORKSPACE_MIB", "2"))
    if not 1 <= workspace_mib <= 128:
        raise RuntimeError("down Stream-K workspace MiB must be in [1,128]")

    def _down_streamk(A: torch.Tensor, B: torch.Tensor,
                      As: torch.Tensor, Bs: torch.Tensor) -> torch.Tensor:
        nonlocal workspace_mib
        # This assertion guards calls made directly through torch.ops as well
        # as calls selected by the patched vLLM class method.
        if (A.ndim != 2 or tuple(A.shape) != (5, 17408) or
                tuple(B.shape) != (5120, 17408) or
                tuple(As.shape) != (5, 136) or
                tuple(Bs.shape) != (40, 136) or
                A.dtype != torch.float8_e4m3fn or
                B.dtype != torch.float8_e4m3fn or
                As.dtype != torch.float32 or Bs.dtype != torch.float32 or
                A.device.type != "cuda" or B.device != A.device or
                As.device != A.device or Bs.device != A.device or
                not A.is_contiguous() or not B.is_contiguous() or
                not Bs.is_contiguous() or As.stride() != (1, 5)):
            raise RuntimeError("down Stream-K custom op received unsupported operands")
        with torch.cuda.device(A.device):
            if torch.cuda.get_device_capability(A.device) != (12, 0):
                raise RuntimeError("down Stream-K library requires SM120")
            phase = ("capture" if torch.cuda.is_current_stream_capturing()
                     else "eager")
            output = torch.empty((5, 5120), dtype=torch.bfloat16, device=A.device)
            stream = torch.cuda.current_stream(A.device)
            # The isolated adapter has process-global host launch parameters.
            # Serialize entry into it; GPU execution may overlap afterwards
            # because each call owns scratch and uses its own CUDA stream.
            with _LAUNCH_LOCK:
                while True:
                    # The C ABI checks CUTLASS get_workspace_size before
                    # launch and reports exactly "workspace too small". Grow
                    # only on that pre-launch failure; the first success is
                    # this invocation's real GEMM. Cache the needed size for
                    # later calls/captures instead of allocating 128 MiB per op.
                    scratch = torch.empty(workspace_mib * 1024 * 1024,
                                          dtype=torch.uint8, device=A.device)
                    code = fn(output.data_ptr(), A.data_ptr(), B.data_ptr(),
                              As.data_ptr(), Bs.data_ptr(), 5, 5120, 17408,
                              1, 1, scratch.data_ptr(), scratch.numel(),
                              stream.cuda_stream)
                    if not code:
                        break
                    message = cuda_library.targeted_last_error()
                    decoded = (message or b"unknown").decode(errors="replace")
                    if decoded != "workspace too small" or workspace_mib >= 128:
                        raise RuntimeError("down Stream-K launch failed: " + decoded)
                    workspace_mib = min(workspace_mib * 2, 128)
                # The CUDA launch is asynchronous. Associate every borrowed
                # allocation with its launch stream before releasing the lock.
                for tensor in (output, scratch, A, B, As, Bs):
                    tensor.record_stream(stream)
                if phase not in _LOGGED_PHASES:
                    logging.getLogger("pro6000d_down_streamk").warning(
                        "down Stream-K GEMM launched: phase=%s M=5 K=17408 "
                        "N=5120 workspace_mib=%d",
                        phase, workspace_mib,
                    )
                    _LOGGED_PHASES.add(phase)
            return output

    def _down_streamk_fake(A: torch.Tensor, B: torch.Tensor,
                           As: torch.Tensor, Bs: torch.Tensor) -> torch.Tensor:
        return torch.empty((A.shape[0], B.shape[0]), dtype=torch.bfloat16,
                           device=A.device)

    op_library = torch.library.Library("pro6000d_exp", "DEF")
    direct_register_custom_op("down_streamk", _down_streamk,
                              fake_impl=_down_streamk_fake,
                              target_lib=op_library, dispatch_key="CUDA")
    kernel_class = module.CutlassFp8BlockScaledMMKernel
    original = kernel_class.apply_block_scaled_mm

    def _patched(self, A, B, As, Bs):
        if (A.ndim == 2 and A.shape[0] == 5 and A.shape[1] == 17408 and
                B.shape == (5120, 17408) and
                As.shape == (5, 136) and Bs.shape == (40, 136) and
                self.config.out_dtype == torch.bfloat16 and
                A.dtype == torch.float8_e4m3fn and
                B.dtype == torch.float8_e4m3fn and
                As.dtype == torch.float32 and Bs.dtype == torch.float32 and
                A.device.type == "cuda" and B.device == A.device and
                As.device == A.device and Bs.device == A.device and
                A.is_contiguous() and B.is_contiguous() and Bs.is_contiguous() and
                As.stride() == (1, 5)):
            return torch.ops.pro6000d_exp.down_streamk(A, B, As, Bs)
        return original(self, A, B, As, Bs)

    kernel_class.apply_block_scaled_mm = _patched
    _LIBRARY = op_library
    _CUDA_LIBRARY = cuda_library
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


class _CutlassImportHook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _AfterImportLoader(spec.loader)
        return spec


if os.environ.get("PRO6000D_DOWN_STREAMK_ENABLE") == "1":
    # This runs before the service imports vLLM. Deferring registration until
    # the exact cutlass module loads avoids early CUDA/vLLM initialization.
    loaded = sys.modules.get(_TARGET)
    if loaded is not None and hasattr(loaded, "CutlassFp8BlockScaledMMKernel"):
        _register_and_patch(loaded)
    else:
        sys.meta_path.insert(0, _CutlassImportHook())
