"""Host-only loader/shape queries for the versioned candidate C ABI.

This module does not register a PyTorch op or enable candidate mode. The
isolated dispatcher uses it only after FX and numerical gates have passed.
"""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path


class CandidateABI:
    def __init__(self, library: str, expected_sha256: str):
        path = Path(library)
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError("candidate library must be an absolute file")
        if len(expected_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in expected_sha256.lower()
        ):
            raise RuntimeError("candidate library SHA256 must be 64 hex digits")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256.lower():
            raise RuntimeError("candidate library SHA256 mismatch")
        library_handle = ctypes.CDLL(str(path))
        version = library_handle.gateup_quant_abi_version
        version.argtypes = []
        version.restype = ctypes.c_int
        if version() != 2:
            raise RuntimeError("candidate C ABI must be version 2")

        self._temp_scale_bytes = library_handle.gateup_quant_temp_scale_bytes
        self._temp_scale_bytes.argtypes = [ctypes.c_int]
        self._temp_scale_bytes.restype = ctypes.c_size_t
        self._workspace_bytes = library_handle.gateup_quant_workspace_bytes
        self._workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
        self._workspace_bytes.restype = ctypes.c_size_t
        self._launch = library_handle.gateup_quant_fp8
        self._launch.argtypes = (
            [ctypes.c_void_p] * 8
            + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
               ctypes.c_size_t, ctypes.c_ulonglong]
        )
        self._launch.restype = ctypes.c_int
        self._last_error = library_handle.gateup_quant_last_error
        self._last_error.argtypes = []
        self._last_error.restype = ctypes.c_char_p
        self._handle = library_handle
        self.sha256 = actual

    def storage_bytes(self, m: int, swizzle: int) -> tuple[int, int]:
        """Query per-invocation native-scale and workspace sizes on the host.

        The zero/unsupported response is an error. Do not allocate a fixed
        128 MiB workspace per layer or infer it from the output shape.
        """
        if not isinstance(m, int) or not 256 < m <= 32768:
            raise RuntimeError("candidate M outside validated ABI range")
        if swizzle not in (1, 8):
            raise RuntimeError("candidate swizzle must be 1 or 8")
        scale_bytes = int(self._temp_scale_bytes(m))
        workspace_bytes = int(self._workspace_bytes(m, swizzle))
        if scale_bytes < m * 136 * 4 or workspace_bytes <= 0:
            raise RuntimeError("candidate storage query failed")
        return scale_bytes, workspace_bytes

    def launch(self, *args) -> None:
        """Launch bridge: caller must validate tensors and own buffers."""
        if len(args) != 13:
            raise RuntimeError("candidate launch requires exact ABI v2 arguments")
        code = self._launch(*args)
        if code:
            # The library error is local diagnostic text, not user input.
            message = self._last_error()
            safe = message.decode("utf-8", "replace")[:256] if message else ""
            raise RuntimeError(f"candidate launch failed ({code}): {safe}")
