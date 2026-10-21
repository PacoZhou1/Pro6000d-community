"""Host-only, SHA-pinned ABI1 bridge for the official M128N128 BF16 GEMM.

The sole experimental setting is AlongM raster / swizzle 1 at M=16160.
This bridge never calls the fused FP8 or paired-weight entry points.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path


class CandidateABI:
    M = 16160
    RASTER_ALONG_M = 1
    SWIZZLE = 1

    def __init__(self, library: str, expected_sha256: str):
        path = Path(library)
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError("official-tuning library must be an absolute file")
        if len(expected_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in expected_sha256.lower()
        ):
            raise RuntimeError("official-tuning library SHA256 must be 64 hex digits")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256.lower():
            raise RuntimeError("official-tuning library SHA256 mismatch")
        handle = ctypes.CDLL(str(path))
        version = handle.sm120_scheduler_abi_version
        version.argtypes = []
        version.restype = ctypes.c_int
        if version() != 1:
            raise RuntimeError("official-tuning C ABI must be version 1")
        metadata_fn = handle.sm120_scheduler_metadata
        metadata_fn.argtypes = []
        metadata_fn.restype = ctypes.c_char_p
        metadata = json.loads(metadata_fn().decode())
        if (metadata.get("abi_version") != 1 or metadata.get("N") != 34816
            or metadata.get("K") != 5120 or metadata.get("cuda_arch") != "sm_120a"):
            raise RuntimeError("official-tuning library metadata mismatch")
        query = handle.sm120_official_bf16_workspace_bytes
        query.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        query.restype = ctypes.c_size_t
        launch = handle.sm120_official_bf16
        launch.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 3
                           + [ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.c_ulonglong])
        launch.restype = ctypes.c_int
        last_error = handle.sm120_scheduler_last_error
        last_error.argtypes = []
        last_error.restype = ctypes.c_char_p
        self._handle = handle
        self._query = query
        self._launch = launch
        self._last_error = last_error
        self.sha256 = actual
        self.metadata = metadata

    def workspace_bytes(self, m: int) -> int:
        if m != self.M:
            raise RuntimeError("official-tuning runtime accepts only M=16160")
        size = int(self._query(m, self.RASTER_ALONG_M, self.SWIZZLE))
        if size < 1:
            raise RuntimeError("official-tuning workspace query failed")
        return size

    def launch(self, out: int, a: int, b: int, asc: int, bs: int, m: int,
               workspace: int, workspace_bytes: int, stream: int) -> None:
        if m != self.M or workspace_bytes < 1:
            raise RuntimeError("official-tuning launch shape/workspace guard failed")
        code = self._launch(out, a, b, asc, bs, m,
                            self.RASTER_ALONG_M, self.SWIZZLE,
                            workspace, workspace_bytes, stream)
        if code:
            error = self._last_error()
            safe = error.decode("utf-8", "replace")[:256] if error else ""
            raise RuntimeError(f"official-tuning GEMM launch failed ({code}): {safe}")
