"""SHA-pinned host ABI1 bridge for one isolated negative-result model test.

Only warp24n64_v3 at M16160 / AlongM / swizzle1 is accepted. This module
does not allocate CUDA buffers, register an op or change service state.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path

LIBRARY_SHA = "a6ae4155f65ec3b992ade3943902ccd88eea76b211dda5ceab26626245b7ae5d"
VARIANT, TARGET_M, RASTER, SWIZZLE = "warp24n64_v3", 16160, 1, 1


class CandidateABI:
    def __init__(self, library: str, expected_sha256: str):
        path = Path(library)
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError("mainloop library must be an absolute file")
        if expected_sha256 != LIBRARY_SHA:
            raise RuntimeError("mainloop experiment requires the locked v3 library SHA256")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != LIBRARY_SHA:
            raise RuntimeError("mainloop library SHA256 mismatch")
        handle = ctypes.CDLL(str(path))
        version = handle.sm120_scheduler_abi_version
        version.argtypes, version.restype = [], ctypes.c_int
        if version() != 1:
            raise RuntimeError("mainloop C ABI must be version 1")
        metadata_fn = handle.sm120_scheduler_metadata
        metadata_fn.argtypes, metadata_fn.restype = [], ctypes.c_char_p
        metadata = json.loads(metadata_fn().decode())
        required = {"abi_version": 1, "N": 34816, "K": 5120,
                    "cuda_arch": "sm_120a", "min_M": 257, "max_M": 32768,
                    "mainloop_variant": VARIANT,
                    "private_namespace": "pro6000d_mainloop_warp24n64_v3",
                    "same_m64_mainloop_type": True,
                    "m64_mma_atom_layout_mnk": [2, 4, 1],
                    "m64_mma_permutation_n": 64,
                    "m64_smem_copy_atom_b": "SM75_U32x4_LDSM_N",
                    "m64_smem_copy_b_bytes_per_thread": 16,
                    "fused_epilogue_tile": "64x256"}
        if any(metadata.get(k) != v for k, v in required.items()):
            raise RuntimeError("mainloop library variant/layout metadata mismatch")
        self._temp_scale_bytes = handle.sm120_fused_fp8_temp_scale_bytes
        self._temp_scale_bytes.argtypes = [ctypes.c_int]
        self._temp_scale_bytes.restype = ctypes.c_size_t
        self._workspace_bytes = handle.sm120_fused_fp8_workspace_bytes
        self._workspace_bytes.argtypes = [ctypes.c_int] * 3
        self._workspace_bytes.restype = ctypes.c_size_t
        self._launch = handle.sm120_fused_fp8
        self._launch.argtypes = ([ctypes.c_void_p] * 8 + [ctypes.c_int] * 3
                                + [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong])
        self._launch.restype = ctypes.c_int
        self._last_error = handle.sm120_scheduler_last_error
        self._last_error.argtypes, self._last_error.restype = [], ctypes.c_char_p
        self._handle, self.sha256, self.metadata = handle, actual, metadata

    @staticmethod
    def _guard(m: int, raster: int, swizzle: int) -> None:
        if (m, raster, swizzle) != (TARGET_M, RASTER, SWIZZLE):
            raise RuntimeError("mainloop runtime accepts only M16160 / AlongM / swizzle1")

    def storage_bytes(self, m: int, raster: int, swizzle: int) -> tuple[int, int]:
        self._guard(m, raster, swizzle)
        scale_bytes = int(self._temp_scale_bytes(m))
        workspace_bytes = int(self._workspace_bytes(m, raster, swizzle))
        if scale_bytes < m * 136 * 4 or workspace_bytes < 1:
            raise RuntimeError("mainloop storage query failed")
        return scale_bytes, workspace_bytes

    def launch(self, *args) -> None:
        if len(args) != 14:
            raise RuntimeError("mainloop launch requires exactly 14 ABI1 arguments")
        self._guard(*args[8:11])
        if not all(args[:8]) or not args[11] or args[12] < 1:
            raise RuntimeError("mainloop launch received null buffers or empty workspace")
        # Stream address zero is legal. No host setter or shared CUDA scratch.
        code = self._launch(*args)
        if code:
            message = self._last_error()
            safe = message.decode("utf-8", "replace")[:256] if message else ""
            raise RuntimeError(f"mainloop fused launch failed ({code}): {safe}")
