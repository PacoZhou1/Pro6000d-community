"""Probe-only adapter for the explicitly agreed paired-FP8 epilogue C ABI.

ABI agreed with kernel agent, 2026-10-14. Build/run not verified locally.
The backend prepares per-M temporary storage before graph capture or timing.
It never owns service lifecycle or overrides a registered vLLM operation.
"""
from __future__ import annotations

import ctypes
from pathlib import Path


class Backend:
    def __init__(self, library: Path, workspace, swizzle: int):
        import torch
        self.torch = torch
        self.workspace = workspace
        self.swizzle = swizzle
        self.temp = None
        self.temp_scales = None
        self.lib = ctypes.CDLL(str(library.resolve()))
        # A mismatched ctypes signature can corrupt host state before CUDA can
        # report an error. Refuse unversioned or later ABIs until their exact
        # argument order and temporary-storage query have been agreed.
        try:
            abi_version = self.lib.gateup_quant_abi_version
        except AttributeError as exc:
            raise RuntimeError("candidate library lacks gateup_quant_abi_version; ABI not locked") from exc
        abi_version.argtypes = []
        abi_version.restype = ctypes.c_int
        version = abi_version()
        if version != 2:
            raise RuntimeError(f"candidate ABI v{version} is not implemented by this v2 adapter")
        self.temp_scale_bytes_fn = self.lib.gateup_quant_temp_scale_bytes
        self.temp_scale_bytes_fn.argtypes = [ctypes.c_int]
        self.temp_scale_bytes_fn.restype = ctypes.c_size_t
        self.fn = self.lib.gateup_quant_fp8
        self.fn.argtypes = ([ctypes.c_void_p] * 8 + [ctypes.c_int] * 2 +
                            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong])
        self.fn.restype = ctypes.c_int
        self.lib.gateup_quant_last_error.argtypes = []
        self.lib.gateup_quant_last_error.restype = ctypes.c_char_p
        self.metadata = {
            "abi": "gateup_quant_fp8-v2-paired-FP8-and-native-scale-temps",
            "abi_version": version,
            "symbol": "gateup_quant_fp8",
            "launches": "GEMM fused SwiGLU/quant epilogue plus FP8 pack kernel",
            "eliminates_bf16_intermediate": True,
            "materializes_paired_fp8_intermediate": True,
            "materializes_native_scale_intermediate": True,
            "temporary_layout": "FP8 [M,34816], each N256 group has 128 valid bytes then 128 zeros",
            "output_layout": "FP8[M,17408] row major; scale FP32[M,136] stride(1,M)",
            "build_and_source_sha256": "record separately in kernel build manifest",
            "local_cuda_verification": False,
            "swizzle": swizzle,
        }

    @property
    def owned_buffers(self):
        return [] if self.temp is None else [self.temp, self.temp_scales]

    def supports(self, m):
        return m > 256

    def prepare(self, m):
        if not self.supports(m):
            raise RuntimeError(f"unsupported fused M={m}")
        self.temp = self.torch.empty((m, 34816), device=self.workspace.device,
                                     dtype=self.torch.float8_e4m3fn)
        temp_scale_bytes = self.temp_scale_bytes_fn(m)
        if temp_scale_bytes < m * 136 * 4:
            raise RuntimeError(f"candidate native scale storage query is too small: {temp_scale_bytes}")
        self.temp_scales = self.torch.empty(temp_scale_bytes, device=self.workspace.device,
                                            dtype=self.torch.uint8)
        self.metadata["temporary_bytes_current_M"] = self.temp.numel()
        self.metadata["native_scale_temporary_bytes_current_M"] = temp_scale_bytes

    def launch(self, q, scales, a, a_scales, packed_b, packed_bs):
        m = a.shape[0]
        if self.temp is None or self.temp.shape != (m, 34816):
            raise RuntimeError("prepare(M) must execute outside capture before launch")
        if (q.shape != (m, 17408) or not q.is_contiguous() or
                scales.shape != (m, 136) or scales.stride() != (1, m) or
                a.shape != (m, 5120) or not a.is_contiguous() or
                a_scales.shape != (m, 40) or a_scales.stride() != (1, m) or
                packed_b.shape != (34816, 5120) or not packed_b.is_contiguous() or
                packed_bs.shape != (272, 40) or not packed_bs.is_contiguous()):
            raise RuntimeError("candidate probe shape/stride mismatch")
        stream = self.torch.cuda.current_stream(a.device).cuda_stream
        code = self.fn(q.data_ptr(), scales.data_ptr(), self.temp.data_ptr(),
                       self.temp_scales.data_ptr(),
                       a.data_ptr(), packed_b.data_ptr(), a_scales.data_ptr(),
                       packed_bs.data_ptr(), m, self.swizzle, self.workspace.data_ptr(),
                       self.workspace.numel(), stream)
        if code:
            raise RuntimeError(f"candidate {code}: {self.lib.gateup_quant_last_error()!r}")


def create_backend(library: Path, workspace, swizzle: int):
    return Backend(library, workspace, swizzle)
