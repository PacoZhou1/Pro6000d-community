#!/usr/bin/env python3
"""Numerical-only isolation of Cooperative paired-visitor and SF128 layout.

Run only in the fixed GPU image when the root experiment scheduler gives the
GPU to this probe. Both diagnostic identity libraries have ABI 1002 and must
never be loaded by the model adapter (which requires ABI 2). No timings are
emitted. The paired FP8 temporary and *both* native scale halves are checked.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
import traceback
from pathlib import Path

import torch

from pipeline_probe import (DiagnosticGemm, K, N, GROUP, make_activation,
                            make_weight, packed_weights)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bit_compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("bit comparison shape/dtype mismatch")
    bit_dtype = torch.uint8 if actual.dtype == torch.float8_e4m3fn else torch.int32
    different = actual.contiguous().view(bit_dtype) != expected.contiguous().view(bit_dtype)
    positions = different.nonzero()
    return {"mismatches": int(different.sum().item()), "elements": actual.numel(),
            "first_coordinates": positions[:8].tolist()}


class IdentityVisitor:
    def __init__(self, library: Path, m: int, swizzle: int):
        self.lib = ctypes.CDLL(str(library.resolve()))
        abi = self.lib.gateup_quant_abi_version
        abi.argtypes, abi.restype = [], ctypes.c_int
        if abi() != 1002:
            raise RuntimeError(f"{library}: identity diagnostic ABI 1002 required")
        self.scale_bytes = self.lib.gateup_quant_temp_scale_bytes
        self.scale_bytes.argtypes, self.scale_bytes.restype = [ctypes.c_int], ctypes.c_size_t
        workspace_bytes = self.lib.gateup_quant_workspace_bytes
        workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
        workspace_bytes.restype = ctypes.c_size_t
        needed = int(workspace_bytes(m, swizzle))
        if needed <= 0:
            raise RuntimeError("diagnostic workspace query failed")
        self.workspace = torch.empty(needed, device="cuda", dtype=torch.uint8)
        self.temp = torch.empty((m, N), device="cuda", dtype=torch.float8_e4m3fn)
        self.native_scales = torch.empty(int(self.scale_bytes(m)), device="cuda", dtype=torch.uint8)
        self.paired_scales = torch.empty_strided((m, N // GROUP), (1, m), device="cuda", dtype=torch.float32)
        self.output = torch.empty((m, N // 2), device="cuda", dtype=torch.float8_e4m3fn)
        self.output_scales = torch.empty_strided((m, N // (2 * GROUP)), (1, m), device="cuda", dtype=torch.float32)
        self.launch_fn = self.lib.gateup_quant_fp8
        self.launch_fn.argtypes = ([ctypes.c_void_p] * 8 + [ctypes.c_int] * 2 +
                                   [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong])
        self.launch_fn.restype = ctypes.c_int
        self.copy_fn = self.lib.gateup_quant_diagnostic_copy_scales
        self.copy_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong]
        self.copy_fn.restype = ctypes.c_int
        self.lib.gateup_quant_last_error.argtypes = []
        self.lib.gateup_quant_last_error.restype = ctypes.c_char_p
        self.swizzle = swizzle

    def run(self, a, asc, packed_b, packed_bs):
        m = a.shape[0]
        stream = torch.cuda.current_stream().cuda_stream
        status = self.launch_fn(
            self.output.data_ptr(), self.output_scales.data_ptr(),
            self.temp.data_ptr(), self.native_scales.data_ptr(),
            a.data_ptr(), packed_b.data_ptr(), asc.data_ptr(), packed_bs.data_ptr(),
            m, self.swizzle, self.workspace.data_ptr(), self.workspace.numel(), stream)
        if status:
            raise RuntimeError(f"identity visitor launch: {self.lib.gateup_quant_last_error()!r}")
        status = self.copy_fn(self.paired_scales.data_ptr(), self.native_scales.data_ptr(), m, stream)
        if status:
            raise RuntimeError(f"native scale extraction: {self.lib.gateup_quant_last_error()!r}")
        torch.cuda.synchronize()


def reference_quant(paired_bf16: torch.Tensor):
    # This is a secondary mathematical reference. The decisive diagnostic is
    # the exact pingpong-vs-Cooperative comparison below, which uses the same
    # CUDA conversion/reduction code and differs only in scheduling/layout.
    m = paired_bf16.shape[0]
    grouped = paired_bf16.float().reshape(m, N // GROUP, GROUP)
    scale = torch.maximum(grouped.abs().amax(dim=2) / 448.0,
                          torch.full((m, N // GROUP), 1.0 / (448.0 * 512.0), device="cuda"))
    quant = (grouped / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quant.reshape(m, N), scale


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coop-bf16-library", required=True, type=Path)
    p.add_argument("--coop-identity-library", required=True, type=Path)
    p.add_argument("--pingpong-identity-library", required=True, type=Path)
    p.add_argument("--m", type=int, default=257)
    p.add_argument("--swizzle", type=int, choices=(1, 8), default=8)
    p.add_argument("--workspace-mib", type=int, default=64)
    args = p.parse_args()
    if not 256 < args.m <= 32768:
        p.error("diagnostic requires 256 < M <= 32768")
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    b, bs = make_weight(torch, N, K, 20261014)
    pb, pbs = packed_weights(torch, b, bs)
    a, asc = make_activation(torch, per_token_group_quant_fp8, args.m, 20261015)
    bf16_backend = DiagnosticGemm(torch, args.coop_bf16_library, args.workspace_mib * 1024**2, args.swizzle)
    paired_bf16 = torch.empty((args.m, N), device="cuda", dtype=torch.bfloat16)
    bf16_backend.launch(paired_bf16, a, asc, pb, pbs)
    pingpong = IdentityVisitor(args.pingpong_identity_library, args.m, args.swizzle)
    coop = IdentityVisitor(args.coop_identity_library, args.m, args.swizzle)
    pingpong.run(a, asc, pb, pbs)
    coop.run(a, asc, pb, pbs)
    ref_q, ref_s = reference_quant(paired_bf16)
    checks = {
        "coop_vs_pingpong_fp8": bit_compare(coop.temp, pingpong.temp),
        "coop_vs_pingpong_scales": bit_compare(coop.paired_scales, pingpong.paired_scales),
        "pingpong_vs_math_fp8": bit_compare(pingpong.temp, ref_q),
        "pingpong_vs_math_scales": bit_compare(pingpong.paired_scales, ref_s),
        "coop_vs_math_fp8": bit_compare(coop.temp, ref_q),
        "coop_vs_math_scales": bit_compare(coop.paired_scales, ref_s),
    }
    output = {"diagnostic": "identity-visitor-SF128-layout", "M": args.m,
              "libraries_sha256": {k: digest(getattr(args, k)) for k in
                                    ("coop_bf16_library", "coop_identity_library", "pingpong_identity_library")},
              "checks": checks,
              "interpretation": "Pingpong-vs-Cooperative compares the same visitor source; math reference is secondary."
              }
    print(json.dumps(output, sort_keys=True), flush=True)
    # A mismatch is the finding this experiment is designed to reveal, not a
    # process failure. Setup/launch exceptions return nonzero below.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({"diagnostic": "identity-visitor-SF128-layout", "error": str(exc),
                          "pass": False}), flush=True)
        raise SystemExit(1)
