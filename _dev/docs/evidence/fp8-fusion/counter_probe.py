#!/usr/bin/env python3
"""Two-launch NCU counter window for fixed FP8 SwiGLU block quant.

Run only during a reserved GPU window in the fixed vLLM 0.31.0 SM120 image.
An example NCU invocation uses --profile-from-start off --launch-count 2.
The script first validates the V2 library against the official op on the
same synthetic input, then opens cudaProfilerStart/Stop around exactly one
official kernel launch followed by one candidate kernel launch. This is a
hardware-counter probe, not a model-level performance or quality test.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path


M = 16160
N = 17408
GROUP = 128
GROUPS = N // GROUP


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cuda_status_ok(value) -> bool:
    # PyTorch's cudart binding may return an enum or a one-element tuple.
    code = value[0] if isinstance(value, tuple) else value
    return int(code) == 0


def run(library_path: Path, expected_sha: str) -> dict:
    import torch
    from vllm import _custom_ops as _ops  # noqa: F401 - registers fixed _C op

    if not library_path.is_file():
        raise FileNotFoundError(library_path)
    actual_sha = sha256(library_path)
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError("candidate library SHA256 mismatch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("reserved SM120 GPU required")
    torch.manual_seed(20261014)

    cdll = ctypes.CDLL(str(library_path.resolve()))
    launch = cdll.pro6000d_fp8_silu_quant_launch
    launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [
        ctypes.c_uint64]
    launch.restype = ctypes.c_int
    cdll.pro6000d_fp8_silu_quant_last_error.restype = ctypes.c_char_p

    x = torch.empty((M, 2 * N), dtype=torch.bfloat16, device="cuda")
    x.normal_(mean=0.0, std=0.5)
    q_official = torch.empty((M, N), dtype=torch.float8_e4m3fn, device="cuda")
    q_candidate = torch.empty_like(q_official)
    s_official = torch.empty((GROUPS, M), dtype=torch.float32,
                             device="cuda").t()
    s_candidate = torch.empty((GROUPS, M), dtype=torch.float32,
                              device="cuda").t()
    assert s_official.stride() == s_candidate.stride() == (1, M)
    stream = torch.cuda.current_stream(x.device)

    def official_launch():
        torch.ops._C.silu_and_mul_per_block_quant(
            q_official, x, s_official, GROUP, None, True)

    def candidate_launch():
        status = launch(q_candidate.data_ptr(), s_candidate.data_ptr(),
                        x.data_ptr(), M, N, s_candidate.stride(0),
                        s_candidate.stride(1), stream.cuda_stream)
        if status:
            error = cdll.pro6000d_fp8_silu_quant_last_error()
            raise RuntimeError(f"candidate launch failed: {status} {error!r}")

    for _ in range(3):
        official_launch()
        candidate_launch()
    stream.synchronize()
    byte_mismatches = int((q_official.view(torch.uint8) !=
                           q_candidate.view(torch.uint8)).sum().item())
    scale_bit_mismatches = int((s_official.contiguous().view(torch.int32) !=
                                s_candidate.contiguous().view(torch.int32)).sum().item())
    if byte_mismatches or scale_bit_mismatches:
        raise RuntimeError(
            f"numeric gate failed: FP8 byte mismatches={byte_mismatches}, "
            f"FP32 scale-bit mismatches={scale_bit_mismatches}")

    # No allocations, reductions, synchronization kernels or extra warmups
    # occur between profilerStart and profilerStop. Both launches use the
    # current stream; its synchronization waits for exactly these launches.
    stream.synchronize()
    cudart = torch.cuda.cudart()
    if not cuda_status_ok(cudart.cudaProfilerStart()):
        raise RuntimeError("cudaProfilerStart failed")
    try:
        official_launch()
        candidate_launch()
        stream.synchronize()
    finally:
        if not cuda_status_ok(cudart.cudaProfilerStop()):
            raise RuntimeError("cudaProfilerStop failed")

    return {
        "pass": True,
        "library": str(library_path.resolve()),
        "library_sha256": actual_sha,
        "fixed_vllm_commit": "db9527a46873454610df6dbedf79a36d6bf1a7f6",
        "shape": {"M": M, "input_width": 2 * N, "output_width": N,
                  "group_size": GROUP, "groups": GROUPS},
        "dtype": {"input": "bfloat16", "output": "float8_e4m3fn",
                  "scales": "float32"},
        "scale_stride": list(s_candidate.stride()),
        "synthetic_input": True,
        "numeric_gate": {"fp8_byte_mismatches": byte_mismatches,
                         "fp32_scale_bit_mismatches": scale_bit_mismatches},
        "profiled_launch_order": ["official", "candidate"],
        "profiled_launch_count": 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sha256", default="")
    args = parser.parse_args()
    try:
        result = run(args.library, args.sha256)
    except Exception as exc:
        result = {"pass": False, "error_type": type(exc).__name__,
                  "error": str(exc)}
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
