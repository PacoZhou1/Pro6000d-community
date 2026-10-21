#!/usr/bin/env python3
"""Exact full-pipeline gate and paired SM120 GEMM→SwiGLU/blockquant timings.

See probe-design.md for the candidate adapter contract. No CUDA ABI is guessed.
--baseline-only develops/checks the reference path and emits no timings.
--ncu opens one complete-pipeline window per arm, after all numerical gates.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib.util
import json
import math
import random
import statistics
import time
import traceback
from pathlib import Path

K, N, H, GROUP = 5120, 34816, 17408, 128
DEFAULT_M = (16160, 2741, 2786, 256, 257, 1, 5)
BASELINE_SHA = "f7abfcd8c04ad9a93c924364beadd305f773b1f909bca013f2838d109f17b7b2"
QUANT_SHA = "087b7caa92c4742ece3875b6dd1421e25360b834ce4106f7fa166f90bc933b3f"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(kind, value):
    print(json.dumps({kind: value}, allow_nan=False, sort_keys=True), flush=True)


def check_library(path, expected):
    actual = sha(path)
    if expected and actual != expected:
        raise RuntimeError(f"{path}: SHA256 mismatch: {actual}")
    return actual


class Baseline:
    """Known previous-round C ABIs; each instance owns separate workspace."""
    def __init__(self, torch, gemm_path, quant_path, workspace_bytes, swizzle):
        self.torch = torch
        self.workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
        self.gemm_lib = ctypes.CDLL(str(gemm_path.resolve()))
        self.gemm_fn = self.gemm_lib.targeted_block_fp8
        self.gemm_fn.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 +
                                [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong])
        self.gemm_fn.restype = ctypes.c_int
        self.gemm_lib.targeted_last_error.restype = ctypes.c_char_p
        self.quant_lib = ctypes.CDLL(str(quant_path.resolve()))
        self.quant_fn = self.quant_lib.pro6000d_fp8_silu_quant_launch
        self.quant_fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_uint64]
        self.quant_fn.restype = ctypes.c_int
        self.quant_lib.pro6000d_fp8_silu_quant_last_error.restype = ctypes.c_char_p
        self.swizzle = swizzle

    def gemm(self, output, a, scales, b, bs):
        code = self.gemm_fn(output.data_ptr(), a.data_ptr(), b.data_ptr(),
                            scales.data_ptr(), bs.data_ptr(), a.shape[0], b.shape[0],
                            a.shape[1], 0, self.swizzle, self.workspace.data_ptr(),
                            self.workspace.numel(), self.torch.cuda.current_stream().cuda_stream)
        if code:
            raise RuntimeError(f"baseline GEMM {code}: {self.gemm_lib.targeted_last_error()!r}")

    def quant(self, x, q, scales):
        code = self.quant_fn(q.data_ptr(), scales.data_ptr(), x.data_ptr(), x.shape[0], H,
                             scales.stride(0), scales.stride(1),
                             self.torch.cuda.current_stream().cuda_stream)
        if code:
            raise RuntimeError(f"V2 quant {code}: {self.quant_lib.pro6000d_fp8_silu_quant_last_error()!r}")


class DiagnosticGemm:
    """N256/M64 ordinary BF16 epilogue: triangulation, not fused internals."""
    def __init__(self, torch, library, workspace_bytes, swizzle):
        self.torch, self.swizzle = torch, swizzle
        self.workspace = torch.empty(workspace_bytes, device="cuda", dtype=torch.uint8)
        self.lib = ctypes.CDLL(str(library.resolve()))
        self.fn = self.lib.n256_block_fp8
        self.fn.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 +
                            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong])
        self.fn.restype = ctypes.c_int
        self.lib.n256_last_error.argtypes = []
        self.lib.n256_last_error.restype = ctypes.c_char_p

    def launch(self, paired_bf16, a, scales, packed_b, packed_bs):
        code = self.fn(paired_bf16.data_ptr(), a.data_ptr(), packed_b.data_ptr(),
                       scales.data_ptr(), packed_bs.data_ptr(), a.shape[0], N, K,
                       self.swizzle, self.workspace.data_ptr(), self.workspace.numel(),
                       self.torch.cuda.current_stream().cuda_stream)
        if code:
            raise RuntimeError(f"N256 diagnostic GEMM {code}: {self.lib.n256_last_error()!r}")


def load_candidate(args, workspace, swizzle):
    spec = importlib.util.spec_from_file_location("epilogue_candidate_adapter", args.candidate_adapter)
    if not spec or not spec.loader:
        raise RuntimeError("cannot load candidate adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.create_backend(args.candidate_library, workspace, swizzle)
    for name in ("supports", "prepare", "launch", "metadata"):
        if not hasattr(backend, name):
            raise RuntimeError(f"adapter missing {name}")
    if not isinstance(backend.metadata, dict):
        raise RuntimeError("adapter metadata must be a JSON-serializable dict")
    return backend


def make_weight(torch, n, k, seed):
    torch.manual_seed(seed)
    x = torch.randn((n, k), dtype=torch.float32, device="cuda") / math.sqrt(k)
    blocks = x.view(n // GROUP, GROUP, k // GROUP, GROUP)
    scales = blocks.abs().amax(dim=(1, 3)).div(448.0).clamp_min(1e-7).contiguous()
    q = ((blocks / scales[:, None, :, None]).clamp(-448, 448)
         .to(torch.float8_e4m3fn).reshape(n, k).contiguous())
    return q, scales


def make_activation(torch, quant_fn, m, seed, distribution="normal"):
    torch.manual_seed(seed)
    x = torch.randn((m, K), dtype=torch.bfloat16, device="cuda")
    if distribution == "tiny":
        x.mul_(1e-5)
    elif distribution == "large":
        x.mul_(20)
    elif distribution == "zero":
        x.zero_()
    q, scales = quant_fn(x, group_size=GROUP, dtype=torch.float8_e4m3fn,
                        column_major_scales=True, use_ue8m0=False)
    q = q.contiguous()
    if distribution == "exceptional":
        # Direct FP8/scale injection follows quantization so the exceptional
        # values actually reach GEMM. This is deliberately selected coverage.
        q.view(torch.uint8)[0, 0] = 0x7f  # E4M3FN NaN
        scales[1, 0] = float("inf")
        scales[2, 0] = float("-inf")
        scales[3, 0] = float("nan")
        scales[4, 0] = torch.finfo(torch.float32).max
    if q.shape != (m, K) or scales.shape != (m, K // GROUP) or scales.stride() != (1, m):
        raise RuntimeError("activation shape/scale stride mismatch")
    return q, scales


def packed_weights(torch, b, bs):
    # Reinterpret bytes: no unsupported FP8 indexing/conversion is required.
    packed = b.view(torch.uint8).reshape(2, H // GROUP, GROUP, K).transpose(0, 1)
    packed = packed.contiguous().reshape(N, K).view(torch.float8_e4m3fn)
    packed_bs = bs.reshape(2, H // GROUP, K // GROUP).transpose(0, 1).contiguous().reshape(N // GROUP, K // GROUP)
    return packed, packed_bs


def inverse_packed_output(torch, x):
    # Dtype-preserving permutation of gate128/up128 to [gate || up].
    return x.reshape(x.shape[0], H // GROUP, 2, GROUP).transpose(1, 2).contiguous().reshape(x.shape[0], N)


def bit_comparison(torch, x, y, label, integer_dtype):
    xb, yb = x.contiguous().view(integer_dtype), y.contiguous().view(integer_dtype)
    different = xb != yb
    count = int(different.sum().item())
    record = {"label": label, "mismatches": count, "elements": x.numel(), "pass": count == 0}
    if count:
        # Only first flat mismatch is copied. Never materialize all mismatch
        # coordinates for a large output.
        flat = different.reshape(-1).to(torch.int8)
        index = int(flat.argmax().item())
        record["first_mismatch"] = {"row": index // x.shape[1], "column": index % x.shape[1],
                                    "actual_bits": int(xb.reshape(-1)[index].item()),
                                    "expected_bits": int(yb.reshape(-1)[index].item())}
        max_error = 0.0
        jointly_finite = 0
        actual_nonfinite, expected_nonfinite = 0, 0
        for row in range(0, x.shape[0], 128):
            xf, yf = x[row:row + 128].float(), y[row:row + 128].float()
            finite_x, finite_y = torch.isfinite(xf), torch.isfinite(yf)
            finite = finite_x & finite_y
            jointly_finite += int(finite.sum().item())
            actual_nonfinite += int((~finite_x).sum().item())
            expected_nonfinite += int((~finite_y).sum().item())
            error = torch.where(finite, (xf - yf).abs(), 0.0).max().item()
            max_error = max(max_error, float(error))
        record["finite_error_diagnostics"] = {"jointly_finite_elements": jointly_finite,
                                              "actual_nonfinite_elements": actual_nonfinite,
                                              "expected_nonfinite_elements": expected_nonfinite,
                                              "max_abs_on_jointly_finite": max_error if math.isfinite(max_error) else None}
    return record


def first_coordinates(torch, x, y, integer_dtype, limit=8):
    xb, yb = x.contiguous().view(integer_dtype), y.contiguous().view(integer_dtype)
    return first_mask_coordinates(xb != yb, limit)


def first_mask_coordinates(mask, limit=8):
    coordinates = []
    for row in range(0, mask.shape[0], 32):
        local = mask[row:row + 32].nonzero()
        for r, c in local[:limit - len(coordinates)].tolist():
            coordinates.append((row + r, c))
        if len(coordinates) == limit:
            break
    return coordinates


def finite_scalar(value):
    value = float(value)
    return {"value": value if math.isfinite(value) else None,
            "classification": "nan" if math.isnan(value) else
            ("+inf" if value == math.inf else ("-inf" if value == -math.inf else "finite"))}


def mismatch_group_diagnostics(torch, q1, s1, q0, s0):
    qdiff = q1.view(torch.uint8) != q0.view(torch.uint8)
    sdiff = s1.contiguous().view(torch.int32) != s0.contiguous().view(torch.int32)
    grouped = qdiff.reshape(q0.shape[0], H // GROUP, GROUP)
    same_scale_mask = qdiff & (~sdiff)[:, :, None].expand_as(grouped).reshape_as(qdiff)
    diff_scale_bytes = int((grouped & sdiff[:, :, None]).sum().item())
    total = int(qdiff.sum().item())
    record = {"FP8_mismatch_bytes": total,
              "scale_mismatch_groups": int(sdiff.sum().item()),
              "FP8_mismatch_groups": int(grouped.any(dim=2).sum().item()),
              "FP8_mismatch_bytes_in_scale_mismatch_groups": diff_scale_bytes,
              "FP8_mismatch_bytes_in_bit_identical_scale_groups": total - diff_scale_bytes,
              "all_FP8_mismatches_within_scale_mismatch_groups": total == diff_scale_bytes,
              "first_scale_mismatch_group_coordinates": first_mask_coordinates(sdiff),
              "first_same_scale_FP8_mismatch_coordinates": first_mask_coordinates(same_scale_mask)}
    return record


def bits_at(torch, x, row, col):
    integer_dtype = {torch.bfloat16: torch.int16, torch.float32: torch.int32,
                     torch.float8_e4m3fn: torch.uint8}[x.dtype]
    return int(x.contiguous().view(integer_dtype)[row, col].item())


def output_buffers(torch, m):
    q = torch.empty((m, H), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty((H // GROUP, m), device="cuda", dtype=torch.float32).t()
    return q, scales


def assert_nonalias(tensors):
    intervals = sorted((x.data_ptr(), x.data_ptr() + x.numel() * x.element_size()) for x in tensors)
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise RuntimeError("probe arm buffers overlap")


def l2_size(runtime, device):
    value = ctypes.c_int()
    runtime.cudaDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    runtime.cudaDeviceGetAttribute.restype = ctypes.c_int
    if runtime.cudaDeviceGetAttribute(ctypes.byref(value), 38, device) or value.value <= 0:
        raise RuntimeError("cannot query positive CUDA L2 size")
    return value.value


def capture(torch, call):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        call()
    torch.cuda.synchronize()
    return graph


def paired_timings(torch, graphs, flush, rng, repeats):
    output = {}
    for mode in ("warm", "cold"):
        values = {name: [] for name in graphs}
        orders = []
        for pair in range(repeats):
            order = list(graphs)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                if mode == "cold":
                    flush.fill_(pair & 255)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                values[name].append(begin.elapsed_time(end) * 1000)
        ratios = [x / y for x, y in zip(values["baseline"], values["candidate"])]
        output[mode] = {"orders": orders, "values_us": values,
                        "median_us": {name: statistics.median(v) for name, v in values.items()},
                        "paired_baseline_over_candidate": ratios,
                        "median_paired_baseline_over_candidate": statistics.median(ratios), "pairs": repeats}
    return output


def profile_once(torch, runtime, name, call):
    runtime.cudaProfilerStart.restype = ctypes.c_int
    runtime.cudaProfilerStop.restype = ctypes.c_int
    if runtime.cudaProfilerStart():
        raise RuntimeError("cudaProfilerStart failed")
    try:
        with torch.cuda.nvtx.range(f"GEMM_EPILOGUE_PIPELINE:{name}"):
            call()
        torch.cuda.synchronize()
    finally:
        if runtime.cudaProfilerStop():
            raise RuntimeError("cudaProfilerStop failed")


def run(args):
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("reserved SM120 GPU required")
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
    l2 = l2_size(runtime, torch.cuda.current_device())
    swizzle = 8 if N * K > l2 else 1
    baseline_hash = check_library(args.baseline_library, args.baseline_sha256)
    quant_hash = check_library(args.quant_library, args.quant_sha256)
    base = Baseline(torch, args.baseline_library, args.quant_library, args.workspace_mib * 1024**2, swizzle)
    fallback = Baseline(torch, args.baseline_library, args.quant_library, args.workspace_mib * 1024**2, swizzle)
    candidate_workspace = torch.empty(args.workspace_mib * 1024**2, device="cuda", dtype=torch.uint8)
    backend = None if args.baseline_only else load_candidate(args, candidate_workspace, swizzle)
    candidate_hash = None if args.baseline_only else check_library(args.candidate_library, args.candidate_sha256)
    diagnostic = None
    diagnostic_hash = None
    if args.diagnostic_gemm_library is not None:
        diagnostic_hash = check_library(args.diagnostic_gemm_library, args.diagnostic_gemm_sha256)
        diagnostic = DiagnosticGemm(torch, args.diagnostic_gemm_library,
                                    args.workspace_mib * 1024**2, swizzle)
    b, bs = make_weight(torch, N, K, 20261014)
    down_b, down_bs = make_weight(torch, K, H, 20261015)
    torch.cuda.synchronize()
    pack_begin, pack_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    pack_begin.record()
    pb, pbs = packed_weights(torch, b, bs)
    pack_end.record()
    pack_end.synchronize()
    setup = {"event_us": pack_begin.elapsed_time(pack_end) * 1000,
             "wall_us": (time.perf_counter() - wall) * 1e6,
             "extra_packed_bytes_held_both_arms": pb.numel() + pbs.numel() * 4}
    # Byte roundtrip checks cover every packed weight and scale element.
    roundtrip_b = pb.view(torch.uint8).reshape(H // GROUP, 2, GROUP, K).transpose(0, 1).contiguous().reshape(N, K)
    roundtrip_bs = pbs.reshape(H // GROUP, 2, K // GROUP).transpose(0, 1).contiguous().reshape(N // GROUP, K // GROUP)
    setup["weight_roundtrip_exact"] = bool(torch.equal(roundtrip_b, b.view(torch.uint8)))
    setup["scale_roundtrip_bits_exact"] = bool(torch.equal(roundtrip_bs.view(torch.int32), bs.view(torch.int32)))
    del roundtrip_b, roundtrip_bs
    if not setup["weight_roundtrip_exact"] or not setup["scale_roundtrip_bits_exact"]:
        raise RuntimeError("prepack roundtrip mismatch")
    emit("metadata", {"torch": torch.__version__, "vllm": __import__("vllm").__version__,
          "expected_fixed_vllm_commit": "db9527a46873454610df6dbedf79a36d6bf1a7f6",
          "official_ops_source_sha256": sha(Path(ops.__file__)),
          "device": torch.cuda.get_device_name(), "l2_bytes": l2, "swizzle": swizzle,
          "baseline_library_sha256": baseline_hash, "quant_library_sha256": quant_hash,
          "candidate_library_sha256": candidate_hash, "candidate": None if backend is None else backend.metadata,
          "diagnostic_gemm_library_sha256": diagnostic_hash,
          "diagnostic_gemm_scope": "N256/M64 ordinary BF16 epilogue, not direct fused visitor BF16 observation" if diagnostic else None,
          "candidate_adapter_sha256": None if backend is None else sha(args.candidate_adapter),
          "probe_sha256": sha(Path(__file__)), "setup": setup,
          "order_seed": args.order_seed, "timing_pairs": args.repeats,
          "validation_scope": "selected synthetic operands, not exhaustive finite-FP8 proof or model quality",
          "warm_scope": "whole-pipeline graph replay, all weights need not fit L2",
          "cold_scope": "owned >2xL2 flush approximation excluded from timed events",
          "both_weight_layouts_held_for_both_arms": True})

    def official_gemm(a, asc, weight=b, scales=bs):
        return ops.cutlass_scaled_mm(a, weight.T, out_dtype=torch.bfloat16,
                                     scale_a=asc, scale_b=scales.T)

    def official_quant(x, q, sc):
        torch.ops._C.silu_and_mul_per_block_quant(q, x, sc, GROUP, None, True)

    def allocate(m, distribution):
        a, asc = make_activation(torch, per_token_group_quant_fp8, m, 1000 + m, distribution)
        x0 = torch.empty((m, N), device="cuda", dtype=torch.bfloat16)
        # Keep a distinct fallback intermediate even for active fusion; both
        # arms have equal retained allocation conditions during timing.
        x1 = torch.empty_like(x0)
        q0, s0 = output_buffers(torch, m)
        q1, s1 = output_buffers(torch, m)
        assert_nonalias([x0, x1, q0, s0, q1, s1, base.workspace,
                         fallback.workspace, candidate_workspace])
        active = backend is not None and backend.supports(m)
        if active:
            backend.prepare(m)
            assert_nonalias([x0, x1, q0, s0, q1, s1, base.workspace,
                             fallback.workspace, candidate_workspace, *backend.owned_buffers])
        def run_base():
            base.gemm(x0, a, asc, b, bs)
            base.quant(x0, q0, s0)
        def run_candidate():
            if active:
                backend.launch(q1, s1, a, asc, pb, pbs)
            else:
                fallback.gemm(x1, a, asc, b, bs)
                fallback.quant(x1, q1, s1)
        return a, asc, x0, x1, q0, s0, q1, s1, active, run_base, run_candidate

    def validate(m, distribution, graph_check):
        a, asc, x0, x1, q0, s0, q1, s1, active, run_base, run_candidate = allocate(m, distribution)
        comparisons = []
        default = torch.cuda.default_stream()
        with torch.cuda.stream(default):
            run_base()
            if backend is not None:
                run_candidate()
            official_x = official_gemm(a, asc)
            oq, os = output_buffers(torch, m)
            official_quant(official_x, oq, os)
            packed_x = torch.empty_like(x0)
            base.gemm(packed_x, a, asc, pb, pbs)
        torch.cuda.synchronize()
        comparisons.append(bit_comparison(torch, x0, official_x, "baseline_GEMM_vs_official_BF16", torch.int16))
        comparisons.append(bit_comparison(torch, inverse_packed_output(torch, packed_x), official_x,
                                          "packed_fixed_GEMM_inverse_vs_official_BF16", torch.int16))
        comparisons.append(bit_comparison(torch, q0, oq, "baseline_pipeline_vs_official_FP8", torch.uint8))
        comparisons.append(bit_comparison(torch, s0, os, "baseline_pipeline_vs_official_scale", torch.int32))
        if backend is not None:
            comparisons.append(bit_comparison(torch, q1, q0, "candidate_FP8", torch.uint8))
            comparisons.append(bit_comparison(torch, s1, s0, "candidate_scale", torch.int32))
            down0 = official_gemm(q0, s0, down_b, down_bs)
            down1 = official_gemm(q1, s1, down_b, down_bs)
            torch.cuda.synchronize()
            comparisons.append(bit_comparison(torch, down1, down0, "candidate_down_GEMM_BF16", torch.int16))
            del down0, down1
        mismatch_groups = None if backend is None else mismatch_group_diagnostics(torch, q1, s1, q0, s0)
        diagnostic_record = None
        if diagnostic is not None and m > 256:
            xd_paired = torch.empty_like(x0)
            assert_nonalias([xd_paired, x0, x1, q0, s0, q1, s1, diagnostic.workspace,
                             base.workspace, fallback.workspace, candidate_workspace])
            diagnostic.launch(xd_paired, a, asc, pb, pbs)
            xd = inverse_packed_output(torch, xd_paired)
            dq, ds = output_buffers(torch, m)
            doq, dos = output_buffers(torch, m)
            base.quant(xd, dq, ds)
            official_quant(xd, doq, dos)
            torch.cuda.synchronize()
            checks = [bit_comparison(torch, xd, official_x, "N256_BF16_inverse_vs_official", torch.int16),
                      bit_comparison(torch, dq, doq, "N256_BF16_V2_vs_official_quant_FP8", torch.uint8),
                      bit_comparison(torch, ds, dos, "N256_BF16_V2_vs_official_quant_scale", torch.int32),
                      bit_comparison(torch, dq, q0, "N256_BF16_requant_vs_baseline_FP8", torch.uint8),
                      bit_comparison(torch, ds, s0, "N256_BF16_requant_vs_baseline_scale", torch.int32)]
            witnesses, same_scale_witnesses = [], []
            if backend is not None:
                checks += [bit_comparison(torch, q1, dq, "candidate_vs_N256_BF16_requant_FP8", torch.uint8),
                           bit_comparison(torch, s1, ds, "candidate_vs_N256_BF16_requant_scale", torch.int32)]
                coords = first_coordinates(torch, q1, q0, torch.uint8)
                same_scale_coords = mismatch_groups["first_same_scale_FP8_mismatch_coordinates"]
                coords = list(dict.fromkeys(coords + same_scale_coords))
                for row, col in coords:
                    group = col // GROUP
                    witness = {"row": row, "column": col, "group": group,
                        "official_gate_BF16_bits": bits_at(torch, official_x, row, col),
                        "official_up_BF16_bits": bits_at(torch, official_x, row, H + col),
                        "N256_gate_BF16_bits": bits_at(torch, xd, row, col),
                        "N256_up_BF16_bits": bits_at(torch, xd, row, H + col),
                        "baseline_FP8_bits": bits_at(torch, q0, row, col),
                        "N256_requant_FP8_bits": bits_at(torch, dq, row, col),
                        "candidate_FP8_bits": bits_at(torch, q1, row, col),
                        "baseline_scale_bits": bits_at(torch, s0, row, group),
                        "N256_requant_scale_bits": bits_at(torch, ds, row, group),
                        "candidate_scale_bits": bits_at(torch, s1, row, group),
                        "official_gate": finite_scalar(official_x[row, col].item()),
                        "official_up": finite_scalar(official_x[row, H + col].item()),
                        "N256_gate": finite_scalar(xd[row, col].item()),
                        "N256_up": finite_scalar(xd[row, H + col].item()),
                        "candidate_fused_BF16_not_observed": True}
                    witnesses.append(witness)
                    if (row, col) in same_scale_coords:
                        same_scale_witnesses.append(witness)
            diagnostic_record = {"comparisons": checks, "first_candidate_mismatch_witnesses": witnesses,
                "first_same_scale_candidate_mismatch_witnesses": same_scale_witnesses,
                "first_BF16_mismatch_coordinates": first_coordinates(torch, xd, official_x, torch.int16),
                "scope": "triangulation only; ordinary BF16 epilogue can change stages/resource carveout",
                "candidate_matches_N256_requant_all_bits": (all(c["pass"] for c in checks[-2:]) if backend else None),
                "original_candidate_gate_unchanged": True}
            del xd_paired, xd, dq, ds, doq, dos
        del official_x, packed_x, oq, os
        record = {"M": m, "K": K, "N": N, "distribution": distribution,
                  "candidate_active": active, "candidate_fallback": backend is not None and not active,
                  "default_stream_address": default.cuda_stream, "scale_stride": list(s1.stride()),
                  "comparisons": comparisons, "pass": all(x["pass"] for x in comparisons)}
        if diagnostic_record is not None:
            record["N256_BF16_diagnostic"] = diagnostic_record
        if mismatch_groups is not None:
            record["candidate_mismatch_group_diagnostics"] = mismatch_groups
        if active:
            record["candidate_temporary_buffers"] = [
                {"shape": list(x.shape), "dtype": str(x.dtype), "bytes": x.numel() * x.element_size()}
                for x in backend.owned_buffers]
        if record["pass"] and graph_check and backend is not None:
            graphs = {"baseline": capture(torch, run_base), "candidate": capture(torch, run_candidate)}
            old_q = q1.view(torch.uint8).clone()
            new_a, new_s = make_activation(torch, per_token_group_quant_fp8, m, 9000 + m)
            a.copy_(new_a)
            asc.copy_(new_s)
            for graph in graphs.values():
                graph.replay()
            fresh_x = official_gemm(a, asc)
            fq, fs = output_buffers(torch, m)
            official_quant(fresh_x, fq, fs)
            torch.cuda.synchronize()
            gc = [bit_comparison(torch, q1, fq, "candidate_replay_vs_fresh_FP8", torch.uint8),
                  bit_comparison(torch, s1, fs, "candidate_replay_vs_fresh_scale", torch.int32),
                  bit_comparison(torch, q0, fq, "baseline_replay_vs_fresh_FP8", torch.uint8),
                  bit_comparison(torch, s0, fs, "baseline_replay_vs_fresh_scale", torch.int32)]
            changed = not bool(torch.equal(old_q, q1.view(torch.uint8)))
            record["graph"] = {"changed_input": True, "output_changed": changed,
                               "comparisons": gc, "pass": changed and all(x["pass"] for x in gc)}
            record["pass"] = record["pass"] and record["graph"]["pass"]
        emit("numerics", record)
        return record["pass"]

    # Every gate precedes the whole performance phase. A later failure cannot
    # leave apparently valid timing results for the same experimental binary.
    cases = [(m, "normal", True) for m in args.m]
    if not args.no_special:
        cases += [(257, label, False) for label in ("zero", "tiny", "large", "exceptional")]
    passed = True
    for m, distribution, graph_check in cases:
        case_ok = validate(m, distribution, graph_check)
        passed = passed and case_ok
    if not passed:
        emit("summary", {"pass": False, "timings_emitted": False, "reason": "exact numerical gate failed"})
        return 1
    if args.baseline_only:
        emit("summary", {"pass": True, "baseline_only": True, "candidate_validated": False, "timings_emitted": False})
        return 0
    flush = None if args.ncu else torch.empty(max(2 * l2 + 1, 256 * 1024**2), device="cuda", dtype=torch.uint8)
    rng = random.Random(args.order_seed)
    for m in args.m:
        state = allocate(m, "normal")
        active, run_base, run_candidate = state[-3:]
        if args.ncu:
            if not active:
                raise RuntimeError("NCU candidate shape unsupported; refuse to label fallback as fusion")
            for _ in range(3):
                run_base()
                run_candidate()
            torch.cuda.synchronize()
            profile_once(torch, runtime, "baseline", run_base)
            profile_once(torch, runtime, "candidate", run_candidate)
            emit("profiled_once", {"M": m, "baseline_complete_pipelines": 1, "candidate_complete_pipelines": 1,
                                   "denominator": "one complete GEMM→SwiGLU/blockquant pipeline per arm",
                                   "candidate_active": active})
        else:
            graphs = {"baseline": capture(torch, run_base), "candidate": capture(torch, run_candidate)}
            timing = paired_timings(torch, graphs, flush, rng, args.repeats)
            emit("timing", {"M": m, "candidate_active": active, "candidate_fallback": not active,
                            "eligible_as_fusion_speedup": active, "full_pipeline": True, **timing})
            del graphs
        del state
    emit("summary", {"pass": True, "candidate_validated": True, "timings_emitted": not args.ncu,
                     "ncu": args.ncu, "numeric_cases": len(cases)})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-library", required=True, type=Path)
    parser.add_argument("--quant-library", required=True, type=Path)
    parser.add_argument("--baseline-sha256", default=BASELINE_SHA)
    parser.add_argument("--quant-sha256", default=QUANT_SHA)
    parser.add_argument("--candidate-adapter", type=Path)
    parser.add_argument("--candidate-library", type=Path)
    parser.add_argument("--candidate-sha256", default="")
    parser.add_argument("--diagnostic-gemm-library", "--diagnose-n256-bf16-library", type=Path,
                        help="optional n256_block_fp8 ordinary BF16 GEMM, numerical diagnostics only")
    parser.add_argument("--diagnostic-gemm-sha256", default="")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--m", nargs="+", type=int, default=list(DEFAULT_M))
    parser.add_argument("--workspace-mib", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--order-seed", type=int, default=20261014)
    parser.add_argument("--no-special", action="store_true", help="explicitly skip selected exceptional-input checks")
    parser.add_argument("--ncu", action="store_true")
    args = parser.parse_args()
    if not args.baseline_only and (args.candidate_adapter is None or args.candidate_library is None):
        parser.error("candidate adapter/library required unless --baseline-only")
    if args.workspace_mib < 1 or args.repeats < 1 or not args.m or any(m < 1 for m in args.m):
        parser.error("positive workspace, repeats and M required")
    if args.ncu and (args.baseline_only or len(args.m) != 1):
        parser.error("--ncu requires candidate and exactly one --m")
    try:
        return run(args)
    except Exception as exc:
        traceback.print_exc()
        emit("summary", {"pass": False, "error_type": type(exc).__name__, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
