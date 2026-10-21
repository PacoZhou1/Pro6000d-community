#!/usr/bin/env python3
"""Isolated numerical gate and kernel timing for SM120 FP8 SwiGLU quant.

Run only in a reserved GPU window with the fixed vLLM 0.31.0 image. No model
weights are needed: this tests one synthetic operator, not model quality.
Any FP8 byte or FP32 scale-bit mismatch makes the probe fail and invalidates
the timing as an optimization candidate. All timings exclude allocations.
Direct-call CUDA Events can include CPU dispatch gaps; the reported CUDA
Graph replay comparison is the preferred low-overhead kernel timing.
"""

import argparse
import ctypes
import hashlib
import json
import traceback
from pathlib import Path

K = 17408
GROUP = 128
G = K // GROUP
SHAPES = (1, 5, 16, 255, 256, 257, 2741, 2786, 16160)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    import torch
    from vllm import _custom_ops as _ops  # noqa: F401 - registers fixed _C ops

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("reserved SM120 GPU required")
    if args.sha256 and _hash(args.library) != args.sha256:
        raise RuntimeError("library SHA256 mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261014)
    lib = ctypes.CDLL(str(args.library.resolve()))
    fn = lib.pro6000d_fp8_silu_quant_launch
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_uint64]
    fn.restype = ctypes.c_int
    lib.pro6000d_fp8_silu_quant_last_error.restype = ctypes.c_char_p

    def buffers(m):
        output = torch.empty((m, K), dtype=torch.float8_e4m3fn, device="cuda")
        scales = torch.empty((G, m), dtype=torch.float32, device="cuda").t()
        assert scales.shape == (m, G) and scales.stride() == (1, m)
        return output, scales

    def official(x, q, s):
        torch.ops._C.silu_and_mul_per_block_quant(q, x, s, GROUP, None, True)

    def candidate(x, q, s):
        m = x.shape[0]
        if m < 256:
            official(x, q, s)
            return False
        stream = torch.cuda.current_stream(x.device)
        status = fn(q.data_ptr(), s.data_ptr(), x.data_ptr(), m, K,
                    s.stride(0), s.stride(1), stream.cuda_stream)
        if status:
            error = lib.pro6000d_fp8_silu_quant_last_error()
            raise RuntimeError(f"candidate launch {status}: {error!r}")
        # Probe buffers stay alive until explicit synchronization below.
        return True

    def measure(call, cold, flush, repeats):
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        elapsed = []
        for i in range(repeats):
            if cold:
                flush.fill_(i & 255)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            end.synchronize()
            elapsed.append(start.elapsed_time(end) * 1000.0)
        elapsed.sort()
        return {"median_us": elapsed[len(elapsed) // 2],
                "min_us": elapsed[0], "max_us": elapsed[-1],
                "repeats": repeats}

    # This is an owned cache eviction tensor. Cold measurements time only
    # the following target kernel, not the fill. 256 MiB exceeds this GPU's
    # stated 128 MiB L2 but is only a synthetic approximation of model reuse.
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    results = {"library": str(args.library.resolve()),
               "library_sha256": _hash(args.library),
               "device": torch.cuda.get_device_name(),
               "commit": "db9527a46873454610df6dbedf79a36d6bf1a7f6",
               "synthetic_single_operator_not_model_quality": True,
               "direct_event_timing_may_include_host_dispatch_gaps": True,
               "graph_replay_is_preferred_timing": True,
               "cases": []}

    cases = [(m, "normal", 0.5, 1000 + m, True) for m in SHAPES]
    cases += [(256, "small_random", 1e-5, 2001, False),
              (256, "wide_random", 20.0, 2002, False),
              (256, "special_values", 0.5, 2003, False)]
    for m, distribution, std, seed, timed in cases:
        torch.manual_seed(seed)
        x = torch.empty((m, 2 * K), dtype=torch.bfloat16, device="cuda")
        x.normal_(mean=0.0, std=std)
        if distribution == "special_values":
            # Use separate groups/rows for exceptional and boundary cases.
            x[0, :] = 0
            x[1, :] = 1e-7
            x[2, :K] = 400
            x[2, K:] = -400
            x[3, 0] = float("nan")
            x[3, K] = float("inf")
            x[4, :GROUP] = float("nan")
            x[5, :GROUP] = float("inf")
            x[6, :GROUP] = float("-inf")
            x[7, :GROUP] = torch.finfo(torch.bfloat16).max
            x[7, K:K + GROUP] = torch.finfo(torch.bfloat16).max
        q0, s0 = buffers(m)
        q1, s1 = buffers(m)
        official(x, q0, s0)
        active = candidate(x, q1, s1)
        torch.cuda.synchronize()

        q0_bytes, q1_bytes = q0.view(torch.uint8), q1.view(torch.uint8)
        # The col-major scale tensors have stride (1,M); materialize equal
        # logical layouts before reinterpretation to avoid Float8/FP32 index
        # kernel or non-contiguous dtype-view assumptions.
        s0_bits = s0.contiguous().view(torch.int32)
        s1_bits = s1.contiguous().view(torch.int32)
        q_mismatches = int((q0_bytes != q1_bytes).sum().item())
        s_mismatches = int((s0_bits != s1_bits).sum().item())
        finite_input = bool(torch.isfinite(x).all().item())
        record = {"M": m, "K": K, "group_size": GROUP,
                  "distribution": distribution, "seed": seed, "std": std,
                  "special_values": (["zero", "tiny", "large_finite",
                                      "nan", "+inf", "-inf",
                                      "finite_product_overflow_to_inf"]
                                     if distribution == "special_values" else []),
                  "input_all_finite": finite_input,
                  "candidate_active": active,
                  "fp8_byte_mismatches": q_mismatches,
                  "fp32_scale_bit_mismatches": s_mismatches,
                  "scale_stride": list(s1.stride()),
                  "pass": q_mismatches == 0 and s_mismatches == 0}
        if not record["pass"]:
            results["cases"].append(record)
            results["pass"] = False
            return results

        if timed:
            order = ("candidate", "official") if (m % 2) else ("official", "candidate")
            times = {}
            calls = {"official": lambda: official(x, q0, s0),
                     "candidate": lambda: candidate(x, q1, s1)}
            for cache_state in ("warm", "cold"):
                for label in order:
                    times[f"{label}_{cache_state}"] = measure(
                        calls[label], cache_state == "cold", flush, args.repeats)
            record["timings_us"] = times
            record["timing_order"] = list(order)
            if m in args.graph_shapes:
                # Capture both fixed-output ops, mutate the *same* input
                # buffer after capture, and compare their replay results.
                official_graph = torch.cuda.CUDAGraph()
                candidate_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(official_graph):
                    official(x, q0, s0)
                with torch.cuda.graph(candidate_graph):
                    candidate(x, q1, s1)
                torch.cuda.synchronize()
                before_replay = q1.view(torch.uint8).clone()
                x.normal_(mean=0.0, std=1.25)
                official_graph.replay()
                candidate_graph.replay()
                qfresh, sfresh = buffers(m)
                official(x, qfresh, sfresh)
                torch.cuda.synchronize()
                replay_changed = not bool(torch.equal(before_replay,
                                        q1.view(torch.uint8)))
                graph_q_mismatch = int((qfresh.view(torch.uint8) !=
                                        q1.view(torch.uint8)).sum().item())
                graph_s_mismatch = int((sfresh.contiguous().view(torch.int32) !=
                                        s1.contiguous().view(torch.int32)).sum().item())
                official_graph_q_mismatch = int((qfresh.view(torch.uint8) !=
                                                 q0.view(torch.uint8)).sum().item())
                record["cuda_graph"] = {
                    "input_changed_after_capture": True,
                    "replay_output_changed": replay_changed,
                    "fp8_byte_mismatches": graph_q_mismatch,
                    "fp32_scale_bit_mismatches": graph_s_mismatch,
                    "official_graph_vs_fresh_byte_mismatches": official_graph_q_mismatch,
                    "official_warm_us": measure(official_graph.replay, False,
                                                flush, args.repeats),
                    "candidate_warm_us": measure(candidate_graph.replay, False,
                                                 flush, args.repeats),
                    "official_cold_us": measure(official_graph.replay, True,
                                                flush, args.repeats),
                    "candidate_cold_us": measure(candidate_graph.replay, True,
                                                 flush, args.repeats),
                    "pass": replay_changed and graph_q_mismatch == 0 and
                            graph_s_mismatch == 0 and official_graph_q_mismatch == 0,
                }
                record["pass"] = record["pass"] and record["cuda_graph"]["pass"]
                if not record["pass"]:
                    results["cases"].append(record)
                    results["pass"] = False
                    return results
        results["cases"].append(record)
        del x, q0, q1, s0, s1
    results["pass"] = all(case["pass"] for case in results["cases"])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sha256", default="")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--graph-shapes", type=int, nargs="*", default=[256, 2741],
                        help="Shapes to validate and time with CUDA Graph replay")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    try:
        result = run(args)
    except Exception as exc:
        traceback.print_exc()
        result = {"pass": False, "error_type": type(exc).__name__,
                  "error": str(exc)}
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
