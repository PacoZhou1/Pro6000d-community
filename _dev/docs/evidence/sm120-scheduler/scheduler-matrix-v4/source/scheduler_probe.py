#!/usr/bin/env python3
"""SM120 raster/swizzle isolation; GPU execution is reserved for the parent.

Imports exact comparison, input-generation and graph helpers from the
previously validated gemm-epilogue pipeline probe. All scheduler C ABI
arguments are per-call; there is no process-global raster setting.
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
import traceback
from pathlib import Path

K, N, H, GROUP = 5120, 34816, 17408, 128
RASTERS = {"heuristic": 0, "along_m": 1, "along_n": 2}
ARMS = tuple(f"v6_{r}_s{s}" for r in RASTERS for s in (1, 8))
DEFAULT_GATE_M = (1, 5, 16, 256, 257, 2741, 2786, 16160, 32768)
LEGACY_SHA = "e4039ddc95fa3b56f02743357b7a7193156dc500c5d8804bb77ebd3af85450d6"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def emit(key, value):
    print(json.dumps({key: value}, allow_nan=False, sort_keys=True), flush=True)


def check_sha(path, expected):
    actual = sha(path)
    if expected and actual != expected:
        raise RuntimeError(f"{path}: SHA256 mismatch {actual} != {expected}")
    return actual


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SchedulerLibrary:
    """Exact ABI 1 agreed with the isolated kernel builder."""
    def __init__(self, torch, path):
        self.torch = torch
        self.lib = ctypes.CDLL(str(path.resolve()))
        self.lib.sm120_scheduler_abi_version.argtypes = []
        self.lib.sm120_scheduler_abi_version.restype = ctypes.c_int
        if self.lib.sm120_scheduler_abi_version() != 1:
            raise RuntimeError("scheduler library ABI must be 1")
        self.lib.sm120_scheduler_last_error.argtypes = []
        self.lib.sm120_scheduler_last_error.restype = ctypes.c_char_p
        self.lib.sm120_scheduler_metadata.argtypes = []
        self.lib.sm120_scheduler_metadata.restype = ctypes.c_char_p
        self.metadata = json.loads(self.lib.sm120_scheduler_metadata().decode())
        self.query = {}
        self.launch = {}
        for name in ("official_bf16", "paired_bf16", "fused_fp8"):
            q = getattr(self.lib, f"sm120_{name}_workspace_bytes")
            q.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
            q.restype = ctypes.c_size_t
            self.query[name] = q
            fn = getattr(self.lib, f"sm120_{name}")
            pointer_count = 8 if name == "fused_fp8" else 5
            fn.argtypes = ([ctypes.c_void_p] * pointer_count + [ctypes.c_int] * 3 +
                           [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong])
            fn.restype = ctypes.c_int
            self.launch[name] = fn
        self.lib.sm120_fused_fp8_temp_scale_bytes.argtypes = [ctypes.c_int]
        self.lib.sm120_fused_fp8_temp_scale_bytes.restype = ctypes.c_size_t

    def prepare(self, kind, m, raster, swizzle):
        need = self.query[kind](m, raster, swizzle)
        if not 257 <= m <= 32768 or need < 1:
            raise RuntimeError(f"unsupported scheduler launch {kind} M={m} raster={raster} swizzle={swizzle}")
        workspace = self.torch.empty(need, dtype=self.torch.uint8, device="cuda")
        temp = native = None
        if kind == "fused_fp8":
            temp = self.torch.empty((m, N), dtype=self.torch.float8_e4m3fn, device="cuda")
            bytes_needed = self.lib.sm120_fused_fp8_temp_scale_bytes(m)
            if bytes_needed < m * (H // GROUP) * 4:
                raise RuntimeError("native scale query is smaller than output scales")
            native = self.torch.empty(bytes_needed, dtype=self.torch.uint8, device="cuda")
        return {"workspace": workspace, "temp": temp, "native": native,
                "kind": kind, "m": m, "raster": raster, "swizzle": swizzle}

    def call(self, state, out, scales, a, asc, b, bs):
        m, kind = state["m"], state["kind"]
        if (a.shape != (m, K) or not a.is_contiguous() or a.dtype != self.torch.float8_e4m3fn or
            asc.shape != (m, K // GROUP) or asc.stride() != (1, m) or asc.dtype != self.torch.float32 or
            b.shape != (N, K) or not b.is_contiguous() or b.dtype != self.torch.float8_e4m3fn or
            bs.shape != (N // GROUP, K // GROUP) or not bs.is_contiguous() or bs.dtype != self.torch.float32):
            raise RuntimeError("scheduler tensor dtype/shape/stride mismatch")
        if kind == "fused_fp8":
            if (out.shape != (m, H) or out.dtype != self.torch.float8_e4m3fn or
                not out.is_contiguous() or scales.shape != (m, H // GROUP) or
                scales.dtype != self.torch.float32 or scales.stride() != (1, m)):
                raise RuntimeError("scheduler fused output layout mismatch")
            ptrs = (out.data_ptr(), scales.data_ptr(), state["temp"].data_ptr(),
                    state["native"].data_ptr(), a.data_ptr(), b.data_ptr(),
                    asc.data_ptr(), bs.data_ptr())
        else:
            if out.shape != (m, N) or out.dtype != self.torch.bfloat16 or not out.is_contiguous():
                raise RuntimeError("scheduler BF16 output layout mismatch")
            ptrs = (out.data_ptr(), a.data_ptr(), b.data_ptr(), asc.data_ptr(), bs.data_ptr())
        stream = self.torch.cuda.current_stream(a.device).cuda_stream
        code = self.launch[kind](*ptrs, m, state["raster"], state["swizzle"],
                                 state["workspace"].data_ptr(), state["workspace"].numel(), stream)
        if code:
            raise RuntimeError(f"sm120_{kind} returned {code}: {self.lib.sm120_scheduler_last_error()!r}")


def parse_arm(name):
    for raster in RASTERS:
        for swizzle in (1, 8):
            if name == f"v6_{raster}_s{swizzle}":
                return RASTERS[raster], swizzle
    raise ValueError(name)


def input_fingerprint(torch, a, asc):
    """Hash actual synthetic inputs once, outside graph capture/timing."""
    h = hashlib.sha256()
    h.update(a.view(torch.uint8).cpu().numpy().tobytes())
    h.update(asc.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def cuda_versions(runtime):
    result = {}
    for label, symbol in (("runtime", "cudaRuntimeGetVersion"),
                          ("driver", "cudaDriverGetVersion")):
        value = ctypes.c_int()
        fn = getattr(runtime, symbol)
        fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.c_int
        if fn(ctypes.byref(value)):
            raise RuntimeError(f"{symbol} failed")
        result[label] = value.value
    return result


def profile_once(torch, runtime, arm, call):
    runtime.cudaProfilerStart.argtypes = []
    runtime.cudaProfilerStart.restype = ctypes.c_int
    runtime.cudaProfilerStop.argtypes = []
    runtime.cudaProfilerStop.restype = ctypes.c_int
    if runtime.cudaProfilerStart():
        raise RuntimeError("cudaProfilerStart failed")
    try:
        with torch.cuda.nvtx.range(f"SM120_SCHEDULER:{arm}"):
            call()
        torch.cuda.synchronize()
    finally:
        if runtime.cudaProfilerStop():
            raise RuntimeError("cudaProfilerStop failed")


def measure_graphs(torch, graphs, flush, seed, repeats):
    output = {}
    for mode in ("warm", "cache_disturbed"):
        rng = random.Random(seed + (0 if mode == "warm" else 1))
        values = {name: [] for name in graphs}
        orders = []
        for pair in range(repeats):
            order = list(graphs)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                if mode == "cache_disturbed":
                    flush.fill_(pair & 255)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                values[name].append(begin.elapsed_time(end) * 1000)
        output[mode] = {"orders": orders, "values_us": values,
                        "median_us": {k: statistics.median(v) for k, v in values.items()},
                        "pairs": repeats}
    return output


def run(args):
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    # A timed shape must have both a numerical gate and a changed-input graph
    # replay. Compute the effective sets here so CLI overrides cannot omit one.
    numeric_shapes = tuple(dict.fromkeys(
        args.timed_m if args.ncu_arm else (*args.gate_m, *args.graph_m, *args.timed_m)))
    graph_shapes = tuple(dict.fromkeys(
        args.timed_m if args.ncu_arm else (*args.graph_m, *args.timed_m)))

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("reserved SM120 GPU required")
    torch.backends.cuda.matmul.allow_tf32 = False
    old = import_file(args.old_probe, "fixed_pipeline_probe")
    adapter = import_file(args.legacy_adapter, "fixed_v6_adapter")
    hashes = {"old_probe": check_sha(args.old_probe, args.old_probe_sha256),
              "legacy_adapter": check_sha(args.legacy_adapter, args.legacy_adapter_sha256),
              "baseline": check_sha(args.baseline_library, args.baseline_sha256),
              "quant": check_sha(args.quant_library, args.quant_sha256),
              "legacy_v6": check_sha(args.legacy_library, args.legacy_sha256),
              "scheduler": check_sha(args.scheduler_library, args.scheduler_sha256),
              "probe": sha(Path(__file__))}
    scheduler = SchedulerLibrary(torch, args.scheduler_library)
    runtime = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
    l2 = old.l2_size(runtime, torch.cuda.current_device())
    if N * K <= l2:
        raise RuntimeError("fixed 170 MiB packed weight no longer exceeds measured L2; control changed")
    b, bs = old.make_weight(torch, N, K, args.input_seed)
    pb, pbs = old.packed_weights(torch, b, bs)
    down_b, down_bs = old.make_weight(torch, K, H, args.input_seed + 1)
    torch.cuda.synchronize()
    # Roundtrip must cover every packed weight and scale bit before any ABI call.
    rb = pb.view(torch.uint8).reshape(H // GROUP, 2, GROUP, K).transpose(0, 1).contiguous().reshape(N, K)
    rs = pbs.reshape(H // GROUP, 2, K // GROUP).transpose(0, 1).contiguous().reshape(N // GROUP, K // GROUP)
    if not torch.equal(rb, b.view(torch.uint8)) or not torch.equal(rs.view(torch.int32), bs.view(torch.int32)):
        raise RuntimeError("packed weight or scale roundtrip mismatch")
    del rb, rs
    emit("metadata", {"scope": "synthetic FP8 operator comparison; not model quality or API speed",
                      "device": torch.cuda.get_device_name(), "torch": torch.__version__,
                      "torch_cuda_build": torch.version.cuda,
                      "cuda_versions": cuda_versions(runtime),
                      "vllm": __import__("vllm").__version__, "l2_bytes": l2,
                      "official_ops_sha256": sha(Path(ops.__file__)), "sha256": hashes,
                      "scheduler_abi": 1, "scheduler_metadata": scheduler.metadata,
                      "legacy_anchor_raster": "heuristic", "legacy_anchor_swizzle": 8,
                      "settings": {a: {"raster": parse_arm(a)[0], "swizzle": parse_arm(a)[1]} for a in ARMS},
                      "input_seed": args.input_seed, "order_seed": args.order_seed,
                      "timed_shapes": list(args.timed_m), "numeric_shapes": list(numeric_shapes),
                      "requested_graph_shapes": list(args.graph_m),
                      "changed_input_graph_shapes": list(graph_shapes),
                      "timing_pairs": args.repeats,
                      "cache_disturbed_scope": "owned >2x L2 flush outside event; no fixed cold-L2 guarantee",
                      "ncu_cache_none_scope": "not fixed cold cache"})

    def official_ref(a, asc, weight=b, scales=bs):
        return ops.cutlass_scaled_mm(a, weight.T, out_dtype=torch.bfloat16,
                                     scale_a=asc, scale_b=scales.T)

    def official_quant(x, q, sc):
        torch.ops._C.silu_and_mul_per_block_quant(q, x, sc, GROUP, None, True)

    def controls(m, a, asc):
        base = old.Baseline(torch, args.baseline_library, args.quant_library, args.workspace_mib << 20, 8)
        x0 = torch.empty((m, N), device="cuda", dtype=torch.bfloat16)
        q0, s0 = old.output_buffers(torch, m)
        def call():
            base.gemm(x0, a, asc, b, bs)
            base.quant(x0, q0, s0)
        return base, x0, q0, s0, call

    numeric_cases = [(m, "normal") for m in numeric_shapes]
    if not args.ncu_arm:
        numeric_cases += [(257, kind) for kind in ("zero", "tiny", "large", "exceptional")]
    all_pass = True
    for m, distribution in numeric_cases:
        a, asc = old.make_activation(torch, per_token_group_quant_fp8, m,
                                     args.input_seed + 100 + m, distribution)
        input_hash = input_fingerprint(torch, a, asc) if distribution == "normal" and m in args.timed_m else None
        base, x0, q0, s0, run_base = controls(m, a, asc)
        run_base()
        xr = official_ref(a, asc)
        qr, sr = old.output_buffers(torch, m)
        official_quant(xr, qr, sr)
        torch.cuda.synchronize()
        checks = [old.bit_comparison(torch, x0, xr, "baseline_vs_fixed_vllm_BF16", torch.int16),
                  old.bit_comparison(torch, q0, qr, "baseline_vs_fixed_vllm_FP8", torch.uint8),
                  old.bit_comparison(torch, s0, sr, "baseline_vs_fixed_vllm_scales", torch.int32)]
        graph_records = []
        if m >= 257:
            # Legacy binary is the immutable anchor, always heuristic/s8.
            legacy_workspace = torch.empty(args.workspace_mib << 20, dtype=torch.uint8, device="cuda")
            legacy = adapter.Backend(args.legacy_library, legacy_workspace, 8)
            legacy.prepare(m)
            ql, sl = old.output_buffers(torch, m)
            def run_legacy(): legacy.launch(ql, sl, a, asc, pb, pbs)
            run_legacy()
            torch.cuda.synchronize()
            checks += [old.bit_comparison(torch, ql, q0, "legacy_v6_FP8", torch.uint8),
                       old.bit_comparison(torch, sl, s0, "legacy_v6_scales", torch.int32)]
            if distribution == "normal" and m in graph_shapes:
                anchor_graphs = {"official": old.capture(torch, run_base),
                                 "legacy_v6": old.capture(torch, run_legacy)}
                original_a = a.view(torch.uint8).clone()
                original_asc = asc.contiguous().view(torch.uint8).clone()
                prior = {"official": q0.view(torch.uint8).clone(),
                         "legacy_v6": ql.view(torch.uint8).clone()}
                next_a, next_sc = old.make_activation(torch, per_token_group_quant_fp8,
                                                       m, args.input_seed + 9000 + m)
                a.copy_(next_a)
                asc.copy_(next_sc)
                anchor_input_changed = not bool(torch.equal(original_a, a.view(torch.uint8)))
                for graph in anchor_graphs.values(): graph.replay()
                fresh = official_ref(a, asc)
                fq, fs = old.output_buffers(torch, m)
                official_quant(fresh, fq, fs)
                torch.cuda.synchronize()
                for label, output_q, output_s in (("official", q0, s0), ("legacy_v6", ql, sl)):
                    gc = [old.bit_comparison(torch, output_q, fq, f"{label}_replay_FP8", torch.uint8),
                          old.bit_comparison(torch, output_s, fs, f"{label}_replay_scale", torch.int32)]
                    changed = not bool(torch.equal(prior[label], output_q.view(torch.uint8)))
                    graph_records.append({"arm": label, "input_changed": anchor_input_changed,
                                          "output_changed": changed,
                                          "pass": anchor_input_changed and changed and all(c["pass"] for c in gc),
                                          "comparisons": gc})
                a.copy_(original_a.view(torch.float8_e4m3fn))
                asc.copy_(original_asc.view(torch.float32).reshape(m, K // GROUP))
                run_base()
                run_legacy()
                del anchor_graphs, original_a, original_asc, prior, next_a, next_sc, fresh, fq, fs
            # BF16 diagnostics are required for the full gate, but omitted
            # from a single-arm NCU witness to avoid six unrelated replays.
            for raster in (() if args.ncu_arm else RASTERS):
                for swizzle in (1, 8):
                    r = RASTERS[raster]
                    st0 = scheduler.prepare("official_bf16", m, r, swizzle)
                    x = torch.empty_like(x0)
                    scheduler.call(st0, x, None, a, asc, b, bs)
                    stp = scheduler.prepare("paired_bf16", m, r, swizzle)
                    xp = torch.empty_like(x0)
                    scheduler.call(stp, xp, None, a, asc, pb, pbs)
                    canonical = old.inverse_packed_output(torch, xp)
                    torch.cuda.synchronize()
                    label = f"{raster}_s{swizzle}"
                    checks += [old.bit_comparison(torch, x, xr, f"official_tile_{label}_BF16", torch.int16),
                               old.bit_comparison(torch, canonical, xr, f"paired_tile_inverse_{label}_BF16", torch.int16)]
                    del st0, stp, x, xp, canonical
            # All six fused scheduler controls must satisfy the same exact
            # FP8/scale and downstream BF16 semantics as the fixed op.
            numeric_arms = ARMS if not args.ncu_arm else ((args.ncu_arm,) if args.ncu_arm in ARMS else ())
            for arm in numeric_arms:
                raster, swizzle = parse_arm(arm)
                state = scheduler.prepare("fused_fp8", m, raster, swizzle)
                q, sc = old.output_buffers(torch, m)
                def call(): scheduler.call(state, q, sc, a, asc, pb, pbs)
                call()
                torch.cuda.synchronize()
                checks += [old.bit_comparison(torch, q, q0, f"{arm}_FP8", torch.uint8),
                           old.bit_comparison(torch, sc, s0, f"{arm}_scales", torch.int32)]
                if distribution == "normal":
                    down0 = official_ref(q0, s0, down_b, down_bs)
                    down1 = official_ref(q, sc, down_b, down_bs)
                    torch.cuda.synchronize()
                    checks.append(old.bit_comparison(torch, down1, down0,
                                                     f"{arm}_downstream_BF16", torch.int16))
                    del down0, down1
                if distribution == "normal" and m in graph_shapes:
                    graph = old.capture(torch, call)
                    original = a.view(torch.uint8).clone()
                    original_asc = asc.contiguous().view(torch.uint8).clone()
                    prior_q = q.view(torch.uint8).clone()
                    next_a, next_sc = old.make_activation(torch, per_token_group_quant_fp8,
                                                           m, args.input_seed + 9000 + m)
                    a.copy_(next_a)
                    asc.copy_(next_sc)
                    graph.replay()
                    fresh = official_ref(a, asc)
                    fq, fs = old.output_buffers(torch, m)
                    official_quant(fresh, fq, fs)
                    torch.cuda.synchronize()
                    graph_checks = [old.bit_comparison(torch, q, fq, f"{arm}_replay_FP8", torch.uint8),
                                    old.bit_comparison(torch, sc, fs, f"{arm}_replay_scale", torch.int32)]
                    changed = not bool(torch.equal(original, a.view(torch.uint8)))
                    output_changed = not bool(torch.equal(prior_q, q.view(torch.uint8)))
                    graph_records.append({"arm": arm, "input_changed": changed,
                                          "output_changed": output_changed,
                                          "pass": changed and output_changed and all(c["pass"] for c in graph_checks),
                                          "comparisons": graph_checks})
                    a.copy_(original.view(torch.float8_e4m3fn))
                    asc.copy_(original_asc.view(torch.float32).reshape(m, K // GROUP))
                    del graph, next_a, next_sc, fresh, fq, fs, original, original_asc, prior_q
                del state, q, sc
            del legacy_workspace, legacy, ql, sl
        passed = all(c["pass"] for c in checks) and all(g["pass"] for g in graph_records)
        emit("numerics", {"M": m, "distribution": distribution, "pass": passed,
                          "input_seed": args.input_seed + 100 + m,
                          "input_sha256": input_hash,
                          "comparisons": checks, "graph_replay": graph_records,
                          "fused_scheduler_supported": m >= 257})
        all_pass = all_pass and passed
        del a, asc, base, x0, q0, s0, xr, qr, sr
        torch.cuda.empty_cache()
    if not all_pass:
        emit("summary", {"pass": False, "numeric_cases": len(numeric_cases),
                         "timings_emitted": False, "reason": "exact numerical gate failed"})
        return 1

    flush = torch.empty(max(2 * l2 + 1, 256 << 20), dtype=torch.uint8, device="cuda") if not args.ncu_arm else None
    for m in args.timed_m:
        a, asc = old.make_activation(torch, per_token_group_quant_fp8, m, args.input_seed + 100 + m)
        input_hash = input_fingerprint(torch, a, asc)
        base, x0, q0, s0, run_base = controls(m, a, asc)
        calls = {"official": run_base}
        arm_outputs = {"official": (q0, s0)}
        owned = [base, x0, q0, s0]
        if not args.ncu_arm or args.ncu_arm == "legacy_v6":
            legacy_workspace = torch.empty(args.workspace_mib << 20, dtype=torch.uint8, device="cuda")
            legacy = adapter.Backend(args.legacy_library, legacy_workspace, 8)
            legacy.prepare(m)
            ql, sl = old.output_buffers(torch, m)
            def run_legacy(): legacy.launch(ql, sl, a, asc, pb, pbs)
            calls["legacy_v6"] = run_legacy
            arm_outputs["legacy_v6"] = (ql, sl)
            owned.extend((legacy_workspace, legacy, ql, sl))
        timing_arms = ARMS if not args.ncu_arm else ((args.ncu_arm,) if args.ncu_arm in ARMS else ())
        for arm in timing_arms:
            raster, swizzle = parse_arm(arm)
            state = scheduler.prepare("fused_fp8", m, raster, swizzle)
            q, sc = old.output_buffers(torch, m)
            def make_call(state=state, q=q, sc=sc):
                return lambda: scheduler.call(state, q, sc, a, asc, pb, pbs)
            calls[arm] = make_call()
            arm_outputs[arm] = (q, sc)
            owned.extend((state, q, sc))
        if args.ncu_arm:
            for _ in range(3): calls[args.ncu_arm]()
            torch.cuda.synchronize()
            profile_once(torch, runtime, args.ncu_arm, calls[args.ncu_arm])
            emit("profiled_once", {"M": m, "arm": args.ncu_arm,
                                   "nvtx": f"SM120_SCHEDULER:{args.ncu_arm}",
                                   "input_seed": args.input_seed + 100 + m,
                                   "input_sha256": input_hash,
                                   "complete_pipeline": True, "call_count": 1})
        else:
            estimated_live = sum(x.numel() * x.element_size() for x in (a, asc, b, bs, pb, pbs, x0, q0, s0, ql, sl))
            for item in owned:
                if isinstance(item, dict):
                    estimated_live += sum(x.numel() * x.element_size() for x in item.values() if isinstance(x, torch.Tensor))
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            if free_bytes < args.min_free_gib * (1 << 30):
                raise RuntimeError(f"insufficient free GPU memory before capture: {free_bytes} bytes")
            graphs = {name: old.capture(torch, call) for name, call in calls.items()}
            fresh = official_ref(a, asc)
            fq, fs = old.output_buffers(torch, m)
            official_quant(fresh, fq, fs)
            for graph in graphs.values(): graph.replay()
            torch.cuda.synchronize()
            capture_checks = []
            for name, (q, sc) in arm_outputs.items():
                capture_checks.extend((old.bit_comparison(torch, q, fq, f"{name}_timing_capture_FP8", torch.uint8),
                                       old.bit_comparison(torch, sc, fs, f"{name}_timing_capture_scales", torch.int32)))
            if not all(check["pass"] for check in capture_checks):
                raise RuntimeError("timing graph output failed fresh official reference; no events recorded")
            timing = measure_graphs(torch, graphs, flush, args.order_seed + m, args.repeats)
            emit("timing", {"M": m, "full_pipeline": True, "pass": True,
                            "input_seed": args.input_seed + 100 + m,
                            "input_sha256": input_hash,
                            "estimated_explicit_live_bytes_lower_bound": estimated_live,
                            "free_bytes_before_capture": free_bytes,
                            "total_gpu_bytes": total_bytes,
                            "timing_capture_checks": capture_checks,
                            "arm_count": len(calls), **timing})
            del graphs, fresh, fq, fs
        del a, asc, calls, arm_outputs, owned, base, x0, q0, s0
        if "legacy_workspace" in locals():
            del legacy_workspace, legacy, ql, sl
        torch.cuda.empty_cache()
    emit("summary", {"pass": True, "numeric_cases": len(numeric_cases),
                     "timings_emitted": not bool(args.ncu_arm), "ncu": bool(args.ncu_arm),
                     "timed_shapes": list(args.timed_m),
                     "changed_input_graph_shapes": list(graph_shapes)})
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-probe", type=Path, required=True)
    p.add_argument("--old-probe-sha256", default="")
    p.add_argument("--legacy-adapter", type=Path, required=True)
    p.add_argument("--legacy-adapter-sha256", default="")
    p.add_argument("--baseline-library", type=Path, required=True)
    p.add_argument("--baseline-sha256", default="")
    p.add_argument("--quant-library", type=Path, required=True)
    p.add_argument("--quant-sha256", default="")
    p.add_argument("--legacy-library", type=Path, required=True)
    p.add_argument("--legacy-sha256", default=LEGACY_SHA)
    p.add_argument("--scheduler-library", type=Path, required=True)
    p.add_argument("--scheduler-sha256", default="")
    p.add_argument("--gate-m", nargs="+", type=int, default=DEFAULT_GATE_M)
    p.add_argument("--graph-m", nargs="+", type=int, default=(257, 2741, 16160))
    p.add_argument("--timed-m", nargs="+", type=int, default=(2741, 16160))
    p.add_argument("--workspace-mib", type=int, default=64)
    p.add_argument("--input-seed", type=int, default=20261014)
    p.add_argument("--order-seed", type=int, default=20261015)
    p.add_argument("--repeats", type=int, default=31)
    p.add_argument("--min-free-gib", type=float, default=8.0,
                   help="fail before multi-arm graph capture if free memory is below this guard")
    p.add_argument("--ncu-arm", choices=("official", "legacy_v6") + ARMS)
    args = p.parse_args()
    if (args.repeats < 31 or args.workspace_mib < 1 or args.min_free_gib < 1 or
        any(not 257 <= m <= 32768 for m in args.timed_m) or
        any(not 1 <= m <= 32768 for m in args.gate_m) or
        any(not 257 <= m <= 32768 for m in args.graph_m) or
        (args.ncu_arm and len(args.timed_m) != 1)):
        p.error("invalid pairs/workspace/shapes or NCU shape count")
    try:
        return run(args)
    except Exception as exc:
        traceback.print_exc()
        emit("summary", {"pass": False, "error_type": type(exc).__name__, "error": str(exc),
                         "timings_valid": False})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
