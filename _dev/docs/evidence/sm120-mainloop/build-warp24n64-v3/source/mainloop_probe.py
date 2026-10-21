#!/usr/bin/env python3
"""Isolated multi-library FP8 complete-chain probe. GPU execution is by root."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEDULER_SHA = "f27f27083dec90dcf3884cd8190c876df9957c285a1cd3b97b265a4145ac67b8"
PIPELINE_SHA = "eed1a8d2fa5e97d0b0dfefcb504c6ca9d8493d1c00997788e1343aadd60c42ae"
BASELINE_SHA = "f7abfcd8c04ad9a93c924364beadd305f773b1f909bca013f2838d109f17b7b2"
QUANT_SHA = "087b7caa92c4742ece3875b6dd1421e25360b834ce4106f7fa166f90bc933b3f"
K, N, H, GROUP = 5120, 34816, 17408, 128
RASTERS = {"heuristic": 0, "along_m": 1, "along_n": 2}
VARIANTS = {"control42": [4, 2, 1], "warp24x2": [2, 4, 1], "warp24n64": [2, 4, 1]}
COPY_CONFIG = {"control42": (32, 4), "warp24x2": (32, 2), "warp24n64": (64, 4)}
NORMAL_M = (257, 2741, 2786, 16160, 32768)
FALLBACK_M = (1, 5, 16, 256)
SPECIAL = ("zero", "tiny", "large", "exceptional")
OFFICIAL_ARMS = ("official_vllm", "official_h8", "official_m1")
FUSED_ARMS = tuple(f"{v}_{r}_s{s}" for v in VARIANTS for r in RASTERS for s in (1, 8))
DIAGNOSTICS = tuple("paired_" + a for a in FUSED_ARMS)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned_import(path, expected, name):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"pinned dependency changed: {path}: {actual}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit(key, data):
    print(json.dumps({key: data}, sort_keys=True, allow_nan=False), flush=True)


def keyed(values):
    result = {}
    for value in values:
        label, sep, payload = value.partition("=")
        if not sep or not payload or label not in VARIANTS or label in result:
            raise ValueError(f"expected unique supported LABEL=VALUE: {value}")
        result[label] = payload
    return result


def parse_arm(arm):
    kind = "paired_bf16" if arm.startswith("paired_") else "fused_fp8"
    stem = arm.removeprefix("paired_")
    for label in VARIANTS:
        for raster, code in RASTERS.items():
            for swizzle in (1, 8):
                if stem == f"{label}_{raster}_s{swizzle}":
                    return label, kind, code, swizzle
    raise ValueError(f"unknown arm: {arm}")


def metadata_check(label, meta):
    perm_n, b_width = COPY_CONFIG[label]
    wanted = {"abi_version": 1, "N": N, "K": K, "cuda_arch": "sm_120a",
              "min_M": 257, "max_M": 32768, "same_m64_mainloop_type": True,
              "mainloop_variant": label, "m64_mma_atom_layout_mnk": VARIANTS[label],
              "m64_mma_permutation_n": perm_n, "fused_epilogue_tile": "64x256",
              "m64_smem_copy_atom_b": f"SM75_U32x{b_width}_LDSM_N",
              "m64_smem_copy_b_bytes_per_thread": b_width * 4}
    for key, value in wanted.items():
        if meta.get(key) != value:
            raise ValueError(f"{label}: metadata {key} differs: {meta.get(key)!r}")
    if meta.get("private_namespace") != f"pro6000d_mainloop_{label}":
        raise ValueError("library private namespace mismatch")
    for kind in ("paired_bf16", "fused_fp8", "official_bf16"):
        resource = meta.get("resources", {}).get(kind, {})
        if (resource.get("mainloop_stages") != 2 or resource.get("mma_threads") != 256
                or resource.get("max_threads_per_block") != 384
                or not 0 < resource.get("kernel_shared_storage_bytes", 0) <= 101376):
            raise ValueError(f"{label}/{kind}: unexpected stage/thread/shared budget")
    if (meta["resources"]["paired_bf16"]["mainloop_shared_storage_bytes"] !=
            meta["resources"]["fused_fp8"]["mainloop_shared_storage_bytes"]):
        raise ValueError("matched BF16/fused mainloop storage differs")


def run(args):
    # Hash/import only these frozen helper sources; neither import initializes CUDA.
    helper = pinned_import(args.scheduler_probe, SCHEDULER_SHA, "fixed_scheduler_helpers")
    old = pinned_import(args.old_probe, PIPELINE_SHA, "fixed_mainloop_pipeline_helpers")
    paths, expected = keyed(args.library), keyed(args.library_sha)
    if set(paths) != set(expected):
        raise ValueError("every library requires exactly one LABEL=SHA")
    if any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in expected.values()):
        raise ValueError("every library requires a lowercase 64-digit SHA256")
    hashes = {label: helper.check_sha(Path(path), expected[label]) for label, path in paths.items()}
    selected = list(args.arms or (*OFFICIAL_ARMS, *FUSED_ARMS))
    if len(selected) != len(set(selected)) or any(a not in (*OFFICIAL_ARMS, *FUSED_ARMS) for a in selected):
        raise ValueError("arms must be distinct explicit complete-chain labels")
    if args.mode == "ncu":
        selected = list(dict.fromkeys(("official_vllm", args.ncu_arm)))
    elif not set(OFFICIAL_ARMS) <= set(selected):
        raise ValueError("gate/timing must include all three official complete-chain controls")
    labels = {parse_arm(a)[0] for a in selected if a not in OFFICIAL_ARMS}
    if args.mode != "ncu":
        labels.add("control42")  # unchanged official tile control in one fixed library
    elif not labels and args.ncu_arm != "official_vllm":
        labels.add("control42")
    if not labels <= set(paths):
        raise ValueError(f"missing libraries: {labels - set(paths)}")
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("reserved SM120 GPU required")
    torch.backends.cuda.matmul.allow_tf32 = False
    libraries = {label: helper.SchedulerLibrary(torch, Path(paths[label])) for label in sorted(labels)}
    for label, library in libraries.items():
        metadata_check(label, library.metadata)
    official_library = libraries.get("control42") or next(iter(libraries.values()), None)
    dependencies = {"scheduler_probe": SCHEDULER_SHA, "old_probe": PIPELINE_SHA,
                    "baseline": helper.check_sha(args.baseline_library, BASELINE_SHA),
                    "quant": helper.check_sha(args.quant_library, QUANT_SHA),
                    "probe": sha(Path(__file__))}
    runtime = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
    l2 = old.l2_size(runtime, torch.cuda.current_device())
    if N * K <= l2:
        raise RuntimeError("fixed weight no longer exceeds device L2")
    b, bs = old.make_weight(torch, N, K, args.input_seed)
    pb, pbs = old.packed_weights(torch, b, bs)
    rb = pb.view(torch.uint8).reshape(H // GROUP, 2, GROUP, K).transpose(0, 1).contiguous().reshape(N, K)
    rs = pbs.reshape(H // GROUP, 2, K // GROUP).transpose(0, 1).contiguous().reshape(N // GROUP, K // GROUP)
    if not torch.equal(rb, b.view(torch.uint8)) or not torch.equal(rs.view(torch.int32), bs.view(torch.int32)):
        raise RuntimeError("packed-weight roundtrip bits differ")
    del rb, rs
    down_b, down_bs = old.make_weight(torch, K, H, args.input_seed + 1)
    weight_hashes = {"canonical": helper.input_fingerprint(torch, b, bs),
                     "packed": helper.input_fingerprint(torch, pb, pbs),
                     "down": helper.input_fingerprint(torch, down_b, down_bs)}
    cases = ([(args.timed_m[0], "normal")] if args.mode == "ncu" else
             [(m, "normal") for m in (*FALLBACK_M, *NORMAL_M)] + [(257, d) for d in SPECIAL])
    graph_m = set(NORMAL_M) if args.mode != "ncu" else set(args.timed_m)
    meta = {"schema": "sm120-mainloop-v1", "mode": args.mode,
            "scope": "synthetic FP8 operator; no model/API gain or all-input proof",
            "arms": selected, "libraries_sha256": hashes,
            "loaded_libraries": sorted(libraries), "library_metadata": {k: v.metadata for k, v in libraries.items()},
            "official_control_library": next((k for k, v in libraries.items() if v is official_library), None),
            "dependencies_sha256": dependencies, "official_ops_sha256": sha(Path(ops.__file__)),
            "weight_sha256": weight_hashes, "packed_roundtrip_pass": True,
            "cases": [{"M": m, "distribution": d} for m, d in cases],
            "timed_shapes": list(args.timed_m) if args.mode != "gate" else [],
            "graph_shapes": sorted(graph_m), "input_seed": args.input_seed,
            "input_seed_rule": "seed+100+M", "changed_seed_rule": "seed+9000+M",
            "order_seed": args.order_seed, "repeats": args.repeats,
            "minimum_free_bytes": args.min_free_gib * (1 << 30),
            "settings": {a: ([0, 8] if a in ("official_vllm", "official_h8") else
                              [1, 1] if a == "official_m1" else list(parse_arm(a)[2:])) for a in selected},
            "device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda, "vllm": __import__("vllm").__version__,
            "cuda_versions": helper.cuda_versions(runtime), "l2_bytes": l2,
            "cache_scope": "owned >2xL2 disturbance outside events; not fixed cold cache",
            "ncu_selected_arm": args.ncu_arm, "legacy_libraries_loaded": False}
    emit("metadata", meta)

    def free_guard():
        free, total = torch.cuda.mem_get_info()
        if free < args.min_free_gib * (1 << 30):
            raise RuntimeError(f"free memory below guard: {free}")
        return free, total

    def official_ref(a, asc, weight=b, scales=bs):
        return ops.cutlass_scaled_mm(a, weight.T, out_dtype=torch.bfloat16,
                                     scale_a=asc, scale_b=scales.T)

    def reference(a, asc):
        x = official_ref(a, asc)
        q, sc = old.output_buffers(torch, a.shape[0])
        torch.ops._C.silu_and_mul_per_block_quant(q, x, sc, GROUP, None, True)
        torch.cuda.synchronize()
        return {"BF16": x, "FP8": q, "scales": sc}

    def base_instance():
        return old.Baseline(torch, args.baseline_library, args.quant_library, args.workspace_mib << 20, 8)

    def make_arm(arm, m, a, asc, base):
        free_guard()
        owned = []
        if arm == "official_vllm":
            x = torch.empty((m, N), dtype=torch.bfloat16, device="cuda")
            q, sc = old.output_buffers(torch, m)
            def call():
                base.gemm(x, a, asc, b, bs)
                base.quant(x, q, sc)
            outputs = {"BF16": x, "FP8": q, "scales": sc}
        elif arm in ("official_h8", "official_m1"):
            r, s = (0, 8) if arm == "official_h8" else (1, 1)
            state = official_library.prepare("official_bf16", m, r, s)
            owned.append(state)
            x = torch.empty((m, N), dtype=torch.bfloat16, device="cuda")
            q, sc = old.output_buffers(torch, m)
            def call():
                official_library.call(state, x, None, a, asc, b, bs)
                base.quant(x, q, sc)
            outputs = {"BF16": x, "FP8": q, "scales": sc}
        else:
            label, kind, r, s = parse_arm(arm)
            library = libraries[label]
            state = library.prepare(kind, m, r, s)
            owned.append(state)
            if kind == "paired_bf16":
                x = torch.empty((m, N), dtype=torch.bfloat16, device="cuda")
                def call(): library.call(state, x, None, a, asc, pb, pbs)
                outputs = {"paired_BF16": x}
            else:
                q, sc = old.output_buffers(torch, m)
                def call(): library.call(state, q, sc, a, asc, pb, pbs)
                outputs = {"FP8": q, "scales": sc}
        tensors = list(outputs.values())
        for state in owned:
            tensors.extend(v for v in state.values() if isinstance(v, torch.Tensor))
        old.assert_nonalias(tensors)
        return call, outputs, owned

    def compare(outputs, ref, prefix):
        checks = []
        for kind, tensor in outputs.items():
            value = old.inverse_packed_output(torch, tensor) if kind == "paired_BF16" else tensor
            target = "BF16" if kind == "paired_BF16" else kind
            dtype = {"BF16": torch.int16, "FP8": torch.uint8, "scales": torch.int32}[target]
            checks.append(old.bit_comparison(torch, value, ref[target], f"{prefix}_{kind}", dtype))
        return checks

    def verify_arm(arm, m, distribution, a, asc, ref, base):
        call, outputs, owned = make_arm(arm, m, a, asc, base)
        free_guard()
        call()
        torch.cuda.synchronize()
        checks = compare(outputs, ref, arm)
        # Matched ordinary BF16 mainloop is mandatory, independent of Q agreement.
        if arm in FUSED_ARMS:
            label, _, r, s = parse_arm(arm)
            state = libraries[label].prepare("paired_bf16", m, r, s)
            xp = torch.empty((m, N), dtype=torch.bfloat16, device="cuda")
            libraries[label].call(state, xp, None, a, asc, pb, pbs)
            torch.cuda.synchronize()
            checks += compare({"paired_BF16": xp}, ref, arm + "_mainloop")
            del state, xp
        if distribution == "normal" and "FP8" in outputs:
            d0 = official_ref(ref["FP8"], ref["scales"], down_b, down_bs)
            d1 = official_ref(outputs["FP8"], outputs["scales"], down_b, down_bs)
            torch.cuda.synchronize()
            checks.append(old.bit_comparison(torch, d1, d0, arm + "_downstream_BF16", torch.int16))
            del d0, d1
        graphs = []
        if distribution == "normal" and m in graph_m:
            graph = old.capture(torch, call)
            original_a = a.view(torch.uint8).clone()
            original_sc = asc.clone()
            principal = outputs.get("FP8", outputs.get("paired_BF16", outputs.get("BF16")))
            previous = principal.view(torch.uint8).clone()
            next_a, next_sc = old.make_activation(torch, per_token_group_quant_fp8, m, args.input_seed + 9000 + m)
            a.copy_(next_a)
            asc.copy_(next_sc)
            bytes_changed = not torch.equal(original_a, a.view(torch.uint8))
            scales_changed = not torch.equal(original_sc.view(torch.int32), asc.view(torch.int32))
            graph.replay()
            fresh = reference(a, asc)
            gc_checks = compare(outputs, fresh, arm + "_replay")
            output_changed = not torch.equal(previous, principal.view(torch.uint8))
            graphs.append({"arm": arm, "input_bytes_changed": bool(bytes_changed),
                           "input_scales_changed": bool(scales_changed), "output_changed": bool(output_changed),
                           "changed_input_sha256": helper.input_fingerprint(torch, a, asc),
                           "comparisons": gc_checks,
                           "pass": bool(bytes_changed and scales_changed and output_changed and all(c["pass"] for c in gc_checks))})
            a.copy_(original_a.view(torch.float8_e4m3fn))
            asc.copy_(original_sc)
        return checks, graphs

    def numeric_case(m, distribution):
        a, asc = old.make_activation(torch, per_token_group_quant_fp8, m, args.input_seed + 100 + m, distribution)
        input_hash = helper.input_fingerprint(torch, a, asc)
        ref, base = reference(a, asc), base_instance()
        arms = selected if m > 256 else ["official_vllm"]
        checks, graphs = [], []
        for arm in arms:
            c, g = verify_arm(arm, m, distribution, a, asc, ref, base)
            checks.extend(c)
            graphs.extend(g)
            gc.collect()
            torch.cuda.empty_cache()
        row = {"M": m, "distribution": distribution, "input_seed": args.input_seed + 100 + m,
               "input_sha256": input_hash, "arms": arms, "fallback_only": m <= 256,
               "comparisons": checks, "graph_replay": graphs,
               "pass": all(c["pass"] for c in checks) and all(g["pass"] for g in graphs)}
        emit("numerics", row)
        return row["pass"]

    # Complete gates precede every timing; NCU explicitly gates only its selected shape.
    passed = True
    for m, distribution in cases:
        passed = numeric_case(m, distribution) and passed
        gc.collect()
        torch.cuda.empty_cache()
    if not passed:
        emit("summary", {"pass": False, "numeric_cases": len(cases), "timings_valid": False})
        return 1

    def measured_shape(m):
        a, asc = old.make_activation(torch, per_token_group_quant_fp8, m, args.input_seed + 100 + m)
        input_hash = helper.input_fingerprint(torch, a, asc)
        base, ref = base_instance(), reference(a, asc)
        calls, outputs, ownership = {}, {}, []
        for arm in selected:
            calls[arm], outputs[arm], owned = make_arm(arm, m, a, asc, base)
            ownership.extend(owned)
        mutable = [base.workspace]
        mutable.extend(t for o in outputs.values() for t in o.values())
        mutable.extend(t for state in ownership for t in state.values() if isinstance(t, torch.Tensor))
        old.assert_nonalias(mutable)
        free, total = free_guard()
        if args.mode == "ncu":
            arm = args.ncu_arm
            for _ in range(3): calls[arm]()
            torch.cuda.synchronize()
            runtime.cudaProfilerStart.argtypes = []
            runtime.cudaProfilerStart.restype = ctypes.c_int
            runtime.cudaProfilerStop.argtypes = []
            runtime.cudaProfilerStop.restype = ctypes.c_int
            if runtime.cudaProfilerStart():
                raise RuntimeError("cudaProfilerStart failed")
            try:
                with torch.cuda.nvtx.range("SM120_MAINLOOP:" + arm):
                    calls[arm]()
                torch.cuda.synchronize()
            finally:
                if runtime.cudaProfilerStop():
                    raise RuntimeError("cudaProfilerStop failed")
            emit("profiled_once", {"M": m, "arm": arm, "call_count": 1,
                                   "nvtx": "SM120_MAINLOOP:" + arm, "input_sha256": input_hash,
                                   "input_seed": args.input_seed + 100 + m,
                                   "complete_pipeline": arm not in DIAGNOSTICS,
                                   "scope": "paired_BF16_GEMM_only" if arm in DIAGNOSTICS else "complete_chain"})
            return
        flush = torch.empty(max(2 * l2 + 1, 256 << 20), dtype=torch.uint8, device="cuda")
        free_guard()
        graphs = {arm: old.capture(torch, call) for arm, call in calls.items()}
        checks = []
        for arm, graph in graphs.items():
            graph.replay()
        torch.cuda.synchronize()
        for arm, values in outputs.items():
            checks.extend(compare(values, ref, arm + "_timing_capture"))
        if not all(c["pass"] for c in checks):
            raise RuntimeError("fresh timing-capture output mismatch; no timing allowed")
        timing = helper.measure_graphs(torch, graphs, flush, args.order_seed + m, args.repeats)
        emit("timing", {"M": m, "input_seed": args.input_seed + 100 + m,
                        "input_sha256": input_hash, "arms": selected, "pass": True,
                        "full_pipeline": True, "timing_capture_checks": checks,
                        "free_bytes_before_capture": free, "total_gpu_bytes": total,
                        "explicit_mutable_buffer_bytes": sum(t.numel() * t.element_size() for t in mutable),
                        **timing})

    if args.mode != "gate":
        for m in args.timed_m:
            measured_shape(m)
            gc.collect()
            torch.cuda.empty_cache()
    emit("summary", {"pass": True, "mode": args.mode, "numeric_cases": len(cases),
                     "timed_shapes": list(args.timed_m) if args.mode != "gate" else [],
                     "timings_emitted": args.mode == "timing", "ncu": args.mode == "ncu"})
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("gate", "timing", "ncu"), default="gate")
    p.add_argument("--library", action="append", required=True, help="control42/warp24x2/warp24n64=PATH")
    p.add_argument("--library-sha", action="append", required=True, help="LABEL=SHA256; mandatory")
    p.add_argument("--scheduler-probe", type=Path, default=HERE.parent / "sm120-scheduler/scheduler_probe.py")
    p.add_argument("--old-probe", type=Path, default=HERE.parent / "gemm-epilogue/pipeline_probe.py")
    p.add_argument("--baseline-library", type=Path, required=True)
    p.add_argument("--quant-library", type=Path, required=True)
    p.add_argument("--arms", nargs="+", choices=(*OFFICIAL_ARMS, *FUSED_ARMS))
    p.add_argument("--timed-m", nargs="+", type=int, default=(2741, 16160))
    p.add_argument("--ncu-arm", choices=(*OFFICIAL_ARMS, *FUSED_ARMS, *DIAGNOSTICS))
    p.add_argument("--input-seed", type=int, default=20261014)
    p.add_argument("--order-seed", type=int, default=20261015)
    p.add_argument("--workspace-mib", type=int, default=64)
    p.add_argument("--repeats", type=int, default=31)
    p.add_argument("--min-free-gib", type=float, default=8)
    args = p.parse_args()
    if (args.repeats < 31 or args.min_free_gib < 8 or args.workspace_mib < 1
            or len(set(args.timed_m)) != len(args.timed_m)
            or any(m not in NORMAL_M for m in args.timed_m)
            or (args.mode == "ncu" and (len(args.timed_m) != 1 or not args.ncu_arm))
            or (args.mode != "ncu" and args.ncu_arm)):
        p.error("invalid repeats, memory guard, shapes or NCU selection")
    try:
        return run(args)
    except Exception as exc:
        traceback.print_exc()
        emit("summary", {"pass": False, "timings_valid": False,
                         "error_type": type(exc).__name__, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
