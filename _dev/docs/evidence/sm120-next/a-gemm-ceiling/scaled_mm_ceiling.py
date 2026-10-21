#!/usr/bin/env python3
"""Task A2: cuBLASLt (torch._scaled_mm) FP8 GEMM ceiling vs fixed vLLM CUTLASS blockwise.

Runs inside the fixed vLLM 0.31.0 image (torch 2.13.0+cu130) on the SM120 GPU.
Speed only: random FP8 operands, no numeric claims. NOT YET RUN when written.

For every (shape, M): try each scale mode (accepted / rejected + exception text),
capture each accepted arm plus the CUTLASS blockwise reference in CUDA graphs, then
time all arms in ONE process with a random arm order per pair (default 31 pairs),
warm and cache-perturbed (>2xL2 buffer written outside the timed events). Every
sample records the NVML SM clock averaged over the sample window; summaries report
cycles = ms * mean clock. Output is JSONL (kind = meta | attempt | sample | summary |
decision) so downstream analysis can recompute everything.
"""
from __future__ import annotations

import argparse
import bisect
import ctypes
import ctypes.util
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

GROUP = 128
SHAPES = {"gate_up": (34816, 5120), "down": (5120, 17408)}  # name -> (N, K)
DEFAULT_M = (2741, 2786, 16160, 32768)
CUTLASS_REF_TFLOPS = 617.0  # 14-sm120-scheduler-epilogue.md section 4 (M=16160 gate_up)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Out:
    def __init__(self, path):
        self.f = Path(path).open("w")

    def row(self, kind, **kw):
        kw["kind"] = kind
        line = json.dumps(kw, sort_keys=True, allow_nan=False, default=str)
        self.f.write(line + "\n")
        self.f.flush()
        if kind in ("attempt", "summary", "decision", "meta"):
            print(line, flush=True)


# ---------------------------------------------------------------- NVML (ctypes)
class Nvml:
    """Minimal NVML wrapper; every failure degrades to clock=None, never aborts."""

    def __init__(self, pci_bus_id=None, index=0):
        self.lib = None
        self.handle = ctypes.c_void_p()
        self.error = None
        try:
            name = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
            self.lib = ctypes.CDLL(name)
            if self.lib.nvmlInit_v2():
                raise RuntimeError("nvmlInit_v2 failed")
            rc = 1
            if pci_bus_id:
                rc = self.lib.nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id.encode(), ctypes.byref(self.handle))
            if rc:
                rc = self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle))
            if rc:
                raise RuntimeError(f"cannot get NVML handle rc={rc}")
        except Exception as exc:  # noqa: BLE001
            self.lib = None
            self.error = f"{type(exc).__name__}: {exc}"

    def sm_mhz(self):
        if self.lib is None:
            return None
        v = ctypes.c_uint()
        if self.lib.nvmlDeviceGetClockInfo(self.handle, 1, ctypes.byref(v)):  # 1 = NVML_CLOCK_SM
            return None
        return int(v.value)

    def max_sm_mhz(self):
        if self.lib is None:
            return None
        v = ctypes.c_uint()
        if self.lib.nvmlDeviceGetMaxClockInfo(self.handle, 1, ctypes.byref(v)):
            return None
        return int(v.value)


class ClockPoller:
    """Background poll (~2 ms); window_mean(t0, t1) averages samples inside a host window."""

    def __init__(self, nvml, period=0.002):
        self.nvml, self.period = nvml, period
        self.t, self.v = [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            mhz = self.nvml.sm_mhz()
            if mhz is not None:
                self.t.append(time.perf_counter())
                self.v.append(mhz)
            time.sleep(self.period)

    def __enter__(self):
        if self.nvml.lib is not None:
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join()

    def window_mean(self, t0, t1, pad=0.008):
        # padded window: a graph replay (~1-10 ms) is shorter than/near the ~2 ms NVML poll + GIL jitter
        t0, t1 = t0 - pad, t1 + pad
        i, j = bisect.bisect_left(self.t, t0), bisect.bisect_right(self.t, t1)
        vals = self.v[i:j]
        if vals:
            return sum(vals) / len(vals), len(vals)
        if self.v:  # window shorter than the poll period: nearest earlier sample
            k = max(0, min(len(self.v) - 1, bisect.bisect_left(self.t, t1) - 1))
            return float(self.v[k]), 0
        return None, 0


# ---------------------------------------------------------------- helpers
def l2_bytes(torch):
    return torch.cuda.get_device_properties(0).L2_cache_size  # expected 128 MiB


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


def rand_fp8(torch, shape, gen_seed):
    torch.manual_seed(gen_seed)
    rows = shape[0]
    out = torch.empty(shape, dtype=torch.float8_e4m3fn, device="cuda")
    step = max(1, (1 << 28) // shape[1])  # chunk to bound bf16 temporaries
    for i in range(0, rows, step):
        out[i:i + step] = (torch.randn((min(step, rows - i), shape[1]), device="cuda",
                                       dtype=torch.bfloat16) * 0.5).to(torch.float8_e4m3fn)
    return out


def pos_scales(torch, shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda").uniform_(0.5, 1.5)


def ceil_div(a, b):
    return -(-a // b)


def blocked_e8m0(torch, rows, k_blocks):
    """Swizzled (to_blocked) MXFP8 scale buffer: 128x4 tiles, flattened, random UE8M0 near 2^0."""
    rr, cc = ceil_div(rows, 128) * 128, ceil_div(k_blocks, 4) * 4
    raw = torch.randint(124, 130, (rr * cc,), dtype=torch.uint8, device="cuda")
    return raw.view(torch.float8_e8m0fnu)


def build_arms(torch, ops, m, n, k, seed):
    """Return (arms dict name -> callable, attempts list). Callables take no args."""
    attempts, arms = [], {}
    a = rand_fp8(torch, (m, k), seed)
    w = rand_fp8(torch, (n, k), seed + 1)  # [N,K]; w.t() is the (K,N) column-major B operand
    bt = w.t()
    a_scale_cols = pos_scales(torch, (k // GROUP, m)).t()           # [M,K/128] stride (1,M)
    w_scale = pos_scales(torch, (n // GROUP, k // GROUP))           # [N/128,K/128]
    keep = (a, w, a_scale_cols, w_scale)

    def cutlass():
        return ops.cutlass_scaled_mm(a, bt, out_dtype=torch.bfloat16,
                                     scale_a=a_scale_cols, scale_b=w_scale.T)

    s32 = lambda: torch.ones((), dtype=torch.float32, device="cuda")  # noqa: E731
    sa_t, sb_t = s32(), s32()
    sa_r = pos_scales(torch, (m, 1))
    sb_r = pos_scales(torch, (1, n))
    mx_k = k // 32
    sa_mx, sb_mx = blocked_e8m0(torch, m, mx_k), blocked_e8m0(torch, n, mx_k)
    b_blk_cm = w_scale.T                                             # [K/128,N/128] stride (1,K/128)
    b_blk_rm = w_scale.T.contiguous()                                # [K/128,N/128] row-major
    b_1x128 = pos_scales(torch, (k // GROUP, n)).t().contiguous().t()  # [N,K/128] -> passed as (K/128,N)
    b_1x128_c = pos_scales(torch, (k // GROUP, n))                   # [K/128,N] contiguous (outer-dim N major)
    candidates = {
        "cublas_tensorwise": lambda: torch._scaled_mm(a, bt, sa_t, sb_t, out_dtype=torch.bfloat16),
        "cublas_rowwise": lambda: torch._scaled_mm(a, bt, sa_r, sb_r, out_dtype=torch.bfloat16),
        "cublas_mxfp8_vec32": lambda: torch._scaled_mm(a, bt, sa_mx, sb_mx, out_dtype=torch.bfloat16),
        "cublas_blk_1x128_128x128_colmajorB": lambda: torch._scaled_mm(
            a, bt, a_scale_cols, b_blk_cm, out_dtype=torch.bfloat16),
        "cublas_blk_1x128_128x128_rowmajorB": lambda: torch._scaled_mm(
            a, bt, a_scale_cols, b_blk_rm, out_dtype=torch.bfloat16),
        "cublas_blk_1x128_1x128": lambda: torch._scaled_mm(
            a, bt, a_scale_cols, b_1x128, out_dtype=torch.bfloat16),
        "cublas_blk_1x128_1x128_contigB": lambda: torch._scaled_mm(
            a, bt, a_scale_cols, b_1x128_c, out_dtype=torch.bfloat16),
    }
    nnf = getattr(torch.nn.functional, "scaled_mm", None)
    st = getattr(torch.nn.functional, "ScalingType", None)
    sw = getattr(torch.nn.functional, "SwizzleType", None)
    if nnf is not None and st is not None and sw is not None:
        # SPECULATIVE signature (torch >= 2.10 nightly API as remembered); rejection text is the datum.
        candidates["cublas_mxfp8_nn_functional"] = lambda: nnf(
            a, bt, sa_mx, st.BlockWise1x32, sb_mx, st.BlockWise1x32,
            swizzle_a=sw.SWIZZLE_32_4_4, swizzle_b=sw.SWIZZLE_32_4_4, output_dtype=torch.bfloat16)
        # block-wise through the v2 API (per-operand recipes); layouts as the v1 error text asks for
        for nm, sb, rb in (("v2_blk_1x128_128x128_colmajorB", b_blk_cm, st.BlockWise128x128),
                           ("v2_blk_1x128_128x128_rowmajorB", b_blk_rm, st.BlockWise128x128),
                           ("v2_blk_1x128_1x128_colmajorB", b_1x128, st.BlockWise1x128),
                           ("v2_blk_1x128_1x128_contigB", b_1x128_c, st.BlockWise1x128)):
            candidates["cublas_" + nm] = (lambda sb=sb, rb=rb: nnf(
                a, bt, a_scale_cols, st.BlockWise1x128, sb, rb, output_dtype=torch.bfloat16))

    for name, call in [("cutlass_blockwise", cutlass), *candidates.items()]:
        row = {"arm": name, "m": m, "n": n, "k": k}
        try:
            out = call()
            torch.cuda.synchronize()
            ok = tuple(out.shape) == (m, n)
            row.update(accepted=bool(ok), out_dtype=str(out.dtype), out_shape=list(out.shape))
            del out
            if ok:
                arms[name] = call
        except Exception as exc:  # noqa: BLE001 - the exception text is the result
            row.update(accepted=False, exception_type=type(exc).__name__,
                       exception=str(exc)[:2500])
            try:
                torch.cuda.synchronize()
            except Exception as sync_exc:  # noqa: BLE001
                row["poisoned_context"] = str(sync_exc)[:200]
                attempts.append(row)
                raise
        attempts.append(row)
    return arms, attempts, keep


def timing(torch, graphs, flush, rng, pairs, poller, mode):
    """Return samples list of dicts. One random permutation of all arms per pair."""
    rows = []
    for pair in range(pairs):
        order = list(graphs)
        rng.shuffle(order)
        for pos, name in enumerate(order):
            if mode == "cold":
                flush.fill_(pair & 255)  # >2xL2 write, outside timed events
                torch.cuda.synchronize()
            b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            b.record()
            graphs[name].replay()
            e.record()
            e.synchronize()
            t1 = time.perf_counter()
            mhz, nsamp = poller.window_mean(t0, t1)
            rows.append({"mode": mode, "pair": pair, "pos": pos, "arm": name,
                         "ms": b.elapsed_time(e), "sm_mhz": mhz, "clock_samples": nsamp})
    return rows


def summarize(out, rows, shape_name, m, n, k):
    flop = 2.0 * m * n * k
    result = {}
    for mode in ("warm", "cold"):
        sel = [r for r in rows if r["mode"] == mode]
        arms = sorted({r["arm"] for r in sel})
        by = {a: [r for r in sel if r["arm"] == a] for a in arms}
        ref = by.get("cutlass_blockwise")
        for a in arms:
            ms = [r["ms"] for r in by[a]]
            mhz = [r["sm_mhz"] for r in by[a] if r["sm_mhz"] is not None]
            med = statistics.median(ms)
            row = {"shape": shape_name, "m": m, "n": n, "k": k, "mode": mode, "arm": a,
                   "pairs": len(ms), "median_ms": med, "min_ms": min(ms),
                   "median_tflops": flop / (med * 1e-3) / 1e12,
                   "mean_sm_mhz": (sum(mhz) / len(mhz)) if mhz else None}
            if mhz:
                row["median_sm_mhz"] = statistics.median(mhz)
                row["median_cycles"] = med * 1e-3 * (sum(mhz) / len(mhz)) * 1e6
            if ref is not None and a != "cutlass_blockwise":
                # ratio = cutlass_ms / arm_ms; > 1 means the arm is faster than CUTLASS
                ratios = [x["ms"] / y["ms"] for x, y in
                          zip(sorted(ref, key=lambda r: r["pair"]), sorted(by[a], key=lambda r: r["pair"]))]
                row["paired_cutlass_over_arm_median"] = statistics.median(ratios)
                row["paired_cutlass_over_arm_min"] = min(ratios)
                row["paired_cutlass_over_arm_max"] = max(ratios)
                row["paired_ratios"] = ratios
            out.row("summary", **row)
            result[(mode, a)] = row
    return result


def lock_clocks(out, nvml):
    """Record only whether `nvidia-smi -lgc` works; never required, never raises."""
    mhz = nvml.max_sm_mhz()
    cmd = ["nvidia-smi", "-lgc", f"{mhz},{mhz}"] if mhz else None
    info = {"requested_mhz": mhz, "command": cmd, "success": False}
    if cmd:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            info.update(success=(r.returncode == 0), returncode=r.returncode,
                        stdout=r.stdout[-300:], stderr=r.stderr[-300:])
        except Exception as exc:  # noqa: BLE001
            info["error"] = f"{type(exc).__name__}: {exc}"
    out.row("meta", what="lock_clocks_attempt", **info)
    return info["success"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="scaled_mm_ceiling.jsonl")
    p.add_argument("--m", type=int, nargs="+", default=list(DEFAULT_M))
    p.add_argument("--shapes", nargs="+", choices=list(SHAPES), default=list(SHAPES))
    p.add_argument("--pairs", type=int, default=31)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--skip-cold", action="store_true")
    p.add_argument("--lock-clocks", action="store_true",
                   help="try nvidia-smi -lgc at max SM clock; only records success/failure")
    args = p.parse_args()

    import torch
    from vllm import _custom_ops as ops
    from importlib.metadata import version

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 GPU required")
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Out(args.out)
    props = torch.cuda.get_device_properties(0)
    bus = getattr(props, "pci_bus_id", None)
    nvml = Nvml(bus if isinstance(bus, str) else None, torch.cuda.current_device())
    l2 = l2_bytes(torch)
    out.row("meta", what="environment", script_sha256=sha(__file__), torch=torch.__version__,
            cuda=torch.version.cuda, vllm=version("vllm"), device=props.name,
            sm_count=props.multi_processor_count, l2_bytes=l2, nvml_error=nvml.error,
            nvml_max_sm_mhz=nvml.max_sm_mhz(), start_sm_mhz=nvml.sm_mhz(), seed=args.seed,
            pairs=args.pairs, python=sys.version.split()[0],
            api={"has_aten__scaled_mm_v2": hasattr(torch.ops.aten, "_scaled_mm_v2"),
                 "has_F_scaled_mm": hasattr(torch.nn.functional, "scaled_mm"),
                 "has_F_ScalingType": hasattr(torch.nn.functional, "ScalingType"),
                 "has_float8_e8m0fnu": hasattr(torch, "float8_e8m0fnu"),
                 "cutlass_scaled_mm_op": hasattr(torch.ops._C, "cutlass_scaled_mm")},
            cutlass_reference_tflops_from_doc14=CUTLASS_REF_TFLOPS)
    locked = lock_clocks(out, nvml) if args.lock_clocks else None

    flush = torch.empty(max(2 * l2 + 1, 256 << 20), dtype=torch.uint8, device="cuda")
    rng = random.Random(args.seed)
    best_at_16160 = {}
    try:
        for shape_name in args.shapes:
            n, k = SHAPES[shape_name]
            for m in args.m:
                tag = f"{shape_name}-M{m}"
                try:
                    arms, attempts, keep = build_arms(torch, ops, m, n, k, args.seed + m)
                except Exception as exc:  # noqa: BLE001
                    out.row("attempt", arm="*", shape=shape_name, m=m, accepted=False,
                            exception=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-800:])
                    continue
                for row in attempts:
                    out.row("attempt", shape=shape_name, **row)
                if "cutlass_blockwise" not in arms:
                    out.row("meta", what="skip_no_cutlass_reference", tag=tag)
                    continue
                graphs = {}
                for name, call in arms.items():
                    try:
                        graphs[name] = capture(torch, call)
                    except Exception as exc:  # noqa: BLE001
                        out.row("attempt", arm=name, shape=shape_name, m=m, accepted=False,
                                stage="graph_capture", exception=f"{type(exc).__name__}: {exc}"[:2500])
                if len(graphs) < 2:
                    out.row("meta", what="only_reference_arm", tag=tag)
                with ClockPoller(nvml) as poller:
                    t_w = time.perf_counter()  # clock ramp-up: ~1.5 s of back-to-back replays, not timed
                    while time.perf_counter() - t_w < 1.5:
                        for g in graphs.values():
                            g.replay()
                        torch.cuda.synchronize()
                    rows = timing(torch, graphs, flush, rng, args.pairs, poller, "warm")
                    if not args.skip_cold:
                        rows += timing(torch, graphs, flush, rng, args.pairs, poller, "cold")
                for r in rows:
                    out.row("sample", shape=shape_name, m=m, **r)
                summ = summarize(out, rows, shape_name, m, n, k)
                if m == 16160:
                    for (mode, arm), row in summ.items():
                        best_at_16160.setdefault((shape_name, mode), {})[arm] = row["median_tflops"]
                del graphs, arms, keep, rows
                torch.cuda.empty_cache()
    finally:
        if locked:
            subprocess.run(["nvidia-smi", "-rgc"], capture_output=True)

    # Decision rule (plan section A): best < CUTLASS * 1.10 at M=16160 => GEMM line closes.
    for (shape_name, mode), tf in sorted(best_at_16160.items()):
        cut = tf.get("cutlass_blockwise")
        others = {a: v for a, v in tf.items() if a != "cutlass_blockwise"}
        if cut is None or not others:
            continue
        arm = max(others, key=others.get)
        out.row("decision", shape=shape_name, mode=mode, m=16160, cutlass_tflops_same_run=cut,
                best_other_arm=arm, best_other_tflops=others[arm],
                best_over_cutlass_same_run=others[arm] / cut,
                best_over_doc14_617=others[arm] / CUTLASS_REF_TFLOPS,
                rule="gemm line closes if best_over_cutlass_same_run < 1.10 (and < 1.10 vs 617)",
                gemm_line_closes=bool(others[arm] < 1.10 * cut and others[arm] < 1.10 * CUTLASS_REF_TFLOPS),
                note="MMA-rate probe (mma_rate_probe) results must be combined by the operator")


if __name__ == "__main__":
    main()
