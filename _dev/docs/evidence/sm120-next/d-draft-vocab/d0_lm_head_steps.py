#!/usr/bin/env python3
"""D0 offline analyzer: lm_head-shaped GEMM kernels per MTP4 decode step.

Input: an Nsight Systems SQLite export (CUPTI kernels, RUNTIME/DRIVER API rows,
NVTX_EVENTS, StringIds). Works for eager and normal CUDA Graph captures.

Join (same exact-correlation idea as evidence/gemm-targeted/analyze_attribution.py):
  kernel.correlationId == CUDA API row correlationId (cudaGraphLaunch or
  cuLaunchKernel); that API row's (globalTid, start) lies inside the shortest
  same-thread NVTX stage range ``execute_context_*_generation_<n>(<m>)`` with
  n>=1 (a real decode step). Kernels inside a graph replay share the
  cudaGraphLaunch correlationId, so graph replay needs no per-kernel NVTX
  (NVTX inside a captured graph is NOT replayed; GEMM_ATTRIB labels exist only
  in eager runs).

lm_head-shaped kernel (no shape in the kernel name, so shape is inferred):
  BF16 GEMM-named kernel with duration >= --min-us (default 1000 us) whose
  launch geometry (grid/block) matches an N=248320 projection. A 248320x5120
  BF16 weight is 2.543 GB; at ~1.58 ms that is ~1.6 TB/s, which no other BF16
  GEMM in this model approaches. Signatures are reported verbatim so a human
  can confirm; override with --grid-y to pin exact gridY values.
  Rows are labelled M1-like / M5-like only via an optional --m1-name /
  --m5-name substring; otherwise by signature group.

Fallback when no stage NVTX covers a launch: steps are delimited by sorted
kernel time using the M5-like (verification) kernels: step i = (end of
verification i-1, end of verification i]. Reported as step_source=time_partition.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sqlite3, sys
from bisect import bisect_right
from collections import Counter, defaultdict

STAGE = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
GEMM = re.compile(r"gemm|cublas|matmul|gemv|\bmm\b", re.I)
BF16 = re.compile(r"bf16|bfloat16", re.I)
FP8 = re.compile(r"fp8|float8", re.I)
PID_MASK = ~0xFFFFFF
LM_HEAD_BYTES = 248320 * 5120 * 2


def cols(db, t):
    return {r[1] for r in db.execute(f'PRAGMA table_info("{t}")')}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def analyze(db, min_us=1000.0, grid_y=None, m1_name=None, m5_name=None, pids=None):
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    kc = cols(db, "CUPTI_ACTIVITY_KIND_KERNEL")
    want = [c for c in ("start", "end", "globalPid", "correlationId", "demangledName",
                        "shortName", "gridX", "gridY", "gridZ", "blockX", "blockY",
                        "blockZ", "graphId") if c in kc]
    allowed = None
    if pids:
        allowed = {g for g, p in db.execute("SELECT globalPid,pid FROM PROCESSES") if p in pids}
    kernels = []
    for vals in db.execute("SELECT " + ",".join(want) + " FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        k = dict(zip(want, vals))
        if allowed is not None and k.get("globalPid") not in allowed:
            continue
        k["name"] = strings.get(k.get("demangledName")) or strings.get(k.get("shortName")) or "?"
        kernels.append(k)
    total_sum = sum(k["end"] - k["start"] for k in kernels)
    graph_kernels = sum(1 for k in kernels if k.get("graphId") is not None)
    mode = "cuda_graph_replay" if graph_kernels > 0.5 * len(kernels) else (
        "mixed" if graph_kernels else "eager")

    cand = []
    for k in kernels:
        n = k["name"]
        dur = k["end"] - k["start"]
        if not (GEMM.search(n) and BF16.search(n) and not FP8.search(n)):
            continue
        if dur < min_us * 1000:
            continue
        if grid_y and k.get("gridY") not in grid_y:
            continue
        cand.append(k)

    def sig(k):
        return (k["name"], k.get("gridX"), k.get("gridY"), k.get("gridZ"),
                k.get("blockX"), k.get("blockY"), k.get("blockZ"))

    def label(k):
        if m1_name and m1_name in k["name"]:
            return "M1"
        if m5_name and m5_name in k["name"]:
            return "M5"
        return None

    sigs = Counter(sig(k) for k in cand)
    sig_ids = {s: f"S{i}" for i, (s, _) in enumerate(sorted(sigs.items(), key=lambda kv: -kv[1]))}

    # API rows for candidate correlations
    corr = {k["correlationId"] for k in cand}  # correlationId only: globalTid/globalPid bit layouts differ in this nsys version
    api = {}
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if t not in {r[0] for r in db.execute("SELECT name FROM sqlite_master")}:
            continue
        for start, end, tid, cid, nid in db.execute(
                f"SELECT start,end,globalTid,correlationId,nameId FROM {t}"):
            key = cid
            if key in corr:
                api[key] = (start, tid, strings.get(nid, ""))
    # stage ranges per tid
    stages = defaultdict(list)
    if "NVTX_EVENTS" in {r[0] for r in db.execute("SELECT name FROM sqlite_master")}:
        for s, e, tid, text, textid in db.execute(
                "SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
            text = text or strings.get(textid, "") or ""
            m = STAGE.match(text)
            if m and int(m.group(3)) >= 1:
                stages[tid].append((s, e, text))
    for v in stages.values():
        v.sort()

    # In the real MTP4 capture the stage NVTX marks only the target forward; the
    # lm_head projection and the draft graph launches happen AFTER the range ends
    # and before the next stage starts. So a step window is [stage_i.start,
    # stage_{i+1}.start) on the same thread (last window: +inf).
    starts = {tid: [x[0] for x in lst] for tid, lst in stages.items()}

    def stage_of(tid, when):
        lst = stages.get(tid)
        if not lst:
            return None
        i = bisect_right(starts[tid], when) - 1
        return lst[i] if i >= 0 else None

    per_step = defaultdict(list)
    unassigned = []
    for k in cand:
        a = api.get(k["correlationId"])
        st = stage_of(a[1], a[0]) if a else None
        rec = (k["start"], k["end"] - k["start"], sig_ids[sig(k)], k.get("graphId"),
               a[2] if a else None)
        if st:
            per_step[(st[0], st[1], st[2])].append(rec)
        else:
            unassigned.append(rec)
    step_source = "nvtx_stage_window_cpu_launch_api"
    steps = [sorted(v) for _, v in sorted(per_step.items())]
    if not steps and cand:
        step_source = "time_partition_by_most_numerous_sig_ordered"
        # fallback: delimit by the least-frequent signature (verification, once/step)
        ver = min(sigs.items(), key=lambda kv: kv[1])[0]
        ver_ends = sorted(k["end"] for k in cand if sig(k) == ver)
        buckets = defaultdict(list)
        for k in cand:
            idx = sum(1 for e in ver_ends if e < k["start"])
            buckets[idx].append((k["start"], k["end"] - k["start"], sig_ids[sig(k)], k.get("graphId"), None))
        steps = [sorted(v) for _, v in sorted(buckets.items())]
        unassigned = []

    shape_hist = Counter()
    dur_by_sig = defaultdict(list)
    for s in steps:
        shape_hist[tuple(sorted(Counter(r[2] for r in s).items()))] += 1
        for r in s:
            dur_by_sig[r[2]].append(r[1])
    sig_report = []
    for s, sid in sig_ids.items():
        d = dur_by_sig.get(sid, []) or [k["end"] - k["start"] for k in cand if sig(k) == s]
        sig_report.append({
            "id": sid, "kernel": s[0], "grid": list(s[1:4]), "block": list(s[4:7]),
            "count_total": sigs[s],
            "mean_us": sum(d) / len(d) / 1e3 if d else None,
            "min_us": min(d) / 1e3 if d else None, "max_us": max(d) / 1e3 if d else None,
            "implied_GBps_if_reads_lm_head_once": LM_HEAD_BYTES / (sum(d) / len(d)) if d else None,
            "name_hint": ("M1-like" if m1_name and m1_name in s[0] else
                          "M5-like" if m5_name and m5_name in s[0] else None),
        })
    cand_sum = sum(k["end"] - k["start"] for k in cand)
    graphs = Counter(r[3] for s in steps for r in s)
    return {
        "mode": mode, "kernel_count": len(kernels), "gpu_kernel_sum_ns": total_sum,
        "graph_kernel_fraction": graph_kernels / len(kernels) if kernels else None,
        "min_us": min_us, "grid_y_filter": sorted(grid_y) if grid_y else None,
        "step_source": step_source, "decode_steps_with_candidates": len(steps),
        "unassigned_candidates": len(unassigned),
        "signatures": sig_report,
        "per_step_signature_count_histogram": [
            {"counts": dict(k), "steps": v} for k, v in shape_hist.most_common()],
        "candidate_kernel_count": len(cand), "candidate_sum_ns": cand_sum,
        "candidate_share_of_gpu_kernel_sum": cand_sum / total_sum if total_sum else None,
        "mean_candidate_ns_per_step": cand_sum / len(steps) if steps else None,
        "graph_ids_of_candidates": {str(k): v for k, v in graphs.items()},
        "note": ("Graph-replay capture: durations are real replay kernel durations under nsys; "
                 "not a latency claim. Eager vs graph: eager runs launch every kernel with "
                 "cuLaunchKernel and carry GEMM_ATTRIB NVTX; in graph mode the kernels of one "
                 "cudaGraphLaunch share a correlationId and graphId, no per-kernel NVTX exists."),
    }


def self_test():
    db = sqlite3.connect(":memory:")
    for ddl in (
        "CREATE TABLE StringIds(id INTEGER,value TEXT)",
        "CREATE TABLE PROCESSES(globalPid INTEGER,pid INTEGER,name TEXT)",
        "CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER)",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER)",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER,shortName INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER,graphId INTEGER)"):
        db.execute(ddl)
    pid = 0x1000000; tid = pid + 5
    db.executemany("INSERT INTO StringIds VALUES (?,?)", [(1, "gemm_bf16_16x16"), (2, "gemm_bf16_32x32"), (3, "cudaGraphLaunch")])
    db.execute("INSERT INTO PROCESSES VALUES (?,?,?)", (pid, 1, "x"))
    rt, kr, nv = [], [], []
    t = 0
    for step in range(3):
        nv.append((t, t + 1000, tid, "execute_context_0(0)_generation_1(5)", None))
        for j in range(4):
            c = step * 10 + j
            rt.append((t + 10 + j, t + 11 + j, tid, c, 3))
            kr.append((t + 100 + j * 100, t + 100 + j * 100 + 50, pid, c, 1, None, 8, 1940, 1, 32, 1, 1, 7))
        c = step * 10 + 9
        rt.append((t + 20, t + 21, tid, c, 3))
        kr.append((t + 600, t + 650, pid, c, 2, None, 8, 970, 1, 128, 1, 1, None))
        t += 2000
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)", nv)
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)", rt)
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", kr)
    r = analyze(db, min_us=0.0)
    assert r["decode_steps_with_candidates"] == 3, r
    assert r["per_step_signature_count_histogram"][0]["steps"] == 3
    assert sorted(r["per_step_signature_count_histogram"][0]["counts"].values()) == [1, 4]
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite", nargs="?")
    ap.add_argument("--min-us", type=float, default=1000.0)
    ap.add_argument("--grid-y", type=int, nargs="*")
    ap.add_argument("--m1-name", help="kernel-name substring that identifies the M=1 draft kernel")
    ap.add_argument("--m5-name", help="kernel-name substring for the M=5 verification kernel")
    ap.add_argument("--pid", type=int, nargs="*", help="OS pids to include")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.sqlite:
        ap.error("sqlite path required")
    db = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
    res = analyze(db, a.min_us, set(a.grid_y) if a.grid_y else None, a.m1_name, a.m5_name,
                  set(a.pid) if a.pid else None)
    res["input"] = {"path": a.sqlite, "sha256": sha256(a.sqlite)}
    text = json.dumps(res, indent=2, ensure_ascii=False)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
