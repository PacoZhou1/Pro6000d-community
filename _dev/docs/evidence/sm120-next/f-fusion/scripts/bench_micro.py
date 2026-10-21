"""Microbench gate: in-process, randomized interleaved official-vs-candidate, warm and cache-flushed, 31+ rounds.
usage: bench_micro.py OUT.json [rounds]"""
import json, random, statistics, sys
import torch
import f_kernels as fk
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep

dev = "cuda"; NH = 48; HD = 128; H, HV, K, V = 16, 48, 128, 128; DIM = 10240
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 31
flush_buf = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)


def timed(fn, flush):
    if flush:
        flush_buf.zero_()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); e.synchronize()
    return s.elapsed_time(e) * 1000.0  # us


def run(name, M, official, cand):
    for _ in range(3):
        official(); cand()
    torch.cuda.synchronize()
    out = {"name": name, "M": M, "rounds": ROUNDS}
    for mode in ("warm", "cold"):
        flush = mode == "cold"
        t = {"official": [], "candidate": []}
        paired = []
        rng = random.Random(1234)
        for r in range(ROUNDS):
            order = ["official", "candidate"]
            rng.shuffle(order)
            cur = {}
            for k in order:
                cur[k] = timed(official if k == "official" else cand, flush)
            for k in cur: t[k].append(cur[k])
            paired.append(100 * (cur["official"] - cur["candidate"]) / cur["official"])
        out[mode] = {"official_median_us": statistics.median(t["official"]), "candidate_median_us": statistics.median(t["candidate"]),
                     "paired_saving_pct_median": statistics.median(paired), "paired_saving_pct_min": min(paired),
                     "candidate_faster_rounds": sum(p > 0 for p in paired)}
    print(json.dumps(out), flush=True)
    return out


res = []
for M in (16160, 2741):
    g = torch.Generator(device=dev).manual_seed(1)
    x = (torch.randn(M, NH, HD, device=dev, generator=g) * 2).bfloat16()
    full = (torch.randn(M, 16384, device=dev, generator=g) * 2).bfloat16()
    z = full[:, 10240:].reshape(M, NH, HD)
    w = (torch.randn(HD, device=dev, generator=g) * 0.3 + 1).float()
    q = torch.empty(M, NH * HD, dtype=torch.float8_e4m3fn, device=dev)
    s = torch.empty(NH, M, dtype=torch.float32, device=dev).permute(1, 0)

    def off1():
        (a, b), _ = fk.gdn_norm_quant_reference(x, z, w, 1e-6)
    res.append(run("F1 gated-norm+quant (incl. z copy)", M, off1, lambda: fk.gdn_norm_quant_op(q, s, x, z, w, 1e-6)))

    xin = full[:, :DIM]
    cw = (torch.randn(DIM, 4, device=dev, generator=g) * 0.5).bfloat16()
    cs = torch.randn(8, DIM, 3, device=dev, generator=g).bfloat16()
    a_ = torch.randn(M, HV, device=dev, generator=g).bfloat16().contiguous(); b_ = torch.randn(M, HV, device=dev, generator=g).bfloat16().contiguous()
    A_log = torch.randn(HV, device=dev, generator=g).float(); dtb = torch.randn(HV, device=dev, generator=g).float()
    idx = torch.tensor([2], dtype=torch.int32, device=dev); hi = torch.tensor([True], device=dev)
    qsl = torch.tensor([0, M], dtype=torch.int32, device=dev)

    def off2():
        co = causal_conv1d_fn(xin.transpose(0, 1), cw, None, activation="silu", conv_states=cs, has_initial_state=hi,
                              cache_indices=idx, query_start_loc=qsl).transpose(0, 1)
        fused_post_conv_prep(conv_output=co, a=a_, b=b_, A_log=A_log, dt_bias=dtb, num_k_heads=H, head_k_dim=K, head_v_dim=V,
                             apply_l2norm=True, output_g_exp=False)
    res.append(run("F2 conv+post_conv", M, off2, lambda: fk.conv_post_fused(xin, cw, cs, idx, hi, a_, b_, A_log, dtb, H, K, V)))
json.dump(res, open(sys.argv[1], "w"), indent=1)
