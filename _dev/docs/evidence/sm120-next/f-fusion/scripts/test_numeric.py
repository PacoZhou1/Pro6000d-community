"""Numeric gate (run inside the fixed image on the GPU host).
F1: byte-exact FP8 + bit-exact scales vs [rmsnorm_fn (z copy) -> per_token_group_quant_fp8].
F2: bit-exact q/k/v/g/beta + conv_state vs [causal_conv1d_fn -> fused_post_conv_prep].
Special values: zero / tiny / large finite / NaN / Inf.  Also a changed-input CUDA Graph check.
usage: test_numeric.py [OUT.json]"""
import json, sys
import torch
import f_kernels as fk
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep

dev = "cuda"
NH, HD = 48, 128
results = []


def rec(name, ok, **kw):
    results.append(dict(name=name, ok=bool(ok), **kw))
    print(("PASS " if ok else "FAIL ") + name, kw, flush=True)


def eq_bits(a, b):
    a = a.contiguous().view(torch.uint8) if a.dtype == torch.float8_e4m3fn else a.contiguous()
    b = b.contiguous().view(torch.uint8) if b.dtype == torch.float8_e4m3fn else b.contiguous()
    if a.dtype in (torch.float32,):
        a, b = a.view(torch.int32), b.view(torch.int32)
    elif a.dtype == torch.bfloat16:
        a, b = a.view(torch.int16), b.view(torch.int16)
    return int((a != b).sum()), a.numel()


# ------------------------------------------------------------------ F1
def make_f1_inputs(M, seed, mode="random"):
    g = torch.Generator(device=dev).manual_seed(seed)
    xs = torch.randn(M, NH, HD, device=dev, generator=g) * (torch.rand(M, NH, 1, device=dev, generator=g) * 4 + 0.05)
    full = torch.randn(M, 16384, device=dev, generator=g) * 2.0           # qkvz-like buffer
    x = xs.bfloat16().contiguous()
    z = full.bfloat16()[:, 10240:].reshape(M, -1, HD)                      # strided view like the model (6144 of 16384 cols)
    assert M == 1 or z.stride() == (16384, HD, 1), z.stride()
    w = (torch.randn(HD, device=dev, generator=g) * 0.3 + 1).float()      # norm weight (fp32 in model)
    if mode == "zeros":
        x.zero_()
    elif mode == "tiny":
        x = (x.float() * 1e-30).bfloat16()
    elif mode == "large":
        x = (x.float() * 1e18).bfloat16()
    elif mode == "nan":
        x[::7, 3, 5] = float("nan"); x[::11, 7, 100] = float("inf"); x[::13, 9, 0] = float("-inf")
    elif mode == "znan":
        zz = z.clone(); zz[::5, 2, 3] = float("nan"); zz[::9, 4, 8] = float("inf")
        full2 = torch.zeros(M, 16384, dtype=torch.bfloat16, device=dev); full2[:, 10240:] = zz.reshape(M, -1)
        z = full2[:, 10240:].reshape(M, NH, HD)
    elif mode == "mixed_zero_rows":
        x[::3].zero_()
    return x, z, w


def run_f1(M, mode, seed=1, eps=1e-6):
    x, z, w = make_f1_inputs(M, seed, mode)
    (qr, sr), _ = fk.gdn_norm_quant_reference(x, z, w, eps)
    q = torch.empty(M, NH * HD, dtype=torch.float8_e4m3fn, device=dev)
    s = torch.empty(NH, M, dtype=torch.float32, device=dev).permute(1, 0)
    fk.gdn_norm_quant_op(q, s, x, z, w, eps)
    torch.cuda.synchronize()
    bq, nq = eq_bits(q, qr)
    bs, ns = eq_bits(s, sr)
    rec(f"F1 M={M} mode={mode}", bq == 0 and bs == 0 and s.stride() == sr.stride(), fp8_byte_mismatch=bq, fp8_bytes=nq,
        scale_bit_mismatch=bs, scales=ns)


for M in (16160, 2741, 1, 7, 255, 256, 1000):
    run_f1(M, "random")
for mode in ("zeros", "tiny", "large", "nan", "znan", "mixed_zero_rows"):
    run_f1(2741, mode)
    run_f1(33, mode)
# different seeds at the big size
for seed in (2, 3):
    run_f1(16160, "random", seed=seed)


# ------------------------------------------------------------------ F2
H, HV, K, V = 16, 48, 128, 128
DIM = 2 * H * K + HV * V


def run_f2(L, has_init, seed=1, mode="random", slot=5):
    g = torch.Generator(device=dev).manual_seed(seed)
    full = (torch.randn(L, 16384, device=dev, generator=g) * 1.5).bfloat16()
    if mode == "special":
        full[::17, 100] = float("nan"); full[::19, 3000] = float("inf"); full[::23, 6000] = 0
        full[::29, 200] = 1e30
    x = full[:, :DIM]
    cw = (torch.randn(DIM, 4, device=dev, generator=g) * 0.5).bfloat16()
    cs0 = (torch.randn(32, DIM, 3, device=dev, generator=g)).bfloat16()
    a = (torch.randn(L, HV, device=dev, generator=g)).bfloat16().contiguous()
    b = (torch.randn(L, HV, device=dev, generator=g)).bfloat16().contiguous()
    A_log = torch.randn(HV, device=dev, generator=g).float()
    dt_bias = torch.randn(HV, device=dev, generator=g).float()
    idx = torch.tensor([slot], dtype=torch.int32, device=dev)
    hi = torch.tensor([has_init], dtype=torch.bool, device=dev)
    qsl = torch.tensor([0, L], dtype=torch.int32, device=dev)

    cs_ref = cs0.clone()
    conv_out = causal_conv1d_fn(x.transpose(0, 1), cw, None, activation="silu", conv_states=cs_ref,
                                has_initial_state=hi, cache_indices=idx, query_start_loc=qsl).transpose(0, 1)
    qr, kr, vr, gr, br = fused_post_conv_prep(conv_output=conv_out, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
                                              num_k_heads=H, head_k_dim=K, head_v_dim=V, apply_l2norm=True,
                                              output_g_exp=False)
    cs_new = cs0.clone()
    q, k, v, gg, bb = fk.conv_post_fused(x, cw, cs_new, idx, hi, a, b, A_log, dt_bias, H, K, V)
    torch.cuda.synchronize()
    out = {}
    ok = True
    for n, t, r in (("q", q, qr), ("k", k, kr), ("v", v, vr), ("g", gg, gr), ("beta", bb, br), ("conv_state", cs_new, cs_ref)):
        mism, tot = eq_bits(t, r)
        out[n + "_mismatch"] = mism
        ok &= mism == 0
    rec(f"F2 L={L} has_init={has_init} mode={mode}", ok, **out)


for L in (16160, 2741):
    for hi in (False, True):
        run_f2(L, hi)
for L in (3, 4, 5, 15, 16, 17, 100, 257):
    run_f2(L, True)
    run_f2(L, False)
run_f2(2741, True, mode="special")
run_f2(16160, False, mode="special", seed=4)


# ------------------------------------------------------------------ CUDA Graph, changed inputs
def graph_check():
    M = 2048
    x, z, w = make_f1_inputs(M, 11)
    q = torch.empty(M, NH * HD, dtype=torch.float8_e4m3fn, device=dev)
    s = torch.empty(NH, M, dtype=torch.float32, device=dev).permute(1, 0)
    fk.gdn_norm_quant(q, s, x, z, w, 1e-6)  # warm/compile
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fk.gdn_norm_quant(q, s, x, z, w, 1e-6)
    ok = True
    for seed in (21, 22, 23):
        x2, z2, _ = make_f1_inputs(M, seed)
        x.copy_(x2)
        z.copy_(z2)
        gr.replay()
        torch.cuda.synchronize()
        (qr, sr), _ = fk.gdn_norm_quant_reference(x, z, w, 1e-6)
        ok &= eq_bits(q, qr)[0] == 0 and eq_bits(s, sr)[0] == 0
    rec("F1 CUDA Graph replay with changed inputs", ok)

    L = 2741
    g = torch.Generator(device=dev).manual_seed(5)
    full = (torch.randn(L, 16384, device=dev, generator=g) * 1.5).bfloat16()
    x = full[:, :DIM]
    cw = (torch.randn(DIM, 4, device=dev, generator=g) * 0.5).bfloat16()
    cs = (torch.randn(8, DIM, 3, device=dev, generator=g)).bfloat16()
    a = torch.randn(L, HV, device=dev, generator=g).bfloat16().contiguous()
    b = torch.randn(L, HV, device=dev, generator=g).bfloat16().contiguous()
    A_log = torch.randn(HV, device=dev, generator=g).float(); dt_bias = torch.randn(HV, device=dev, generator=g).float()
    idx = torch.tensor([2], dtype=torch.int32, device=dev); hi = torch.tensor([True], device=dev)
    fk.conv_post_fused(x, cw, cs.clone(), idx, hi, a, b, A_log, dt_bias, H, K, V)
    torch.cuda.synchronize()
    cs_g = cs.clone()
    gr2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr2):
        outs = fk.conv_post_fused(x, cw, cs_g, idx, hi, a, b, A_log, dt_bias, H, K, V)
    ok = True
    for seed in (31, 32):
        g2 = torch.Generator(device=dev).manual_seed(seed)
        full.copy_((torch.randn(L, 16384, device=dev, generator=g2) * 1.5).bfloat16())
        cs_g.copy_(cs)
        gr2.replay(); torch.cuda.synchronize()
        cs_r = cs.clone()
        co = causal_conv1d_fn(x.transpose(0, 1), cw, None, activation="silu", conv_states=cs_r, has_initial_state=hi,
                              cache_indices=idx, query_start_loc=torch.tensor([0, L], dtype=torch.int32, device=dev)).transpose(0, 1)
        ref = fused_post_conv_prep(conv_output=co, a=a, b=b, A_log=A_log, dt_bias=dt_bias, num_k_heads=H, head_k_dim=K,
                                   head_v_dim=V, apply_l2norm=True, output_g_exp=False)
        for t, r in zip(outs, ref):
            ok &= eq_bits(t, r)[0] == 0
        ok &= eq_bits(cs_g, cs_r)[0] == 0
    rec("F2 CUDA Graph replay with changed inputs", ok)


graph_check()
out = sys.argv[1] if len(sys.argv) > 1 else "/work/results/numeric.json"
json.dump(dict(all_pass=all(r["ok"] for r in results), results=results, torch=torch.__version__), open(out, "w"), indent=1)
print("ALL PASS" if all(r["ok"] for r in results) else "SOME FAIL")
