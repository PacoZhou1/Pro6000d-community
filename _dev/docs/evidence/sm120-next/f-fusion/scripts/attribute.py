"""Step 1 attribution: per-layer kernel sequences from an nsys SQLite capture (offline, no GPU).
usage: attribute.py CAPTURE.sqlite [--out FILE]
Segments the kernel stream into decoder layers at the end of each down_proj GEMM (the GEMM that
follows silu_and_mul_per_block_quant_kernel) and prints, for each (layer type, chunk M), the mean
duration per kernel position."""
import sqlite3, sys, re, collections, json, statistics

def short(n):
    n = re.sub(r'\(.*', '', n)
    m = re.search(r'(per_token_group_quant_8bit_register_kernel|cutlass_3x_gemm_fp8_blockwise|silu_and_mul_per_block_quant_kernel'
                  r'|flashinfer::BatchPrefill\w+|cutlass_80_\w+gemm\w*|layer_norm_fwd_kernel|_causal_conv1d_fwd_kernel|_fused_post_conv_kernel'
                  r'|triton_\w+|_fused_qk_rmsnorm_rope_gate_kernel|reshape_and_cache_flash_kernel|CPDeltaRule\w+?Sm120|direct_copy_kernel_cuda|elementwise_kernel|index_elementwise_kernel|exp_kernel_cuda|bitwise_not|_\w+_kernel)', n)
    return m.group(1) if m else n[:60]

def main():
    db = sys.argv[1]
    c = sqlite3.connect(db)
    rows = c.execute("select k.start,k.end,s.value,k.gridX,k.gridY,k.blockX,k.tensorSizeMinusOneElements from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start").fetchall()
    mem = c.execute("select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
    # layer segmentation
    layers = []; cur = []; seen_silu = False; down_done = False
    for r in rows:
        nm = r[2]
        cur.append(r)
        if 'silu_and_mul_per_block_quant_kernel' in nm:
            seen_silu = True
        elif seen_silu and 'cutlass_3x_gemm_fp8_blockwise' in nm:
            layers.append(cur); cur = []; seen_silu = False
    groups = collections.defaultdict(list)
    for L in layers:
        raw = [r[2] for r in L]
        typ = 'ATTN' if any('BatchPrefill' in n for n in raw) else ('GDN' if any('CPDeltaRule' in n for n in raw) else 'OTHER')
        mx = max((r[3] for r in L if 'silu_and_mul' in r[2]), default=0)
        groups[(typ, mx)].append(L)
    out = []
    for (typ, M), Ls in sorted(groups.items()):
        out.append(f"=== {typ} layers with silu grid-x(M)={M}: n={len(Ls)} ===")
        # use the most common signature
        sig = collections.Counter(tuple((short(r[2]), r[3], r[4]) for r in L) for L in Ls)
        best, cnt = sig.most_common(1)[0]
        out.append(f"(signature shared by {cnt}/{len(Ls)} layers)")
        mats = [L for L in Ls if tuple((short(r[2]), r[3], r[4]) for r in L) == best]
        tot = 0
        for i, (nm, gx, gy) in enumerate(best):
            d = statistics.mean((L[i][1] - L[i][0]) for L in mats) / 1000
            tot += d
            out.append(f"{i:3d} {d:9.1f} us  grid=({gx},{gy})  {nm}")
        out.append(f"    sum {tot:.1f} us")
    txt = "\n".join(out)
    print(txt)
    # trace-wide per-candidate totals
    tot_k = sum(r[1]-r[0] for r in rows)
    def S(pred):
        s = [r for r in rows if pred(r[2])]
        return len(s), sum(r[1]-r[0] for r in s)/1e6
    cands = {
      'quant_group (all)': lambda n: 'per_token_group_quant_8bit' in n,
      'layer_norm_fwd (GDN gated norm)': lambda n: n.startswith('layer_norm_fwd_kernel'),
      'causal_conv1d_fwd': lambda n: n.startswith('_causal_conv1d_fwd_kernel'),
      'fused_post_conv': lambda n: n.startswith('_fused_post_conv_kernel'),
      'at::native::elementwise_kernel': lambda n: 'at::native::elementwise_kernel' in n,
      'silu_and_mul_quant (V2 target)': lambda n: 'silu_and_mul_per_block_quant' in n,
    }
    out2 = [f"total kernel sum {tot_k/1e6:.1f} ms"]
    for k, f in cands.items():
        n, ms = S(f); out2.append(f"{k}: n={n} {ms:.2f} ms ({100*ms*1e6/tot_k:.2f}%)")
    mc = collections.Counter(); md = collections.Counter(); mb = collections.Counter()
    for s, e, b, kind in mem:
        mc[kind] += 1; md[kind] += e - s; mb[kind] += b or 0
    for k in mc: out2.append(f"memcpy kind={k}: n={mc[k]} {md[k]/1e6:.2f} ms bytes={mb[k]/1e6:.1f} MB")
    print("\n".join(out2))
    if '--out' in sys.argv:
        open(sys.argv[sys.argv.index('--out')+1], 'w').write(txt + "\n\n" + "\n".join(out2) + "\n")
main()
