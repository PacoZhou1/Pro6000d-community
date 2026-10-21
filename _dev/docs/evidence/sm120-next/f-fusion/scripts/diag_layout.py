import torch, triton, re, f_kernels as fk
from vllm.third_party.flash_linear_attention.ops import layernorm_guard as lg
dev='cuda'; NH=48; HD=128; M=16160
x = torch.randn(M*NH,128,device=dev).bfloat16(); z = torch.randn(M*NH,128,device=dev).bfloat16()
w = torch.ones(128,device=dev); y = torch.empty_like(x); rstd = torch.empty(M*NH,device=dev)
rpb = lg.calc_rows_per_block(M*NH, x.device); print('rows per block', rpb)
k = lg.layer_norm_fwd_kernel[(triton.cdiv(M*NH, rpb), 1)](x, y, w, None, z, None, rstd, 128, 128, 128, M*NH, 128, 1e-6,
    BLOCK_N=128, ROWS_PER_BLOCK=rpb, HAS_BIAS=False, HAS_Z=True, NORM_BEFORE_GATE=True, IS_RMS_NORM=True, ACTIVATION='swish', num_warps=1)
def blocked(tt): return sorted(set(re.findall(r'#blocked\d* = #ttg.blocked<\{[^}]*\}>', tt)))
tt = k.asm['ttgir']
open('/work/results/official_norm.ttgir','w').write(tt)
print('OFFICIAL layouts:'); [print(' ', b) for b in blocked(tt)]
print('reduce ops:', re.findall(r'"tt.reduce"\(.*', tt)[:2])
print('PTX shfl count', k.asm['ptx'].count('shfl.sync.bfly'), 'fma', k.asm['ptx'].count('fma.rn.f32'))
# mine
q = torch.empty(M, NH*HD, dtype=torch.float8_e4m3fn, device=dev); s = torch.empty(NH, M, device=dev).permute(1,0)
zz = torch.randn(M, 16384, device=dev).bfloat16()[:, 10240:].reshape(M, NH, HD)
xx = x.reshape(M, NH, HD)
km = fk._gdn_norm_quant_kernel[(triton.cdiv(M*NH, rpb),)](xx, zz, w, q, s, q, s, 128, 128, 16384, 128, M*NH, M, 1e-6, NH=NH, N=128, BLOCK_N=128, ROWS_PER_BLOCK=rpb, SCALE_MODE=0, STORE_Y=False, num_warps=1)
tm = km.asm['ttgir']; open('/work/results/mine_norm.ttgir','w').write(tm)
print('MINE layouts:'); [print(' ', b) for b in blocked(tm)]
print('PTX shfl count', km.asm['ptx'].count('shfl.sync.bfly'), 'fma', km.asm['ptx'].count('fma.rn.f32'))

import re as _re
for nm,t in (("official",tt),("mine",tm)):
    lines=t.split("\n")
    for i,l in enumerate(lines):
        if "tt.reduce" in l: print(nm,"REDUCE:",l.strip()[:220])
    print(nm,"loads:",[l.strip()[:160] for l in lines if "tt.load" in l][:6])
