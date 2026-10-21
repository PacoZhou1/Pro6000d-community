import torch, f_kernels as fk, sys
sys.argv = ['x']
exec(open('/work/test_numeric.py').read().split('# ------------------------------------------------------------------ F1')[1].split('def run_f1')[0].replace('def make_f1_inputs','def make_f1_inputs'), globals()) if False else None
import importlib.util
dev='cuda'; NH=48; HD=128
M=16160
g = torch.Generator(device=dev).manual_seed(1)
xs = torch.randn(M, NH, HD, device=dev, generator=g) * (torch.rand(M, NH, 1, device=dev, generator=g) * 4 + 0.05)
full = torch.randn(M, 16384, device=dev, generator=g) * 2.0
x = xs.bfloat16().contiguous(); z = full.bfloat16()[:, 10240:].reshape(M, -1, HD)
w = (torch.randn(HD, device=dev, generator=g) * 0.3 + 1).float()
(qr, sr), yref = fk.gdn_norm_quant_reference(x, z, w, 1e-6)
q = torch.empty(M, NH*HD, dtype=torch.float8_e4m3fn, device=dev)
s = torch.empty(NH, M, dtype=torch.float32, device=dev).permute(1, 0)
ydbg = torch.empty(M, NH, HD, dtype=torch.bfloat16, device=dev)
fk.gdn_norm_quant(q, s, x, z, w, 1e-6, y_debug=ydbg)
print('y mismatches', int((ydbg.view(torch.int16) != yref.view(torch.int16)).sum()), 'of', ydbg.numel())
# quant-only: apply official quant to MY y
(q2, s2) = fk.__dict__['_ref_quant'](ydbg) if '_ref_quant' in fk.__dict__ else (None, None)
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
q2, s2 = per_token_group_quant_fp8(ydbg.flatten(-2), 128, column_major_scales=True, dtype=torch.float8_e4m3fn, use_ue8m0=False)
print('quant-only (my y): q mismatches', int((q.view(torch.uint8) != q2.view(torch.uint8)).sum()), 'scale mismatches', int((s.contiguous().view(torch.int32) != s2.contiguous().view(torch.int32)).sum()))

from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd
zc = z.reshape(-1, 128).contiguous()
yo, _, rstd_o = layer_norm_fwd(x.reshape(-1,128), w, None, 1e-6, z=zc, group_size=128, norm_before_gate=True, is_rms_norm=True, activation='swish')
rstd_m = torch.empty(M*NH, device=dev, dtype=torch.float32)
ydbg2 = torch.empty(M, NH, HD, dtype=torch.bfloat16, device=dev)
fk.gdn_norm_quant(q, s, x, z, w, 1e-6, y_debug=ydbg2, rstd_debug=rstd_m)
print('rstd mismatches', int((rstd_m.view(torch.int32) != rstd_o.view(torch.int32)).sum()))
print('y vs layer_norm_fwd direct', int((ydbg2.reshape(-1,128).view(torch.int16) != yo.view(torch.int16)).sum()))
bad = (ydbg2.reshape(-1,128).view(torch.int16) != yo.view(torch.int16)).any(1).nonzero().flatten()
print('bad rows', bad.numel(), 'of which rstd differs', int((rstd_m[bad].view(torch.int32) != rstd_o[bad].view(torch.int32)).sum()))
