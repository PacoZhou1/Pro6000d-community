"""Probe the official per_token_group_fp8_quant op semantics (run inside the image on the GPU host)."""
import torch, json
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
dev = 'cuda'
torch.manual_seed(0)
M = 4096
x = (torch.randn(M, 6144, device=dev) * torch.rand(M, 48, 1, device=dev).repeat_interleave(128, 2).reshape(M, 6144) * 3).bfloat16()
q, s = per_token_group_quant_fp8(x, 128, column_major_scales=True, dtype=torch.float8_e4m3fn, use_ue8m0=False)
print('scale stride', s.stride(), s.shape, q.dtype)
xf = x.float().reshape(M, 48, 128)
amax = xf.abs().amax(-1)
res = {}
for name, f in {
 'eps_max_div448': lambda a: a.clamp_min(1e-10) / 448.0,
 'div448_then_max_minscale': lambda a: (a / 448.0).clamp_min(1.0 / (448.0 * 512.0)),
 'plain_div448': lambda a: a / 448.0,
}.items():
    ref_s = f(amax)
    res[name + '_scale_bits_equal'] = bool((ref_s.view(torch.int32) == s.view(torch.int32)).all())
ref_s = amax.clamp_min(1e-10) / 448.0
for qn, qf in {
 'div': lambda: (xf / ref_s[..., None]).clamp(-448, 448),
 'mulrecip': lambda: (xf * (1.0 / ref_s[..., None])).clamp(-448, 448),
}.items():
    qq = qf().reshape(M, 6144).to(torch.float8_e4m3fn)
    res['q_' + qn + '_mismatch_bytes'] = int((qq.view(torch.uint8) != q.view(torch.uint8)).sum())
print(json.dumps(res, indent=1))
r = s / (amax / 448.0)
print('ratio min/max', r.min().item(), r.max().item(), 'frac equal', (r == 1).float().mean().item())
inv = amax * (1.0 / 448.0)
print('amax*(1/448) equal frac', (inv.view(torch.int32) == s.view(torch.int32)).float().mean().item())
print('amax shape', amax.shape, s.shape)
print(s[0, :4], (amax/448)[0, :4])
import numpy as np
c0 = np.float32(1.0 / 448.0)
for k in range(-3, 4):
    c = c0
    for _ in range(abs(k)):
        c = np.nextafter(c, np.float32(-1 if k < 0 else 1), dtype=np.float32)
    cand = amax * torch.tensor(float(c), dtype=torch.float32, device=dev)
    print('mulconst', k, repr(float(c)), 'equal frac', (cand.view(torch.int32) == s.view(torch.int32)).float().mean().item())
# also: scale from amax rounded
cand = torch.nextafter(amax / 448.0, torch.zeros((), device=dev))
print('nextafter-down frac', (cand.view(torch.int32) == s.view(torch.int32)).float().mean().item())
d = amax.double() / 448.0
rn = d.float()
rz = torch.where(rn.double() > d, torch.nextafter(rn, torch.zeros((), device=dev)), rn)
print('RZ division equal frac', (rz.view(torch.int32) == s.view(torch.int32)).float().mean().item())
sb = s.contiguous()  # (M,48)
for qn, qf in {
 'div_by_official_s': lambda: (xf / sb[..., None]).clamp(-448, 448),
 'mulrecip_of_official_s': lambda: (xf * (1.0 / sb[..., None])).clamp(-448, 448),
}.items():
    qq = qf().reshape(M, 6144).to(torch.float8_e4m3fn)
    print(qn, 'mismatch bytes', int((qq.view(torch.uint8) != q.view(torch.uint8)).sum()))
