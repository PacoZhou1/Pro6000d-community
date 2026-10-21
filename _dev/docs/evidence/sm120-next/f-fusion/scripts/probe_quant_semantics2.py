import torch, triton, triton.language as tl
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

@triton.jit
def k(x_ptr, s_a, s_b, s_c, s_d, n, BLOCK: tl.constexpr):
    g = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + g * BLOCK + c).to(tl.float32)
    a = tl.max(tl.abs(x), axis=0)
    a = tl.maximum(a, 1e-10)
    tl.store(s_a + g, a / 448.0)                       # default triton fp32 div
    tl.store(s_b + g, tl.extra.cuda.libdevice.div_rn(a, 448.0))        # IEEE rn
    tl.store(s_c + g, a * (1.0 / 448.0))
    tl.store(s_d + g, tl.extra.cuda.libdevice.div_rz(a, 448.0))

dev = 'cuda'
torch.manual_seed(0)
M = 4096
x = (torch.randn(M, 6144, device=dev) * torch.rand(M, 48, 1, device=dev).repeat_interleave(128, 2).reshape(M, 6144) * 3).bfloat16()
q, s = per_token_group_quant_fp8(x, 128, column_major_scales=True, dtype=torch.float8_e4m3fn, use_ue8m0=False)
sref = s.contiguous().reshape(-1)  # (M,48) row-major view of values
outs = [torch.empty(M * 48, device=dev) for _ in range(4)]
k[(M * 48,)](x, *outs, M * 48, BLOCK=128)
for n, o in zip('default rn mulrecip rz'.split(), outs):
    print(n, (o.view(torch.int32) == sref.view(torch.int32)).float().mean().item())
