"""BF16 lm_head real-shape synthetic counter probe; no model quality claim."""
import argparse,json,torch
p=argparse.ArgumentParser();p.add_argument('--m',type=int,required=True);a=p.parse_args()
assert a.m in [1,5,16]
torch.manual_seed(20261014)
x=torch.randn((a.m,5120),device='cuda',dtype=torch.bfloat16)
w=torch.randn((248320,5120),device='cuda',dtype=torch.bfloat16)
for _ in range(5):out=torch.nn.functional.linear(x,w)
torch.cuda.synchronize()
assert bool(torch.isfinite(out).all())
print(json.dumps({'M':a.m,'N':248320,'K':5120,'x_stride':list(x.stride()),'w_stride':list(w.stride()),'dtype':'bfloat16','weight_bytes':w.numel()*w.element_size(),'scope':'synthetic operands, actual eager-attributed lm_head shapes/layout; not real checkpoint or full-model performance'}),flush=True)
torch.cuda.cudart().cudaProfilerStart()
out=torch.nn.functional.linear(x,w)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print(json.dumps({'finite':bool(torch.isfinite(out).all())}),flush=True)
