import torch,json
x=torch.ones(65536,device='cuda')
for _ in range(5): y=x+1
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
y=x+2
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print(json.dumps({'finite':bool(torch.isfinite(y).all()),'expected':bool((y==3).all()),'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability()}))
