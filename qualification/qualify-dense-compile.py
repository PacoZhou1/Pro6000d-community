import json,pathlib,torch
from vllm.model_executor.layers.homelab_dense import prepare_homelab_dense
from vllm.model_executor.layers.utils import default_unquantized_gemm
torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(731)
class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.prefix='model.language_model.layers.0.self_attn.o_proj'
        self.weight=torch.nn.Parameter(torch.randn((4096,2048),device='cuda',dtype=torch.bfloat16),requires_grad=False)
        prepare_homelab_dense(self);assert self._homelab_dense_cute
    def forward(self,x):return default_unquantized_gemm(self,x,self.weight)
layer=Layer();compiled=torch.compile(layer,fullgraph=True)
rows=[]
for m in [1,2,3,4,8,32]:
    x=torch.randn((m,2048),device='cuda',dtype=torch.bfloat16)
    expected=torch.nn.functional.linear(x,layer.weight)
    for _ in range(3):out=compiled(x)
    torch.testing.assert_close(out,expected,atol=.125,rtol=.015)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=compiled(x)
    x.neg_();graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(out,-expected,atol=.125,rtol=.015)
    rows.append({'m':m,'fullgraph_compile':True,'cuda_graph_mutation':True})
    print('PASS_COMPILE_GRAPH',m,flush=True)
pathlib.Path('/work/dense-bf16-integration/compile-qualification.json').write_text(json.dumps({'all_pass':len(rows)==6,'cases':rows},indent=2))
