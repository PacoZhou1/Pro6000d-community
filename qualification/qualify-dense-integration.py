"""Bounded BF16 dense projection probe, real TP4 weights, no serving patch.

Compare existing PyTorch BLAS choices and vLLM CuTe router GEMM plus BF16
cast. Graph timings are hot-weight microbenchmarks, not model throughput.
"""
import json, pathlib, statistics, gc, hashlib, subprocess
import torch
from safetensors import safe_open
from vllm.model_executor.layers.utils import default_unquantized_gemm
from vllm.model_executor.layers.homelab_dense import prepare_homelab_dense
from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm, ll_bf16_gemm_kernel

root=pathlib.Path('/work/dense-bf16-integration'); root.mkdir(exist_ok=True)
weights=pathlib.Path('/models/GLM-5.3-Flash-NVFP4')
index=json.loads((weights/'model.safetensors.index.json').read_text())['weight_map']
report={'scope':'exact patched default_unquantized_gemm, actual BF16 checkpoint weights, TP4 rank0, M1..4 candidate and M8/32 fallback; hot micro only',
        'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'torch':torch.__version__,'device':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,power.limit,clocks.mem','--format=csv'],text=True),
        'cases':[]}
report['overlay_sha256']={name:hashlib.sha256(pathlib.Path('/opt/glm53-flash/vllm/vllm/model_executor/layers',name).read_bytes()).hexdigest() for name in ['linear.py','utils.py','homelab_dense.py']}
def save(): (root/'measurements.json').write_text(json.dumps(report,indent=2))
torch.cuda.set_device(0);torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
report['allow_bf16_reduced_precision_reduction']=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
def check(actual, expected):
    a=actual.float(); b=expected.float()
    assert torch.isfinite(a).all() and a.abs().max()>0
    rel=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))
    cos=float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))
    assert rel<0.006 and cos>0.99998,(rel,cos)
    return {'relative_l2':rel,'cosine':cos}

for suffix, axis in [('layers.3.self_attn.q_b_proj.weight',0),
                      ('layers.3.self_attn.o_proj.weight',1),
                      ('layers.0.self_attn.o_proj.weight',1)]:
    key='model.language_model.'+suffix
    with safe_open(str(weights/index[key]),framework='pt',device='cpu') as f: w=f.get_tensor(key)
    w=w.chunk(4,dim=axis)[0].contiguous().cuda()
    layer=torch.nn.Module(); layer.prefix=key.removesuffix('.weight'); layer.weight=w
    prepare_homelab_dense(layer)
    assert layer._homelab_dense_cute, 'candidate was not selected'
    for m in [1,2,3,4,8,32]:
        gen=torch.Generator(device='cuda');gen.manual_seed(731+m)
        variants=[torch.randn((m,w.shape[1]),device='cuda',dtype=torch.bfloat16,generator=gen) for _ in range(2)]
        refs=[torch.nn.functional.linear(x.float(),w.float()).to(torch.bfloat16) for x in variants]
        x=variants[0].clone()
        for backend in ['cublas-before','candidate','cublas-after']:
            case={'weight':key,'weight_shape':list(w.shape),'m':m,'backend':backend,'passed':False}
            report['cases'].append(case);save()
            try:
                torch.backends.cuda.preferred_blas_library('cublaslt' if backend=='cublaslt' else 'cublas')
                fn=(lambda: default_unquantized_gemm(layer,x,w)) if backend=='candidate' else (lambda: torch.nn.functional.linear(x,w))
                x.copy_(variants[0]);out=fn();check(out,refs[0])
                for _ in range(3):out=fn()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8):out=fn()
                compiled=(len(ll_bf16_gemm_kernel._compiled_cache),len(ll_bf16_gemm_kernel._splitk_cache))
                before=torch.cuda.memory_allocated()
                case['mutations']=[]
                for n in [0,1,0]:
                    x.copy_(variants[n]);out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
                    case['mutations'].append(check(out,refs[n]))
                assert torch.cuda.memory_allocated()==before,'replay allocation changed'
                assert compiled==(len(ll_bf16_gemm_kernel._compiled_cache),len(ll_bf16_gemm_kernel._splitk_cache))
                samples=[]
                for _ in range(5):
                    for _ in range(10):graph.replay()
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100):graph.replay()
                    end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000/800)
                case.update(passed=True,hot_us=statistics.median(samples),samples_us=samples,graph_calls=8)
                print(json.dumps(case),flush=True)
                del graph,out
            except Exception as exc:
                case['error']=repr(exc);print('FAILED',case,flush=True)
            save()
        del x,variants,refs
    del w,layer;gc.collect();torch.cuda.empty_cache()
report['completed']=True;save();print('PROBE_COMPLETED',flush=True)
