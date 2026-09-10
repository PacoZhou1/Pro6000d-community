"""All-rank fused-projection qualification, including beta/fa tails.
Real TP4 loading semantics; replicated fa outputs must be bitwise identical.
"""
import os,json,pathlib,hashlib
import torch,torch.distributed as dist
from safetensors import safe_open
from vllm.model_executor.layers.homelab_dense import prepare_homelab_dense
from vllm.model_executor.layers.utils import default_unquantized_gemm
from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm_kernel
rank=int(os.environ['RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
dist.init_process_group('gloo');assert dist.get_world_size()==4
torch.backends.cuda.matmul.allow_tf32=False
root=pathlib.Path('/work/fused-allranks');root.mkdir(exist_ok=True)
path=pathlib.Path('/models/GLM-5.3-Flash-NVFP4')
index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
report={'rank':rank,'passed':False,'cases':[],'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
def save():(root/f'rank{rank}.json').write_text(json.dumps(report,indent=2))
segments=[('q',0,2048),('k',2048,4096),('v',4096,6144),('beta',6144,6160),('replicated_fa',6160,6288)]
def check(a,b):
    metrics={}
    for name,lo,hi in segments:
        aa=a[:,lo:hi].float();bb=b[:,lo:hi].float()
        assert torch.isfinite(aa).all() and aa.abs().max()>0,name
        torch.testing.assert_close(aa,bb,atol=.001,rtol=.008)
        metrics[name]={'max_abs_error':float((aa-bb).abs().max()),'relative_l2':float(torch.linalg.vector_norm(aa-bb)/torch.linalg.vector_norm(bb))}
    return metrics
for layer_id in [0,30]:
    prefix=f'model.language_model.layers.{layer_id}.self_attn'
    parts=[]
    for name in ['q_proj','k_proj','v_proj','b_proj','f_a_proj']:
        key=f'{prefix}.{name}.weight'
        with safe_open(str(path/index[key]),framework='pt',device='cpu') as f:part=f.get_tensor(key)
        parts.append(part if name=='f_a_proj' else part.chunk(4,dim=0)[rank])
    w=torch.cat(parts).contiguous().cuda();assert tuple(w.shape)==(6288,4096)
    del part,parts
    layer=torch.nn.Module();layer.prefix=prefix+'.in_proj_qkvgfab';layer.weight=w
    prepare_homelab_dense(layer);assert layer._homelab_dense_cute
    for m in [1,2,3,4,8,32]:
        gen=torch.Generator(device=torch.device('cuda',rank));gen.manual_seed(731+m)
        variants=[torch.randn((m,4096),device=rank,dtype=torch.bfloat16,generator=gen) for _ in range(2)]
        refs=[torch.nn.functional.linear(x.float(),w.float()).to(torch.bfloat16) for x in variants]
        baseline=[check(torch.nn.functional.linear(x,w),ref) for x,ref in zip(variants,refs)]
        x=variants[0].clone()
        for _ in range(3):out=default_unquantized_gemm(layer,x,w)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=default_unquantized_gemm(layer,x,w)
        cache_before=(len(ll_bf16_gemm_kernel._compiled_cache),len(ll_bf16_gemm_kernel._splitk_cache))
        alloc_before=torch.cuda.memory_allocated()
        case={'layer':layer_id,'m':m,'baseline':baseline,'mutations':[],'passed':False}
        report['cases'].append(case);save()
        for seed_index in [0,1,0]:
            x.copy_(variants[seed_index]);out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            metrics=check(out,refs[seed_index])
            fa_hash=hashlib.sha256(out[:,6160:6288].contiguous().view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
            all_hashes=[None]*4;dist.all_gather_object(all_hashes,fa_hash)
            assert len(set(all_hashes))==1,('replicated_fa differs',rank,layer_id,m,all_hashes)
            case['mutations'].append({'seed_index':seed_index,'segments':metrics,'replicated_fa_hash_all_ranks':all_hashes})
        assert cache_before==(len(ll_bf16_gemm_kernel._compiled_cache),len(ll_bf16_gemm_kernel._splitk_cache))
        case['allocated_before']=alloc_before;case['allocated_after']=torch.cuda.memory_allocated()
        assert case['allocated_after']==alloc_before
        case['passed']=True;save();print('PASS_ALLRANK_TAIL',rank,layer_id,m,flush=True)
        del graph,out,x,variants,refs,baseline
    del w,layer
report['passed']=len(report['cases'])==12 and all(c['passed'] for c in report['cases']);save()
assert report['passed'];dist.barrier();dist.destroy_process_group()
