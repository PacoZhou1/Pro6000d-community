#!/usr/bin/env python3
"""Random FP8 gate/up GEMM comparison on fixed vLLM0.31/torch2.13/SM120.

GEMM BF16 output only; each scale scheme has its own numeric reference.
No checkpoint loading, activation quantization, or hidden M padding.
"""
import argparse
import ctypes as C
import ctypes.util
import hashlib
import json
import math
import random
import statistics
import threading
import time
from pathlib import Path

N,K=34816,5120

class NvmlPoller:
    """Best-effort per-sample SM-clock/power witness; not used for event time."""
    def __init__(self, index=0):
        self.lib=None;self.handle=C.c_void_p();self.error=None
        self.samples=[];self.stop=threading.Event();self.worker=None
        try:
            self.lib=C.CDLL(ctypes.util.find_library('nvidia-ml') or 'libnvidia-ml.so.1')
            if self.lib.nvmlInit_v2():raise RuntimeError('nvmlInit_v2 failed')
            if self.lib.nvmlDeviceGetHandleByIndex_v2(index,C.byref(self.handle)):
                raise RuntimeError('nvmlDeviceGetHandleByIndex_v2 failed')
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}';self.lib=None
    def read(self):
        if self.lib is None:return None,None
        clock=C.c_uint();power=C.c_uint()
        c=self.lib.nvmlDeviceGetClockInfo(self.handle,1,C.byref(clock))
        p=self.lib.nvmlDeviceGetPowerUsage(self.handle,C.byref(power))
        return (int(clock.value) if c==0 else None,
                float(power.value)/1000 if p==0 else None)
    def __enter__(self):
        if self.lib is not None:
            self.worker=threading.Thread(target=self._run,daemon=True);self.worker.start()
        return self
    def _run(self):
        while not self.stop.is_set():
            self.samples.append((time.perf_counter(),*self.read()))
            self.stop.wait(.002)
    def window(self,t0,t1):
        # A 1–10 ms replay may have no exact poll. Mark padded readings as such.
        exact=[x for x in self.samples if t0<=x[0]<=t1]
        chosen=exact or [x for x in self.samples if t0-.008<=x[0]<=t1+.008]
        def mean(col):
            values=[x[col] for x in chosen if x[col] is not None]
            return statistics.mean(values) if values else None
        return {'sm_clock_mhz':mean(1),'power_w':mean(2),
                'nvml_samples':len(chosen),'nvml_exact_window':bool(exact)}
    def __exit__(self,*exc):
        self.stop.set()
        if self.worker is not None:self.worker.join()

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def bind(lib,name,restype,args):
    fn=getattr(lib,name);fn.restype=restype;fn.argtypes=args;return fn

def mx_library(path):
    lib=C.CDLL(str(Path(path).resolve()))
    if bind(lib,'recovery_mxfp8_abi_version',C.c_int,[])()!=1:raise RuntimeError('MX ABI mismatch')
    q=[bind(lib,'recovery_mxfp8_'+x+'_bytes',C.c_size_t,[C.c_int]*3)
       for x in ('scale_a','scale_b','workspace')]
    pack=bind(lib,'recovery_mxfp8_pack_scales',C.c_int,
        [C.c_void_p]*4+[C.c_int]*3+[C.c_size_t]*2)
    launch=bind(lib,'recovery_mxfp8_launch',C.c_int,
        [C.c_void_p]*5+[C.c_int]*3+[C.c_void_p,C.c_size_t,C.c_ulonglong])
    last=bind(lib,'recovery_mxfp8_last_error',C.c_char_p,[])
    return q,pack,launch,last

def pr_library(path):
    lib=C.CDLL(str(Path(path).resolve()))
    if bind(lib,'recovery_pr5521_abi_version',C.c_int,[])()!=1:raise RuntimeError('PR-derived ABI mismatch')
    fn=bind(lib,'recovery_pr5521_launch',C.c_int,[C.c_void_p]*5+[C.c_int]*3+[C.c_ulonglong])
    last=bind(lib,'recovery_pr5521_last_error',C.c_char_p,[])
    return fn,last

def emit(path,kind,**data):
    row={'kind':kind,**data}
    with open(path,'a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    if kind!='sample':print(json.dumps(row,sort_keys=True),flush=True)

def fp8(torch,shape,seed):
    torch.manual_seed(seed)
    out=torch.empty(shape,dtype=torch.float8_e4m3fn,device='cuda')
    chunk=max(1,(1<<27)//shape[1])
    for lo in range(0,shape[0],chunk):
        out[lo:lo+chunk]=(torch.randn((min(chunk,shape[0]-lo),shape[1]),device='cuda',dtype=torch.bfloat16)*.5).to(out.dtype)
    return out

def sample_reference(torch,a,w,scale_kind,sa,sb,rows,cols):
    # FP8 advanced indexing is not assumed: index uint8 bytes, then reinterpret.
    ri=torch.tensor(rows,device='cuda');ci=torch.tensor(cols,device='cuda')
    av=a.view(torch.uint8).index_select(0,ri).view(torch.float8_e4m3fn).float()
    bv=w.view(torch.uint8).index_select(0,ci).view(torch.float8_e4m3fn).float()
    if scale_kind=='rowwise':
        return ((av@bv.T)*sa.index_select(0,ri)[:,None]*sb.index_select(0,ci)[None,:]).to(torch.bfloat16).float()
    group=32 if scale_kind=='mxfp8' else 128
    acc=torch.zeros((len(rows),len(cols)),device='cuda',dtype=torch.float32)
    for kb in range(K//group):
        term=av[:,kb*group:(kb+1)*group]@bv[:,kb*group:(kb+1)*group].T
        if scale_kind=='block128':
            scale_a=sa.index_select(0,ri)[:,kb]
            scale_b=sb.index_select(0,ci//128)[:,kb]
        else:
            scale_a=sa.index_select(0,ri)[:,kb]
            scale_b=sb.index_select(0,ci)[:,kb]
        acc+=term*scale_a[:,None]*scale_b[None,:]
    return acc.to(torch.bfloat16).float()

def numeric(torch,out,ref,rows,cols):
    allfinite=bool(torch.isfinite(out).all().item())
    q=out.index_select(0,torch.tensor(rows,device='cuda')).index_select(1,torch.tensor(cols,device='cuda')).float()
    diff=q-ref
    nrmse=float((diff.square().sum()/ref.square().sum().clamp_min(1e-30)).sqrt().item())
    cosine=float(torch.nn.functional.cosine_similarity(q.flatten(),ref.flatten(),dim=0).item())
    maxabs=float(diff.abs().max().item())
    passed=allfinite and math.isfinite(nrmse) and math.isfinite(cosine) and nrmse<.005 and cosine>.9999
    return {'finite':allfinite,
            'nrmse':nrmse if math.isfinite(nrmse) else None,
            'cosine':cosine if math.isfinite(cosine) else None,
            'max_abs':maxabs if math.isfinite(maxabs) else None,
            'pass':passed}

def full_compare(torch, actual, reference_chunks, max_nrmse):
    """Compare every BF16 output element in row chunks; no huge FP32 reference."""
    mismatch=0;count=0;diff2=0.;ref2=0.;dot=0.;q2=0.;maxabs=0.;finite=True
    blocks=[]
    for lo,ref in reference_chunks:
        q=actual[lo:lo+ref.shape[0]]
        if q.dtype!=torch.bfloat16 or ref.dtype!=torch.bfloat16:
            raise RuntimeError('full reference must be BF16')
        block_finite=bool(torch.isfinite(q).all().item()) and bool(torch.isfinite(ref).all().item())
        finite &= block_finite
        block_mismatch=int(torch.count_nonzero(q.view(torch.int16)!=ref.view(torch.int16)).item())
        mismatch+=block_mismatch
        qf=q.float();rf=ref.float();d=qf-rf
        block_diff2=float(d.square().sum(dtype=torch.float64).item())
        block_ref2=float(rf.square().sum(dtype=torch.float64).item())
        block_maxabs=float(d.abs().max().item())
        diff2+=block_diff2;ref2+=block_ref2
        q2+=float(qf.square().sum(dtype=torch.float64).item())
        dot+=float((qf*rf).sum(dtype=torch.float64).item())
        maxabs=max(maxabs,block_maxabs)
        count+=q.numel()
        block_nrmse=math.sqrt(block_diff2/max(block_ref2,1e-30))
        blocks.append({'row_begin':lo,'row_end_exclusive':lo+ref.shape[0],
                       'bf16_bit_mismatch':block_mismatch,
                       'nrmse':block_nrmse if math.isfinite(block_nrmse) else None,
                       'max_abs':block_maxabs if math.isfinite(block_maxabs) else None,
                       'finite':block_finite})
    nrmse=math.sqrt(diff2/max(ref2,1e-30))
    cosine=dot/math.sqrt(max(q2*ref2,1e-30))
    passed=finite and count==actual.numel() and math.isfinite(nrmse) and math.isfinite(cosine) and nrmse<max_nrmse and cosine>.9999
    return {'elements':count,'blocks':blocks,'bf16_bit_mismatch':mismatch,
            'bf16_bit_match_fraction':1-mismatch/max(count,1),
            'finite':finite,
            'nrmse':nrmse if math.isfinite(nrmse) else None,
            'cosine':cosine if math.isfinite(cosine) else None,
            'max_abs':maxabs if math.isfinite(maxabs) else None,
            'pass':passed}

def mx_bf16_reference_chunks(torch,a,w,sa,sb,row_chunk=128,col_chunk=512):
    """Independent MX math: exact power-of-two dequant to BF16 then BF16 GEMM."""
    wb=torch.empty(w.shape,dtype=torch.bfloat16,device='cuda')
    for lo in range(0,w.shape[0],col_chunk):
        hi=min(lo+col_chunk,w.shape[0])
        wb[lo:hi]=(w[lo:hi].float().reshape(hi-lo,K//32,32)
                   *sb[lo:hi,:,None]).reshape(hi-lo,K).to(torch.bfloat16)
    for lo in range(0,a.shape[0],row_chunk):
        hi=min(lo+row_chunk,a.shape[0])
        ab=(a[lo:hi].float().reshape(hi-lo,K//32,32)
            *sa[lo:hi,:,None]).reshape(hi-lo,K).to(torch.bfloat16)
        yield lo,ab@wb.T

def existing_chunks(reference,row_chunk=128):
    for lo in range(0,reference.shape[0],row_chunk):
        yield lo,reference[lo:lo+row_chunk]

def capture(torch,call):
    side=torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=side):out=call()
    torch.cuda.synchronize()
    graph.replay();torch.cuda.synchronize()
    return graph,out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mx-library',type=Path,required=True)
    p.add_argument('--pr-library',type=Path)
    p.add_argument('--m',type=int,choices=[2741,16160],required=True)
    p.add_argument('--pairs',type=int,default=31)
    p.add_argument('--seed',type=int,default=5521)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.pairs<1: p.error('positive pairs required')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():p.error('fresh output required')
    import numpy as np
    import torch
    import vllm._custom_ops as ops
    torch.backends.cuda.matmul.allow_tf32=False
    if torch.cuda.get_device_capability()!=(12,0):raise RuntimeError('fixed SM120 required')
    nvml=NvmlPoller(torch.cuda.current_device())
    q,pack,mxlaunch,mxerr=mx_library(a.mx_library)
    pr=None if a.pr_library is None else pr_library(a.pr_library)
    emit(a.output,'meta',m=a.m,n=N,k=K,seed=a.seed,pairs=a.pairs,device=torch.cuda.get_device_name(),
         torch=torch.__version__,mx_library_sha256=sha(a.mx_library),
         pr_library_sha256=sha(a.pr_library) if a.pr_library else None,
         bench_sha256=sha(Path(__file__)),nvml_error=nvml.error,
         scope='random FP8 GEMM->BF16 only; scales differ by scheme; no model quality or conversion/packing cost')
    A=fp8(torch,(a.m,K),a.seed);W=fp8(torch,(N,K),a.seed+1);Bt=W.t()
    gen=np.random.default_rng(a.seed+2)
    # Fixed vLLM block128 ABI requires activation scales [M,K/128] with
    # column-major physical storage (stride 1,M), not a contiguous matrix.
    ba=torch.from_numpy(gen.uniform(.5,1.5,(K//128,a.m)).astype('float32')).to('cuda').T
    bs=torch.from_numpy(gen.uniform(.5,1.5,(N//128,K//128)).astype('float32')).to('cuda')
    ra=torch.from_numpy(gen.uniform(.5,1.5,a.m).astype('float32')).to('cuda')
    rb=torch.from_numpy(gen.uniform(.5,1.5,N).astype('float32')).to('cuda')
    # UE8M0 codes are powers of two, deliberately nontrivial; native packing is CPU preparation.
    ca=gen.integers(124,131,(a.m,K//32),dtype=np.uint8)
    cb=gen.integers(124,131,(N,K//32),dtype=np.uint8)
    sizes=[fn(a.m,N,K) for fn in q]
    if any(x<1 for x in sizes):raise RuntimeError('MX query failed: '+str(mxerr()))
    pa=np.empty(sizes[0],dtype=np.uint8);pb=np.empty(sizes[1],dtype=np.uint8)
    rc=pack(ca.ctypes.data,cb.ctypes.data,pa.ctypes.data,pb.ctypes.data,a.m,N,K,pa.nbytes,pb.nbytes)
    if rc:raise RuntimeError('MX pack failed: '+str(mxerr()))
    MXA=torch.from_numpy(pa).to('cuda');MXB=torch.from_numpy(pb).to('cuda')
    workspace=torch.empty((sizes[2],),dtype=torch.uint8,device='cuda')
    mxout=torch.empty((a.m,N),dtype=torch.bfloat16,device='cuda')
    def mx():
        rc=mxlaunch(mxout.data_ptr(),A.data_ptr(),W.data_ptr(),MXA.data_ptr(),MXB.data_ptr(),
            a.m,N,K,workspace.data_ptr(),workspace.numel(),torch.cuda.current_stream().cuda_stream)
        if rc:raise RuntimeError('MX launch failed: '+str(mxerr()))
        return mxout
    arms={'cutlass_block128':lambda:ops.cutlass_scaled_mm(A,Bt,out_dtype=torch.bfloat16,
                               scale_a=ba,scale_b=bs.T),
          'cublas_rowwise':lambda:torch._scaled_mm(A,Bt,ra[:,None],rb[None,:],out_dtype=torch.bfloat16),
          'cutlass_mxfp8':mx}
    mxa=torch.from_numpy(np.ldexp(np.ones(ca.shape,dtype='float32'),ca.astype('int32')-127)).to('cuda')
    mxb=torch.from_numpy(np.ldexp(np.ones(cb.shape,dtype='float32'),cb.astype('int32')-127)).to('cuda')
    if pr:
        prlaunch,prerr=pr
        half=N//2
        order=np.empty(N,dtype=np.int64)
        for g in range(half//8):
            base=16*g
            order[base:base+8]=np.arange(8*g,8*g+8)
            order[base+8:base+16]=np.arange(half+8*g,half+8*g+8)
        idx=torch.from_numpy(order).to('cuda')
        packedW=W.view(torch.uint8).index_select(0,idx).view(torch.float8_e4m3fn)
        packedRb=rb.index_select(0,idx)
        prout=torch.empty((a.m,N),dtype=torch.bfloat16,device='cuda')
        def prcall():
            rc=prlaunch(prout.data_ptr(),A.data_ptr(),packedW.data_ptr(),ra.data_ptr(),
                packedRb.data_ptr(),a.m,N,K,torch.cuda.current_stream().cuda_stream)
            if rc:raise RuntimeError('PR-derived launch failed: '+str(prerr()))
            return prout
        arms['pr5521_derived_rowwise']=prcall
    # Always include both tails and the midpoint; a sorted prefix would
    # silently drop the last row for some M values.
    rows=sorted(set([0,a.m//2,a.m-1,*[round(i*(a.m-1)/12) for i in range(13)]]))
    cols=sorted(set([*range(0,64),*range(N//2,N//2+64),*range(N-64,N)]))
    emit(a.output,'shape_scope',logical_m=a.m,logical_n=N,logical_k=K,
         cutlass_m_tile=128,cutlass_executed_m_tiles=(a.m+127)//128,
         cutlass_tile_covered_rows=((a.m+127)//128)*128,
         cutlass_valid_rows=a.m,
         pr5521_derived_m_tile=128 if pr else None,
         pr5521_derived_tile_covered_rows=((a.m+127)//128)*128 if pr else None,
         pr5521_derived_tail_valid_rows=a.m%128 if pr else None,
         notes='tile coverage is not extra logical input rows; per-arm actual launch grid must be verified from compiled kernel')
    refs={'cutlass_block128':sample_reference(torch,A,W,'block128',ba,bs,rows,cols),
          'cublas_rowwise':sample_reference(torch,A,W,'rowwise',ra,rb,rows,cols),
          'cutlass_mxfp8':sample_reference(torch,A,W,'mxfp8',mxa,mxb,rows,cols)}
    if pr:refs['pr5521_derived_rowwise']=refs['cublas_rowwise']
    numeric_outputs={}
    for name,call in arms.items():
        out=call();torch.cuda.synchronize();check=numeric(torch,out,refs[name],rows,cols)
        emit(a.output,'numeric',arm=name,rows=rows,columns_sampled=len(cols),**check)
        if not check['pass']:raise RuntimeError(name+' numeric gate failed')
        if name in ('cublas_rowwise','pr5521_derived_rowwise','cutlass_mxfp8'):
            numeric_outputs[name]=out
    if pr:
        check=full_compare(torch,numeric_outputs['pr5521_derived_rowwise'],
                           existing_chunks(numeric_outputs['cublas_rowwise']),.002)
        emit(a.output,'full_matrix_numeric',arm='pr5521_derived_rowwise',
             reference='same-input cublas_rowwise BF16',**check)
        if not check['pass']:raise RuntimeError('PR-derived full BF16 rowwise comparison failed')
    check=full_compare(torch,numeric_outputs['cutlass_mxfp8'],
                       mx_bf16_reference_chunks(torch,A,W,mxa,mxb),.005)
    emit(a.output,'full_matrix_numeric',arm='cutlass_mxfp8',
         reference='power-of-two K32 dequant BF16 A/W then BF16 matmul',**check)
    if not check['pass']:raise RuntimeError('MX full BF16 dequant comparison failed')
    del numeric_outputs
    graphs={}
    for name,call in arms.items():graphs[name]=capture(torch,call)
    # Changed-input replay catches stale captured outputs. Restore common input for timing.
    old=A.view(torch.uint8).clone();alt=fp8(torch,(a.m,K),a.seed+3)
    A.view(torch.uint8).copy_(alt.view(torch.uint8));torch.cuda.synchronize()
    for name,(graph,out) in graphs.items():
        before=out[0,:32].clone();graph.replay();torch.cuda.synchronize()
        changed=not torch.equal(before.view(torch.uint8),out[0,:32].view(torch.uint8))
        scheme={'cutlass_block128':('block128',ba,bs),
                'cublas_rowwise':('rowwise',ra,rb),
                'cutlass_mxfp8':('mxfp8',mxa,mxb),
                'pr5521_derived_rowwise':('rowwise',ra,rb)}[name]
        new_ref=sample_reference(torch,alt,W,*scheme,rows,cols)
        check=numeric(torch,out,new_ref,rows,cols)
        emit(a.output,'graph_changed_input',arm=name,output_changed=changed,
             reference='changed FP8 A, same W/scales',**check)
        if not changed or not check['pass']:
            raise RuntimeError(name+' graph replay did not match changed-input reference')
    if pr:
        check=full_compare(torch,graphs['pr5521_derived_rowwise'][1],
                           existing_chunks(graphs['cublas_rowwise'][1]),.002)
        emit(a.output,'graph_changed_full_matrix_numeric',
             arm='pr5521_derived_rowwise',reference='changed-input cublas_rowwise BF16',**check)
        if not check['pass']:raise RuntimeError('PR-derived changed-input full BF16 comparison failed')
    A.view(torch.uint8).copy_(old);torch.cuda.synchronize()
    for graph,_ in graphs.values():graph.replay()
    torch.cuda.synchronize()
    l2=torch.cuda.get_device_properties(0).L2_cache_size
    flush=torch.empty((2*l2+4096,),dtype=torch.uint8,device='cuda')
    rng=random.Random(a.seed+9)
    samples=[]
    with nvml:
        for mode in ('warm','disturbed'):
            for pair in range(a.pairs):
                order=list(graphs);rng.shuffle(order)
                for pos,name in enumerate(order):
                    if mode=='disturbed':flush.fill_(pair&255);torch.cuda.synchronize()
                    beg=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    t0=time.perf_counter();beg.record();graphs[name][0].replay();end.record();end.synchronize()
                    t1=time.perf_counter();ms=beg.elapsed_time(end)
                    sample=dict(mode=mode,pair=pair,pos=pos,arm=name,ms=ms,
                                **nvml.window(t0,t1))
                    samples.append(sample);emit(a.output,'sample',**sample)
    for mode in ('warm','disturbed'):
        for name in arms:
            vals=[x['ms'] for x in samples if x['mode']==mode and x['arm']==name]
            emit(a.output,'summary',mode=mode,arm=name,n=len(vals),median_ms=statistics.median(vals),
                 min_ms=min(vals),max_ms=max(vals),tflops=2*a.m*N*K/(statistics.median(vals)*1e-3)/1e12)
            if name!='cutlass_block128':
                base={x['pair']:x['ms'] for x in samples if x['mode']==mode and x['arm']=='cutlass_block128'}
                gains=[base[x['pair']]/x['ms'] for x in samples if x['mode']==mode and x['arm']==name]
                emit(a.output,'paired_gain',mode=mode,arm=name,
                     numerator='same-pair cutlass_block128 GEMM ms',denominator='arm GEMM ms',
                     n=len(gains),median_ratio=statistics.median(gains),
                     min_ratio=min(gains),max_ratio=max(gains))
    emit(a.output,'complete',passed=True,limitations='Disturbance writes own >2xL2 buffer, not guaranteed cold L2; GPU event graph replay excludes CPU dispatch, all arm preprocessing excluded.')
if __name__=='__main__':main()
