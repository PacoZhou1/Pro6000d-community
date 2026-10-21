#!/usr/bin/env python3
"""CPU-only mock ABI/dispatch contract check. No torch/CUDA library is loaded."""
import argparse
import contextlib
import copy
import importlib
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    from candidate_abi import CandidateABI, LIBRARY_SHA
    import candidate_abi
    meta = dict(abi_version=1, N=34816, K=5120, cuda_arch='sm_120a', min_M=257, max_M=32768,
                mainloop_variant='warp24n64_v3', private_namespace='pro6000d_mainloop_warp24n64_v3',
                same_m64_mainloop_type=True, m64_mma_atom_layout_mnk=[2,4,1], m64_mma_permutation_n=64,
                m64_smem_copy_atom_b='SM75_U32x4_LDSM_N', m64_smem_copy_b_bytes_per_thread=16,
                fused_epilogue_tile='64x256')
    class Fn:
        def __init__(self, result): self.result, self.calls = result, []
        def __call__(self, *values): self.calls.append(values); return self.result
    def handle(metadata):
        return types.SimpleNamespace(sm120_scheduler_abi_version=Fn(1),
           sm120_scheduler_metadata=Fn(json.dumps(metadata).encode()),
           sm120_fused_fp8_temp_scale_bytes=Fn(16160*136*4),
           sm120_fused_fp8_workspace_bytes=Fn(1), sm120_fused_fp8=Fn(0),
           sm120_scheduler_last_error=Fn(b'fake_error'))
    def rejected(fn):
        try: fn()
        except RuntimeError: return True
        raise AssertionError('invalid contract accepted')
    h = handle(meta)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)/'mock-library'; path.write_bytes(b'mock; never loaded')
        with patch.object(candidate_abi.hashlib, 'sha256', return_value=types.SimpleNamespace(hexdigest=lambda:LIBRARY_SHA)), patch.object(candidate_abi.ctypes, 'CDLL', return_value=h):
            abi = CandidateABI(str(path),LIBRARY_SHA)
            assert len(h.sm120_fused_fp8.argtypes)==14
            assert len(h.sm120_fused_fp8_workspace_bytes.argtypes)==3
            assert abi.storage_bytes(16160,1,1)==(16160*136*4,1)
            abi.launch(*range(1,9),16160,1,1,9,1,0)
            assert h.sm120_fused_fp8.calls[-1][-1]==0
            rejection = {
                'wrong_sha':rejected(lambda:CandidateABI(str(path),'0'*64)),
                'wrong_M':rejected(lambda:abi.storage_bytes(2741,1,1)),
                'wrong_raster':rejected(lambda:abi.storage_bytes(16160,2,1)),
                'wrong_swizzle':rejected(lambda:abi.storage_bytes(16160,1,8)),
                '13_argument_ABI2':rejected(lambda:abi.launch(*range(13))),
                'null_buffer':rejected(lambda:abi.launch(0,*range(2,9),16160,1,1,9,1,0)),
            }
            bad = copy.deepcopy(meta); bad['mainloop_variant']='warp24n64'
            with patch.object(candidate_abi.ctypes,'CDLL',return_value=handle(bad)):
                rejection['v2_variant']=rejected(lambda:CandidateABI(str(path),LIBRARY_SHA))
    # Structural dispatcher execution using metadata-only mock tensors.
    class Device:
        type='cuda'
    device=Device()
    class Tensor:
        next_ptr=4096
        def __init__(self, shape, dtype, stride=None):
            self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype;self.device=device
            self._stride=stride or (shape[1],1) if len(shape)==2 else (1,)
            self.ptr=Tensor.next_ptr;Tensor.next_ptr+=4096
        def stride(self): return self._stride
        def data_ptr(self): return self.ptr
        def numel(self):
            import math
            return math.prod(self.shape)
        def element_size(self): return {'fp8':1,'bf16':2,'fp32':4,'u8':1}[self.dtype]
        def record_stream(self,stream): pass
        def t(self): return Tensor(self.shape[::-1],self.dtype,self._stride[::-1])
    events=[]
    fake=types.SimpleNamespace(float8_e4m3fn='fp8',bfloat16='bf16',float32='fp32',uint8='u8',
        empty=lambda shape,dtype,device:Tensor(shape,dtype),
        cuda=types.SimpleNamespace(device=lambda d:contextlib.nullcontext(),get_device_capability=lambda d:(12,0),current_stream=lambda d:types.SimpleNamespace(cuda_stream=0)),
        ops=types.SimpleNamespace(_C=types.SimpleNamespace(cutlass_scaled_mm=lambda *a:events.append('canonical_gemm')),
            pro6000d_fp8=types.SimpleNamespace(silu_and_mul_per_block_quant=lambda *a:events.append('v2_quant'))))
    with patch.dict(sys.modules, {'torch':fake}):
        op=importlib.import_module('candidate_op'); op._ABI=abi; op._DISPATCH_EVIDENCE=False
        w=Tensor((34816,5120),'fp8');bs=Tensor((272,40),'fp32')
        entry={'packed_weight':Tensor((34816,5120),'fp8'),'packed_scale':Tensor((272,40),'fp32')}
        op.lookup_packed=lambda b,s:entry if b is w and s is bs else None
        branches=[]
        for m,registered in [(16160,True),(16160,False),(1,True),(5,True),(256,True),(2741,True),(32768,True)]:
            a=Tensor((m,5120),'fp8');asc=Tensor((m,40),'fp32',(1,m))
            events.clear();before=len(h.sm120_fused_fp8.calls)
            out,sc=op._dispatch(a,w if registered else Tensor((34816,5120),'fp8'),asc,bs)
            candidate=m==16160 and registered
            assert (len(h.sm120_fused_fp8.calls)==before+1) is candidate
            assert events==([] if candidate else ['canonical_gemm','v2_quant'])
            assert out.shape==(m,17408) and sc.shape==(m,136) and sc.stride()==(1,m)
            branches.append({'M':m,'registered':registered,'dispatch':op.last_dispatch_for_eager_smoke()})
        rejection['wrong_input_dtype']=rejected(lambda:op._dispatch(Tensor((16160,5120),'bf16'),w,Tensor((16160,40),'fp32',(1,16160)),bs))
    report={'passed':True,'scope':'CPU mock ABI/shape/identity branch contract only; no CUDA/numerics/Graph/Inductor test',
            'ctypes_launch_arguments':14,'workspace_query_arguments':3,'stream_zero_accepted':True,
            'rejected':rejection,'branches':branches}
    text=json.dumps(report,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
