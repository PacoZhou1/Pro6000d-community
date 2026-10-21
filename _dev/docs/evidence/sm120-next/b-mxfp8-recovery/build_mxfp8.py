#!/usr/bin/env python3
"""Build the isolated CUTLASS 4.5.0 MXFP8xMXFP8 C ABI; no GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
EXPECTED_VERSION_SHA = 'b4eb73fe13a5d2c22ac514d468280db5c7f590acf89d0115e53882b26d9693ed'
EXPECTED_HEADERS = {
    'include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp': '6a347ec5c6550800410e2bda53c61213a78ecd74ed7343176bd8c2972c4c3141',
    'include/cutlass/detail/sm100_blockscaled_layout.hpp': '598e054bef21edf94b1fd6bb1447cfa9cfcf5a5907ab370128102448dbb6d530',
    'tools/util/include/cutlass/util/packed_stride.hpp': 'b43a26572905cebf99e0a9842903efd2232421f7d1c1a686711ddd636708fcbb',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--cutlass-root', type=Path, required=True,
               help='CUTLASS 4.5.0 root with include/ and tools/util/include/')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
p.add_argument('--emit-only', action='store_true')
a = p.parse_args()
root = a.cutlass_root.resolve()
version = root/'include/cutlass/version.h'
if not version.is_file() or sha(version) != EXPECTED_VERSION_SHA:
    raise SystemExit('fixed installed CUTLASS 4.5.0 version.h SHA mismatch')
for relative, expected in EXPECTED_HEADERS.items():
    if not (root/relative).is_file() or sha(root/relative) != expected:
        raise SystemExit('fixed installed CUTLASS header SHA mismatch: '+relative)
if not (root/'tools/util/include').is_dir():
    raise SystemExit('CUTLASS tools/util/include missing')
a.output.mkdir(parents=True, exist_ok=True)
for name in ('mxfp8_adapter.cu', 'mxfp8_abi.h'):
    shutil.copy2(HERE/name, a.output/name)
lib = a.output/'librecovery_mxfp8.so'
cmd = [a.nvcc, '-std=c++17', '-O3', '--expt-relaxed-constexpr',
       '--expt-extended-lambda', '-gencode', 'arch=compute_120a,code=sm_120a',
       '-lineinfo', '--resource-usage',
       '-Xcompiler=-fPIC,-fvisibility=hidden,-fvisibility-inlines-hidden,-fno-gnu-unique',
       '-shared', '-I', str(a.output), '-I', str(root/'include'),
       '-I', str(root/'tools/util/include'), str(a.output/'mxfp8_adapter.cu'),
       '-o', str(lib), '-lcudart']
manifest = {
    'status': 'emitted' if a.emit_only else 'building',
    'compile_only_no_gpu': True,
    'cutlass_version_header_sha256': sha(version),
    'fixed_cutlass_headers_sha256': EXPECTED_HEADERS,
    'cutlass_root': str(root),
    'source_sha256': {name:sha(a.output/name) for name in ('mxfp8_adapter.cu','mxfp8_abi.h')},
    'upstream_example_sha256': sha(HERE/'source/cutlass_79c_v450.cu'),
    'upstream_example_ref': 'NVIDIA/cutlass@e406c186f510a15091cce01f782020ceb7ba8eb5 examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu',
    'command':cmd,
    'library':str(lib),
}
if not a.emit_only:
    with (a.output/'compile.log').open('w') as out:
        done=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
    manifest['compile_exit_code']=done.returncode
    manifest['status']='compiled' if done.returncode == 0 else 'failed'
    manifest['compile_log_sha256']=sha(a.output/'compile.log')
    if done.returncode == 0: manifest['library_sha256']=sha(lib)
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='command'}))
if manifest['status']=='failed': raise SystemExit(1)
