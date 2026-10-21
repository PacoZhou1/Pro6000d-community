"""Compile and execute CuTe ownership enumeration in a CPU-only container."""
import hashlib,json,subprocess
from pathlib import Path
root=Path('/work/sm120-mainloop');out=Path('/results');inc=Path('/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=root/'ownership_probe.cu';binary=out/'ownership-probe'
cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O0','--expt-relaxed-constexpr','-arch=sm_120','-I',str(inc),str(source),'-o',str(binary)]
r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'compile.log').write_text(r.stdout)
status={'scope':'CPU-only CuTe coordinate enumeration; not GPU correctness','command':cmd,'source_sha256':sha(source),'compile_exit_code':r.returncode,'header_sha256':{str(p.relative_to(inc)):sha(p) for p in [inc/'cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl',inc/'cute/arch/mma_sm120.hpp']}}
if r.returncode==0:
 with (out/'ownership.jsonl').open('w') as f,(out/'run.stderr').open('w') as err:proc=subprocess.run([str(binary),'--full'],text=True,stdout=f,stderr=err)
 status.update(run_exit_code=proc.returncode,binary_sha256=sha(binary),output_sha256=sha(out/'ownership.jsonl'))
 if proc.returncode==0:
  rows=[json.loads(x) for x in (out/'ownership.jsonl').read_text().splitlines()];status['geometries']=[{k:v for k,v in x.items() if k!='thread_coordinates'} for x in rows]
(out/'ownership-status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status))
raise SystemExit(r.returncode or status.get('run_exit_code',1))
