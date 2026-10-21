"""CPU-only export of one completed owned witness; no model-service controls."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path('/home/llm/.local/share/qwen-sm120-scheduler')
OUTPUT=ROOT/'model-witness-v1/witness'
REPORTS=list(OUTPUT.glob('*.nsys-rep'))
assert len(REPORTS)==1, REPORTS
IMAGE='sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
NSYS='/home/llm/.local/share/qwen-profiling/nsys-2026.5.1'
command=['docker','run','--rm','--name','codex-sm120-export-witness','--runtime','runc','--network','none','--memory','8g','--cpus','2','--mount',f'type=bind,src={OUTPUT},dst=/profiles','--mount',f'type=bind,src={NSYS},dst=/nsys,readonly','--entrypoint','/nsys/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys',IMAGE,'export','--type=sqlite','--output=/profiles/dispatch.sqlite','/profiles/'+REPORTS[0].name]
with (OUTPUT/'export.log').open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=120)
status={'exit_code':r.returncode,'command':command,'report':REPORTS[0].name,'report_sha256':hashlib.sha256(REPORTS[0].read_bytes()).hexdigest()}
if (OUTPUT/'dispatch.sqlite').exists():status['sqlite_sha256']=hashlib.sha256((OUTPUT/'dispatch.sqlite').read_bytes()).hexdigest()
(OUTPUT/'export-status.json').write_text(json.dumps(status,indent=2)+'\n')
assert r.returncode==0
for script in ['analyze_dispatch.py','summarize_integration.py']:
 cmd=['python3',str(ROOT/'work/gemm-epilogue'/script),str(OUTPUT/'dispatch.sqlite') if script=='analyze_dispatch.py' else str(OUTPUT)]
 if script=='analyze_dispatch.py':cmd+=['--output',str(OUTPUT/'dispatch-summary.json')]
 subprocess.run(cmd,check=True)
