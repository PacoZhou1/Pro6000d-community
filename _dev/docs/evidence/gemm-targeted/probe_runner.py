"""Run an owned isolated GPU probe and record embedding counters around it."""
import argparse,datetime,json,pathlib,subprocess,urllib.request,time
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--name',required=True);p.add_argument('--output',required=True);p.add_argument('--ncu',action='store_true');p.add_argument('--ncu-cold',action='store_true');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
assert a.name.startswith('codex-gemm-')
run=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
assert not any(n in run for n in ['vllm-qwen-27b','vllm-qwen3.5-9b','vllm-qwen3-asr'])
assert not any(n.startswith('codex-gemm-') for n in run)
def emb():
 d=json.loads(subprocess.check_output(['docker','inspect','vllm-embedding']))[0];cmd=d['Config']['Cmd'];key=cmd[cmd.index('--api-key')+1]
 out={'id':d['Id'],'started_at':d['State']['StartedAt'],'restart_count':d['RestartCount'],'running':d['State']['Running']}
 h={'Authorization':'Bearer '+key}
 for route in ['health','metrics']:
  with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8002/'+route,headers=h),timeout=10) as r:
   if route=='health':out['health']=r.status
   else:
    met={}
    for line in r.read().decode().splitlines():
     if not line.startswith('vllm:'):continue
     name=line.split('{')[0].split(' ')[0]
     if name.endswith('_total'):met[name]=met.get(name,0)+float(line.rsplit(' ',1)[1])
    out['metrics']=met
 return out
root=pathlib.Path(a.output);root.mkdir(parents=True,exist_ok=True)
before=emb();t=time.time()
command=['docker','run','--rm','--name',a.name,'--gpus','all','--ipc','host','--memory=24g','--memory-swap=24g','--mount','type=bind,src=/home/llm/.local/share/qwen-gemm-targeted/build,dst=/work']
if a.ncu:
 command += ['--cap-add=SYS_ADMIN','--mount','type=bind,src=/home/llm/.local/share/qwen-gemm-targeted/ncu-2026.3.0,dst=/ncu,readonly','--mount',f'type=bind,src={root},dst=/results','--entrypoint','/ncu/opt/nvidia/nsight-compute/2026.3.0/ncu','sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b','--clock-control','none','--cache-control','all' if a.ncu_cold else 'none','--profile-from-start','off','--launch-count','1','--metrics','gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active','--export','/results/counter','--force-overwrite','python3']+a.args
else:
 command += ['--entrypoint','python3','sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b']+a.args
with (root/'results.jsonl').open('w') as out,(root/'stderr.log').open('w') as err:
 result=subprocess.run(command,stdout=out,stderr=err)
after=emb();data={'at_unix':t,'elapsed_s':time.time()-t,'exit_code':result.returncode,'command':command,'embedding_before':before,'embedding_after':after}
(root/'status.json').write_text(json.dumps(data,indent=2));print(json.dumps(data))
raise SystemExit(result.returncode)
