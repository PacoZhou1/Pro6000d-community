"""Cold single-stream prefill qualification: fixed12 samples,3 rounds."""
import argparse,pathlib,subprocess,json,time,urllib.request,hashlib,re,traceback
p=argparse.ArgumentParser();p.add_argument('container');p.add_argument('case');p.add_argument('--rounds',type=int,default=3);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent.parent;out=root/'runs'/a.case;out.mkdir(exist_ok=False,parents=True)
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def request(path):
 with op.open('http://127.0.0.1:8000'+path,timeout=5) as r:return r.read().decode()
def cap(cmd):return subprocess.check_output(cmd,text=True)
def dump(name,v):(out/name).write_text(json.dumps(v,indent=2))
def phase(v):(out/'phase').write_text(v+' '+time.strftime('%Y-%m-%dT%H:%M:%S%z'));print(v,flush=True)
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text();(out/'boot-before.txt').write_text(boot)
assert request('/health')==''
info=json.loads(cap(['docker','inspect',a.container]))[0]
env=dict(x.split('=',1) for x in info['Config']['Env'] if '=' in x)
assert env['SPECULATOR']=='mtp' and env['MTP_DEPTH']=='3'
assert all(float(x)==350 for x in cap(['nvidia-smi','--query-gpu=power.limit','--format=csv,noheader,nounits']).split())
keys=['SPECULATOR','MTP_DEPTH','TP','DCP','MAX_NUM_SEQS','MAX_NUM_BATCHED_TOKENS','VLLM_B12X_MOE_FP4_FORCE_A16','NCCL_MIN_NCHANNELS','NCCL_MAX_NCHANNELS']
dump('runtime.json',{'container':a.container,'image':info['Image'],'started':info['State']['StartedAt'],'cmd':info['Config']['Cmd'],'env':{k:env.get(k) for k in keys}})
for src,dst in [(root/'benchmarks/vendor/bench-0.6.1.py','/tmp/lab-community.py'),(root/'benchmarks/bench-prefill12.py','/tmp/lab-prefill12.py')]:subprocess.run(['docker','cp',str(src),a.container+':'+dst],check=True)
dump('sources.json',{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'benchmarks/vendor/bench-0.6.1.py',root/'benchmarks/bench-prefill12.py',pathlib.Path(__file__)]})
f=open(out/'gpu.csv','w');gpu=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem','--format=csv','-l','1'],stdout=f)
results=[]
try:
 for n in range(1,a.rounds+1):
  phase('ROUND_'+str(n));m=request('/metrics');(out/f'r{n}-metrics-before.prom').write_text(m)
  counts=[float(x) for x in re.findall(r'^vllm:num_requests_(?:running|waiting)\{[^\n]*\}\s+([0-9.e+\-]+)',m,re.M)]
  assert counts and sum(counts)==0,('preexisting workload',counts)
  dest=f'/tmp/{a.case}-r{n}.json'
  cmd=['docker','exec',a.container,'timeout','--signal=TERM','--kill-after=10','600','python3','-u','/tmp/lab-prefill12.py','--host','127.0.0.1','--port','8000','--model','GLM-5.3-Flash-NVFP4','--contexts','0','--concurrency','1','--prefill-only','--standalone-prefill','--prefill-contexts','32k','--prefill-metric','auto','--prefill-duration','45','--max-tokens','8192','--max-total-tokens','2000000','--display-mode','plain','--no-hw-monitor','--no-resume','--temperature','1','--output',dest]
  dump(f'r{n}-command.json',cmd)
  with open(out/f'r{n}.log','w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=620)
  for suffix in ['', '.samples.jsonl']:subprocess.run(['docker','cp',a.container+':'+dest+suffix,str(out/(f'r{n}.json'+suffix))],check=True)
  data=json.loads((out/f'r{n}.json').read_text());cell=data['prefill']['32768']
  samples=[json.loads(x) for x in (out/f'r{n}.json.samples.jsonl').read_text().splitlines()]
  assert len(samples)==12 and cell['samples']==12 and cell['method']=='client'
  assert all(x['sample']['prompt_tokens']>32000 and x['sample']['ttft']>0 for x in samples)
  # Exact server validation availability is reported rather than fabricated.
  validity=[x['sample'].get('server_valid',False) for x in samples]
  cache=[x['sample'].get('server_cached_tokens') for x in samples]
  if all(validity):assert all(x==0 for x in cache),('noncold server tokens',cache)
  results.append({'round':n,**cell,'all_server_validation_available':all(validity),'server_cached_tokens_per_sample':cache})
  dump('partial.json',results);(out/f'r{n}-metrics-after.prom').write_text(request('/metrics'))
 assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text()==boot
 dump('validation.json',{'passed':True,'records':results,'target':11000,'all_rounds_exceed_target':all(x['tok_per_sec']>11000 for x in results),'scope':'MTP3 single-stream cold nominal32K. Public0.6.1 TTFT/payload unchanged, fixed12 samples replaces duration stop. Each original sample persisted. Server validation unavailability is not proof of no cache.'})
 phase('DONE')
except BaseException as exc:
 dump('failure.json',{'error':repr(exc),'traceback':traceback.format_exc()});phase('FAILED');raise
finally:gpu.terminate();gpu.wait(timeout=10);f.close()
