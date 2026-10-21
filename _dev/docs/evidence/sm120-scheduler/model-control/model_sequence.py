"""Serial owned model witness/ABA. Original service state is never changed."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path('/home/llm/.local/share/qwen-sm120-scheduler')
sys.path.insert(0, str(ROOT))
from run_control import monitor, verify
LIBRARY = ROOT/'work/build-v3/libsm120_scheduler.so'
SHA = 'af1d814966a746ff09edc55d40196b871067d88d22ed0e50700c1f3a4eb185d6'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['witness','aba'],required=True)
    p.add_argument('--label',required=True)
    a=p.parse_args()
    import re
    if not re.fullmatch('[a-z0-9][a-z0-9-]{0,40}',a.label):p.error('fresh simple label required')
    output=ROOT/a.label;output.mkdir(exist_ok=False)
    verify(output/'services-initial.json',functional=True)
    if hashlib.sha256(LIBRARY.read_bytes()).hexdigest()!=SHA:raise RuntimeError('library SHA mismatch')
    events=[]
    def event(**data):
        data['at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();events.append(data)
        (output/'sequence.json').write_text(json.dumps(events,indent=2)+'\n');print(json.dumps(data),flush=True)
    stopped=threading.Event();watcher=threading.Thread(target=monitor,args=(output/'embedding-monitor.jsonl',stopped),daemon=True);watcher.start()
    try:
        for arm in (['witness'] if a.mode=='witness' else ['a1','b','a2']):
            name='codex-fp8-epi-'+a.label+'-'+arm
            destination=output/arm;mode='candidate' if arm in ['witness','b'] else 'baseline'
            verify(output/(arm+'-services-before.json'),functional=True)
            event(arm=arm,phase='launch',name=name,mode=mode)
            command=['python3',str(ROOT/'work/gemm-epilogue/launch_model_arm.py'),'--root',str(ROOT),'--name',name,'--label',a.label+'/'+arm,'--mode',mode]
            if mode=='candidate':command+=['--library-source',str(LIBRARY),'--library-sha256',SHA]
            if a.mode=='witness':command+=['--witness']
            try:
                subprocess.run(command,check=True,timeout=60)
                api=['python3',str(ROOT/'work/gemm-epilogue/run_api_arm.py'),'--name',name,'--output',str(destination),'--tag','sm120-official-m16160-fixed-aba']
                if a.mode=='witness':api+=['--witness']
                subprocess.run(api,check=True,timeout=2400)
                event(arm=arm,phase='measurements_completed')
            finally:
                state=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
                if state.returncode==0:
                    destination.mkdir(parents=True,exist_ok=True)
                    with (destination/'server.log').open('w') as f:subprocess.run(['docker','logs',name],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
                    subprocess.run(['docker','stop','--time','30',name],check=True,timeout=45)
                    obj=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
                    if obj['State']['Running']:raise RuntimeError('owned model still running')
                    # Exact name created by this fresh sequence; preserve all profile files.
                    subprocess.run(['docker','rm',name],check=True,timeout=30)
                verify(output/(arm+'-services-after.json'),functional=True)
            event(arm=arm,phase='removed_owned_arm')
        event(phase='completed')
    finally:
        stopped.set();watcher.join(timeout=25)
        verify(output/'services-final.json',functional=True)
        expected=next(x for x in json.loads((output/'services-initial.json').read_text())['services'] if x['name']=='vllm-embedding')
        samples=[json.loads(x) for x in (output/'embedding-monitor.jsonl').read_text().splitlines()]
        bad=[x for x in samples if x.get('error_type') or x.get('health')!=200 or not x.get('running') or any(x.get(k)!=expected[k] for k in ['id','started_at','restart_count'])]
        result={'samples':len(samples),'bad_samples':len(bad),'passed':bool(samples) and not bad,'scope':'Sampled health and identity, not zero per-request latency impact','controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (output/'monitor-summary.json').write_text(json.dumps(result,indent=2)+'\n')
        if not result['passed']:raise RuntimeError('embedding monitor failed')
if __name__=='__main__':main()
