"""Detached synthetic FP8 repeat with 15 s GPU thermal settling. Only exact owned experiment names are controlled."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import urllib.request

from run_f_diagnostic import ROOT, verify, monitor

IMAGE = 'sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def metrics(path):
    obj = json.loads(subprocess.check_output(['docker', 'inspect', 'vllm-embedding']))[0]
    cmd = obj['Config']['Cmd']
    request = urllib.request.Request('http://127.0.0.1:8002/metrics',
        headers={'Authorization': 'Bearer '+cmd[cmd.index('--api-key')+1]})
    with urllib.request.urlopen(request, timeout=15) as response:
        lines = response.read().decode().splitlines()
    path.write_text('\n'.join(x for x in lines if x.startswith('vllm:'))+'\n')


def run(label):
    out = ROOT/label
    source = ROOT/'b-mxfp8-recovery'
    status = {'state':'running', 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope':'random GEMM only; 15 s settling before timings; no model loaded or converted', 'image':IMAGE, 'arms':[]}
    dump(out/'status.json',status)
    active_name = None
    stopped = threading.Event()
    watcher = None
    try:
        verify(out/'services-before.json')
        names = subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
        if any(x.startswith('codex-') for x in names):
            raise RuntimeError('another owned experiment/build is still running')
        files = [source/'bench_recovery_steady.py',source/'build-mxfp8/librecovery_mxfp8.so',
                 source/'build-pr5521/libpr5521_derived.so',Path(__file__),
                 ROOT/'run_f_diagnostic.py']
        dump(out/'source-manifest.json', {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in files})
        watcher = threading.Thread(target=monitor,args=(out/'embedding-monitor.jsonl',stopped),daemon=True)
        watcher.start()
        for m in (2741,16160):
            metrics(out/f'embedding-before-m{m}.prom')
            active_name = f'codex-fp8-takeover-{label}-m{m}'
            command = ['docker','run','--name',active_name,'--gpus','all','--ipc=host',
                       '--network','none','--memory','24g','--cpus','6',
                       '-v',str(source)+':/work:ro','-v',str(out)+':/results',
                       '-w','/work','--entrypoint','python3',IMAGE,'/work/bench_recovery_steady.py',
                       '--mx-library','/work/build-mxfp8/librecovery_mxfp8.so',
                       '--pr-library','/work/build-pr5521/libpr5521_derived.so',
                       '--m',str(m),'--pairs','31','--output',f'/results/m{m}.jsonl']
            dump(out/f'command-m{m}.json',command)
            with (out/f'run-m{m}.log').open('w') as log:
                result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            state = json.loads(subprocess.check_output(['docker','inspect',active_name]))[0]
            dump(out/f'container-m{m}.json', {'id':state['Id'],'image':state['Image'],
                 'state':state['State'],'command':state['Config']['Cmd'],
                 'mounts':state['Mounts'],'runtime':state['HostConfig']['Runtime']})
            subprocess.run(['docker','rm',active_name],check=True,stdout=subprocess.DEVNULL)
            active_name = None
            metrics(out/f'embedding-after-m{m}.prom')
            status['arms'].append({'m':m,'exit_code':result.returncode})
            dump(out/'status.json',status)
            if result.returncode:
                raise RuntimeError(f'M{m} numerical/build/runtime gate failed; logs retained')
        status['state'] = 'completed'
    except Exception as exc:
        status.update(state='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        if active_name:
            subprocess.run(['docker','stop','--timeout','60',active_name],stdout=subprocess.DEVNULL)
            subprocess.run(['docker','rm',active_name],stdout=subprocess.DEVNULL)
        stopped.set()
        if watcher: watcher.join(timeout=20)
        try:
            verify(out/'services-after.json')
        except Exception as exc:
            status.update(state='failed',service_verification_error=type(exc).__name__)
        status['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump(out/'status.json',status)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    assert re.fullmatch(r'gemm-v[0-9]+',args.label)
    if args.run:
        run(args.label)
    else:
        out=ROOT/args.label
        out.mkdir(exist_ok=False)
        with (out/'detached.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--label',args.label,'--run'],
                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'label':args.label,'pid':child.pid,'output':str(out)}))
