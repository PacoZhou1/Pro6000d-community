"""Fresh A1/B/A2: V2 baseline versus F1+F2 with only null-metadata fix.

No diagnostic reference work or synchronization is enabled in these arms.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading

from run_f_diagnostic import ROOT, LEGACY, LIBSHA, verify, monitor, dump


def run(label):
    out=ROOT/label
    instrument=ROOT/'f-mdnone-fix'
    status={'state':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'fresh V2-only / V2+F1+F2-null-md-fix / V2-only, no diagnostic', 'arms':[]}
    dump(out/'status.json',status)
    active=None
    stopped=threading.Event(); watcher=None
    try:
        verify(out/'services-before.json')
        names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
        if any(x.startswith('codex-') for x in names):raise RuntimeError('another owned job is active')
        hashes={str(p.relative_to(instrument)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in instrument.rglob('*') if p.is_file() and p.suffix in {'.py','.so'}}
        assert hashes['v2/libfp8_silu_quant.so']==LIBSHA
        assert hashes['sitecustomize.py']=='854043107312ee708f4107e35362d6096ce270cc33d4a50293d1f2026df1bbcc'
        dump(out/'source-manifest.json',hashes)
        watcher=threading.Thread(target=monitor,args=(out/'embedding-monitor.jsonl',stopped),daemon=True)
        watcher.start()
        for arm in ('a1','b','a2'):
            p=out/arm;p.mkdir()
            verify(p/'services-before.json')
            active='codex-fp8-takeover-'+label+'-'+arm
            command=['python3',str(LEGACY/'experiment_control.py'),'--name',active,
                     '--method','mtp','--tokens','4','--profiler','none',
                     '--output-dir',str(p/'prof'),'--instrument-dir',str(instrument)]
            env={'PRO6000D_F_V2':'1','PRO6000D_FP8_FUSION_ENABLE':'1',
                 'PRO6000D_FP8_FUSION_LIBRARY':'/instrument/v2/libfp8_silu_quant.so',
                 'PRO6000D_FP8_FUSION_LIBRARY_SHA256':LIBSHA,
                 'PRO6000D_F_KERNELS_SHA256':hashes['f_kernels.py'],
                 'PRO6000D_F_ENABLE':'norm,conv' if arm=='b' else ''}
            for key,value in env.items():command+=['--extra-env',key+'='+value]
            dump(p/'launch-command.json',command)
            with (p/'launch.log').open('w') as log:
                subprocess.run(command,check=True,timeout=120,stdout=log,stderr=subprocess.STDOUT)
            with (p/'api.log').open('w') as log:
                result=subprocess.run(['python3',str(LEGACY/'run_api_arm.py'),'--name',active,
                     '--output',str(p),'--tag','takeover-'+label+'-'+arm],
                     timeout=1800,stdout=log,stderr=subprocess.STDOUT)
            with (p/'server.log').open('w') as log:
                subprocess.run(['docker','logs',active],stdout=log,stderr=subprocess.STDOUT)
            subprocess.run(['docker','stop','--timeout','60',active],stdout=subprocess.DEVNULL)
            subprocess.run(['docker','rm',active],stdout=subprocess.DEVNULL)
            active=None
            verify(p/'services-after.json')
            status['arms'].append({'arm':arm,'exit_code':result.returncode})
            dump(out/'status.json',status)
            if result.returncode:raise RuntimeError(arm+' API arm failed')
        status['state']='completed'
    except Exception as exc:
        status.update(state='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        if active:
            with (out/'failure-server.log').open('w') as log:
                subprocess.run(['docker','logs',active],stdout=log,stderr=subprocess.STDOUT)
            subprocess.run(['docker','stop','--timeout','60',active],stdout=subprocess.DEVNULL)
            subprocess.run(['docker','rm',active],stdout=subprocess.DEVNULL)
        stopped.set()
        if watcher:watcher.join(timeout=20)
        try:verify(out/'services-after.json')
        except Exception as exc:status.update(state='failed',service_error=type(exc).__name__)
        status['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump(out/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args();assert re.fullmatch(r'ffix-v[0-9]+',args.label)
    if args.run:run(args.label)
    else:
        out=ROOT/args.label;out.mkdir(exist_ok=False)
        with (out/'detached.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--label',args.label,'--run'],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'label':args.label,'pid':proc.pid,'output':str(out)}))
