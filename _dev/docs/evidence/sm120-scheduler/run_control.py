"""Run an isolated owned build/probe; never change any model service state."""
import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/llm/.local/share/qwen-sm120-scheduler')
OLD = Path('/home/llm/.local/share/qwen-gemm-epilogue')
IMAGE = 'sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
NCU_ROOT = Path('/home/llm/.local/share/qwen-gemm-targeted/ncu-2026.3.0')
NCU = '/ncu/opt/nvidia/nsight-compute/2026.3.0/ncu'


def verify(path, functional=False):
    command = ['python3', str(OLD/'work/gemm-epilogue/verify_service_state.py'),
               '--initial-state', str(OLD/'initial-state.json'), '--output', str(path)]
    if functional:
        command.append('--functional')
    subprocess.run(command, check=True)


def monitor(path, stopped):
    with path.open('w') as f:
        while not stopped.is_set():
            row = {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            try:
                obj = json.loads(subprocess.check_output(['docker', 'inspect', 'vllm-embedding'], timeout=10))[0]
                cmd = obj['Config']['Cmd']
                key = cmd[cmd.index('--api-key')+1]
                row.update(id=obj['Id'], started_at=obj['State']['StartedAt'],
                           restart_count=obj['RestartCount'], running=obj['State']['Running'])
                req = urllib.request.Request('http://127.0.0.1:8002/health',
                    headers={'Authorization': 'Bearer '+key})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    row['health'] = resp.status
            except Exception as exc:
                row['error_type'] = type(exc).__name__
            f.write(json.dumps(row)+'\n'); f.flush()
            stopped.wait(15)


def absent(name):
    return subprocess.run(['docker', 'inspect', name], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0


def main():
    import re
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--command-json', type=Path, required=True)
    p.add_argument('--mode', choices=['build', 'probe', 'ncu'], required=True)
    p.add_argument('--timeout', type=int, default=900)
    a = p.parse_args()
    if not re.fullmatch('[a-z0-9][a-z0-9-]{0,60}', a.label) or not 1 <= a.timeout <= 1200:
        p.error('invalid owned label/timeout')
    argv = json.loads(a.command_json.read_text())
    if not isinstance(argv, list) or not argv or argv[0] != 'python3' or not all(isinstance(x,str) for x in argv):
        p.error('command must be a Python argv list')
    output = ROOT/a.label
    output.mkdir(exist_ok=False)
    preflight={'stage':'preflight','mode':a.mode,'argv':argv,'started_unix':time.time()}
    (output/'status.json').write_text(json.dumps(preflight,indent=2)+'\n')
    try:
        verify(output/'services-before.json')
        names = subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
        if any(n.startswith(('codex-fp8-', 'codex-gemm-', 'codex-sm120-')) for n in names):
            raise RuntimeError('another owned GPU experiment is running')
    except Exception as exc:
        preflight.update(stage='preflight_failed',error_type=type(exc).__name__)
        (output/'status.json').write_text(json.dumps(preflight,indent=2)+'\n')
        raise
    snapshot = output/'source'; snapshot.mkdir()
    for path in (ROOT/'work/sm120-scheduler').iterdir():
        if path.is_file() and path.suffix in {'.py','.cu','.cuh','.h','.json'}:
            shutil.copy2(path,snapshot/path.name)
    shutil.copy2(a.command_json,output/'command.json')
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()}
    name = 'codex-sm120-'+a.label
    command = ['docker','run','--rm','--name',name,'--network','none',
               '--memory=24g','--memory-swap=24g','--cpus=3']
    command += ['--runtime','runc'] if a.mode == 'build' else ['--gpus','all','--ipc','host']
    if a.mode == 'ncu': command += ['--cap-add','SYS_ADMIN']
    for src,dst,ro in [(ROOT/'work','/work',False),(OLD/'work','/oldwork',True),
                       (ROOT/a.label,'/results',False),
                       (Path('/home/llm/.local/share/qwen-gemm-targeted/build/compiled'),'/baseline',True),
                       (Path('/home/llm/.local/share/qwen-fp8-fusion'),'/v2',True),
                       (NCU_ROOT,'/ncu',True)]:
        command += ['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    if a.mode == 'ncu':
        import importlib.util
        spec=importlib.util.spec_from_file_location('hw_control',ROOT/'work/sm120-scheduler/hardware_control.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        query=OLD/'ncu-metric-query-cap.txt'
        wanted=set(module.METRICS);found=set()
        for line in query.open():
            fields=line.split()
            if fields and fields[0] in wanted:found.add(fields[0])
        if wanted != found:raise RuntimeError('missing requested NCU metrics')
        (output/'metrics.json').write_text(json.dumps({'metrics':module.METRICS,'query_sha256':hashlib.sha256(query.read_bytes()).hexdigest()},indent=2)+'\n')
        command += ['--entrypoint',NCU,IMAGE,'--clock-control','none','--cache-control','none',
                    '--profile-from-start','off','--nvtx','--nvtx-include','regex:SM120_SCHEDULER:.*/',
                    '--section','LaunchStats','--metrics',','.join(module.METRICS),
                    '--export','/results/capture',*argv]
    else:
        command += ['--entrypoint',argv[0],IMAGE,*argv[1:]]
    status={'command':command,'mode':a.mode,'started_unix':time.time(),
            'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_sha256':source_hashes,'exit_code':None,'owned_container_present_after':None}
    (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    stopped=threading.Event();watcher=threading.Thread(target=monitor,args=(output/'embedding-monitor.jsonl',stopped),daemon=True)
    watcher.start()
    try:
        with (output/'stdout.log').open('w') as out,(output/'stderr.log').open('w') as err:
            status['exit_code']=subprocess.run(command,stdout=out,stderr=err,timeout=a.timeout).returncode
    except subprocess.TimeoutExpired:
        status['timeout']=True
        subprocess.run(['docker','stop','--time','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    finally:
        stopped.set();watcher.join(timeout=25)
        status['elapsed_s']=time.time()-status['started_unix']
        status['owned_container_present_after']=not absent(name)
        expected=next(x for x in json.loads((output/'services-before.json').read_text())['services']
                      if x['name']=='vllm-embedding')
        samples=[json.loads(line) for line in (output/'embedding-monitor.jsonl').read_text().splitlines()]
        bad=[x for x in samples if x.get('error_type') or x.get('health')!=200 or not x.get('running')
             or any(x.get(k)!=expected[k] for k in ('id','started_at','restart_count'))]
        status['embedding_monitor']={'samples':len(samples),'bad_samples':len(bad),
            'passed':bool(samples) and not bad,
            'scope':'Health sampled every 15 seconds; does not prove uninterrupted request-level availability or zero latency impact.'}
        (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        try:
            verify(output/'services-after.json',functional=True)
            status['service_after_passed']=True
        except Exception:
            status['service_after_passed']=False
            raise
        finally:
            (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    if a.mode == 'ncu' and status['exit_code'] == 0:
        reports=list(output.glob('capture.ncu-rep*'))
        if len(reports)!=1:raise RuntimeError('missing/ambiguous NCU report')
        export_name=name+'-export'
        export=['docker','run','--rm','--name',export_name,'--network','none','--runtime','runc',
                '--mount',f'type=bind,src={output},dst=/results,readonly',
                '--mount',f'type=bind,src={NCU_ROOT},dst=/ncu,readonly','--entrypoint',NCU,IMAGE,
                '--import','/results/'+reports[0].name,'--page','raw','--csv']
        try:
            with (output/'raw.csv').open('w') as out,(output/'export.stderr').open('w') as err:
                status['export_exit_code']=subprocess.run(export,stdout=out,stderr=err,timeout=120).returncode
        except subprocess.TimeoutExpired:
            status['export_timeout']=True
            subprocess.run(['docker','stop','--time','5',export_name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        finally:
            status['export_container_present_after']=not absent(export_name)
            (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps({'label':a.label,**{k:status.get(k) for k in ['exit_code','elapsed_s','export_exit_code','owned_container_present_after']}}))
    if status['exit_code'] != 0 or status['owned_container_present_after'] or not status['embedding_monitor']['passed']:
        raise SystemExit(1)
    if a.mode=='ncu' and (status.get('export_exit_code')!=0 or status.get('export_container_present_after')):
        raise SystemExit(1)


if __name__=='__main__':main()
