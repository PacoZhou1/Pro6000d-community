"""Root-owned detached diagnostic, preserving embedding and stopped originals.

GPU timings from this diagnostic are explicitly not performance evidence.
Uses the unchanged FP8 experiment controller and request payload builder.
"""
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
import time
import urllib.request

ROOT = Path('/home/llm/.local/share/qwen-sm120-takeover')
LEGACY = Path('/home/llm/.local/share/qwen-fp8-fusion')
VERIFY = Path('/home/llm/.local/share/qwen-gemm-epilogue')
LIBSHA = '087b7caa92c4742ece3875b6dd1421e25360b834ce4106f7fa166f90bc933b3f'


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def verify(path):
    subprocess.run(['python3', str(VERIFY/'work/gemm-epilogue/verify_service_state.py'),
                    '--initial-state', str(VERIFY/'initial-state.json'),
                    '--output', str(path), '--functional'], check=True)


def monitor(path, stopped):
    with path.open('w') as out:
        while not stopped.is_set():
            row = {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            try:
                o = json.loads(subprocess.check_output(['docker', 'inspect', 'vllm-embedding']))[0]
                cmd = o['Config']['Cmd']
                row.update(id=o['Id'], started_at=o['State']['StartedAt'],
                           restart_count=o['RestartCount'], running=o['State']['Running'])
                req = urllib.request.Request('http://127.0.0.1:8002/health',
                    headers={'Authorization': 'Bearer ' + cmd[cmd.index('--api-key')+1]})
                with urllib.request.urlopen(req, timeout=10) as r:
                    row['health'] = r.status
            except Exception as e:
                row['error_type'] = type(e).__name__
            out.write(json.dumps(row)+'\n'); out.flush(); stopped.wait(15)


def run(label):
    out = ROOT/label
    instr = ROOT/'f-diagnostic'
    name = 'codex-fp8-takeover-'+label
    status = {'state': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'real model diagnostic; all timings invalid for speed claims', 'name': name}
    dump(out/'controller-status.json', status)
    verify(out/'services-before.json')
    names = subprocess.check_output(['docker','ps','--format','{{.Names}}'], text=True).splitlines()
    if any(n.startswith('codex-') for n in names):
        raise RuntimeError('another owned experiment is active')
    hashes = {str(p.relative_to(instr)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in instr.rglob('*') if p.is_file() and p.suffix in {'.py','.so'}}
    assert hashes['v2/libfp8_silu_quant.so'] == LIBSHA
    dump(out/'source-manifest.json', hashes)
    args = ['python3', str(LEGACY/'experiment_control.py'), '--name', name, '--method', 'mtp',
            '--tokens', '4', '--profiler', 'none', '--output-dir', str(out/'prof'),
            '--instrument-dir', str(instr)]
    env = {'PRO6000D_F_V2':'1','PRO6000D_FP8_FUSION_ENABLE':'1',
           'PRO6000D_FP8_FUSION_LIBRARY':'/instrument/v2/libfp8_silu_quant.so',
           'PRO6000D_FP8_FUSION_LIBRARY_SHA256':LIBSHA,
           'PRO6000D_F_ENABLE':'norm,conv', 'PRO6000D_F_KERNELS_SHA256':hashes['f_kernels.py'],
           'PRO6000D_F_VERIFY':'1', 'PRO6000D_F_VERIFY_CONV':'1',
           'PRO6000D_F_DIAGNOSTIC_SHA256':hashes['f_diagnostic.py'],
           'PRO6000D_F_DIAG_DIR':'/profiles'}
    for key, value in env.items():
        args += ['--extra-env', key+'='+value]
    dump(out/'launch-command.json', args)
    stopped = threading.Event()
    worker = threading.Thread(target=monitor, args=(out/'embedding-monitor.jsonl', stopped), daemon=True)
    worker.start()
    try:
        subprocess.run(args, check=True, timeout=120, stdout=open(out/'launch.log','w'), stderr=subprocess.STDOUT)
        with (out/'api.log').open('w') as log:
            subprocess.run(['python3', str(LEGACY/'run_api_arm.py'), '--name', name,
                            '--output',str(out), '--tag','takeover-fdiag'],
                           check=True, timeout=1800, stdout=log, stderr=subprocess.STDOUT)
        status['state'] = 'completed'
    except Exception as e:
        status.update(state='failed', error_type=type(e).__name__)
        raise
    finally:
        # Exact owned name only; original services are never controlled here.
        with (out/'server.log').open('w') as log:
            subprocess.run(['docker','logs',name], stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(['docker','stop','--timeout','60',name], stdout=subprocess.DEVNULL)
        subprocess.run(['docker','rm',name], stdout=subprocess.DEVNULL)
        stopped.set(); worker.join(timeout=20)
        verify(out/'services-after.json')
        status['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump(out/'controller-status.json', status)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    assert re.fullmatch(r'fdiag-v[0-9]+', a.label)
    if a.run:
        run(a.label)
    else:
        out = ROOT/a.label; out.mkdir(exist_ok=False)
        with (out/'detached.log').open('w') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--label', a.label, '--run'], stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'label':a.label,'pid':child.pid,'output':str(out)}))
