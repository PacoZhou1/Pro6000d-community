"""Detached CUDA output-contract probe; no model and no original service mutation."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import threading

from run_f_diagnostic import ROOT, verify, monitor

OUT = ROOT / 'null-v1'
NAME = 'codex-fp8-takeover-null'
IMAGE = 'sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')


def run():
    status = {'state': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    dump('status.json', status)
    stopped = threading.Event()
    watcher = None
    created = False
    try:
        verify(OUT / 'services-before.json')
        names = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()
        if any(n.startswith('codex-') for n in names):
            raise RuntimeError('another owned experiment is running')
        command = ['docker', 'create', '--name', NAME, '--gpus', 'all', '--ipc=host',
                   '--network', 'none', '--memory', '12g', '--cpus', '4',
                   '-v', str(ROOT) + ':/work:ro',
                   '-v', '/tmp/qwen-sm120-next-f/instr:/original:ro',
                   '-v', str(OUT) + ':/results', '--entrypoint', 'python3', IMAGE,
                   '/work/verify_null_metadata.py', '--original-site', '/original/sitecustomize.py',
                   '--fixed-site', '/work/f-mdnone-fix/sitecustomize.py',
                   '--official-source', '/work/source/qwen_gdn_linear_attn.py',
                   '--output', '/results/null-metadata-poison.json']
        dump('command.json', command)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        created = True
        watcher = threading.Thread(target=monitor, args=(OUT / 'embedding-monitor.jsonl', stopped), daemon=True)
        watcher.start()
        with (OUT / 'run.log').open('w') as log:
            result = subprocess.run(['docker', 'start', '-a', NAME], stdout=log, stderr=subprocess.STDOUT, timeout=300)
        obj = json.loads(subprocess.check_output(['docker', 'inspect', NAME]))[0]
        dump('container.json', {'id': obj['Id'], 'image': obj['Image'], 'state': obj['State'],
                                'command': obj['Config']['Cmd'], 'mounts': obj['Mounts']})
        status['exit_code'] = obj['State']['ExitCode']
        if result.returncode or status['exit_code']:
            raise RuntimeError('CUDA poison probe failed; original logs retained')
        if not json.loads((OUT / 'null-metadata-poison.json').read_text()).get('passed'):
            raise RuntimeError('CUDA output-contract result failed')
        status['state'] = 'completed'
    except Exception as exc:
        status.update(state='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if created:
            subprocess.run(['docker', 'stop', '--timeout', '30', NAME], stdout=subprocess.DEVNULL)
            subprocess.run(['docker', 'rm', NAME], check=True, stdout=subprocess.DEVNULL)
        stopped.set()
        if watcher:
            watcher.join(timeout=20)
        try:
            verify(OUT / 'services-after.json')
        except Exception as exc:
            status.update(state='failed', service_error=type(exc).__name__)
        status['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump('status.json', status)


if __name__ == '__main__':
    if '--run' in sys.argv:
        run()
    else:
        OUT.mkdir(exist_ok=False)
        with (OUT / 'detached.log').open('w') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--run'],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps({'pid': child.pid, 'output': str(OUT)}))
