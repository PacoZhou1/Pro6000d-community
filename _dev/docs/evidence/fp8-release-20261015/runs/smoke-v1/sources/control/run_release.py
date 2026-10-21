"""Detached, serial fixed-image FP8 comparison; only exact owned names controlled."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path('/home/llm/.local/share/qwen-fp8-release-20261015')
CONTROL = ROOT / 'control'
BASE = 'sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
OLD = 'sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967'
KV_BYTES = 12 * 1024**3


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]


def verify(path):
    subprocess.run([sys.executable, str(CONTROL / 'verify_service_state.py'),
                    '--initial-state', str(CONTROL / 'protected-state.json'),
                    '--output', str(path), '--functional'], check=True)


def watch_embedding(path, stopped):
    with path.open('w') as out:
        while not stopped.is_set():
            record = {'at_utc': now()}
            try:
                obj = inspect('vllm-embedding')
                cmd = obj['Config']['Cmd']
                record.update(id=obj['Id'], running=obj['State']['Running'],
                              started_at=obj['State']['StartedAt'], restart_count=obj['RestartCount'])
                req = urllib.request.Request('http://127.0.0.1:8002/health',
                        headers={'Authorization': 'Bearer ' + cmd[cmd.index('--api-key') + 1]})
                with urllib.request.urlopen(req, timeout=10) as response:
                    record['health'] = response.status
            except Exception as exc:
                record['error_type'] = type(exc).__name__
            out.write(json.dumps(record) + '\n')
            out.flush()
            stopped.wait(15)


def capture_runtime(name, path):
    obj = inspect(name)
    dump(path, {k: obj[k] for k in ('Id', 'Image', 'State', 'Mounts')} | {
         'cmd': obj['Config']['Cmd'],
         'env': [s for s in obj['Config']['Env'] if s.startswith(
                 ('VLLM_', 'PRO6000D_', 'HF_HUB_', 'TRANSFORMERS_', 'PYTHONPATH='))]})


def command(arm, image, output, label, overrides):
    original = inspect('vllm-qwen-27b')
    args = original['Config']['Cmd'][:]
    i = args.index('--api-key')
    del args[i:i + 2]

    def change(flag, value):
        if flag in args:
            args[args.index(flag) + 1] = str(value)
        else:
            args.extend([flag, str(value)])

    old = arm.startswith('o')
    for flag, value in [('--gpu-memory-utilization', .70), ('--kv-cache-memory-bytes', KV_BYTES),
                        ('--max-model-len', 262144), ('--max-num-seqs', 16),
                        ('--max-num-batched-tokens', 8192 if old else 32768),
                        ('--host', '127.0.0.1'), ('--port', 18000),
                        ('--attention-backend', 'FLASHINFER'), ('--kv-cache-dtype', 'fp8'),
                        ('--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 2 if old else 4}))]:
        change(flag, value)
    model = next(m['Source'] for m in original['Mounts'] if m['Destination'] == '/model')
    name = 'codex-fp8-release-' + label + '-' + arm
    env = ['HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'VLLM_WORKER_MULTIPROC_METHOD=spawn']
    if not old:
        env.append('VLLM_USE_V2_MODEL_RUNNER=1')
    if arm == 'b':
        env.extend(overrides)
    cmd = ['docker', 'create', '--name', name, '--gpus', 'all', '--ipc=host', '--network=host',
           '--restart=no', '--memory=40g', '--memory-swap=40g', '--ulimit', 'memlock=-1:-1',
           '--mount', f'type=bind,src={model},dst=/model,readonly',
           '--mount', f'type=bind,src={output},dst=/profiles']
    for item in env:
        cmd.extend(['--env', item])
    # Each image's own entrypoint is used, including the fixed release image.
    cmd += [image] + args
    return name, cmd


def wait_ready(name):
    deadline = time.monotonic() + 1500
    while True:
        obj = inspect(name)
        if not obj['State']['Running']:
            raise RuntimeError('owned model exited during startup')
        try:
            with urllib.request.urlopen('http://127.0.0.1:18000/health', timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            if time.monotonic() >= deadline:
                raise RuntimeError('startup health deadline exceeded')
        time.sleep(8)


def run(args):
    output = ROOT / 'runs' / args.label
    state = {'state': 'running', 'started_utc': now(), 'stage': args.stage,
             'candidate_image': args.image, 'arms': [], 'kv_bytes': KV_BYTES,
             'candidate_overrides': args.extra_env}
    dump(output / 'status.json', state)
    active = None
    arm_output = None
    clock_proc = None
    clock_file = None
    stopped = threading.Event()
    watcher = None
    try:
        verify(output / 'services-before.json')
        live = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()
        if any(n.startswith('codex-') for n in live):
            raise RuntimeError('another owned build or GPU experiment is running')
        # Protected-service verification above includes authenticated health
        # and an actual embedding request. Never probe the experiment through
        # a pre-existing listener belonging to somebody else.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 18000))
        resolved = json.loads(subprocess.check_output(['docker', 'image', 'inspect', args.image]))[0]
        if resolved['Id'] != args.image:
            raise RuntimeError('candidate must be an immutable image ID')
        files = [*CONTROL.glob('*.py'), CONTROL / 'protected-state.json', * (ROOT / 'clients').glob('*.py')]
        dump(output / 'source-manifest.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in files})
        watcher = threading.Thread(target=watch_embedding,
                 args=(output / 'embedding-monitor.jsonl', stopped), daemon=True)
        watcher.start()
        arms = ['b'] if args.stage == 'smoke' else ['o1', 'a1', 'b', 'a2', 'o2']
        for arm in arms:
            arm_output = output / arm
            arm_output.mkdir()
            verify(arm_output / 'services-before.json')
            chosen = OLD if arm.startswith('o') else BASE if arm.startswith('a') else args.image
            name, cmd = command(arm, chosen, arm_output, args.label, args.extra_env)
            dump(arm_output / 'launch-command.json', cmd)
            with (arm_output / 'launch.log').open('w') as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
                active = name
                subprocess.run(['docker', 'start', active], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            clock_file = (arm_output / 'gpu-clock.csv').open('w')
            clock_proc = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used',
                           '--format=csv', '-lms', '500'], stdout=clock_file, stderr=subprocess.STDOUT)
            wait_ready(active)
            capture_runtime(active, arm_output / 'runtime.json')
            commands = []
            cases = [('prefill', 12), ('decode', 6), ('thinking', 6)] if args.stage == 'compare' else [('prefill', 1), ('decode', 1)]
            for case, n in cases:
                commands.append((case, [sys.executable, str(CONTROL / 'profile_client.py'), '--tag', args.label + '-' + arm,
                    '--mode', 'baseline', '--profiler-installed', 'none', '--cases', case, '--repeats', str(n)]))
            commands.append(('natural', [sys.executable, str(ROOT / 'clients/natural_probe.py'),
                      '--tag', args.label + '-' + arm, '--base', 'http://127.0.0.1:18000',
                      '--allow-no-key',
                      '--model', 'qwen3.6-27b-fp8', '--output', str(arm_output / 'acceptance.jsonl')]))
            phase_results = []
            for phase, cmd in commands:
                suffix = '.log' if phase == 'natural' else '.jsonl'
                with (arm_output / (phase + suffix)).open('w') as out, (arm_output / (phase + '.stderr')).open('w') as err:
                    completed = subprocess.run(cmd, stdout=out, stderr=err, timeout=1800)
                phase_results.append({'kind': phase, 'exit_code': completed.returncode, 'command': cmd})
                dump(arm_output / 'client-status.json', phase_results)
                if completed.returncode:
                    raise RuntimeError(arm + '/' + phase + ' failed; raw result retained')
            with (arm_output / 'server.log').open('w') as log:
                subprocess.run(['docker', 'logs', active], stdout=log, stderr=subprocess.STDOUT, check=True)
            capture_runtime(active, arm_output / 'runtime-after.json')
            subprocess.run(['docker', 'stop', '--timeout', '60', active], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(['docker', 'rm', active], check=True, stdout=subprocess.DEVNULL)
            active = None
            clock_proc.terminate()
            clock_proc.wait(timeout=10)
            clock_proc = None
            clock_file.close()
            clock_file = None
            verify(arm_output / 'services-after.json')
            state['arms'].append({'arm': arm, 'completed_utc': now(), 'image': chosen, 'exit_code': 0})
            dump(output / 'status.json', state)
        state['state'] = 'completed'
    except Exception as exc:
        state.update(state='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if active:
            try:
                with (arm_output / 'failure-server.log').open('w') as log:
                    subprocess.run(['docker', 'logs', active], stdout=log, stderr=subprocess.STDOUT, timeout=30)
                capture_runtime(active, arm_output / 'failure-runtime.json')
            except Exception as exc:
                state['failure_capture_error'] = type(exc).__name__
            finally:
                for cmd in (['docker', 'stop', '--timeout', '60', active], ['docker', 'rm', active]):
                    try:
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, timeout=90, check=True)
                    except Exception as exc:
                        state.update(state='failed', cleanup_error=type(exc).__name__)
        if clock_proc:
            clock_proc.terminate()
            try:
                clock_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                clock_proc.kill()
                clock_proc.wait()
        if clock_file:
            clock_file.close()
        stopped.set()
        if watcher:
            watcher.join(timeout=20)
        try:
            verify(output / 'services-after.json')
        except Exception as exc:
            state.update(state='failed', service_error=type(exc).__name__)
        state['ended_utc'] = now()
        dump(output / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--stage', choices=['smoke', 'compare'], required=True)
    parser.add_argument('--extra-env', action='append', default=[])
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch(r'(smoke|compare)-v[0-9]+', args.label)
    assert re.fullmatch(r'sha256:[a-f0-9]{64}', args.image)
    allowed = {'PRO6000D_RELEASE_V2', 'PRO6000D_RELEASE_F', 'PRO6000D_RELEASE_DRAFT64K'}
    assert all(x.partition('=')[0] in allowed and x.partition('=')[2] in ('0', '1') for x in args.extra_env)
    if args.run:
        run(args)
    else:
        out = ROOT / 'runs' / args.label
        out.mkdir(parents=True, exist_ok=False)
        with (out / 'detached.log').open('w') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--run'],
                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'output': str(out)}))
