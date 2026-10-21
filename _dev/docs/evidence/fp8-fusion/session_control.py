"""Capture/monitor state and pause/restore exactly the three authorized models.

Run on the GPU host. Authentication is read in memory, never written to evidence.
Embedding has no lifecycle operation. No experiments may be running at restore.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time
import urllib.request

SERVICES = [('vllm-qwen-27b', 8000), ('vllm-qwen3.5-9b', 8001),
            ('vllm-qwen3-asr', 8003), ('vllm-embedding', 8002)]


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def health(obj, port):
    cmd = obj['Config']['Cmd']
    key = cmd[cmd.index('--api-key') + 1]
    request = urllib.request.Request(f'http://127.0.0.1:{port}/health',
                                    headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status


def state(name, port):
    obj = inspect(name)
    cmd = list(obj['Config']['Cmd'])
    for i, value in enumerate(cmd):
        if value in ('--api-key', '--password', '--token', '--hf-token'):
            cmd[i + 1] = '<REDACTED>'
        elif any(value.startswith(flag + '=') for flag in
                 ('--api-key', '--password', '--token', '--hf-token')):
            cmd[i] = value.split('=')[0] + '=<REDACTED>'
    result = {'name': name, 'id': obj['Id'], 'image': obj['Image'],
              'cmd': cmd, 'started_at': obj['State']['StartedAt'],
              'restart_count': obj['RestartCount'], 'running': obj['State']['Running']}
    result['health'] = health(obj, port) if result['running'] else None
    return result


def embedding_unchanged(initial):
    current = state('vllm-embedding', 8002)
    expected = initial['containers']['vllm-embedding']
    for key in ('id', 'image', 'cmd', 'started_at', 'restart_count'):
        assert current[key] == expected[key], 'embedding changed: ' + key
    assert current['running'] and current['health'] == 200
    return current


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['snapshot', 'pause', 'restore', 'monitor'])
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    if args.action == 'snapshot':
        target = root / 'initial-state.json'
        assert not target.exists(), 'do not overwrite the original state'
        obj = {'at': utc(), 'containers': {n: state(n, port) for n, port in SERVICES}}
        assert all(s['running'] and s['health'] == 200 for s in obj['containers'].values())
        obj['gpu'] = subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,uuid,driver_version,memory.total,memory.used,power.limit',
            '--format=csv,noheader'], text=True).strip()
        target.write_text(json.dumps(obj, indent=2) + '\n')
        print(json.dumps({'at': obj['at'], 'all_services_healthy': True}))
        return
    initial = json.loads((root / 'initial-state.json').read_text())
    if args.action == 'monitor':
        with (root / 'embedding-monitor.jsonl').open('a', buffering=1) as out:
            while not (root / 'monitor.stop').exists():
                row = {'at_utc': utc()}
                try:
                    row.update(embedding_unchanged(initial))
                except Exception as exc:
                    row['error_type'] = type(exc).__name__
                # Remove command details to keep regular observations small.
                row.pop('cmd', None)
                out.write(json.dumps(row) + '\n')
                time.sleep(30)
        return
    running = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'],
                                      text=True).splitlines()
    assert not any(n.startswith(('codex-gemm-', 'codex-fp8-')) for n in running)
    events = []
    for name, port in SERVICES[:-1]:
        embedding_unchanged(initial)
        current = state(name, port)
        expected = initial['containers'][name]
        for key in ('id', 'image', 'cmd'):
            assert current[key] == expected[key], name + ' changed: ' + key
        start = utc()
        if args.action == 'pause':
            assert current['running'], 'expected original service running'
            subprocess.run(['docker', 'stop', '--time', '60', name], check=True,
                           stdout=subprocess.DEVNULL)
        else:
            subprocess.run(['docker', 'start', name], check=True,
                           stdout=subprocess.DEVNULL)
            deadline = time.monotonic() + 1200
            while True:
                embedding_unchanged(initial)
                try:
                    if state(name, port)['health'] == 200:
                        break
                except Exception:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError(name + ' startup timeout')
                time.sleep(10)
        events.append({'name': name, 'started_utc': start, 'finished_utc': utc(),
                       'state_after': state(name, port)})
        (root / (args.action + '-sequence.json')).write_text(json.dumps(events, indent=2) + '\n')
    embedding_unchanged(initial)
    print(json.dumps({'action': args.action, 'completed': True, 'embedding_unchanged': True}))


if __name__ == '__main__':
    main()
