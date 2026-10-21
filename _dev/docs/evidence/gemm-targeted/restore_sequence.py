"""Restore only the three paused original models; never restart embedding."""
import datetime
import json
from pathlib import Path
import subprocess
import time
import urllib.request

root = Path('/home/llm/.local/share/qwen-gemm-targeted')
initial = json.loads(Path('/tmp/qwen-gemm-targeted/initial-state.json').read_text())['containers']
names = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()
assert not any(n.startswith('codex-gemm-') for n in names), 'stop owned experiments first'
events = []

def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]

def health(name, port):
    obj = inspect(name)
    cmd = obj['Config']['Cmd']
    key = cmd[cmd.index('--api-key') + 1]
    req = urllib.request.Request(f'http://127.0.0.1:{port}/health',
                                 headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(req, timeout=5) as response:
        return response.status

def embedding_unchanged():
    obj, ref = inspect('vllm-embedding'), initial['vllm-embedding']
    assert obj['Id'] == ref['id']
    assert obj['State']['StartedAt'] == ref['started_at']
    assert obj['RestartCount'] == ref['restart_count']
    assert health('vllm-embedding', 8002) == 200

for name, port in [('vllm-qwen-27b', 8000), ('vllm-qwen3.5-9b', 8001), ('vllm-qwen3-asr', 8003)]:
    embedding_unchanged()
    before = inspect(name)
    assert before['Id'] == initial[name]['id']
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    subprocess.run(['docker', 'start', name], check=True, stdout=subprocess.DEVNULL)
    deadline = time.monotonic() + 1200
    while True:
        embedding_unchanged()
        try:
            if health(name, port) == 200:
                break
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError(name + ' health deadline exceeded')
        time.sleep(10)
    after = inspect(name)
    events.append({'name': name, 'requested_start_utc': started,
                   'healthy_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'id': after['Id'], 'started_at': after['State']['StartedAt'],
                   'restart_count_before': before['RestartCount'],
                   'restart_count_after': after['RestartCount'], 'health': 200})
    (root / 'restore-sequence.json').write_text(json.dumps(events, indent=2) + '\n')
    print(json.dumps(events[-1]), flush=True)
subprocess.run(['python3', '/tmp/qwen-gemm-targeted/restore_verify.py',
                '--initial-state', '/tmp/qwen-gemm-targeted/initial-state.json',
                '--output', str(root / 'restoration.json')], check=True,
               stdout=subprocess.DEVNULL)
print('Original services functional verification passed.', flush=True)
