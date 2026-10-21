"""Wait for the owned service, then run identical unprofiled API/quality cases."""
import argparse
import json
import pathlib
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--output', required=True)
p.add_argument('--tag', required=True)
p.add_argument('--witness', action='store_true', help='Short Nsight real-kernel dispatch capture; not performance')
a = p.parse_args()
assert a.name.startswith('codex-sm120-mainloop-model-')
root = pathlib.Path(a.output)
root.mkdir(parents=True, exist_ok=True)
deadline = time.monotonic() + 1200
while True:
    state = json.loads(subprocess.check_output(['docker', 'inspect', a.name]))[0]
    if not state['State']['Running']:
        raise RuntimeError('owned service exited during startup')
    try:
        with urllib.request.urlopen('http://127.0.0.1:18000/health', timeout=3) as r:
            if r.status == 200:
                break
    except Exception:
        if time.monotonic() >= deadline:
            raise RuntimeError('startup deadline exceeded')
        time.sleep(10)

# Performance uses the exact existing payload builder: independent UUID prefixes
# per index, same inputs across fresh A1/B/A2 engines, no captured timing.
# Thinking quality is covered by the normally completed open-output probes.
# This owned service deliberately has no API key in its command/environment.
runtime = {k: state[k] for k in ['Id', 'Image', 'State', 'Mounts']}
runtime['cmd'] = state['Config']['Cmd']
runtime['env'] = [v for v in state['Config']['Env']
                  if v.startswith(('VLLM_', 'HF_HUB_', 'TRANSFORMERS_', 'PYTHONPATH=',
                                   'GEMM_', 'PRO6000D_'))]
(root / 'runtime.json').write_text(json.dumps(runtime, indent=2))
commands = [
    ('prefill', ['python3', '/tmp/qwen-profile-experiment/profile_client.py',
                     '--tag', a.tag, '--mode', 'baseline', '--profiler-installed', 'none',
                     '--cases', 'prefill', '--repeats', '12']),
    ('decode', ['python3', '/tmp/qwen-profile-experiment/profile_client.py',
                 '--tag', a.tag, '--mode', 'baseline', '--profiler-installed', 'none',
                 '--cases', 'decode', '--repeats', '6']),
    ('quality', ['python3', '/tmp/qwen-spec-experiment/quality_probe.py', '--tag', a.tag]),
    ('open-output', ['python3', '/tmp/qwen-spec-experiment/open_output_probe.py', '--tag', a.tag]),
]
if a.witness:
    commands = [('requests', ['python3', '/tmp/qwen-profile-experiment/profile_client.py',
                              '--tag', a.tag, '--mode', 'capture', '--profiler-installed', 'cuda',
                              '--cases', 'prefill', '--repeats', '1'])]
result = []
for label, command in commands:
    with (root / (label + '.jsonl')).open('w') as out, (root / (label + '.stderr')).open('w') as err:
        completed = subprocess.run(command, stdout=out, stderr=err, timeout=1200)
    result.append({'kind': label, 'exit_code': completed.returncode, 'command': command})
    (root / 'status.json').write_text(json.dumps(result, indent=2))
    if completed.returncode:
        raise RuntimeError(label + ' failed; inspect recorded evidence')
print(json.dumps(result), flush=True)
