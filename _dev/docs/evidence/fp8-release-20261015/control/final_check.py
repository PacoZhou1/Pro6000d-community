"""Read-only identity checks plus synthetic API calls for the retained service."""
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT = Path('/home/llm/.local/share/qwen-fp8-release-20261015')
NAME = 'pro6000d-qwen36-fp8-release'
IMAGE = 'sha256:3835800bf3d91cd3cb9225e3877428cc565f3e4dd7243bfe911759d828f59fb5'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label', required=True)
args = parser.parse_args()
assert re.fullmatch(r'final[12]', args.label)
output = ROOT / 'final' / args.label
output.mkdir(parents=True, exist_ok=False)
status = json.loads((ROOT / 'runs/measure-v2/status.json').read_text())
assert status['state'] == 'completed' and status['retained_service']['name'] == NAME
subprocess.run(['python3', str(ROOT / 'control/verify_service_state.py'),
    '--initial-state', str(ROOT / 'control/protected-state.json'),
    '--output', str(output / 'protected-services.json'), '--functional'], check=True)
service = json.loads(subprocess.check_output(['docker', 'inspect', NAME]))[0]
assert service['Image'] == IMAGE and service['State']['Running'] and service['RestartCount'] == 0
runtime = json.loads((ROOT / 'runs/measure-v2/b/runtime-after.json').read_text())
assert service['Id'] == runtime['Id'] and service['State']['StartedAt'] == runtime['State']['StartedAt']
live = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()
assert not any(x.startswith('codex-') for x in live)
with urllib.request.urlopen('http://127.0.0.1:18000/health', timeout=15) as response:
    health = response.status
payload = {'model': 'qwen3.6-27b-fp8', 'temperature': 0, 'max_tokens': 128,
           'chat_template_kwargs': {'enable_thinking': False},
           'response_format': {'type': 'json_object'},
           'messages': [{'role': 'user', 'content': '三组设备分别19、7、24台，已领用13台。只输出JSON对象，字段remaining为整数剩余量，status为字符串"ok"。'}]}
request = urllib.request.Request('http://127.0.0.1:18000/v1/chat/completions',
          data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=90) as response:
    body = json.load(response)
choice = body['choices'][0]
assert health == 200 and choice['finish_reason'] == 'stop'
assert json.loads(choice['message']['content']) == {'remaining': 37, 'status': 'ok'}
record = {'passed': True, 'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'name': NAME, 'id': service['Id'], 'image': service['Image'],
          'started_at': service['State']['StartedAt'], 'restart_count': service['RestartCount'],
          'health': health, 'response': body, 'running_names': live,
          'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw', '--format=csv'], text=True)}
(output / 'candidate.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: record[k] for k in ('passed', 'name', 'observed_at_utc', 'health', 'restart_count')}))
