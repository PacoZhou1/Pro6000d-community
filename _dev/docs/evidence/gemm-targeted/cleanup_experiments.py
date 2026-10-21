"""Remove this run's stopped temporary containers, preserving evidence files."""
import datetime
import json
from pathlib import Path
import subprocess

names = ['codex-gemm-attribution-mtp4', 'codex-gemm-counters-mtp4',
         'codex-gemm-api-baseline', 'codex-gemm-api-streamk',
         'codex-gemm-api-streamk-v2', 'codex-gemm-api-baseline-repeat',
         'codex-gemm-streamk-replay']
image = 'sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
removed = []
for name in names:
    obj = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert not obj['State']['Running'] and obj['Image'] == image
    subprocess.run(['docker', 'rm', name], check=True, stdout=subprocess.DEVNULL)
    removed.append({'name': name, 'id': obj['Id'], 'state_before': obj['State']['Status']})
remaining = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
result = {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'removed': removed,
          'remaining_gemm_experiments': [n for n in remaining if n.startswith('codex-gemm-')],
          'preserved': 'remote model files, local tool installation, compiled library and captured evidence'}
Path('/home/llm/.local/share/qwen-gemm-targeted/cleanup.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
