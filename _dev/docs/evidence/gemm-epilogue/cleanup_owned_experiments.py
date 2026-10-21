"""Remove only already-stopped containers owned by this evidence root.

Never stops a container. Original services, including embedding, are excluded
by immutable IDs as well as the required experiment name and mount boundary.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--names', nargs='+', required=True)
a = p.parse_args()
root = a.root.resolve()
original = json.loads((root / 'initial-state.json').read_text())['containers']
protected_ids = {value['id'] for value in original.values()}
report = {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'policy': 'Only remove already-stopped owned experiment containers',
          'containers': []}
objects = []
for name in a.names:
    assert name.startswith('codex-fp8-epi-') and name not in original
    obj = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert obj['Id'] not in protected_ids and not obj['State']['Running']
    assert obj['Image'] == 'sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
    profiles = next(m['Source'] for m in obj['Mounts'] if m['Destination'] == '/profiles')
    Path(profiles).resolve().relative_to(root)
    objects.append((name, obj, profiles))
# Validate the complete requested set before any deletion.
for name, obj, profiles in objects:
    subprocess.run(['docker', 'rm', name], check=True, stdout=subprocess.DEVNULL)
    report['containers'].append({'name': name, 'id': obj['Id'],
        'image': obj['Image'], 'running_before': False,
        'retained_profiles': profiles, 'removed': True})
    (root / 'owned-cleanup.json').write_text(json.dumps(report, indent=2) + '\n')
report['completed'] = True
(root / 'owned-cleanup.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'completed': True, 'removed': len(objects)}))
