"""Actual fixed-image CPU FX/Inductor checks, without a GPU or model load."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path('/work/model-runtime')
out = Path('/results')
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(out / 'runtime-source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
results = []
for name in ('serialization_selfcheck', 'inductor_cpu_selfcheck'):
    with (out / (name + '.stdout')).open('w') as stdout, (out / (name + '.stderr')).open('w') as stderr:
        result = subprocess.run([sys.executable, str(root / (name + '.py'))], stdout=stdout, stderr=stderr)
    results.append({'name': name, 'exit_code': result.returncode})
(out / 'cpu-checks.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results))
raise SystemExit(int(any(r['exit_code'] for r in results)))
