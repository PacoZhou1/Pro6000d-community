"""Read only generated Python in the owned model's compile cache."""
import hashlib
import json
from pathlib import Path
roots = [Path('/root/.cache/vllm'), Path('/tmp/torchinductor_root'), Path('/root/.cache/torch/inductor')]
records = []
for path in sorted({p for root in roots for p in root.rglob('*.py')}):
    data = path.read_bytes()
    if b'pro6000d_fp8.silu_and_mul_per_block_quant' not in data:
        continue
    text = data.decode('utf-8', errors='replace')
    lines = [{'line': i, 'text': line[:800]} for i, line in enumerate(text.splitlines(), 1)
             if 'pro6000d_fp8.silu_and_mul_per_block_quant' in line]
    records.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
                    'custom_op_lines': lines})
print(json.dumps({'files': records, 'file_count': len(records),
                  'scope': 'generated model Python custom-op references; actual execution proved separately by Nsight'}, indent=2))
