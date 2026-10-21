"""Summarize recorded synthetic timings without claiming model-level speedup."""
import json
from pathlib import Path

root = Path(__file__).parent
out = {'scope': 'synthetic same-operand CUDA events; not model throughput', 'candidates': {}}
for name in ['micro-1', 'micro-splitn', 'micro-streamk']:
    rows = []
    for line in (root / name / 'results.jsonl').read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # Keep original framework logs in the source file.
    status = json.loads((root / name / 'status.json').read_text())
    assert status['exit_code'] == 0
    checks = [r['numerics'] for r in rows if 'numerics' in r]
    assert checks and all(r['pass'] for r in checks)
    timings = [r['timing'] for r in rows if 'timing' in r]
    compact = [{k: v for k, v in t.items() if k not in ['values_us', 'min_us', 'max_us']}
               for t in timings]
    out['candidates'][name] = {
        'numeric_cases_passed': len(checks),
        'embedding_unchanged': status['embedding_before'] == status['embedding_after'],
        'timings': compact,
    }
(root / 'micro-summary.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({name: len(data['timings']) for name, data in out['candidates'].items()}))
