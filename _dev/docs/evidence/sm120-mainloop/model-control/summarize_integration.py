"""Summarize frozen model compile, weight identity and dispatch evidence.

Python records describe dispatch/capture, never graph replay. Actual CUDA
launch correlation is independently established by analyze_dispatch.py.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
sources = {}

def read(path):
    sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

dispatch = [read(f) for f in a.root.glob('dispatch-*.json')
            if f.name != 'dispatch-summary.json']
rewrites = [read(f) for f in a.root.glob('fx-rewrite-*.json')]
packs = [read(f) for f in a.root.glob('startup-pack-*.json')
         if not f.name.endswith('-summary.json')]
modules = [m for r in packs for m in r['modules']]
registered = {m['canonical_weight_data_ptr']: m for m in modules}
candidates = [r for r in dispatch if r['dispatch'] == 'mainloop_warp24n64_v3_m16160_r1_s1']
fallback = [r for r in dispatch if r['dispatch'] == 'canonical_v2_fallback']
assert len(modules) == 65 and len(registered) == 65
assert len(candidates) == 65 and len({r['module_name'] for r in candidates}) == 65
assert len(dispatch) == len(candidates) + len(fallback)
assert len({(r['canonical_weight_data_ptr'], r['canonical_scale_data_ptr']) for r in candidates}) == 65
assert sum(r['rewrite_count'] for r in rewrites) > 0
for row in candidates:
    entry = registered[row['canonical_weight_data_ptr']]
    assert row['registered_identity_match']
    assert all(row[key] == entry[key] for key in (
        'canonical_scale_data_ptr', 'packed_weight_data_ptr', 'packed_scale_data_ptr'))
    assert row['module_name'] == entry['name']
    temporary=row['temporary_buffer_bytes']
    assert set(temporary)=={'paired_fp8_temp','native_scale_temp','cutlass_workspace'}
    assert temporary['paired_fp8_temp']==16160*34816
    assert temporary['native_scale_temp']>=16160*136*4
    assert temporary['cutlass_workspace']>0
assert all(r['M'] == 16160 for r in candidates)
assert all(r['M'] != 16160 for r in fallback)
assert {5, 32768}.issubset({r['M'] for r in fallback}), 'small and largest-M runtime guard witness missing'
result = {
    'passed': True,
    'target_M': 16160,
    'candidate_python_pids': sorted({r['pid'] for r in candidates}),
    'kernel_consumes_packed_weights': True,
    'packed_modules': len(modules),
    'additional_packed_bytes': sum(r['total_bytes'] for r in packs),
    'fx_graphs': len(rewrites),
    'fx_rewrites': sum(r['rewrite_count'] for r in rewrites),
    'candidate_modules_identity_checked': len({r['module_name'] for r in candidates}),
    'candidate_records_by_M': dict(sorted(collections.Counter(r['M'] for r in candidates).items())),
    'fallback_records_by_M': dict(sorted(collections.Counter(r['M'] for r in fallback).items())),
    'candidate_temporary_bytes_by_M': {r['M']: r['temporary_buffer_bytes'] for r in candidates},
    'peak_allocated_bytes_observed': max(r['memory_after']['peak_allocated_bytes'] for r in dispatch),
    'peak_reserved_bytes_observed': max(r['memory_after']['peak_reserved_bytes'] for r in dispatch),
    'scope': 'Frozen whole-model startup and first Python dispatch/capture records; not per-request peak or graph replay counts. Kernel launches are separately verified with Nsight.',
    'source_hashes': sources,
}
(a.root / 'integration-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}))
