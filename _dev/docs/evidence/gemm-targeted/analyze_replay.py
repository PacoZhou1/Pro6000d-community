"""Prove the candidate kernel executes in real CUDA graph replay, not just capture."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3

p = argparse.ArgumentParser()
p.add_argument('sqlite', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
c = sqlite3.connect(a.sqlite)
c.row_factory = sqlite3.Row
assert c.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
names = {r['id']: r['value'] for r in c.execute('SELECT id,value FROM StringIds')}
kernels = [dict(r) for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL')
           if 'StreamKScheduler' in names[r['demangledName']]]
assert kernels, 'no Stream-K kernel in capture: do not claim the candidate was exercised'
launches = collections.defaultdict(list)
for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
    if 'cudaGraphLaunch' in names[r['nameId']] and r['returnValue'] == 0:
        launches[(r['globalTid'] & ~0xFFFFFF, r['correlationId'])].append(dict(r))
hist = collections.Counter()
matched = 0
samples = []
for k in kernels:
    key = (k['globalPid'], k['correlationId'])
    exact = len(launches[key]) == 1
    graph_node = bool(k.get('graphNodeId'))
    if exact and graph_node:
        matched += 1
    hist[(names[k['demangledName']], k['gridX'], k['gridY'], k['gridZ'])] += 1
    if len(samples) < 8:
        samples.append({key: k[key] for key in ['start', 'end', 'globalPid', 'correlationId',
                                               'graphNodeId', 'graphId', 'gridX', 'gridY', 'gridZ']}
                       | {'exact_successful_graph_launch': exact})
result = {
    'sqlite_sha256': hashlib.sha256(a.sqlite.read_bytes()).hexdigest(),
    'sqlite_integrity': 'ok', 'streamk_kernel_count': len(kernels),
    'kernel_duration_sum_ms': sum(k['end'] - k['start'] for k in kernels) / 1e6,
    'with_graph_node_id': sum(bool(k.get('graphNodeId')) for k in kernels),
    'exact_successful_graph_launch_matches': matched,
    'distinct_graph_launches': len({(k['globalPid'], k['correlationId']) for k in kernels}),
    'kernels_and_grids': [{'name': key[0], 'grid': list(key[1:]), 'count': count}
                          for key, count in hist.items()],
    'samples': samples,
    'scope': 'normal-mode model decode diagnostic with Nsight Systems; proof of replayed candidate, not unprofiled API timing',
    'pass': matched == len(kernels),
}
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ['kernels_and_grids', 'samples']}))
raise SystemExit(0 if result['pass'] else 1)
