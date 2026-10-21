"""Verify the two private candidate kernels in one normal-model Nsight capture.

SQLite kernels establish actual GPU work; runtime JSON separately establishes M.
A graph node is called replay only with a matching successful cudaGraphLaunch.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('sqlite', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
c = sqlite3.connect(a.sqlite)
c.row_factory = sqlite3.Row
assert c.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
names = {r['id']: r['value'] for r in c.execute('SELECT id,value FROM StringIds')}
GEMM = 'MainloopPairedGateUpFp8_warp24n64_v3'
PACK = 'mainloop_pack_gateup_fp8_warp24n64_v3'
NAMESPACE = 'pro6000d_mainloop_warp24n64_v3'
rows = [dict(r) for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL')]
gemm = [r for r in rows if GEMM in names[r['demangledName']]
        and NAMESPACE in names[r['demangledName']]]
pack = [r for r in rows if PACK in names[r['demangledName']]]
assert len(gemm) == len(pack) == 65, f'expected 65 GEMM and 65 pack; got {len(gemm)}, {len(pack)}'
assert all('float_e4m3_t' in names[r['demangledName']] for r in gemm)
assert not any(GEMM in names[r['demangledName']] and NAMESPACE not in names[r['demangledName']] for r in rows)
launches = collections.defaultdict(list)
direct = collections.defaultdict(list)
for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
    name = names[r['nameId']]
    if r['returnValue'] != 0:
        continue
    key = (r['globalTid'] & ~0xFFFFFF, r['correlationId'])
    if 'cudaGraphLaunch' in name:
        launches[key].append(dict(r))
    if 'cudaLaunchKernel' in name:
        direct[key].append(dict(r))

def classify(k):
    key = (k['globalPid'], k['correlationId'])
    graph = bool(k.get('graphNodeId')) and len(launches[key]) == 1
    launched = len(direct[key]) == 1
    return 'graph_replay' if graph else 'direct_launch' if launched else 'unproven'

per_kind = {'fused_gemm': gemm, 'pack': pack}
counts = {kind: dict(collections.Counter(classify(k) for k in group))
          for kind, group in per_kind.items()}
ordered = sorted([(r['start'], 'fused_gemm', r) for r in gemm] +
                 [(r['start'], 'pack', r) for r in pack], key=lambda x: x[0])
# Every target invocation calls the GEMM first, then its pack, on the same CUDA process.
pairs = []
for i in range(0, len(ordered), 2):
    first, second = ordered[i], ordered[i + 1]
    pairs.append({'gemm_start_ns': first[2]['start'], 'pack_start_ns': second[2]['start'],
                  'gemm_pid': first[2]['globalPid'], 'pack_pid': second[2]['globalPid'],
                  'ordered': first[1] == 'fused_gemm' and second[1] == 'pack'
                             and first[2]['globalPid'] == second[2]['globalPid']
                             and first[2]['streamId'] == second[2]['streamId']
                             and first[2]['end'] <= second[2]['start'],
                  'gemm_launch': classify(first[2]), 'pack_launch': classify(second[2])})
result = {
    'sqlite_sha256': hashlib.sha256(a.sqlite.read_bytes()).hexdigest(),
    'sqlite_integrity': 'ok', 'target_M': 16160,
    'target_M_evidence_scope': 'M is established by same-run registered runtime dispatch JSON; SQLite kernel records do not decode kernel arguments.',
    'fused_gemm_count': len(gemm), 'pack_count': len(pack), 'paired_invocations': len(pairs),
    'counts_by_launch_kind': counts,
    'pair_order_and_pid_valid': all(x['ordered'] for x in pairs),
    'kernel_duration_sum_ms': {kind: sum(k['end'] - k['start'] for k in group) / 1e6
                               for kind, group in per_kind.items()},
    'kernels_and_grids': [{'kind': kind, 'name': name, 'grid': list(grid), 'count': count}
                          for kind, group in per_kind.items()
                          for (name, *grid), count in collections.Counter(
                              (names[k['demangledName']], k['gridX'], k['gridY'], k['gridZ'])
                              for k in group).items()],
    'first_pairs': pairs[:8],
    'scope': 'Normal-mode model prefill Nsight witness; report direct launches and graph replay separately. Not unprofiled API timing.',
    'pass': all(x['ordered'] and x['gemm_launch'] == x['pack_launch']
                and x['gemm_launch'] != 'unproven'
                for x in pairs),
}
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('kernels_and_grids', 'first_pairs')}))
raise SystemExit(0 if result['pass'] else 1)
