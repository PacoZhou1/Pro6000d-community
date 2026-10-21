"""Parse Nsight Compute's horizontal CSV; preserve units and replay caveats."""
import csv
import json
from pathlib import Path

root = Path(__file__).parent / 'counters'
names = [
    'gpu__time_duration.sum',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'lts__t_sector_hit_rate.pct',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'launch__grid_size', 'launch__registers_per_thread',
    'launch__occupancy_limit_registers', 'launch__occupancy_limit_shared_mem',
    'profiler__replayer_passes',
]
summary = []
for path in sorted(root.glob('*/metrics.csv')):
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2, f'expected units and one kernel: {path}'
    units, row = rows
    status = json.loads((path.parent / 'status.json').read_text())
    assert status['exit_code'] == 0
    metrics = {n: {'value': float(row[n].replace(',', '')), 'unit': units[n]}
               for n in names if n in row and row[n]}
    anomalies = {n: float(v.replace(',', '')) for n, v in row.items()
                 if 'pct_of_peak_sustained_active' in n and v
                 and float(v.replace(',', '')) > 100}
    summary.append({
        'name': path.parent.name, 'metrics': metrics,
        'embedding_unchanged': status['embedding_before'] == status['embedding_after'],
        'kernel': row['Kernel Name'],
        'cache_control': status['command'][status['command'].index('--cache-control') + 1],
        'active_cycle_metrics_over_100': anomalies,
        'scope': 'synthetic operands matching measured model M N K layout; isolated NCU replay; not full model counters or latency',
        'limit': ('small-grid multi-pass active-cycle metrics are unstable; do not use them to infer physical utilization'
                  if anomalies else 'pipeline active ratio is not achieved FLOPS'),
    })
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({'counter_cases': len(summary),
                  'embedding_unchanged': all(x['embedding_unchanged'] for x in summary)}))
