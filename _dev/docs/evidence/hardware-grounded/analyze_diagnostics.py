"""Normalize saved NCU wide CSV exports; preserve units and exact source hashes."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BYTE_FACTORS = {'byte': 1, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9}
TIME_FACTORS = {'ns': 1, 'nsecond': 1, 'us': 1e3, 'usecond': 1e3,
                'ms': 1e6, 'msecond': 1e6, 'second': 1e9}


def parse(path):
    status = json.loads((path.parent/'status.json').read_text())
    assert status['exit_code'] == status['export_exit_code'] == 0
    assert status['owned_container_present_after'] is False
    with path.open() as f:
        rows = list(csv.DictReader(f))
    units, raw = rows[0], rows[1:]
    assert units['ID'] == '' and [r['ID'] for r in raw] == ['0', '1', '2', '3']
    assert 'PairedGateUpFp8' in raw[2]['Kernel Name']
    assert 'silu_quant' in raw[1]['Kernel Name'] and 'pack_gateup_fp8' in raw[3]['Kernel Name']
    metrics = json.loads((path.parent/'metric-selection.json').read_text())['selected']
    results = []
    for row in raw:
        vals = {}
        for name in [k for k in row if k in metrics or k.startswith(('launch__', 'device__attribute_'))]:
            value = row[name]
            if value == '':
                continue
            unit = units.get(name, '')
            try:
                num = float(value.replace(',', ''))
            except ValueError:
                vals[name] = {'value': value, 'unit': unit}
                continue
            factor = BYTE_FACTORS.get(unit, TIME_FACTORS.get(unit, 1))
            vals[name] = {'value': num, 'unit': unit, 'base_value': num*factor,
                          'base_unit': 'byte' if unit in BYTE_FACTORS else 'ns' if unit in TIME_FACTORS else unit}
        results.append({'id': row['ID'], 'kernel': row['Kernel Name'],
                        'block': row['Block Size'], 'grid': row['Grid Size'], 'metrics': vals})
    def val(i,k): return results[i]['metrics'][k]['base_value']
    pipeline = {}
    for name, ids in [('baseline',[0,1]),('v6',[2,3])]:
        pipeline[name] = {
            'duration_ms': sum(val(i,'gpu__time_duration.sum') for i in ids)/1e6,
            'dram_read_bytes': sum(val(i,'dram__bytes_op_read.sum') for i in ids),
            'dram_write_bytes': sum(val(i,'dram__bytes_op_write.sum') for i in ids),
            'l2_request_bytes': sum(val(i,'lts__t_bytes.sum') for i in ids)}
        pipeline[name]['dram_total_bytes'] = pipeline[name]['dram_read_bytes'] + pipeline[name]['dram_write_bytes']
    ratios = {k:pipeline['v6'][k]/pipeline['baseline'][k] for k in pipeline['baseline']}
    gemm_ratios = {k:val(2,k)/val(0,k) for k in ['smsp__inst_executed.sum',
        'smsp__inst_executed_pipe_tensor.sum','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum']}
    return {'source':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'cache_control':status.get('cache_control','none'), 'kernels':results,
            'pipeline':pipeline,'v6_over_baseline':ratios,'gemm_counter_ratios':gemm_ratios,
            'limits':'Synthetic operands at real model dimensions; NCU replay and uncontrolled clocks. Stall percentages are not wall-time fractions. Read counters do not identify individual input/weight buffers.'}

if __name__ == '__main__':
    result = {p.parent.name:parse(p) for p in sorted(ROOT.glob('ncu-m*/raw.csv'))}
    (ROOT/'diagnostics-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    # Keep failed captures visible even when no CSV could be exported.
    inventory = []
    for path in sorted(ROOT.glob('ncu-m*/status.json')):
        status = json.loads(path.read_text())
        usable = path.parent.name in result
        assert usable or status.get('exit_code') != 0 or status.get('export_exit_code') != 0
        inventory.append({'capture': path.parent.name, 'usable_counters': usable,
                          'exit_code': status['exit_code'],
                          'cache_control': status.get('cache_control', 'none'),
                          'status_source': str(path.relative_to(ROOT)),
                          'status_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'owned_container_present_after': status['owned_container_present_after']})
    (ROOT/'capture-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    for name, row in result.items():
        print(name,json.dumps({'pipeline':row['pipeline'],'v6_over_baseline':row['v6_over_baseline'],'gemm_counter_ratios':row['gemm_counter_ratios']}))
