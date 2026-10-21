"""Profile fixed existing libraries in an owned container; never manage model services."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import time

IMAGE = 'sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
ROOT = Path('/home/llm/.local/share/qwen-sm120-hardware')
OLD = Path('/home/llm/.local/share/qwen-gemm-epilogue')
NCU = '/ncu/opt/nvidia/nsight-compute/2026.3.0/ncu'
METRICS = [
    'gpu__time_duration.sum', 'sm__cycles_elapsed.avg.per_second',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'dram__bytes_op_read.sum', 'dram__bytes_op_write.sum',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'lts__t_bytes.sum', 'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'smsp__warps_eligible.avg.per_cycle_active', 'smsp__inst_executed.sum',
    *['smsp__inst_executed_pipe_' + p + '.sum' for p in ['tensor', 'fma', 'alu', 'lsu', 'xu']],
    *['smsp__warp_issue_stalled_' + p + '_per_warp_active.pct' for p in [
        'long_scoreboard', 'short_scoreboard', 'mio_throttle', 'math_pipe_throttle',
        'barrier', 'wait', 'not_selected', 'lg_throttle']],
    *['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_' + p + '.sum' for p in ['ld', 'st']],
    *['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_' + p + '.sum' for p in ['ld', 'st']],
    *['l1tex__t_sectors_pipe_lsu_mem_local_op_' + p + '.sum' for p in ['ld', 'st']],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, choices=[2741, 16160], required=True)
    parser.add_argument('--label', default=None)
    parser.add_argument('--cache-control', choices=['none', 'all'], default='none')
    args = parser.parse_args()
    label = args.label or ('ncu-m' + str(args.m))
    if not re.fullmatch(r'ncu-m(?:2741|16160)(?:-[a-z0-9-]+)?', label):
        parser.error('invalid owned diagnostic label')
    output = ROOT / label
    output.mkdir(parents=True, exist_ok=False)
    name = 'codex-fp8-hardware-' + label
    # Existing model services must be in the user-requested state. Do not change them.
    subprocess.run(['python3', str(OLD / 'work/gemm-epilogue/verify_service_state.py'),
                    '--initial-state', str(OLD / 'initial-state.json'),
                    '--output', str(output / 'services-before.json')], check=True)
    running = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()
    if any(n.startswith(('codex-fp8-', 'codex-gemm-')) for n in running):
        raise RuntimeError('another experiment is running')
    query = OLD / 'ncu-metric-query-cap.txt'
    wanted = set(METRICS)
    found = set()
    for line in query.open():
        fields = line.split()
        if fields and fields[0] in wanted:
            found.add(fields[0])
    metrics = [m for m in METRICS if m in found]
    if len(metrics) < 24:
        raise RuntimeError('insufficient supported diagnostic metrics')
    (output / 'metric-selection.json').write_text(json.dumps({
        'selected': metrics, 'unavailable': sorted(wanted - found),
        'query_sha256': hashlib.sha256(query.read_bytes()).hexdigest(),
        'interpretation': 'Stall percentages describe active warps, not additive fractions of API time.'}, indent=2)+'\n')
    docker = ['docker', 'run', '--rm', '--name', name, '--gpus', 'all', '--cap-add', 'SYS_ADMIN',
              '--ipc', 'host', '--network', 'none', '--memory=24g', '--memory-swap=24g', '--cpus=3']
    for source, destination, ro in [
        (OLD / 'work', '/work', True), (output, '/results', False),
        (Path('/home/llm/.local/share/qwen-gemm-targeted/build/compiled'), '/baseline', True),
        (Path('/home/llm/.local/share/qwen-fp8-fusion'), '/v2', True),
        (Path('/home/llm/.local/share/qwen-gemm-targeted/ncu-2026.3.0'), '/ncu', True),
    ]:
        docker += ['--mount', f'type=bind,src={source},dst={destination}' + (',readonly' if ro else '')]
    command = docker + ['--entrypoint', NCU, IMAGE, '--clock-control', 'none', '--cache-control', args.cache_control,
        '--profile-from-start', 'off', '--nvtx', '--nvtx-include', 'regex:GEMM_EPILOGUE_PIPELINE:.*/',
        '--section', 'LaunchStats', '--metrics', ','.join(metrics), '--export', '/results/pipeline',
        'python3', '/work/gemm-epilogue/pipeline_probe.py',
        '--baseline-library', '/baseline/libtargeted_fp8.so',
        '--quant-library', '/v2/libfp8_silu_quant_group_major.so',
        '--candidate-adapter', '/work/gemm-epilogue/candidate_adapter.py',
        '--candidate-library', '/work/build-gateup-coop-v6/libgateup_quant_fp8.so',
        '--candidate-sha256', 'e4039ddc95fa3b56f02743357b7a7193156dc500c5d8804bb77ebd3af85450d6',
        '--m', str(args.m), '--ncu']
    started = time.time()
    status = {'command': command, 'started_unix': started, 'scope': 'Synthetic actual-shape full pipeline, existing validated binaries; NCU kernel replay; clocks uncontrolled.', 'cache_control': args.cache_control, 'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        with (output / 'stdout.log').open('w') as out, (output / 'stderr.log').open('w') as err:
            result = subprocess.run(command, stdout=out, stderr=err, timeout=240)
        status['exit_code'] = result.returncode
        reports = list(output.glob('pipeline.ncu-rep*'))
        if result.returncode == 0 and len(reports) == 1:
            export = ['docker', 'run', '--rm', '--network', 'none', '--runtime', 'runc',
                      '--mount', f'type=bind,src={output},dst=/results,readonly',
                      '--mount', 'type=bind,src=/home/llm/.local/share/qwen-gemm-targeted/ncu-2026.3.0,dst=/ncu,readonly',
                      '--entrypoint', NCU, IMAGE, '--import', '/results/' + reports[0].name,
                      '--page', 'raw', '--csv']
            with (output / 'raw.csv').open('w') as out, (output / 'export.stderr').open('w') as err:
                imported = subprocess.run(export, stdout=out, stderr=err, timeout=60)
            status['export_exit_code'] = imported.returncode
    except subprocess.TimeoutExpired:
        status['timeout'] = True
        # Exact unique owned diagnostic name only; never any model service.
        subprocess.run(['docker', 'stop', '--time', '5', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        status['elapsed_s'] = time.time() - started
        status['owned_container_present_after'] = subprocess.run(['docker', 'inspect', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        (output / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
        subprocess.run(['python3', str(OLD / 'work/gemm-epilogue/verify_service_state.py'),
                        '--initial-state', str(OLD / 'initial-state.json'),
                        '--output', str(output / 'services-after.json')], check=True)
    print(json.dumps({'m': args.m, 'exit_code': status.get('exit_code'), 'elapsed_s': status['elapsed_s'], 'container_present': status['owned_container_present_after']}))
    if status.get('exit_code') != 0 or status.get('export_exit_code') != 0 or status['owned_container_present_after']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
