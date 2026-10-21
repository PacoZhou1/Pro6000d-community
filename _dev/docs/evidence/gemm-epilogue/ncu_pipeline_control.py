"""Run one owned synthetic pipeline capture; no service lifecycle operations."""
import argparse
import json
import pathlib
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--build-dir', required=True)
p.add_argument('--library-sha256', required=True)
p.add_argument('--label', required=True)
p.add_argument('--m', type=int, default=16160)
a = p.parse_args()
root = pathlib.Path(a.root)
output = root / a.label
output.mkdir(parents=True, exist_ok=True)
image = 'sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
names = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()
for name in names:
    if name.startswith(('codex-fp8-', 'codex-gemm-')):
        obj = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
        host = obj['HostConfig']
        if host.get('Runtime') != 'runc' or host.get('DeviceRequests'):
            raise RuntimeError('another GPU experiment is running')
metrics = ['dram__bytes_op_read.sum', 'dram__bytes_op_write.sum',
           'lts__t_bytes.sum', 'gpu__time_duration.sum',
           'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
           'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
           'sm__warps_active.avg.pct_of_peak_sustained_active',
           'sm__throughput.avg.pct_of_peak_sustained_elapsed']
command = ['docker', 'run', '--rm', '--name', 'codex-fp8-epi-ncu-pipeline',
           '--gpus', 'all', '--cap-add', 'SYS_ADMIN', '--ipc', 'host', '--network', 'none',
           '--memory=24g', '--memory-swap=24g', '--cpus=3']
for source, dest, readonly in [
    (root / 'work', '/work', False), (output, '/results', False),
    ('/home/llm/.local/share/qwen-gemm-targeted/build/compiled', '/baseline', True),
    ('/home/llm/.local/share/qwen-fp8-fusion', '/v2', True),
    ('/home/llm/.local/share/qwen-gemm-targeted/ncu-2026.3.0', '/ncu', True),
]:
    command += ['--mount', f'type=bind,src={source},dst={dest}' + (',readonly' if readonly else '')]
command += ['--entrypoint', '/ncu/opt/nvidia/nsight-compute/2026.3.0/ncu', image,
            '--clock-control', 'none', '--cache-control', 'none',
            '--profile-from-start', 'off', '--nvtx',
            '--nvtx-include', 'regex:GEMM_EPILOGUE_PIPELINE:.*/',
            '--metrics', ','.join(metrics), '--export', '/results/pipeline',
            '--force-overwrite', 'python3', '/work/gemm-epilogue/pipeline_probe.py',
            '--baseline-library', '/baseline/libtargeted_fp8.so',
            '--quant-library', '/v2/libfp8_silu_quant_group_major.so',
            '--candidate-adapter', '/work/gemm-epilogue/candidate_adapter.py',
            '--candidate-library', '/work/' + a.build_dir + '/libgateup_quant_fp8.so',
            '--candidate-sha256', a.library_sha256, '--m', str(a.m), '--ncu']
started = time.time()
with (output / 'stdout.log').open('w') as out, (output / 'stderr.log').open('w') as err:
    result = subprocess.run(command, stdout=out, stderr=err)
(output / 'status.json').write_text(json.dumps({
    'command': command, 'exit_code': result.returncode, 'started_unix': started,
    'elapsed_s': time.time() - started,
    'scope': 'one complete synthetic pipeline per arm, kernel replay, cache/clock control none',
}, indent=2) + '\n')
raise SystemExit(result.returncode)
