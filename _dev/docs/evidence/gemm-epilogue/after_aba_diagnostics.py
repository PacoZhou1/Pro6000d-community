"""Wait for the owned A/B/A to release the GPU, then run numerical diagnostics.

The identity experiment emits no timings. Its GPU work may overlap the CPU-only
v6 compilation; both start strictly after the unprofiled model sequence ends.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root
deadline = time.monotonic() + 1200
while True:
    rows = json.loads((root / 'aba-v4/sequence.json').read_text())
    if rows[-1].get('phase') == 'completed':
        break
    if time.monotonic() > deadline:
        raise RuntimeError('A/B/A did not complete; no diagnostic started')
    time.sleep(10)
names = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()
assert not any(n.startswith(('codex-fp8-', 'codex-gemm-')) for n in names), names
image = 'sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
common = ['docker', 'run', '--rm', '--network', 'none', '--cpus', '2',
          '-v', str(root / 'work') + ':/work']
compile_command = common + ['--runtime=runc', '--memory', '12g', '--entrypoint',
    'python3', image, '/work/gemm-epilogue/build_candidate.py',
    '--source', '/work/fixed_dispatch.cuh', '--cutlass-include',
    '/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include',
    '--build-dir', '/work/build-gateup-coop-v6', '--tile-m', '64',
    '--cooperative-experimental', '--match-official-scale-order',
    '--cooperative-sf128-two-warps']
identity_command = common + ['--name', 'codex-fp8-epi-identity', '--gpus', 'all',
    '--ipc', 'host', '--memory', '16g', '--entrypoint', 'python3', image,
    '/work/gemm-epilogue/identity_visitor_probe.py', '--coop-bf16-library',
    '/work/build-coop-bf16-debug/libn256_m64_epiauto_bf16_probe.so',
    '--coop-identity-library', '/work/build-identity-coop/libgateup_quant_fp8.so',
    '--pingpong-identity-library', '/work/build-identity-pingpong/libgateup_quant_fp8.so',
    '--m', '257', '--swizzle', '8']
status = {'started_unix': time.time(), 'commands': {
    'compile_v6': compile_command, 'identity': identity_command}}
(root / 'post-aba-diagnostics-status.json').write_text(json.dumps(status, indent=2))
with (root / 'candidate-build-coop-v6.log').open('w') as build_log, \
     (root / 'identity-visitor.json').open('w') as out, \
     (root / 'identity-visitor.stderr').open('w') as err:
    build = subprocess.Popen(compile_command, stdout=build_log, stderr=subprocess.STDOUT)
    identity = subprocess.run(identity_command, stdout=out, stderr=err)
    status['identity_exit_code'] = identity.returncode
    status['compile_exit_code'] = build.wait()
status['finished_unix'] = time.time()
(root / 'post-aba-diagnostics-status.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status), flush=True)
if status['identity_exit_code'] or status['compile_exit_code']:
    raise SystemExit(1)
