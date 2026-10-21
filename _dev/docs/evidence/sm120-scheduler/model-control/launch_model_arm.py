"""Freeze experiment instrumentation and launch one owned model arm."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--name', required=True)
p.add_argument('--label', required=True)
p.add_argument('--mode', choices=['baseline', 'candidate'], required=True)
p.add_argument('--library-source')
p.add_argument('--library-sha256')
p.add_argument('--witness', action='store_true')
a = p.parse_args()
root = pathlib.Path(a.root)
instrument = root / 'instruments' / a.name
output = root / a.label
if not a.name.startswith('codex-fp8-epi-') or instrument.exists() or output.exists():
    raise RuntimeError('fresh owned experiment name and evidence directory required')
shutil.copytree(root / 'runtime', instrument)
if a.mode == 'candidate':
    if not a.library_source or not a.library_sha256:
        raise RuntimeError('candidate requires validated library path and SHA')
    library = pathlib.Path(a.library_source)
    if hashlib.sha256(library.read_bytes()).hexdigest() != a.library_sha256:
        raise RuntimeError('candidate library hash mismatch')
    shutil.copy2(library, instrument / 'libgateup.so')
output.mkdir(parents=True, exist_ok=False)
manifest = {str(f.relative_to(instrument)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in instrument.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
(output / 'instrument-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
command = ['python3', str(root / 'work/gemm-epilogue/experiment_control.py'),
           '--name', a.name, '--profiler', 'cuda' if a.witness else 'none',
           '--output-dir', str(output), '--instrument-dir', str(instrument),
           '--extra-arg=--kv-cache-memory-bytes', '--extra-arg=12884901888']
env = [f'PRO6000D_OFFICIAL_MODE={a.mode}',
       'PRO6000D_OFFICIAL_EVIDENCE_DIR=/profiles',
       'PRO6000D_OFFICIAL_V2_SITECUSTOMIZE=/instrument/v2/sitecustomize.py',
       'PRO6000D_FP8_FUSION_ENABLE=1',
       'PRO6000D_FP8_FUSION_LIBRARY=/instrument/v2/libquant.so',
       'PRO6000D_FP8_FUSION_LIBRARY_SHA256=087b7caa92c4742ece3875b6dd1421e25360b834ce4106f7fa166f90bc933b3f']
if a.mode == 'candidate':
    env += ['PRO6000D_OFFICIAL_LIBRARY=/instrument/libgateup.so',
            'PRO6000D_OFFICIAL_LIBRARY_SHA256=' + a.library_sha256,
            'PRO6000D_OFFICIAL_VALIDATED_LIBRARY_SHA256=' + a.library_sha256]
if a.witness:
    env += ['PRO6000D_OFFICIAL_DISPATCH_EVIDENCE=1']
    command += ['--nsys-root', '/home/llm/.local/share/qwen-profiling/nsys-2026.5.1',
                '--nsys-relative', 'opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys']
for value in env:
    command += ['--extra-env', value]
(output / 'launch-command.json').write_text(json.dumps(command, indent=2) + '\n')
subprocess.run(command, check=True)
