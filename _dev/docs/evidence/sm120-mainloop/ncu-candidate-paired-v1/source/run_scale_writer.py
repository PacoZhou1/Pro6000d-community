"""Build/run the actual epilogue destination geometry probe without a GPU."""
import hashlib
import json
import subprocess
from pathlib import Path

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

source = Path('/work/sm120-mainloop/scale_writer_probe.cu')
out = Path('/results')
inc = Path('/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include')
results = []
for variant, rev in [('control42', 'v2'), ('warp24x2', 'v2'), ('warp24n64', 'v2'), ('warp24x2', 'v3'), ('warp24n64', 'v3')]:
    label = variant + '-' + rev
    root = Path('/work') / ('build-' + label)
    manifest = json.loads((root / 'build-manifest.json').read_text())
    binary = out / (label + '.bin')
    cmd = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O0', '--expt-relaxed-constexpr', '-arch=sm_120a',
           '-DPROBE_NAMESPACE=' + manifest['private_namespace'], '-DPROBE_LABEL="' + label + '"',
           '-DPROBE_WRITER_MAX_WARP=' + ('2' if rev == 'v3' else '4'),
           '-I', str(root / 'include'), '-I', str(root), '-I', str(inc),
           '-I', str(inc.parent / 'tools/util/include'), str(source), '-o', str(binary)]
    r = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / (label + '-compile.log')).write_text(r.stdout)
    row = {'label': label, 'command': cmd, 'compile_exit_code': r.returncode,
           'source_sha256': sha(source), 'manifest_sha256': sha(root / 'build-manifest.json'),
           'generated_source_sha256': manifest['generated_sources']}
    if r.returncode == 0:
        run = subprocess.run([str(binary)], text=True, capture_output=True)
        row.update(run_exit_code=run.returncode, binary_sha256=sha(binary))
        (out / (label + '.json')).write_text(run.stdout)
        (out / (label + '.stderr')).write_text(run.stderr)
        if run.returncode == 0:
            row['result'] = json.loads(run.stdout)
    results.append(row)
(out / 'writer-status.json').write_text(json.dumps({'scope': 'CPU epilogue destination address geometry, not GPU race or edge-tile proof', 'variants': results}, indent=2) + '\n')
print(json.dumps({r['label']: r.get('result', {'compile_exit_code': r['compile_exit_code']}) for r in results}))
raise SystemExit(int(any(r['compile_exit_code'] or r.get('run_exit_code', 1) for r in results)))
