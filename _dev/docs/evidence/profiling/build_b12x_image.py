"""Reproduce the isolated experimental image; never alter the base image.

Run on GPU host. No GPU is requested. The B12X version is outside vLLM 0.31's
official extra pin; see backend-source-notes.md for compatibility boundaries.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

BASE = 'sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
WHEEL_SHA = '4b6635e93ce672db1415cca1817c81eab461a245fb96a5e2de7efa2e36a36114'
p = argparse.ArgumentParser()
p.add_argument('--output-dir', required=True)
p.add_argument('--name', default='codex-kernel-build-b12x')
p.add_argument('--tag', default='qwen-profile-b12x:1.5.0')
a = p.parse_args()
assert a.name.startswith('codex-kernel-build-')
root = pathlib.Path(a.output_dir).resolve()
root.mkdir(parents=True, exist_ok=True)
code = '''import subprocess,glob,importlib.metadata as m,json,pathlib,hashlib
assert m.version('nvidia-cutlass-dsl') == '4.7.1'
subprocess.run(['python3','-m','pip','download','--no-deps','--dest','/artifacts','b12x==1.5.0'],check=True)
w=glob.glob('/artifacts/b12x-1.5.0-*.whl');assert len(w)==1
assert hashlib.sha256(pathlib.Path(w[0]).read_bytes()).hexdigest() == WHEEL_SHA
subprocess.run(['python3','-m','pip','install','--no-deps',w[0]],check=True)
d={p:{'version':m.version(p),'requires':m.requires(p)} for p in ['b12x','nvidia-cutlass-dsl','flashinfer-python','torch','vllm']}
pathlib.Path('/artifacts/packages.json').write_text(json.dumps(d,indent=2))
'''.replace('WHEEL_SHA', repr(WHEEL_SHA))
r = subprocess.run(['docker', 'run', '--name', a.name, '--mount',
    f'type=bind,src={root},dst=/artifacts', '--entrypoint', 'python3', BASE,
    '-c', code], capture_output=True, text=True)
(root / 'build.log').write_text(r.stdout + r.stderr)
r.check_returncode()
image = subprocess.check_output(['docker', 'commit', a.name, a.tag], text=True).strip()
record = {'base_image': BASE, 'image_id': image, 'b12x_version': '1.5.0',
          'pip_no_deps': True, 'wheel_sha256': {
              p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.whl')},
          'scope': 'experimental integration; fixed vLLM0.31.0 official extra pins b12x1.3.0 instead'}
(root / 'image.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
