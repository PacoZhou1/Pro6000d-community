"""Initialize the opt-in runtime before any Torch/vLLM import, inside owned probe."""
import os
import hashlib
import json
import runpy
import sys
from pathlib import Path
root=Path('/work/model-runtime')
sha='a6ae4155f65ec3b992ade3943902ccd88eea76b211dda5ceab26626245b7ae5d'
os.environ.update({
 'PRO6000D_MAINLOOP_MODE':'candidate',
 'PRO6000D_MAINLOOP_EVIDENCE_DIR':'/results',
 'PRO6000D_MAINLOOP_V2_SITECUSTOMIZE':str(root/'v2/sitecustomize.py'),
 'PRO6000D_MAINLOOP_LIBRARY':'/work/build-warp24n64-v3/libsm120_mainloop_warp24n64_v3.so',
 'PRO6000D_MAINLOOP_LIBRARY_SHA256':sha,
 'PRO6000D_MAINLOOP_VALIDATED_LIBRARY_SHA256':sha,
 'PRO6000D_FP8_FUSION_ENABLE':'1',
 'PRO6000D_FP8_FUSION_LIBRARY':str(root/'v2/libquant.so'),
 'PRO6000D_FP8_FUSION_LIBRARY_SHA256':'087b7caa92c4742ece3875b6dd1421e25360b834ce4106f7fa166f90bc933b3f'})
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
Path('/results/runtime-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
sys.path.insert(0,str(root))
runpy.run_path(str(root/'sitecustomize.py'))
sys.argv=['runtime_smoke.py','--output','/results/runtime-smoke.json']
runpy.run_path(str(root/'runtime_smoke.py'),run_name='__main__')
