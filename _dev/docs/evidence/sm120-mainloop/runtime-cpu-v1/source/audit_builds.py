"""CPU-only library metadata, ELF isolation and compiler-resource audit."""
import ctypes
import hashlib
import json
import re
import subprocess
from pathlib import Path

out = Path('/results')
rows = []
for label in ('control42-v2', 'warp24x2-v2', 'warp24n64-v2', 'warp24x2-v3', 'warp24n64-v3'):
    root = Path('/work') / ('build-' + label)
    manifest = json.loads((root / 'build-manifest.json').read_text())
    library = Path(manifest['library'])
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    assert digest == manifest['library_sha256']
    handle = ctypes.CDLL(str(library))
    handle.sm120_scheduler_metadata.restype = ctypes.c_char_p
    metadata = json.loads(handle.sm120_scheduler_metadata())
    symbols = subprocess.check_output(['readelf', '-Ws', str(library)], text=True)
    (out / (label + '-elf.txt')).write_text(symbols)
    globals_ = [line for line in symbols.splitlines()
                if re.search(r'\b(?:GLOBAL|UNIQUE)\s+DEFAULT\b', line)
                and ' UND ' not in line]
    assert globals_ and all('sm120_' in line and ' FUNC ' in line for line in globals_), globals_
    resources = []
    current = None
    for line in (root / 'compile.log').read_text().splitlines():
        match = re.search(r"Compiling entry function '([^']+)'", line)
        if match:
            current = {'symbol': match[1], 'compiler_resources': []}
            current['kind'] = ('pack' if 'mainloop_pack_' in match[1] else
                'fused_fp8' if 'MainloopPairedGateUpFp8' in match[1] else
                'paired_bf16' if 'BlockwiseCooperativeSm120' in match[1] else 'official_bf16')
            resources.append(current)
        if current is not None and ('Used ' in line or 'spill stores' in line):
            current['compiler_resources'].append(line.strip())
    assert {r['kind'] for r in resources} == {'pack', 'fused_fp8', 'paired_bf16', 'official_bf16'}
    rows.append({'build': label, 'sha256': digest, 'metadata': metadata,
                 'resources': resources, 'only_abi_functions_global': True})
(out / 'audit.json').write_text(json.dumps({'scope': 'CPU metadata and compiler resources; not GPU correctness or occupancy', 'libraries': rows}, indent=2) + '\n')
print(json.dumps({'pass': True, 'builds': len(rows)}))
