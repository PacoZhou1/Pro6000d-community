"""Detach an owned model sequence so an SSH interruption cannot abandon cleanup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path('/home/llm/.local/share/qwen-sm120-mainloop')
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode', choices=['witness', 'aba'], required=True)
p.add_argument('--label', required=True)
p.add_argument('--child', action='store_true')
a = p.parse_args()
if not re.fullmatch('[a-z0-9][a-z0-9-]{0,40}', a.label):
    p.error('fresh simple label required')
status_path = root / (a.label + '-controller.json')
script = root / 'work/model-control/model_sequence.py'
command = [sys.executable, str(script), '--mode', a.mode, '--label', a.label]
if not a.child:
    if status_path.exists() or (root / a.label).exists():
        raise RuntimeError('refuse to reuse model sequence label')
    status_path.write_text(json.dumps({'state': 'launching', 'command': command}) + '\n')
    with (root / (a.label + '-sequence.log')).open('w') as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--mode', a.mode,
                                 '--label', a.label, '--child'], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'pid': proc.pid, 'status': str(status_path)}))
else:
    status = {'state': 'running', 'pid': os.getpid(), 'command': command, 'started_unix': time.time(),
              'sequence_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
              'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    try:
        status['exit_code'] = subprocess.run(command).returncode
        status['state'] = 'completed' if status['exit_code'] == 0 else 'failed'
    except Exception as exc:
        status.update(state='failed', error_type=type(exc).__name__)
        raise
    finally:
        status['elapsed_s'] = time.time() - status['started_unix']
        status_path.write_text(json.dumps(status, indent=2) + '\n')
