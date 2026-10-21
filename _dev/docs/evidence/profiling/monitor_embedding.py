"""Read-only embedding availability monitor; no inference load is generated."""
import datetime
import json
import pathlib
import subprocess
import time
import urllib.request

stop = pathlib.Path('/tmp/qwen-profile-experiment/monitor.stop')
while not stop.exists():
    row = {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        d = json.loads(subprocess.check_output(['docker', 'inspect', 'vllm-embedding']))[0]
        cmd = d['Config']['Cmd']
        key = cmd[cmd.index('--api-key') + 1]
        row.update(id=d['Id'], started_at=d['State']['StartedAt'],
                   restart_count=d['RestartCount'], running=d['State']['Running'])
        with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8002/health',
                headers={'Authorization': 'Bearer ' + key}), timeout=10) as r:
            row['health'] = r.status
    except Exception as exc:
        row['error_type'] = type(exc).__name__
    print(json.dumps(row), flush=True)
    time.sleep(30)
