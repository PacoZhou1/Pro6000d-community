"""Read-only follow-up for live production identities, authentication and APIs."""
import datetime
import http.client
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import urllib.error
import urllib.request

ROOT=Path('/home/llm/.local/share/qwen-fp8-release-20261015')
OUT=ROOT/'production/20261015-r1'
spec=importlib.util.spec_from_file_location('activation',ROOT/'activate_production.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
assert json.loads((OUT/'status.json').read_text())['state']=='completed'
original=json.loads((OUT/'functional.json').read_text())
results={}
for name,port in helpers.PORTS.items():
    obj=helpers.inspect(name);secret=helpers.key(obj)
    assert obj['Id']==original[name]['id'] and obj['State']['StartedAt']==original[name]['started_at']
    assert obj['RestartCount']==original[name]['restart_count'] and obj['State']['Running']
    assert helpers.call(port,secret,'/health')[0]==200
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=15) as response:
            unauth=response.status
    except urllib.error.HTTPError as error:
        unauth=error.code
    assert unauth==401
    models=json.loads(helpers.call(port,secret,'/v1/models')[1])
    results[name]={**helpers.snapshot(obj),'health':200,'unauthenticated_models_status':unauth,
        'served_models':[x['id'] for x in models['data']],'identity_stable':True}
old=helpers.inspect(helpers.BACKUP);new=helpers.inspect(helpers.OLD)
assert helpers.key(old)==helpers.key(new)
assert old['HostConfig']['PortBindings']==new['HostConfig']['PortBindings']
assert new['Image']==helpers.IMAGE and not old['State']['Running']
command=new['Config']['Cmd']
assert command[command.index('--max-model-len')+1]=='262144'
assert command[command.index('--kv-cache-memory-bytes')+1]==str(10*1024**3)
log=subprocess.run(['docker','logs',helpers.OLD],capture_output=True,text=True)
text=log.stdout+log.stderr
markers=('draft-vocab enabled: 65536','FP8 fusion candidate launched',
         'P3 gdn core+norm+quant fused path used','P3 conv_post_fused kernel launched','Graph capturing finished')
observed={marker:marker in text for marker in markers}
record={'passed':True,'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'services':results,'new_27b_api_key_matches_backup':True,'new_27b_ports_match_backup':True,
        'old_backup_stopped':True,'runtime_markers':observed,
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu,power.draw','--format=csv'],text=True)}
assert not (OUT/'followup.json').exists()
(OUT/'followup.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'passed':True,'services':{name:{'port':helpers.PORTS[name],'health':x['health'],
                  'restart_count':x['restart_count']} for name,x in results.items()}}))
