"""Authorized production replacement; keys stay in memory on the GPU host."""
import copy
import datetime
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path('/home/llm/.local/share/qwen-fp8-release-20261015')
OUT = ROOT / 'production/20261015-r1'
IMAGE = 'sha256:3835800bf3d91cd3cb9225e3877428cc565f3e4dd7243bfe911759d828f59fb5'
OLD = 'vllm-qwen-27b'
BACKUP = 'vllm-qwen-27b-before-fp8-r1-20261015'
TEST = 'pro6000d-qwen36-fp8-release'
NINE = 'vllm-qwen3.5-9b'
ASR = 'vllm-qwen3-asr'
EMBED = 'vllm-embedding'
PORTS = {OLD:8000, NINE:8001, EMBED:8002, ASR:8003}


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]


def key(obj):
    c = obj['Config']['Cmd']
    return c[c.index('--api-key') + 1]


def call(port, secret, route, payload=None, timeout=20):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{route}',
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization':'Bearer '+secret, 'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read()


def run_docker(*args):
    subprocess.run(['docker', *args], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=120)


def snapshot(obj):
    cmd = obj['Config']['Cmd'][:]
    if '--api-key' in cmd:
        cmd[cmd.index('--api-key') + 1] = '<REDACTED>'
    return {'id':obj['Id'], 'image':obj['Image'], 'running':obj['State']['Running'],
            'started_at':obj['State']['StartedAt'], 'restart_count':obj['RestartCount'],
            'ports':obj['HostConfig']['PortBindings'], 'restart_policy':obj['HostConfig']['RestartPolicy'],
            'cmd_redacted':cmd}


def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2)+'\n')


class DockerSocket(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(90)
        self.sock.connect('/var/run/docker.sock')


def api(method, route, data=None):
    conn = DockerSocket('localhost')
    try:
        conn.request(method, route, body=None if data is None else json.dumps(data),
                     headers={'Content-Type':'application/json'})
        result = conn.getresponse(); body = result.read()
        if result.status not in (200, 201, 204):
            raise RuntimeError('Docker API HTTP '+str(result.status))
        return json.loads(body) if body else None
    finally:
        conn.close()


def wait(name, port, secret):
    end = time.monotonic()+1200
    while time.monotonic()<end:
        obj=inspect(name)
        if obj['State']['Running']:
            try:
                if call(port, secret, '/health')[0]==200:
                    return
            except Exception:
                pass
        if obj['State']['OOMKilled'] or obj['RestartCount']>2:
            raise RuntimeError(name+' startup exited or repeatedly restarted')
        time.sleep(8)
    raise RuntimeError(name+' startup deadline')


def main():
    baseline={n:inspect(n) for n in (*PORTS,TEST)}
    secrets=[key(baseline[n]) for n in PORTS]
    embedding_identity=snapshot(baseline[EMBED])
    state={'state':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'image':IMAGE,'kv_cache_bytes':10*1024**3,'events':[]}
    save('before.json',{n:snapshot(x) for n,x in baseline.items()})
    stopped=threading.Event()

    def embedding_check():
        obj=inspect(EMBED)
        assert obj['Id']==embedding_identity['id'] and obj['State']['StartedAt']==embedding_identity['started_at']
        assert obj['RestartCount']==embedding_identity['restart_count'] and obj['State']['Running']
        assert call(8002,key(obj),'/health')[0]==200

    def monitor():
        with (OUT/'embedding-monitor.jsonl').open('w') as log:
            while not stopped.is_set():
                row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
                try:embedding_check();row['passed']=True
                except Exception as exc:row.update(passed=False,error_type=type(exc).__name__)
                log.write(json.dumps(row)+'\n');log.flush();stopped.wait(15)

    def phase(name):
        embedding_check()
        state['phase']=name
        state['events'].append({'phase':name,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        save('status.json',state)

    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    renamed=False;created=None
    try:
        phase('prepare')
        assert baseline[OLD]['Image']!='' and not baseline[OLD]['State']['Running']
        existing=subprocess.run(['docker','inspect',BACKUP],capture_output=True)
        assert existing.returncode!=0, 'backup name already exists'
        assert baseline[TEST]['Image']==IMAGE
        image=json.loads(subprocess.check_output(['docker','image','inspect',IMAGE]))[0]
        args=baseline[OLD]['Config']['Cmd'][:]
        def change(flag,value):
            if flag in args:args[args.index(flag)+1]=str(value)
            else:args.extend([flag,str(value)])
        for flag,value in [('--gpu-memory-utilization','.45'),('--kv-cache-memory-bytes',10*1024**3),
             ('--max-num-batched-tokens',32768),('--attention-backend','FLASHINFER'),
             ('--speculative-config',json.dumps({'method':'mtp','num_speculative_tokens':4}))]:change(flag,value)
        host=copy.deepcopy(baseline[OLD]['HostConfig'])
        env=image['Config']['Env'][:]+['HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1']
        config={'Image':IMAGE,'Cmd':args,'Entrypoint':image['Config']['Entrypoint'],
                'WorkingDir':image['Config']['WorkingDir'],'Env':env,
                'ExposedPorts':baseline[OLD]['Config']['ExposedPorts'],'HostConfig':host,
                'Labels':image['Config'].get('Labels',{})}
        if baseline[OLD]['Config'].get('Healthcheck'):
            config['Healthcheck']=baseline[OLD]['Config']['Healthcheck']
        phase('stop_test_and_replace_name')
        if baseline[TEST]['State']['Running']:run_docker('stop','--timeout','60',baseline[TEST]['Id'])
        run_docker('rename',OLD,BACKUP);renamed=True
        version=api('GET','/version')['ApiVersion']
        created=api('POST',f'/v{version}/containers/create?name='+urllib.parse.quote(OLD),config)['Id']
        current=inspect(created)
        assert key(current)==key(baseline[OLD])
        assert current['HostConfig']['PortBindings']==baseline[OLD]['HostConfig']['PortBindings']
        assert current['HostConfig']['RestartPolicy']==baseline[OLD]['HostConfig']['RestartPolicy']
        phase('starting_new_27b')
        run_docker('start',created);wait(OLD,8000,key(baseline[OLD]))
        for name in (NINE,ASR):
            phase('starting_'+name)
            if not inspect(name)['State']['Running']:run_docker('start',name)
            wait(name,PORTS[name],key(baseline[name]))
        phase('functional_checks')
        spec=importlib.util.spec_from_file_location('restore_helpers','/tmp/qwen-spec-experiment/restore_verify.py')
        helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
        results={}
        for name,port in PORTS.items():
            obj=inspect(name);cmd=obj['Config']['Cmd'];model=cmd[cmd.index('--served-model-name')+1]
            secret=key(obj);health=call(port,secret,'/health')[0]
            models=json.loads(call(port,secret,'/v1/models')[1])
            result=snapshot(obj)
            result.update(health=health,model_id_matches=model in [m['id'] for m in models['data']],
                          api_key_unchanged=secret==key(baseline[name]),ports_unchanged=obj['HostConfig']['PortBindings']==baseline[name]['HostConfig']['PortBindings'])
            result['functional']=(helpers.chat(port,secret,model) if port in (8000,8001) else
                  helpers.embedding(port,secret,model) if port==8002 else helpers.asr(port,secret,model))
            result['passed']=health==200 and result['model_id_matches'] and result['api_key_unchanged'] and result['ports_unchanged'] and result['functional']['passed']
            results[name]=result
        embedding_check();assert all(x['passed'] for x in results.values())
        save('functional.json',results)
        state.update(state='completed',new_27b_id=created,old_27b_backup=BACKUP,all_four_functional=True)
    except Exception as exc:
        state.update(state='failed',error_type=type(exc).__name__)
        # Persist diagnostics without serializing an API key, even if a log
        # line echoed startup configuration.
        for name in (OLD,NINE,ASR):
            log=subprocess.run(['docker','logs','--tail','100',name],capture_output=True,text=True)
            text=log.stdout+log.stderr
            for secret in secrets:text=text.replace(secret,'<REDACTED>')
            (OUT/(name+'-failure.log')).write_text(text)
        if renamed:
            try:
                if created:
                    run_docker('stop','--timeout','60',created)
                    run_docker('rename',created,OLD+'-fp8-r1-failed-20261015')
                run_docker('rename',BACKUP,OLD)
                for name in (OLD,NINE,ASR):
                    if not inspect(name)['State']['Running']:run_docker('start',name)
                    wait(name,PORTS[name],key(baseline[name]))
                state['rollback_original_services_healthy']=True
            except Exception as recovery:
                state['rollback_error_type']=type(recovery).__name__
        raise
    finally:
        stopped.set();watcher.join(timeout=20)
        state['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        save('status.json',state)


if __name__=='__main__':
    if '--run' in sys.argv:main()
    else:
        OUT.mkdir(parents=True,exist_ok=False)
        with (OUT/'detached.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run'],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':process.pid,'output':str(OUT)}))
