"""E3: V2 SiLU+quant cross-session retest. Same protocol as doc 11 (A1 official / B V2 / A2 official), driven by the ORIGINAL
experiment_control.py + run_api_arm.py from the first session (host copies under qwen-fp8-fusion/), only names/dirs differ.
Adds host-side monitors (gpu clocks, embedding) and docker logs capture.  usage: e3_run.py a1 b a2"""
import json, os, pathlib, subprocess, sys, threading, time
sys.path.insert(0, '/tmp/qwen-sm120-next-p/bin')
import common
FUS = '/home/llm/.local/share/qwen-fp8-fusion'
R = pathlib.Path('/tmp/qwen-sm120-next-p/e3'); R.mkdir(exist_ok=True)
LIBSHA = '087b7caa92c4742ece3875b6dd1421e25360b834ce4106f7fa166f90bc933b3f'
initial = json.load(open('/tmp/qwen-sm120-next-p/logs/initial-snapshot.json'))
import hashlib
assert hashlib.sha256(open(FUS + '/runtime/libfp8_silu_quant.so', 'rb').read()).hexdigest() == LIBSHA
for arm in sys.argv[1:]:
    d = R / arm
    if (d / 'DONE').exists(): continue
    d.mkdir(exist_ok=True); name = 'codex-fp8-p-' + arm
    common.verify(d / 'services-before.json', initial)
    stop = threading.Event()
    def gpu():
        with open(d / 'gpu-clock.csv', 'w') as f:
            pr = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active,memory.used', '--format=csv', '-lms', '500'], stdout=f)
            stop.wait(); pr.terminate(); pr.wait()
    def emb():
        with open(d / 'embedding-monitor.jsonl', 'w') as f:
            while not stop.is_set():
                try:
                    o = common.inspect('vllm-embedding'); r = {'at': common.now(), 'running': o['State']['Running'], 'id': o['Id'], 'started_at': o['State']['StartedAt'], 'restart_count': o['RestartCount']}
                except Exception as e: r = {'at': common.now(), 'error': repr(e)}
                f.write(json.dumps(r) + '\n'); f.flush(); stop.wait(10)
    ts = [threading.Thread(target=gpu, daemon=True), threading.Thread(target=emb, daemon=True)]
    for t in ts: t.start()
    try:
        cmd = ['python3', FUS + '/experiment_control.py', '--name', name, '--method', 'mtp', '--tokens', '4', '--profiler', 'none', '--output-dir', str(d / 'prof')]
        if arm.startswith('b'):
            cmd += ['--instrument-dir', FUS + '/runtime', '--extra-env', 'PRO6000D_FP8_FUSION_ENABLE=1',
                    '--extra-env', 'PRO6000D_FP8_FUSION_LIBRARY=/instrument/libfp8_silu_quant.so',
                    '--extra-env', 'PRO6000D_FP8_FUSION_LIBRARY_SHA256=' + LIBSHA]
        (d / 'control-cmd.json').write_text(json.dumps(cmd))
        subprocess.run(cmd, check=True, stdout=open(d / 'control.out', 'w'), stderr=subprocess.STDOUT)
        subprocess.run(['python3', FUS + '/run_api_arm.py', '--name', name, '--output', str(d), '--tag', 'p-' + arm], check=True,
                       stdout=open(d / 'run_api.out', 'w'), stderr=subprocess.STDOUT, timeout=3600)
        (d / 'DONE_PENDING').write_text(common.now())
    finally:
        stop.set()
        with open(d / 'server.log', 'w') as f: subprocess.run(['docker', 'logs', name], stdout=f, stderr=subprocess.STDOUT)
        subprocess.run(['docker', 'stop', '--time', '30', name]); subprocess.run(['docker', 'rm', name])
        for t in ts: t.join(timeout=20)
        common.verify(d / 'services-after.json', initial)
    (d / 'DONE').write_text(common.now())
print('E3 DONE')
