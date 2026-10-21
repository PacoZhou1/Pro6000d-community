"""P3 A1/B/A2 (and nsys witness) driver, run on the GPU host with nohup.  Same protocol as doc 11 / E3:
original experiment_control.py + run_api_arm.py (host copies in FUS), 12 long prefill + 6 decode + 5 quality + 4 open-output
per arm.  V2 is ENABLED in every arm (new baseline); the P3 fusions (PRO6000D_F_ENABLE=norm,conv) only in B / witness.
usage: f_run.py ARM [ARM...]   with ARM in a1 b a2 w (clean nsys witness) wv (nsys witness + in-situ VERIFY of norm+quant), w0 (V2-only nsys baseline, keeps inductor cache), wc (candidate nsys, keeps inductor cache)"""
import hashlib, json, pathlib, subprocess, sys, threading, time
sys.path.insert(0, '/tmp/qwen-sm120-next-f/bin')
import common
FUS = '/home/llm/.local/share/qwen-fp8-fusion'
R = pathlib.Path('/tmp/qwen-sm120-next-f/aba'); R.mkdir(exist_ok=True)
INSTR = '/tmp/qwen-sm120-next-f/instr'
LIBSHA = '087b7caa92c4742ece3875b6dd1421e25360b834ce4106f7fa166f90bc933b3f'
KSHA = hashlib.sha256(open(INSTR + '/f_kernels.py', 'rb').read()).hexdigest()
assert hashlib.sha256(open(INSTR + '/v2/libfp8_silu_quant.so', 'rb').read()).hexdigest() == LIBSHA
initial = json.load(open('/tmp/qwen-sm120-next-f/logs/initial-snapshot.json'))
NSYS_ROOT = '/home/llm/.local/share/qwen-profiling/nsys-2026.5.1'
NSYS_REL = 'opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys'
for arm in sys.argv[1:]:
    d = R / arm
    if (d / 'DONE').exists():
        continue
    d.mkdir(exist_ok=True)
    name = 'codex-fp8-f-' + arm
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
                except Exception as e:
                    r = {'at': common.now(), 'error': repr(e)}
                f.write(json.dumps(r) + '\n'); f.flush(); stop.wait(10)
    ts = [threading.Thread(target=gpu, daemon=True), threading.Thread(target=emb, daemon=True)]
    for t in ts: t.start()
    try:
        cmd = ['python3', FUS + '/experiment_control.py', '--name', name, '--method', 'mtp', '--tokens', '4',
               '--profiler', 'cuda' if arm in ('w', 'wv', 'w0', 'wc') else 'none', '--output-dir', str(d / 'prof'),
               '--instrument-dir', INSTR,
               '--extra-env', 'PRO6000D_F_V2=1', '--extra-env', 'PRO6000D_FP8_FUSION_ENABLE=1',
               '--extra-env', 'PRO6000D_FP8_FUSION_LIBRARY=/instrument/v2/libfp8_silu_quant.so',
               '--extra-env', 'PRO6000D_FP8_FUSION_LIBRARY_SHA256=' + LIBSHA]
        if arm in ('w', 'wv', 'w0', 'wc'):
            cmd += ['--nsys-root', NSYS_ROOT, '--nsys-relative', NSYS_REL]
        if arm == 'wv':
            cmd += ['--extra-env', 'PRO6000D_F_VERIFY=1']
        if arm in ('w0', 'wc'):
            cmd += ['--extra-env', 'VLLM_CACHE_ROOT=/profiles/vllm_cache']   # keep Inductor output code for inspection
        if arm in ('bn', 'bc'):   # isolation arms: F1-only / F2-only
            cmd += ['--extra-env', 'PRO6000D_F_ENABLE=' + ('norm' if arm == 'bn' else 'conv'),
                    '--extra-env', 'PRO6000D_F_KERNELS_SHA256=' + KSHA]
        if arm in ('b', 'w', 'wv', 'wc'):
            cmd += ['--extra-env', 'PRO6000D_F_ENABLE=norm,conv', '--extra-env', 'PRO6000D_F_KERNELS_SHA256=' + KSHA]
        (d / 'control-cmd.json').write_text(json.dumps(cmd))
        subprocess.run(cmd, check=True, stdout=open(d / 'control.out', 'w'), stderr=subprocess.STDOUT)
        api = ['python3', FUS + '/run_api_arm.py', '--name', name, '--output', str(d), '--tag', 'f-' + arm]
        if arm in ('w', 'wv', 'w0', 'wc'):
            api += ['--witness']
        subprocess.run(api, check=True, stdout=open(d / 'run_api.out', 'w'), stderr=subprocess.STDOUT, timeout=3600)
        (d / 'DONE_PENDING').write_text(common.now())
    finally:
        stop.set()
        with open(d / 'server.log', 'w') as f:
            subprocess.run(['docker', 'logs', name], stdout=f, stderr=subprocess.STDOUT)
        subprocess.run(['docker', 'stop', '--time', '60', name]); subprocess.run(['docker', 'rm', name])
        for t in ts: t.join(timeout=20)
        common.verify(d / 'services-after.json', initial)
    (d / 'DONE').write_text(common.now())
print('F RUN DONE')
