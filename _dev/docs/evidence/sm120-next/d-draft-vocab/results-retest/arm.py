"""Launch ONE owned model container (baseline or draft-vocab candidate), run clients, collect, remove.

Resumable: an arm directory containing DONE is skipped.  Flags copied from the stopped production
container exactly as evidence/sm120-mainloop/model-control/experiment_control.py (+ 12 GiB KV).
"""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys, threading, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

p = argparse.ArgumentParser()
p.add_argument('--label', required=True)           # arm dir name under runs/
p.add_argument('--subset')                          # file name inside inputs/ ; omit => baseline
p.add_argument('--no-sha', action='store_true')
p.add_argument('--no-strict', action='store_true')
p.add_argument('--patched', default=ROOT + '/src/patched')
p.add_argument('--client', nargs='*', default=[], help='client kinds: d3, perf, smoke, nsys')
p.add_argument('--d3-passes', type=int, default=2)
p.add_argument('--extra-env', action='append', default=[])
a = p.parse_args()

out = pathlib.Path(os.environ.get('RUNS_DIR', ROOT + '/runs')) / a.label
if (out / 'DONE').exists():
    print('already done', a.label); sys.exit(0)
if out.exists():
    shutil.move(str(out), str(out) + '.incomplete-' + time.strftime('%H%M%S'))
out.mkdir(parents=True)
name = PREFIX + a.label.replace('_', '-').lower()
nsys = 'nsys' in a.client
initial = json.load(open(os.environ.get('INITIAL_SNAPSHOT', ROOT + '/logs/initial-snapshot.json')))
verify(out / 'services-before.json', initial)

# ---- command: copy production flags, same overrides as earlier rounds
orig = inspect(OWN_BASELINE_SRC)
cmd = orig['Config']['Cmd'][:]
i = cmd.index('--api-key'); del cmd[i:i + 2]
def change(flag, value):
    if flag in cmd: cmd[cmd.index(flag) + 1] = str(value)
    else: cmd.extend([flag, str(value)])
for flag, value in [('--gpu-memory-utilization', .70), ('--max-num-batched-tokens', 32768),
                    ('--max-model-len', 262144), ('--max-num-seqs', 16), ('--host', '127.0.0.1'),
                    ('--port', 18000), ('--kv-cache-dtype', 'fp8'), ('--attention-backend', 'FLASHINFER')]:
    change(flag, value)
change('--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 4}))
if nsys:
    change('--profiler-config', json.dumps({'profiler': 'cuda'}))
cmd += ['--kv-cache-memory-bytes', '12884901888']
model = next(m['Source'] for m in orig['Mounts'] if m['Destination'] == '/model')
create = ['docker', 'create', '--name', name, '--gpus', 'all', '--ipc', 'host', '--network', 'host',
          '--restart', 'no', '--memory=40g', '--memory-swap=40g', '--ulimit', 'memlock=-1:-1',
          '--mount', f'type=bind,src={model},dst=/model,readonly',
          '--mount', f'type=bind,src={out}/prof,dst=/profiles']
(out / 'prof').mkdir()
env = ['HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'VLLM_USE_V2_MODEL_RUNNER=1', 'VLLM_WORKER_MULTIPROC_METHOD=spawn'] + a.extra_env
meta = {'label': a.label, 'name': name, 'subset': a.subset, 'clients': a.client}
if a.subset:
    sp = pathlib.Path(ROOT) / 'inputs' / a.subset
    sha = hashlib.sha256(sp.read_bytes()).hexdigest()
    site = sh(['docker', 'run', '--rm', '--entrypoint', 'python3', IMAGE, '-c',
               'import vllm,os;print(os.path.dirname(vllm.__file__))']).strip()
    base = site + '/v1/worker/gpu/spec_decode/'
    create += ['--mount', f'type=bind,src={a.patched}/speculator.py,dst={base}speculator.py,readonly',
               '--mount', f'type=bind,src={a.patched}/draft_vocab.py,dst={base}draft_vocab.py,readonly',
               '--mount', f'type=bind,src={sp},dst=/subsets/{a.subset},readonly']
    env += [f'VLLM_DRAFT_VOCAB_IDS=/subsets/{a.subset}']
    if not a.no_sha: env += [f'VLLM_DRAFT_VOCAB_SHA256={sha}']
    if not a.no_strict: env += ['VLLM_DRAFT_VOCAB_STRICT=1']
    meta.update(subset_sha256=sha, patched_sha256={f: hashlib.sha256((pathlib.Path(a.patched) / f).read_bytes()).hexdigest()
                                                  for f in ('speculator.py', 'draft_vocab.py')})
for e in env: create += ['--env', e]
if nsys:
    nr = '/home/llm/.local/share/qwen-profiling/nsys-2026.5.1'
    rel = 'opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys'
    create += ['--mount', f'type=bind,src={nr},dst=/nsys,readonly', '--entrypoint', '/nsys/' + rel]
    wrapped = ['profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none', '--trace-fork-before-exec=true',
               '--cuda-graph-trace=node', '--capture-range=cudaProfilerApi', '--capture-range-end=repeat:sync',
               '--output=/profiles/capture', 'vllm', 'serve'] + cmd
else:
    create += ['--entrypoint', 'vllm']
    wrapped = ['serve'] + cmd
meta.update(env=env, command=wrapped, image=IMAGE, create=create)
(out / 'arm-meta.json').write_text(json.dumps(meta, indent=2))

stop_mon = threading.Event()
def gpu_monitor():
    with open(out / 'gpu-clock.csv', 'w') as f:
        proc = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active,memory.used',
                                 '--format=csv', '-lms', '500'], stdout=f, stderr=subprocess.STDOUT)
        stop_mon.wait(); proc.terminate(); proc.wait()
def emb_monitor():
    with open(out / 'embedding-monitor.jsonl', 'w') as f:
        while not stop_mon.is_set():
            try:
                o = inspect('vllm-embedding'); r = {'at': now(), 'running': o['State']['Running'], 'id': o['Id'],
                  'started_at': o['State']['StartedAt'], 'restart_count': o['RestartCount']}
            except Exception as e:
                r = {'at': now(), 'error': repr(e)}
            f.write(json.dumps(r) + '\n'); f.flush(); stop_mon.wait(10)
threads = [threading.Thread(target=gpu_monitor, daemon=True), threading.Thread(target=emb_monitor, daemon=True)]
status = {'phases': []}
def phase(x):
    status['phases'].append({'at': now(), 'phase': x}); (out / 'status.json').write_text(json.dumps(status, indent=2)); print(x, flush=True)
try:
    for t in threads: t.start()
    sh(create + [IMAGE] + wrapped)
    sh(['docker', 'start', name]); phase('started ' + name)
    t0 = time.time()
    while True:
        st = inspect(name)['State']
        if not st['Running']: raise RuntimeError('container exited during startup')
        try:
            with urllib.request.urlopen('http://127.0.0.1:18000/health', timeout=3) as r:
                if r.status == 200: break
        except Exception:
            if time.time() - t0 > 1500: raise RuntimeError('startup deadline')
            time.sleep(8)
    phase('healthy after %.0fs' % (time.time() - t0))
    with open(out / 'startup.log', 'w') as f:
        subprocess.run(['docker', 'logs', name], stdout=f, stderr=subprocess.STDOUT)
    S = ROOT + '/bin/'
    tag = a.label
    for kind in a.client:
        phase('client ' + kind)
        if kind == 'smoke':
            cl = ['python3', S + 'd3_client.py', '--tag', tag, '--out', str(out / 'smoke.json'), '--passes', '1', '--only', 'smoke']
        elif kind == 'd3':
            cl = ['python3', S + 'd3_client.py', '--tag', tag, '--out', str(out / 'd3.json'), '--passes', str(a.d3_passes)]
        elif kind == 'perf':
            cl = ['python3', ROOT + '/bin/profile_client.py', '--tag', tag, '--mode', 'baseline', '--profiler-installed', 'none',
                  '--cases', 'decode', 'thinking', '--repeats', '6']
        elif kind == 'x':
            cl = ['python3', S + 'xclient.py', '--tag', tag, '--out', str(out / 'x.json')]
        elif kind == 'tf':
            cl = ['python3', S + 'tf_client.py']
        elif kind == 'nsys':
            cl = ['python3', ROOT + '/bin/profile_client.py', '--tag', tag, '--mode', 'capture', '--profiler-installed', 'cuda',
                  '--cases', 'decode', '--repeats', '2']
        else:
            raise RuntimeError(kind)
        with open(out / (kind + '.jsonl'), 'w') as o, open(out / (kind + '.stderr'), 'w') as e:
            rc = subprocess.run(cl, stdout=o, stderr=e, timeout=3000).returncode
        status[kind + '_rc'] = rc; phase(f'client {kind} rc={rc}')
        if rc: raise RuntimeError(kind + ' failed')
    with urllib.request.urlopen('http://127.0.0.1:18000/metrics', timeout=10) as r:
        (out / 'metrics-final.txt').write_text(r.read().decode())
    (out / 'DONE_PENDING').write_text(now())
finally:
    stop_mon.set()
    try:
        with open(out / 'server.log', 'w') as f:
            subprocess.run(['docker', 'logs', name], stdout=f, stderr=subprocess.STDOUT, timeout=60)
        if nsys:
            subprocess.run(['docker', 'stop', '--time', '60', name], timeout=90)
        else:
            subprocess.run(['docker', 'stop', '--time', '30', name], timeout=60)
        subprocess.run(['docker', 'rm', name], timeout=60)
    except Exception as e:
        print('cleanup problem', e)
    for t in threads: t.join(timeout=20)
    verify(out / 'services-after.json', initial)
if (out / 'DONE_PENDING').exists():
    (out / 'DONE').write_text(now())
