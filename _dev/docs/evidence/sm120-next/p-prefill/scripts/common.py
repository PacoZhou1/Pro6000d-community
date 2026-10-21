"""Shared helpers for task D host scripts (run on the GPU host). Never touches non-own containers except read-only inspect."""
import datetime, json, math, subprocess, urllib.request

ROOT = '/tmp/qwen-sm120-next-p'
IMAGE = 'sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
PREFIX = 'codex-next-p-'
OWN_BASELINE_SRC = 'vllm-qwen-27b'  # inspect only (read-only) to copy flags


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sh(cmd, **kw):
    return subprocess.check_output(cmd, text=True, **kw)


def inspect(name):
    return json.loads(sh(['docker', 'inspect', name]))[0]


def snapshot():
    """ID/StartedAt/RestartCount/Running for every container that is not ours."""
    out = {}
    for name in sh(['docker', 'ps', '-a', '--format', '{{.Names}}']).split():
        if name.startswith((PREFIX, 'codex-fp8-')):
            continue
        o = inspect(name)
        out[name] = {'id': o['Id'], 'image': o['Image'], 'running': o['State']['Running'],
                     'started_at': o['State']['StartedAt'], 'restart_count': o['RestartCount']}
    return out


def embedding_check():
    o = inspect('vllm-embedding')
    cmd = o['Config']['Cmd']
    key = cmd[cmd.index('--api-key') + 1]
    model = cmd[cmd.index('--served-model-name') + 1]
    h = {'Authorization': 'Bearer ' + key}
    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8002/health', headers=h), timeout=15) as r:
        health = r.status
    body = json.dumps({'model': model, 'input': 'task D service check'}).encode()
    req = urllib.request.Request('http://127.0.0.1:8002/v1/embeddings', data=body,
                                 headers={**h, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        vec = json.load(r)['data'][0]['embedding']
    return {'health': health, 'dims': len(vec), 'finite': all(math.isfinite(v) for v in vec)}


def verify(path, baseline_snapshot=None):
    snap = snapshot()
    res = {'at_utc': now(), 'snapshot': snap, 'embedding': embedding_check()}
    ok = res['embedding']['health'] == 200 and res['embedding']['dims'] == 1024 and res['embedding']['finite']
    ok = ok and snap['vllm-embedding']['running']
    if baseline_snapshot:
        diffs = {k: [f for f in ('id', 'started_at', 'restart_count', 'running') if baseline_snapshot[k][f] != v[f]]
                 for k, v in snap.items() if k in baseline_snapshot}
        diffs = {k: v for k, v in diffs.items() if v}
        res['differences_vs_initial'] = diffs
        ok = ok and not diffs and set(snap) == set(baseline_snapshot)
    res['passed'] = bool(ok)
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)
    if not ok:
        raise RuntimeError('service verification failed: ' + path)
    return res
