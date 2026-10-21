"""D3 correctness client: fixed greedy requests, token ids + text + spec-decode counters per request.

Prompts come from the SAME builder as evidence/profiling/profile_client.py (payload()) and
/tmp/qwen-spec-experiment/open_output_probe.py. Non-streaming, return_token_ids so token-by-token
comparison is possible.  Runs `--passes` passes (pass 2 is prefix-cache warm).
"""
import argparse, hashlib, json, random, re, time, urllib.request, uuid

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--out', required=True)
p.add_argument('--passes', type=int, default=2)
p.add_argument('--only', choices=['all', 'smoke'], default='all')
p.add_argument('--base', default='http://127.0.0.1:18000')
a = p.parse_args()


def post(payload, timeout=600):
    req = urllib.request.Request(a.base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def metrics():
    with urllib.request.urlopen(a.base + '/metrics', timeout=15) as r:
        lines = r.read().decode().splitlines()
    out = {}
    for l in lines:
        if l.startswith('vllm:spec_decode'):
            key = l.rsplit(' ', 1)[0]  # name{labels}
            out[key] = float(l.rsplit(' ', 1)[1])
    return out


def payload(case, label, max_tokens):   # verbatim logic of profile_client.payload (non-capture)
    rng = random.Random(label)
    words = ["router", "switch", "latency", "packet", "gateway", "service",
             "storage", "memory", "network", "alert", "recovery", "capacity"]
    count = 3200 if case == "prefill" else 800
    content = " ".join(rng.choice(words) + "=" + str(rng.randrange(10000)) for _ in range(count))
    content += "\n请按网络、存储、计算资源分别分析以上合成观测，详细解释原因和排查步骤。"
    return {"model": "qwen3.6-27b-fp8", "temperature": 0, "ignore_eos": True, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": case == "thinking"},
            "messages": [{"role": "system", "content": str(uuid.uuid5(uuid.NAMESPACE_URL, label)) +
                          " Synthetic inference profiling. Explain in clear Chinese prose."},
                         {"role": "user", "content": content}]}


OPEN = [
    ("open_chinese_operations_summary", [
        {"role": "system", "content": "你是运维助手。只依据用户给出的合成事件回答。"},
        {"role": "user", "content": ("合成事件：14:00，节点lab-router-7出口丢包率升至12%，CPU占用21%；"
            "14:03，将出口流量切换至备用链路；14:06，丢包率回落到0.2%。"
            "请用两句中文概述故障、处置与恢复。必须原样保留节点名、"
            "12%、0.2%和备用链路，不要输出JSON。")}], False, 384),
    ("open_thinking_arithmetic", [
        {"role": "system", "content": "请认真推理合成算术题，并按用户要求给出最终答复。"},
        {"role": "user", "content": ("仓库原有17箱设备，每箱8台，另到货9台，已发出53台。"
            "还剩多少台？可以在思考中计算，最终正文只用一句中文和阿拉伯数字回答，"
            "不要写算式或JSON。")}], True, 1024)]

items = []
n = 2 if a.only == 'smoke' else 6
for case in ('decode', 'thinking'):
    for i in range(1, n + 1):
        items.append({'kind': case, 'index': i,
                      'payload': payload(case, f'qwen-profile-v1:baseline:{case}:r{i}:warmFalse', 1024)})
if a.only != 'smoke':
    for name, msgs, think, mt in OPEN:
        for rep in (1, 2):
            items.append({'kind': name, 'index': rep, 'payload': {
                "model": "qwen3.6-27b-fp8", "messages": msgs, "temperature": 0, "max_tokens": mt,
                "chat_template_kwargs": {"enable_thinking": think}}})
# 256-token smoke: 1 normal short
results = {'tag': a.tag, 'passes': []}
# warm-up (not recorded): compile/caches
post({**payload('decode', 'warmup-d3', 32)})
for ps in range(1, a.passes + 1):
    recs = []
    for it in items:
        pl = dict(it['payload']); pl['return_token_ids'] = True
        before = metrics(); t0 = time.perf_counter()
        body = post(pl); el = time.perf_counter() - t0
        after = metrics()
        ch = body['choices'][0]; msg = ch['message']
        toks = ch.get('token_ids')
        d = {k: after.get(k, 0) - v for k, v in before.items()}
        agg = lambda pre: sum(v for k, v in d.items() if k.startswith(pre))
        recs.append({
            'kind': it['kind'], 'index': it['index'], 'pass': ps,
            'payload_sha256': hashlib.sha256(json.dumps(it['payload'], sort_keys=True).encode()).hexdigest(),
            'finish_reason': ch.get('finish_reason'), 'usage': body.get('usage'),
            'token_ids': toks, 'token_ids_present': toks is not None,
            'content': msg.get('content'), 'reasoning': msg.get('reasoning_content') or msg.get('reasoning'),
            'elapsed_s': el,
            'spec': {'drafts': agg('vllm:spec_decode_num_drafts_total'),
                     'draft_tokens': agg('vllm:spec_decode_num_draft_tokens_total'),
                     'accepted': agg('vllm:spec_decode_num_accepted_tokens_total'),
                     'per_pos': {k: v for k, v in d.items() if 'per_pos' in k and v}}})
        print(json.dumps({k: recs[-1][k] for k in ('kind', 'index', 'pass', 'finish_reason', 'elapsed_s', 'token_ids_present')}), flush=True)
    results['passes'].append(recs)
    json.dump(results, open(a.out, 'w'), ensure_ascii=False)
