"""Extra correctness client (retest): 24 new greedy requests, each sent non-streaming AND streaming with return_token_ids.
Payload builder = d3_client.payload (generic topic verbatim); zh/code topics only change the word list / tail text.
Matrix: 3 output lengths (256/1024/2048) x 2 modes (normal/thinking) x 4 topics (generic, zh, code, zh2) = 24."""
import argparse, hashlib, json, random, time, urllib.request, uuid

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--out', required=True)
p.add_argument('--base', default='http://127.0.0.1:18000')
a = p.parse_args()

EN = ["router", "switch", "latency", "packet", "gateway", "service", "storage", "memory", "network", "alert", "recovery", "capacity"]
ZH = ["路由器", "交换机", "时延", "丢包", "网关", "服务", "存储", "内存", "网络", "告警", "恢复", "容量", "带宽", "链路", "主机", "磁盘"]
TAILS = {
    'generic': "\n请按网络、存储、计算资源分别分析以上合成观测，详细解释原因和排查步骤。",
    'zh': "\n以上是合成的中文运维观测。请逐项归纳异常，写一份结构化的中文故障分析报告，包含现象、根因假设、验证步骤和处置建议。",
    'zh2': "\n请把以上合成观测改写成一篇通俗易懂的中文科普文章，面向非技术读者，使用比喻，分小节叙述。",
    'code': "\nReview the synthetic Python functions above. Explain what each does, find edge cases, then rewrite them as one typed module with docstrings and unit tests.",
}

def payload(case, label, max_tokens, topic):
    rng = random.Random(label)
    if topic == 'code':
        content = "\n".join(f"def f_{rng.randrange(1000)}(x, y={rng.randrange(50)}):\n    return (x * {rng.randrange(1, 97)} + y) % {rng.randrange(2, 1000)}"
                            for _ in range(220))
    else:
        words = ZH if topic in ('zh', 'zh2') else EN
        sep = "：" if topic in ('zh', 'zh2') else "="
        content = " ".join(rng.choice(words) + sep + str(rng.randrange(10000)) for _ in range(800))
    content += TAILS[topic]
    return {"model": "qwen3.6-27b-fp8", "temperature": 0, "ignore_eos": True, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": case == "thinking"},
            "messages": [{"role": "system", "content": str(uuid.uuid5(uuid.NAMESPACE_URL, label)) +
                          " Synthetic inference profiling. Explain in clear Chinese prose."},
                         {"role": "user", "content": content}]}

items = []
for i in range(24):
    mt = [256, 1024, 2048][i % 3]; think = (i // 3) % 2 == 1; topic = ['generic', 'zh', 'code', 'zh2'][i // 6]
    case = 'thinking' if think else 'decode'
    items.append({'i': i, 'topic': topic, 'case': case, 'max_tokens': mt,
                  'payload': payload(case, f'retest-x-v1:{topic}:{case}:{mt}:r{i}', mt, topic)})

def post(pl):
    req = urllib.request.Request(a.base + '/v1/chat/completions', data=json.dumps(pl).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        b = json.load(r)
    ch = b['choices'][0]; m = ch['message']
    return ch.get('token_ids'), m.get('content'), m.get('reasoning_content') or m.get('reasoning'), ch.get('finish_reason')

def stream(pl):
    pl = dict(pl); pl['stream'] = True
    req = urllib.request.Request(a.base + '/v1/chat/completions', data=json.dumps(pl).encode(), headers={'Content-Type': 'application/json'})
    toks, content, reasoning, fin, present = [], [], [], None, False
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith('data:'): continue
            d = line[5:].strip()
            if d == '[DONE]': break
            for ch in json.loads(d).get('choices', []):
                if ch.get('token_ids') is not None: present = True; toks += ch['token_ids']
                dl = ch.get('delta') or {}
                if dl.get('content'): content.append(dl['content'])
                rs = dl.get('reasoning_content') or dl.get('reasoning')
                if rs: reasoning.append(rs)
                if ch.get('finish_reason'): fin = ch['finish_reason']
    return (toks if present else None), ''.join(content), ''.join(reasoning) or None, fin

post({**items[0]['payload'], 'max_tokens': 16})  # warm-up, not recorded
res = {'tag': a.tag, 'records': []}
for mode, fn in (('nonstream', post), ('stream', stream)):
    for it in items:
        pl = dict(it['payload']); pl['return_token_ids'] = True
        t0 = time.perf_counter(); toks, c, rs, fin = fn(pl); el = time.perf_counter() - t0
        res['records'].append({'mode': mode, 'i': it['i'], 'topic': it['topic'], 'case': it['case'], 'max_tokens': it['max_tokens'],
            'payload_sha256': hashlib.sha256(json.dumps(it['payload'], sort_keys=True).encode()).hexdigest(),
            'token_ids': toks, 'token_ids_present': toks is not None, 'content': c, 'reasoning': rs, 'finish_reason': fin, 'elapsed_s': el})
        print(json.dumps({'mode': mode, 'i': it['i'], 'n': len(toks or []), 'fin': fin, 's': round(el, 1)}), flush=True)
        json.dump(res, open(a.out, 'w'), ensure_ascii=False)
