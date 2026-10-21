"""Shared client helpers for sm120-next-p (run on the GPU host against the owned arm on 127.0.0.1:18000)."""
import hashlib, json, random, time, urllib.error, urllib.request

BASE = 'http://127.0.0.1:18000'
MODEL = 'qwen3.6-27b-fp8'
WORDS = ["router", "switch", "latency", "packet", "gateway", "service", "storage", "memory", "network", "alert", "recovery", "capacity"]
QUESTION = "\n请按网络、存储、计算资源分别分析以上合成观测，详细解释原因和排查步骤。"


def http(path, payload=None, timeout=1800):
    req = urllib.request.Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


KEYS = ('vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total', 'vllm:prompt_tokens_total',
        'vllm:prompt_tokens_by_source_total', 'vllm:prompt_tokens_cached_total', 'vllm:iteration_tokens_total_sum',
        'vllm:iteration_tokens_total_count', 'vllm:iteration_tokens_total_bucket', 'vllm:request_success_total')


def metrics():
    with urllib.request.urlopen(BASE + '/metrics', timeout=20) as r:
        lines = r.read().decode().splitlines()
    out = {}
    for l in lines:
        if l.startswith(KEYS):
            name, val = l.rsplit(' ', 1)
            name = name.replace('engine="0",', '').replace(',model_name="qwen3.6-27b-fp8"', '').replace('model_name="qwen3.6-27b-fp8",', '')
            out[name] = out.get(name, 0) + float(val)
    return out


def mdelta(b, a):
    return {k: a.get(k, 0) - v for k, v in b.items() if a.get(k, 0) - v}


def sget(d, prefix):
    return sum(v for k, v in d.items() if k.startswith(prefix))


def items(label, n, start=0):
    """First n items of a deterministic item stream; items(label,n1) is a prefix of items(label,n2)."""
    rng = random.Random(label)
    return " ".join(rng.choice(WORDS) + "=" + str(rng.randrange(10000)) for _ in range(n))


def tok_count(text):
    return http('/tokenize', {'model': MODEL, 'prompt': text, 'add_special_tokens': False})['count']


def tokens_of(text):
    return http('/tokenize', {'model': MODEL, 'prompt': text, 'add_special_tokens': False})['tokens']


_ratio = {}
def text_for(label, target_tokens, tol=3):
    """Deterministic text built from item stream `label` whose raw token count is within tol of target."""
    n = max(1, int(target_tokens / _ratio.get('r', 5.9)))
    for _ in range(6):
        t = items(label, n)
        c = tok_count(t)
        _ratio['r'] = c / n
        if abs(c - target_tokens) <= tol:
            return t, c
        n = max(1, n + round((target_tokens - c) / (c / n)))
    return t, c


def chat_tokens(messages, thinking=False):
    r = http('/tokenize', {'model': MODEL, 'messages': messages, 'add_generation_prompt': True,
                           'chat_template_kwargs': {'enable_thinking': thinking}})
    return r['tokens']


def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i


def first_div(x, y):
    if x is None or y is None: return None
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q: return i
    return None if len(x) == len(y) else min(len(x), len(y))


LOGPROBS_OK = [True]


def stream(payload, path='/v1/chat/completions'):
    """wrapper: chat requests ask for top-2 logprobs (margin analysis at divergence); retried without if the server rejects it."""
    if path == '/v1/chat/completions' and LOGPROBS_OK[0] and payload.get('max_tokens', 0) > 1:
        pl = dict(payload); pl['logprobs'] = True; pl['top_logprobs'] = 2
        try:
            return _stream(pl, path)
        except urllib.error.HTTPError as e:
            if e.code != 400: raise
            LOGPROBS_OK[0] = False
            print('logprobs rejected, continuing without:', e.read()[:200], flush=True)
    return _stream(payload, path)


def _stream(payload, path='/v1/chat/completions'):
    """Streaming request. TTFT = first chunk carrying any token/text. Returns record incl. metrics delta."""
    pl = dict(payload); pl['stream'] = True; pl['stream_options'] = {'include_usage': True}; pl['return_token_ids'] = True
    b = metrics(); t0 = time.perf_counter(); first = None; toks = []; content = []; reasoning = []; usage = None; fin = None; lps = []
    req = urllib.request.Request(BASE + path, data=json.dumps(pl).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith('data:'): continue
            d = line[5:].strip()
            if d == '[DONE]': break
            ch = json.loads(d)
            if ch.get('usage'): usage = ch['usage']
            for c in ch.get('choices', []):
                ids = c.get('token_ids') or []
                dl = c.get('delta') or {}
                txt = (dl.get('content') or '') + (dl.get('reasoning') or dl.get('reasoning_content') or '') + (c.get('text') or '')
                if (ids or txt) and first is None: first = time.perf_counter() - t0
                toks += ids
                lp = (c.get('logprobs') or {}).get('content') or []
                for e in lp: lps.append([e.get('logprob'), [[t.get('token'), t.get('logprob')] for t in (e.get('top_logprobs') or [])]])
                if dl.get('content'): content.append(dl['content'])
                if c.get('text'): content.append(c['text'])
                rs = dl.get('reasoning') or dl.get('reasoning_content')
                if rs: reasoning.append(rs)
                if c.get('finish_reason'): fin = c['finish_reason']
    total = time.perf_counter() - t0
    time.sleep(0.3)
    a = metrics(); d = mdelta(b, a)
    hit_src = d.get('vllm:prompt_tokens_by_source_total{source="local_cache_hit"}', 0)
    return {'ttft_s': first, 'total_s': total, 'usage': usage, 'finish': fin, 'token_ids': toks, 'logprobs': lps,
            'content': ''.join(content), 'reasoning': ''.join(reasoning),
            'prompt_tokens': (usage or {}).get('prompt_tokens'),
            'completion_tokens': (usage or {}).get('completion_tokens'),
            'hit_tokens': sget(d, 'vllm:prefix_cache_hits_total'), 'hit_tokens_by_source': hit_src,
            'queries': sget(d, 'vllm:prefix_cache_queries_total'),
            'iter_count': sget(d, 'vllm:iteration_tokens_total_count'), 'iter_sum': sget(d, 'vllm:iteration_tokens_total_sum'),
            'iter_buckets': {k.split('le="')[1].split('"')[0]: v for k, v in d.items() if 'iteration_tokens_total_bucket' in k}}


def chat_payload(messages, max_tokens, thinking=False, salt=None, ignore_eos=True):
    p = {'model': MODEL, 'temperature': 0, 'ignore_eos': ignore_eos, 'max_tokens': max_tokens,
         'chat_template_kwargs': {'enable_thinking': thinking}, 'messages': messages}
    if salt: p['cache_salt'] = salt
    return p


def sha(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
