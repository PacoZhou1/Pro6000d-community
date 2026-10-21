"""Teacher-forcing diagnostic on the BASELINE server: for each 32K-vs-baseline divergence, feed prompt + baseline prefix
(tokens[:fd]) and get top-5 logprobs for the next token; report baseline vs candidate token logprob margin."""
import json, sys, urllib.request, random
sys.argv = [sys.argv[0]] + sys.argv[1:]
R = '/tmp/qwen-sm120-next-d/runs/'
B = 'http://127.0.0.1:18000'
def post(path, pl):
    req = urllib.request.Request(B + path, data=json.dumps(pl).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=600))
base = json.load(open(R + 'd3-base/d3.json'))['passes'][0]
cand = json.load(open(R + 'd3-32768/d3.json'))['passes'][0]
import importlib.util, types
# rebuild payloads identical to d3_client
src = open('/tmp/qwen-sm120-next-d/bin/d3_client.py').read()
ns = {}
pre = src.split('OPEN = [')[0]
pre = pre.split('p = argparse')[0] + 'import argparse\n' + src[src.index('def payload'):src.index('OPEN = [')]
exec(compile('import random, uuid\n' + src[src.index('def payload'):src.index('OPEN = [')], 'p', 'exec'), ns)
out = []
for b, c in zip(base, cand):
    if b['kind'] not in ('decode', 'thinking'): continue
    bt, ct = b['token_ids'], c['token_ids']
    fd = next((i for i, (x, y) in enumerate(zip(bt, ct)) if x != y), None)
    if fd is None: continue
    pl = ns['payload'](b['kind'], f"qwen-profile-v1:baseline:{b['kind']}:r{b['index']}:warmFalse", 1024)
    tk = post('/tokenize', {'model': pl['model'], 'messages': pl['messages'], 'add_generation_prompt': True,
                            'chat_template_kwargs': pl['chat_template_kwargs']})['tokens']
    r = post('/v1/completions', {'model': pl['model'], 'prompt': tk + bt[:fd], 'max_tokens': 1, 'temperature': 0, 'logprobs': 10,
                                 'return_tokens_as_token_ids': True})
    lp = r['choices'][0]['logprobs']['top_logprobs'][0]
    d = {k.replace('token_id:', ''): v for k, v in lp.items()}
    out.append({'kind': b['kind'], 'index': b['index'], 'first_div': fd, 'prompt_tokens': len(tk), 'base_tok': bt[fd], 'cand_tok': ct[fd],
                'tf_top1': r['choices'][0]['logprobs']['tokens'][0], 'lp_base_tok': d.get(str(bt[fd])), 'lp_cand_tok': d.get(str(ct[fd])),
                'top': d})
    print(json.dumps({k: out[-1][k] for k in out[-1] if k != 'top'}), flush=True)
json.dump(out, open(R + 'tf-base/tf.json', 'w'), indent=1)
