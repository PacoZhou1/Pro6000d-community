"""E2 cold prefill at several lengths: n unique-prefix requests per length, 1 output token, streaming TTFT, prefix-hit delta must be 0."""
import argparse, json, time
from hc import *

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--out', required=True)
p.add_argument('--lens', default='18900,44000,64000,128000'); p.add_argument('--n', type=int, default=4)
a = p.parse_args()
res = {'tag': a.tag, 'records': []}
for L in [int(x) for x in a.lens.split(',')]:
    for i in range(0, (2 if L >= 128000 else min(a.n, 3)) + 1):        # i==0 is an unrecorded-in-stats warm-up (flag warmup)
        lab = f'len-v1-{L}-r{i}'       # same label across arms => same prompt text per (L,i)
        t, c = text_for(lab, L - 120)
        msgs = [{'role': 'system', 'content': lab + ' Synthetic prefill latency probe.'}, {'role': 'user', 'content': t + QUESTION}]
        r = stream(chat_payload(msgs, 1, ignore_eos=True))
        r.update(L=L, index=i, warmup=(i == 0), payload_sha256=sha(msgs))
        r.pop('token_ids', None)
        res['records'].append(r); json.dump(res, open(a.out, 'w'))
        print(json.dumps({k: r[k] for k in ('L', 'index', 'prompt_tokens', 'hit_tokens', 'ttft_s', 'iter_count', 'iter_sum')}), flush=True)
print('LEN_DONE', flush=True)
