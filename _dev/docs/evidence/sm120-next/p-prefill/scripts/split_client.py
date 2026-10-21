"""P4 split probe: cold prompts of several lengths (1 output token); server stderr (instr arm) logs every chunking decision;
iteration_tokens histogram delta is recorded as an independent check."""
import argparse, json
from hc import *
p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--out', required=True)
p.add_argument('--lens', default='3000,10000,18900,25000,33000,44000')
p.add_argument('--n', type=int, default=2)
a = p.parse_args()
res = {'tag': a.tag, 'records': []}
for L in [int(x) for x in a.lens.split(',')]:
    for i in range(a.n):
        lab = f'split-v1-{L}-r{i}'
        t, c = text_for(lab, L - 120)
        msgs = [{'role': 'system', 'content': lab + ' split probe.'}, {'role': 'user', 'content': t + QUESTION}]
        r = stream(chat_payload(msgs, 1)); r.pop('token_ids', None); r.update(L=L, index=i)
        res['records'].append(r); json.dump(res, open(a.out, 'w'))
        print(json.dumps({k: r[k] for k in ('L', 'index', 'prompt_tokens', 'hit_tokens', 'ttft_s', 'iter_count', 'iter_sum')}), flush=True)
