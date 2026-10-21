"""Chunk-structure noise floor: 6 cold ~19K prompts, greedy 256 tokens, token ids kept. Compare across arms (e.g. prefix caching on = 16160+2741 chunks vs off = 1 chunk)."""
import argparse, json
from hc import *
p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--out', required=True)
p.add_argument('--n', type=int, default=6)
a = p.parse_args()
res = {'tag': a.tag, 'records': []}
for i in range(a.n):
    lab = f'gen-v1-{i}'
    t, c = text_for(lab, 18900 - 120)
    msgs = [{'role': 'system', 'content': lab + ' chunk noise floor probe.'}, {'role': 'user', 'content': t + QUESTION}]
    r = stream(chat_payload(msgs, 256)); r.update(index=i, payload_sha256=sha(msgs))
    res['records'].append(r); json.dump(res, open(a.out, 'w'))
    print(json.dumps({k: r[k] for k in ('index', 'prompt_tokens', 'hit_tokens', 'ttft_s', 'completion_tokens')}), flush=True)
