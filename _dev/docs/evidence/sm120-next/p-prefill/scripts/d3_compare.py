"""Offline: compare d3.json files token-by-token and summarize acceptance.  usage: d3_compare.py runs_dir out.json"""
import json, sys, pathlib

root = pathlib.Path(sys.argv[1]); out = {}
arms = {}
for d in sorted(root.iterdir()):
    f = d / 'd3.json'
    if f.exists(): arms[d.name] = json.load(open(f))
def key(r): return (r['kind'], r['index'])
def first_div(x, y):
    if x is None or y is None: return None
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q: return i
    return None if len(x) == len(y) else min(len(x), len(y))
def spec_sum(recs, kinds=None):
    dr = sum(r['spec']['draft_tokens'] for r in recs if kinds is None or r['kind'] in kinds)
    ac = sum(r['spec']['accepted'] for r in recs if kinds is None or r['kind'] in kinds)
    nd = sum(r['spec']['drafts'] for r in recs if kinds is None or r['kind'] in kinds)
    return {'drafts': nd, 'draft_tokens': dr, 'accepted': ac, 'accept_rate': ac / dr if dr else None,
            'mean_accept_len': ac / nd if nd else None, 'mean_tokens_per_step': (ac / nd + 1) if nd else None}
GROUPS = {'normal': ['decode'], 'thinking': ['thinking'], 'open': ['open_chinese_operations_summary', 'open_thinking_arithmetic']}
out['acceptance'] = {}
for name, d in arms.items():
    for ps in d['passes']:
        pn = ps[0]['pass']
        out['acceptance'][f'{name}/pass{pn}'] = {'all': spec_sum(ps), **{g: spec_sum(ps, k) for g, k in GROUPS.items()}}
def cmp(a, b, pa, pb):
    A = {key(r): r for r in arms[a]['passes'][pa]}; B = {key(r): r for r in arms[b]['passes'][pb]}
    rows = []
    for k in A:
        if k not in B: continue
        x, y = A[k], B[k]
        fd = first_div(x['token_ids'], y['token_ids'])
        rows.append({'kind': k[0], 'index': k[1], 'tokens_a': len(x['token_ids'] or []), 'tokens_b': len(y['token_ids'] or []),
                     'identical_tokens': fd is None, 'first_divergence_token': fd,
                     'content_equal': x['content'] == y['content'], 'reasoning_equal': x['reasoning'] == y['reasoning'],
                     'finish_a': x['finish_reason'], 'finish_b': y['finish_reason'], 'payload_equal': x['payload_sha256'] == y['payload_sha256']})
    return {'n': len(rows), 'identical': sum(r['identical_tokens'] for r in rows), 'rows': rows}
out['comparisons'] = {}
names = list(arms)
base = next((n for n in names if n.startswith('d3-base')), None)
if base:
    if len(arms[base]['passes']) > 1: out['comparisons'][f'{base}.pass1 vs {base}.pass2 (self-consistency)'] = cmp(base, base, 0, 1)
    for n in names:
        if n == base: continue
        out['comparisons'][f'{base}.pass1 vs {n}.pass1'] = cmp(base, n, 0, 0)
        if len(arms[n]['passes']) > 1: out['comparisons'][f'{base}.pass2 vs {n}.pass2'] = cmp(base, n, 1, 1)
out['token_ids_present'] = {n: all(r['token_ids_present'] for ps in d['passes'] for r in ps) for n, d in arms.items()}
out['finish'] = {n: {f"{r['kind']}:{r['index']}:p{r['pass']}": r['finish_reason'] for ps in d['passes'] for r in ps} for n, d in arms.items()}
json.dump(out, open(sys.argv[2], 'w'), indent=1, ensure_ascii=False)
for k, v in out['comparisons'].items(): print(k, f"{v['identical']}/{v['n']} identical")
for k, v in out['acceptance'].items(): print(k, {g: (round(x['accept_rate'], 4) if x['accept_rate'] else None, round(x['mean_accept_len'], 3) if x['mean_accept_len'] else None) for g, x in v.items()})
