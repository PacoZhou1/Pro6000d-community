"""usage: xcompare.py base_x.json cand_x.json out.json"""
import json, sys
B = json.load(open(sys.argv[1]))['records']; C = json.load(open(sys.argv[2]))['records']
def fd(x, y):
    if x is None or y is None: return 'n/a'
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q: return i
    return None if len(x) == len(y) else min(len(x), len(y))
def idx(R): return {(r['mode'], r['i']): r for r in R}
b, c = idx(B), idx(C); out = {}
def cmp(name, A, Z):
    rows = []
    for k in sorted(A):
        if k not in Z: continue
        x, y = A[k], Z[k]; d = fd(x['token_ids'], y['token_ids'])
        rows.append({'mode': k[0], 'i': k[1], 'topic': x['topic'], 'case': x['case'], 'max_tokens': x['max_tokens'],
                     'tokens': [len(x['token_ids'] or []), len(y['token_ids'] or [])], 'first_divergence': d,
                     'identical': d is None, 'text_equal': x['content'] == y['content'] and x['reasoning'] == y['reasoning']})
    out[name] = {'n': len(rows), 'identical': sum(r['identical'] for r in rows), 'rows': rows}
    print(name, f"{out[name]['identical']}/{len(rows)}", [(r['i'], r['first_divergence']) for r in rows if not r['identical']])
for m in ('nonstream', 'stream'):
    cmp(f'base vs cand [{m}]', {k: v for k, v in b.items() if k[0] == m}, {k: v for k, v in c.items() if k[0] == m})
def cross(src, tag):
    A = {('x', k[1]): v for k, v in src.items() if k[0] == 'nonstream'}; Z = {('x', k[1]): v for k, v in src.items() if k[0] == 'stream'}
    cmp(f'{tag}: nonstream vs stream', A, Z)
cross(b, 'base'); cross(c, 'cand')
out['token_ids_present'] = {'base': all(r['token_ids_present'] for r in B), 'cand': all(r['token_ids_present'] for r in C)}
json.dump(out, open(sys.argv[3], 'w'), indent=1)
