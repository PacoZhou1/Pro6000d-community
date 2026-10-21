"""E2 analysis: usage analyze_e2.py RUNS_DIR OUT_DIR"""
import json, statistics as st, sys, pathlib
runs = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2]); out.mkdir(exist_ok=True, parents=True)
ARMS = ['e2-fi-a1', 'e2-flash-attn', 'e2-triton', 'e2-fi-a2']
L = []; P = L.append
def load(n):
    f = runs / n / 'len.json'
    return [r for r in json.load(open(f))['records']] if f.exists() else None
D = {n: load(n) for n in ARMS}
lens = sorted({r['L'] for v in D.values() if v for r in v})
P('| length | ' + ' | '.join(f'{n} median s (min-max)' for n in ARMS) + ' | prompt tokens | hit tokens |'); P('|---|' + '---|' * (len(ARMS) + 2))
for l in lens:
    row = [str(l)]; pt = None; hits = 0
    for n in ARMS:
        rs = [r for r in (D[n] or []) if r['L'] == l and not r['warmup']]
        if rs:
            t = [r['ttft_s'] for r in rs]; row.append(f'{st.median(t):.3f} ({min(t):.3f}-{max(t):.3f}, n={len(t)})'); pt = rs[0]['prompt_tokens']; hits += sum(r['hit_tokens'] for r in rs)
        else: row.append('n/a')
    P('| ' + ' | '.join(row) + f' | {pt} | {hits:.0f} |')
P('\npaired change vs mean(A1,A2) per index (positive = slower than FlashInfer):')
if D['e2-fi-a1'] and D['e2-fi-a2']:
    for l in lens:
        def g(n): return {r['index']: r['ttft_s'] for r in (D[n] or []) if r['L'] == l and not r['warmup']}
        a1, a2 = g('e2-fi-a1'), g('e2-fi-a2')
        idx = [i for i in a1 if i in a2]
        if not idx: continue
        drift = [100 * (a2[i] - a1[i]) / a1[i] for i in idx]
        P(f'- L={l}: A1->A2 drift median {st.median(drift):+.3f}% (n={len(idx)})')
        for n in ('e2-flash-attn', 'e2-triton'):
            b = g(n)
            if b:
                d = [100 * (b[i] - (a1[i] + a2[i]) / 2) / ((a1[i] + a2[i]) / 2) for i in idx if i in b]
                P(f'  - {n}: median {st.median(d):+.3f}% range {min(d):+.3f}..{max(d):+.3f}')
# d3 correctness vs A1
def fd(x, y):
    if x is None or y is None: return 'n/a'
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q: return i
    return None if len(x) == len(y) else min(len(x), len(y))
P('\nD3 correctness (16 requests, 1 pass, greedy) vs e2-fi-a1: first divergence token index (None = identical)')
f = runs / 'e2-fi-a1' / 'd3.json'
if f.exists():
    A = {(r['kind'], r['index']): r for r in json.load(open(f))['passes'][0]}
    for n in ARMS[1:]:
        g = runs / n / 'd3.json'
        if not g.exists(): P(f'- {n}: no d3'); continue
        B = {(r['kind'], r['index']): r for r in json.load(open(g))['passes'][0]}
        rows = [(k, fd(A[k]['token_ids'], B[k]['token_ids']), len(A[k]['token_ids'] or []), len(B[k]['token_ids'] or [])) for k in A if k in B]
        same = sum(1 for _, d, _, _ in rows if d is None)
        P(f'- {n}: identical {same}/{len(rows)}; ' + '; '.join(f'{k[0]}#{k[1]}:{d}' for k, d, _, _ in rows if d is not None))
(out / 'e2-analysis.md').write_text('\n'.join(L)); print('\n'.join(L))
