"""Offline E1 analysis: usage analyze_e1.py RUNS_DIR OUT_DIR   (reads runs/*/{pfx,gen,split}.json and prefill.jsonl)"""
import json, statistics as st, sys, pathlib, re
runs = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2]); out.mkdir(exist_ok=True, parents=True)
def jl(p):
    return [json.loads(l) for l in open(p) if l.strip().startswith('{')] if p.exists() else []
def fd(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q: return i
    return None if len(x) == len(y) else min(len(x), len(y))
L = []; P = L.append
# ---- prefill TTFT per arm (profile_client prefill, 6 cold requests)
P('## prefill cold TTFT (profile_client prefill, 6 requests, same prompts per index)')
arms = {}
for d in sorted(runs.iterdir()):
    rows = [r for r in jl(d / 'prefill.jsonl') if not r.get('warmup')]
    if rows: arms[d.name] = rows
for n, rows in arms.items():
    t = [r['first_text_s'] for r in rows]
    P(f"- {n}: n={len(t)} median {st.median(t):.4f} s, range {min(t):.4f}-{max(t):.4f}, prompt_tokens {rows[0]['usage']['prompt_tokens']}, hits {sum(r['metrics_delta'].get('vllm:prefix_cache_hits_total',0) for r in rows)}")
def pair(a, b):
    A = {r['index']: r['first_text_s'] for r in arms[a]}; B = {r['index']: r['first_text_s'] for r in arms[b]}
    return [100 * (B[i] - A[i]) / A[i] for i in A if i in B]
if all(k in arms for k in ('e1-base1', 'e1-nopc', 'e1-base2')):
    ref = {i: (x + y) / 2 for i, x, y in [(r['index'], r['first_text_s'], next(q['first_text_s'] for q in arms['e1-base2'] if q['index'] == r['index'])) for r in arms['e1-base1']]}
    nop = {r['index']: r['first_text_s'] for r in arms['e1-nopc']}
    d = [100 * (nop[i] - ref[i]) / ref[i] for i in ref]
    drift = pair('e1-base1', 'e1-base2')
    P(f"- no-prefix-caching vs mean(base1,base2): paired median {st.median(d):+.3f}% (range {min(d):+.3f}..{max(d):+.3f}); base1->base2 drift median {st.median(drift):+.3f}%")
# ---- chunk splits from instr arms
P('\n## scheduler chunking (SPLITPROBE in server.log)')
for d in sorted(runs.iterdir()):
    f = d / 'server.log'
    if f.exists():
        rows = [l for l in f.read_text(errors='ignore').splitlines() if 'SPLITPROBE' in l]
        if rows:
            seen = {}
            for l in rows:
                m = dict(re.findall(r'(\w+)=(\S+)', l))
                if int(m['asked']) > 5 or int(m['computed']) < int(m['prompt']):   # prefill chunks only
                    seen.setdefault(int(m['prompt']), []).append((int(m['computed']), int(m['asked']), int(m['got'])))
            P(f'### {d.name}')
            for pr, ch in sorted(seen.items()):
                P(f'- prompt {pr}: ' + ' + '.join(f'{g} (asked {a}, start {c})' for c, a, g in ch))
# ---- iteration histogram based split (non-instr arms)
for d in sorted(runs.iterdir()):
    j = d / 'split.json'
    if j.exists():
        P(f'### {d.name} iteration_tokens deltas (count/sum per request)')
        for r in json.load(open(j))['records']:
            P(f"- L={r['L']} prompt={r['prompt_tokens']} steps={r['iter_count']:.0f} sum={r['iter_sum']:.0f} ttft={r['ttft_s']:.3f}")
# ---- pfx scenarios
for d in sorted(runs.iterdir()):
    j = d / 'pfx.json'
    if not j.exists(): continue
    D = json.load(open(j)); R = D['records']
    P(f'\n## {d.name}: prefix scenarios')
    P('| scen | request | prompt tok | hit tok | recomputed | TTFT s | role |'); P('|---|---|---:|---:|---:|---:|---|')
    for r in R:
        if r['scen'] == 'S4': continue
        P(f"| {r['scen']} | {r['name']} | {r['prompt_tokens']} | {r['hit_tokens']:.0f} | {r['prompt_tokens'] - r['hit_tokens']:.0f} | {r['ttft_s']:.3f} | {r.get('role','')} |")
    s4 = [r for r in R if r['scen'] == 'S4']
    if s4:
        P('\n### S4 sweep (shared prefix P -> hit tokens; test prompt = P + ~1000 new tokens)')
        P('| seed L | P | P mod 1616 | hit | hit/1616 | recomputed | TTFT s |'); P('|---:|---:|---:|---:|---:|---:|---:|')
        for r in s4:
            if r['role'] == 'hit':
                P(f"| {r['L']} | {r['P']} | {r['P'] % 1616} | {r['hit_tokens']:.0f} | {r['hit_tokens'] / 1616:.2f} | {r['prompt_tokens'] - r['hit_tokens']:.0f} | {r['ttft_s']:.3f} |")
    P('\n### correctness gate (greedy 256 tokens; hit vs salted-cold)')
    P('| request | hit tok | first div hit-vs-coldA | coldA-vs-coldB | margin(top1-top2) at div, cold / hit |'); P('|---|---:|---|---|---|')
    byname = {r['name']: r for r in R}
    for c in D.get('comparisons', []):
        div = c.get('hit_vs_cold_a_first_div', c.get('hit_vs_cold_first_div'))
        marg = ''
        if div is not None:
            hit = byname.get(c['name']); cold = byname.get(c['name'] + '/cold-a') or byname.get(c['name'].rsplit('/', 1)[0] + '/cold')
            def m(r):
                try:
                    t = r['logprobs'][div][1]; return f"{t[0][1] - t[1][1]:.3f}"
                except Exception: return 'n/a'
            if hit and cold: marg = f"{m(cold)} / {m(hit)}"
        P(f"| {c['scen']} {c['name']} | {c['hit_tokens']:.0f} | {div} | {c.get('cold_a_vs_cold_b_first_div','-')} | {marg} |")
# ---- gen noise floor across arms
G = {d.name: json.load(open(d / 'gen.json'))['records'] for d in sorted(runs.iterdir()) if (d / 'gen.json').exists()}
if len(G) > 1:
    P('\n## chunk-structure noise floor (same 6 cold prompts, greedy 256): first divergence index vs e1-base1')
    base = G.get('e1-base1')
    for n, rr in G.items():
        if n == 'e1-base1' or not base: continue
        P(f"- {n}: " + ', '.join(str(fd(a['token_ids'], b['token_ids'])) for a, b in zip(base, rr)))
(out / 'e1-analysis.md').write_text('\n'.join(L)); print('\n'.join(L))
