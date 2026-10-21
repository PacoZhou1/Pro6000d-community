"""P1 prefix-reuse scenarios S1-S4 (+ cold-salt correctness gate). Runs on the host against the owned arm.
Every record: TTFT, prompt tokens, server hit-token delta, queries delta, token ids.  cache_salt isolates 'cold' re-runs
(salt is hashed into the first block, so no block/Mamba snapshot can be shared with unsalted or other-salted requests)."""
import argparse, json, random, sys, time
from hc import *

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--out', required=True)
p.add_argument('--scen', default='S1,S2,S3,S4')
p.add_argument('--lens', default='19000,44000')
p.add_argument('--max-new', type=int, default=256)
p.add_argument('--s4-limit', type=int, default=0)
p.add_argument('--shuffle', action='store_true', help='add cold-c after an unrelated filler (free-block-order control)')
a = p.parse_args()
SCEN = a.scen.split(','); LENS = [int(x) for x in a.lens.split(',')]
res = {'tag': a.tag, 'records': [], 'notes': {}}
RUN = time.strftime('%H%M%S')
seq = [0]

def save(): json.dump(res, open(a.out, 'w'), ensure_ascii=False)

def rec(scen, name, payload, path='/v1/chat/completions', **extra):
    r = stream(payload, path)
    seq[0] += 1
    r.update(scen=scen, name=name, seq=seq[0], salt=payload.get('cache_salt'), **extra)
    res['records'].append(r); save()
    print(json.dumps({k: r[k] for k in ('scen', 'name', 'prompt_tokens', 'hit_tokens', 'queries', 'ttft_s', 'completion_tokens', 'iter_count')}), flush=True)
    return r

def sysmsg(label): return {'role': 'system', 'content': label + ' Synthetic prefix-reuse probe. Explain in clear Chinese prose.'}

def cold_pair(scen, name, payload, hit_rec, path='/v1/chat/completions'):
    """two salted cold runs of the identical request + compare against hit_rec; returns comparison dict"""
    outs = []
    for k in ('a', 'b', 'c'):
        if k == 'c' and not a.shuffle: break
        if k == 'c':   # control: unrelated salted filler between cold-b and cold-c reshuffles the free-block order
            t, _ = text_for(f'filler-{RUN}-{seq[0]}', 3000)
            stream(chat_payload([sysmsg('filler'), {'role': 'user', 'content': t}], 1, salt=f'filler-{RUN}-{seq[0]}'))
        pl = dict(payload); pl['cache_salt'] = f'cold-{RUN}-{seq[0]}-{k}'
        outs.append(rec(scen, name + f'/cold-{k}', pl, path, role='cold'))
    cmpd = {'scen': scen, 'name': name, 'hit_vs_cold_a_first_div': first_div(hit_rec['token_ids'], outs[0]['token_ids']),
            'hit_vs_cold_b_first_div': first_div(hit_rec['token_ids'], outs[1]['token_ids']),
            'cold_a_vs_cold_b_first_div': first_div(outs[0]['token_ids'], outs[1]['token_ids']),
            'cold_a_vs_cold_c_first_div': first_div(outs[0]['token_ids'], outs[2]['token_ids']) if len(outs) > 2 else 'n/a',
            'ttft_cold_c_s': outs[2]['ttft_s'] if len(outs) > 2 else None,
            'n_tokens': [len(hit_rec['token_ids']), len(outs[0]['token_ids']), len(outs[1]['token_ids'])],
            'hit_tokens': hit_rec['hit_tokens'], 'prompt_tokens': hit_rec['prompt_tokens'],
            'ttft_hit_s': hit_rec['ttft_s'], 'ttft_cold_a_s': outs[0]['ttft_s'], 'ttft_cold_b_s': outs[1]['ttft_s']}
    res.setdefault('comparisons', []).append(cmpd); save()
    print('CMP', json.dumps(cmpd), flush=True)
    return cmpd

# ---- warm-up (not recorded): compile/JIT shapes for 19K and 44K
for L in (19000, 44000):
    t, _ = text_for(f'warm-{L}', L - 120)
    stream(chat_payload([sysmsg(f'warm-{L}'), {'role': 'user', 'content': t + QUESTION}], 1, salt=f'warm-{RUN}-{L}'))

def user_prompt(label, L, extra=''):
    t, c = text_for(label, L - 120)
    return [sysmsg(label), {'role': 'user', 'content': t + QUESTION + extra}]

# ---------------- S1 exact repeat
if 'S1' in SCEN:
    for L in LENS:
        for s in range(2):
            lab = f'S1-{RUN}-{L}-{s}'
            pl = chat_payload(user_prompt(lab, L), a.max_new)
            cold = rec('S1', f'{L}/s{s}/cold', pl, role='cold', L=L)
            for k in (1, 2):
                h = rec('S1', f'{L}/s{s}/repeat{k}', pl, role='hit', L=L)
                res.setdefault('comparisons', []).append({'scen': 'S1', 'name': f'{L}/s{s}/repeat{k}', 'hit_vs_cold_first_div': first_div(h['token_ids'], cold['token_ids']),
                      'hit_tokens': h['hit_tokens'], 'prompt_tokens': h['prompt_tokens'], 'ttft_hit_s': h['ttft_s'], 'ttft_cold_s': cold['ttft_s']})
            save()

# ---------------- S2 shared prefix + new tail
if 'S2' in SCEN:
    plan = {19000: (15000, [1000, 2000, 4000]), 44000: (40000, [1000, 4000])}
    for L in LENS:
        sh_n, tails = plan[L]
        for s in range(2):
            lab = f'S2-{RUN}-{L}-{s}'
            shared, shc = text_for(lab + '-shared', sh_n)
            def mk(tn, tag):
                tt, tc = text_for(lab + '-tail-' + tag, tn)
                return [sysmsg(lab), {'role': 'user', 'content': shared + '\n\n' + tt + QUESTION}], tc
            seed_m, _ = mk(max(tails), 'seed')
            seed = rec('S2', f'{L}/s{s}/seed(cold)', chat_payload(seed_m, a.max_new), role='seed', L=L)
            seed_ids = chat_tokens(seed_m)
            for tn in (tails if s == 0 else tails[::-1]):   # s0: ascending tails, s1: descending (order decides which chunk-end snapshots exist)
                m, tc = mk(tn, f'v{tn}')
                ids = chat_tokens(m)
                pl = chat_payload(m, a.max_new)
                h = rec('S2', f'{L}/s{s}/tail{tn}', pl, role='hit', L=L, tail_tokens=tc, ideal_lcp_vs_seed=lcp(ids, seed_ids), prompt_ids_len=len(ids))
                cold_pair('S2', f'{L}/s{s}/tail{tn}', pl, h)

# ---------------- S3 multi-turn chat
if 'S3' in SCEN:
    for L in LENS:
        for s in range(2):
            lab = f'S3-{RUN}-{L}-{s}'
            msgs = user_prompt(lab, L)
            followups = ['请用一句话总结上面你的回答。', '再用三个要点列出最需要优先处理的事项。']
            prev_ids = chat_tokens(msgs)
            for turn in (1, 2, 3):
                pl = chat_payload(msgs, a.max_new)
                ids = chat_tokens(msgs)
                h = rec('S3', f'{L}/s{s}/turn{turn}', pl, role='turn%d' % turn, L=L, prompt_ids_len=len(ids), ideal_lcp_vs_prev_prompt=lcp(ids, prev_ids))
                if turn >= 2: cold_pair('S3', f'{L}/s{s}/turn{turn}', pl, h)
                prev_ids = ids
                if turn < 3:
                    msgs = msgs + [{'role': 'assistant', 'content': h['content']}, {'role': 'user', 'content': followups[turn - 1]}]
                    # prompt of the *next* turn begins with this turn's prompt + the assistant's generated tokens

# ---------------- S4 granularity sweep (token-id prompts via /v1/completions, isolated by cache_salt)
if 'S4' in SCEN:
    B = 1616
    sweeps = {19000: sorted(set([1, 800, 1615, 1616, 1617] + [k * B + d for k in range(2, 12) for d in (-1, 0, 1)] + [15000, 18000, 18900, 19000])),
              44000: sorted(set([1616, 3232, 8080, 15000] + [k * B + d for k in (10, 15, 19, 20, 21, 25, 27) for d in (-1, 0, 1)] + [30000, 40000, 43000, 44000]))}
    for L in LENS:
        plist = sweeps.get(L, [])
        if a.s4_limit: plist = plist[:: max(1, len(plist) // a.s4_limit)][:a.s4_limit]
        for P in plist:
            lab = f'S4-{RUN}-{L}-{P}'
            seed_ids = tokens_of(items(lab + '-seed', int(L / 5.9) + 400))[:L]
            tail_all = tokens_of(items(lab + '-tail', 400))
            tail = [t for t in tail_all]
            while tail and P < len(seed_ids) and tail[0] == seed_ids[P]: tail.pop(0)
            tail = tail[:1000]
            salt = f'S4-{RUN}-{L}-{P}'
            def comp(ids, name, role):
                return rec('S4', name, {'model': MODEL, 'prompt': ids, 'temperature': 0, 'max_tokens': 1, 'cache_salt': salt}, '/v1/completions',
                           role=role, L=L, P=P, prompt_ids_len=len(ids))
            sd = comp(seed_ids, f'{L}/P{P}/seed', 'seed')
            test = comp(seed_ids[:P] + tail, f'{L}/P{P}/test', 'hit')
            test['reuse_granularity_note'] = {'P': P, 'hit': test['hit_tokens'], 'P_mod_1616': P % B, 'hit_mod_1616': test['hit_tokens'] % B}
save()
print('PFX_DONE', flush=True)
