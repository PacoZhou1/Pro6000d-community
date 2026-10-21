"""Offline A1/B/A2 for decode + thinking (profile_client jsonl).  usage: analyze_d4.py runs_dir out.json [a1 b a2 labels]"""
import json, statistics as st, sys, pathlib
root = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
labels = sys.argv[3:6] or ['d4-a1', 'd4-b', 'd4-a2']
EMB = ('vllm:request_success_total', 'vllm:prompt_tokens_total', 'vllm:request_prompt_tokens_count')
def load(l):
    return [json.loads(x) for x in (root / l / 'perf.jsonl').read_text().splitlines() if x.strip()]
def emb(r):
    b, a = r['embedding_before'], r['embedding_after']
    ch = {k: a[k] - b[k] for k in b if k in a and a[k] != b[k]}
    return ch
data = {l: load(l) for l in labels}
res = {'labels': labels, 'cases': {}}
for case in ('decode', 'thinking'):
    per = {}
    for l in labels:
        rows = [r for r in data[l] if r['case'] == case and not r['warmup']]
        per[l] = rows
    s = {}
    for l, rows in per.items():
        def m(k): return [r['metrics_delta'].get(k, 0) for r in rows]
        tps = [r['decode_estimate_tps'] for r in rows]
        dr, ac, nd = m('vllm:spec_decode_num_draft_tokens_total'), m('vllm:spec_decode_num_accepted_tokens_total'), m('vllm:spec_decode_num_drafts_total')
        s[l] = {'n': len(rows), 'tps': tps, 'tps_median': st.median(tps), 'tps_mean': st.mean(tps),
                'accept_rate': sum(ac) / sum(dr), 'mean_accept_len': sum(ac) / sum(nd), 'tokens_per_step': sum(ac) / sum(nd) + 1,
                'step_ms_est': 1000 / (st.median(tps) / (sum(ac) / sum(nd) + 1)),
                'valid_all': all(r['valid'] for r in rows),
                'payload_hashes': [r['payload_sha256'][:12] for r in rows],
                'embedding_counter_changes': [emb(r) for r in rows],
                'content_sha': [r['content_sha256'][:12] for r in rows], 'reasoning_sha': [r['reasoning_sha256'][:12] for r in rows]}
    a1, b, a2 = labels
    same = all(s[a1]['payload_hashes'] == s[x]['payload_hashes'] for x in (b, a2))
    paired = []
    for i in range(len(s[b]['tps'])):
        base = (s[a1]['tps'][i] + s[a2]['tps'][i]) / 2
        paired.append(s[b]['tps'][i] / base - 1)
    drift = [s[a2]['tps'][i] / s[a1]['tps'][i] - 1 for i in range(len(s[b]['tps']))]
    res['cases'][case] = {'arms': s, 'payload_hash_identical_all_arms': same,
        'paired_change_B_vs_meanA': paired, 'paired_median': st.median(paired), 'paired_mean': st.mean(paired),
        'paired_range': [min(paired), max(paired)], 'ab_drift_A2_vs_A1_median': st.median(drift),
        'median_tps_change_B_vs_meanA': s[b]['tps_median'] / ((s[a1]['tps_median'] + s[a2]['tps_median']) / 2) - 1,
        'content_identical_A1_A2': s[a1]['content_sha'] == s[a2]['content_sha'],
        'content_identical_B_A1': s[b]['content_sha'] == s[a1]['content_sha']}
pooled = []
for i in range(6):
    pass
allp = res['cases']['decode']['paired_change_B_vs_meanA'] + res['cases']['thinking']['paired_change_B_vs_meanA']
res['pooled_paired_median'] = st.median(allp); res['pooled_paired_mean'] = st.mean(allp)
out.write_text(json.dumps(res, indent=1))
for c, v in res['cases'].items():
    print(c, {l: round(a['tps_median'], 2) for l, a in v['arms'].items()}, 'paired median %.2f%%' % (100 * v['paired_median']),
          'drift %.2f%%' % (100 * v['ab_drift_A2_vs_A1_median']), {l: round(a['mean_accept_len'], 3) for l, a in v['arms'].items()})
print('pooled paired median %.2f%%' % (100 * res['pooled_paired_median']))
