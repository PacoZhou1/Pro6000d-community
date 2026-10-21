"""Compare actual unprofiled API arms; retain output and interference limits."""
import hashlib
import json
import statistics
from pathlib import Path

root = Path(__file__).parent
arms = ['api-baseline', 'api-streamk-v2', 'api-baseline-repeat']
raw = {}
summary = {'scope': 'C1 synthetic fixed prompts, normal CUDA graphs, no profiler installed',
           'arms': {}, 'pairs': {}}
for arm in arms:
    folder = root / arm
    requests = [json.loads(l) for l in (folder / 'performance.jsonl').read_text().splitlines()]
    assert len(requests) == 12 and all(r['valid'] and not r['profiled'] for r in requests)
    runtime = json.loads((folder / 'runtime.json').read_text())
    groups = {}
    for case in ['prefill', 'decode', 'thinking']:
        rows = [r for r in requests if r['case'] == case and not r['warmup']]
        assert len(rows) == 3
        metric = lambda key: statistics.median(r['metrics_delta'][key] for r in rows)
        groups[case] = {
            'n': len(rows),
            'ttft_median_s': statistics.median(r['first_text_s'] for r in rows),
            'server_prefill_median_s': metric('vllm:request_prefill_time_seconds_sum'),
            'total_median_s': statistics.median(r['total_s'] for r in rows),
            'embedding_counts_unchanged': all(r['embedding_before'] == r['embedding_after'] for r in rows),
            'prefix_hits': [r['metrics_delta'].get('vllm:prefix_cache_hits_total') for r in rows],
            'prompt_tokens': [r['usage']['prompt_tokens'] for r in rows],
            'completion_tokens': [r['usage']['completion_tokens'] for r in rows],
            'reasoning_tokens': [r['usage'].get('completion_tokens_details', {}).get('reasoning_tokens') for r in rows],
        }
        if case != 'prefill':
            tps = [r['decode_estimate_tps'] for r in rows]
            groups[case].update(decode_median_tps=statistics.median(tps), decode_tps=tps,
                                accepted_per_draft=[
                                    r['metrics_delta']['vllm:spec_decode_num_accepted_tokens_total'] /
                                    r['metrics_delta']['vllm:spec_decode_num_drafts_total']
                                    for r in rows])
    quality = {}
    for name in ['quality', 'open-output']:
        rows = [json.loads(l) for l in (folder / (name + '.jsonl')).read_text().splitlines()]
        quality[name] = {'passed': sum(r['passed'] for r in rows), 'total': len(rows),
                         'scope': 'synthetic API checks only; not MoBase or broad quality'}
    raw[arm] = {(r['case'], r['index']): r for r in requests if not r['warmup']}
    summary['arms'][arm] = {'cases': groups, 'quality': quality, 'image': runtime['Image'],
                            'performance_sha256': hashlib.sha256((folder / 'performance.jsonl').read_bytes()).hexdigest()}

for baseline in ['api-baseline', 'api-baseline-repeat']:
    rows = []
    for key, b in raw[baseline].items():
        c = raw['api-streamk-v2'][key]
        assert b['payload_sha256'] == c['payload_sha256']
        rows.append({'case': key[0], 'index': key[1], 'same_payload': True,
                     'content_equal': b['content_sha256'] == c['content_sha256'],
                     'reasoning_equal': b['reasoning_sha256'] == c['reasoning_sha256'],
                     'ttft_change_pct': (c['first_text_s'] / b['first_text_s'] - 1) * 100,
                     'decode_tps_change_pct': None if key[0] == 'prefill' else
                         (c['decode_estimate_tps'] / b['decode_estimate_tps'] - 1) * 100})
    cmd_b = json.loads((root / baseline / 'runtime.json').read_text())['cmd']
    cmd_c = json.loads((root / 'api-streamk-v2/runtime.json').read_text())['cmd']
    assert cmd_b == cmd_c
    summary['pairs'][baseline + '_vs_candidate'] = {'same_server_args': True, 'requests': rows}

summary['limits'] = [
    'Three samples per workload per arm; A-B-A restart sequence, not a randomized sustained-load trial.',
    'Streaming token rate uses usage tokens and first/last text timestamps, not SSE chunk count.',
    'Different output hashes/acceptance ratios can affect timings; equal prompts are not equal generated work.',
    '1024-token thinking may contain only reasoning and cannot establish completed-task throughput.',
]
(root / 'api-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({a: {k: (v['ttft_median_s'] if k == 'prefill' else v['decode_median_tps'])
                          for k, v in d['cases'].items()} for a, d in summary['arms'].items()}))
