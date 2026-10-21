#!/usr/bin/env python3
"""Offline validation and descriptive statistics for one combined-image run."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


CASES = {'prefill': 12, 'decode': 6, 'thinking': 6}
ACCEPTANCE = {(case, mode, repeat)
              for case in ('normal_long', 'thinking_long', 'json_object', 'tool_arguments')
              for mode in ('nonstream', 'sse') for repeat in (1, 2)}
EMBED_KEYS = ('vllm:request_success_total', 'vllm:prompt_tokens_total',
              'vllm:request_prompt_tokens_count')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    if not path.is_file():
        raise ValueError(f'missing input: {path}')
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def number(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not positive or value > 0))


def summary(values):
    values = [float(x) for x in values if number(x)]
    return {'n': len(values), 'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def embedding_status(row):
    before, after = row.get('embedding_before'), row.get('embedding_after')
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or set(before) != set(after) or not set(EMBED_KEYS) <= set(before)):
        return {'known': False, 'quiet': False, 'changed': None}
    changed = {key: [before[key], after[key]] for key in before if before[key] != after[key]}
    return {'known': True, 'quiet': not changed, 'changed': changed}


def perf_reasons(row, case, index):
    usage, delta = row.get('usage') or {}, row.get('metrics_delta') or {}
    reasons = []
    if row.get('case') != case or row.get('index') != index or row.get('warmup') is not (index == 0):
        reasons.append('case_index_or_warmup_mismatch')
    if row.get('valid') is not True:
        reasons.append('client_valid_false')
    if row.get('profiled') is not False or row.get('timing_scope') != 'no profiler installed':
        reasons.append('profiler_or_timing_scope')
    if row.get('stream_done') is not True or row.get('finish_reason') != 'length':
        reasons.append('fixed_output_incomplete')
    expected_completion = 1 if case == 'prefill' else (32 if index == 0 else 1024)
    if case == 'prefill':
        expected_completion = 1
    if usage.get('completion_tokens') != expected_completion or not number(usage.get('prompt_tokens'), True):
        reasons.append('usage_or_expected_completion')
    if delta.get('vllm:request_success_total') != 1:
        reasons.append('target_request_counter')
    if delta.get('vllm:prompt_tokens_total') != usage.get('prompt_tokens'):
        reasons.append('prompt_counter')
    if delta.get('vllm:generation_tokens_total') != usage.get('completion_tokens'):
        reasons.append('generation_counter')
    if delta.get('vllm:prefix_cache_hits_total') != 0:
        reasons.append('prefix_cache_hit_or_unknown')
    embedded = embedding_status(row)
    if not embedded['known'] or not embedded['quiet']:
        reasons.append('embedding_active_or_unknown')
    if not isinstance(row.get('payload_sha256'), str) or len(row['payload_sha256']) != 64:
        reasons.append('payload_hash_missing')
    if not number(row.get('first_text_s'), True) or not number(row.get('total_s'), True):
        reasons.append('latency_missing')
    if case != 'prefill' and index > 0 and not number(row.get('decode_estimate_tps'), True):
        reasons.append('decode_tps_missing')
    if any(not isinstance(row.get(key), str) or len(row[key]) != 64
           for key in ('content_sha256', 'reasoning_sha256')):
        reasons.append('output_hash_missing')
    return reasons, embedded


def performance(root):
    result, hashes, all_pass = {}, {}, True
    for case, expected_n in CASES.items():
        path = root / f'{case}.jsonl'
        hashes[case] = file_hash(path)
        rows = read_rows(path)
        by_index, structural = {}, []
        for row in rows:
            index = row.get('index')
            if not isinstance(index, int) or isinstance(index, bool) or index in by_index:
                structural.append({'reason': 'duplicate_or_invalid_index', 'index': index})
            else:
                by_index[index] = row
        expected = set(range(expected_n + 1))
        if set(by_index) != expected:
            structural.append({'reason': 'index_matrix_incomplete',
                               'missing': sorted(expected - set(by_index)),
                               'extra': sorted(set(by_index) - expected)})
        records = []
        for index in sorted(by_index):
            row = by_index[index]
            reasons, embedded = perf_reasons(row, case, index)
            delta = row.get('metrics_delta') or {}
            records.append({'index': index, 'warmup': index == 0, 'valid': not reasons,
                            'reasons': reasons, 'payload_sha256': row.get('payload_sha256'),
                            'usage': row.get('usage'), 'first_text_s': row.get('first_text_s'),
                            'total_s': row.get('total_s'),
                            'decode_estimate_tps': row.get('decode_estimate_tps'),
                            'content_sha256': row.get('content_sha256'),
                            'reasoning_sha256': row.get('reasoning_sha256'),
                            'accepted_tokens': delta.get('vllm:spec_decode_num_accepted_tokens_total'),
                            'draft_tokens': delta.get('vllm:spec_decode_num_draft_tokens_total'),
                            'prefix_cache_hits': delta.get('vllm:prefix_cache_hits_total'),
                            'embedding': embedded})
        formal = [r for r in records if r['index'] > 0 and r['valid']]
        valid_all = not structural and len(formal) == expected_n and all(r['valid'] for r in records)
        all_pass = all_pass and valid_all
        result[case] = {'expected_formal_n': expected_n, 'valid_formal_n': len(formal),
                        'all_requests_valid': valid_all, 'structural_errors': structural,
                        'first_text_s': summary(r['first_text_s'] for r in formal),
                        'total_s': summary(r['total_s'] for r in formal),
                        'decode_estimate_tps': summary(r['decode_estimate_tps'] for r in formal),
                        'prompt_tokens': summary((r['usage'] or {}).get('prompt_tokens') for r in formal),
                        'completion_tokens': summary((r['usage'] or {}).get('completion_tokens') for r in formal),
                        'accepted_tokens': summary(r['accepted_tokens'] for r in formal),
                        'draft_tokens': summary(r['draft_tokens'] for r in formal),
                        'records': records}
    return result, hashes, all_pass


def natural(path):
    rows = read_rows(path)
    metas = [r for r in rows if r.get('kind') == 'meta']
    completes = [r for r in rows if r.get('kind') == 'complete']
    indexed, structural = {}, []
    for row in rows:
        if row.get('kind') != 'request':
            continue
        key = (row.get('case'), row.get('mode'), row.get('repeat'))
        if key in indexed:
            structural.append({'reason': 'duplicate_acceptance_request', 'key': key})
        indexed[key] = row
    if len(metas) != 1 or len(completes) != 1:
        structural.append({'reason': 'meta_or_complete_count', 'meta': len(metas),
                           'complete': len(completes)})
    if set(indexed) != ACCEPTANCE:
        structural.append({'reason': 'acceptance_matrix_incomplete',
                           'missing': [list(k) for k in sorted(ACCEPTANCE - set(indexed))],
                           'extra': [list(k) for k in sorted(set(indexed) - ACCEPTANCE)]})
    records = []
    for key, row in sorted(indexed.items()):
        usage = row.get('usage') or {}
        flags = row.get('loop_flags') or {}
        reasons = list(row.get('failure_reasons') or [])
        if row.get('observed_valid') is not True or row.get('normal_eos_or_tool_stop') is not True:
            reasons.append('natural_completion_invalid')
        if row.get('fixture_checks_pass') is not True:
            reasons.append('fixture_checks_failed')
        if row.get('truncated') is not False:
            reasons.append('truncated_or_unknown')
        if flags.get('obvious_loop_suspect') is not False:
            reasons.append('loop_suspect_or_unknown')
        if not number(usage.get('completion_tokens'), True):
            reasons.append('completion_usage_missing')
        if not isinstance(row.get('payload_sha256'), str) or len(row['payload_sha256']) != 64:
            reasons.append('payload_hash_missing')
        if any(not isinstance(row.get(field), str) or len(row[field]) != 64
               for field in ('content_sha256', 'reasoning_sha256')):
            reasons.append('output_hash_missing')
        prior = indexed.get((key[0], key[1], 1)) if key[2] == 2 else None
        if prior is not None and row.get('payload_sha256') != prior.get('payload_sha256'):
            reasons.append('repeat_payload_mismatch')
        if row.get('acceptance_pass') is not True:
            reasons.append('client_acceptance_failed')
        records.append({'case': key[0], 'mode': key[1], 'repeat': key[2],
                        'valid': not reasons, 'reasons': sorted(set(reasons)),
                        'payload_sha256': row.get('payload_sha256'),
                        'finish_reason': row.get('finish_reason'), 'usage': usage,
                        'content_sha256': row.get('content_sha256'),
                        'reasoning_sha256': row.get('reasoning_sha256'),
                        'token_ids_status': row.get('token_ids_status'),
                        'token_ids_available': row.get('token_ids_available'),
                        'truncated': row.get('truncated'), 'loop_flags': flags,
                        'fixture_checks_pass': row.get('fixture_checks_pass'),
                        'same_repeat_content': row.get('content_sha256') == prior.get('content_sha256') if prior else None,
                        'same_repeat_reasoning': row.get('reasoning_sha256') == prior.get('reasoning_sha256') if prior else None})
    complete_ok = len(completes) == 1 and completes[0].get('all_acceptance_pass') is True
    all_pass = not structural and len(records) == 16 and all(r['valid'] for r in records) and complete_ok
    return {'all_pass': all_pass, 'expected_n': 16, 'valid_n': sum(r['valid'] for r in records),
            'structural_errors': structural, 'client_complete_pass': complete_ok,
            'token_ids_available_n': sum(r['token_ids_available'] is True for r in records),
            'loop_suspect_n': sum((r['loop_flags'] or {}).get('obvious_loop_suspect') is True for r in records),
            'truncated_n': sum(r['truncated'] is True for r in records),
            'same_repeat_content_n': sum(r['same_repeat_content'] is True for r in records),
            'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='Combined B performance arm directory')
    parser.add_argument('--acceptance', required=True, type=Path, help='B smoke acceptance.jsonl')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('fresh output required; keep original evidence immutable')
    perf, hashes, perf_ok = performance(args.root)
    acceptance = natural(args.acceptance)
    result = {'scope': 'one combined image only; no paired baseline or causal gain estimate',
              'source_sha256': {**hashes, 'acceptance': file_hash(args.acceptance)},
              'performance': perf, 'acceptance': acceptance,
              'all_checks_pass': perf_ok and acceptance['all_pass'],
              'limitations': ['Synthetic prompts do not establish broad model quality.',
                              'Performance client runs fixed-output ignore_eos requests; acceptance separately observes natural EOS.',
                              'The performance client checks idle before each request but its row does not store a separate busy-before field.',
                              'Historical 0.27/0.31/draft64 results have different sessions and some different payloads; no paired gain is computed.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'all_checks_pass': result['all_checks_pass'],
                      'performance_valid_n': {case: perf[case]['valid_formal_n'] for case in CASES},
                      'acceptance_valid_n': acceptance['valid_n']}, ensure_ascii=False))
    if not result['all_checks_pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
