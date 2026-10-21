#!/usr/bin/env python3
"""Offline O1/A1/B/A2/O2 FP8 combination comparison; never touches services."""
import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

ARMS=('o1','a1','b','a2','o2')
CASES={'prefill':12,'decode':6,'thinking':6}
ACCEPTANCE={(c,m,r) for c in ('normal_long','thinking_long','json_object','tool_arguments')
            for m in ('nonstream','sse') for r in (1,2)}
EMBED_KEYS=('vllm:request_success_total','vllm:prompt_tokens_total','vllm:request_prompt_tokens_count')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    if not path.is_file():raise ValueError(f'missing {path}')
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def stats(values):
    vals=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {'n':len(vals),'median':statistics.median(vals) if vals else None,
            'min':min(vals) if vals else None,'max':max(vals) if vals else None}


def quiet(row):
    a=row.get('embedding_before');b=row.get('embedding_after')
    if not isinstance(a,dict) or not isinstance(b,dict) or set(a)!=set(b) or not set(EMBED_KEYS)<=set(a):
        return {'known':False,'quiet':False,'changed':None}
    changed={k:b[k]-a[k] for k in a if b[k]!=a[k]}
    return {'known':True,'quiet':not changed,'changed':changed}


def perf_valid(row,case,index):
    usage=row.get('usage') or {};delta=row.get('metrics_delta') or {}
    errors=[]
    if row.get('case')!=case or row.get('index')!=index or row.get('warmup')!=(index==0):errors.append('case/index/warmup')
    if row.get('valid') is not True or row.get('profiled') is not False:errors.append('valid/profiled')
    if row.get('timing_scope')!='no profiler installed':errors.append('profiler_scope')
    if row.get('stream_done') is not True or row.get('finish_reason')!='length':errors.append('stream_or_length')
    if delta.get('vllm:request_success_total')!=1:errors.append('target_request_count')
    if delta.get('vllm:prompt_tokens_total')!=usage.get('prompt_tokens'):errors.append('prompt_counter')
    if delta.get('vllm:generation_tokens_total')!=usage.get('completion_tokens'):errors.append('generation_counter')
    if not isinstance(row.get('payload_sha256'),str) or len(row['payload_sha256'])!=64:errors.append('payload_hash')
    if not isinstance(row.get('first_text_s'),(int,float)) or row['first_text_s']<=0:errors.append('first_text')
    if case in ('decode','thinking') and index>0 and not isinstance(row.get('decode_estimate_tps'),(int,float)):
        errors.append('decode_tps')
    return errors


def load_arm(root,arm,acceptance_name):
    out={'source_hashes':{},'perf':{},'acceptance':{}}
    for case,n in CASES.items():
        path=root/arm/f'{case}.jsonl';out['source_hashes'][case]=sha(path)
        rows=read_jsonl(path)
        by={}
        for row in rows:
            i=row.get('index')
            if not isinstance(i,int) or i in by:raise ValueError(f'{path}: duplicate/invalid index {i}')
            by[i]=row
        if set(by)!=set(range(n+1)):raise ValueError(f'{path}: expected warmup 0 and formal 1..{n}')
        out['perf'][case]=by
    path=root/arm/acceptance_name;out['source_hashes']['acceptance']=sha(path)
    rows=read_jsonl(path)
    metas=[x for x in rows if x.get('kind')=='meta']
    completes=[x for x in rows if x.get('kind')=='complete']
    if len(metas)!=1 or len(completes)!=1:raise ValueError(f'{path}: meta/complete count')
    out['acceptance_meta']=metas[0];out['acceptance_complete']=completes[0]
    for row in rows:
        if row.get('kind')!='request':continue
        key=(row.get('case'),row.get('mode'),row.get('repeat'))
        if key in out['acceptance']:raise ValueError(f'{path}: duplicate {key}')
        out['acceptance'][key]=row
    if set(out['acceptance'])!=ACCEPTANCE:raise ValueError(f'{path}: acceptance case/mode/repeat matrix incomplete')
    return out


def paired(raw,comparison,case):
    left,right=comparison
    target=(left,right,'b')
    metric='first_text_s' if case=='prefill' else 'decode_estimate_tps'
    pairs=[]
    for i in range(1,CASES[case]+1):
        rows={arm:raw[arm]['perf'][case][i] for arm in ARMS}
        hashes={rows[a].get('payload_sha256') for a in target}
        prompt_workloads={(rows[a].get('usage') or {}).get('prompt_tokens') for a in target}
        completion_workloads={(rows[a].get('usage') or {}).get('completion_tokens') for a in target}
        workloads=(prompt_workloads,completion_workloads)
        usage_equal=all(len(s)==1 and None not in s for s in workloads)
        errors={arm:perf_valid(r,case,i) for arm,r in rows.items()}
        embedding={arm:quiet(r) for arm,r in rows.items()}
        cache={arm:(r.get('metrics_delta') or {}).get('vllm:prefix_cache_hits_total') for arm,r in rows.items()}
        ready=(len(hashes)==1 and usage_equal and not any(errors[a] for a in target) and
               all(embedding[a]['known'] and embedding[a]['quiet'] for a in target) and
               all(cache[a]==0 for a in target))
        baseline=(rows[left].get(metric)+rows[right].get(metric))/2 if ready else None
        bvalue=rows['b'].get(metric) if ready else None
        gain=(baseline/bvalue if case=='prefill' else bvalue/baseline) if ready and baseline and bvalue else None
        reasons=[]
        if len(hashes)!=1:reasons.append('payload_sha256_mismatch')
        if not usage_equal:reasons.append('prompt_or_completion_workload_mismatch')
        if any(errors[a] for a in target):reasons.append('request_validation_or_profiler_scope')
        if not all(embedding[a]['known'] and embedding[a]['quiet'] for a in target):reasons.append('embedding_active_or_unknown')
        if not all(cache[a]==0 for a in target):reasons.append('prefix_cache_hit_or_unknown')
        pairs.append({'index':i,'payload_sha256':next(iter(hashes)) if len(hashes)==1 else None,
                      'comparison_arms':list(target),
                      'all_five_payload_hashes_match':len({r.get('payload_sha256') for r in rows.values()})==1,
                      'usage_by_arm':{a:rows[a].get('usage') for a in ARMS},
                      'metrics':{a:rows[a].get(metric) for a in ARMS},
                      'accepted_tokens_by_arm':{a:(rows[a].get('metrics_delta') or {}).get('vllm:spec_decode_num_accepted_tokens_total') for a in ARMS},
                      'embedding':embedding,'prefix_hits':cache,'validation_errors':errors,
                      'workload_equal':usage_equal,'comparable':bool(ready and gain is not None),
                      'exclusion_reasons':reasons,'baseline_mean':baseline,'b':bvalue,'b_gain_ratio':gain})
    return {'case':case,'metric':metric,'interpretation':'ratio >1 favors B',
            'baseline_arms':[left,right],'pairs':pairs,'comparable_pairs':sum(x['comparable'] for x in pairs),
            'gain_ratio':stats(x['b_gain_ratio'] for x in pairs)}


def acceptance_summary(raw):
    rows=[]
    for key in sorted(ACCEPTANCE):
        current={arm:raw[arm]['acceptance'][key] for arm in ARMS}
        hashes={v.get('payload_sha256') for v in current.values()}
        per_arm={}
        for arm,row in current.items():
            prior=raw[arm]['acceptance'][(key[0],key[1],1)]
            per_arm[arm]={'payload_sha256':row.get('payload_sha256'),
                'observed_valid':row.get('observed_valid'),
                'acceptance_pass':row.get('acceptance_pass'),
                'finish_reason':row.get('finish_reason'),'truncated':row.get('truncated'),
                'loop_suspect':(row.get('loop_flags') or {}).get('obvious_loop_suspect'),
                'usage':row.get('usage'),'token_ids_present':isinstance(row.get('token_ids'),list),
                'token_ids_status':row.get('token_ids_status'),
                'content_sha256':row.get('content_sha256'),'reasoning_sha256':row.get('reasoning_sha256'),
                'checks':row.get('checks'),'fixture_checks_pass':row.get('fixture_checks_pass'),
                'same_arm_repeat_content':row.get('content_sha256')==prior.get('content_sha256') if key[2]==2 else None,
                'same_arm_repeat_reasoning':row.get('reasoning_sha256')==prior.get('reasoning_sha256') if key[2]==2 else None,
                'same_arm_repeat_token_ids':row.get('token_ids')==prior.get('token_ids') if key[2]==2 and key[1]=='nonstream' else None}
        rows.append({'case':key[0],'mode':key[1],'repeat':key[2],
                     'same_payload_all_arms':len(hashes)==1,'payload_sha256':next(iter(hashes)) if len(hashes)==1 else None,
                     'cross_arm_content_hash_equal':len({v.get('content_sha256') for v in current.values()})==1,
                     'cross_arm_nonempty_content_equal':(len({v.get('content_sha256') for v in current.values()})==1 and
                        all(bool(v.get('content')) for v in current.values())),
                     'cross_arm_reasoning_hash_equal':len({v.get('reasoning_sha256') for v in current.values()})==1,
                     'cross_arm_nonempty_reasoning_equal':(len({v.get('reasoning_sha256') for v in current.values()})==1 and
                        all(bool(v.get('reasoning')) for v in current.values())),
                     'cross_arm_token_ids_equal':len({json.dumps(v.get('token_ids')) for v in current.values()})==1 if key[1]=='nonstream' else None,
                     'arms':per_arm})
    return {'rows':rows,'all_payload_hashes_match':all(x['same_payload_all_arms'] for x in rows),
            'all_observed_valid':all(v['observed_valid'] is True for x in rows for v in x['arms'].values()),
            'all_fixture_checks_pass':all(v['fixture_checks_pass'] is True for x in rows for v in x['arms'].values()),
            'cross_arm_content_equal_count':sum(x['cross_arm_content_hash_equal'] for x in rows),
            'cross_arm_nonempty_content_equal_count':sum(x['cross_arm_nonempty_content_equal'] for x in rows),
            'cross_arm_reasoning_equal_count':sum(x['cross_arm_reasoning_hash_equal'] for x in rows),
            'cross_arm_nonempty_reasoning_equal_count':sum(x['cross_arm_nonempty_reasoning_equal'] for x in rows),
            'same_arm_repeat_content_equal_count_by_arm':{
                arm:sum(x['arms'][arm]['same_arm_repeat_content'] is True for x in rows if x['repeat']==2)
                for arm in ARMS},
            'truncated':[{k:x[k] for k in ('case','mode','repeat')}|{'arm':a} for x in rows for a,v in x['arms'].items() if v['truncated']],
            'loop_suspects':[{k:x[k] for k in ('case','mode','repeat')}|{'arm':a} for x in rows for a,v in x['arms'].items() if v['loop_suspect']]}


def acceptance_for_arms(acceptance,arms):
    selected=tuple(dict.fromkeys((*arms,'b')))
    rows=acceptance['rows']
    payload_match=all(len({x['arms'][a].get('payload_sha256') for a in selected})==1 and
                      all(isinstance(x['arms'][a].get('payload_sha256'),str) and
                          len(x['arms'][a]['payload_sha256'])==64 for a in selected)
                      for x in rows)
    checks_ok=all(x['arms'][a]['observed_valid'] is True and
                  x['arms'][a]['fixture_checks_pass'] is True and
                  x['arms'][a]['truncated'] is False and
                  x['arms'][a]['loop_suspect'] is False
                  for x in rows for a in selected)
    return {'arms':list(selected),'all_observed_and_fixture_checks_pass':checks_ok,
            'all_payload_hashes_match':payload_match,
            'pass':payload_match and checks_ok}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acceptance-name',default='acceptance.jsonl')
    a=p.parse_args()
    raw={arm:load_arm(a.root,arm,a.acceptance_name) for arm in ARMS}
    payload_mismatches=[]
    for case in CASES:
        for i in range(CASES[case]+1):
            # Even an excluded warmup must have the same request bytes.
            if len({raw[arm]['perf'][case][i].get('payload_sha256') for arm in ARMS})!=1:
                payload_mismatches.append({'case':case,'index':i,
                    'hashes_by_arm':{arm:raw[arm]['perf'][case][i].get('payload_sha256') for arm in ARMS}})
    acceptance=acceptance_summary(raw)
    result={'scope':'five-arm fixed synthetic API comparison; O=.27 MTP2, A=.31 stock MTP4, B=combination; 12 GiB KV must be verified by controller',
            'source_sha256':{arm:raw[arm]['source_hashes'] for arm in ARMS},
            'payload_mismatches_including_warmups':payload_mismatches,
            'acceptance':acceptance,'comparisons':{},
            'limits':['No broad model quality conclusion from synthetic fixtures.',
                      'If token workloads, prefix cache, embedding quiet, payload hash, or profiler scope differ, those pairs are excluded.',
                      'Streaming decode tps estimates from first/last text and usage, not token-level timestamps.',
                      'Accepted tokens are descriptive; MTP2 and MTP4 denominators differ.',
                      'Single O1/A1/B/A2/O2 sequence does not establish restart/time-window significance.']}
    for label,arms in {'b_vs_mean_a':('a1','a2'),'b_vs_mean_o':('o1','o2')}.items():
        comparison={case:paired(raw,arms,case) for case in CASES}
        target=(*arms,'b')
        warmup_hashes_match=all(len({raw[arm]['perf'][case][0].get('payload_sha256') for arm in target})==1
                                 for case in CASES)
        comparison['acceptance']=acceptance_for_arms(acceptance,arms)
        comparison['warmup_payload_hashes_match']=warmup_hashes_match
        comparison['performance_comparable_all_requested']=(warmup_hashes_match and
            all(comparison[case]['comparable_pairs']==CASES[case] for case in CASES))
        comparison['comparable_all_requested']=(comparison['performance_comparable_all_requested'] and
            comparison['acceptance']['pass'])
        result['comparisons'][label]=comparison
    result['performance_comparable_all_requested']=(not payload_mismatches and
        all(result['comparisons'][comp][case]['comparable_pairs']==CASES[case]
            for comp in result['comparisons'] for case in CASES))
    result['synthetic_acceptance_pass']=(acceptance['all_payload_hashes_match'] and
        acceptance['all_observed_valid'] and acceptance['all_fixture_checks_pass'] and
        not acceptance['truncated'] and not acceptance['loop_suspects'])
    result['comparable_all_requested']=(result['performance_comparable_all_requested'] and
        result['synthetic_acceptance_pass'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(a.output),'comparable_all_requested':result['comparable_all_requested'],
                      'paired_counts':{c:{case:d[case]['comparable_pairs'] for case in CASES}
                                       for c,d in result['comparisons'].items()}},
                     ensure_ascii=False))
    if not result['comparable_all_requested']:raise SystemExit(2)

if __name__=='__main__':main()
