"""Audit completed A/B/A configuration, payload outputs and adoption decision."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
rows={arm:json.loads((root/arm/'runtime.json').read_text()) for arm in ('a1','b','a2')}
manifests={arm:json.loads((root/arm/'instrument-manifest.json').read_text()) for arm in rows}
assert rows['a1']['Image']==rows['b']['Image']==rows['a2']['Image']
assert rows['a1']['cmd']==rows['b']['cmd']==rows['a2']['cmd']
assert '--profiler-config' not in rows['b']['cmd']
assert rows['b']['cmd'][rows['b']['cmd'].index('--kv-cache-memory-bytes')+1]=='12884901888'
assert manifests['a1']==manifests['a2']
assert {k:v for k,v in manifests['b'].items() if k!='libgateup.so'}==manifests['a1']
LIB_SHA='a6ae4155f65ec3b992ade3943902ccd88eea76b211dda5ceab26626245b7ae5d'
assert manifests['b']['libgateup.so']==LIB_SHA
ignore={'PRO6000D_MAINLOOP_MODE','PRO6000D_MAINLOOP_LIBRARY','PRO6000D_MAINLOOP_LIBRARY_SHA256','PRO6000D_MAINLOOP_VALIDATED_LIBRARY_SHA256'}
def env(arm):return {s.split('=',1)[0]:s.split('=',1)[1] for s in rows[arm]['env'] if s.split('=',1)[0] not in ignore}
assert env('a1')==env('b')==env('a2')
def full_env(arm):return {s.split('=',1)[0]:s.split('=',1)[1] for s in rows[arm]['env']}
assert [full_env(x)['PRO6000D_MAINLOOP_MODE'] for x in ('a1','b','a2')]==['baseline','candidate','baseline']
assert full_env('b')['PRO6000D_MAINLOOP_LIBRARY_SHA256']==LIB_SHA
assert full_env('b')['PRO6000D_MAINLOOP_VALIDATED_LIBRARY_SHA256']==LIB_SHA
assert full_env('b')['PRO6000D_MAINLOOP_LIBRARY']=='/instrument/libgateup.so'
assert 'PRO6000D_MAINLOOP_LIBRARY' not in full_env('a1') and 'PRO6000D_MAINLOOP_LIBRARY' not in full_env('a2')
assert env('a1')['PRO6000D_MAINLOOP_EVIDENCE_DIR']=='/profiles'
assert env('a1')['PRO6000D_MAINLOOP_V2_SITECUSTOMIZE']=='/instrument/v2/sitecustomize.py'
# Exact mounted model path is retained internally in saved, credential-free runtime files.
models={arm:next(x['Source'] for x in rows[arm]['Mounts'] if x['Destination']=='/model') for arm in rows};assert len(set(models.values()))==1
packed={arm:sum(json.loads(f.read_text())['total_bytes'] for f in (root/arm).glob('startup-pack-*.json') if not f.name.endswith('-summary.json')) for arm in rows}
assert set(packed.values())=={11589593600}
j=json.loads((root/'analysis.json').read_text());pre=j['paired_ttft']['prefill'];pairs=pre['pairs']
drift=statistics.median(100*(x['a2_first_text_s']/x['a1_first_text_s']-1) for x in pairs)
assert len(pairs)==12 and all(j['validation'][k] for k in (
    'all_measured_requests_valid','one_target_service_request_per_metrics_window',
    'all_index_payload_hashes_equal','quality_all_passed',
    'open_output_all_normal_stop_with_final_body'))
prefill_gain=pre['all_pairs_reduction_pct']['median']
quiet=pre['embedding_quiet_pairs_reduction_pct']
def prompt_usage(arm):
    records=[json.loads(line) for line in (root/arm/'prefill.jsonl').read_text().splitlines() if line.strip()]
    measured={r['index']:r for r in records if r.get('case')=='prefill' and r.get('warmup') is False}
    assert set(measured)==set(range(1,13))
    return {index:{'payload_sha256':row['payload_sha256'],
                   'prompt_tokens':row['usage']['prompt_tokens']}
            for index,row in sorted(measured.items())}
usage={arm:prompt_usage(arm) for arm in rows}
assert all(usage['a1'][i]==usage[arm][i] for arm in ('b','a2') for i in range(1,13))
# This signal describes only the aggregate prefill timing. The strict M=16160
# guard is not inferred from HTTP token counts; the profiled witness proves
# candidate GPU launches for its one capture request only.
prefill_above_drift_floor=prefill_gain>max(1.0,abs(drift))
reason=(f'One isolated A/B/A window shows paired prefill TTFT reduction '
        f'{prefill_gain:.3f}% versus A2/A1 median drift {drift:.3f}%, but does '
        'not establish whole-model improvement or candidate hits on every '
        'prefill payload. Decode, output, embedding-quiet cases and 11.59 GB '
        'packed-weight cost must be considered; no deployment is authorized.')
result={'passed':True,'scope':'audit of fixed unprofiled A1/B/A2 experiment; no production deployment',
 'model_same':True,'image_same':True,'cmd_same':True,'environment_same_except_explicit_candidate_optin':True,
 'instrument_same_except_candidate_library':True,'kv_bytes':12884901888,'packed_bytes_by_arm':packed,
 'candidate_library_sha256':LIB_SHA,
 'target_M':16160,'raster':1,'swizzle':1,
 'analysis_sha256':hashlib.sha256((root/'analysis.json').read_bytes()).hexdigest(),
 'ttft_median_s':{arm:j['arms'][arm]['prefill']['statistics']['first_text_s']['median'] for arm in rows},
 'decode_tps_median':{arm:j['arms'][arm]['decode']['statistics']['decode_estimate_tps']['median'] for arm in rows},
 'prefill_paired_reduction_pct':prefill_gain,
 'prefill_embedding_quiet_reduction_pct':quiet['median'],
 'prefill_embedding_quiet_pair_count':quiet['n'],
 'prefill_prompt_usage_by_index':usage['a1'],
 'prefill_prompt_usage_cross_arm_equal':True,
 'candidate_hit_scope':'One separately profiled witness request proves 65 fused GEMM and 65 pack launches; this A/B/A JSONL has no per-request kernel/M witness. Prompt-token counts do not prove M=16160, and not all 12 prefill payloads are claimed as candidate hits.',
 'prefill_only_decision':'inconclusive_without_per_request_M_witness',
 'prefill_timing_above_simple_drift_floor':prefill_above_drift_floor,
 'paired_a2_over_a1_drift_pct_median':drift,
 'decode_nonempty_content_equal_counts':{key:value['content']['meaningful_nonempty_equal_count'] for key,value in j['decode_output_hashes'].items()},
 'quality_checks_passed':j['validation']['quality_all_passed'],
 'adoption':'not_adopted',
 'overall_decision':'not_adopted',
 'reason':reason}
(root/'model-contract.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
