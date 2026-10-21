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
assert manifests['b']['libgateup.so']=='af1d814966a746ff09edc55d40196b871067d88d22ed0e50700c1f3a4eb185d6'
ignore={'PRO6000D_OFFICIAL_MODE','PRO6000D_OFFICIAL_LIBRARY','PRO6000D_OFFICIAL_LIBRARY_SHA256','PRO6000D_OFFICIAL_VALIDATED_LIBRARY_SHA256'}
def env(arm):return {s.split('=',1)[0]:s.split('=',1)[1] for s in rows[arm]['env'] if s.split('=',1)[0] not in ignore}
assert env('a1')==env('b')==env('a2')
# Exact mounted model path is retained internally in saved, credential-free runtime files.
models={arm:next(x['Source'] for x in rows[arm]['Mounts'] if x['Destination']=='/model') for arm in rows};assert len(set(models.values()))==1
packed={arm:sum(json.loads(f.read_text())['total_bytes'] for f in (root/arm).glob('startup-pack-*.json') if not f.name.endswith('-summary.json')) for arm in rows}
assert set(packed.values())=={11589593600}
j=json.loads((root/'analysis.json').read_text());pre=j['paired_ttft']['prefill'];pairs=pre['pairs']
drift=statistics.median(100*(x['a2_first_text_s']/x['a1_first_text_s']-1) for x in pairs)
result={'passed':True,'scope':'audit of fixed unprofiled A1/B/A2 experiment; no production deployment',
 'model_same':True,'image_same':True,'cmd_same':True,'environment_same_except_explicit_candidate_optin':True,
 'instrument_same_except_candidate_library':True,'kv_bytes':12884901888,'packed_bytes_by_arm':packed,
 'analysis_sha256':hashlib.sha256((root/'analysis.json').read_bytes()).hexdigest(),
 'ttft_median_s':{arm:j['arms'][arm]['prefill']['statistics']['first_text_s']['median'] for arm in rows},
 'decode_tps_median':{arm:j['arms'][arm]['decode']['statistics']['decode_estimate_tps']['median'] for arm in rows},
 'prefill_paired_reduction_pct':pre['all_pairs_reduction_pct']['median'],
 'prefill_embedding_quiet_reduction_pct':pre['embedding_quiet_pairs_reduction_pct']['median'],
 'paired_a2_over_a1_drift_pct_median':drift,
 'decode_nonempty_content_equal_counts':{key:value['content']['meaningful_nonempty_equal_count'] for key,value in j['decode_output_hashes'].items()},
 'quality_checks_passed':j['validation']['quality_all_passed'],
 'adoption':'not_adopted',
 'reason':'Observed paired TTFT difference is tiny relative to A1/A2 drift; no independent repeated windows establish a useful model gain. Synthetic kernel improvement is not a deployment gain.'}
(root/'model-contract.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
