"""Community 0.6.2 Burst/E2E transport and metrics, with unique request cache_salt.
Only extension: isolate each measured prompt cache. Prompt/generation/scorer intact.
Use request-count == concurrency, warmup-request-count=0, max-tokens=4096.
"""
import importlib.util,sys,pathlib,json,hashlib,uuid,time,copy,inspect
source=pathlib.Path('/tmp/burst-community-0.6.2.py')
spec=importlib.util.spec_from_file_location('community_burst',source)
bench=importlib.util.module_from_spec(spec);sys.modules[spec.name]=bench;spec.loader.exec_module(bench)
original=bench.stream_one_request
records=[];auxiliary_records=[];run_id=uuid.uuid4().hex
signature=inspect.signature(original)
async def isolated_request(client,url,payload,index,*args,**kwargs):
    assert payload['max_tokens']==4096
    bound=signature.bind(client,url,payload,index,*args,**kwargs);bound.apply_defaults()
    viewer=bound.arguments.get('output_viewer')
    measured=bound.arguments['target_request_count']>0 and getattr(viewer,'phase',None)=='measurement'
    payload=copy.deepcopy(payload)
    payload['cache_salt']=f'homelab-cold-{run_id}-{index}-{uuid.uuid4().hex}'
    item={'worker':index,'measured':measured,'target_request_count':bound.arguments['target_request_count'],'viewer_phase':getattr(viewer,'phase',None),'prompt_sha256':hashlib.sha256(json.dumps(payload['messages'],sort_keys=True).encode()).hexdigest(),'cache_salt_sha256':hashlib.sha256(payload['cache_salt'].encode()).hexdigest(),'start_monotonic':time.monotonic()}
    result=await original(client,url,payload,index,*args,**kwargs)
    item.update(request_samples=len(result.request_samples),error=result.error)
    if len(result.request_samples)==1:
        s=result.request_samples[0]
        item.update(first_token_monotonic=item['start_monotonic']+s.ttft,input_tokens=s.input_tokens,output_tokens=s.output_tokens,completed=s.completed)
    (records if measured else auxiliary_records).append(item)
    # A repeated request within a worker would reuse its salt, invalidating cold scope.
    if measured:
        assert len(result.request_samples)==1,(index,'Expected one measured request per worker')
    return result
bench.stream_one_request=isolated_request
try:
    bench.main()
finally:
    bench._restore_terminal()
    if '--output' in sys.argv:
        output=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
        metadata={'scope':'Community 0.6.2 finite Burst/E2E with unique per-invocation cache_salt; exact prompt targeting, unchanged prompt text, requested 4096-token output (completion verified separately). Default hidden decode warmup preserved and excluded from measured records. This is NOT the community context-zero sustained decode metric.','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'records':records,'auxiliary_records':auxiliary_records}
        if records and all(r.get('completed') and r.get('first_token_monotonic') for r in records):
            span=max(r['first_token_monotonic'] for r in records)-min(r['start_monotonic'] for r in records)
            metadata.update(all_first_tokens_elapsed_seconds=span,first_wave_input_rate_including_queue_and_mixed_decode=sum(r['input_tokens'] for r in records)/span)
        output.with_suffix('.cold-metadata.json').write_text(json.dumps(metadata,indent=2))
