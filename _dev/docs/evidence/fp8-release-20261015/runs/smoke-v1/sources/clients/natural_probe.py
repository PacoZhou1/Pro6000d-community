#!/usr/bin/env python3
"""Fixed synthetic combination checks. No tool execution or model control."""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
import urllib.parse
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def read_key(args):
    key=os.environ.get(args.api_key_env)
    if not key and args.inspect_container:
        obj=json.loads(subprocess.check_output(['docker','inspect',args.inspect_container]))[0]
        cmd=obj['Config']['Cmd']
        key=cmd[cmd.index('--api-key')+1]
    if not key and args.allow_no_key:
        host=urllib.parse.urlsplit(args.base).hostname
        if host not in ('127.0.0.1','::1'):
            raise RuntimeError('--allow-no-key is restricted to an explicit loopback IP')
        return None
    if not key:raise RuntimeError('API key unavailable: set requested env or --inspect-container')
    return key


def open_request(base,key,route,payload=None,timeout=600):
    headers={'Content-Type':'application/json'}
    if key is not None:headers['Authorization']='Bearer '+key
    req=urllib.request.Request(base.rstrip('/')+route,
        data=None if payload is None else canonical(payload),headers=headers)
    return urllib.request.urlopen(req,timeout=timeout)


def fixtures(model):
    event=(
      '合成事件：lab-router-7 于14:00丢包12%，CPU仅21%；14:03切换备用链路；'
      '14:06丢包降至0.2%。storage-3 同时有单盘延迟升高但无数据丢失。')
    return [
      ('normal_long',{
        'messages':[{'role':'system','content':'你是运维助手，只依据合成事件作答。'},
                    {'role':'user','content':event+'请给出至少八段完整中文分析，分别谈时间线、网络、存储、计算、证据缺口、验证、回滚条件与后续监控。自然结束，不要反复续写。'}],
        'chat_template_kwargs':{'enable_thinking':False},'max_tokens':1536},'long'),
      ('thinking_long',{
        'messages':[{'role':'system','content':'请推理后给出完整最终答复。'},
                    {'role':'user','content':event+'请先审视证据是否足以断定根因，再给出六项具体排查步骤与最终建议。正文以结论收尾，避免重复。'}],
        'chat_template_kwargs':{'enable_thinking':True},'max_tokens':2048},'long'),
      ('json_object',{
        'messages':[{'role':'user','content':'三组设备分别19、7、24台，已领用13台。只输出JSON对象，字段remaining为整数剩余量，status为字符串"ok"。'}],
        'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_object'},
        'max_tokens':512},'json'),
      ('tool_arguments',{
        'messages':[{'role':'user','content':'请调用 get_metrics 查询 lab-node-7 的 cpu 指标。只生成工具调用参数，不要臆造查询结果。'}],
        'chat_template_kwargs':{'enable_thinking':False},
        'tools':[{'type':'function','function':{'name':'get_metrics','description':'读取合成主机的一项指标；此实验不执行调用。',
          'parameters':{'type':'object','properties':{'host':{'type':'string'},'metric':{'type':'string'}},
                        'required':['host','metric'],'additionalProperties':False}}}],
        'tool_choice':{'type':'function','function':{'name':'get_metrics'}},'max_tokens':512},'tool')]


def loop_flags(text):
    lines=[re.sub(r'\s+',' ',x).strip() for x in text.splitlines() if x.strip()]
    repeated_lines=any(lines[i]==lines[i+1]==lines[i+2]==lines[i+3] and len(lines[i])>=8
                       for i in range(max(0,len(lines)-3)))
    normalized=re.sub(r'\s+',' ',text)
    repeated_block=any(normalized[i:i+96]*3==normalized[i:i+288]
                       for i in range(0,max(0,len(normalized)-287),24))
    return {'four_identical_consecutive_lines':repeated_lines,
            'three_identical_consecutive_96char_blocks':repeated_block,
            'obvious_loop_suspect':repeated_lines or repeated_block}


def nonstream(base,key,payload):
    with open_request(base,key,'/v1/chat/completions',payload) as response:
        body=json.load(response);status=response.status
    choice=body['choices'][0];msg=choice.get('message') or {}
    return {'http_status':status,'finish_reason':choice.get('finish_reason'),
            'content':msg.get('content') or '',
            'reasoning':msg.get('reasoning_content') or msg.get('reasoning') or '',
            'tool_calls':msg.get('tool_calls') or [],
            'token_ids':choice.get('token_ids',msg.get('token_ids')),
            'usage':body.get('usage'),'stream_done':None}


def streaming(base,key,payload):
    content=[];reasoning=[];calls={};usage=None;finish=None;done=False;status=None
    with open_request(base,key,'/v1/chat/completions',payload) as response:
        status=response.status
        for raw in response:
            line=raw.decode('utf-8').strip()
            if not line.startswith('data: '):continue
            if line=='data: [DONE]':done=True;break
            item=json.loads(line[6:])
            if item.get('usage') is not None:usage=item['usage']
            for choice in item.get('choices') or []:
                if choice.get('finish_reason') is not None:finish=choice['finish_reason']
                d=choice.get('delta') or {}
                content.append(d.get('content') or '')
                reasoning.append(d.get('reasoning_content') or d.get('reasoning') or '')
                for tc in d.get('tool_calls') or []:
                    idx=tc['index'];out=calls.setdefault(idx,{'id':'','type':'function','function':{'name':'','arguments':''}})
                    out['id']+=tc.get('id') or ''
                    fn=tc.get('function') or {}
                    out['function']['name']+=fn.get('name') or ''
                    out['function']['arguments']+=fn.get('arguments') or ''
    return {'http_status':status,'finish_reason':finish,'content':''.join(content),
            'reasoning':''.join(reasoning),'tool_calls':[calls[k] for k in sorted(calls)],
            'token_ids':None,'usage':usage,'stream_done':done}


def check_case(kind,reply):
    content=reply['content'];calls=reply['tool_calls']
    if kind=='json':
        try:parsed=json.loads(content)
        except (ValueError,TypeError):parsed=None
        return {'json_parseable':isinstance(parsed,dict),
                'expected_fixture_fields':parsed=={'remaining':37,'status':'ok'}}
    if kind=='tool':
        try:args=json.loads(calls[0]['function']['arguments']) if len(calls)==1 else None
        except (ValueError,TypeError,KeyError):args=None
        return {'one_get_metrics_call':len(calls)==1 and calls[0]['function']['name']=='get_metrics',
                'arguments_parseable':isinstance(args,dict),
                'expected_fixture_arguments':args=={'host':'lab-node-7','metric':'cpu'},
                'tool_executed':False}
    return {'substantial_final_body':len(content.strip())>=160}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',default='http://127.0.0.1:18000')
    p.add_argument('--model',help='Exact served model ID; omit only when /v1/models has one entry')
    p.add_argument('--api-key-env',default='PRO6000D_COMBO_API_KEY')
    p.add_argument('--inspect-container',help='Optional owned service container for in-memory key lookup')
    p.add_argument('--allow-no-key',action='store_true',
                   help='Allow unauthenticated requests only to an explicit loopback IP')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():p.error('fresh output required')
    key=read_key(a)
    with open_request(a.base,key,'/v1/models',timeout=30) as response:models=json.load(response)['data']
    ids=[x['id'] for x in models]
    if a.model and a.model not in ids:raise RuntimeError('requested model ID not served')
    if not a.model and len(ids)!=1:raise RuntimeError('pass --model when multiple IDs are served')
    model=a.model or ids[0]
    script_sha=digest(Path(__file__).read_bytes())
    failures=[]
    all_observed_valid=True
    with a.output.open('x') as out:
        def emit(row):
            out.write(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n');out.flush()
        emit({'kind':'meta','tag':a.tag,'model_id':model,'served_model_ids':ids,
              'script_sha256':script_sha,'scope':'synthetic API completion checks, no tool execution or broad business quality',
              'api_key_recorded':False,'authorization_header_sent':key is not None})
        for case,spec,kind in fixtures(model):
            for mode in ('nonstream','sse'):
                payload={'model':model,'temperature':0,**spec,'stream':mode=='sse'}
                if mode=='nonstream':payload['return_token_ids']=True
                else:payload['stream_options']={'include_usage':True}
                ph=digest(canonical(payload))
                for repeat in (1,2):
                    t0=time.perf_counter()
                    reply=nonstream(a.base,key,payload) if mode=='nonstream' else streaming(a.base,key,payload)
                    elapsed=time.perf_counter()-t0
                    content=reply.pop('content');reasoning=reply.pop('reasoning')
                    flags=loop_flags(content+'\n'+reasoning)
                    checks=check_case(kind,{**reply,'content':content,'reasoning':reasoning})
                    fixture_ok=all(v is True for k,v in checks.items() if k!='tool_executed')
                    usage=reply.get('usage') or {}
                    normal=reply['finish_reason'] in ('stop','tool_calls')
                    token_ids=reply.get('token_ids')
                    token_ids_status=('not_requested_in_sse' if mode=='sse' else
                                      'available' if isinstance(token_ids,list) and len(token_ids)>0 else
                                      'unavailable')
                    observed_reasons=[]
                    if reply['http_status']!=200:observed_reasons.append('http_status_not_200')
                    if mode=='sse' and not reply['stream_done']:observed_reasons.append('sse_done_missing')
                    if not normal:observed_reasons.append('natural_eos_missing_or_truncated')
                    if usage.get('completion_tokens',0)<=0:observed_reasons.append('completion_usage_missing_or_zero')
                    valid=not observed_reasons
                    failure_reasons=list(observed_reasons)
                    if not fixture_ok:failure_reasons.append('fixture_checks_failed')
                    if flags['obvious_loop_suspect']:failure_reasons.append('obvious_loop_suspect')
                    accepted=not failure_reasons
                    all_observed_valid=all_observed_valid and valid
                    row={'kind':'request','tag':a.tag,'case':case,'mode':mode,'repeat':repeat,
                         'observed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                         'payload':payload,'payload_sha256':ph,'elapsed_s':elapsed,
                         'content':content,'reasoning':reasoning,
                         'content_sha256':digest(content.encode()),'reasoning_sha256':digest(reasoning.encode()),
                         'checks':checks,'fixture_checks_pass':fixture_ok,
                         'normal_eos_or_tool_stop':normal,
                         'truncated':reply['finish_reason']=='length','loop_flags':flags,
                         'token_ids_requested':mode=='nonstream',
                         'token_ids_status':token_ids_status,
                         'token_ids_available':token_ids_status=='available',
                         'observed_valid':valid,'acceptance_pass':accepted,
                         'failure_reasons':failure_reasons,**reply}
                    emit(row)
                    if failure_reasons:failures.append({'case':case,'mode':mode,'repeat':repeat,
                                                        'reasons':failure_reasons})
        emit({'kind':'complete','all_observed_valid':all_observed_valid,
              'all_acceptance_pass':not failures,'failed_requests':failures,
              'limitations':'normal stop and simple fixture checks do not establish broad quality; token ID availability is reported, not assumed'})
    print(json.dumps({'output':str(a.output),'model_id':model,'failed_requests':failures}))
    if failures:raise SystemExit(2)

if __name__=='__main__':main()
