"""Synthetic chat / structured output / tool-call compatibility smoke; no tools run."""
import argparse
import datetime
import json
import subprocess
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--model", default="qwen3.6-27b-fp8")
p.add_argument("--base", default="http://127.0.0.1:18000")
p.add_argument("--tag", required=True)
args = p.parse_args()
cmd = json.loads(subprocess.check_output(["docker", "inspect", "vllm-qwen-27b"], text=True))[0]["Config"]["Cmd"]
key = cmd[cmd.index("--api-key") + 1]
common = {"model": args.model, "temperature": 0, "max_tokens": 128,
          "chat_template_kwargs": {"enable_thinking": False}}
cases = [
    ("arithmetic", {"messages": [{"role": "user", "content": "计算17乘23，只输出十进制整数。"}]}),
    ("json", {"messages": [{"role": "user", "content": '只返回JSON对象，包含ok为true，以及value为17乘23的整数结果。'}],
              "response_format": {"type": "json_object"}}),
    ("tool_call", {"messages": [{"role": "user", "content": "请调用get_metrics工具查询cpu指标。不要猜测指标值。"}],
                   "tools": [{"type": "function", "function": {"name": "get_metrics", "description": "查询系统指标",
                              "parameters": {"type": "object", "properties": {"metric": {"type": "string", "enum": ["cpu", "memory"]}}, "required": ["metric"], "additionalProperties": False}}}],
                   "tool_choice": "auto"})]
for name, extra in cases:
    payload = dict(common, **extra)
    req = urllib.request.Request(args.base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=90) as r:
        body = json.load(r)
    choice = body["choices"][0]
    message = choice["message"]
    passed = False
    if name == "arithmetic":
        passed = (message.get("content") or "").strip() == "391" and choice["finish_reason"] == "stop"
    elif name == "json":
        try:
            content = json.loads(message.get("content") or "")
            passed = content.get("ok") is True and content.get("value") == 391 and choice["finish_reason"] == "stop"
        except ValueError:
            pass
    else:
        try:
            tool = message["tool_calls"][0]["function"]
            passed = (tool["name"] == "get_metrics" and json.loads(tool["arguments"]) == {"metric": "cpu"}
                      and choice["finish_reason"] == "tool_calls")
        except (KeyError, IndexError, ValueError):
            pass
    print(json.dumps({"tag": args.tag, "model": args.model, "case": name, "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "passed": passed, "finish_reason": choice["finish_reason"], "message": message,
                      "usage": body.get("usage"), "scope": "synthetic API smoke, not MoBase E2E or model quality evaluation"}, ensure_ascii=False), flush=True)
