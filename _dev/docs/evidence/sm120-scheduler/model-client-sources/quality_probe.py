"""Synthetic API checks, including repeated long-prefix retrieval.

These exercise live inference and cache reuse, not MoBase integration or broad
model quality. All prompts and expected answers are public synthetic fixtures.
"""
import argparse
import datetime
import json
import random
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--base", default="http://127.0.0.1:18000")
a = p.parse_args()
cmd = json.loads(subprocess.check_output(["docker", "inspect", "vllm-qwen-27b"]))[0]["Config"]["Cmd"]
key = cmd[cmd.index("--api-key") + 1]


def request(route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(a.base + route, data=data,
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=180)


def metrics():
    with request("/metrics") as r:
        lines = r.read().decode().splitlines()
    out = {}
    for line in lines:
        if line.startswith(("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")):
            name = line.split("{", 1)[0]
            out[name] = out.get(name, 0) + float(line.rsplit(" ", 1)[1])
    return out


rng = random.Random(714)
noise = [f"Synthetic row {i}: router={rng.randrange(10000)} storage={rng.randrange(10000)}."
         for i in range(2000)]
noise.insert(713, "权威记录：工单ZX627的回滚标识为FALCON-2741。")
noise.insert(1628, "权威记录：工单ZX628的回滚标识为MAPLE-9037。")
long_prompt = "\n".join(noise) + '\n从权威记录提取两张工单的回滚标识。仅输出JSON，字段为ZX627和ZX628。'
cases = [
    ("json_arithmetic", [{"role": "user", "content": '库存三组各19、7、24台；已领用13台。仅输出JSON {"remaining":剩余总数}。'}], {"remaining": 37}),
    ("history_correction", [{"role": "user", "content": "机柜容量是500GB。"},
                            {"role": "assistant", "content": "已记录容量500GB。"},
                            {"role": "user", "content": '更正：容量750GB，已用125GB。只输出JSON，capacity和free用GB整数。'}],
     {"capacity": 750, "free": 625}),
    ("tool_result_reading", [{"role": "user", "content": "请读取cpu指标。"},
                             {"role": "assistant", "content": None,
                              "tool_calls": [{"id": "call_synthetic_metrics", "type": "function",
                                              "function": {"name": "get_metrics", "arguments": '{"metric":"cpu"}'}}]},
                             {"role": "tool", "tool_call_id": "call_synthetic_metrics",
                              "content": '{"cpu_percent":93,"host":"synthetic-node"}'},
                             {"role": "user", "content": '只依据工具结果返回JSON，字段host、cpu_percent，不添加其他字段。'}],
     {"host": "synthetic-node", "cpu_percent": 93}),
    ("long_retrieval_cold", [{"role": "user", "content": long_prompt}],
     {"ZX627": "FALCON-2741", "ZX628": "MAPLE-9037"}),
    ("long_retrieval_repeated", [{"role": "user", "content": long_prompt}],
     {"ZX627": "FALCON-2741", "ZX628": "MAPLE-9037"}),
]
for name, messages, expected in cases:
    payload = {"model": "qwen3.6-27b-fp8", "messages": messages, "temperature": 0,
               "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False},
               "response_format": {"type": "json_object"}}
    before = metrics()
    start = time.perf_counter()
    with request("/v1/chat/completions", payload) as r:
        body = json.load(r)
    elapsed = time.perf_counter() - start
    after = metrics()
    choice = body["choices"][0]
    text = choice["message"].get("content")
    try:
        answer = json.loads(text)
    except (TypeError, ValueError):
        answer = None
    print(json.dumps({"tag": a.tag, "case": name, "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "passed": answer == expected and choice["finish_reason"] == "stop",
                      "expected": expected, "message": choice["message"], "finish_reason": choice["finish_reason"],
                      "usage": body.get("usage"), "elapsed_s": elapsed,
                      "cache_delta": {k: after.get(k, 0) - v for k, v in before.items()},
                      "scope": "synthetic API checks; no external tools executed; not MoBase E2E"}, ensure_ascii=False), flush=True)
