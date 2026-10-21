"""Synthetic loopback requests, separate unprofiled performance from trace windows.

Run on GPU host. Decode profiling starts after the first streamed text, so the
request prefill is excluded. Verify that boundary using the captured NVTX ranges.
"""
import argparse
import datetime
import hashlib
import json
import random
import subprocess
import time
import urllib.request
import uuid

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--mode", choices=["baseline", "capture"], required=True)
p.add_argument("--cases", nargs="+", choices=["prefill", "decode", "thinking"], required=True)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--base", default="http://127.0.0.1:18000")
p.add_argument("--profiler-installed", choices=["cuda", "torch", "none"], default="cuda")
a = p.parse_args()
cmd = json.loads(subprocess.check_output(["docker", "inspect", "vllm-embedding"]))[0]["Config"]["Cmd"]
embedding_key = cmd[cmd.index("--api-key") + 1]

def request(route, payload=None, base=None, key=None, timeout=300):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = None if payload is None else json.dumps(payload).encode()
    return urllib.request.urlopen(urllib.request.Request((base or a.base) + route,
                                  data=data, headers=headers), timeout=timeout)

def metrics(embedding=False):
    kwargs = {"base": "http://127.0.0.1:8002", "key": embedding_key} if embedding else {}
    with request("/metrics", **kwargs) as r:
        lines = r.read().decode().splitlines()
    result = {}
    for line in lines:
        if not line.startswith("vllm:"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name.endswith(("_total", "_sum", "_count")) or name in (
                "vllm:num_requests_running", "vllm:num_requests_waiting"):
            result[name] = result.get(name, 0) + float(line.rsplit(" ", 1)[1])
    return result

def profile(action):
    start = time.perf_counter()
    with request("/" + action + "_profile", {}, timeout=300) as r:
        r.read()
        return {"status": r.status, "http_elapsed_s": time.perf_counter() - start}

def payload(case, label, capture):
    rng = random.Random(label)
    words = ["router", "switch", "latency", "packet", "gateway", "service",
             "storage", "memory", "network", "alert", "recovery", "capacity"]
    count = 3200 if case == "prefill" else 800
    content = " ".join(rng.choice(words) + "=" + str(rng.randrange(10000)) for _ in range(count))
    content += "\n请按网络、存储、计算资源分别分析以上合成观测，详细解释原因和排查步骤。"
    return {"model": "qwen3.6-27b-fp8", "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0, "ignore_eos": True,
            "max_tokens": 1 if case == "prefill" else (256 if capture else 1024),
            "chat_template_kwargs": {"enable_thinking": case == "thinking"},
            "messages": [{"role": "system", "content": str(uuid.uuid5(uuid.NAMESPACE_URL, label)) +
                          " Synthetic inference profiling. Explain in clear Chinese prose."},
                         {"role": "user", "content": content}]}

def run(case, index, warmup=False):
    capture = a.mode == "capture" and not warmup
    label = f"qwen-profile-v1:{a.mode}:{case}:r{index}:warm{warmup}"
    data = payload(case, label, capture)
    if warmup:
        data["max_tokens"] = 1 if case == "prefill" else 32
    before, eb = metrics(), metrics(True)
    if before.get("vllm:num_requests_running", 0) or before.get("vllm:num_requests_waiting", 0):
        raise RuntimeError("temporary service busy")
    result = {"tag": a.tag, "case": case, "index": index, "warmup": warmup,
              "profiled": capture, "payload_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
              "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    active = False
    done = False
    content_hash, reasoning_hash = hashlib.sha256(), hashlib.sha256()
    first = last = first_content = None
    usage = finish = None
    try:
        if capture and case == "prefill":
            result["profile_start"] = profile("start")
            active = True
        t0 = time.perf_counter()
        with request("/v1/chat/completions", data) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    break
                item = json.loads(line[6:])
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    d = choice.get("delta", {})
                    content = d.get("content") or ""
                    reasoning = d.get("reasoning") or d.get("reasoning_content") or ""
                    if content or reasoning:
                        now = time.perf_counter() - t0
                        if first is None:
                            first = now
                            if capture and case != "prefill":
                                result["profile_start"] = profile("start")
                                result["profile_start_returned_s"] = time.perf_counter() - t0
                                active = True
                        last = time.perf_counter() - t0
                    if content and first_content is None:
                        first_content = time.perf_counter() - t0
                    content_hash.update(content.encode())
                    reasoning_hash.update(reasoning.encode())
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        result["total_s"] = time.perf_counter() - t0
    finally:
        if active:
            result["profile_stop"] = profile("stop")
    time.sleep(.25)
    after, ea = metrics(), metrics(True)
    delta = {k: after[k] - v for k, v in before.items()
             if k in after and k.endswith(("_sum", "_count", "_total"))}
    valid = (done and finish == "length" and usage and usage["completion_tokens"] == data["max_tokens"]
             and delta.get("vllm:request_success_total") == 1
             and delta.get("vllm:prompt_tokens_total") == usage["prompt_tokens"]
             and delta.get("vllm:generation_tokens_total") == usage["completion_tokens"])
    result.update(usage=usage, finish_reason=finish, stream_done=done, valid=bool(valid),
                  first_text_s=first, first_content_s=first_content, last_text_s=last,
                  metrics_delta=delta, embedding_before=eb, embedding_after=ea,
                  content_sha256=content_hash.hexdigest(), reasoning_sha256=reasoning_hash.hexdigest())
    if first is not None and last > first and usage:
        result["decode_estimate_tps"] = (usage["completion_tokens"] - 1) / (last - first)
    result["timing_scope"] = ("profiling diagnostics only" if capture else
        "no profiler installed" if a.profiler_installed == "none" else "no active capture; profiler-equipped server")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not valid:
        raise RuntimeError("incomplete request or contaminated metrics window")

for case in a.cases:
    run(case, 0, warmup=True)
    for i in range(1, a.repeats + 1):
        run(case, i)
