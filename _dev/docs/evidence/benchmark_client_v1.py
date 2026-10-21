"""Read-only deployment baseline; synthetic requests only, C=1, no credentials saved.

Run on the GPU host with python3 via SSH stdin. --public-local is run on the
Mac and reads the API key through an existing SSH multiplex connection in memory.
The script never changes container or model configuration.
"""
import argparse
import datetime
import hashlib
import json
import random
import re
import statistics
import subprocess
import time
import urllib.request
import uuid


def credential():
    command = ["docker", "inspect", "vllm-qwen-27b"]
    if args.public_local:
        command = ["ssh", "-S", args.ssh_socket, "-p", str(args.ssh_port), args.ssh_target,
                   "docker inspect vllm-qwen-27b"]
    data = json.loads(subprocess.check_output(command, text=True))[0]
    cmd = data["Config"]["Cmd"]
    return cmd[cmd.index("--api-key") + 1]


def call(path, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    return urllib.request.urlopen(urllib.request.Request(
        args.base + path, data=data,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=timeout)


def metrics():
    try:
        with call("/metrics", timeout=10) as response:
            body = response.read().decode()
        result = {}
        for line in body.splitlines():
            if line.startswith("#") or not line.startswith("vllm:"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name.endswith(("_sum", "_count", "_total")) or name in (
                    "vllm:num_requests_running", "vllm:num_requests_waiting"):
                # Sum series, including finish reasons and per-position counters.
                result[name] = result.get(name, 0) + float(line.rsplit(" ", 1)[1])
        return result
    except Exception as exc:
        return {"unavailable": type(exc).__name__}


def text_for(n, seed):
    rng = random.Random(seed)
    words = ["router", "switch", "latency", "packet", "gateway", "service",
             "storage", "memory", "network", "alert", "recovery", "capacity"]
    return " ".join(rng.choice(words) + "=" + str(rng.randrange(10000)) for _ in range(n))


def run(label, words, output, thinking=False):
    run_id = str(uuid.uuid4())
    messages = [
        {"role": "system", "content": run_id + " Synthetic inference performance test. "
         "Explain the operational observations in clear Chinese prose."},
        {"role": "user", "content": text_for(words, run_id) +
         "\n请按网络、存储、计算资源分别分析以上合成观测。详细解释可能原因和排查步骤。"}]
    payload = {"model": "qwen3.6-27b-fp8", "messages": messages, "stream": True,
               "stream_options": {"include_usage": True}, "temperature": 0,
               "max_tokens": output, "ignore_eos": True,
               "chat_template_kwargs": {"enable_thinking": thinking}}
    before = metrics()
    # Avoid deliberately overlapping another active request; recheck briefly.
    waits = 0
    while before.get("vllm:num_requests_running", 0) + before.get("vllm:num_requests_waiting", 0) > 0 and waits < 6:
        time.sleep(5)
        before = metrics()
        waits += 1
    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.perf_counter()
    first_sse = first_text = first_content = last_text = None
    usage, finish, chunks, reason_chars, content_chars = None, None, 0, 0, 0
    with call("/v1/chat/completions", payload) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            obj = json.loads(line[6:])
            now = time.perf_counter() - t0
            if first_sse is None:
                first_sse = now
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if content or reasoning:
                    chunks += 1
                    if first_text is None:
                        first_text = now
                    last_text = now
                if content and first_content is None:
                    first_content = now
                reason_chars += len(reasoning)
                content_chars += len(content)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    total = time.perf_counter() - t0
    time.sleep(.25)
    after = metrics()
    delta = {k: after[k] - v for k, v in before.items()
             if isinstance(v, (int, float)) and isinstance(after.get(k), (int, float))
             and k.endswith(("_sum", "_count", "_total"))}
    keep = ["request_success_total", "prompt_tokens_total", "generation_tokens_total",
            "time_to_first_token_seconds", "request_prefill_time_seconds", "request_decode_time_seconds",
            "request_queue_time_seconds", "prefix_cache_hits_total", "prefix_cache_queries_total",
            "spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total"]
    delta = {k: v for k, v in delta.items() if any(k == "vllm:" + name or k.startswith("vllm:" + name + "_") for name in keep)}
    result = {"label": label, "route": "mac_public_https" if args.public_local else "gpu_host_loopback",
              "started_at_utc": start_utc, "request_id": run_id,
              "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
              "synthetic_word_count": words, "enable_thinking": thinking, "temperature": 0,
              "ignore_eos": True, "requested_output_tokens": output, "concurrency": 1,
              "first_sse_s": first_sse, "first_text_s": first_text, "first_content_s": first_content,
              "last_text_s": last_text, "total_s": total, "usage": usage, "finish_reason": finish,
              "text_chunks": chunks, "reasoning_chars": reason_chars, "content_chars": content_chars,
              "metrics_delta": delta,
              "running_before": before.get("vllm:num_requests_running"),
              "waiting_before": before.get("vllm:num_requests_waiting"),
              "running_after": after.get("vllm:num_requests_running")}
    if usage and first_text is not None and last_text is not None and last_text > first_text:
        result["client_decode_tps_estimate"] = (usage["completion_tokens"] - 1) / (last_text - first_text)
    result["single_request_metrics_window"] = (
        delta.get("vllm:request_success_total") == 1
        and delta.get("vllm:request_prefill_time_seconds_count") == 1
        and delta.get("vllm:prompt_tokens_total") == (usage or {}).get("prompt_tokens")
        and delta.get("vllm:generation_tokens_total") == (usage or {}).get("completion_tokens"))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://127.0.0.1:8000")
parser.add_argument("--public-local", action="store_true")
parser.add_argument("--ssh-target")
parser.add_argument("--ssh-port", type=int, default=22222)
parser.add_argument("--ssh-socket", default="/tmp/pro6000d-research-ssh")
args = parser.parse_args()
key = credential()
if args.public_local:
    for i in range(3):
        run("public_decode_~4k_r" + str(i + 1), 800, 256)
else:
    run("warmup_excluded", 60, 32)
    for i in range(3):
        for label, words, tokens in [("short_decode", 100, 256), ("medium_decode", 800, 256),
                                     ("long_prefill", 3200, 1)]:
            run(label + "_r" + str(i + 1), words, tokens)
    run("thinking_observation", 800, 256, True)
