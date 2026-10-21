"""Four read-only, synthetic API requests for open-output correctness checks.

Run on the GPU host. Output is JSONL and contains complete synthetic responses
so identical inputs can be compared within and across experiment arms. No model
configuration is changed and no credentials are printed or saved.
"""

import argparse
import datetime
import hashlib
import json
import re
import subprocess
import time
import urllib.request


def sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tag", required=True, help="Experiment-arm label")
parser.add_argument("--base", default="http://127.0.0.1:18000")
args = parser.parse_args()

# Reuse the production container's existing API key without exposing it in JSONL.
cmd = json.loads(subprocess.check_output(["docker", "inspect", "vllm-qwen-27b"]))[0]["Config"]["Cmd"]
key = cmd[cmd.index("--api-key") + 1]

cases = [
    {
        "case": "open_chinese_operations_summary",
        "messages": [
            {"role": "system", "content": "你是运维助手。只依据用户给出的合成事件回答。"},
            {"role": "user", "content": (
                "合成事件：14:00，节点lab-router-7出口丢包率升至12%，CPU占用21%；"
                "14:03，将出口流量切换至备用链路；14:06，丢包率回落到0.2%。"
                "请用两句中文概述故障、处置与恢复。必须原样保留节点名、"
                "12%、0.2%和备用链路，不要输出JSON。"
            )},
        ],
        "thinking": False,
        "max_tokens": 384,
        "gold": {"required_literals": ["lab-router-7", "12%", "0.2%", "备用链路"]},
    },
    {
        "case": "open_thinking_arithmetic",
        "messages": [
            {"role": "system", "content": "请认真推理合成算术题，并按用户要求给出最终答复。"},
            {"role": "user", "content": (
                "仓库原有17箱设备，每箱8台，另到货9台，已发出53台。"
                "还剩多少台？可以在思考中计算，最终正文只用一句中文和阿拉伯数字回答，"
                "不要写算式或JSON。"
            )},
        ],
        "thinking": True,
        "max_tokens": 1024,
        "gold": {"remaining_devices": 92},
    },
]

for case in cases:
    first = None
    for repeat in (1, 2):
        payload = {
            "model": "qwen3.6-27b-fp8",
            "messages": case["messages"],
            "temperature": 0,
            "max_tokens": case["max_tokens"],
            "chat_template_kwargs": {"enable_thinking": case["thinking"]},
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        request = urllib.request.Request(
            args.base.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.load(response)
        elapsed = time.perf_counter() - start

        choice = body["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if case["case"] == "open_chinese_operations_summary":
            checks = {literal: literal in content for literal in case["gold"]["required_literals"]}
        else:
            # Only the final body is scored; intermediate reasoning may include
            # other numbers. The response requests one Arabic-numeral answer.
            device_counts = [int(n) for n in re.findall(r"(?<!\d)(\d+)\s*台(?!\d)", re.sub(r"[*_`]", "", content))]
            checks = {"final_answer_92_devices": device_counts == [92]}
        record = {
            "tag": args.tag,
            "case": case["case"],
            "repeat": repeat,
            "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "payload": payload,
            "payload_sha256": sha256(encoded),
            "gold": case["gold"],
            "fact_checks": checks,
            "passed": choice.get("finish_reason") == "stop" and all(checks.values()),
            "message": message,
            "content": content,
            "reasoning": reasoning,
            "content_sha256": sha256(content),
            "reasoning_sha256": sha256(reasoning),
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage"),
            "elapsed_s": round(elapsed, 6),
            "intra_case_first_diff_char_content": None if first is None else first_difference(first[0], content),
            "intra_case_first_diff_char_reasoning": None if first is None else first_difference(first[1], reasoning),
            "scope": "fixed synthetic API checks; not MoBase end-to-end or broad model quality evaluation",
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if first is None:
            first = (content, reasoning)
