"""Run on the GPU host. Manage only named temporary inference containers.

Credentials stay in memory, read from the original container. No model download
or production configuration changes are performed by this helper.
"""
import argparse
import json
import subprocess
import time
import urllib.request


def inspect(name):
    return json.loads(subprocess.check_output(["docker", "inspect", name]))[0]


def health(name, port):
    d = inspect(name)
    cmd = d["Config"]["Cmd"]
    key = cmd[cmd.index("--api-key") + 1]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/metrics",
        headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=10) as r:
        body = r.read().decode()
    active = sum(float(line.rsplit(" ", 1)[1]) for line in body.splitlines()
                 if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")))
    if active:
        raise RuntimeError(f"{name} has active or waiting requests")


def wait_ready(name, port, seconds=900):
    """Wait for authenticated HTTP 200 before starting the next original service."""
    cmd = inspect(name)["Config"]["Cmd"]
    key = cmd[cmd.index("--api-key") + 1]
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/health",
                headers={"Authorization": "Bearer " + key})
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    return
        except Exception:
            pass  # The model is still loading; never echo a credential-bearing error.
        time.sleep(5)
    raise RuntimeError(f"{name} did not reach authenticated health 200 in {seconds}s")


p = argparse.ArgumentParser()
p.add_argument("action", choices=["pause", "launch", "stop", "restore"])
p.add_argument("--name", default="codex-spec-mtp2")
p.add_argument("--method", choices=["mtp", "dflash"], default="mtp")
p.add_argument("--draft-tokens", type=int, default=2)
p.add_argument("--draft-path")
p.add_argument("--kv", default="fp8")
p.add_argument("--attention")
p.add_argument("--memory", type=float, default=0.70)
p.add_argument("--max-len", type=int, default=262144)
p.add_argument("--spec-extra", default="{}")
a = p.parse_args()
if not a.name.startswith("codex-spec-"):
    raise ValueError("Temporary container names must begin codex-spec-")

if a.action == "pause":
    for name, port in [("vllm-qwen-27b", 8000), ("vllm-qwen3.5-9b", 8001),
                       ("vllm-qwen3-asr", 8003)]:
        health(name, port)
    print(subprocess.check_output(["docker", "stop", "--time", "60", "vllm-qwen-27b",
                                   "vllm-qwen3.5-9b", "vllm-qwen3-asr"], text=True))
elif a.action == "restore":
    running = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).splitlines()
    if any(name.startswith("codex-spec-") for name in running):
        raise RuntimeError("Stop temporary inference containers before restoring production")
    for name, port in [("vllm-qwen-27b", 8000), ("vllm-qwen3.5-9b", 8001)]:
        if not inspect(name)["State"]["Running"]:
            print(subprocess.check_output(["docker", "start", name], text=True))
        wait_ready(name, port)
    # Start ASR only after the other two finish allocating memory; caller checks health.
elif a.action == "stop":
    print(subprocess.check_output(["docker", "stop", "--time", "60", a.name], text=True))
else:
    original = inspect("vllm-qwen-27b")
    cmd = original["Config"]["Cmd"][:]
    def change(flag, value):
        if flag in cmd:
            cmd[cmd.index(flag) + 1] = str(value)
        else:
            cmd.extend([flag, str(value)])
    change("--gpu-memory-utilization", a.memory)
    change("--max-model-len", a.max_len)
    change("--kv-cache-dtype", a.kv)
    change("--max-num-batched-tokens", 32768)
    change("--host", "127.0.0.1")
    change("--port", 18000)
    spec = {"method": a.method, "num_speculative_tokens": a.draft_tokens}
    if a.method == "dflash":
        if not a.draft_path:
            raise ValueError("DFlash requires a verified local draft path")
        spec["model"] = "/draft"
    spec.update(json.loads(a.spec_extra))
    change("--speculative-config", json.dumps(spec))
    if a.attention:
        change("--attention-backend", a.attention)
    model = next(m["Source"] for m in original["Mounts"] if m["Destination"] == "/model")
    image = "sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b"
    create = ["docker", "create", "--name", a.name, "--gpus", "all", "--ipc", "host",
              "--network", "host", "--restart", "no", "--ulimit", "memlock=-1:-1",
              "--mount", f"type=bind,src={model},dst=/model,readonly",
              "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
              "--env", "VLLM_USE_V2_MODEL_RUNNER=1"]
    if a.draft_path:
        create += ["--mount", f"type=bind,src={a.draft_path},dst=/draft,readonly"]
    # Capture output without echoing CLI arguments containing the API key.
    r = subprocess.run(create + [image] + cmd, text=True, capture_output=True)
    if r.returncode:
        raise RuntimeError("docker create failed: " + r.stderr)
    print(subprocess.check_output(["docker", "start", a.name], text=True))
