"""Run one isolated speculative-decoding arm over an existing SSH connection.

Only the named codex-spec-* container is launched/stopped. Production service
pause/restore is deliberately left to the operator. This script stores no SSH
target, credentials, or unfiltered container logs in its output files.
"""

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
import time


REMOTE = "/tmp/qwen-spec-experiment"
HEALTH_CODE = r"""
import json, subprocess, sys, urllib.request
name = sys.argv[1]
try:
    item = json.loads(subprocess.check_output(['docker', 'inspect', name], stderr=subprocess.DEVNULL))[0]
    if not item['State']['Running']:
        sys.exit(2)
    cmd = item['Config']['Cmd']
    key = cmd[cmd.index('--api-key') + 1]
    req = urllib.request.Request('http://127.0.0.1:18000/metrics',
                                 headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(req, timeout=4) as response:
        sys.exit(0 if response.status == 200 and response.read() else 1)
except Exception:
    sys.exit(1)
"""
VERIFY_CODE = r"""
import json, subprocess, sys
name, method, tokens, kv, attention, draft_path, extra_json = sys.argv[1:]
try:
    item = json.loads(subprocess.check_output(['docker', 'inspect', name], stderr=subprocess.DEVNULL))[0]
    cmd = item['Config']['Cmd']
    value = lambda flag: cmd[cmd.index(flag) + 1]
    spec = json.loads(value('--speculative-config'))
    extra = json.loads(extra_json)
    expected = (spec['method'] == method and
                spec['num_speculative_tokens'] == int(tokens) and
                value('--kv-cache-dtype') == kv and
                value('--attention-backend') == attention and
                value('--gpu-memory-utilization') == '0.7' and
                value('--max-model-len') == '262144' and
                value('--max-num-seqs') == '16' and
                value('--max-num-batched-tokens') == '32768' and
                'VLLM_USE_V2_MODEL_RUNNER=1' in item['Config']['Env'])
    if method == 'dflash':
        mounts = [m for m in item['Mounts'] if m['Destination'] == '/draft']
        expected = expected and spec.get('model') == '/draft' and len(mounts) == 1
        if draft_path:
            expected = expected and mounts[0]['Source'] == draft_path
    expected = expected and all(spec.get(k) == v for k, v in extra.items())
    sys.exit(0 if expected else 1)
except Exception:
    sys.exit(1)
"""


def remote(ssh, argv, timeout, checked=True):
    command = ssh + [shlex.join(argv)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if checked and result.returncode:
        # Remote stderr can contain command arguments copied from vLLM logs.
        raise RuntimeError(f"remote command failed (exit {result.returncode}): {argv[1]}")
    return result


def records(output, kind):
    rows = [json.loads(line) for line in output.splitlines() if line.strip()]
    expected = {"benchmark": 22, "smoke": 3, "quality": 5, "open": 4}[kind]
    if len(rows) != expected or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{kind}: expected {expected} complete JSONL records, got {len(rows)}")
    if kind == "benchmark":
        if not all(row.get("stream_done") is True and
                   row.get("complete_fixed_output") is True and
                   row.get("single_request_metrics_window") is True for row in rows):
            raise RuntimeError("benchmark: an SSE request or metrics window did not complete")
    else:
        if not all(row.get("usage") and row.get("finish_reason") for row in rows):
            raise RuntimeError(f"{kind}: a request did not finish with usage")
        if not all(row.get("passed") is True for row in rows):
            raise RuntimeError(f"{kind}: a synthetic API case failed")
    return rows


def write_new(path, content):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--socket", required=True, help="Existing SSH control socket")
    parser.add_argument("--port", required=True, type=int, help="SSH port")
    parser.add_argument("--name", required=True, help="One codex-spec-* container")
    parser.add_argument("--method", required=True, choices=["mtp", "dflash"])
    parser.add_argument("--draft-tokens", required=True, type=int)
    parser.add_argument("--kv", required=True)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--draft-path", help="Verified host path; required for a new DFlash arm")
    parser.add_argument("--spec-extra", default="{}", help="Additional explicit speculative configuration")
    parser.add_argument("--output-stem", required=True, type=pathlib.Path)
    parser.add_argument("--mode", choices=["new", "existing"], default="new")
    args = parser.parse_args()
    extra = json.loads(args.spec_extra)
    if not isinstance(extra, dict) or any(k in extra for k in ["method", "model", "num_speculative_tokens"]):
        parser.error("spec-extra must be an object without reserved model/method/token keys")
    if not args.name.startswith("codex-spec-") or not args.name[11:]:
        parser.error("--name must identify a codex-spec-* container")
    if args.draft_tokens < 1 or args.port < 1 or args.port > 65535:
        parser.error("--draft-tokens and --port must be positive and valid")
    if args.method == "dflash" and args.mode == "new" and not args.draft_path:
        parser.error("new DFlash arms require --draft-path")
    if args.method == "mtp" and args.draft_path:
        parser.error("MTP arms do not use --draft-path")

    stem = args.output_stem.expanduser()
    paths = {kind: stem.parent / (prefix + stem.name + suffix) for kind, (prefix, suffix) in {
        "benchmark": ("paired-", ".jsonl"), "smoke": ("smoke-", ".jsonl"),
        "quality": ("quality-", ".jsonl"), "open": ("open-", ".jsonl"), "runtime": ("runtime-", ".json")}.items()}
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        parser.error("output files already exist: " + ", ".join(map(str, existing)))
    stem.parent.mkdir(parents=True, exist_ok=True)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-S", args.socket, "-p", str(args.port), args.ssh_target]
    control = ["python3", f"{REMOTE}/experiment_control.py"]
    inspect = remote(ssh, ["docker", "inspect", args.name], 30, checked=False)
    if args.mode == "new" and inspect.returncode == 0:
        raise RuntimeError("named temporary container already exists; use --mode existing or another name")
    if args.mode == "existing" and inspect.returncode != 0:
        raise RuntimeError("named temporary container does not exist")
    running = remote(ssh, ["docker", "ps", "--format", "{{.Names}}"], 30).stdout.splitlines()
    other_arms = [name for name in running if name.startswith("codex-spec-") and name != args.name]
    if other_arms:
        raise RuntimeError("another codex-spec container is running; serial arm required")

    owns_container = args.mode == "existing"
    try:
        if args.mode == "new":
            launch = control + ["launch", "--name", args.name, "--method", args.method,
                                "--draft-tokens", str(args.draft_tokens), "--kv", args.kv,
                                "--attention", args.attention, "--spec-extra", json.dumps(extra)]
            if args.draft_path:
                launch += ["--draft-path", args.draft_path]
            owns_container = True  # A failed launch may have created the named container.
            remote(ssh, launch, 180)

        match = remote(ssh, ["python3", "-c", VERIFY_CODE, args.name,
                             args.method, str(args.draft_tokens), args.kv,
                             args.attention, args.draft_path or "", json.dumps(extra)], 30, checked=False)
        if match.returncode:
            raise RuntimeError("temporary container configuration differs from requested arm")

        print("Waiting for " + args.name, flush=True)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            result = remote(ssh, ["python3", "-c", HEALTH_CODE, args.name], 20,
                            checked=False)
            if result.returncode == 0:
                break
            if result.returncode == 2:
                raise RuntimeError("temporary container exited before becoming ready")
            time.sleep(5)
        else:
            raise RuntimeError("temporary container did not become healthy within 15 minutes")

        base = "http://127.0.0.1:18000"
        tag = args.name.removeprefix("codex-spec-")
        stages = [
            ("benchmark", ["python3", f"{REMOTE}/benchmark_client.py",
                           "--credential-container", args.name, "--base", base,
                           "--profile", "spec", "--tag", tag], 7200),
            ("smoke", ["python3", f"{REMOTE}/api_smoke.py",
                       "--base", base, "--tag", tag], 600),
            ("quality", ["python3", f"{REMOTE}/quality_probe.py",
                         "--base", base, "--tag", tag], 1200),
            ("open", ["python3", f"{REMOTE}/open_output_probe.py",
                      "--base", base, "--tag", tag], 1200),
        ]
        for kind, command, timeout in stages:
            print("Starting " + kind + " for " + args.name, flush=True)
            result = remote(ssh, command, timeout, checked=False)
            write_new(paths[kind], result.stdout)  # Preserve partial output on failure.
            if result.returncode:
                raise RuntimeError(f"{kind} failed with exit {result.returncode}; partial output preserved")
            records(result.stdout, kind)
            print(f"{kind} complete: {paths[kind]}", flush=True)
    finally:
        if owns_container:
            cleanup_errors = []
            try:
                captured = remote(ssh, ["python3", f"{REMOTE}/capture_runtime.py",
                                        args.name], 120)
                runtime = json.loads(captured.stdout)
                write_new(paths["runtime"], json.dumps(runtime, ensure_ascii=False, indent=2) + "\n")
                print(f"runtime captured: {paths['runtime']}", flush=True)
            except Exception as exc:
                print(f"runtime capture failed: {type(exc).__name__}", file=sys.stderr)
                cleanup_errors.append("runtime capture failed")
            try:
                remote(ssh, control + ["stop", "--name", args.name], 120)
                print(f"stopped {args.name}", flush=True)
            except Exception as exc:
                print(f"temporary container stop failed: {type(exc).__name__}", file=sys.stderr)
                cleanup_errors.append("temporary container stop failed")
            if cleanup_errors and sys.exc_info()[0] is None:
                raise RuntimeError("; ".join(cleanup_errors))


if __name__ == "__main__":
    main()
