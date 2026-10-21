"""Verify the four original inference services after a spec-decode experiment.

Run on the GPU host. This script only inspects Docker and calls loopback APIs;
it does not start, stop, or alter containers. --snapshot-only skips inference
requests and is suitable for a later stability check.

Expected IDs, images, commands and restart counts are read from the uploaded
initial-state.json. The current ASR fixture was generated on macOS with
``say -v Samantha`` and converted with ``afconvert``.
"""

import argparse
import datetime
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


SERVICES = (
    ("vllm-qwen-27b", 8000, "qwen3.6-27b-fp8",
     "b35075fff264a08298677ed8bc6e1f1180776f4958aa2d25b71639da209fd288",
     "sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"),
    ("vllm-qwen3.5-9b", 8001, "qwen3.5-9b",
     "47135c52bc4e89cc1b042b0f28978cbcf45b8ccdaa10101786bab3441c1427a9",
     "sha256:d1bd760bf6630f67378206c7945afb6ab9bc046064a51fe421461e91261dcd7b"),
    ("vllm-qwen3-asr", 8003, "qwen3-asr-1.7b",
     "1810da3dc2845ac19a493ea5d365c7296083f3ea9f26638470c85236e04f94a2",
     "sha256:2f9c6c2942f2c991694884ba86941d50745df34378ce05328c42acd92e34ec64"),
    ("vllm-embedding", 8002, "Qwen3-Embedding-0.6B",
     "4224a2b871260bfe78eeec7d39c67859c385750fcee26bd25d55f668c0283e9a",
     "sha256:d1bd760bf6630f67378206c7945afb6ab9bc046064a51fe421461e91261dcd7b"),
)
ASR_WAV = pathlib.Path("/tmp/qwen-spec-experiment/asr-smoke.wav")
ASR_SHA256 = "ca04e02f1738260ae3146718e9581fcb504bf052ddb3b16160dabfe4c57a3990"
SECRET_FLAGS = {"--api-key", "--hf-token", "--token", "--password"}


def inspect(name):
    proc = subprocess.run(["docker", "inspect", name], capture_output=True,
                          text=True, timeout=15)
    if proc.returncode:
        raise RuntimeError("docker inspect failed")
    return json.loads(proc.stdout)[0]


def redacted_cmd(cmd):
    result = list(cmd)
    for index, value in enumerate(result):
        if value in SECRET_FLAGS and index + 1 < len(result):
            result[index + 1] = "<REDACTED>"
        elif any(value.startswith(flag + "=") for flag in SECRET_FLAGS):
            result[index] = value.split("=", 1)[0] + "=<REDACTED>"
    return result


def api_key(cmd):
    return cmd[cmd.index("--api-key") + 1]


def call(port, key, route, payload=None, content_type="application/json", timeout=30):
    headers = {"Authorization": "Bearer " + key}
    if payload is not None:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=payload,
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read()


def json_call(port, key, route, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    status, body = call(port, key, route, data, timeout=timeout)
    return status, json.loads(body)


def safe_error(exc):
    # Exception messages and HTTP response bodies may echo authenticated input.
    if isinstance(exc, urllib.error.HTTPError):
        return {"type": "HTTPError", "status": exc.code}
    return {"type": type(exc).__name__}


def chat(port, key, model):
    payload = {"model": model, "messages": [{"role": "user", "content":
               "Calculate 17 times 23. Answer with the integer only."}],
               "temperature": 0, "max_tokens": 64,
               "chat_template_kwargs": {"enable_thinking": False}}
    status, body = json_call(port, key, "/v1/chat/completions", payload, timeout=120)
    choice = body["choices"][0]
    content = choice["message"].get("content") or ""
    result = {"status": status, "finish_reason": choice.get("finish_reason"),
              "usage": body.get("usage"), "contains_expected_integer":
              bool(re.search(r"(?<!\d)391(?!\d)", content))}
    result["passed"] = (status == 200 and result["contains_expected_integer"]
                        and result["finish_reason"] == "stop"
                        and bool(result["usage"]))
    return result


def embedding(port, key, model):
    status, body = json_call(port, key, "/v1/embeddings",
                             {"model": model, "input": "Synthetic restoration check."},
                             timeout=90)
    vector = body["data"][0]["embedding"]
    finite = all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector)
    norm = math.sqrt(sum(x * x for x in vector)) if finite else None
    result = {"status": status, "vector_dimensions": len(vector),
              "vector_finite": finite, "vector_norm": norm,
              "usage": body.get("usage")}
    result["passed"] = (status == 200 and len(vector) == 1024 and finite
                        and norm is not None and norm > 0 and bool(result["usage"]))
    return result


def asr(port, key, model):
    wav = ASR_WAV.read_bytes()
    digest = hashlib.sha256(wav).hexdigest()
    if digest != ASR_SHA256:
        raise RuntimeError("ASR fixture hash mismatch")
    boundary = "codexrestore" + uuid.uuid4().hex
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"asr-smoke.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode(),
        wav, f"\r\n--{boundary}--\r\n".encode()]
    status, body = call(port, key, "/v1/audio/transcriptions", b"".join(parts),
                        f"multipart/form-data; boundary={boundary}", timeout=120)
    data = json.loads(body)
    transcript = data.get("text", "")
    normalized = " ".join(re.findall(r"[a-z0-9]+", transcript.lower()))
    expected = "hello this is an audio test one two three"
    result = {"status": status, "fixture_sha256": digest,
              "transcript": transcript, "normalized_match": normalized == expected,
              "usage": data.get("usage")}
    result["passed"] = status == 200 and result["normalized_match"]
    return result


def check_service(name, port, model, expected_id, expected_image, snapshot_only,
                  initial, reference=None):
    result = {"container": name, "port": port, "expected_model": model}
    try:
        item = inspect(name)
        cmd = item["Config"]["Cmd"]
        result.update({"id": item["Id"], "image": item["Image"],
                       "image_ref": item["Config"]["Image"],
                       "state": item["State"]["Status"],
                       "started_at": item["State"]["StartedAt"],
                       "restart_count": item["RestartCount"],
                       "cmd_redacted": redacted_cmd(cmd),
                       "original_id_matches": item["Id"] == initial["id"] == expected_id,
                       "original_image_matches": item["Image"] == initial["image"] == expected_image,
                       "original_cmd_matches": redacted_cmd(cmd) == initial["cmd"],
                       "original_restart_count_matches":
                       item["RestartCount"] == initial["restart_count"]})
        if name == "vllm-embedding":
            result["embedding_start_time_matches"] = (
                item["State"]["StartedAt"] == initial["started_at"])
        if reference is not None:
            result["stability_start_time_matches"] = (
                item["State"]["StartedAt"] == reference["started_at"])
            result["stability_restart_count_matches"] = (
                item["RestartCount"] == reference["restart_count"])
        key = api_key(cmd)
        status, _ = call(port, key, "/health", timeout=15)
        result["health_status"] = status
        models_status, models = json_call(port, key, "/v1/models", timeout=15)
        ids = [entry.get("id") for entry in models.get("data", [])]
        result["models_status"] = models_status
        result["model_ids"] = ids
        base_ok = (result["state"] == "running" and result["original_id_matches"]
                   and result["original_image_matches"] and result["original_cmd_matches"]
                   and (name != "vllm-embedding" or
                        (result["original_restart_count_matches"] and
                         result["embedding_start_time_matches"]))
                   and result.get("stability_start_time_matches", True)
                   and result.get("stability_restart_count_matches", True)
                   and status == 200
                   and models_status == 200 and model in ids)
        if not snapshot_only and base_ok:
            start = time.perf_counter()
            try:
                if port in (8000, 8001):
                    result["functional"] = chat(port, key, model)
                elif port == 8002:
                    result["functional"] = embedding(port, key, model)
                else:
                    result["functional"] = asr(port, key, model)
            finally:
                result.setdefault("functional", {})["elapsed_s"] = time.perf_counter() - start
        result["passed"] = base_ok and (snapshot_only or result.get("functional", {}).get("passed") is True)
    except Exception as exc:
        result["error"] = safe_error(exc)
        result["passed"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--initial-state", type=pathlib.Path,
                        default=pathlib.Path("/tmp/qwen-spec-experiment/initial-state.json"))
    parser.add_argument("--stability-reference", type=pathlib.Path,
                        help="Prior JSON snapshot to compare starts and restarts")
    parser.add_argument("--output", type=pathlib.Path, help="Write JSON evidence here as well as stdout")
    args = parser.parse_args()
    evidence = {"observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "mode": "snapshot_only" if args.snapshot_only else "functional"}
    try:
        initial = json.loads(args.initial_state.read_text(encoding="utf-8"))["containers"]
    except Exception as exc:
        evidence["initial_state_error"] = safe_error(exc)
        evidence["services"] = []
    else:
        try:
            reference = None
            if args.stability_reference:
                snapshot = json.loads(args.stability_reference.read_text(encoding="utf-8"))
                reference = {service["container"]: service for service in snapshot["services"]}
                if set(reference) != {service[0] for service in SERVICES}:
                    raise ValueError("stability reference has an incomplete service set")
                if not all("started_at" in service and "restart_count" in service
                           for service in reference.values()):
                    raise ValueError("stability reference lacks start or restart fields")
            evidence["services"] = [check_service(*service, args.snapshot_only,
                                                   initial[service[0]],
                                                   reference[service[0]] if reference else None)
                                    for service in SERVICES]
        except Exception as exc:
            evidence["stability_reference_error"] = safe_error(exc)
            evidence["services"] = []
    evidence["passed"] = (len(evidence["services"]) == len(SERVICES)
                          and all(service["passed"] for service in evidence["services"]))
    rendered = json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        try:
            args.output.write_text(rendered, encoding="utf-8")
        except Exception:
            return 2
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
