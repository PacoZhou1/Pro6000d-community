"""Capture non-secret runtime provenance for a named benchmark container."""
import argparse
import datetime
import json
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument("container")
a = p.parse_args()
d = json.loads(subprocess.check_output(["docker", "inspect", a.container], text=True))[0]
cmd = list(d["Config"]["Cmd"])
secrets = []
for i, value in enumerate(cmd):
    if value in ["--api-key", "--hf-token"]:
        secrets.append(cmd[i + 1])
        cmd[i + 1] = "<REDACTED>"
log = subprocess.run(["docker", "logs", "--tail", "1500", a.container], text=True, capture_output=True)
selected = []
pattern = r"(Using.*backend|Using.*kernel|Using.*Marlin|Using.*Flash|FlashInfer resolved|attention block size|KV cache layout|uncalibrated|scaling factor|GDN prefill|Capturing.*graphs|GDN decode kernel|GPU KV cache size|Available KV|Model loading took|max_num_batched_tokens|CUDAGraphMode|Estimated CUDA graph|CUDA graph pool|Profiling CUDA graph|Engine core initialization failed|ValueError|RuntimeError|ERROR)"
for line in (log.stdout + log.stderr).splitlines():
    if re.search(pattern, line) and not any(x in line for x in ["api_key", "api-key", "non-default args", "Initializing a V1"]):
        for secret in secrets:
            line = line.replace(secret, "<REDACTED>")
        selected.append(line)
version_code = """import importlib.metadata as m,json
out={}
for p in ['vllm','torch','transformers','flashinfer-python','flashinfer-cubin','nvidia-cutlass-dsl','b12x']:
 try:out[p]=m.version(p)
 except m.PackageNotFoundError:out[p]=None
print(json.dumps(out))"""
v = subprocess.run(["docker", "exec", a.container, "python3", "-c", version_code], text=True, capture_output=True)
im = json.loads(subprocess.check_output(["docker", "image", "inspect", d["Image"]], text=True))[0]
print(json.dumps({"observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "container": a.container, "image_ref": d["Config"]["Image"], "image_id": d["Image"],
                  "repo_digests": im.get("RepoDigests"), "cmd_redacted": cmd,
                  "state": {k: d["State"][k] for k in ["Status", "StartedAt", "FinishedAt", "ExitCode"]},
                  "model_mount_read_only": all(not x["RW"] for x in d["Mounts"] if x["Destination"] == "/model"),
                  "selected_logs": selected[-80:], "packages": json.loads(v.stdout) if v.returncode == 0 else {"unavailable": True},
                  "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,power.draw,utilization.gpu", "--format=csv"], text=True)}, ensure_ascii=False, indent=2))
