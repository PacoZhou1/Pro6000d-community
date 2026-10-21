"""Launch one loopback-only profiling service; production containers untouched.

Run on the GPU host. The temporary service deliberately has no production API
key in its argv: Nsight records process command lines. Bind to localhost only.
"""
import argparse
import json
import pathlib
import subprocess

IMAGE = "sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b"
p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--method", choices=["mtp", "dflash"], default="mtp")
p.add_argument("--tokens", type=int, default=4)
p.add_argument("--profiler", choices=["torch", "cuda", "none"], default="cuda")
p.add_argument("--output-dir", required=True)
p.add_argument("--nsys-root")
p.add_argument("--nsys-relative", help="Path to nsys relative to the mounted root")
p.add_argument("--extra-env", action="append", default=[])
p.add_argument("--extra-arg", action="append", default=[])
p.add_argument("--image", default=IMAGE, help="Explicit immutable experimental image ID")
p.add_argument("--instrument-dir")
p.add_argument("--ncu-root")
p.add_argument("--eager", action="store_true")
a = p.parse_args()
assert a.name.startswith("codex-fp8-")
running = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).splitlines()
if any(n.startswith(("codex-fp8-", "codex-gemm-")) for n in running):
    raise RuntimeError("another profiling arm is running")
if any(n in running for n in ["vllm-qwen-27b", "vllm-qwen3.5-9b", "vllm-qwen3-asr"]):
    raise RuntimeError("pause the non-embedding original models before launch")
original = json.loads(subprocess.check_output(["docker", "inspect", "vllm-qwen-27b"]))[0]
cmd = original["Config"]["Cmd"][:]
i = cmd.index("--api-key")
del cmd[i:i + 2]
def change(flag, value):
    if flag in cmd:
        cmd[cmd.index(flag) + 1] = str(value)
    else:
        cmd.extend([flag, str(value)])
for flag, value in [("--gpu-memory-utilization", .70), ("--max-num-batched-tokens", 32768),
                    ("--max-model-len", 262144), ("--max-num-seqs", 16),
                    ("--host", "127.0.0.1"), ("--port", 18000),
                    ("--kv-cache-dtype", "fp8"), ("--attention-backend", "FLASHINFER")]:
    change(flag, value)
spec = {"method": a.method, "num_speculative_tokens": a.tokens}
if a.method == "dflash":
    spec.update(model="/draft", attention_backend="FLASHINFER")
change("--speculative-config", json.dumps(spec))
if a.profiler != "none":
    config = {"profiler": a.profiler}
    if a.profiler == "torch":
        config.update(torch_profiler_dir="/profiles", torch_profiler_record_shapes=True,
                      torch_profiler_with_stack=False, torch_profiler_with_memory=False,
                      torch_profiler_use_gzip=True)
    change("--profiler-config", json.dumps(config))
cmd += a.extra_arg
if a.eager:
    cmd += ["--enforce-eager"]
model = next(m["Source"] for m in original["Mounts"] if m["Destination"] == "/model")
pathlib.Path(a.output_dir).mkdir(parents=True, exist_ok=True)
create = ["docker", "create", "--name", a.name, "--gpus", "all", "--ipc", "host",
          "--network", "host", "--restart", "no", "--memory=40g", "--memory-swap=40g", "--ulimit", "memlock=-1:-1",
          "--mount", f"type=bind,src={model},dst=/model,readonly",
          "--mount", f"type=bind,src={a.output_dir},dst=/profiles"]
if a.instrument_dir:
    create += ["--mount", f"type=bind,src={a.instrument_dir},dst=/instrument,readonly", "--env", "PYTHONPATH=/instrument"]
for env in ["HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "VLLM_USE_V2_MODEL_RUNNER=1",
            "VLLM_WORKER_MULTIPROC_METHOD=spawn"] + a.extra_env:
    create += ["--env", env]
if a.method == "dflash":
    create += ["--mount", "type=bind,src=/home/llm/models/Qwen3.6-27B-DFlash,dst=/draft,readonly"]
if a.profiler == "cuda" and a.ncu_root:
    create += ["--mount", f"type=bind,src={a.ncu_root},dst=/ncu,readonly", "--cap-add=SYS_ADMIN", "--entrypoint", "/ncu/opt/nvidia/nsight-compute/2026.3.0/ncu"]
    metrics = "gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
    wrapped = ["--target-processes", "all", "--clock-control", "none", "--cache-control", "none", "--profile-from-start", "off", "--launch-count", "12", "--nvtx", "--nvtx-include", "regex:GEMM_ATTRIB:fp8_block_cutlass_gemm:language_model.model.layers.0.mlp.gate_up_proj:.*|GEMM_ATTRIB:unquantized_embedding_gemm:language_model.lm_head:.*/", "--metrics", metrics, "--export", "/profiles/counters", "--force-overwrite", "vllm", "serve"] + cmd
elif a.profiler == "cuda":
    if not a.nsys_root or not a.nsys_relative:
        raise ValueError("CUDA capture requires Nsight root and relative binary")
    create += ["--mount", f"type=bind,src={a.nsys_root},dst=/nsys,readonly",
               "--entrypoint", "/nsys/" + a.nsys_relative]
    wrapped = ["profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
               "--trace-fork-before-exec=true", "--cuda-graph-trace=node",
               "--capture-range=cudaProfilerApi", "--capture-range-end=repeat:sync",
               "--output=/profiles/capture", "vllm", "serve"] + cmd
else:
    create += ["--entrypoint", "vllm"]
    wrapped = ["serve"] + cmd
proc = subprocess.run(create + [a.image] + wrapped, capture_output=True, text=True)
if proc.returncode:
    raise RuntimeError("temporary container create failed: " + proc.stderr)
print(subprocess.check_output(["docker", "start", a.name], text=True))
