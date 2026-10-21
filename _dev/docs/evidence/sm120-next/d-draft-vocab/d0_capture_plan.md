# D0 capture plan (for a root operator; NOT executed here)

## Important prior finding: a normal-graph capture already exists

`evidence/profiling/raw/mtp4-capture.3.sqlite` and `mtp4-capture.4.sqlite` were taken with the
production-like flags (CUDA graphs on, `--cuda-graph-trace=node`, MTP4, V2 runner). Running
`d0_lm_head_steps.py` on them gives (see `d0_result_*.json`):

| capture | mode | decode steps | lm_head-shaped kernels per step | mean us |
|---|---|---|---|---|
| mtp4-capture.3 | graph replay (95.8% of kernels carry graphId) | 83 | 4 x gridY=1940 (M=1-like) + 1 x gridY=970 (M=5-like) in 83/83 steps | 1575.96 / 1569.04 |
| mtp4-capture.4 | graph replay | 84 | same, 84/84 | see json |
| gemm-targeted/raw/capture.2 | eager | 87 | 4 + 1 in 87/87 (348 / 87 kernels, 25.54% of GPU kernel sum) | 1580.7 / 1572.0 |

Graph structure observed in capture.3: the M=5 verification lm_head launches eagerly
(`cuLaunchKernel`, graphId NULL); the first draft step is one graph (graphId 10577, 1 lm_head);
the remaining three draft steps are one graph (graphId 10598) launched 3 times per step, each with
1 lm_head. So the claim "4 draft M=1 + 1 verify M=5 per step" already holds in graph mode.

Differences eager vs graph for the analyzer: eager has per-kernel `GEMM_ATTRIB` NVTX and
`cuLaunchKernel` per kernel; graph replay has no per-kernel NVTX, kernels of one `cudaGraphLaunch`
share a correlationId + graphId. The stage NVTX (`execute_context_0(0)_generation_1(5)`) closes
BEFORE logits/draft launches, so the analyzer uses windows [stage_i.start, stage_{i+1}.start).
Shape is inferred (no N/K in kernel names): BF16 GEMM-named, >=1 ms, signatures reported verbatim.

Residual gap the old captures do not cover: they were taken with `--max-num-seqs 16`, one request at a
time, fp8 KV, and pre-date any patch. The capture below is only needed (a) as a fresh-day baseline
for the D4 byte-reduction proof, and (b) after the patch, to prove the draft lm_head kernel shrinks.

## Command (baseline or candidate; re-use the existing harness)

Run on the GPU host as root, following the 15-doc service policy (embedding stays up, record its
ID/start/restart count/health + one real vector request before and after; 27B/9B/ASR stay stopped;
container deleted afterwards). Nsight Systems 2026.5.1 as in `evidence/profiling/nsight-install.json`.

```
python3 evidence/profiling/profile_control.py \
  --name codex-profile-d0-mtp4 --method mtp --tokens 4 --profiler cuda \
  --output-dir /home/llm/profiles/d0 \
  --nsys-root <NSYS_ROOT> --nsys-relative <relative path to nsys>
# candidate arm additionally: --extra-env VLLM_DRAFT_VOCAB_IDS=/model-extra/subset_32768.json  (see d2_design.md)
```

which expands to (from `mtp4-runtime.json`):

```
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --trace-fork-before-exec=true \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=repeat:sync \
  --output=/profiles/capture \
  vllm serve /model --served-model-name qwen3.6-27b-fp8 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7 --max-num-seqs 16 --max-model-len 262144 --quantization fp8 \
  --kv-cache-dtype fp8 --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --max-num-batched-tokens 32768 --host 127.0.0.1 --port 18000 --attention-backend FLASHINFER \
  --profiler-config '{"profiler":"cuda"}'
env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
```
Image: `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`.
Do NOT pass `--enforce-eager` (that is the 09-doc attribution mode, not this test).

Request sequence (reuse `profile_client.py`, same as `mtp4-capture-requests.jsonl`): one warmup
decode request (4.8K prompt / 256 completion), then 2 profiled decode requests, each bracketed by the
profiler start/stop endpoints so `repeat:sync` yields capture.1..N. Exports:
`nsys export --type sqlite capture.N.nsys-rep`.

## Analysis (offline, any machine)

```
python3 -I d0_lm_head_steps.py capture.N.sqlite --out d0_result_capture.N.json
```
Pass criteria for D0: histogram shows `{M1-like:4, M5-like:1}` in >=95% of decode steps with the
M1-like kernel mean ~1.58 ms. If instead draft steps do not show a per-step 248320-row projection,
stop task D and record it (plan section 4 D0). Verified here on existing data: criterion met.

## Operator checklist
1. Embedding baseline record (ID, StartedAt, RestartCount, health, one vector request).
2. Optional `nvidia-smi -lgc` lock; record success. Log SM clock during capture anyway.
3. Launch, warm up, capture 2 decode windows, export SQLite, copy SQLite + `.nsys-rep` + runtime JSON back.
4. Delete container; embedding record again; hash all files (SHA256).
