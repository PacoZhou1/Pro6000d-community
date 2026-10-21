# Fixed-image GEMM attribution and targeted experiments

The Chinese result report is [09-targeted-gemm.md](../../09-targeted-gemm.md).
`module-signatures.json` and `mtp4-*-attribution.json` describe real eager
model requests. `counters/summary.json` describes **synthetic operands at
those measured shapes**, not real-checkpoint hardware counters. The attempted
full-model NCU capture OOMed; `ncu-model-failure.json` preserves that failure.
The event timings in `micro-*` are synthetic operator comparisons, while
`api-*` records contain actual loopback model API requests.

Candidate code is experimental and is not installed in a production image:

- `targeted_kernel.py`: fixed source hash, original dispatch versus gate/up
  Pingpong tile, and a standalone C ABI build.
- `splitn_probe.py` / `splitn_runtime.patch`: split-N plus concatenation;
  measured slower, never installed in a model service.
- `down_streamk_builder.py`: fixed-tile SM120 Stream-K down projection.
- `runtime_down_streamk/sitecustomize.py`: reversible CUDA custom-op wrapper
  for a temporary vLLM container. Only the validated M=5 shape takes Stream-K.
  Other M values use the official operation inside the opaque custom op.
  Each call owns its output/workspace; the host C ABI is protected by a lock.
- `runtime_smoke.py`: real GPU numerical, fallback, non-target-to-target
  dynamic compile, CUDA graph changed-input replay, and two-stream checks.

The old `runtime_guard_v1.py` is a preserved failed integration, not an
alternative to install. `api-streamk/runtime-guard-miss.json` proves its
compiled model graph contained no experimental op. Its apparently normal API
numbers must not be presented as a candidate performance result.

Raw SQLite, NCU reports, and compiled graph text are retained under ignored
`raw/` / report paths and indexed by `raw-artifact-manifest.json`; large raw
binaries are not committed. Remote builds and captures remain beneath
`/home/llm/.local/share/qwen-gemm-targeted/` for reproduction. Model
weights stay in the original model directory and are unchanged.

## Restoration and reproduction boundaries

`restore-sequence.json`, `restoration.json`, and `restoration-stability.json`
record the sequential restart, actual functional APIs, and a later 309.58-second
stability check. `restore_sequence.py` reuses
[`../specdecode/restore_verify.py`](../specdecode/restore_verify.py), copied to
`/tmp/qwen-gemm-targeted/restore_verify.py` on the server; both copies have SHA256
`125d7d830075942a2b62de4489726848a0326c29769fe617633a627ef46cdd44`.
The ASR fixture remains at `/tmp/qwen-spec-experiment/asr-smoke.wav`, with its
hash checked by that verifier. `cleanup.json` records deletion of this run's
seven stopped temporary model containers. All original services are running.

`embedding-monitor.jsonl` contains 203 server-local observations across
101.07 minutes; `embedding-monitor-summary.json` records the source hash,
maximum sample gap and unchanged container identity. This is availability
sampling, not an embedding latency benchmark. The monitor has been stopped
using its own stop file; the embedding service was never stopped.

For offline inspection, `summarize_micro.py` and `summarize_api.py` regenerate
their JSON summaries from the sibling evidence directories. Run the attribution
analyzer on a saved raw SQLite file, or `analyze_replay.py CAPTURE.sqlite
--output RESULT.json` for the real graph witness. Consult
`experiment-manifest.json` and `raw-artifact-manifest.json` for exact inputs and
hashes. GPU reproduction requires the pinned SM120 image and original source;
`probe_runner.py` and `run_api_arm.py` record the exact experimental commands
in their status/runtime artifacts. They intentionally require isolated
temporary containers and must not be aimed at a production service.

## Real module attribution

`attribution_patch.py` is an opt-in diagnostic shim for the pinned vLLM
`0.31.0` image `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`.
The installed source files and their SHA256 values are recorded in
`source-manifest.json`; the local `source/` copy is ignored by Git. The shim
does not edit vLLM or model weights and must only be mounted into a temporary
attribution experiment.

Copy the shim to an experiment-only directory as `sitecustomize.py`, mount it
read-only at `/instrument`, and pass `PYTHONPATH=/instrument` together with
`GEMM_ATTRIBUTION_ENABLE=1` and a writable
`GEMM_ATTRIBUTION_OUTPUT_DIR=/some/experiment/path` to **all** Python worker
processes. Use `--enforce-eager` for the first attribution run. The launcher
owns container/service lifecycle; this shim performs none. The output is one
`gemm-attribution.<pid>.jsonl` file per process. A valid run must show
`patch_installed` for the FP8 method, CUTLASS block-FP8 kernel method,
unquantized linear and embedding methods, and logits processor in the engine
worker; frontend-only records do not establish attribution.

The fixed source provides a `prefix` for each linear layer and the `lm_head`.
`Fp8LinearMethod.apply` records that prefix and input/weight/scale tensor
metadata. Its active CUTLASS block-FP8 path calls
`CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm`; a separate
`GEMM_ATTRIB:fp8_block_cutlass_gemm:<prefix>:M...:N...:K...` NVTX range encloses
the actual GEMM call, with `A`, `B`, and scale shape/stride/dtype in JSONL. In
this path `A` is `[M,K]`, `B` is `[N,K]`, and the fixed CUTLASS adapter passes
`B.T` to `cutlass_scaled_mm`. Unquantized linear and embedding method calls
get their own NVTX ranges. `ParallelLMHead.forward` is not called for logits;
the fixed `LogitsProcessor._apply_head` calls the head's quantization method.
The alternate `head_dtype` path is covered separately. A recorded `lm_head`
prefix together with `M/N/K` and the exact NVTX→CUDA launch→GPU kernel
correlation is evidence for its BF16 GEMM; shape coincidence by itself is not.

Capture CUDA and NVTX with Nsight Systems. Join a `GEMM_ATTRIB:` NVTX CPU range
to a CUDA runtime launch whose **start lies inside that range on the same
`globalTid`**, then join that runtime call to a GPU kernel using the same
`correlationId` and the process portion of `globalTid`. The fixed Nsight
export's process mask is `globalTid & ~0xFFFFFF`; validate against
`PROCESSES.globalPid`. Preserve unmatched kernels as unassigned. For split-K,
one labeled call can launch multiple GPU kernels. Keep full raw names, operands
and shape metadata in the evidence.

After exporting a capture to SQLite, run `python3 analyze_attribution.py
capture.sqlite --label mtp4-eager --output mtp4-eager-attribution.json`. The
offline analyzer reports exact matches by `(kind, module_prefix, M, N, K,
enclosing_stage)` and retains each raw kernel name. It also reports BF16-named
GEMM kernel count/duration for all, attributed, and unmatched kernels. The
`enclosing_stage` is present only when a vLLM stage NVTX range on the same
thread contains the GEMM range; null is intentionally unclassified. Use
`--pid` if the SQLite includes multiple GPU processes. A BF16 kernel under
`lm_head` is supported by the NVTX/API correlation and tensor metadata,
whereas the identical BF16 name without such correlation remains unassigned.

Python hooks may run only during compilation or graph capture. CUDA Graph
replays execute GPU work without re-entering the patched Python methods;
therefore this shim deliberately skips logging/NVTX while Dynamo is compiling.
An eager run proves module and kernel identity for its actual request, while
normal CUDA Graph Nsight captures retain the performance evidence. Eager
latency must not be compared with normal-mode latency. A torch.profiler CPU
module range has the same replay limitation; Nsight CUDA API correlation in an
eager run provides the firmer module→kernel link.
