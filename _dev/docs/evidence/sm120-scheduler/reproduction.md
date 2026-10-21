# Reproducing the SM120 official scheduler experiment

This is an isolated FP8 gate/up experiment on the RTX PRO 6000 Blackwell
Workstation Edition (SM120). It keeps the fixed vLLM 0.31.0 source
`db9527a46873454610df6dbedf79a36d6bf1a7f6`, FP8 blockwise inputs,
BF16 GEMM output rounding, and the already validated V2 SiLU/per-128 FP8
quantizer. The candidate changes only the official M128N128 GEMM tile
scheduler to `AlongM`, swizzle 1, for **M exactly 16160**. All other M,
including M32768 and decode, use the original canonical GEMM + V2 quantizer.
This is not the paired-weight fused epilogue candidate. Both model arms retain
the same paired-weight copies solely to equalize memory pressure.

## Immutable inputs and environment

| Item | Pinned value |
|:---|:---|
| Experimental image | `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b` |
| PyTorch / CUDA build | `2.13.0+cu130` / CUDA 13.0; observed runtime 13000, driver 13020 |
| CUDA compiler | nvcc 13.0.88, `-arch=sm_120a`, no `--use_fast_math` |
| Scheduler C ABI | 1; private namespace `pro6000d_scheduler_v1` |
| Scheduler library | `af1d814966a746ff09edc55d40196b871067d88d22ed0e50700c1f3a4eb185d6` |
| Old M128N128 GEMM control | `f7abfcd8c04ad9a93c924364beadd305f773b1f909bca013f2838d109f17b7b2` |
| V2 quant library | `087b7caa92c4742ece3875b6dd1421e25360b834ce4106f7fa166f90bc933b3f` |
| Old pipeline probe | `eed1a8d2fa5e97d0b0dfefcb504c6ca9d8493d1c00997788e1343aadd60c42ae` |

The [build manifest](build-v3/build-manifest.json) records every fixed v6
source/header SHA, generated shadow header SHA, exact nvcc argv and shared
library SHA. The [official-tuning metadata](official-tuning-v2/analysis.json)
records the runtime versions and all probe binary hashes. Do not use an
unverified installed CUTLASS tree or substitute a later vLLM release.

The GPU host's experiment root in the recorded runs was
`/home/llm/.local/share/qwen-sm120-scheduler`. `/work` below refers
to its mounted `work` directory **inside the immutable image**. These
commands are templates for a new, uniquely named owned container after
checking the recorded mounts; the complete actual `docker run` argv is in
each result directory's `status.json`. No password or production API key is
needed or recorded by these probes.

## Build and numerical gates

Build in an owned CPU-only container (`--runtime runc`, no `--gpus`) using
the exact command recorded in [build-v3/command.json](build-v3/command.json):

```sh
python3 /work/sm120-scheduler/build_scheduler.py \
  --v6-source-dir /work/v6-source \
  --cutlass-include /usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include \
  --build-dir /work/build-v3
```

Require compile exit 0 and the library SHA above before any ctypes launch.
The old v1 library had an ELF/TLS symbol collision; use only the private
namespace v3 build. The `build_scheduler.py` source checks exact input
header hashes before emitting, and the manifest records generated hashes.

The GPU probes are **serial**: do not overlap them with another GPU probe or
model A/B arm. Run the recorded [scheduler matrix command](scheduler-matrix-v4/command.json)
and [official tuning command](official-tuning-v2-command.json) through an
owned container with the `/work`, `/oldwork`, `/baseline`, and `/v2` mounts
shown in their respective `status.json` files. Both commands pin every
loaded library SHA and use 31 randomized matched graph pairs. The matrix
tests the six raster/swizzle fused configurations plus old/v6 anchors, full
bitwise numerical gates, changed-input graph replay, and warm/owned-cache-
disturbed timing. The official tuning probe is the decisive same-process
three-arm comparison: old H8, new-library H8, new-library AlongM/s1, each
the complete GEMM + unchanged V2 quantizer on identical canonical inputs.
Its default shapes are M2741/2786/16160/32768, released one by one.

Analyze the original JSONL, rather than selected medians, with:

```sh
python3 /work/sm120-scheduler/analyze_results.py /results/stdout.jsonl
python3 /work/sm120-scheduler/analyze_official_tuning.py /results/stdout.jsonl
```

`stdout.log` may include container/profiler messages; extract only JSON
object lines if the controller has not already saved a pure JSONL. The
analyzers reject missing shapes, hash/seed mismatches, incomplete BF16/FP8/
scale or graph witnesses, and fewer than 31 matched pairs. The recorded
[official tuning result](official-tuning-v2/analysis.json) passed these gates:
at M16160, warm new-H8 10.432 ms versus AlongM/s1 10.227 ms, while M32768
AlongM/s1 was slower (33.038 versus 21.954 ms). This is synthetic operator
timing, not a model speedup claim. Owned cache flushing does not guarantee a
fixed cold L2 state.

For hardware counters, run only one selected arm/shape with the exact
[NCU command](ncu-m16160-official-along-m-s1/status.json), including
`--profile-from-start off`, NVTX `SM120_SCHEDULER`, and
`--cache-control none`. This counter run is diagnostic and must not be mixed
into unprofiled latency A/B. The [NCU raw CSV](ncu-m16160-official-along-m-s1/raw.csv)
and [official control CSV](ncu-m16160-official/raw.csv) retain launch and
hardware metrics; `cache-control none` is not a cold-cache guarantee.
The two strict M32768 captures, [Heuristic/s8](ncu-m32768-official-heuristic-s8/raw.csv)
and [AlongM/s1](ncu-m32768-official-along-m-s1/raw.csv), are audited in the
[large-M NCU summary](ncu-large-m-summary.json). Both passed capture/export,
source/hash, numerical, graph and window checks on matching input fingerprints.
Their complete GEMM + V2 quantizer times were 20.037 and 32.887 ms,
respectively. The pinned-source [scheduler shape notes](scheduler-shape-notes.md)
show that Heuristic resolves to AlongM at both M16160 and M32768; this
comparison changes the swizzle, not the raster direction. These counters
do not identify which input buffer caused the extra M32768 DRAM traffic.

## Isolated runtime and model witness

Copy [model-runtime](model-runtime/README.md) into a new instrument directory,
place it first on `PYTHONPATH`, and use the exact A/B environment described
there. Both arms run startup packing and the same V2 sitecustomize. Only B
sets `PRO6000D_OFFICIAL_MODE=candidate` and the validated scheduler library
path/SHA; A sets `baseline`. The C ABI bridge verifies ABI1, N/K/SM120 and
the library SHA. The opaque op guards **`M == 16160`** plus exact canonical
weight/scale storage identity and SM120. It then calls only
`sm120_official_bf16` with `(raster=AlongM, swizzle=1)` and V2 quant. The
post-grad pass UUID includes the library SHA to isolate the compile cache.

Before launching the model, run the recorded
[runtime smoke command](runtime-smoke-v1/command.json) in an owned GPU
container with the source and V2/library mounts from its `status.json`.
Its [result](runtime-smoke-v1/runtime-smoke.json) passed FP8-byte and
FP32-scale-bit equality for target M16160, small M5, other prefill M2741,
and unregistered M16160, plus changed-input CUDAGraph replay against a
fresh official reference. This proves only synthetic direct-op semantics.

The model controller is [model_sequence.py](model-control/model_sequence.py)
and the per-arm launcher is [launch_model_arm.py](model-control/launch_model_arm.py).
The witness invocation uses a fresh label, for example:

```sh
python3 /home/llm/.local/share/qwen-sm120-scheduler/work/sm120-scheduler/model-control/model_sequence.py \
  --mode witness --label model-witness-new
```

It creates only owned temporary containers. Its launcher removes any API key
from the copied model command before profiling, binds the temporary API to
loopback, pins the image and an equal 12,884,901,888-byte KV pool, and
records instrument/library hashes. The original 27B, 9B and ASR services
must remain stopped as the user requested; do **not** restore them at cleanup.
The embedding service stays running, with sampled identity and health checked
before, during and after each arm. Existing services are never stopped by
these commands. Review [witness integration](model-witness-v1/witness/integration-summary.json)
and its startup-pack, FX rewrite and dispatch files: 65 canonical module
identities at M16160 used the candidate branch; every observed other M,
including M32768, took canonical fallback; A/B each hold 65 packed copies,
adding 11,589,593,600 bytes. The [monitor summary](model-witness-v1/monitor-summary.json)
reports 12 healthy samples. Nsight's exported SQLite is checked with
`model-control/analyze_dispatch.py`; distinguish a real direct CUDA launch
from a graph replay, and require exactly 65 private official BF16 kernels.
The retained [dispatch summary](model-witness-v1/witness/dispatch-summary.json)
reports 65 private official BF16 kernels, all 65 with an exact successful
direct-launch correlation and none with a graph node. A Python dispatch or
capture log alone does not prove those launches.

```sh
python3 /work/sm120-scheduler/model-control/summarize_integration.py /profiles
python3 /work/sm120-scheduler/model-control/analyze_dispatch.py \
  /profiles/dispatch.sqlite --output /profiles/dispatch-summary.json
```

Run these against the *same* fresh witness directory and retain the Nsight
report/SQLite SHA. The integration summary proves observed module identity
and Python dispatch, while the SQLite correlation proves actual CUDA launches;
neither source alone establishes both facts.

For the no-profiler model comparison, use the same controller with a fresh
label and `--mode aba`. It serially runs A1, B, A2 with identical payload
generation, prefill/decode cases and quality/open-output probes. The completed
[A/B/A run](aba-official-v1/sequence.json) retains all three arms. Recompute
its [API analysis](aba-official-v1/analysis.json) and then the
[model contract audit](aba-official-v1/model-contract.json) from the repository
root with the analyzers' verified `--root`, `--output` and positional `root`
arguments. Keep both new outputs in a temporary review directory:

```sh
review_dir=$(mktemp -d /tmp/sm120-aba-review.XXXXXX)
python3 _dev/docs/evidence/gemm-epilogue/analyze_api.py \
  --root _dev/docs/evidence/sm120-scheduler/aba-official-v1 \
  --output "$review_dir/analysis.json"
cp -R _dev/docs/evidence/sm120-scheduler/aba-official-v1 "$review_dir/input"
cp "$review_dir/analysis.json" "$review_dir/input/analysis.json"
python3 _dev/docs/evidence/sm120-scheduler/model-control/analyze_model_contract.py \
  "$review_dir/input"
```

`analyze_model_contract.py` reads `root/analysis.json` and **writes**
`root/model-contract.json`; the temporary copy is necessary to keep the
recorded evidence unchanged. `analyze_api.py` embeds absolute source file
paths in its output. A local regeneration therefore need not have the same
file SHA as the retained server analysis below; compare validated results and
source JSONL hashes instead. In this working tree, the commands were checked
CPU-only: both analyzers passed, and regenerated `analysis.json` did match the
retained SHA below. They do not need the server or GPU.

The first command requires each arm's `prefill.jsonl`, `decode.jsonl`,
`quality.jsonl` and `open-output.jsonl`. It excludes warmup index 0 and
requires 12 formal prefill and 6 formal decode requests per arm, matching
indexed prefill/decode payload hashes, valid metrics windows and passing
quality/open-output checks. The second command checks image, model, command,
KV allocation and packed-weight bytes across arms, apart from explicit
candidate opt-in. The retained `analysis.json` SHA256 is
`7c4ce02c1f2111bf707afe7f275555487c4d91b9a832a681e7172563d5180adb`.

All three arms passed. Median prefill TTFT was A1 2.246333 s, B 2.250655 s,
A2 2.262705 s; median decode throughput was 106.728, 106.553 and 106.725
tokens/s, respectively. The 12 index-paired prefill comparisons against
`(A1+A2)/2` gave a median B reduction of 0.144349% (0.152455% over 11
embedding-quiet windows), smaller than A1-to-A2 drift. The
[independent model review](model-review.md) recomputes the raw checks and
limitations: open-output payload hashes match, but the historical quality
probes have no payload hashes, so their request-byte equality is unverified.
This single A/B/A sequence supplies no independent repeated time windows or
credible model-level speedup; the contract records `not_adopted`. Preserve
every arm's original JSONL, service snapshots and embedding monitor file,
including failed attempts, rather than overwriting evidence.

The Nsight Systems report `capture.1.nsys-rep` and exported `dispatch.sqlite`
for the model witness are retained on the GPU server under
`/home/llm/.local/share/qwen-sm120-scheduler/model-witness-v1/witness`.
Their recorded SHA256 values are, respectively,
`ddb9d70bc15ef8628c8c825c6380b24d2469cf5e46f5af5b9afccc14d363e62f`
and `f9499224a9647d52a4113daf676f4c3c56caa6bc3902a85b8a0f9cb0bfd8a93e`
in the local [export status](model-witness-v1/witness/export-status.json).
Both complete, hash-matching binaries are also present in this working tree
at the time of writing, but `*.nsys-rep` and `*.sqlite` are Git-ignored and
are not guaranteed to accompany a fresh checkout. The local SQLite copy can
reproduce the dispatch summary with the `analyze_dispatch.py` command above;
the text exports and summaries alone cannot. Re-exporting SQLite from Nsight
requires the hashed `.nsys-rep` and the pinned Nsight tool, so retain the
server originals independently of this text evidence.
