# Isolated M64 mainloop model runtime

This is an isolated **negative-result validation**, not a production deployment.
The selected synthetic M16160 complete chain is still about 6% slower than
official GEMM+V2 in the first timing window; the user explicitly requested a
real-model comparison. Root has independently passed the selected v3 library's
13 numerical cases, 21 arms, five-M changed-input Graph gates and unique
SF128 writer proof. This new Python runtime has only local syntax and mock
contract checks so far; its CUDA smoke, fixed-image Inductor check, real model
witness and A/B/A remain separate requirements.

The source was copied from `gemm-epilogue/frozen-runtime-v6`, without any
shared libraries. The existing strict matcher, packing/identity registry,
source/provenance metadata filtering, fake output shape/column-major scale
layout and canonical fallback are retained. The new C ABI bridge replaces
ABI2 with ABI1 and the custom op is `pro6000d_mainloop_gateup.fused`.
Only a live registered target weight with **M=16160** on SM120 uses
`warp24n64_v3`, raster **AlongM=1**, swizzle **1**. Other legal matched M
values use canonical GEMM + unchanged V2. Nonmatching graph paths are not
rewritten. BF16 lm_head is outside the target matcher. Query/launch errors
raise; they do not silently fall back after an authorized candidate hit.

## Frozen contract

The exact library SHA256 is
`a6ae4155f65ec3b992ade3943902ccd88eea76b211dda5ceab26626245b7ae5d`.
The loader requires an absolute file, exact SHA, `sm120_scheduler_abi_version()==1`
and the locked variant, namespace, MMA/permutation/copy and output metadata.
It queries `sm120_fused_fp8_temp_scale_bytes(M)` and
`sm120_fused_fp8_workspace_bytes(M,raster,swizzle)`. Launch has **14** arguments:

```text
out_fp8, out_scales, temp_fp8, temp_scales,
A, Bpacked, As, Bspacked,
M, raster, swizzle, workspace, workspace_bytes, stream
```

Current stream address zero is valid. All outputs, paired temporary, native
scale storage and workspace belong to the invocation and are recorded on
its stream. No global mutable GPU scratch or raster setter is introduced.
The paired temporary remains `[M,34816]` FP8; final output `[M,17408]`
FP8 and FP32 scale `[M,136]`, stride `(1,M)`. The native scale buffer is
sized by query, not inferred from the output shape.

The candidate FX pass UUID is
`pro6000d-mainloop-warp24n64-v3-m16160-r1-s1:<library SHA>`.
The existing `_inserted_meta` filters stale `eager_input_vals`; the new op
continues to accept four tensor arguments and return two output tensors.
Weight authorization checks exact live canonical weight+scale storage,
dtype, shape, stride and storage offset, with strong references preventing
allocator-pointer reuse. Shape alone does not authorize packed weights.

## Environment and instrumentation

Place this directory first on `PYTHONPATH`, normally `/instrument`. The
shared library is supplied separately as `/instrument/libgateup.so`.
Use the following exact keys; M/raster/swizzle/variant are fixed in code,
with no additional tuning environment variables:

| Key | Value / scope |
|---|---|
| `PRO6000D_MAINLOOP_MODE` | `baseline` for A1/A2, `candidate` for B; optional `probe` records FX without replacing |
| `PRO6000D_MAINLOOP_EVIDENCE_DIR` | Absolute writable directory, normally `/profiles`, all arms |
| `PRO6000D_MAINLOOP_V2_SITECUSTOMIZE` | Absolute existing V2 entry, normally `/instrument/v2/sitecustomize.py`, all arms |
| `PRO6000D_MAINLOOP_LIBRARY` | `/instrument/libgateup.so`, candidate |
| `PRO6000D_MAINLOOP_LIBRARY_SHA256` | The exact SHA above, candidate |
| `PRO6000D_MAINLOOP_VALIDATED_LIBRARY_SHA256` | Same exact SHA, candidate |
| `PRO6000D_MAINLOOP_DISPATCH_EVIDENCE` | `1` for first Python dispatch/capture evidence, normally witness only |

All arms also use the same prior V2 keys:
`PRO6000D_FP8_FUSION_ENABLE=1`,
`PRO6000D_FP8_FUSION_LIBRARY=/instrument/v2/libquant.so`, and
`PRO6000D_FP8_FUSION_LIBRARY_SHA256=087b7caa92c4742ece3875b6dd1421e25360b834ce4106f7fa166f90bc933b3f`.
Python imports one `sitecustomize`, so the new entry explicitly runs V2.
An unset MAINLOOP mode installs no experimental vLLM import hooks.

Both arms hold canonical and identical packed duplicates. The expected fixed
model has 64 main-model gate/up modules plus 1 MTP: packed additional bytes
**11589593600**, independently checked at startup. Keep the KV budget at
**12884901888 B = 12 GiB**, same model/image/MTP/cache/compile options and
prompts in A1/B/A2. An unused library in A need not have B's SHA, but all
executed shared runtime source hashes must match. This extra packed-memory
cost remains even for the baseline and is not an optimization benefit.

First-dispatch label is `mainloop_warp24n64_v3_m16160_r1_s1`; fallback label
is `canonical_v2_fallback`. Candidate temporary evidence has exactly
`paired_fp8_temp`, `native_scale_temp`, `cutlass_workspace`.
Records include variant/target/raster/swizzle/library SHA and exact registered
canonical+packed pointers. They are deduplicated by module/M/branch and do
**not** count API requests or CUDA Graph replay. Expected candidate kernel
markers are `MainloopPairedGateUpFp8_warp24n64_v3` and
`mainloop_pack_gateup_fp8_warp24n64_v3`; both kernels must be demonstrated
on the actual model path, with successful direct launch and Graph replay
correlations reported separately. Do not infer exact M from a kernel name
or grid without same-run dispatch/configuration evidence.

## Checks and execution scope

`abi_cpu_selfcheck.py` uses metadata-only fake tensors and mocked C entry
points. Its saved `abi-cpu-selfcheck.json` passed 14-argument/query/stream0,
registered target, unregistered target, M1/5/256/2741/32768 fallback and
wrong SHA/M/raster/swizzle/ABI-length/null-buffer/v2-variant/input-dtype
rejections. All 13 Python sources passed local `py_compile`; no `.so`
was copied. These checks do not load torch/CUDA and prove no numerical,
Graph, model or performance result. The local Mac Python has no torch.

Root must run `inductor_cpu_selfcheck.py` and
`serialization_selfcheck.py` in the fixed image with MAINLOOP mode unset.
These are CPU checks of the real torch schema, metadata, serialization and
Inductor stack, distinct from this local mock.

Then, in an owned isolated GPU container with candidate environment above:

```sh
python3 /instrument/runtime_smoke.py --output /profiles/runtime-smoke.json
```

The smoke has a fixed target M, no `--m` option. It checks FP8 bytes and
FP32 scale bits for registered M16160, unregistered M16160, and registered
fallback M1/5/256/2741/32768. It then validates fresh capture output,
changes input bytes and scales, replays and compares output with a fresh
canonical+V2 computation. This synthetic direct-op smoke does not start
or stop services and is not a model benchmark.

Only after smoke and real-model FX/identity/dual-kernel dispatch witness
should root run unprofiled A1/B/A2. Report the observed negative or positive
result against baseline drift, separate decode hash relationships, and
limitations. Five fixed quality cases without stored payload hashes establish
their own named-case results, not captured input-byte equality. Empty
reasoning hashes are not quality evidence. Preserve embedding; keep the
other original services stopped. No automatic adoption or deployment is
performed by these files.
