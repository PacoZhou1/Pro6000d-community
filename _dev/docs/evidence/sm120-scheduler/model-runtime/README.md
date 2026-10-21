# Isolated official scheduler model experiment (unverified runtime)

This directory is a copy of `gemm-epilogue/frozen-runtime-v6`, with only the
candidate C ABI, opaque dispatcher, opt-in names, smoke test and pass UUID
changed. It has **not** been executed on GPU or on the model. The experiment
uses the official M128N128 BF16 CUTLASS GEMM with explicit `AlongM` raster and
swizzle 1 only when the runtime A has exactly **M=16160**, then runs the
unchanged V2 SiLU/per-128 FP8 quantizer. This is a scheduler experiment, not
a new fusion kernel. Other legal matched M values call the canonical fixed
vLLM GEMM and V2 quantizer; nonmatching graph paths are not rewritten.
The M guard lives inside the opaque custom op so a compile-range decision
cannot silently remove the targeted branch.

Both A and B must place this directory first on `PYTHONPATH` and set
`PRO6000D_OFFICIAL_MODE=baseline` or `candidate`, respectively. Both set an
absolute `PRO6000D_OFFICIAL_EVIDENCE_DIR` and
`PRO6000D_OFFICIAL_V2_SITECUSTOMIZE` pointing to the already validated V2
entry, plus the same V2 enablement/library/SHA environment. Python imports
only one `sitecustomize`, so this entry executes V2 explicitly. Both modes
retain canonical weights and the same startup paired-weight duplicates, even
though the new official kernel consumes canonical B/Bs and never reads the
duplicates. Keep KV-pool size, prompts, compilation settings and service
configuration identical; report this extra memory as an experiment cost.

The B arm additionally sets absolute `PRO6000D_OFFICIAL_LIBRARY` and exact
`PRO6000D_OFFICIAL_LIBRARY_SHA256`, with the same SHA in
`PRO6000D_OFFICIAL_VALIDATED_LIBRARY_SHA256`. The runtime verifies SHA and
`sm120_scheduler_abi_version()==1`, fixed N/K/SM120 metadata, and queries
workspace bytes specifically for `(M=16160, AlongM, swizzle=1)`. Every call
owns its BF16 intermediate and workspace on its current stream. No global
CUDA scratch or mutable raster setting is used. Optional
`PRO6000D_OFFICIAL_DISPATCH_EVIDENCE=1` writes first Python dispatch/capture
records; they do not count graph replay or API requests.

The existing strict FX matcher and canonical-weight identity registry are
preserved. Candidate mode registers `pro6000d_official_gateup.tuned`, whose
Meta implementation preserves symbolic output M. Its FX pass UUID is
`pro6000d-official-m16160-alongm-s1-v1:<library SHA>` to force a separate
compile cache identity from v6. An unmatched graph is left untouched; a
matched legal call without target M or registered weight uses canonical
GEMM+V2 fallback. Baseline mode performs startup packing and graph proof,
but installs no replacement.

Before model timing, run `runtime_smoke.py --output /absolute/path.json`
with candidate environment in an isolated GPU container. It checks registered
M16160 FP8 bytes/scale FP32 bits, M5 and M2741 fallback, unregistered M16160
fallback, and changed-input graph replay against a fresh official reference.
Then require a real-model FX rewrite witness and an actual M16160 candidate
kernel witness on the path the service uses: a direct execution trace if the
large-prefill path is direct, or a CUDA Graph replay trace if it is captured.
Record the path type and demonstrate that non-target M values use canonical
fallback. A Python first-dispatch/capture log alone cannot prove repeated
request execution. Require matching output checks and a no-profiler A/B/A
run before interpretation. The current same-library
micro result supports testing **only M16160**: AlongM/s1 was approximately
2% faster than H8 there, while M2741/2786 did not improve and M32768 was
about 50% slower. No result here establishes model speed or quality.

The micro evidence is [official-tuning-v2/analysis.json](../official-tuning-v2/analysis.json):
M16160 warm medians were 10.432 ms for new H8 and 10.227 ms for new
AlongM/s1, while M32768 was 21.954 vs 33.038 ms. Those selected synthetic
shapes are the reason for the exact M guard; no broad M-range rule is used.

This source was only syntax checked locally. The original frozen runtime,
services and GPU were not modified while preparing it.
