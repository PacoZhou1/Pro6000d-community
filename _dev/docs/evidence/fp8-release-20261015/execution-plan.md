# Fixed combined FP8 release — measurement contract

2026-10-15, America/Los_Angeles. User authorized combining the positive FP8
optimizations, measuring their actual combined gain, and retaining our own
fixed Docker image. This is not permission to restart the original model
services or replace their endpoint. Embedding remains running throughout.

## Included components

- Fixed vLLM 0.31 image, Qwen3.6 original FP8 model, FlashInfer, FP8 KV, MTP4.
- The actually tested 65,536-token draft subset and speculator patch; original
  target verification/bonus head remains full BF16.
- V2 SiLU/per128 FP8 quantization library and repaired F1/F2 GDN fusions.
- Independent configuration switches; unsupported or corrupted enabled
  components fail explicitly. No FP4, MXFP8, rowwise weight conversion,
  DFlash, or new kernel mathematics in this release.

Draft patch/subset files in `received/` are byte-identical to the actual prior
`d4-b` arm metadata recovered from its tar archive. Here, vLLM's V2 model runner
and the custom V2 activation quantizer are distinct features: the old draft
experiment used the runner but did not mount the custom quantizer.

## Revised execution scope (latest user steering)

At 2026-10-15 ~05:10 PDT the user explicitly cancelled rerunning the old
baseline and asked to run our own combination normally. The planned five-arm
comparison below is **superseded and will not be executed**. Finish the existing
combined-image natural-output smoke, then measure only B with the fixed
prefill12/decode6/thinking6 workload. Keep the validated combined service
running on loopback18000; embedding stays on and original27B/9B/ASR stay off.
Compare observed B numbers to cited historical runs, explicitly as cross-run
reference with no simultaneous control, not a fresh paired causal result.
The 2048-token thinking smoke is retained as a budget-truncated failed gate;
only its thinking max_tokens was raised to8192 for smoke-v2.

## Original controlled comparison (cancelled before any O/A arm ran)

| Arm | Image | MTP | Batch token ceiling | Custom patches |
| --- | --- | ---: | ---: | --- |
| O1 / O2 | Existing original v0.27.1 image | 2 | 8,192 | none |
| A1 / A2 | Pinned upstream v0.31.0 image | 4 | 32,768 | none |
| B | Fixed own image derived from the same v0.31.0 | 4 | 32,768 | draft64K + V2 quantizer + repaired F1/F2 |

Run order O1 → A1 → B → A2 → O2. All use the same read-only model volume,
262,144 maximum length, 16 maximum sequences, FlashInfer/FP8 KV, **explicit
12 GiB KV byte budget**, loopback endpoint and serial clients. Old-stack
runner defaults remain original; the new stack uses its V2 runner. This
measures the old-to-new stack package, separately from the A-versus-B custom
patch effect. The 12 GiB common allocation is a controlled experiment, not a
claim that the original four-service production memory budget was validated.

Each arm receives the same fixed payloads, with one excluded warmup per case:
12 long prefill requests, six normal fixed-output decodes, six thinking
fixed-output decodes. Report same-index paired ratios against both bracket
means, drift, output hashes, actual tokens and draft/accepted differences.
Different output or speculation work is disclosed, not silently filtered.

A separate bounded suite exercises naturally ended normal/thinking answers,
JSON and generated tool-call arguments (without executing tools), repeated
with identical payloads. It records hashes, actual output and repetition
signals; it does not claim full MoBase integration or absence of all loops.

If the full combination fails a real startup/correctness gate, preserve the
failure, isolate component switches and retest a supported profile; do not
promote an unverified image or substitute historical estimates.

## Operations and delivery

Root alone controls SSH/GPU. Reuse the existing local-proxy SSH ControlMaster;
all subordinate SSH/SCP commands specify `ProxyCommand=false` to prohibit
fallback TCP connections. Detached host controllers persist status/logs and
only clean their exact owned container names. Functional embedding checks
run before/after arms, health is sampled during runs, clocks/power are logged
without global locking. Original 27B, 9B and ASR remain stopped.

Deliver Dockerfile, pinned sources, build logs, immutable image ID, run/disable
instructions, rollback procedure, full raw results and a measured report in
`_dev/docs`. The image includes the runtime files and subset, but not model
weights, credentials or temporary host instrument mounts. Finish with two
functional embedding checks at least 30 seconds apart, no temporary tests,
and the validated combined service retained at the user-requested running state.
