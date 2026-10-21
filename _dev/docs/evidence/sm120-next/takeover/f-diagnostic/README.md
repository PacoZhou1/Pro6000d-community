# F1/F2 in-situ diagnostic only

Sources copied from the recovered `f-received/instr` into this independent
folder. Existing Claude sources/results are untouched. `f_kernels.py` and
`v2_sitecustomize.py` are byte-identical to that recovered version. The base
sitecustomize is the post-splitfix source (original SHA ce01159d…b3380c).
B/BN/WC engine logs contain the new splitting op; W/WV/W0 do not. Therefore
old WV's 192 F1 bit-exact checks do not prove the B configuration.

Mount this folder read-only first on PYTHONPATH as `/instrument`, supplying
the unchanged V2 shared library separately at the prior path. No `.so` is
included here. Root controls the image, GPU, services, input and output mounts.

Retain the old settings for a combined diagnostic:

```text
PRO6000D_F_ENABLE=norm,conv
PRO6000D_F_V2=1
PRO6000D_F_KERNELS_SHA256=1b51a6d934e214190fcb586b217d1f890d096a699fdc06a83c97f0f2d61c8dd6
PRO6000D_FP8_FUSION_ENABLE=1
PRO6000D_FP8_FUSION_LIBRARY=/instrument/v2/libfp8_silu_quant.so
PRO6000D_FP8_FUSION_LIBRARY_SHA256=087b7caa92c4742ece3875b6dd1421e25360b834ce4106f7fa166f90bc933b3f
```

Add the diagnostic flags:

```text
PRO6000D_F_VERIFY=1
PRO6000D_F_VERIFY_CONV=1
PRO6000D_F_DIAGNOSTIC_SHA256=37780728f8494127a63172fdea5e5cc3dc8edb94ac1a769bd5010680fed260d0
PRO6000D_F_DIAG_DIR=/profiles
```

For each runtime `post_w` match, F2 clones the full original conv_state for
the official reference, runs official conv+post on the same real x/parameters,
then the candidate updates the real state. It compares every element of
q/k/v/g/beta and the full conv_state as integer bits. The six comparisons
are written to `/profiles/f2-diagnostic-<pid>.jsonl` and server logs, with
shape/stride/counts and first mismatch index/bits only. No tensor dump,
request text or token list is written. Any mismatch raises immediately.
F1 retains its existing cumulative FP8/scale VERIFY log and now also raises
on a mismatch. There is no sampling limit on eligible calls.

The reference conv API is assumed to mutate only the supplied conv_states,
consistent with the original synthetic official-versus-candidate tests.
All other inputs/metadata are shared read-only. This is a local step audit,
not an independent proof that upstream GDN state, metadata or whole-model
outputs are equal. A failure must be interpreted with the inserted reference
and execution ordering in mind; a pass only covers actually executed inputs.

This implementation synchronizes CUDA for numerical summaries. If an eligible
F2 call is made while CUDA graph capture is active, it records
`capture_not_supported_no_silent_skip` and raises. Nothing is silently
omitted. Historical large-prefill execution was direct, so root can first
use normal mode. If startup or a tested call does hit capture, root must
choose an explicit eager diagnostic or add a separate replay audit; the
mode change must be recorded and cannot prove the original Graph path.

Run one short diagnostic request set that exercises the original B long
prefill plus its ~4.8K decode prompt, retaining MTP4 and FP8KV, then inspect
all F1/F2 logs and actual hit lengths/call counts. This adds reference work,
clones, comparisons and synchronization, so its latency and decode TPS are
**not performance measurements**. It diagnoses possible local differences;
it does not presume the combined fusions are safe or explain output
hash differences without evidence.

CPU validation used AST parsing and metadata-only mocks to confirm separate
reference state, official-before-candidate ordering, all six comparison calls,
mismatch rejection and explicit capture rejection. No torch/CUDA/model was
loaded locally. GPU results remain root's responsibility. Execution source
hashes are frozen in `source-manifest.json`.
