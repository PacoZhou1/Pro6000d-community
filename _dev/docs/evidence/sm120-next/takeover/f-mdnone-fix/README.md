# Minimal F1 null-metadata output initialization

2026-10-15 validation update: the frozen code below has now run on the actual
SM120 image. The extracted null-metadata CUDA poison probe passed all three
shapes; its packed warmup is explicitly stubbed, not an integration test.
A separate uninstrumented full-model A1/B/A2 passed the recorded startup,
request, output-hash and MTP-count checks. See [results](../ffix-results.md)
for exact scope and remaining limits. The preparation notes below describe
the code before those root-run checks; no historical-cause claim is added.

Independent copy of recovered `f-received/instr`. The sole execution-source
change is the F1 `md is None` branch: after official packed warmup returns,
run the same official per-token-group FP8 quantizer already used by the
normal fallback, so newly allocated q and s are written before out_proj.

Official fixed source `takeover/source/qwen_gdn_linear_attn.py:1778-1780`
returns after warmup without filling the zero-initialized core buffer. Its
original forward still passes that zero buffer to normal `self.out_proj`,
which performs its input quantization. The F1 replacement bypasses that
input quantizer and previously passed uninitialized torch.empty q/s to
CUTLASS. This draft repairs that explicit output contract; it does **not**
claim to explain or fix B's decode text differences.

No F1/F2 math, metadata-present fallback, MTP selection, graph splitting,
weight layout or kernel changes. Do not replace the already running
`f-diagnostic` with this folder. Root selects a fresh owned run and freezes
this separate source snapshot. No `.so` is included; provide the prior V2
library using the same mount/path/SHA as the existing F experiment.

Environment is unchanged: `PRO6000D_F_ENABLE=norm` or `norm,conv`,
`PRO6000D_F_KERNELS_SHA256=1b51a6d934e214190fcb586b217d1f890d096a699fdc06a83c97f0f2d61c8dd6`,
`PRO6000D_F_V2=1`, and the prior enabled V2 library settings. Optional
`PRO6000D_F_VERIFY=1` retains original F1 logging semantics. This is not
the synchronous F2 diagnostic and does not use VERIFY_CONV.

Original validation plan: poisoned q/s under null metadata versus official packed
warmup+quant, actual fixed-image fake/schema/Inductor and startup/capture,
then metadata-present numerical gates and fresh model output/draft/accepted
comparison. Local AST and mock checks cover code placement and whether both
outputs are written only. The preparation child did not run CUDA; subsequent
root-run CUDA and model results are linked at the top.
Full review and source references are in `../f1-forward-review.md`.
