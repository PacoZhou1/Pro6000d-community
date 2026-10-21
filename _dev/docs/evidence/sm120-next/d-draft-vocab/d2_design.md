# D2 design: opt-in draft-step lm_head row slice for MTP (vLLM v0.31.0, db9527a4...)

Status: PROPOSED PATCH, NOT APPLIED, NOT COMPILED, NOT TESTED. `patch --dry-run` applies cleanly to the
pristine files fetched at the pinned commit (SHA256 of the fetched files in `d2_patch/source-sha256.txt`).
Source was read from `raw.githubusercontent.com/vllm-project/vllm/db9527a46873454610df6dbedf79a36d6bf1a7f6/...`
(not a model/dataset; the ModelScope-only rule applies to weights/tokenizer/corpus).
The V2 model runner itself (`vllm/v1/worker/gpu/model_runner.py`) and the target sampler were NOT read;
statements about the verification path below rest on the speculator code plus the Nsight evidence
(M=5 head kernel is launched eagerly, outside the draft graphs), and must be re-confirmed in review.

## 1. Where draft logits are computed (exact)

Runtime config in all evidence runs: `VLLM_USE_V2_MODEL_RUNNER=1`, `--speculative-config {"method":"mtp","num_speculative_tokens":4}`.

| step | file : function | what it does |
|---|---|---|
| speculator class | `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py : MTPSpeculator(AutoRegressiveSpeculator)` | no logits code of its own |
| draft step 0 (draft "prefill", position 0) | `.../autoregressive/speculator.py : AutoRegressiveSpeculator._prefill` | `_run_model`, then `self.draft_tokens[:n,0] = self.sample_draft(...)` |
| draft steps 1..3 | `.../autoregressive/speculator.py : _generate_draft` (called from `_multi_step_decode` / `_generate_fused_drafts`) | `_run_model`, `sample_draft`, `update_draft_inputs` |
| **single choke point** | `.../spec_decode/speculator.py : DraftModelSpeculator.sample_draft` (greedy branch) | `logits = self.model.compute_logits(hidden_states); sampled = logits.argmax(dim=-1)` |
| model hook | `vllm/model_executor/models/qwen3_5_mtp.py : Qwen3_5MTP.compute_logits` | `return self.logits_processor(self.lm_head, hidden_states)` (`spec_step_idx` unused) |
| projection | `vllm/model_executor/layers/logits_processor.py : LogitsProcessor.forward -> _get_logits -> _apply_head` | `lm_head.quant_method.apply(lm_head, hidden_states)` -> BF16 GEMM, then `[..., :org_vocab_size]` |

So per decode step: 1 x draft-prefill graph (one M=1 head) + 3 x single-step decode graph launches (one M=1 head each)
= 4 draft heads, matching D0 (`d0_result_mtp4-capture.3_graph.json`: graph 10577 x1, graph 10598 x3 per step).
`use_fused_multi_step_decode` was False in that capture (graph 10598 replays 3 times), consistent with
`_configure_fused_multi_step_decode` falling back for the FLASHINFER backend; the patch does not depend on it.

## 2. How lm_head is shared

- Qwen3.6-27B has `tie_word_embeddings=False`, `vocab_size=248320`, `hidden_size=5120`, `mtp_num_hidden_layers=1`,
  EOS ids {248044, 248046} (from the downloaded `config.json` / `generation_config.json`).
- `Qwen3_5MTP.__init__` builds its own `ParallelLMHead` and `load_weights` keeps `lm_head*` checkpoint tensors.
- `eagle/utils.py : load_eagle_model` then calls `_should_share(eagle_model, "has_own_lm_head", ...)`. `Qwen3_5MTP`
  defines no `has_own_lm_head`, so it returns True: the draft's own `lm_head` is deleted and
  `eagle_model.lm_head = target_lm_head` (target `get_language_model().lm_head`). Draft and target therefore use one
  2.54 GB BF16 weight (the draft copy exists only transiently during load).
- Consequence for the patch: slicing `target lm_head.weight[ids]` after `load_draft_model` gives an independent copy;
  the shared head used by target verification is never touched.

## 3. What the patch does (files in `d2_patch/`)

1. `0001-draft_vocab-new-file.diff` adds `vllm/v1/worker/gpu/spec_decode/draft_vocab.py`:
   - env guard `VLLM_DRAFT_VOCAB_IDS=<path to JSON>` (list or `{"ids":[...]}`, ascending, unique), optional
     `VLLM_DRAFT_VOCAB_SHA256`, optional `VLLM_DRAFT_VOCAB_STRICT=1` (raise instead of fall back).
   - `maybe_build_draft_vocab_head(speculator)`: static eligibility checks (below), then
     `sub = lm_head.weight.index_select(0, ids).contiguous()` (BF16, `[K, 5120]`, own allocation) and `ids` on device.
   - `DraftVocabHead.top_tokens(h) = ids[(h @ sub.T).argmax(-1)]`, i.e. argmax over the subset mapped back to full ids.
     Ascending `ids` keeps lowest-index tie-breaking like the full argmax.
2. `0002-speculator-hook.diff` edits `spec_decode/speculator.py`: import, two attributes in `__init__`,
   `self.draft_vocab_head = maybe_build_draft_vocab_head(self)` right after `load_draft_model` (before any capture),
   and one new `elif self.draft_vocab_head is not None:` branch in `sample_draft` between the local-argmax branch and the
   full-logits branch. `ids` is int64 as is `draft_tokens`.

Apply: `cd <vllm tree> && patch -p1 < 0001-... && patch -p1 < 0002-...`.

Not touched: `Qwen3_5MTP.compute_logits`, `LogitsProcessor`, the target model, the rejection sampler, bonus-token path.
The verification (M=5) head and the bonus token come from the target forward in the model runner (not via
`DraftModelSpeculator.sample_draft`), so they keep the full head: lossless greedy verification is unchanged.

## 4. Guards, fallbacks, counters

Static fallbacks (reason recorded in `speculator.draft_vocab_fallbacks`, logged once with WARNING; original path used):
method != mtp; `draft_sample_method == "probabilistic"`; adaptive verification / acceptance estimator on (needs full
draft logits); watermarking; TP > 1; logit scale != 1 or soft cap; lm_head not BF16 or not `UnquantizedEmbeddingMethod`;
`head_dtype` override; ids file SHA mismatch; invalid ids (unsorted, duplicates, out of range); EOS id missing from subset.

Runtime counters: Python ints cannot count graph replays (they run only at capture), so the patch keeps a device
counter `DraftVocabHead.used` incremented in-place inside the captured region (`self.used.add_(1)`), which IS replayed.
Expected after N decode steps with 4 draft heads/step: `used == 4 * N` (+ warmup/capture runs). Reading it requires a
host sync, so read it from a diagnostic endpoint or at shutdown only; not wired in this patch (open item).
Fallback counts are static per process (one decision at load); a mixed fallback during serving cannot happen because
every condition is fixed at construction.

## 5. CUDA graph implications

- The sliced buffer and `ids` are allocated in `load_model`, before `speculator.capture()`. Graph nodes hold fixed
  pointers; never re-slice or reassign `sub`/`ids` after capture (the patch does not).
- `sample_draft` is inside the FULL-graph region (`_prefill` for the prefill graph; `_generate_draft` for decode graphs),
  so the new GEMM, the argmax and the `ids[...]` gather are captured. With PIECEWISE/NONE modes they run eagerly. No
  host sync, no data-dependent shapes: ok for capture.
- M (rows) is fixed per captured batch descriptor; cuBLAS picks its kernel per (M, N=K_subset, K=5120) at capture time,
  so the 32K head may select a different kernel/split than the 248K head. Performance of that kernel is an open
  measurement (D4 diagnostic capture must show the draft head kernel bytes/time dropped, not just its grid).
- Memory: +160 / 320 / 640 MiB for K = 16384 / 32768 / 65536 (BF16), allocated before KV-cache profiling, so it is
  subtracted from the KV budget automatically; with `--gpu-memory-utilization 0.7` expect ~0.3 GiB less KV for 32K
  (re-check the profiled-block count in the log and compare).
- Kernel numerics: the subset GEMM may reduce in a different order than the full GEMM. This can flip near-tied drafts
  (acceptance only), never final tokens, because verification uses the full head.
- `torch.compile`: nothing added inside compiled `forward`; no recompilation expected.
- Alternative considered: reuse `LocalArgmaxMixin.get_top_tokens` + `draft_id_to_target_id` (exists at this commit for
  EAGLE3 reduced-vocab drafts; maps `k -> k + d2t[k]`). It needs `use_local_argmax_reduction` and a smaller
  `ParallelLMHead` registered inside `Qwen3_5MTP`, which is more invasive for MTP. A follow-up could emit `d2t` from the
  subset and reuse it if upstreaming.

## 6. Known gaps / risks
- Untested: import paths, `lm_head.quant_method` class name check, `speculative_config.draft_sample_method` default
  value and attribute names (all read from source at the pinned commit, never executed).
- `getattr(model.config, "eos_token_id")` check uses the draft config; the Qwen3.6 text config exposes 248044 only,
  the chat EOS 248046 is in `generation_config`. Both ids are in every subset produced by `d1_build_vocab.py`, so the check
  passes for ours; a different ids file must include both.
- Repetition/loop drafts: tokens outside the subset can never be drafted (long-tail names, rare CJK, identifiers), so
  acceptance depends on text type; D1 numbers (README) show 32K covers only ~89% of held-out zh tokens.
- Counter readout path unimplemented.
