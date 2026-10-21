# D3 / D4 test plan (not executed; needs the GPU host and the patch from d2_patch/)

Baseline everywhere: fixed image `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`,
MTP4, same flags as `evidence/profiling/mtp4-runtime.json` (V2 runner, FP8 weights, fp8 KV, FLASHINFER). Candidate = same
image + `d2_patch/*.diff` mounted/applied + `VLLM_DRAFT_VOCAB_IDS=/subsets/subset_<K>.json`
(+ `VLLM_DRAFT_VOCAB_SHA256`, `VLLM_DRAFT_VOCAB_STRICT=1`). Subsets from `outputs/` (SHA256s in `outputs/coverage.json`).
Service policy, embedding monitoring, clock logging/locking, SHA-recorded payloads: plan section 1 of `16-sm120-next-round-plan.md`.

## Gate 0 (before any timing): patch is live
- Startup log must contain `draft-vocab enabled: <K> rows ...` and must NOT contain `draft-vocab disabled`.
- One short request; diagnostic Nsight capture (d0_capture_plan.md command, candidate arm) analysed with
  `d0_lm_head_steps.py`: M=1 head kernel count/step still 4, but its grid (gridY) and mean time must drop roughly in proportion
  to K/248320 (32K: expect ~0.2 ms, not 1.58 ms); M=5 kernel unchanged (1.57 ms, 1/step). If M=1 time did not drop, stop.

## D3 correctness gate
1. Greedy token-by-token equality vs same-config MTP4 baseline (temperature 0, top_p 1, seed fixed):
   6 requests of 4.8K prompt / 1024 completion + 4 open-ended outputs, thinking on and off. Compare token ids (request
   `logprobs`/`return_token_ids` if the API gives them, else exact text + completion token count). Run the BASELINE twice
   first to measure baseline self-consistency; 07 doc showed MTP2/3/4 differ among themselves, so shape-dependent numerics
   are a known source of diffs.
   - Pass: candidate == baseline for every request, or every divergence is of the same kind/rate as baseline-vs-baseline-rerun
     (first divergence position inside a near-tie logit, verified by teacher-forcing the baseline prefix and showing the
     candidate's top-1 equals the baseline's top-1 with margin < numeric noise). Any unexplained divergence = fail.
2. Acceptance per subset size K in {16384, 32768, 65536} (+ reasoning-specific 32768 when thinking): from the vLLM spec-decode
   metrics: accepted/draft, mean accepted length per round, per-position acceptance (pos 1..4), baseline vs candidate, per
   prompt type (zh / en / code / thinking). Report ratio = candidate_mean_accepted_len / baseline.
3. Existing 5 synthetic quality items + 4 open outputs + thinking terminates normally (`</think>` then EOS).
4. Fallback sanity: run once with an ids file lacking EOS and check the WARNING + unchanged outputs/timing (guard works).

## D4 performance A1 / B / A2 (only for K that passed D3)
- Same payloads (hash recorded) per index, no profiler: 6 normal decode + 6 thinking decode, 4.8K prompt / 1024 completion,
  warmup requests discarded per class. Order A1, B, A2 on the same booted hardware state; B arm is a separate container
  boot of the same image with the env var (A arms: no env var). Baseline = (A1 + A2) / 2; report A1 vs A2 drift.
- Record per arm: decode tok/s, per-step time, accepted/draft, mean accepted length, SM clock (NVML; lock with `nvidia-smi -lgc`
  if root, record success), embedding ID/start/restart/health + a real vector request before/after, payload hashes.
- Metrics: delta = B / ((A1+A2)/2) - 1 on decode tok/s, per class (normal, thinking) and pooled; also per-step ms.

## Numeric decision rule
Let delta = pooled decode tok/s improvement of the 32K arm over the baseline mean, with greedy outputs passing D3.

| outcome | action |
|---|---|
| delta < +5% , or accepted-length drop explains (>=) the per-step saving (candidate tok/step falls below break-even in README table) | STOP task D, keep negative result (plan section 4 criterion) |
| +5% <= delta < +10% | try K=65536 and the reasoning-specific subset for thinking; no cross-session repeat unless one of them reaches >= +10% |
| delta >= +10% and D3 passes | continue: cross-session repeat (second independent session) and thinking-mode confirmation; then evaluate per-class subsets / upstream form |
| also: baseline A1 vs A2 drift > delta/2 | result inconclusive; rerun the pair before deciding |

Break-even reference (README): at 32K the step shrinks 28.7 -> ~23.3 ms, so the candidate must keep >= ~2.48 tokens/step
(accepted length >= ~1.48 of baseline 2.06) just to tie.
