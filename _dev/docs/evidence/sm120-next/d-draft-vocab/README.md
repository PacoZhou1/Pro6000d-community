# Task D: draft-head vocabulary pruning (FR-Spec style) for Qwen3.6-27B-FP8 MTP4

Status: **prepared, not run on GPU.** D0 and D1 were executed offline on CPU against existing local evidence and ModelScope
data. D2 is a design plus an unapplied, untested patch. D3/D4 are a plan. Nothing here was committed.

## Files
| file | what |
|---|---|
| `d0_lm_head_steps.py`, `d0_result_*.json`, `d0_capture_plan.md` | offline per-step lm_head kernel counter (eager and graph captures), results on 5 local captures, operator capture command |
| `d1_fetch_corpus.py`, `d1_build_vocab.py`, `download_manifest.json` | ModelScope-only corpus + tokenizer fetch (ranged), frequency and subset builder; revisions, licences, SHA256 |
| `outputs/subset_{16384,32768,65536}.json`, `outputs/reasoning_subset_*.json`, `outputs/coverage.{json,md}`, `outputs/token_counts.npz` | the id lists (with SHA256 in `coverage.json`), held-out coverage, reasoning overlap |
| `d2_design.md`, `d2_patch/` | where draft logits are computed, the patch (2 unified diffs), CUDA-graph notes, source SHA256 |
| `d3_d4_test_plan.md` | correctness gate, A1/B/A2 plan, numeric decision rule |
| `download/` (gitignored) | venv, tokenizer files, corpus samples |

## D0 result (already answered from existing data)
`d0_lm_head_steps.py` on `evidence/profiling/raw/mtp4-capture.3.sqlite` (normal CUDA graph replay, `--cuda-graph-trace=node`,
MTP4): **83/83 decode steps contain exactly 4 M=1-like head kernels (grid 8x1940, mean 1575.96 us) and 1 M=5-like (grid 8x970,
mean 1569.04 us)**; capture.4: 84/84 steps. Together 27.87% of all GPU kernel time, 7.87 ms/step. The eager
`gemm-targeted/raw/capture.2.sqlite` reproduces 348 / 87 kernels over 87 steps (25.54% of kernel time, 09 doc). Structure in graph
mode: M=5 verify head launched eagerly (`cuLaunchKernel`); draft step 0 is graph 10577 (1 head); draft steps 1-3 are graph 10598
launched 3x (1 head each). The old captures cover the plan's D0 need, so a new capture is only required as the fresh baseline for
the D4 diagnostic. Caveat: shape is inferred (BF16 GEMM-named, >=1 ms, grid signature), kernel names carry no N/K. The
capture.1/2 prefill captures contain only ~11 candidate kernels (prefill, no decode loop).

## D1 result (held-out documents, token coverage / 4-token chain coverage)
Tokenizer: Qwen/Qwen3.6-27B-FP8 on ModelScope, revision `970cb019bd144f587f4503d02bc69b9dd300d0f0` (tokenizer.json sha256
`5f9e4d49...81cb42`). Corpus (~355 MB text total after sampling, 8 ModelScope sources: wikipedia-cn, chinese-fineweb-edu,
fineweb-edu, starcoderdata python/js, CodeAlpaca-20k, Chinese-DeepSeek-R1-Distill-110k, Bespoke-Stratos-17k). Doc-level 80/20 split;
mix weights zh 0.30 / en 0.40 / code 0.30; 289 forced ids (all added/special tokens incl. `<think>`/`</think>`/`<tool_call>`, EOS 248044
and 248046, the 256 byte-level tokens) are inside every subset.

| K | zh | en | code | reasoning_zh | reasoning_en |
|---|---|---|---|---|---|
| 16384 | 80.27 / 44.20 | 86.64 / 57.95 | 86.75 / 60.35 | 80.42 / 42.63 | 91.16 / 69.58 |
| 32768 | 88.60 / 63.46 | 92.77 / 75.05 | 92.08 / 73.99 | 88.26 / 61.05 | 94.65 / 80.36 |
| 65536 | 95.56 / 84.21 | 97.45 / 90.60 | 96.47 / 87.44 | 94.69 / 80.71 | 97.62 / 90.95 |

Reasoning-specific subset (R1 zh + Stratos en) vs general: K=32768 overlap 17,147 ids (Jaccard 0.354; only 52.3% of the reasoning
subset is inside the general one); reasoning text coverage with its own subset 97.8% (zh) / 99.4% (en) versus 88.3% / 94.7% with the
general subset. General domains lose a lot under the reasoning subset (K=32768: zh 89.5, en 77.7, code 78.5), and the union of
both has 48,389 ids. Hence reasoning text differs materially from general text in this corpus, but the reasoning subset is built
and evaluated on only two datasets (distilled traces), so the gap is partly in-domain fitting; a separate thinking-mode subset is
worth trying only if D4 shows thinking-mode loses acceptance. File SHA256 values are in `outputs/coverage.json` (e.g. subset_32768
`bcbbef0f...fd9`).

Limits: corpus is encyclopedic/web/code, not the production traffic mix; zh coverage at 32K (88.6%) is the weakest point; no
tokens-per-class weighting for actual user workloads.

## Estimated upside (recomputed; model, not measurement)
Inputs: head 2.543 GB per read (248320 x 5120 x 2 B), 1.576 ms M=1 / 1.569 ms M=5 (~1.61 TB/s), 4+1 reads per step, 28.7 ms/step
(106.7 tok/s at 2.06 accepted + 1 bonus), heads = 7.87 ms = 27.4% of step (D0 capture 3 gives 7.87 ms exactly). Assume subset GEMM time
= 1.576 ms x K/248320 + 15 us floor (bandwidth-bound; untested for the smaller cuBLAS kernel).

| K | GB/draft read | draft head ms | saving/step | step ms | speedup if acceptance unchanged | break-even tok/step |
|---|---|---|---|---|---|---|
| 16384 | 0.168 | 0.12 | 5.83 | 22.87 | +25.5% | 2.44 |
| 32768 | 0.336 | 0.22 | 5.41 | 23.29 | +23.2% | 2.48 |
| 65536 | 0.671 | 0.43 | 4.58 | 24.12 | +19.0% | 2.57 |

Acceptance sensitivity: per-position acceptance a=0.7516 reproduces 2.06 with 4 drafts (a + a^2 + a^3 + a^4). Pessimistic model
a' = a x coverage (every target-chosen token must lie in the subset; coverage from the table, ignoring that accepted tokens are
skewed to frequent ones): K=32768 gives tokens/step 2.60 (zh), 2.76 (en), 2.73 (code), i.e. throughput x1.05 / x1.11 / x1.10; K=65536 gives
x1.12 / x1.15 / x1.13; K=16384 gives x0.95 / x1.04 / x1.04. So the plan's "+20%" is the zero-loss ceiling; the realistic band under this
model is about +5% to +15%, and zh traffic is the risk. This sits around the plan's 5% / 10% thresholds, so the measurement matters.

## Open risks
1. Acceptance loss is the whole question; the coverage-product model is crude both ways (misses correlate with wrong drafts; chain
   correlations unknown). Measure per class.
2. The 32K cuBLAS kernel at M=1/N=32768 may not reach the bandwidth ratio (small grids, split-K); real saving may be below the table.
3. Patch untested (imports, attribute names, `quant_method` class, `draft_sample_method` default, graph capture of the gather);
   V2 model runner verification path not read, only inferred from speculator code and Nsight evidence.
4. Memory +0.16/0.32/0.64 GiB reduces KV budget (profiling order to confirm).
5. Numerics: subset GEMM reduction order may differ; only affects drafts, but D3 equality must account for the known shape-dependent
   differences (07 doc).
6. Device counter readout and fallback telemetry not wired.
7. Probabilistic draft sampling, TP>1, adaptive verification: guarded off (falls back).
8. Corpus transfer: the only large shard fetch (zh_web, whole 779 MB parquet row group) pushed total ModelScope transfer to about
   1.0 GB, above the "few hundred MB" budget; stored samples are ~355 MB and the rest is not retained as such. Licence note: starcoderdata has no
   single SPDX id (BigCode terms, permissive per-file); samples are used for token counts only, not redistributed.

## Blocked / not done
No GPU, SSH or remote host used. Nothing blocked on data. Not done by design: applying/compiling the patch, any GPU run (D3/D4),
a fresh normal-graph capture (existing captures already answer D0).
