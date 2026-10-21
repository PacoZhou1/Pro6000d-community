# Historical performance references (not this run's baselines)

Primary comparison corrected by user request: initial stock vLLM0.31/MTP2, 94.2336 tok/s normal decode and 2.17895 s prefill; quiet prefill subset 2.16683 s (n=2). Current normal gain is 25.4042%; prefill is effectively unchanged. Initial FP8-KV/FlashInfer thinking baseline was not measured. Source: `../../paired-v031.jsonl` and `../../summary.json` -> `paired-v031.jsonl`. The v0.27 figures below are historical context, not the release headline denominator.

These numbers came from separate GPU sessions and request suites. They are descriptive context for a **single-arm combined-image validation**, not matched A/B measurements or evidence that separate gains add.

| Stack and metric | Historical observation | Inspectable local source | Comparability limit |
|---|---:|---|---|
| vLLM 0.27 FP8 MTP2, 1024-token sustained decode | 90.7344 tok/s median, 3 valid requests | [`summary.json`](../../summary.json) → `paired-v027.jsonl.groups.sustained_4k_decode`; raw [`paired-v027.jsonl`](../../paired-v027.jsonl) | Old client and session; 4.8K-token prompt, not today's fixed 800-word payload. |
| vLLM 0.27 FP8 MTP2, long prefill | 2.37779 s text TTFT median, 3 valid requests | Same `summary.json` → `paired-v027.jsonl.groups.long_prefill` and raw file | Roughly 18.9K tokens; historical context only. |
| Stock vLLM 0.31 MTP4, decode | 106.61 tok/s A1, 106.59 tok/s A2 | [`../../sm120-next/p-prefill/summary.md`](../../sm120-next/p-prefill/summary.md), E3 table; raw under `../../sm120-next/p-prefill/runs/` | Separate V2 fusion A/B/A session and payloads. |
| Stock vLLM 0.31 MTP4, thinking | 130.45/130.97 tok/s A1/A2 | [`../../sm120-next/d-draft-vocab/results-retest/summary.md`](../../sm120-next/d-draft-vocab/results-retest/summary.md), D4 table; [`d4_analysis.json`](../../sm120-next/d-draft-vocab/results-retest/logs/d4_analysis.json) | Separate draft-vocabulary retest and prompts. |
| vLLM 0.31 MTP4 with 64K draft vocabulary, normal/thinking | 119.29 / 148.10 tok/s B medians | Same D4 retest summary and `d4_analysis.json` | 64K draft subset alone, not the current combined image. |

The old 0.27 result came from `paired-v027.jsonl`, not `paired-v027-repeat.jsonl` or `paired-v027-nomtp.jsonl`. The 0.31 normal-decode and thinking figures above were measured in different historical sessions. Do not put them into `analyze_combined_only.py` as denominators or claim an aggregate speedup. For the current run, inspect the result's own `source_sha256`, request validity, prompt/completion usage, and median/range.
