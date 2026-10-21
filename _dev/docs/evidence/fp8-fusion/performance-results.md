# No-profiler model comparison

Same fixed image/model/command and base environment in A1, B(V2), A2; only B mounts the opt-in runtime and selects the recorded library SHA. All 54 measured request windows contain exactly one target request, cold prefixes (zero cache-hit counter delta), valid usage, identical per-index payloads, and no observed embedding counter activity. Before/after counters do not rule out every transient overlap.

| Median | A1 | B | A2 |
|---|---:|---:|---:|
| Long prefill TTFT (12 requests each, seconds) | 2.268213 | 2.244258 | 2.284222 |
| Ordinary decode (6 requests each, tok/s) | 106.73580 | 106.73173 | 106.70545 |

Median paired TTFT reduction against (A1+A2)/2 is 1.4144%, with 12 per-input reductions spanning 1.3697%-1.5054%. A1/A2 medians themselves drift by about 0.706%; one A/B/A cycle does not establish robustness across independent times or production load. The bootstrap in the JSON resamples prompt indices only and is descriptive, not a causal confidence interval for restart/time variability.

All 12 prefill first-token contents and six ordinary decode content hashes match across all three arms. Their reasoning is empty and is not evidence about thinking performance. Every arm passes five structured quality cases and four open-output cases (including two normally completed thinking arithmetic answers): 27/27 total. This is synthetic quality, not full MoBase or business evaluation.

Retain V2 as an experimental prefill candidate with a small observed API improvement. Do not replace production: the original 0.45 memory budget, four-service coexistence, broader requests and multiple time windows remain unverified for the candidate. BF16 lm_head and model weights remain unchanged.

Sources: api-comparison.json, configuration-comparison.json, and a1/b/a2 raw JSONL, runtime, status and server logs.
