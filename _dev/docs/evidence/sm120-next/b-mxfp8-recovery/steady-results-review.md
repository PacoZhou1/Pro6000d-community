# FP8 GEMM steady-window replication

The second GPU window [gemm-v2](../takeover/gemm-v2/) completed both shapes on the same fixed image, random seed, MX/PR library SHAs, and 31-pair arm orders as [gemm-v1](../takeover/gemm-v1/). The only timing change in the [frozen probe](bench_recovery_steady.py) was a 15-second four-arm round-robin Graph warmup before sampling; its exact source change is recorded in [steady-diff.md](steady-diff.md). I independently checked every JSONL gate, pair, median, and service snapshot; the [machine-readable recomputation](steady-summary.json) includes raw SHA256 values. Both runs have 19 distinct arm orders per mode, with all 31 pair indices containing exactly one sample of every arm at positions 0–3. No GPU work was performed for this review.

The warmup actually lasted 15.004 s / 2,338 full rounds at M2741 and 15.032 s / 406 rounds at M16160. Each arm therefore replayed that many times. NVML before→after changed from 2625 MHz and 80 W to 1897 MHz and 600 W at M2741, and from 2625 MHz and 82 W to 1860 MHz and 600 W at M16160. In v2's measured warm windows, SM clock spans only 1897 MHz at M2741 and 1860–1867 MHz at M16160, versus the broad/lagged readings in v1. NVML is a coarse witness, not a per-kernel energy measurement.

| M | Arm | v1 warm paired | v2 warm paired | v1 disturbed paired | v2 disturbed paired | v2 warm GEMM ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2741 | fixed CUTLASS block128 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.614 |
| 2741 | cuBLAS rowwise | 1.0392 | 1.0374 | 1.0395 | 1.0377 | 1.553 |
| 2741 | CUTLASS MXFP8 | 0.9514 | 0.9503 | 0.9488 | 0.9471 | 1.698 |
| 2741 | PR #5521-derived rowwise | 1.0418 | 1.0377 | 1.0432 | 1.0361 | 1.555 |
| 16160 | fixed CUTLASS block128 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 9.512 |
| 16160 | cuBLAS rowwise | 1.0619 | 1.0670 | 1.0606 | 1.0667 | 8.914 |
| 16160 | CUTLASS MXFP8 | 0.9689 | 0.9726 | 0.9670 | 0.9724 | 9.778 |
| 16160 | PR #5521-derived rowwise | 1.0487 | 1.0478 | 1.0496 | 1.0484 | 9.077 |

“Paired” is `median(baseline_ms[pair]/arm_ms[pair])`, not a ratio of independently computed medians. This distinction mattered at M2741 in v1: its warm PR ratio of separate medians was 1.0813 while the paired median was 1.0418, because absolute speed shifted during the run. After settling, v2's warm PR values are 1.0382 by separate medians and 1.0377 paired. Absolute v1/v2 milliseconds should not be compared as a kernel improvement: v2 deliberately enters a lower-clock ~600 W steady state. The two within-run paired ratios both show a small ~3.6–4.3% PR advantage at M2741. The cache disturbance still does not guarantee cold L2.

All v2 numerical evidence repeated v1 exactly: four finite sampled-reference gates, four changed-input Graph replay gates with refreshed references, and both full-matrix BF16 checks passed for each M. PR-derived rowwise output has **zero BF16 bit differences** versus same-input cuBLAS rowwise over 95,430,656 values at M2741 and 562,626,560 at M16160, including the tail, and zero after changed-input Graph replay. MXFP8 versus independent K32 power-of-two dequantized BF16 A/W matmul has full-matrix NRMSE `5.37e-5` and `5.34e-5`, respectively. These are independent synthetic scale schemes: block128, MXFP8, and rowwise numbers cannot be interchanged with the existing model's weights or treated as model-quality evidence.

The [v2 service snapshots](../takeover/gemm-v2/services-after.json) passed. Embedding retained the same container/start time, remained healthy (HTTP 200), and its prompt-token and successful-request counters stayed at 110,056 and 2,130 before/after **each** shape; running/waiting requests remained zero. Four 15-second monitor samples reported healthy service. The counters differ from v1's 109,984/2,118 because requests occurred between experiments; that does not imply traffic during either measured shape. Original 27B, 9B, and ASR remained stopped as instructed.

No arm meets the proposed **1.10 same-pair GEMM threshold** in either v2 mode. The largest value is cuBLAS rowwise 1.0670 at M16160; PR-derived rowwise reaches 1.0478, and MXFP8 is slower. Rowwise and MX use different scaling/weight representations from the fixed block128 model, and the comparison excludes activation quantization, weight reformatting, MX scale packing, and the PR ABI's per-call host TMA descriptor encoding. A separate host-dispatch measurement could quantify that excluded cost, but it cannot rescue the unmet operator threshold. This evidence does not support model integration or deployment of these arms.
