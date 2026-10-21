# Steady-window second measurement: exact source delta

This is an **additional** probe. The [first-run benchmark](bench_recovery.py), its SHA256 `2a7c39ed9f669ecdeee815a5be20bad99a854f32393b4cfa4572c182976a8477`, and `takeover/gemm-v1` outputs are unchanged. The [steady version](bench_recovery_steady.py) has SHA256 `fcd43e3b8549ba1763c2ac21e46bf3a6a28ced5c7eeb4ff7e17e73c1683ab86a`. Its `meta.bench_sha256` records that new hash at runtime. Local `python3 -m py_compile` passed at preparation; the parent subsequently ran both GPU shapes in [gemm-v2](../takeover/gemm-v2/), and their independent comparison is in [steady-results-review.md](steady-results-review.md).

The diff has only these behavior changes:

1. The old `shape_scope.cutlass_m_tile` label is split into `baseline_source_m_tile=128`, `mx_adapter_m_tile=128`, and `pr5521_derived_m_tile=128` when present. The fixed vLLM source selects its M128 default for both tested M>256 shapes; MX/PR configs also use M128. cuBLAS tiling remains unknown, and no measured grid is claimed.
2. `--pr-library` is required, so the requested warmup cannot silently run only three arms. After all original numeric and changed-input Graph gates pass and common input is restored, the script allocates its existing L2 disturbance buffer, then round-robins the four captured graphs for at least 15 seconds. It synchronizes after each replay, completes whole rounds, and records `steady_warmup` with target/actual seconds, rounds, replays per arm, arm order, and before/after NVML clock/power. This phase is outside all CUDA Event samples.
3. The original seed, inputs, four arms, thresholds, 31 randomized pairs, per-pair order generator, warm/disturbed modes, sample fields, and paired-gain formulas are copied unchanged. The owned-buffer disturbance retains the same size and is not claimed to guarantee cold L2. PR host TMA descriptor encoding remains outside Graph/Event timing and is not measured here.

Inspect the precise line diff with:

```sh
diff -u _dev/docs/evidence/sm120-next/b-mxfp8-recovery/bench_recovery.py \
  _dev/docs/evidence/sm120-next/b-mxfp8-recovery/bench_recovery_steady.py
```

Run only after the current GPU experiment has completed, with fresh output files:

```sh
python3 /work/b-mxfp8-recovery/bench_recovery_steady.py \
  --mx-library /work/build-mxfp8/librecovery_mxfp8.so \
  --pr-library /work/build-pr5521/libpr5521_derived.so \
  --m 2741 --pairs 31 --output /work/results/steady-m2741.jsonl
```

Repeat for `--m 16160`. Treat the second window as replication of a small synthetic operator difference, not a model A/B or an altered numerical contract.
