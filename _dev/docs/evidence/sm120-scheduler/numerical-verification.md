# v3 library numerical and graph acceptance

Validated binary: `af1d814966a746ff09edc55d40196b871067d88d22ed0e50700c1f3a4eb185d6`. The accepted [v4 run](scheduler-matrix-v4/stdout.log) uses its own frozen source snapshot (probe SHA `715608b02bfda9aba256d9e879c0c90ab5dc5e995179a3e4b8f82031409cb0da`); a subsequent NCU-only loader change must be recorded separately.

13 numerical cases passed: M1/5/16/256 reference/fallback only; M257/2741/2786/16160/32768 normal and M257 zero/tiny/large/exceptional. Supported candidates compare all six scheduler combinations against pinned vLLM BF16 and FP8 byte/FP32 scale references; normal cases also validate downstream BF16 output. No tolerance relaxation. Small M does not prove M64 candidate support.

Changed-input CUDAGraph witnesses cover M257/2741/2786/16160, 8 arms each, and require changed output plus exact fresh-reference bytes/scales. The timing capture outputs are checked separately for all timed M and arms. The raw analyzer result is [analysis.json](scheduler-matrix-v4/analysis.json).

Preserved failures: first two matrix launches exited139 before numerical results because of host symbol collision. v3 completed numerical cases but omitted changed-input Graph coverage for the added timing tail M2786; the strict analyzer rejected this omission. The probe was corrected to union timed M into the Graph gate and the full v4 run passed. Neither failure is reclassified as a pass.

These are synthetic GPU operator and capture checks, not model quality, API speed or actual model graph-replay evidence.
