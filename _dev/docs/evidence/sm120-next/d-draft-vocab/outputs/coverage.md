# D1 coverage (held-out docs, token coverage / 4-token chain coverage)

| K | zh | en | code | reasoning_zh | reasoning_en |
|---|---|---|---|---|---|
| 16384 | 80.27% / 44.20% | 86.64% / 57.95% | 86.75% / 60.35% | 80.42% / 42.63% | 91.16% / 69.58% |
| 32768 | 88.60% / 63.46% | 92.77% / 75.05% | 92.08% / 73.99% | 88.26% / 61.05% | 94.65% / 80.36% |
| 65536 | 95.56% / 84.21% | 97.45% / 90.60% | 96.47% / 87.44% | 94.69% / 80.71% | 97.62% / 90.95% |

## Reasoning-specific subset vs general
| K | overlap | Jaccard | reasoning subset ids inside general | reasoning_zh cov (general / reasoning-subset) | reasoning_en cov (general / reasoning-subset) |
|---|---|---|---|---|---|
| 16384 | 7942 | 0.320 | 48.5% | 80.42% / 94.24% | 91.16% / 98.68% |
| 32768 | 17147 | 0.354 | 52.3% | 88.26% / 97.77% | 94.65% / 99.37% |
| 65536 | 43351 | 0.494 | 66.1% | 94.69% / 99.41% | 97.62% / 99.77% |
