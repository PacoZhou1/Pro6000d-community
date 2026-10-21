# Same-process complete-chain timings

Each shape compares all21 selected arms,31 randomized paired rounds per warm/disturbed condition. Both candidate variants and unchanged4×2 fusion are compared to original official GEMM+V2 quant and two rebuilt official controls. Input/weight/library hashes, full bitwise checks and changed-input Graph replay are mandatory. Cache disturbance uses an owned >2×L2 buffer outside timing events; it is not a controlled cold cache.

| M | Condition | Best new candidate | New ms | Official ms | New latency vs official | Same-raster/swizzle4×2 ms | New latency vs4×2 |
|---:|---|---|---:|---:|---:|---:|---:|
| 2741 | warm | warp24n64_v3_heuristic_s8 | 1.9145 | 1.7886 | +6.97% | 1.9576 | -2.20% |
| 2741 | cache_disturbed | warp24n64_v3_heuristic_s8 | 1.8985 | 1.7698 | +7.27% | 1.9457 | -2.43% |
| 2786 | warm | warp24n64_v3_along_m_s8 | 1.9646 | 1.8016 | +9.20% | 2.0155 | -2.52% |
| 2786 | cache_disturbed | warp24n64_v3_heuristic_s8 | 1.9456 | 1.7838 | +9.11% | 1.9947 | -2.40% |
| 16160 | warm | warp24n64_v3_along_m_s1 | 11.3061 | 10.6286 | +6.36% | 11.6133 | -2.51% |
| 16160 | cache_disturbed | warp24n64_v3_along_m_s1 | 11.4074 | 10.7090 | +6.49% | 11.6961 | -2.52% |
| 32768 | warm | warp24n64_v3_along_m_s8 | 23.9538 | 22.9065 | +4.52% | 24.5453 | -2.53% |
| 32768 | cache_disturbed | warp24n64_v3_along_m_s8 | 24.2196 | 23.1670 | +4.45% | 24.8463 | -2.55% |

Percent changes are medians of paired ratios, not ratios of displayed medians. Best new arm selection is exploratory across the measured21 arms; all results remain in timing-v3/analysis.json, and small best-arm variations are not independent confirmation.

For isolated real-model negative-result validation, select warp24n64_v3 only at exact M16160, AlongM/swizzle1. It is the best new candidate at that shape in both conditions, but still slower than official. M32768 needs swizzle8; no generic large-M scheduler rule is inferred. New warp organization gives a modest improvement over the matched4×2 fused control but does not overcome the official full chain. Actual model dispatch and TTFT remain separate acceptance evidence.
