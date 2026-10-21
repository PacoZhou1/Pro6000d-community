# Numerical and writer gates

Fixed v3 binaries: both 2×4 variants and the 4×2 control passed all 13 cases (normal M1/5/16/256/257/2741/2786/16160/32768; M257 zero/tiny/large/exceptional). There were 688 full-output bitwise comparisons and 105 changed-input CUDA Graph replays. Fused arms compare FP8, FP32 scale and a separately launched matched BF16 mainloop; normal inputs also compare the downstream official BF16 GEMM. The strict analyzer validates complete output sizes, input byte and scale changes, dependency/weight/library hashes, and graph coverage before allowing timing.

Small M uses only the official fallback in this synthetic probe; it does not yet prove the eventual runtime guard. Changed-input Graph checks cover five normal shapes and every selected arm. These sampled distributions are not an all-input numerical proof or a language-quality assessment.

The v2 diagnostic also passed all numerical checks, but was rejected for timing: the CPU reconstruction of actual epilogue destination partitions found two writers per scale slot for each 2×4 variant (256 writes to128 addresses). The v3 writer predicate selects warp0/1, leaving128 writes to128 addresses, no missing/unexpected slots. The original4×2 control also has128 unique writes. This is a full-CTA address-geometry proof, not a GPU race-sanitizer or edge-tile proof; actual tail correctness is separately covered by the GPU bitwise gates.

Evidence: gate-v2/analysis.json, gate-v3/analysis.json, writer-v1/writer-status.json and scale-writer-probe.md. Failed ownership-v1 compilation is preserved; ownership-v2 fixed initialization without relaxing the compiler gate.
