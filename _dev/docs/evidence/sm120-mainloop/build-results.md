# Actual fixed-image compilation and ELF audit

The CPU-only compiler and audit containers used the pinned vLLM image; none initialized CUDA. No installed CUTLASS header was modified. Source snapshots, command argv, compiler logs and hashes are retained per attempt.

| Build | Result | Library SHA256 |
|---|---|---|
| build-warp24-v1 | compile_failed | none |
| build-control42-v2 | compiled | 0a658bdd25b2a9eeb4e55c240e5221bb4f94494c457a31e0cbb689042b2d02d6 |
| build-warp24x2-v2 | compiled | 0b632f1dd0f8bd6c245c1d8367fc6ffe1e3d6658123137017dfafcb3a154472c |
| build-warp24n64-v2 | compiled | 195ca947d66bedb91c824150328dd3367ee27c1e58ca14317466eeed8384451d |
| build-warp24x2-v3 | compiled | 8a49155295fec6de66ac05a603342b9aba0caa5ffa3ce89f16f11b72a8a7728b |
| build-warp24n64-v3 | compiled | a6ae4155f65ec3b992ade3943902ccd88eea76b211dda5ceab26626245b7ae5d |

`warp24-v1` failed at CuTe CopyAtom coverage: 2×4/PermN32 supplies 8 B/thread for B, but x4 needs 16 B/thread. `warp24x2` changes B copy to x2; `warp24n64` keeps x4 and raises permutation N to64. Both compile. v3 additionally restricts the SF128/4-warp scale writer to warp0/1; v2 retains a duplicate-writer predicate and is rejected for performance/model use.

All five audited libraries expose only intended ABI C functions as defined GLOBAL/DEFAULT symbols. Metadata confirms identical matched paired/fused mainloops, two stages, 256 MMA threads and 384 total threads.

| Kernel | User shared bytes | Mainloop shared bytes | Registers/thread | ptxas spill stores/loads B |
|---|---:|---:|---:|---:|
| Fused M64N256, every layout |101376|83968|168|0/0|
| Paired BF16 M64N256, every layout |93184|83968|168|0/0|
| Official rebuilt M128N128 |77824|68608|168|12/20|
| Output pack |0|n/a|16|0/0|

The official rebuilt control has an 8-byte stack frame; it is not correct to describe every compiled kernel as spill-free. Compiled register/shared values are not measured occupancy, bank conflicts, memory throughput or model speed. Runtime numerical/Graph evidence lives in gate-v3, and timing/profiling is separate.
