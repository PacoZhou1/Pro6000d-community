# SM120 scheduler isolation probe

This experiment keeps the fixed FP8 blockwise gate/up math, BF16 rounding,
SiLU and per-128 quantization unchanged. It varies only CUTLASS persistent
tile scheduler `raster_order` and `max_swizzle_size` on the validated v6
M64N256 Cooperative fused kernel. The earlier official M128N128 BF16 GEMM +
V2 quant pipeline is the performance control; the previously validated v6
library is an anchor. A same-tile M64N256 ordinary-BF16 epilogue is a
diagnostic for scheduler effects inside GEMM. Its paired output must be
inverse-permuted before quantization, and that inverse cost is reported
separately; it cannot be presented as fused pipeline performance.

The source-confirmed scheduler heuristic changes direction at M16160:
official CTA grid (127,272) gives `AlongM` while v6 (253,136) gives
`AlongN`; both use `AlongM` at M2741. In CUTLASS's coordinate mapping,
`AlongM` advances N fastest and `AlongN` advances M fastest. That proves
different visitation order, not a causal explanation for the observed
long-chunk DRAM read increase. The clean comparison crosses three raster
settings (`Heuristic`, `AlongM`, `AlongN`) with swizzle 1 and 8, holding
tile, source binary, operands, workspace and output precision fixed. The
explicit forms that coincide with the heuristic are still recorded so
driver behavior can be checked.

The probe first runs numerical gates for M=1,5,16,256,257,2741,2786,
16160,32768 and normal/zero/tiny/large/exceptional M257 distributions.
For supported candidate M>256, every arm must match the fixed official
pipeline FP8 bytes and scale FP32 bits exactly; selected downstream FP8
down-projection BF16 output must also match. The official BF16 GEMM and
packed-weight inverse must match the fixed vLLM op bit for bit. A changed-
input CUDAGraph replay must change output and match a fresh official
reference. Ordinary BF16 epilogue results are compared after inverse
pair permutation; any mismatch is a diagnostic failure or separately
classified, never silently used as a fused result. Full numerical gates
precede all performance rows; any failure exits nonzero without timings.
Synthetic random weights/activations prove only these selected operator
semantics, not model quality.

`--timed-m` is automatically included in both the numerical gate and the
changed-input CUDAGraph replay set, even when the caller does not add it to
`--gate-m` or `--graph-m`. The effective `numeric_shapes` and
`changed_input_graph_shapes` are recorded in metadata; `--graph-m` remains
an additional requested replay set. Thus, timing M2786 cannot produce rows
without an M2786 changed-input replay witness. The analyzer continues to
reject any missing witness.

The separate `epilogue_probe.py` provides the matched ordinary-BF16
ablation without changing the fused eight-arm statistical denominator. For
one M/raster/swizzle at a time, it validates the new official and paired
ordinary-BF16 GEMMs against the fixed vLLM operation, inverse-permutes the
paired output into a preallocated canonical BF16 buffer, then checks V2
FP8 bytes/scales and changed-input graph replay. It reports seven distinct
timing scopes: old official complete chain; new official GEMM-only and
complete chain; paired GEMM-only, inverse-only and complete
GEMM→inverse→V2 chain; and quant-only on precomputed canonical BF16.
The paired ordinary chain has three kernels and is **diagnostic only**;
its inverse copy and quant cost must not be subtracted as if they were
additive wall-time contributions or confused with the fused GEMM+pack.
It accepts a single `--ncu-target` to profile exactly one of its labeled
GEMM or complete-chain scopes.

For timing, all per-arm tensors, native scale scratch and queried workspace
are preallocated before capture. Each measured graph contains the complete
GEMM→SwiGLU/blockquant pipeline; the fused graph contains GEMM+pack. A
separate GEMM-only event, if emitted, is labeled as such. At least 31
paired iterations use deterministic randomized arm order, one common
input and seed, CUDA events around graph replay, and both a warm sequence
and a sequence with an owned >2×L2 flush tensor filled outside the timed
event. This latter case is an *approximate cache disturbance*, not a
guaranteed cold-L2 state. Nsight Compute `--cache-control none` likewise
does not establish a fixed cold cache. Report raw pair values and
medians/ranges; do not call small differences significant or extrapolate
to model/API speedup. During NCU, run only a selected shape and arm, after
numeric gates; use a profilerStart/Stop window around exactly one complete
pipeline and label each kernel, avoiding all-arm replay overhead.

The dedicated C ABI is pinned in `scheduler_abi.h` at version 1. The probe
checks ABI version and all loaded library SHA values before any ctypes
launch. Raster is 0=Heuristic, 1=AlongM, 2=AlongN, and swizzle is 1 or 8.
Explicit raster
and swizzle are per-call values, never process-global setters, so graph
capture/replay cannot silently change scheduler state. The CUDA launch
path must not allocate. The resulting JSONL includes binary hashes,
device/driver/runtime versions, shape, exact control tuple, per-arm
correctness and timing scope. `analyze_results.py` will reject missing
arms, failed gates, unmatched seeds or mixed shapes before calculating
paired contrasts. The old v6 anchor retains its known SHA and heuristic
swizzle 8. A separate NCU invocation gates only its selected M and arm's
changed-input graph before opening the profiler window; it does not replay
all large numerical shapes before every selected counter capture.
The NCU path numerically validates and changed-input graph-replays the fixed
official reference plus only the selected arm. It loads the legacy v6 binary
only when `legacy_v6` itself is selected; unrelated NCU targets exclude that
binary to avoid an independently observed ELF/TLS collision. Metadata records
`ncu_selected_arm`, `numerical_arms`, and `excluded_arms`, and the analyzer
checks the selected BF16/FP8/scale/downstream witnesses, replay output change,
input seed/hash and single profiled launch. The full eight-arm matrix still
loads and validates the legacy anchor as before.

`official_tuning_probe.py` is a separate same-process test of the promising
ordinary official tile, without loading the legacy v6 binary. For each of
M2741/2786/16160/32768 it shares canonical A/B/As/Bs among three complete
chains: fixed old official H8, new library official H8, and new library
official AlongM/s1, all followed by the same V2 quantizer. It first requires
full-bit BF16, FP8 and scale equality against fixed vLLM, plus changed-input
Graph replay and a fresh-output timing-capture check. It then records at
least 31 randomized matched graph pairs in warm and owned-cache-disturbed
scopes, releasing per-M buffers/graphs before advancing. The strict CPU-only
`analyze_official_tuning.py` verifies hashes, each shape's seed/content hash,
all three numerical and graph witnesses, capture checks and complete paired
orders. This separates a different library/build from a raster/swizzle
change; even a positive micro result only licenses a narrow real-model
experiment after its own model gate.
