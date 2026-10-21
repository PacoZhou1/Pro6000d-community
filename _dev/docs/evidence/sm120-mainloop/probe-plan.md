# SM120 M64 mainloop/warp-layout experiment plan

Status: local source/design review only. No new kernel has been compiled or
launched. The target remains the fixed vLLM 0.31.0
`db9527a46873454610df6dbedf79a36d6bf1a7f6` FP8 blockwise gate/up GEMM,
`[M,5120] × [34816,5120]`, followed by BF16 rounding, SiLU×up, and the
existing per-128 FP8 quantizer. BF16 `lm_head`, FP4 and services are outside
this experiment.

## Smallest attributable source change

Use [build_scheduler.py](../sm120-scheduler/build_scheduler.py) as the
source-pinning and private-ELF template, copied into this new experiment rather
than edited in place. Its fused baseline is M64×N256×K128 Cooperative,
cluster 1×1×1, two mainloop stages, 384 block threads and 256 MMA threads.
The [fixed CUTLASS blockwise builder](../gemm-epilogue/source/cutlass/include/cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl)
chooses `AtomLayoutMNK=Layout<Shape<_4,_2,_1>>` for Cooperative and
`<_2,_2,_1>` for Pingpong. The [generated scheduler dispatch](../sm120-scheduler/build-v3/standalone_dispatch.cuh)
already pins the M64N256 fused epilogue and constructs a matched ordinary
BF16 GEMM with **the exact same `CollectiveMainloop` type**. Keep that
static assertion, fixed K128 block scales, tile, schedule, weight packing,
epilogue visitor, pack kernel, raster and swizzle unchanged in the first
variant. The one proposed experimental edit is a pinned shadow-header change
to the Cooperative `AtomLayoutMNK` from 4×2 to **2×4** MMA warps. This tests
whether distributing the same eight consumer warps differently over M/N
improves operand/scale reuse or instruction issue at the same CTA output
area. It is a hypothesis, not an assumed legal or faster layout: CUTLASS
compile assertions, the paired gate/up fragment mapping and SF128 reduction
must all be revalidated. Do not relax a compile-time safety assertion to get
this variant built. If 2×4 is unsupported, retain the compiler diagnostic
and stop this axis; do not silently change tile or schedule.

The baseline 4×2 and experimental 2×4 should be separate private libraries
with an otherwise identical ABI. Give the new build a unique namespace and
visitor type names (for example `pro6000d_mainloop_v1`) and export only its
versioned C entry points. Retain `static thread_local` C++ state,
`-fvisibility=hidden,-fvisibility-inlines-hidden,-fno-gnu-unique`, and an
`readelf` check of exported symbols. The earlier v1 scheduler library caused
an ELF/TLS collision with the old baseline; merely changing the `.so`
filename is insufficient. Each build manifest must hash the pinned v6
inputs, fixed CUTLASS headers, exact one-line shadow edit, generated
sources, nvcc argv, compile log and library. Compile CPU-only in an owned
container; this plan authorizes no GPU or service operation.

Do not infer that 2×4 retains the old scale-factor mapping. The previous
M64 Cooperative fix in the [v6 visitor](../gemm-epilogue/build-gateup-coop-v6/paired_gateup_visitor.cuh)
set SF128 `WarpsPerSF=2` after a 4-warp assumption produced massive output
errors, despite a correct gate/up fragment pairing map. The exact same
FP32 scale multiplication order as the validated v6 shadow mainloop must
remain pinned. A changed warp layout may require a new *derived* SF/epilogue
lane map; if so, that is a second, explicitly hashed semantic patch, and
strict full-bit gates still apply. Never carry over the numeric success of
the old layout by assertion.

## Isolated measurement extension

The least code is a small `mainloop_probe.py` adapting
[scheduler_probe.py](../sm120-scheduler/scheduler_probe.py) helpers for
SHA-checked ctypes, input construction, bit comparison, graph replay and
`measure_graphs`, together with the complete-chain arm logic in
[official_tuning_probe.py](../sm120-scheduler/official_tuning_probe.py).
Keep [epilogue_probe.py](../sm120-scheduler/epilogue_probe.py)'s separate
BF16 diagnostic scopes. Do not load the old legacy v6 library during a
single-arm NCU run; the prior TLS collision occurred before a target kernel
was profiled.

Use one fixed synthetic A/As and canonical B/Bs per shape and distribution;
pack B/Bs once and verify the inverse mapping bitwise before any GEMM. Record
input seed rule (`seed+100+M`), FP8 A/B bytes, scale bits, packed-weight
fingerprint, build/library SHA and actual CUDA/tool versions. Allocate all
outputs, native scales, temp FP8, pack output and each arm's queried
workspace **before** capture/timing; keep each call's buffers and stream
lifetime independent. Test M=257, 2741, 2786, 16160, 32768 serially,
releasing a shape's graphs/buffers before the next. M257 covers the strict
`M>256` threshold; tail rows 2741/2786 stress edge tiles. This is the
gate/up N34816/K5120 path only, not a down or decode optimization.

Within one process, the decisive arms are: (1) fixed vLLM official
M128N128 BF16 GEMM + unchanged V2 quant; (2) validated 4×2 M64N256 fused
GEMM + its pack kernel; (3) 2×4 M64N256 fused GEMM + **the same** pack
kernel. All three must be compared as full output FP8 `[M,17408]` and
column-major scales `[M,136]` with the same input. As a decomposition,
also run old and new M64N256 **ordinary BF16** GEMM on the same packed
weights: inverse-pair the BF16 result and compare its `int16` bits with
official BF16. Time GEMM-only, inverse-only and V2 quant-only separately;
report ordinary BF16 GEMM + inverse + V2 as a *three-kernel diagnostic*,
not as a deployable fused-chain speedup. The matched ordinary/fused
mainloop-type assertion isolates the mainloop source change structurally,
while different epilogues and resource carveouts remain explicit limitations.

Before timing, require for every shape and tested finite/extreme distribution:
official-vs-fixed-vLLM BF16 bits; ordinary M64 inverse BF16 bits; full FP8
`uint8` bytes and FP32 scale `int32` bits for both fused arms against
official; finite status and downstream BF16 bits where applicable. Reuse the
previous zero, tiny, large and nonfinite/overflow cases, including NaN/Inf
conversion behavior, rather than relaxing a threshold. Then capture every
timed arm, change input bytes **and** scales, replay, prove output bytes
changed, and compare replay with a fresh official computation. A compile
success or capture log is not a replay witness. Any gate failure suppresses
all performance conclusions and model integration.

After these gates, time the three complete chains in one process using 31
or more randomly interleaved matched CUDA Graph pairs per M. Keep each
pair's seed, arm order and samples, report warm and owned-cache-disturbed
scopes separately, and verify fresh captured outputs before timing. The
owned >2×L2 flush stays outside event intervals; it does not guarantee a
fixed cold L2 state. A second small diagnostic may time old/new ordinary
BF16 GEMM-only with identical arm ordering. Compare candidate/old M64 to
attribute the warp edit; compare both to official to decide if the full
chain is worthwhile. Do not claim model benefit from a GEMM-only median.

For NCU, run one shape and selected arm per process under a single
`SM120_MAINLOOP:<arm>` NVTX range and `--profile-from-start off`, after
that shape's numeric and changed-input graph gate. Start with M2741 and
M16160, add M32768 because earlier scheduler data showed a large-shape
cache regression; M257/2786 need counters only if numerical or timing
results are anomalous. Record actual launch grid, registers/thread,
static/dynamic/driver-accounted shared memory, stage count, occupancy
limits, local load/store sectors, tensor instructions/active cycles,
executed instructions, shared-load wavefronts, L2 request bytes and DRAM
read/write bytes **per kernel**, plus full-chain totals. `lts__t_bytes.sum`
is L2 traffic, not DRAM, and Tensor active percent is not peak-FLOPS
efficiency. `--cache-control none` makes these diagnostic replay counters,
not controlled-cold-cache or unprofiled API latency. Save raw CSV,
command/status, source manifest, report SHA and a parser that rejects
missing kernels or mismatched input fingerprints.

Only after full-chain bitwise, Graph and 31-pair evidence shows a stable
gain at an actual model M should an isolated runtime be considered. Guard
that exact M inside the opaque op, retain canonical fallback for every other
M, prove real model kernel dispatch and run unprofiled A/B/A with equal
packed-weight/KV memory and embedding health. The previous AlongM/s1
M16160 synthetic ~2% result did **not** yield a credible model gain; this
new variant must meet that higher integration bar independently.
