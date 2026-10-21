# SM120 gate/up FP8 GEMM resource model

This is a fixed-shape, evidence-backed model for the RTX PRO 6000 Blackwell
Workstation GPU running vLLM 0.31.0 at commit
`db9527a46873454610df6dbedf79a36d6bf1a7f6`. The target gate/up is
FP8 A `[M,5120]` times FP8 B `[34816,5120]`, block scales `(M1,N128,K128)`,
followed by BF16 rounding, SiLU×up and per-128 FP8 quantization. The
official GEMM is M128×N128×K128 Cooperative and writes BF16. The v6 fused
candidate is M64×N256×K128 Cooperative, writes paired FP8 plus native
scales, and uses a separate pack kernel. The packed-weight copy adds
11,589,593,600 bytes across the 65 targeted layers in both controlled
model arms. It is an actual deployment cost, not a free layout change.

The reproducible arithmetic is in `resource_model.py` and its generated
`resource-model.json`. It reads saved NCU CSVs; it does not initialize CUDA.
Hardware numbers below come from `ncu-coop-v6-m2741/raw.csv` device attributes:
188 SMs, 65,536 registers/SM, 102,400 shared bytes/SM (101,376 dynamic
bytes/block opt-in plus 1,024 driver-accounted bytes; kernel static shared
memory is zero in these launches), 1,536
threads/SM, 48 warps/SM, and 128 MiB reported L2. Cluster launch reports
supported; no candidate in this round used a cluster. These are measured
attributes of this card/driver, not general SM120 specifications.

| GEMM | CTA output | MMA warps | FP32 accumulators/MMA thread | Total block threads | Registers/thread; nominal registers/CTA | Shared/CTA | Resource bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official Cooperative | 128×128 | 8 | 64 | 384 | 168; 64,512 | 78,848 B | 1 CTA/SM |
| v3 Pingpong | 64×256 | 4 active per tile | 128 | 384 | 168; 64,512 | 102,400 B | 1 CTA/SM |
| v4 Pingpong | 32×256 | 4 active per tile | 64 | 384 | 168; 64,512 | 86,016 B | 1 CTA/SM |
| v6 Cooperative | 64×256 | 8 | 64 | 384 | 168; 64,512 | 102,400 B | 1 CTA/SM |

The 384 launch threads include producer/scheduler roles; dividing
accumulators by all 384 would understate pressure. An M128N128 and M64N256
CTA each produce 16,384 FP32 accumulator values, but the former and v6
distribute them over 256 MMA threads (64 values or 256 B/thread), while v3
Pingpong's active 128-thread consumer owns 128 (512 B/thread). The
corresponding FP32 accumulator payload is 8 KiB per active MMA warp for
official/v6 and 16 KiB for v3. These are lower bounds on live register
demand; operands, scale factors, addresses, epilogue values and compiler
temporaries also consume registers. `cuobjdump` reports REG168 and STACK8
for official, REG168/STACK568 for v3, and REG168/STACK0 for v4/v6. The
v3 NCU GEMM requests 440,143,872 local-load and the same local-store
32-byte sectors versus official's 8,623,164 loads/12,032 stores and zero
local sectors for v4/v6. The v3 excess is consistent with substantial
register spilling, but STACK bytes alone are not a traffic measurement.

At REG168×384, a CTA nominally reserves 64,512 of 65,536 registers, so
two CTAs cannot reside on one SM. All four also exceed half the SM shared
capacity; NCU reports both `launch__occupancy_limit_registers=1` and
`launch__occupancy_limit_shared_mem=1`. Equal one-CTA occupancy does not
imply equal throughput: v3 has twice the accumulator payload per active
MMA warp and measured local traffic, while v6 has more operand traffic and
a fuller shared-memory footprint. The one-CTA limit leaves only 12 launch
warps resident per SM, although this metric includes non-MMA roles and
does not describe temporal activity of those warps.

For FP8, the raw A+B payload of one K128 shared-memory stage is
`(tileM+tileN)×128` bytes: 32,768 official, 40,960 v3/v6, and 36,864 v4.
Two stages consume 65,536, 81,920 and 73,728 bytes respectively. Measured
whole-block shared memory leaves 13,312 B official, 20,480 B v3/v6, and
12,288 B v4 beyond those *raw operand* bytes; this remainder includes
scales, pipeline state, epilogue and padding, not free space. The fixed
CUTLASS builder computes stage count from A/B, scale and pipeline bytes
after an epilogue carveout, requires at least two stages, and asserts
tile K equals scale granularity K=128. Thus an M128N256 tile already needs
98,304 raw A+B bytes for two stages; only 3,072 dynamic bytes remain under
the 101,376-byte opt-in limit before scales/pipeline/epilogue. The previous
M128N256 compile reached Stages=1 and failed. Reducing K to 64 is forbidden
by the fixed builder's scale-granularity assertion. A larger equal-area tile
is not automatically feasible or efficient.

For M=2741, the grid and logical CTA operand reads are:

| GEMM | CTA grid | One-CTA/SM waves | Logical FP8 A reads | Logical FP8 B reads | Logical A+B | NCU L2 requests | NCU DRAM reads | GEMM time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official | 5,984 | 31.83 | 3.922 GB | 3.922 GB | 7.843 GB | 8.162 GB | 192.87 MB | 1.498 ms |
| v3 | 5,848 | 31.11 | 1.916 GB | 7.665 GB | 9.581 GB | 37.745 GB | 192.96 MB | 4.770 ms |
| v4 | 11,696 | 62.21 | 1.916 GB | 15.330 GB | 17.246 GB | 17.432 GB | 193.54 MB | 2.342 ms |
| v6 | 5,848 | 31.11 | 1.916 GB | 7.665 GB | 9.581 GB | 9.752 GB | 192.89 MB | 1.685 ms |

GB/MB above are decimal. Each CTA logically loads `tileM×5120` A bytes
and `tileN×5120` B bytes over 40 K stages. N128→N256 halves the number of
N CTAs, so A duplication halves. M128→M64 nearly doubles the number of M
CTAs, so B duplication nearly doubles. B is 178,257,920 unique bytes
(170 MiB), larger than the device's 128 MiB L2, but the observed DRAM read
for each GEMM is only about 193 MB, close to one B copy plus the 14.03 MB
unique M2741 A tensor. This *does not* mean B fits in L2: cache lifetime,
streaming and profiler replay affect DRAM reads. It does show that the extra
CTA weight requests largely appear at L2 rather than DRAM in this captured
run. NCU `lts__t_bytes.sum` counts L2 requests and must never be labeled
DRAM traffic or cache capacity. v4's 2.14× official L2 requests track the
2.20× logical A+B model, while v6's 1.19× tracks its 1.22× model. v3's
37.745 GB is far above 9.581 GB logical operand reads; its ~28.2 GB excess
is close to its local load+store sector requests (~28.17 GB), explaining
why equal-area Pingpong tiles were especially slow.

At M16160, official and v6 have 34,544 and 34,408 CTAs respectively,
about 184 and 183 theoretical one-CTA/SM waves; v4 has 68,680 (~365 waves).
At M32768 official and v6 each have 69,632 CTAs; v4 has 139,264. Similar
CTA counts do not make official and v6 equivalent: v6 repeats B across
twice as many M tiles, and its output epilogue performs BF16 rounding,
SiLU/FP8 group reduction and conversion. An earlier, separate baseline
M16160 NCU report records tensor-pipe active cycles at 88.80% of the
active-cycle reference, SM throughput 88.45% of sustained elapsed reference,
and DRAM throughput 14.36% of sustained elapsed reference. In the newer
paired full-pipeline capture below, the baseline tensor-active percentage
is 83.59%; these are different runs, not an inconsistency to average away.
**Tensor-pipe active-cycle percent is not achieved tensor FLOPS/peak
FLOPS**; instruction counts, issue rate, clock and utilization are required
for that claim.

The new same-run NCU captures in `ncu-m2741/raw.csv` and
`ncu-m16160/raw.csv` include four kernels in this order: official GEMM,
official V2 SiLU+quant, v6 GEMM, and v6 pack. `resource-model.json`
preserves every selected counter with NCU's **display unit**: the CSV uses
Mbyte at M2741 for DRAM and Gbyte at M16160, so their numeric fields must
not be mixed without unit conversion. The runs used `--clock-control none`
and `--cache-control none`; they are kernel-replay observations, not API
latency or controlled-cold-cache measurements.

| GEMM-only counter | M2741 official | M2741 v6 | M16160 official | M16160 v6 |
|---|---:|---:|---:|---:|
| GEMM time, ms | 1.494 | 1.672 | 8.638 | 10.661 |
| Tensor-pipe active, % of elapsed reference | 82.00 | 67.91 | 83.59 | 67.21 |
| Total executed instructions, million | 523.40 | 597.83 | 3018.54 | 3518.67 |
| Tensor instructions, million | 122.55 | 119.77 | 707.46 | 704.68 |
| Shared load wavefronts, million | 193.66 | 275.54 | 1117.94 | 1621.09 |
| L2 request bytes, GB | 8.162 | 9.752 | 47.133 | 57.310 |
| DRAM read bytes, GB | 0.193 | 0.193 | 0.951 | 6.385 |
| DRAM write bytes, GB | 0.150 | 0.065 | 1.093 | 0.583 |

The M2741 v6 GEMM executes 74.43 million more total instructions (+14.2%)
but 2.79 million fewer tensor instructions (−2.3%; edge/padding and tile
layout also differ). ALU, FMA and LSU counts rise by 17.00, 17.45 and
14.07 million, and XU instructions rise from zero to 9.36 million. Its
shared-load wavefront count rises by 81.88 million (+42.3%), while shared
load bank conflicts rise only from 133,215 to 144,413. At M16160, total
instructions rise 16.6%, shared-load wavefronts 45.0%, and tensor-active
percentage falls by 16.38 points. Source provides plausible mechanisms:
the N256 mainloop stages 40,960 rather than 32,768 raw FP8 operand bytes
per K128 step and has two N128 B-scale groups instead of one; the patched
`sm120_mma_tma_blockwise_scaling.hpp` copies and applies per-fragment
block scales; `paired_gateup_visitor.cuh` converts both accumulators through
BF16 and evaluates `expf`, product and FP8 quant; its patched SF128 store
uses `sAmaxs`, named-barrier interwarp reductions and shared loads. The
extra XU activity is compatible with nonlinear math in the fused GEMM, but
this counter alone cannot assign individual instructions to `expf`. Nor can
shared wavefront totals distinguish mainloop operand reads from scale
reduction/epilogue reads without SASS/source-address attribution. At M2741,
`long_scoreboard` stall percentage **falls** from 6.21 to 4.83 while
`mio_throttle` rises 0.67→2.19 and `short_scoreboard` 0.46→1.83; eligible
warps per cycle are 0.283→0.279 and issue-active 22.32→21.72%. M16160
shows the same directions. These are percentages of warp states, not
exclusive wall-time slices; the lower long-scoreboard percentage does not
prove memory is irrelevant, and the larger shared-load count does not by
itself prove shared memory is the latency limiter.

The longer chunk reverses the short-chunk DRAM result. Across both pipeline
kernels, official M16160 DRAM reads+writes total 3.404 GB, versus 7.511 GB
for v6+pack. v6 GEMM alone reads 6.385 GB versus 0.951 GB official, even
though its BF16-eliminating output reduces GEMM writes by about 0.510 GB.
This is material: the packed B is 178.26 MB, larger than the reported
134.22-MB L2, and B's **logical** CTA reads grow from 22.64 to 45.10 GB
when M tiles double from 127 to 253. A's logical reads halve from 22.64
to 11.27 GB because N tiles halve. Thus, an altered cache-reuse distance or
CTA schedule could convert a fraction of repeated B requests into DRAM
misses for long M even if short M mostly serves them from L2. This is a
falsifiable cache hypothesis, **not** a buffer-level attribution: aggregate
NCU DRAM and L2 counters do not identify B versus A, temp, scale, or cache
eviction cause; the capture does not fix clocks/cache state. The parent is
repeated M16160 with the same `cache-control none` policy: pipeline DRAM
total remained 7.450 versus 3.409 GB (v6/official 2.185×), and time
11.348 versus 9.504 ms (1.194×). This confirms the observation across
two captures but not its buffer or cache mechanism. The separate
`ncu-m16160-cache-all` attempt exited 11 and produced **no usable numeric
or counter result**; it cannot establish a cold-cache effect.

The later [fixed-source audit](../sm120-scheduler/scheduler-shape-notes.md)
corrects the original scheduler explanation here. SM120's
`TileSchedulerSelector<void,...>` selects `PersistentTileSchedulerSm100`
(CLC), **not** the Sm90 scheduler. Its params reuse the Sm90 raster
heuristic on unrounded tile counts, but its physical grid is the full
CTA grid, transposed for AlongN and never rounded up for swizzle.
At M2741 the official (22,272) and v6 (43,136) logical grids both select
AlongM. At M16160 official (127,272) selects AlongM while v6 (253,136)
selects AlongN. Thus the direction-change observation remains valid,
but the former rounded-grid and unconditional fast-axis explanation does not.

For cluster=1, s1 leaves coordinates unchanged before the AlongN transpose;
physical x neighbors change logical M under AlongM and N under AlongN.
For s8, groups of eight x coordinates in the divisible interior instead
vary the other logical dimension; residual regions are not remapped.
These are coordinate mappings, not measured execution order: CLC dynamically
claims work. The corrected `resource-model.json` records `clc_*` fields
and preserves the old assumed Sm90 arithmetic only as `legacy_sm90_*`,
explicitly invalid for this launch. Operand byte counts and raw NCU records
are unchanged.

The [controlled follow-up](../../14-sm120-scheduler-epilogue.md) has now
compared raster/swizzle on the same v6 tile and math. M16160 fused GEMM
DRAM reads fall from 6.389 GB (AlongN/s8) to 0.261 GB (AlongM/s1), but
the complete fused chain still loses to the official pipeline. This supports
a schedule-dependent aggregate traffic difference, not byte-level buffer
attribution. Even official H8 versus AlongM/s1 changes sign across shapes:
M16160 improves slightly, while M32768 regresses materially, with increased
DRAM reads and reduced Tensor activity. Source locality and A/L2 capacity
are useful hypotheses; counters do not prove that all excess reads are A.

The M2741 full-pipeline NCU window clarifies the remaining v6 gap. In the
new capture, official GEMM+V2 quant is 1.494+0.135≈1.629 ms; v6 GEMM+pack
is 1.672+0.118≈1.790 ms. The ~0.177 ms slower GEMM, rather than the pack
alone, explains the ~0.161 ms complete-pipeline gap. Removing pack with
zero replacement cost would at most remove its measured 0.118 ms here and
still leave the fused GEMM slower than the entire official chain by about
0.042 ms under this profiled condition. Unprofiled paired graph-warm
times have the same direction (1.950 versus 1.777 ms). Profiler replay
times and normal API latency are distinct measurements.

The two new runs supplied instruction, pipe, shared-wavefront and stall
counters. The next controlled pass should repeat M16160 under fixed replay
and explicit cache policy, retaining four separate kernel rows and
launch/grid/DRAM/L2 counters. To locate the new DRAM reads, collect an
available L2 hit/read/write split, global versus TMA transactions or source
addresses, and a cold/warm comparison for the same synthetic allocation;
the aggregate `lts__t_bytes.sum` is insufficient. To separate the extra
shared reads, collect source-correlated SASS PCs or instruction sampling
for mainloop and epilogue and shared transaction/throughput counters. Query
the installed GB202 metric catalog first; an unavailable metric is missing
evidence, not zero. These observations can falsify the following explanations:

1. **Repeated B/cache reuse:** v6 adds ~1.59 GB L2 requests at M2741 and
   ~10.18 GB at M16160, with much larger extra DRAM reads only at M16160.
   The same-policy repeat shows the DRAM gap persists; a successful
   cache-policy contrast is still missing.
   Forced raster options on the *same tile and binary* isolate the source-
   confirmed scheduling change. They may improve or worsen DRAM reuse;
   the direction must be measured. If only total L2 changes without buffer
   attribution, the B-specific claim remains unproved.
2. **Instruction/shared pressure:** v6 tensor instruction count is slightly
   lower while total instructions and shared-load wavefronts rise markedly;
   tensor-active fraction falls. PC/source attribution should show whether
   added loads are operand fragments, scale application or SF128 reduction.
   Cutting one source must reduce its counters and time without changing
   numerical bits; stall-state percentages alone cannot apportion time.
3. **Resource-limited latency hiding:** both launches already have one
   CTA/SM. If eligible warps collapse during TMA/epilogue waits, changing
   only register count without cutting shared below 51,200 B cannot add a
   second resident CTA. If occupancy remains fully used, extra CTAs are not
   the explanation for v6 versus official, but v4's doubled waves are.

Resource-feasible *hypotheses* for a next round are ordered by evidence
and implementation cost. None is implemented or numerically validated here:

1. **Control raster and swizzle independently in the existing scheduler.**
   Expose `TileSchedulerArguments.raster_order`; cross it with current
   `max_swizzle_size` values 1 and 8 in the four-arm probe above, without
   changing tile, fragment mapping, FP8 math, shared footprint or CTA count.
   M16160 is decisive because the current heuristic flips there; M2741
   checks whether either control has a cost absent a flip. The ABI/runtime
   should retain the present heuristic and swizzle by default; forced
   controls belong in the isolated probe. Same bits are expected because
   each CTA's K reduction is unchanged, but must be checked. The paired
   time/DRAM/L2 response will determine whether visitation order, swizzle,
   an interaction, or neither merits further work; no direction is assumed.
2. **Direct compressed FP8 output from the v6 epilogue.** Map each paired
   CTA's gate 128-product group directly to contiguous `[M,17408]` output
   and column-major scales, eliminating paired FP8 temp and pack while
   keeping the same BF16-rounded SiLU and per128 reduction. The existing
   register-held values and scale can in principle be stored without an
   extra M/N CTA or shared stage, but CUTLASS's TMA D store currently
   assumes the paired `[M,34816]` layout, so a coordinate-aware non-TMA
   store or new epilogue mapping must prove coalescing and bit identity.
   The 0.120-ms M2741 pack time is an upper-bound *opportunity*, not a
   predicted speedup; alone it does not close the measured gap.
3. **Conditional two-CTA B multicast across M.** Only after the raster
   experiment and a compile-only support check, investigate a cluster
   shape `(2,1,1)` with adjacent M64N256 CTAs sharing a B K128 tile.
   A theoretical B logical-read reduction could approach `ceil(M/128)`
   from `ceil(M/64)` without exceeding one CTA per SM, because two CTAs
   would occupy two SMs. The present fixed builder instantiates cluster
   `(1,1,1)`; launch capability alone does **not** prove its TMA multicast
   path works or improves GB202 performance. Support, numerical identity,
   TMA/L2/DRAM byte reduction and time are all feasibility gates. This is
   lower priority than raster because it requires a new kernel and may add
   cluster synchronization while the source of the DRAM gap is unresolved.

No design may alter FP8 weight/activation precision, BF16 GEMM-output
rounding, scale semantics, or the model's BF16 lm_head. The embedding service
stays running; the original 27B/9B/ASR services remain stopped until the
user explicitly requests a restart. This document and script neither use
GPU resources nor change service state.
