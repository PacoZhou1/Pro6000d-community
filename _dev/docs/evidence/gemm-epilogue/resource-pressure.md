# v3 resource-pressure hypothesis and M32 probe

The full v3 pipeline is bit-exact on all 12 selected synthetic cases and
changed-input CUDA Graph replay, but its graph-warm median is 29.37 ms at
M16160 versus 10.37 ms for the official GEMM plus V2 quant. At M2741 it is
5.11 ms versus 1.76 ms. The complete FP8 temporary and pack are included.
These timings reject v3 as a speedup.

CPU-only `cuobjdump --dump-resource-usage` reports 168 registers and an
8-byte stack per thread for the original N128 Cooperative GEMM, versus 168
registers and a 568-byte stack for the paired M64N256 Pingpong GEMM. Both
reports say `LOCAL:0`; the stack difference alone did not quantify spills.
An initial NCU counter query returned `ERR_NVGPUCTRPERM`, then a separate
owned profiling container with `SYS_ADMIN` collected the required counters
without a host-driver change. Sources: `baseline-resource.log`,
`candidate-v3-resource.log`, and `ncu-v3-m2741/{raw.csv,summary.json}`.

The M2741 full-pipeline NCU window contains two kernels per arm. In the GEMM
kernel alone, the candidate requests 440,143,872 local-load sectors and
440,143,872 local-store sectors, versus the official kernel's 8,623,164
and 12,032. At 32 bytes per sector, the candidate issues roughly 14.08 GB
of local-load and 14.08 GB of local-store sector requests. L2 request bytes
rise from 8.16 GB to 37.74 GB, while GEMM duration grows from 1.486 ms to
4.770 ms. The candidate pack kernel takes only 0.121 ms in this capture.
Whole-pipeline DRAM read+write bytes are 0.871× the baseline despite the
3.017× duration, so the bottleneck is not increased DRAM bytes. The observed
local traffic supports the spill diagnosis; NCU replay and uncontrolled
GPU cache/clock conditions limit direct comparison to unprofiled API timing.

The fixed CUTLASS SM120 blockwise builder uses 8 MMA warps for Cooperative
(`AtomLayoutMNK=(4,2,1)`) and 4 per active Pingpong consumer warpgroup
(`(2,2,1)`). Both official M128N128 and candidate M64N256 cover 16,384
output elements per tile, so Pingpong's active warpgroup has roughly twice
the FP32 accumulator elements per thread, before paired activation and
quantization. The SM120 shared-memory cap is 101,376 bytes. For M128N256,
two 128-K stages consume almost the entire budget, leaving too little for
the paired FP8 epilogue; merely switching this tile to Cooperative is not a
viable minimal change.

`build_candidate.py --tile-m 32 --match-official-scale-order` is the isolated
next experiment. It keeps each CTA's 256 paired N columns and the v3 FP32
scale-multiplication order, but halves its M tile and epilogue M extent.
Per-thread accumulators should be smaller, with the explicit cost of more
CTAs and repeated weight loads. Its compile success, resource use, full-bit
numerics, and whole-pipeline latency are all unverified. This flag does not
modify or replace the v3 binary or its evidence.

The M32 v4 result is now available: the owned binary builds with REG168 and
STACK0, and eight full-bit cases pass, including changed-input graph replay
for the normal-input shapes. Yet the
M2741 graph-warm pipeline remains 2.474 ms versus 1.675 ms for baseline,
and M16160 is 16.950 ms versus 10.380 ms. Removing spills alone does not
make this CTA shape competitive; the smaller M tile nearly doubles CTA count
and repeated weight reads. v4 is therefore not a deployment candidate.
M2741 NCU confirms zero local-load and local-store sectors in the v4 GEMM,
but L2 requests are 17.43 GB versus 8.16 GB official, with GEMM duration
2.342 ms versus 1.493 ms. DRAM reads are nearly equal at 193.5 MB. Thus
the remaining cost is not the spill seen in v3 and cannot be inferred from
DRAM traffic alone; doubled CTA work and L2 requests are measured.

An additional **compile-only** v5 probe tests whether the existing 8-warp
Cooperative schedule can run M64N256. The fixed CUTLASS kernel explicitly
asserts M>=128, so `--cooperative-experimental --tile-m 64
--match-official-scale-order` copies that pinned header into the owned build
directory and changes only its assertion to M>=64. This crosses an upstream
support-domain boundary. Compilation, if successful, is no evidence of
numerical or memory safety; full-bit, Graph, and full-pipeline checks remain
mandatory before any runtime integration.

The v5 Cooperative binary compiles with REG168/STACK0, but the strict GPU
gate fails badly on nonzero inputs. For M257 normal, 2,701,283 of 4,473,856
FP8 bytes and 23,251 of 34,952 scales differ from baseline; the downstream
BF16 projection differs at 1,309,201 of 1,315,840 elements. Zero and tiny
input cases pass because their quantized results collapse to zero or the
scale floor. These are not evidence that the Cooperative layout is correct.
The failed binary must not be used in the model or timed as an optimization.
Source: `candidate-coop-v5-full.jsonl`.

The same M64N256 Cooperative mainloop with an ordinary BF16 epilogue was
built and tested separately. Its inverse-packed BF16 output matches the
official GEMM at **all 8,947,712 elements** in each of the five M257
distributions (normal, zero, tiny, large, exceptional). Feeding this BF16
output through the V2 quantizer also matches the official FP8 and scales
bit-for-bit in all five. Yet the fused v5 output differs from that exact
Cooperative-BF16→quant reference by precisely the same counts reported
above. This rules out the Cooperative mainloop and scale-order patch as the
source of v5's observed numerical error. It localizes the fault to the fused
visitor and/or its native SF128 scale epilogue. Source:
`candidate-coop-v5-diagnostic.jsonl`.

The visitor currently pairs gate/up using two fragments of the *same thread*
at offsets `frag` and `frag + frags_per_half`. That mapping was validated
only for Pingpong's four-warps-per-consumer arrangement. Cooperative uses
eight MMA warps with a different atom layout. A CPU CuTe probe now finds
that all 8,192 M64N256 MMA-coordinate gate/up pairs do belong to the same
thread at a flattened half-fragment offset, with no missing or duplicate
coordinates (`layout-coop.json`). Thus the raw MMA mapping alone does **not**
explain v5; epilogue register-to-store retiling still needs inspection. The
SF128 reduction is also a
local shadow extension of fixed CUTLASS's documented SF16/32/64 visitor and
has not been independently validated for Cooperative. A numerical-only
identity-visitor diagnostic now retains both BF16-rounded halves and uses
the same SF128 conversion/store; comparing its raw FP8 temporary and all
native scales against a Pingpong identity build will distinguish the
cross-half pairing from scale reduction/store faults. Until that runs and a
full-bit retest passes, there is no justified one-line v5 repair.

The isolated v6 flag `--cooperative-sf128-two-warps` tests a specific warp
ownership hypothesis without altering v5. The CPU layout's row-0 examples
place N=0 on warp 0 and N=8 on warp 4 within the same 128-column group.
The v5 shadow assumes four contributing warps and folds warp 0 with warps
2, 4, and 6; fixed CUTLASS's existing two-warp Cooperative branch pairs
warp 0 with warp 4. v6 uses that branch and eight local fragments per 128
group. The example coordinates alone do not prove the entire group mask,
so the identity visitor's GPU comparison and the v6 numerical gate test
the hypothesis. The expanded CPU warp-mask probe remains unverified.

Those numerical checks have now run. The v5 identity visitor, which skips
SiLU and quantizes both BF16-rounded halves, differs from the Pingpong
identity visitor at 6,810,443/8,947,712 FP8 bytes and 57,485/69,904
scales (`identity-visitor.json`). This confirms an error remains in the
Cooperative visitor/SF path even without gate/up multiplication. Its separate
PyTorch mathematical reference has small FP8 and substantial low-bit scale
differences from the Pingpong path, so that secondary reference is not used
as a bit-exact oracle. The direct Pingpong-vs-Cooperative comparison and the
Cooperative BF16 exact result are the decisive evidence.

The v6 shadow differs from v5's visitor shadow in exactly one generated
line: for SF128, `WarpsPerSF` is `2` instead of `4`. The existing CUTLASS
two-warp Cooperative path then combines warp `w` with `w ^ 4`; the local
fragment count per scale changes from four to eight. The v6 library SHA256
is `e4039ddc95fa3b56f02743357b7a7193156dc500c5d8804bb77ebd3af85450d6`.
The full-pipeline GPU probe reports **12/12 numerical cases passed**,
including active M=257, 2741, 2786, 16160, 32768; fallback M=1, 5,
256; and zero, tiny, large, exceptional M257 inputs. Its changed-input CUDA
Graph replay also passes. Every reported FP8-byte, FP32-scale-bit, and
downstream BF16 comparison passes within this finite tested matrix; this is
an empirical fixed-shape result, not a proof for all shapes or inputs.
Source: `candidate-coop-v6-full.jsonl`.

The complete candidate GEMM+pack remains slower than the official GEMM+V2
quant in 31 paired graph-warm trials: M16160 12.606 versus 10.509 ms
(~20.0% slower), M2741 1.950 versus 1.777 ms (~9.7% slower), M2786
2.015 versus 1.802 ms (~11.8% slower), and M32768 25.295 versus 22.174 ms
(~14.1% slower). Cold measurements have the same direction. M257 and
fallback small-M differences are near the timing resolution and do not
establish a benefit. v6 substantially reduces v3/v4's latency penalty but
is not a synthetic-pipeline speedup. NCU and normal-model A/B/A remain
separate evidence; no whole-model gain or quality equivalence follows from
these synthetic tests alone.

The v6 build still shadow-relaxes CUTLASS's explicit Cooperative CTA M>=128
assert to M>=64. Despite the tested numerical/Graph passes, it lies outside
the upstream supported configuration and cannot be described as generally
safe for arbitrary shapes, layouts, inputs, drivers, or future CUTLASS builds.
The binary is isolated from installed packages and is not a deployment
recommendation.
