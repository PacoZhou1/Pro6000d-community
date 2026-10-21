# Ordinary BF16 epilogue diagnostic (four independent GPU jobs)

All four jobs exited 0 and reported `numerics.pass=true` and
`summary.pass=true`. Each measured 31 randomized-order graph pairs for warm
and owned-cache-disturbed scopes. M16160 used the same synthetic activation
seed `20277274` and input SHA256
`e8c9023113f4e6b803d7326a17d03af5cd1d991ad88bda3b453bd065f1b95714`
in all three jobs; M2741 used seed `20263855` and SHA256
`754a3821009be6cd6304d3c5d529e3ce5d9c48c4b146642aad7382b61204d4de`.
The synthetic weights and candidate library were also fixed by the recorded
SHA values. These are operator checks, not model-quality checks.

Warm graph median, milliseconds (each cell from its own job):

| M | Raster / swizzle | Old official GEMM + V2 quant | New official GEMM + V2 quant | New official GEMM only | Paired BF16 GEMM only | Inverse only | V2 quant only | Paired GEMM + inverse + V2 quant |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 16160 | AlongM / 1 | 10.429 | 10.203 | 9.336 | 10.646 | 1.513 | 0.888 | 13.046 |
| 16160 | AlongM / 8 | 10.470 | 10.464 | 9.589 | 10.815 | 1.516 | 0.888 | 13.233 |
| 16160 | AlongN / 8 | 10.505 | 11.114 | 10.239 | 11.807 | 1.514 | 0.886 | 14.211 |
| 2741 | AlongM / 8 | 1.779 | 1.770 | 1.620 | 1.797 | 0.255 | 0.147 | 2.201 |

The same direction holds for the independently measured cache-disturbed
scope: at M16160, new official complete-chain medians were 10.225, 10.457,
and 11.104 ms for AlongM/s1, AlongM/s8, and AlongN/s8 respectively. Its
owned flush is outside the event; it does not establish a fixed cold-L2
state. The 2.1% old/new complete-chain advantage within the AlongM/s1 job
is a microbenchmark contrast only. The three M16160 configurations ran in
different jobs, so their medians are **not** strictly paired across jobs,
despite identical synthetic inputs. Paired BF16's inverse copy and V2 quant
are separately timed, but subtracting these medians does not allocate wall
time to an individual stage.

The paired BF16 GEMM and fused FP8 candidate have a compile-time identical
M64N256 CUTLASS `CollectiveMainloop` type and two stages. Their epilogues
and outputs differ: paired BF16 stores paired `[M,34816]` BF16, while fusion
uses the custom FP8 visitor plus a second pack kernel. Compiled shared
storage is 93,184 B versus 101,376 B per kernel respectively. The official
ordinary BF16 arm is a different M128N128 mainloop and uses 77,824 B kernel
shared storage; therefore official-versus-paired timing cannot isolate an
epilogue effect. All three report 384 max block threads and 256 MMA threads,
which does not make their CTA count, instruction mix, or effective occupancy
equal. The build's `ptxas` resource record reports 168 registers/thread for
the three kernels, zero spill for paired/fused and a small official spill;
the runtime effect still needs NCU. Exact mainloop type equivalence also
does not eliminate epilogue-dependent shared-memory carveout, launch/grid,
write-volume, or cache effects.

Sources: [four analysis JSON files](epilogue-m16160-along-m-s1/analysis.json),
[M16160 AlongM/s8](epilogue-m16160-along-m-s8/analysis.json),
[M16160 AlongN/s8](epilogue-m16160-along-n-s8/analysis.json),
[M2741 AlongM/s8](epilogue-m2741-along-m-s8/analysis.json), and the
[compiled resource log](build-v3/compile.log). Each job's `stdout.log`
contains the original metadata, numerical/graph witnesses and raw paired
timings. The scheduler eight-arm matrix must be analyzed separately; these
BF16 diagnostics do not establish a fused FP8 pipeline or API speedup.

## Minimum isolated long-prefill model experiment

The existing [frozen v6 sitecustomize](../gemm-epilogue/frozen-runtime-v6/sitecustomize.py)
registers a gate/up functional custom op through a strict post-grad FX
matcher. The [dispatcher](../gemm-epilogue/frozen-runtime-v6/candidate_op.py)
already sees the runtime row count and canonical FP8 `B/Bs`, while the
[matcher](../gemm-epilogue/frozen-runtime-v6/fx_matcher.py) proves the matched
BF16 GEMM has only the quantizer as a data consumer. A new opt-in copy of
that runtime can therefore keep the same FX replacement and packed-weight
startup in both A and B, but initially dispatch only `M == 16160` to the scheduler library's
`sm120_official_bf16` with canonical `B/Bs`, then call the existing V2 SiLU
quantizer on its BF16 output. Other legal matched M values take the
canonical fixed-vLLM GEMM + V2 path; unmatched FX chains remain on their
original graph path. This exact-M guard uses the long-prefill shape already
tested in the ordinary-BF16 diagnostic; other prefill shapes need their own
numeric, graph and paired performance gates before eligibility is widened.
Keep the M guard *inside* the opaque op so a large-M
compile range cannot statically eliminate the small-M fallback or vice versa.
Use the library's workspace query for each actual M, own the BF16 output and
workspace per invocation/capture, and record them on the current stream;
never use a process-global scratch buffer. The tested library and exact
raster `AlongM`/swizzle 1 are pinned by ABI and SHA. Leave the frozen v6 files
unchanged. The isolated A/B/A arms must retain the same extra packed weights,
KV-pool size, V2 quantizer, service configuration and prompts, so any
observed difference is attributable to the long-prefill GEMM dispatch rather
than a memory-capacity change. Small-M/decode must demonstrably take the
canonical path.

Before model performance interpretation, require full-bit BF16 and FP8/scale
equality on M16160 and a non-target prefill/decode shape, changed-input graph
replay, real post-grad FX and compiled-graph witnesses, and a trace/NCU
kernel witness from actual model replay (capture-time Python logs alone are
insufficient). Then run the same no-profiler model A/B/A prompt set with
matching payload hashes, output checks, embedding health and identical
memory conditions. Report TTFT/decode separately and retain startup/memory
cost; a small synthetic advantage is not evidence of model speed or quality.
This is a proposed integration route only; no runtime or service was changed
for this document.
