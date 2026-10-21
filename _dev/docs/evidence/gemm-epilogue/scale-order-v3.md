# M64×N256 block-scale arithmetic diagnostic

The v2 fused epilogue did not add a numerical discrepancy: across all five
tested M257 synthetic distributions, its FP8 output and scale were bit-exact
to the *same* M64×N256 CUTLASS GEMM's BF16 output followed by the already
verified V2 SiLU/quant kernel. For normal inputs, that N256 GEMM differed from
the fixed vLLM M128×N128 BF16 GEMM at 791 of 8,947,712 elements; the ensuing
quantization reproduced the candidate's 53 FP8-byte and two scale mismatches.
See `candidate-v2-diagnostic.jsonl`. This identifies the block-scaled GEMM
mainloop as the source of the discrepancy; the fused visitor remains subject
to subsequent full-shape numerical validation.

The pinned CUTLASS
`cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp`
(SHA-256 `88bfdb527044d520c98fead4f0d1d59c051950642722b5344ad656e679287bd8`)
uses two FP32 expression orders. The official N128 tile has one B scale per
tile and first forms `scale_a * scale_b`, then accumulates
`tmp_accum * scale_ab`. The N256 tile spans two B scale groups and uses the
generic `tmp_accum * scale_a * scale_b` expression, whose first product is
rounded before applying B. Floating-point associativity makes a BF16-rounding
change at rare output values plausible and explains why this is an N-tile
specific issue. This is a source-level mechanism, not proof that changing it
is sufficient.

`build_candidate.py --match-official-scale-order` creates a shadow CUTLASS
header only in its owned build directory. It changes the generic branch to
`const ElementSF scale_ab = scale_a * scale_b; accum += tmp_accum * scale_ab;`
and leaves the original N128 branch, MMA K loop, tile, schedule, and visitor
untouched. The builder verifies the original header hash. The local generated
shadow header hashes to
`2920355ce886064701479b567dfec56a778042ad3cdd854b677abd7e7721c303`;
the captured v3 compile metadata records the source-header hash and scale
order but not this generated-header hash. The same builder also exports a
host-only workspace-size query; C ABI launch version remains 2. The v2
library and evidence are preserved.

The v3 gate is unchanged: compiled launch plus full-bit comparison against
the official GEMM→V2 quant path for every tested shape and finite/exceptional
distribution, followed by CUDA Graph correctness and whole-pipeline timing.
No v3 numerical or performance result is implied by this patch.
