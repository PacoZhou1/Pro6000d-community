# M257 v2 failure: independent numerical diagnosis

Reviewed locally on 2026-10-14. Input evidence is
`candidate-v2-m257.jsonl` and the root-run GPU diagnosis
`candidate-v2-diagnostic.jsonl`; this reviewer made no GPU connection. Candidate
library SHA256 was `ad1aa671156bfce18792da32474d3654aac0ccce17b2117aa56295e44e366cd0`.
The exact gate failed and no timing was emitted. The diagnosis reproduces every
candidate FP8/scale bit by quantizing the ordinary same-tile N256 BF16 output.
The underlying mainloop source-level cause is still under investigation. This
document does not accept the candidate or weaken the gate.

## Root-run same-tile diagnostic result

Diagnostic BF16 library SHA256 is
`ea042680cfb4b81a9bb235a27be919ceaaa1c99e58c5cfda1b2d964cffb39e7d`;
probe SHA256 was
`eed1a8d2fa5e97d0b0dfefcb504c6ca9d8493d1c00997788e1343aadd60c42ae`.
The ordinary M64N256 Pingpong BF16 GEMM uses the same packed operands as the
candidate. Every distribution has **zero** candidate-versus-N256-BF16-requant
FP8 and scale differences. V2 and official quant also have zero differences
when receiving the same N256 BF16 tensor.

| Distribution, M257 | N256 BF16 vs official bit differences | N256 BF16 re-quant vs baseline FP8 bytes / scales | Candidate vs N256 BF16 re-quant FP8 bytes / scales |
| --- | ---: | ---: | ---: |
| normal | 791 | 53 / 2 | 0 / 0 |
| zero | 0 | 0 / 0 | 0 / 0 |
| tiny | 840 | 0 / 0 | 0 / 0 |
| large | 928 | 35 / 0 | 0 / 0 |
| exceptional | 2,146 | 63 / 2 | 0 / 0 |

Thus the same-tile GEMM BF16 differences, followed by unchanged reference
quantization, reproduce the entire measured candidate mismatch. There is no
additional observed final-output error attributable to the fused quantization
visitor in these five cases. This is a measured statement for the selected
operands, not an exhaustive proof of visitor correctness.

For normal input, 20 of 53 differing bytes occur in the two scale-mismatch
groups `(19,126)` and `(234,97)`; **33 occur in bit-identical-scale groups**.
At row12/column520, the scale bits match, official gate is 0.037353515625,
N256 gate is 0.037109375, and up is unchanged at -1.546875. Re-quantized
N256 and candidate both give byte196 while baseline gives byte197. Large
input has all 35 differing bytes in bit-identical-scale groups: at
row0/column13726, gate changes 20.25→20.375 with up fixed at 25.625; both N256
re-quant and candidate give byte115 versus baseline byte114.

Exceptional row4/column3349 now has an actual witness: official rounded gate/up
remain finite at approximately -1.0767e38/-9.3046e37, while the ordinary N256
GEMM yields -Inf/-Inf. Both group scales are +Inf. Official result becomes zero
after quantization, whereas N256 re-quant and candidate both become +448
(byte126), exactly consistent with the reference exceptional arithmetic below.
This replaces the earlier overflow hypothesis with a same-tile measured
BF16 overflow witness; it does not identify the precise scale-accumulation
instruction order responsible.

## What the measured failure establishes

For M257 normal input, fixed baseline GEMM BF16, fixed packed GEMM after inverse
permutation, and V2 FP8/scale all exactly reproduce the corresponding official
operations. Candidate output differs in 53 FP8 bytes and two FP32 scales. The
first byte difference is row12/column520; the first scale difference is
row19/group126. Down GEMM has 5,799 BF16 bit differences, so the small number of
quantized byte differences is not a reason to relax the gate.

Zero and tiny inputs pass. Large input differs in 35 FP8 bytes while every
scale bit matches; therefore scale disagreement cannot explain all failure
distributions. The first large byte difference is row0/column13726.

Exceptional input has 63 FP8 differences and repeats the same two normal scale
differences. Its first FP8 difference is row4/column3349, with candidate byte126
(+448) versus official byte0. Row4 is the injected maximum finite FP32 A-scale
case, not a row where the probe directly injects NaN or infinity. Its GEMM
intermediates can nevertheless overflow. The unchanged row19 scale difference
belongs to the ordinary numerical issue. The candidate and baseline FP8 outputs
both have zero nonfinite elements in the recorded diagnostic, which does not
establish finite GEMM input, finite intermediate products, or exhaustive
finite-FP8 correctness.

## Fixed reference semantics

`pipeline_probe.py` first quantizes a BF16 normal activation, then exceptional
mode injects E4M3FN byte `0x7f` at A[0,0], A-scale +Inf/-Inf/NaN at rows1/2/3
group0, and maximum finite FP32 at row4/group0. Rows >=5 use the same normal
activation. This tests selected exceptional A/scale operands; it does not
enumerate every exceptional BF16 gate/up pattern or all finite FP8 inputs.

The official `fused_silu_mul_block_quant.cu` loads **rounded BF16** gate/up as
FP32, computes `sigmoid = 1/(1+expf(-gate))`, `silu = gate*sigmoid`, and
`product = silu*up`. Per-group maximum uses `fabsf` and a tree of `fmaxf`.
With no upper bound, scale is `fmaxf(group_max/448, 1/(448*512))`.
`ScaledQuant<FP8,false>` computes **division** `product/scale`, then applies
`fmax(-448, fmin(value,448))`, then FP8 conversion. The V2 implementation uses
the same operation order, division, minimum scale and saturation, changing
thread organization. The recorded V2/official full-bit equality is the
reference evidence for these particular inputs.

Consequences for exceptional values:

- `fmaxf` chooses the numeric operand when paired with one NaN. A lone NaN does
  not necessarily make the scale NaN. If every product magnitude is NaN,
  `fmaxf(NaN,min_scale)` produces the minimum scale.
- Positive infinite group maximum gives infinite scale. A finite product
  divided by infinite scale gives signed zero; Inf/Inf gives NaN.
- The final clamp turns NaN into +448 under this `fmin/fmax` sequence. Thus an
  otherwise finite product changed to NaN can produce candidate +448 versus
  official zero even when the two infinite scale bits match.
- Negative-infinite gate goes through `-Inf * sigmoid(-Inf)` with a zero
  sigmoid, which can produce NaN. Replacing it with a stable algebraic
  expression that happens to return zero changes the reference semantics.

These are source-level consequences. The later ordinary same-tile BF16 witness
at row4/column3349, described above, demonstrates the finite→-Inf difference
that reproduces its observed +448-versus-zero output.

## Diagnostic added without changing acceptance

`--diagnostic-gemm-library` accepts the root-built ordinary M64N256 Pingpong
BF16 epilogue library (`n256_block_fp8`), with independent owned workspace and
`--diagnostic-gemm-sha256`. For each M>256 distribution, the probe:

1. Inverse-permutes its full paired BF16 result and compares every BF16 bit to
   the official canonical GEMM.
2. Sends that exact BF16 result through V2 and official quant, checks their
   entire FP8/scale output, then compares it to baseline and candidate.
3. Counts FP8 mismatches inside scale-mismatch versus bit-identical-scale
   `(row,group128)` pairs, retaining a few coordinates and scalar gate/up bit
   witnesses. No activation/weight tensors are written to evidence.

If candidate equals N256-BF16 re-quantization exactly and N256 BF16 differs from
official, the N256 GEMM change explains the observed final-output differences
in this triangulation. If N256 BF16 matches official but candidate does not
match its re-quantization, visitor math, group reduction, conversion or mapping
becomes the next investigation. If candidate differs from re-quantization even
when N256 BF16 differs, both paths may contribute.

The ordinary BF16 epilogue uses different epilogue/store resources from the
fused FP8 epilogue. Its stage carveout can also differ. Therefore these outcomes
do not directly observe the fused kernel's actual rounded BF16 values. A
conclusive second-stage diagnostic would side-store the rounded gate/up inside
the same fused visitor/mainloop, without changing its math, tile, stage count
or output path, and use an explicitly versioned debug ABI. That ABI has not
been assumed or implemented here.

## Lesson for the next implementation

Tile/schedule changes need a GEMM BF16 equivalence check before debugging the
activation/quantization math. Final quantization can hide intermediate
differences: tiny input has 840 BF16 differences but zero FP8/scale differences.
Conversely, a one-BF16-step gate/up change can cross an FP8 threshold while the
scale stays bit-identical, and an overflowing mainloop can change exceptional
behavior by hundreds in FP8 numeric value. The next source investigation should
compare the official Cooperative and candidate Pingpong block-scale
accumulation order and stage organization, then rerun the unchanged exact gate.

The kernel agent subsequently identified a concrete source difference: the
official N128 path computes `tmp_accum*(A_scale*B_scale)`, while the N256 branch
computes `(tmp_accum*A_scale)*B_scale`. FP32 rounding and overflow can differ
between those expressions. `build_candidate.py --match-official-scale-order`
now provides a v3 experiment that groups A/B scales as in the official path.
This source observation is consistent with the diagnostic, but the v3 fix has
not been accepted by this review; root must run the original full-bit gate on
the rebuilt binary, all required shapes and changed-input graph replay.
