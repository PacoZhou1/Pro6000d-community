# Group-fast CTA ordering experiment

`fused_silu_quant_group_major.cu` is an isolated V2 of the bit-exact V1
`fused_silu_quant.cu`. It changes only the CTA-to-row/group-block mapping:

| | Neighboring CTAs | Kernel grid |
| --- | --- | --- |
| V1 | Same 4-group block, successive token rows | `(M, 34)` |
| V2 | Successive 4-group blocks within one token row | `(M * 34)` |

In V2, `row = blockIdx.x / 34` and `group_block = blockIdx.x % 34`.
Each CTA still has four warps, each warp still handles one 128-column group,
and gate/up arithmetic, scale floor, FP8 conversion, addresses, C ABI and
runtime shape guards are unchanged. All element addresses remain 64-bit.
The added C ABI grid-size check only rejects implausibly large `M` values
whose flattened CUDA x dimension would overflow the supported grid.

The possible gain is closer temporal locality across adjacent gate/up slices
of one token's 69 KiB BF16 input row. The possible loss is poorer locality
for the column-major FP32 scale stores, which V1 writes to adjacent rows
across CTAs; integer division/modulo per CTA also adds a small cost. Neither
effect can be inferred reliably from layout alone. Compare the variants on
the same synthetic input and the same GPU window. V1 has already passed the
full byte/scale and exceptional-input gate; V2 must pass it independently.

Build V2 with the existing builder, using a distinct output path:

```sh
python3 build_candidate.py \
  --source fused_silu_quant_group_major.cu \
  --output /build/libfp8_silu_quant_group_major.so --build
```

Run the unchanged `micro_probe.py --library` gate on each library, including
CUDA Graph replay timing. Only compare a V2 latency result if every FP8 byte
and FP32 scale bit matches the fixed official kernel for all probe cases.
Direct-call Event times can include host dispatch gaps; CUDA Graph replay is
the preferred isolated timing. Normal API A/B remains necessary to assess
model-level impact, and neither microbenchmark represents model quality.
