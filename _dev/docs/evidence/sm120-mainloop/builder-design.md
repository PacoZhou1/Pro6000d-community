# First M64 mainloop builder (CPU source emission only)

`build_mainloop.py` imports the **SHA-pinned**
[`build_scheduler.py`](../sm120-scheduler/build_scheduler.py) helpers for the
validated v6 input checks, dispatch adapter, private host state and unchanged
FP8 pack body. It additionally verifies the fixed CUTLASS
`sm120_blockwise_mma_builder.inl` SHA before writing a shadow header under
its *own* build directory. It never edits vLLM, installed CUTLASS or earlier
experiment sources. The emitted source tree and manifest are reviewable
without CUDA initialization or a compiler.

`control42` retains Cooperative `4×2×1` MMA-warp layout for the target
M64×N256×K128 tile and the v6 SF128 two-warp visitor. `warp24` changes
only that tile's Cooperative atom layout to `2×4×1`, leaves its MMA
permutation N32, and switches the SF128 visitor to the existing four-warp
branch, whose source explicitly assumes a 2×4 layout. This SF ownership
patch is a **second, separately hashed source difference** necessary for
correctness, so a performance difference cannot be described as a pure
mainloop-only edit. The official M128N128 path does not match the shadow
header's target predicate. The ordinary paired BF16 and fused FP8 variants
still have a compile-time `CollectiveMainloop` type equality assertion;
the BF16 epilogue remains `Auto` and fusion math/pack remain v6.

The first actual `warp24` nvcc build failed at fixed CuTe
`copy_atom.hpp:206`: `make_tiled_copy_B<SM75_U32x4_LDSM_N,uint8_t>` had only
8 B per B fragment/thread with atom 2×4, while x4 requires 16 B. That
variant remains an immutable failed base in this builder. Two independent
compile candidates address exactly that constraint:

| Variant | Cooperative target atom | Permutation N | Target B smem copy | SF128 reduction | Change against failed `warp24` |
|:---|:---:|---:|:---|---:|:---|
| `warp24x2` | 2×4×1 | 32 | `SM75_U32x2_LDSM_N` (8 B/thread) | 4 warps | B copy atom only; A remains x4 |
| `warp24n64` | 2×4×1 | 64 | `SM75_U32x4_LDSM_N` (16 B/thread) | 4 warps | MMA permutation N only |

The shadow-header predicates restrict both changes to Cooperative
M64N256K128. The official M128N128 and any other tile retain the fixed
CUTLASS builder behavior. Whether either candidate compiles, preserves
the CuTe lane/SF ownership, or passes numeric/Graph gates remains open.

## SF128 scale-store writer revision v3

The fixed row-store visitor first reduces a scale group across four MMA
warps in the 2×4 layout (warp sets `{0,2,4,6}` or `{1,3,5,7}` in the
[host ownership record](ownership-v2/ownership.jsonl)). The validated 4×2
visitor's Cooperative store guard still permits all `warp_idx<4`. Under
the 2×4 ownership sets this permits two warp leaders for each group
(0 and 2, or 1 and 3). `filter_zeros(tC_rSFD)` removes zero-stride modes
**within a thread**; the subsequent `tC_cSFD` boundary comparison does
not itself remove a duplicate writer across warps. This is a strong
source-level duplicate-store risk, but the host `partition_C` probe does
not instantiate the actual epilogue `tiled_copy` and cannot by itself
prove two global addresses coincide.

For the two compile candidates, add `--visitor-revision v3`. The separately
hashed v3 shadow visitor narrows the Cooperative
`SFVecSize==128 && WarpsPerSF==4` write
condition to `warp_idx<2`: warp 0 represents even-row sets, warp 1 odd-row
sets, one quad leader for each local row/scale; the four-warp amax reduction
and quantization math are unchanged. The 4×2 `control42` and recorded
`warp24` v2 behavior remain selectable with the default
`--visitor-revision v2`; v3 rejects those two labels rather than silently
changing their meanings. v3 produces `_v3` namespaces and `.so` names so
the compiler/numerical result cannot be confused with v2. The emitted
v3-v2 visitor diff is only this write guard, and v2 regenerated shadow
mainloop/visitor files were byte-identical to their earlier emission.

This patch still needs actual FP8-byte/FP32-scale-bit and changed-input
Graph tests. Compute Sanitizer `memcheck` can find invalid accesses;
`racecheck` primarily targets shared-memory hazards and cannot prove that
equal-valued duplicate **global** scale stores are absent. A conclusive
writer test would enumerate the actual `tC_cSFD`/`tC_gSFD` epilogue
retile addresses for all participating threads, or use a separate debug
kernel/visitor with an integer atomic counter per scale address and assert
exactly one store at every expected address. Such instrumentation is
diagnostic only and must not be used for performance measurement.

`warp18` is intentionally rejected at argument parsing. Its requested
eight-warps-in-N layout needs a proven SF128 ownership/reduction path;
the pinned visitor only supports 1/2/4 warps per scale. There is no
compile-only placeholder that could be mistaken for a numerically valid
kernel. Compilation of `warp24` may itself fail or its epilogue may map
fragments differently; neither emit-only nor compile success is a numeric
or Graph pass. Root should first check CuTe lane→(M,N) and SF128 ownership,
then run strict full-bit and changed-input Graph gates before timing.

The C ABI keeps the scheduler probe's version-1 function names and pointer
order so its ctypes adapter can load each `.so` independently. The build
chooses a distinct C++ namespace (`pro6000d_mainloop_control42` or
`pro6000d_mainloop_warp24`), variant-specific visitor and pack-kernel type
names, static thread-local host state and hidden ELF visibility with
GNU-unique disabled. Check
`readelf -Ws` after compilation: only ABI-1 C entry points should be
globally visible. This addresses the old scheduler v1 TLS/OBJECT collision;
the new library still needs a combined old-baseline/new-library process
smoke before measuring. C ABI metadata reports the variant, namespace,
target atom layout/permutation and compiled shared-memory, stage and thread
counts; the build manifest records exact source hashes. ptxas register/spill and NCU values must be
collected separately; metadata does not claim measured occupancy.

From the repository root, the local CPU-only source checks used were:

```sh
python3 _dev/docs/evidence/sm120-mainloop/build_mainloop.py \
  --variant control42 \
  --v6-source-dir _dev/docs/evidence/gemm-epilogue/build-gateup-coop-v6 \
  --cutlass-include _dev/docs/evidence/gemm-epilogue/source/cutlass/include \
  --build-dir /tmp/pro6000d-mainloop-emit-control42 --emit-only
python3 _dev/docs/evidence/sm120-mainloop/build_mainloop.py \
  --variant warp24 \
  --v6-source-dir _dev/docs/evidence/gemm-epilogue/build-gateup-coop-v6 \
  --cutlass-include _dev/docs/evidence/gemm-epilogue/source/cutlass/include \
  --build-dir /tmp/pro6000d-mainloop-emit-warp24 --emit-only
python3 _dev/docs/evidence/sm120-mainloop/build_mainloop.py \
  --variant warp24x2 \
  --v6-source-dir _dev/docs/evidence/gemm-epilogue/build-gateup-coop-v6 \
  --cutlass-include _dev/docs/evidence/gemm-epilogue/source/cutlass/include \
  --build-dir /tmp/pro6000d-mainloop-emit-warp24x2 --emit-only
python3 _dev/docs/evidence/sm120-mainloop/build_mainloop.py \
  --variant warp24n64 \
  --v6-source-dir _dev/docs/evidence/gemm-epilogue/build-gateup-coop-v6 \
  --cutlass-include _dev/docs/evidence/gemm-epilogue/source/cutlass/include \
  --build-dir /tmp/pro6000d-mainloop-emit-warp24n64 --emit-only
python3 _dev/docs/evidence/sm120-mainloop/build_mainloop.py \
  --variant warp24x2 --visitor-revision v3 \
  --v6-source-dir _dev/docs/evidence/gemm-epilogue/build-gateup-coop-v6 \
  --cutlass-include _dev/docs/evidence/gemm-epilogue/source/cutlass/include \
  --build-dir /tmp/pro6000d-mainloop-emit-warp24x2-v3 --emit-only
python3 _dev/docs/evidence/sm120-mainloop/build_mainloop.py \
  --variant warp24n64 --visitor-revision v3 \
  --v6-source-dir _dev/docs/evidence/gemm-epilogue/build-gateup-coop-v6 \
  --cutlass-include _dev/docs/evidence/gemm-epilogue/source/cutlass/include \
  --build-dir /tmp/pro6000d-mainloop-emit-warp24n64-v3 --emit-only
```

All four emitted `sources_emitted` with distinct namespace and output paths;
`warp18` is still a deliberate argument error.
The generated mainloop header differs from fixed CUTLASS only at the
target-predicate/atom-layout block, plus one target-only B-copy or PermN
line for the respective new variants. All 2×4 visitors differ from pinned
v6 in one `WarpsPerSF` line. The generated dispatch variants differ only in
their private namespace/type names. `warp24` has the compiler failure above;
`warp24x2` and `warp24n64` have **not** yet been nvcc-compiled or GPU-tested.
When root compiles on the server, omit `--emit-only` and use separate owned
build directories; retain each `build-manifest.json`, compile log and `.so`
SHA. The unprofiled complete-chain experiment specified in
[`probe-plan.md`](probe-plan.md) is the next gate, before any model runtime
change.
