# SM120 FP8 GEMM recovery: pinned sources and test boundary

This is a random-input, single-operator experiment for `A[M,5120] × W[34816,5120]^T → BF16[M,34816]`, with `M=2741` and `16160`. It does **not** load or convert Qwen weights and cannot establish model quality. Both operator timings and numeric references start with already quantized FP8 inputs. Activation quantization, weight conversion, scale packing, and any PR-derived weight reordering are outside the GPU GEMM interval and must be disclosed as deployment costs.

| Arm | Scaling semantics | Implementation | Relationship to fixed model |
| --- | --- | --- | --- |
| `cutlass_block128` | A FP32 `[M,40]`, physical stride `(1,M)`; W FP32 `[272,40]`; each K128/N128 block | Fixed vLLM 0.31 `cutlass_scaled_mm` | Same scale scheme and kernel as gate/up baseline; random tensors, not model weights |
| `cutlass_mxfp8` | E4M3 A/W with UE8M0 power-of-two scales per K32 | CUTLASS 4.5 SM120 block-scaled MMA, derived from example 79c, BF16 output | Different scaling/weight representation; only an alternative operator ceiling |
| `cublas_rowwise` | A FP32 per row, W FP32 per output channel | Fixed torch 2.13 `torch._scaled_mm` | Public rowwise reference, not FlashInfer PR #5521 |
| `pr5521_derived_rowwise` | Same rowwise scaling, gate/up weight rows reordered in groups of 8, BF16 output in canonical order | Locally derived FP8 GEMM core from pinned FlashInfer PR #5521; included only after a compiled and numerically validated library exists | Research derivative; original PR is a fused MiniMax-H3 operator, not directly callable for this GEMM |

Each arm has **its own** nontrivial random scales and FP32 sampled reference. The sampled gate covers first/middle/tail rows and three output-column regions, requiring finite full output, normalized RMSE below 0.005 and cosine above 0.9999. In addition, the PR-derived rowwise output is compared against the same-input cuBLAS rowwise BF16 output for **every element** in row chunks (NRMSE < 0.002), and MXFP8 is compared for every element against an independent reference that applies K32 power-of-two scales to A/W, casts to BF16, and performs BF16 matmul (NRMSE < 0.005). Those thresholds allow different BF16 accumulation order; the JSONL also records all BF16 bit mismatches and max absolute error. All output rows, including the tail tile, are checked. Changed-input CUDA Graph replay must both change output and pass a new sampled numeric reference; the PR-derived arm also repeats full BF16 comparison. These tests check a synthetic operator, not equality between scaling schemes. The BF16 cast belongs after the scale math. The BF16 output covers all `M×34816` logical values; tail tiles can execute more physical rows. For the 128-row CUTLASS and derived PR tiles, M2741 covers 2816 tile rows (53 valid in last tile) and M16160 covers 16256 (32 valid in last tile); this is tile coverage, not a changed logical M. Confirm each compiled arm's grid from the actual binary before interpreting results. GPU Event graph replay excludes host launch/TMA descriptor setup; if the PR-derived ABI builds a descriptor per call, report that cost separately. A >2×L2 owned-buffer write is only a cache disturbance, not guaranteed cold L2. Best-effort NVML SM clock/power observations accompany each sample with poll count and exact-window status; missing NVML data is explicitly null and does not invalidate CUDA Event timing.

## Upstream pin and scope

- [CUTLASS SM120 example 79c at v4.5.0](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu) uses MXFP8 A and MXFP6 B; the local C ABI substitutes MXFP8 E4M3 B and keeps SM120 block-scaled MMA. The fixed-image headers are CUTLASS 4.5.0, pinned by SHA in `build_mxfp8.py`, including `tools/util/include/cutlass/util/packed_stride.hpp` (SHA256 `b43a26572905cebf99e0a9842903efd2232421f7d1c1a686711ddd636708fcbb`). The first CPU compile failed because that header was not included; the updated adapter adds it, with a new source hash and separate build required. Example snapshot: `source/cutlass_79c_v450.cu`, SHA256 `5de04bc0bf718b558b69b7d1060af3f57a71cd5980fe690370ba94d62a9b998d`.
- [FlashInfer PR #5521](https://github.com/flashinfer-ai/flashinfer/pull/5521), last pinned PR commit `860ed84451fa86c468969ab1ab37d2255937884f`, targets fused RMSNorm/indexed AdaLN/activation quantization/FC1/SwiGLU for MiniMax-H3 with K5376 and N28672. Source snapshot: `source/flashinfer_pr5521_fc1.cu`, SHA256 `e9b3c49e10b63ecd439737738e7ea0cbd713dcab9fddcc89ef1a025b221cba63`. Its published fused performance is not an isolated Qwen GEMM result. The derivative must change K/N mapping, remove fusion, reorder weights before timing, and prove canonical BF16 output.
- [FlashInfer PR #5524](https://github.com/flashinfer-ai/flashinfer/pull/5524) targets a separate specialized attention output projection and is not an interchangeable rowwise gate/up GEMM.
- At the same pinned FlashInfer commit, public [`gemm_base.py`](https://github.com/flashinfer-ai/flashinfer/blob/860ed84451fa86c468969ab1ab37d2255937884f/flashinfer/gemm/gemm_base.py) routes SM120 `bmm_fp8(backend="cutlass")` through groupwise scales `1×128` on A and `128×128` on B; it is not a generic per-token/per-channel rowwise equivalent. Snapshot `source/flashinfer_gemm_base_860ed.py`, SHA256 `d977aa4991e7b4bf5b8904436d7aa4cff5b5b66a9a394f2a6003b10125166fa8`.
- Prior fixed-stack operator timings are in [A2 ceiling summary](../a-gemm-ceiling/results/summary.md). Its torch MX/rowwise arms used different scaling semantics and reported speed only, so none establishes a model-preserving win.

## Build and validation order

In a CPU-only owned container using the fixed vLLM 0.31 / torch 2.13 cu130 image and extracted CUTLASS 4.5.0 tree:

```sh
python3 /work/b-mxfp8-recovery/build_mxfp8.py \
  --cutlass-root /work/cutlass \
  --output /work/build-mxfp8
```

`/work/cutlass` must contain `include/` and `tools/util/include/`; the builder rejects a different version/header hash. It emits compile log, exact command, source hashes, and library SHA. No GPU is needed for compilation. Then, after the parent has exclusive GPU scheduling, run one M at a time, keeping full JSONL even on failure:

```sh
python3 /work/b-mxfp8-recovery/bench_recovery.py \
  --mx-library /work/build-mxfp8/librecovery_mxfp8.so \
  --pr-library /work/build-pr/libpr5521_derived.so \
  --m 2741 --pairs 31 --output /work/results/recovery-m2741.jsonl
```

Omit `--pr-library` until its separate build passes. Repeat at `--m 16160`; do not substitute a cuBLAS call for the PR-derived arm. Strict progression: compile → shape/layout and launch check → sampled and full-matrix numeric gates → changed-input graph replay with new reference → 31 randomized paired warm and disturbed observations. Every sample records randomized arm order, paired index, CUDA Event milliseconds, and best-effort NVML clock/power. Per-arm `paired_gain` uses the same pair's fixed CUTLASS interval divided by that arm's interval; >1 favors the arm. The parent launcher must record embedding/service isolation outside this script. Report timing and memory/packing costs separately. Neither an upstream PR speed claim nor a single-operator result authorizes model deployment.
