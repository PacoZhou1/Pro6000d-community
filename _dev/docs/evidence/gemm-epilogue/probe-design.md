# GEMM → SwiGLU/blockquant probe contract

This probe is developed locally. A syntax check is not a CUDA correctness or
performance result. Only root runs it on the reserved SM120 GPU, without
stopping embedding. Synthetic checks do not establish model quality or cover
every finite FP8 input.

`pipeline_probe.py` owns its buffers and changes no service or installed code.
The fixed baseline is `targeted_block_fp8` variant 0 followed by the already
validated V2 `pro6000d_fp8_silu_quant_launch`. Before any timing it must exactly
reproduce fixed vLLM `cutlass_scaled_mm` + official blockquant, including BF16
GEMM bits, FP8 bytes and FP32 scale bits. Candidate validity requires exact
FP8/scale bits, exact official downstream down-GEMM BF16 bits, and changed-input
CUDA Graph replay. Failure means no timing for that case. All numerical cases
run before any timing or profiler window, so a later failure invalidates the
entire performance phase.

Canonical B `[34816,5120]` and Bs `[272,40]` are row major. The one-time setup
copies each gate128/up128 pair into adjacent rows. The full canonical and packed
weight/scale tensors stay alive for both arms and through all timing. Setup
wall/event duration and extra held memory are reported separately. A uses fixed
vLLM per-token/group quant with column-major scales; output Q `[M,17408]` is row
major and S `[M,136]` has stride `(1,M)`. Each arm owns distinct intermediate,
output and workspace storage. Direct numerical calls test the CUDA default
stream, including a legal stream address of zero. Captures use a side stream.

The candidate C ABI is deliberately **not assumed** while kernel development
is in progress. `--candidate-adapter PATH.py --candidate-library PATH.so` loads
an adapter exposing `create_backend(library: Path, workspace: Tensor, swizzle: int)`. Its
returned object must expose:

- `metadata: dict`: exact symbol, ABI version, source/build hashes, algorithm
  and intermediate-storage claims, including `eliminates_bf16_intermediate`.
- `supports(m: int) -> bool`: unsupported shapes use the explicit baseline
  fallback, labelled in each record; do not count fallback as fusion speedup.
- `prepare(m: int) -> None` and `owned_buffers: list[Tensor]`: allocate any
  temporary outside capture/timing and expose it for nonaliasing checks.
- `launch(q, scales, a, a_scales, packed_b, packed_bs) -> None`: writes provided
  nonaliasing tensors, passes current CUDA stream without rejecting zero,
  owns no mutable shared workspace, and throws on every nonzero C status.

The explicitly agreed v2 ABI is implemented in `candidate_adapter.py`:
`gateup_quant_fp8(out, scales, temp, temp_scales, A, packedB, As, packedBs, M, swizzle,
workspace, workspace_bytes, stream)`. It removes BF16 materialization but
retains paired FP8 `[M,34816]` and native-layout scale intermediate storage,
plus a second pack kernel. `gateup_quant_abi_version()` must return 2;
`gateup_quant_temp_scale_bytes(M)` provides the native scale-buffer size.
Its timings and NCU window must include both launches. This ABI agreement is
not evidence of compilation, correctness or speedup.

Normal random input shapes default to 16160, 2741, 2786, 256, 257, 1, 5.
Additional M257 zero/tiny/large/NaN/Inf inputs are numerical cases only. These
inputs exercise selected exceptional cases; they are not an exhaustive proof.
Every numerical case verifies the packed baseline GEMM after inverse N-group
permutation and verifies V2 against official quant. Candidate comparisons use
all output elements, never only sample coordinates. Diagnostics retain first
mismatching logical coordinates and maximum finite error without replacing
the exact gate with a tolerance.

Warm and cold each have 31 paired samples. Each pair independently shuffles
arm order with a recorded seed. Both arms replay pre-captured **whole pipeline**
graphs, and events bracket the complete replay. Cold fills an owned >2×L2
buffer before the events; this is an eviction approximation. No allocation,
pack, or flush is timed. Warm means repeated operands, not all weights fit L2.
Both median per-arm times and paired baseline/candidate ratios are reported.

`--ncu --m ONE_M` runs numerical gates first and then opens two profiler
windows, exactly one full baseline pipeline and one candidate pipeline each.
No timing loop runs. Use NCU `--profile-from-start off` and NVTX labels to
attribute both baseline kernels versus candidate kernel(s). Select at least
`dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,
lts__t_bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed` after checking
these metric names on the pinned NCU installation. Sum read/write bytes across
all kernels in an arm. Report candidate/baseline read, write and total ratios,
with denominator = one complete GEMM→quant pipeline and measured kernel count.
Logical BF16 bytes removed are not measured DRAM bytes. NCU replay/cache
policy and any missing metrics must be recorded by the launcher.
