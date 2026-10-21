// Experimental SM120 FP8 SwiGLU + per-128-group quantization kernel.
// V2: adjacent CTAs traverse 4-group blocks within one row before the next
// row. The arithmetic and per-warp group mapping are identical to V1.
// Baseline: vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6
// csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu.
// This standalone library does not register or replace any vLLM op by itself.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

thread_local std::string last_error;
constexpr int kGroup = 128;
constexpr int kWarpsPerCta = 4;
constexpr int kGroupBlocksPerRow = 17408 / (kGroup * kWarpsPerCta);
constexpr float kFp8Max = 448.0f;
constexpr float kMinScale = 1.0f / (kFp8Max * 512.0f);

// Each warp owns one complete 128-column group. Four values per lane retain
// the upstream FP32 SiLU and product until scale computation and conversion.
// The baseline maps one whole CTA to one group and uses nine CTA barriers.
__global__ void silu_quant_warp_groups(
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    int hidden_size,
    int scale_stride_m,
    int scale_stride_group) {
  const int row = blockIdx.x / kGroupBlocksPerRow;
  const int group_block = blockIdx.x % kGroupBlocksPerRow;
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int group = group_block * kWarpsPerCta + warp;
  if (group >= hidden_size / kGroup) return;

  const int64_t row_base = static_cast<int64_t>(row) * (2 * hidden_size);
  const int group_base = group * kGroup;
  float values[4];
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = group_base + lane + i * warpSize;
    const float gate = __bfloat162float(input[row_base + col]);
    const float up = __bfloat162float(input[row_base + hidden_size + col]);
    const float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    const float silu_gate = gate * sigmoid_gate;
    const float result = silu_gate * up;
    values[i] = result;
    max_abs = (i == 0) ? fabsf(result) : fmaxf(max_abs, fabsf(result));
  }

  // The maximum is order-independent for finite inputs. This also uses the
  // upstream fmaxf semantics for exceptional inputs; micro_probe checks the
  // resulting FP8 bytes and scale bits before any performance numbers count.
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, offset));
  }
  const float group_max = __shfl_sync(0xffffffffu, max_abs, 0);
  const float group_scale = fmaxf(group_max / kFp8Max, kMinScale);
  if (lane == 0) {
    scales[static_cast<int64_t>(row) * scale_stride_m +
           static_cast<int64_t>(group) * scale_stride_group] = group_scale;
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float scaled = values[i] / group_scale;
    const float clamped = fmaxf(-kFp8Max, fminf(scaled, kFp8Max));
    output[static_cast<int64_t>(row) * hidden_size + group_base + lane +
           i * warpSize] =
        __nv_fp8_e4m3(clamped);
  }
}

}  // namespace

extern "C" int pro6000d_fp8_silu_quant_launch(
    void* output, void* scales, const void* input, int m, int hidden_size,
    int scale_stride_m, int scale_stride_group, uint64_t stream_address) {
  last_error.clear();
  // CUDA's legacy default stream is represented by address 0 and is valid.
  if (!output || !scales || !input || m < 1 || hidden_size != 17408 ||
      scale_stride_m != 1 || scale_stride_group < m) {
    last_error = "unsupported pointer, shape, or scale stride";
    return 1;
  }
  const uint64_t ctas = static_cast<uint64_t>(m) * kGroupBlocksPerRow;
  if (ctas > 2147483647ull) {
    last_error = "CTA grid exceeds CUDA x dimension";
    return 1;
  }
  const dim3 grid(static_cast<unsigned int>(ctas));
  const auto stream = reinterpret_cast<cudaStream_t>(stream_address);
  silu_quant_warp_groups<<<grid, kWarpsPerCta * 32, 0, stream>>>(
      static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
      static_cast<const __nv_bfloat16*>(input), hidden_size, scale_stride_m,
      scale_stride_group);
  const cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    last_error = cudaGetErrorString(status);
    return 2;
  }
  return 0;
}

extern "C" const char* pro6000d_fp8_silu_quant_last_error() {
  return last_error.c_str();
}
