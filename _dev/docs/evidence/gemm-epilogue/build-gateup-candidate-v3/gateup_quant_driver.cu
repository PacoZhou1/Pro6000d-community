#include "standalone_dispatch.cuh"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <exception>

namespace { thread_local char error_message[512] = {0}; }

extern "C" int gateup_quant_abi_version() { return 2; }
extern "C" const char* gateup_quant_last_error() { return error_message; }

extern "C" size_t gateup_quant_temp_scale_bytes(int m) {
  if (m <= 256 || m > 32768) return 0;
  using S = cutlass::detail::Sm1xxBlockScaledOutputConfig<128>;
  auto layout = S::tile_atom_to_shape_SFD(cute::make_shape(m, 34816, 5120, 1));
  return size_t(cute::cosize(layout)) * sizeof(float);
}

// Pure host query: forms the same fixed-shape CUTLASS arguments as launch,
// but c3x::query_only returns before can_implement/op.run or any CUDA API.
// A zero return means unsupported inputs or a query failure.
extern "C" size_t gateup_quant_workspace_bytes(int m, int swizzle) {
  if (m <= 256 || m > 32768 || (swizzle != 1 && swizzle != 8)) return 0;
  constexpr uintptr_t fake = 0x10000;
  vllm::TensorView out{reinterpret_cast<void*>(fake), m, 34816, 0};
  vllm::TensorView a{reinterpret_cast<void*>(fake), m, 5120, 0};
  vllm::TensorView b{reinterpret_cast<void*>(fake), 5120, 34816, 0};
  vllm::TensorView as{reinterpret_cast<void*>(fake), m, 40, 0};
  vllm::TensorView bs{reinterpret_cast<void*>(fake), 40, 272, 0};
  using Gemm = typename vllm::sm120_blockwise_fp8_config_paired_quant<cutlass::float_e4m3_t>::Gemm;
  vllm::c3x::query_only = true;
  vllm::c3x::required_workspace_bytes = 0;
  try {
    vllm::cutlass_gemm_caller_blockwise<Gemm>(
        out, a, b, as, bs, swizzle, reinterpret_cast<void*>(fake));
    const size_t needed = vllm::c3x::required_workspace_bytes;
    vllm::c3x::query_only = false;
    return needed == 0 ? 1 : needed;
  } catch (...) {
    vllm::c3x::query_only = false;
    return 0;
  }
}

__global__ void pack_gateup_fp8(
    cutlass::float_e4m3_t* __restrict__ out,
    float* __restrict__ out_scale,
    const cutlass::float_e4m3_t* __restrict__ paired,
    const float* __restrict__ native_scale, int m) {
  constexpr int H = 17408;
  constexpr int PackedN = 34816;
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= int64_t(m) * H) return;
  const int row = int(idx / H);
  const int col = int(idx % H);
  const int group = col / 128;
  const int within = col % 128;
  out[idx] = paired[int64_t(row) * PackedN + group * 256 + within];
  if (within == 0) {
    using S = cutlass::detail::Sm1xxBlockScaledOutputConfig<128>;
    auto layout = S::tile_atom_to_shape_SFD(cute::make_shape(m, PackedN, 5120, 1));
    const auto native_offset = layout(cute::make_coord(row, group * 256, 0));
    out_scale[int64_t(group) * m + row] = native_scale[native_offset];
  }
}

extern "C" int gateup_quant_fp8(
    void* out_fp8, void* out_scales, void* temp_fp8, void* temp_scales,
    const void* A, const void* Bpacked, const void* As, const void* Bspacked,
    int M, int swizzle, void* workspace, size_t workspace_bytes,
    unsigned long long stream_address) {
  error_message[0] = 0;
  try {
    if (!out_fp8 || !out_scales || !temp_fp8 || !temp_scales ||
        !A || !Bpacked || !As || !Bspacked || !workspace ||
        M <= 256 || M > 32768 || (swizzle != 1 && swizzle != 8))
      throw std::runtime_error("unsupported pointer, shape, or swizzle");
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("cudaGetDevice failed");
    vllm::c3x::workspace = workspace;
    vllm::c3x::workspace_bytes = workspace_bytes;
    vllm::c3x::stream = reinterpret_cast<cudaStream_t>(stream_address);
    vllm::TensorView out{temp_fp8, M, 34816, device};
    vllm::TensorView a{const_cast<void*>(A), M, 5120, device};
    vllm::TensorView b{const_cast<void*>(Bpacked), 5120, 34816, device};
    vllm::TensorView as{const_cast<void*>(As), M, 40, device};
    vllm::TensorView bs{const_cast<void*>(Bspacked), 40, 272, device};
    using Gemm = typename vllm::sm120_blockwise_fp8_config_paired_quant<cutlass::float_e4m3_t>::Gemm;
    vllm::cutlass_gemm_caller_blockwise<Gemm>(out, a, b, as, bs, swizzle, temp_scales);
    constexpr int H = 17408;
    const int64_t total = int64_t(M) * H;
    const int blocks = int((total + 255) / 256);
    pack_gateup_fp8<<<blocks, 256, 0, vllm::c3x::stream>>>(
        static_cast<cutlass::float_e4m3_t*>(out_fp8),
        static_cast<float*>(out_scales),
        static_cast<const cutlass::float_e4m3_t*>(temp_fp8),
        static_cast<const float*>(temp_scales), M);
    auto err = cudaPeekAtLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return 0;
  } catch (const std::exception& ex) {
    std::strncpy(error_message, ex.what(), sizeof(error_message) - 1);
    error_message[sizeof(error_message) - 1] = 0;
    return 1;
  }
}
