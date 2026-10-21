#include "standalone_dispatch.cuh"
#include <cstring>
#include <exception>

namespace { thread_local char error_message[512] = {0}; }
extern "C" const char* n256_last_error() { return error_message; }

// Packed B is [N,K] contiguous, with adjacent gate/up 128-row groups.
// This stage writes BF16 [M,N] to test the paired CTA/mainloop only.
extern "C" int n256_block_fp8(
    void* output, const void* a, const void* packed_b, const void* as,
    const void* packed_bs, int m, int n, int k, int swizzle,
    void* workspace, size_t workspace_bytes, unsigned long long stream) {
  error_message[0] = 0;
  try {
    if (!output || !a || !packed_b || !as || !packed_bs || !workspace ||
        m <= 256 || n != 34816 || k != 5120 || (swizzle != 1 && swizzle != 8))
      throw std::runtime_error("unsupported argument");
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("cudaGetDevice failed");
    vllm::c3x::workspace = workspace;
    vllm::c3x::workspace_bytes = workspace_bytes;
    vllm::c3x::stream = reinterpret_cast<cudaStream_t>(stream);
    vllm::TensorView o{output, m, n, device};
    vllm::TensorView av{const_cast<void*>(a), m, k, device};
    vllm::TensorView bv{const_cast<void*>(packed_b), k, n, device};
    vllm::TensorView asv{const_cast<void*>(as), m, k / 128, device};
    vllm::TensorView bsv{const_cast<void*>(packed_bs), k / 128, n / 128, device};
    using Gemm = typename vllm::sm120_blockwise_fp8_config_paired_n256<cutlass::bfloat16_t>::Gemm;
    vllm::cutlass_gemm_caller_blockwise<Gemm>(o, av, bv, asv, bsv, swizzle);
    return 0;
  } catch (const std::exception& ex) {
    std::strncpy(error_message, ex.what(), sizeof(error_message) - 1);
    error_message[sizeof(error_message) - 1] = 0;
    return 1;
  }
}
