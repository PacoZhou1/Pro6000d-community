#include "mainloop_abi.h"
#include "standalone_dispatch.cuh"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifndef PRO6000D_MAINLOOP_NAMESPACE
#error "build_mainloop.py must define a private namespace"
#endif
#ifndef PRO6000D_MAINLOOP_VARIANT
#error "build_mainloop.py must define a variant string"
#endif
#ifndef PRO6000D_MAINLOOP_ATOM_M
#error "build_mainloop.py must define the atom layout"
#endif
#ifndef PRO6000D_MAINLOOP_ATOM_N
#error "build_mainloop.py must define the atom layout"
#endif
#ifndef PRO6000D_MAINLOOP_PERM_N
#error "build_mainloop.py must define the MMA permutation N"
#endif
#ifndef PRO6000D_MAINLOOP_PACK_KERNEL
#error "build_mainloop.py must define the private pack kernel name"
#endif
#define PRO6000D_STRINGIFY_INNER(x) #x
#define PRO6000D_STRINGIFY(x) PRO6000D_STRINGIFY_INNER(x)

namespace {
thread_local char error_message[512] = {0};
using Official = typename PRO6000D_MAINLOOP_NAMESPACE::sm120_blockwise_fp8_config_default<cutlass::bfloat16_t>::Gemm;
using Paired = typename PRO6000D_MAINLOOP_NAMESPACE::sm120_scheduler_config_paired_bf16::Gemm;
using Fused = typename PRO6000D_MAINLOOP_NAMESPACE::sm120_blockwise_fp8_config_paired_quant<cutlass::float_e4m3_t>::Gemm;
static_assert(std::is_same_v<typename Paired::CollectiveMainloop,
                             typename Fused::CollectiveMainloop>,
              "BF16 diagnostic must retain the exact fused mainloop type");

void save_error(const char* message) {
  std::strncpy(error_message, message, sizeof(error_message) - 1);
  error_message[sizeof(error_message) - 1] = 0;
}
void check_shape(int m, int raster, int swizzle) {
  if (m <= 256 || m > 32768 || raster < 0 || raster > 2 ||
      (swizzle != 1 && swizzle != 8))
    throw std::runtime_error("unsupported M, raster, or swizzle");
}

template<class Gemm>
void call_gemm(void* output, const void* a, const void* b, const void* as,
               const void* bs, int m, int raster, int swizzle,
               void* native_scales, int device) {
  PRO6000D_MAINLOOP_NAMESPACE::TensorView ov{output, m, 34816, device};
  PRO6000D_MAINLOOP_NAMESPACE::TensorView av{const_cast<void*>(a), m, 5120, device};
  PRO6000D_MAINLOOP_NAMESPACE::TensorView bv{const_cast<void*>(b), 5120, 34816, device};
  PRO6000D_MAINLOOP_NAMESPACE::TensorView asv{const_cast<void*>(as), m, 40, device};
  PRO6000D_MAINLOOP_NAMESPACE::TensorView bsv{const_cast<void*>(bs), 40, 272, device};
  PRO6000D_MAINLOOP_NAMESPACE::cutlass_gemm_caller_blockwise<Gemm>(
      ov, av, bv, asv, bsv, swizzle, native_scales, raster);
}

template<class Gemm>
size_t query_workspace(int m, int raster, int swizzle) {
  error_message[0] = 0;
  try {
    check_shape(m, raster, swizzle);
    constexpr uintptr_t fake = 0x10000;
    void* p = reinterpret_cast<void*>(fake);
    PRO6000D_MAINLOOP_NAMESPACE::c3x::query_only = true;
    PRO6000D_MAINLOOP_NAMESPACE::c3x::required_workspace_bytes = 0;
    call_gemm<Gemm>(p, p, p, p, p, m, raster, swizzle, p, 0);
    const size_t needed = PRO6000D_MAINLOOP_NAMESPACE::c3x::required_workspace_bytes;
    PRO6000D_MAINLOOP_NAMESPACE::c3x::query_only = false;
    return needed == 0 ? 1 : needed;
  } catch (const std::exception& ex) {
    PRO6000D_MAINLOOP_NAMESPACE::c3x::query_only = false;
    save_error(ex.what());
    return 0;
  }
}

template<class Gemm>
int launch_bf16(void* output, const void* a, const void* b, const void* as,
                const void* bs, int m, int raster, int swizzle,
                void* workspace, size_t workspace_bytes,
                unsigned long long stream) {
  error_message[0] = 0;
  try {
    check_shape(m, raster, swizzle);
    if (!output || !a || !b || !as || !bs || !workspace)
      throw std::runtime_error("null buffer");
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    PRO6000D_MAINLOOP_NAMESPACE::c3x::query_only = false;
    PRO6000D_MAINLOOP_NAMESPACE::c3x::workspace = workspace;
    PRO6000D_MAINLOOP_NAMESPACE::c3x::workspace_bytes = workspace_bytes;
    PRO6000D_MAINLOOP_NAMESPACE::c3x::stream = reinterpret_cast<cudaStream_t>(stream);
    call_gemm<Gemm>(output, a, b, as, bs, m, raster, swizzle, nullptr, device);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return 0;
  } catch (const std::exception& ex) {
    save_error(ex.what());
    return 1;
  }
}

template<class Gemm>
void append_resources(std::ostringstream& out, const char* name) {
  using K = typename Gemm::GemmKernel;
  out << '"' << name << "\":{\"mainloop_stages\":"
      << K::DispatchPolicy::Stages
      << ",\"max_threads_per_block\":" << K::MaxThreadsPerBlock
      << ",\"mma_threads\":" << K::NumMMAThreads
      << ",\"kernel_shared_storage_bytes\":" << sizeof(typename K::SharedStorage)
      << ",\"mainloop_shared_storage_bytes\":" << sizeof(typename Gemm::CollectiveMainloop::SharedStorage)
      << ",\"epilogue_shared_storage_bytes\":" << sizeof(typename Gemm::CollectiveEpilogue::SharedStorage)
      << '}';
}
}  // namespace

// This pack body is extracted verbatim from the pinned v6 driver at build time.
#include "mainloop_pack.cuh"

extern "C" int sm120_scheduler_abi_version() { return 1; }
extern "C" const char* sm120_scheduler_last_error() { return error_message; }
extern "C" const char* sm120_scheduler_metadata() {
  static const std::string value = [] {
    std::ostringstream out;
    out << "{\"abi_version\":1,\"scope\":\"isolated probe only\","
        << "\"cuda_arch\":\"sm_120a\",\"N\":34816,\"K\":5120,"
        << "\"min_M\":257,\"max_M\":32768,"
        << "\"raster\":{\"0\":\"Heuristic\",\"1\":\"AlongM\",\"2\":\"AlongN\"},"
        << "\"swizzle\":[1,8],\"same_m64_mainloop_type\":true,"
        << "\"mainloop_variant\":\"" << PRO6000D_MAINLOOP_VARIANT << "\","
        << "\"private_namespace\":\"" << PRO6000D_STRINGIFY(PRO6000D_MAINLOOP_NAMESPACE) << "\","
        << "\"m64_mma_atom_layout_mnk\":[" << PRO6000D_MAINLOOP_ATOM_M << ","
        << PRO6000D_MAINLOOP_ATOM_N << ",1],"
        << "\"m64_mma_permutation_n\":" << PRO6000D_MAINLOOP_PERM_N << ","
        << "\"mainloop_patch_scope\":\"Cooperative M64N256K128 only\","
        << "\"paired_bf16_epilogue\":\"Auto\","
        << "\"fused_epilogue_tile\":\"64x256\",\"resources\":{";
    append_resources<Official>(out, "official_bf16"); out << ',';
    append_resources<Paired>(out, "paired_bf16"); out << ',';
    append_resources<Fused>(out, "fused_fp8"); out << "}}";
    return out.str();
  }();
  return value.c_str();
}

extern "C" size_t sm120_official_bf16_workspace_bytes(int m, int raster, int swizzle) {
  return query_workspace<Official>(m, raster, swizzle);
}
extern "C" size_t sm120_paired_bf16_workspace_bytes(int m, int raster, int swizzle) {
  return query_workspace<Paired>(m, raster, swizzle);
}
extern "C" size_t sm120_fused_fp8_workspace_bytes(int m, int raster, int swizzle) {
  return query_workspace<Fused>(m, raster, swizzle);
}
extern "C" size_t sm120_fused_fp8_temp_scale_bytes(int m) {
  error_message[0] = 0;
  if (m <= 256 || m > 32768) { save_error("unsupported M"); return 0; }
  using S = cutlass::detail::Sm1xxBlockScaledOutputConfig<128>;
  auto layout = S::tile_atom_to_shape_SFD(cute::make_shape(m, 34816, 5120, 1));
  return size_t(cute::cosize(layout)) * sizeof(float);
}

extern "C" int sm120_official_bf16(
    void* output, const void* a, const void* b, const void* as, const void* bs,
    int m, int raster, int swizzle, void* workspace, size_t workspace_bytes,
    unsigned long long stream) {
  return launch_bf16<Official>(output, a, b, as, bs, m, raster, swizzle,
                               workspace, workspace_bytes, stream);
}
extern "C" int sm120_paired_bf16(
    void* output, const void* a, const void* b, const void* as, const void* bs,
    int m, int raster, int swizzle, void* workspace, size_t workspace_bytes,
    unsigned long long stream) {
  return launch_bf16<Paired>(output, a, b, as, bs, m, raster, swizzle,
                             workspace, workspace_bytes, stream);
}
extern "C" int sm120_fused_fp8(
    void* out_fp8, void* out_scales, void* temp_fp8, void* temp_scales,
    const void* a, const void* b, const void* as, const void* bs,
    int m, int raster, int swizzle, void* workspace, size_t workspace_bytes,
    unsigned long long stream) {
  error_message[0] = 0;
  try {
    check_shape(m, raster, swizzle);
    if (!out_fp8 || !out_scales || !temp_fp8 || !temp_scales ||
        !a || !b || !as || !bs || !workspace)
      throw std::runtime_error("null buffer");
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    PRO6000D_MAINLOOP_NAMESPACE::c3x::query_only = false;
    PRO6000D_MAINLOOP_NAMESPACE::c3x::workspace = workspace;
    PRO6000D_MAINLOOP_NAMESPACE::c3x::workspace_bytes = workspace_bytes;
    PRO6000D_MAINLOOP_NAMESPACE::c3x::stream = reinterpret_cast<cudaStream_t>(stream);
    call_gemm<Fused>(temp_fp8, a, b, as, bs, m, raster, swizzle, temp_scales, device);
    const int64_t total = int64_t(m) * 17408;
    PRO6000D_MAINLOOP_PACK_KERNEL<<<int((total + 255) / 256), 256, 0, PRO6000D_MAINLOOP_NAMESPACE::c3x::stream>>>(
        static_cast<cutlass::float_e4m3_t*>(out_fp8), static_cast<float*>(out_scales),
        static_cast<const cutlass::float_e4m3_t*>(temp_fp8),
        static_cast<const float*>(temp_scales), m);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return 0;
  } catch (const std::exception& ex) {
    save_error(ex.what());
    return 1;
  }
}
