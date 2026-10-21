#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace vllm {
struct TensorView {
  void* ptr;
  int64_t d0, d1;
  int device_id;
  int64_t size(int i) const { return i == 0 ? d0 : d1; }
  void* data_ptr() const { return ptr; }
  int device() const { return device_id; }
};

// Exact architecture guard shape from fixed vLLM cutlass_extensions/common.hpp.
template <typename Kernel>
struct enable_sm120_family : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
#if (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
    Kernel::operator()(std::forward<Args>(args)...);
#else
    asm("trap;");
#endif
#endif
  }
};

namespace c3x {
// The caller is adapted from fixed vLLM cutlass_gemm_caller.cuh. Workspace
// comes from the Python-owned CUDA tensor so graph capture allocates nothing.
inline thread_local void* workspace = nullptr;
inline thread_local size_t workspace_bytes = 0;
inline thread_local cudaStream_t stream = nullptr;
inline thread_local bool query_only = false;
inline thread_local size_t required_workspace_bytes = 0;

template <typename GemmKernel>
void cutlass_gemm_caller(
    int device, cute::Shape<int, int, int, int> prob_shape,
    typename GemmKernel::MainloopArguments mainloop_args,
    typename GemmKernel::EpilogueArguments epilogue_args,
    typename GemmKernel::TileSchedulerArguments scheduler = {}) {
  (void)device;
  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape, mainloop_args, epilogue_args,
                                      hw_info, scheduler};
  using Op = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  Op op;
  required_workspace_bytes = op.get_workspace_size(args);
  if (query_only) return;
  auto status = op.can_implement(args);
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(status));
  if (op.get_workspace_size(args) > workspace_bytes)
    throw std::runtime_error("workspace too small");
  status = op.run(args, workspace, stream);
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(status));
}
} // namespace c3x
} // namespace vllm
