#pragma once
#include <stddef.h>

#if defined(__GNUC__)
#define SM120_MAINLOOP_API __attribute__((visibility("default")))
#else
#define SM120_MAINLOOP_API
#endif

// Dedicated mainloop probe ABI 1, compatible with scheduler ctypes adapters.
// Each variant resides in a separate private library; only these C symbols export.
#ifdef __cplusplus
extern "C" {
#endif
SM120_MAINLOOP_API int sm120_scheduler_abi_version(void);  // exactly 1
SM120_MAINLOOP_API const char* sm120_scheduler_last_error(void);
SM120_MAINLOOP_API const char* sm120_scheduler_metadata(void);  // host-only, JSON, static lifetime

// raster: 0=Heuristic, 1=AlongM, 2=AlongN; swizzle: 1 or 8.
// Shape is fixed N=34816, K=5120, with 257 <= M <= 32768.
// Queries return 0 on failure, at least 1 byte on success, and do no CUDA work.
SM120_MAINLOOP_API size_t sm120_official_bf16_workspace_bytes(int M, int raster, int swizzle);
SM120_MAINLOOP_API size_t sm120_paired_bf16_workspace_bytes(int M, int raster, int swizzle);
SM120_MAINLOOP_API size_t sm120_fused_fp8_workspace_bytes(int M, int raster, int swizzle);
SM120_MAINLOOP_API size_t sm120_fused_fp8_temp_scale_bytes(int M);

// Return 0 for a successful asynchronous launch, 1 on error (see last_error).
// Stream address 0 is the legal default stream. All buffers are caller-owned.
SM120_MAINLOOP_API int sm120_official_bf16(
    void* output, const void* A, const void* B, const void* As, const void* Bs,
    int M, int raster, int swizzle, void* workspace, size_t workspace_bytes,
    unsigned long long stream);
SM120_MAINLOOP_API int sm120_paired_bf16(
    void* output, const void* A, const void* Bpacked,
    const void* As, const void* Bspacked,
    int M, int raster, int swizzle, void* workspace, size_t workspace_bytes,
    unsigned long long stream);
SM120_MAINLOOP_API int sm120_fused_fp8(
    void* out_fp8, void* out_scales, void* temp_fp8, void* temp_scales,
    const void* A, const void* Bpacked, const void* As, const void* Bspacked,
    int M, int raster, int swizzle, void* workspace, size_t workspace_bytes,
    unsigned long long stream);
#ifdef __cplusplus
}
#endif
