#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#if defined(__GNUC__)
#define RECOVERY_ABI __attribute__((visibility("default")))
#else
#define RECOVERY_ABI
#endif
RECOVERY_ABI int recovery_mxfp8_abi_version(void);
RECOVERY_ABI const char* recovery_mxfp8_last_error(void);
RECOVERY_ABI size_t recovery_mxfp8_scale_a_bytes(int m, int n, int k);
RECOVERY_ABI size_t recovery_mxfp8_scale_b_bytes(int m, int n, int k);
RECOVERY_ABI size_t recovery_mxfp8_workspace_bytes(int m, int n, int k);
/* logical scales are e8m0 exponent bytes [M,K/32] and [N,K/32].
   out buffers must meet *_scale_*_bytes; all pointers are HOST pointers. */
RECOVERY_ABI int recovery_mxfp8_pack_scales(const unsigned char* logical_a,
    const unsigned char* logical_b, unsigned char* packed_a,
    unsigned char* packed_b, int m, int n, int k,
    size_t packed_a_bytes, size_t packed_b_bytes);
/* A is device FP8 row-major [M,K], B is device FP8 [N,K] row-major
   (= column-major GEMM B [K,N]), output device BF16 row-major [M,N].
   Scale buffers are device e8m0 in the CUTLASS 4.5.0 SM120 native layout.
   Stream is the cudaStream_t bit pattern. No allocation or synchronization. */
RECOVERY_ABI int recovery_mxfp8_launch(void* out, const void* a, const void* b,
    const void* scale_a, const void* scale_b, int m, int n, int k,
    void* workspace, size_t workspace_bytes, unsigned long long stream);
#ifdef __cplusplus
}
#endif
