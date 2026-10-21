#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define PR5521_ABI __attribute__((visibility("default")))
#else
#define PR5521_ABI
#endif

PR5521_ABI int recovery_pr5521_abi_version(void);
PR5521_ABI const char* recovery_pr5521_last_error(void);

/* Experimental derivative of FlashInfer PR #5521 FC1, not its public API.
 * A: device FP8 E4M3 row-major [M,5120]; scale_a: device float32 [M].
 * B: device FP8 E4M3 row-major [34816,5120], preinterleaved by
 *   p = 16*(c/8) + c%8 + 8*is_up, c in [0,17408).
 * scale_b: device float32 [34816], in exactly the same preinterleaved order.
 * out: caller-owned device BF16 row-major [M,34816], canonical gate then up.
 * M is exactly 2741 or 16160; N=34816, K=5120. All pointers are device
 * pointers, 16-byte aligned. stream is a cudaStream_t bit pattern. Returns
 * zero on host setup and launch success; asynchronous kernel errors require
 * caller's stream synchronization. No allocation or stream synchronization.
 */
PR5521_ABI int recovery_pr5521_launch(void* out, const void* a, const void* b,
    const float* scale_a, const float* scale_b,
    int m, int n, int k, unsigned long long stream);

#ifdef __cplusplus
}
#endif
