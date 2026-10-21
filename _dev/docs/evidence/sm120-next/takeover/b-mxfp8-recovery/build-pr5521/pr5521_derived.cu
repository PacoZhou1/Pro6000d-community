/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// clang-format off
// Experimental derivative of FlashInfer PR #5521 FC1 FP8 kernel.
// GEMM only: BF16 canonical gate/up output, K=5120, N=34816.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <math_constants.h>
static_assert(sizeof(CUtensorMap) == 128, "CUDA tensor-map ABI");
static_assert(alignof(CUtensorMap) >= 64, "CUDA tensor-map alignment");
namespace pro6000d_pr5521_fp8_derived {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_FC1_INF CUDART_INF_F
#define NUM_FULL_PIPE_STAGES 4
#define NUM_EMPTY_PIPE_STAGES 4
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 8192
#define SMEM_A_STAGE_STRIDE 8192
#define SMEM_B_STAGE_OFF 33792
#define SMEM_B_STAGE_STAGE_BYTES 16384
#define SMEM_B_STAGE_STRIDE 16384
#define SMEM_TOTAL 99328
#define THREADS 256
#define GROUP_M 16

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__global__ __launch_bounds__(256, 1) void
kernel_h3_fc1_swiglu_gemm(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ act_scale, float* __restrict__ w_scale, unsigned int* __restrict__ out, int M, int num_m_tiles, int total_tiles, float alpha)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int B_stage_addr = smem + 33792;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'full_pipe' ---
            // ab_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'empty_pipe' ---
            // ab_empty: 4 barriers, init_count=8
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            mbarrier_init(smem + 48, 8);
            mbarrier_init(smem + 56, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: lead ----
    if (warp == 0) {
        { // lead_main
            unsigned int load_stage = 0;
            unsigned int load_g = 0;
            unsigned int mma_stage = 0;
            unsigned int mma_kk = 0;
            int warp_m = 0;
            int warp_n = 0;
            float accum[128];
            unsigned int a_frag[16];
            unsigned int b_frag[16];
            unsigned int _phase_ab_empty = 1;
            if (lane == 0) {
                for (int _prologue = 0; _prologue < 3; _prologue++) {
                    if (bid + (int)(load_g / 80) * num_bids < total_tiles) {
                        mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                        mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 24576);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(A_stage_addr + load_stage * 8192), "l"((&A)), "r"(((int)load_g - (int)(load_g / 80) * 80) * 64), "r"(((bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * GROUP_M + (bid + (int)(load_g / 80) * num_bids - (bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * (GROUP_M * 136)) % ((num_m_tiles - (bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * GROUP_M : GROUP_M)) * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(B_stage_addr + load_stage * 16384), "l"((&B)), "r"(((int)load_g - (int)(load_g / 80) * 80) * 64), "r"((bid + (int)(load_g / 80) * num_bids - (bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * (GROUP_M * 136)) / ((num_m_tiles - (bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * GROUP_M : GROUP_M) * 256), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                    }
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; _phase_ab_empty ^= 1; }
                    load_g = load_g + 1;
                }
            }
            __syncwarp();
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile = bid; tile < total_tiles; tile += num_bids) {
                int tile_m = tile / (GROUP_M * 136) * GROUP_M + (tile - tile / (GROUP_M * 136) * (GROUP_M * 136)) % ((num_m_tiles - tile / (GROUP_M * 136) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 136) * GROUP_M : GROUP_M);
                int tile_n = (tile - tile / (GROUP_M * 136) * (GROUP_M * 136)) / ((num_m_tiles - tile / (GROUP_M * 136) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 136) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                accum[64] = 0.0f;
                accum[65] = 0.0f;
                accum[66] = 0.0f;
                accum[67] = 0.0f;
                accum[68] = 0.0f;
                accum[69] = 0.0f;
                accum[70] = 0.0f;
                accum[71] = 0.0f;
                accum[72] = 0.0f;
                accum[73] = 0.0f;
                accum[74] = 0.0f;
                accum[75] = 0.0f;
                accum[76] = 0.0f;
                accum[77] = 0.0f;
                accum[78] = 0.0f;
                accum[79] = 0.0f;
                accum[80] = 0.0f;
                accum[81] = 0.0f;
                accum[82] = 0.0f;
                accum[83] = 0.0f;
                accum[84] = 0.0f;
                accum[85] = 0.0f;
                accum[86] = 0.0f;
                accum[87] = 0.0f;
                accum[88] = 0.0f;
                accum[89] = 0.0f;
                accum[90] = 0.0f;
                accum[91] = 0.0f;
                accum[92] = 0.0f;
                accum[93] = 0.0f;
                accum[94] = 0.0f;
                accum[95] = 0.0f;
                accum[96] = 0.0f;
                accum[97] = 0.0f;
                accum[98] = 0.0f;
                accum[99] = 0.0f;
                accum[100] = 0.0f;
                accum[101] = 0.0f;
                accum[102] = 0.0f;
                accum[103] = 0.0f;
                accum[104] = 0.0f;
                accum[105] = 0.0f;
                accum[106] = 0.0f;
                accum[107] = 0.0f;
                accum[108] = 0.0f;
                accum[109] = 0.0f;
                accum[110] = 0.0f;
                accum[111] = 0.0f;
                accum[112] = 0.0f;
                accum[113] = 0.0f;
                accum[114] = 0.0f;
                accum[115] = 0.0f;
                accum[116] = 0.0f;
                accum[117] = 0.0f;
                accum[118] = 0.0f;
                accum[119] = 0.0f;
                accum[120] = 0.0f;
                accum[121] = 0.0f;
                accum[122] = 0.0f;
                accum[123] = 0.0f;
                accum[124] = 0.0f;
                accum[125] = 0.0f;
                accum[126] = 0.0f;
                accum[127] = 0.0f;
                #pragma unroll 1
                for (int k_tile = 0; k_tile < 80; k_tile++) {
                    if (lane == 0) {
                        if (bid + (int)(load_g / 80) * num_bids < total_tiles) {
                            mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 24576);
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(A_stage_addr + load_stage * 8192), "l"((&A)), "r"(((int)load_g - (int)(load_g / 80) * 80) * 64), "r"(((bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * GROUP_M + (bid + (int)(load_g / 80) * num_bids - (bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * (GROUP_M * 136)) % ((num_m_tiles - (bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * GROUP_M : GROUP_M)) * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(B_stage_addr + load_stage * 16384), "l"((&B)), "r"(((int)load_g - (int)(load_g / 80) * 80) * 64), "r"((bid + (int)(load_g / 80) * num_bids - (bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * (GROUP_M * 136)) / ((num_m_tiles - (bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 80) * num_bids) / (GROUP_M * 136) * GROUP_M : GROUP_M) * 256), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        }
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_ab_empty ^= 1; }
                        load_g = load_g + 1;
                    }
                    __syncwarp();
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    for (int k_step = 0; k_step < 2; k_step++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(A_stage_addr + mma_stage * 8192 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                            : "r"(A_stage_addr + mma_stage * 8192 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                            : "r"(A_stage_addr + mma_stage * 8192 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                            : "r"(A_stage_addr + mma_stage * 8192 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    }
                    __syncwarp();
                    if (lane == 0) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 4) { mma_stage = 0; _phase_ab_full ^= 1; }
                    mma_kk = mma_kk + 1;
                }
                float sw[16];
                sw[0] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2];
                sw[2] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 8];
                sw[1] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 1];
                sw[3] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 1 + 8];
                sw[4] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 16];
                sw[6] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 16 + 8];
                sw[5] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 16 + 1];
                sw[7] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 16 + 1 + 8];
                sw[8] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 32];
                sw[10] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 32 + 8];
                sw[9] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 32 + 1];
                sw[11] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 32 + 1 + 8];
                sw[12] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 48];
                sw[14] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 48 + 8];
                sw[13] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 48 + 1];
                sw[15] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 48 + 1 + 8];
                int row_lo_0 = tile_m * 128 + warp_m * 64 + (lane >> 2);
                int row_hi_0 = row_lo_0 + 8;
                float sa_lo_0 = ((row_lo_0 < M) ? act_scale[row_lo_0] : 0.0f);
                float sa_hi_0 = ((row_hi_0 < M) ? act_scale[row_hi_0] : 0.0f);
                float h_raw_0_0[8];
                h_raw_0_0[0] = accum[0] * sa_lo_0 * sw[0];
                h_raw_0_0[2] = accum[2] * sa_hi_0 * sw[0];
                h_raw_0_0[4] = accum[4] * sa_lo_0 * sw[2];
                h_raw_0_0[6] = accum[6] * sa_hi_0 * sw[2];
                h_raw_0_0[1] = accum[1] * sa_lo_0 * sw[1];
                h_raw_0_0[3] = accum[3] * sa_hi_0 * sw[1];
                h_raw_0_0[5] = accum[5] * sa_lo_0 * sw[3];
                h_raw_0_0[7] = accum[7] * sa_hi_0 * sw[3];
                uint32_t h_raw_0_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_0[_lp*2 + 0], h_raw_0_0[_lp*2+1 + 0]));
                    h_raw_0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_0_0 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2;
                if (row_lo_0 < M) {
                    out[(row_lo_0 * 34816 + canonical_col_0_0) / 2] = h_raw_0_0_bf16[0];
                    out[(row_lo_0 * 34816 + 17408 + canonical_col_0_0) / 2] = h_raw_0_0_bf16[2];
                }
                if (row_hi_0 < M) {
                    out[(row_hi_0 * 34816 + canonical_col_0_0) / 2] = h_raw_0_0_bf16[1];
                    out[(row_hi_0 * 34816 + 17408 + canonical_col_0_0) / 2] = h_raw_0_0_bf16[3];
                }
                float h_raw_0_1[8];
                h_raw_0_1[0] = accum[8] * sa_lo_0 * sw[4];
                h_raw_0_1[2] = accum[10] * sa_hi_0 * sw[4];
                h_raw_0_1[4] = accum[12] * sa_lo_0 * sw[6];
                h_raw_0_1[6] = accum[14] * sa_hi_0 * sw[6];
                h_raw_0_1[1] = accum[9] * sa_lo_0 * sw[5];
                h_raw_0_1[3] = accum[11] * sa_hi_0 * sw[5];
                h_raw_0_1[5] = accum[13] * sa_lo_0 * sw[7];
                h_raw_0_1[7] = accum[15] * sa_hi_0 * sw[7];
                uint32_t h_raw_0_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_1[_lp*2 + 0], h_raw_0_1[_lp*2+1 + 0]));
                    h_raw_0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_0_1 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8;
                if (row_lo_0 < M) {
                    out[(row_lo_0 * 34816 + canonical_col_0_1) / 2] = h_raw_0_1_bf16[0];
                    out[(row_lo_0 * 34816 + 17408 + canonical_col_0_1) / 2] = h_raw_0_1_bf16[2];
                }
                if (row_hi_0 < M) {
                    out[(row_hi_0 * 34816 + canonical_col_0_1) / 2] = h_raw_0_1_bf16[1];
                    out[(row_hi_0 * 34816 + 17408 + canonical_col_0_1) / 2] = h_raw_0_1_bf16[3];
                }
                float h_raw_0_2[8];
                h_raw_0_2[0] = accum[16] * sa_lo_0 * sw[8];
                h_raw_0_2[2] = accum[18] * sa_hi_0 * sw[8];
                h_raw_0_2[4] = accum[20] * sa_lo_0 * sw[10];
                h_raw_0_2[6] = accum[22] * sa_hi_0 * sw[10];
                h_raw_0_2[1] = accum[17] * sa_lo_0 * sw[9];
                h_raw_0_2[3] = accum[19] * sa_hi_0 * sw[9];
                h_raw_0_2[5] = accum[21] * sa_lo_0 * sw[11];
                h_raw_0_2[7] = accum[23] * sa_hi_0 * sw[11];
                uint32_t h_raw_0_2_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_2[_lp*2 + 0], h_raw_0_2[_lp*2+1 + 0]));
                    h_raw_0_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_0_2 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16;
                if (row_lo_0 < M) {
                    out[(row_lo_0 * 34816 + canonical_col_0_2) / 2] = h_raw_0_2_bf16[0];
                    out[(row_lo_0 * 34816 + 17408 + canonical_col_0_2) / 2] = h_raw_0_2_bf16[2];
                }
                if (row_hi_0 < M) {
                    out[(row_hi_0 * 34816 + canonical_col_0_2) / 2] = h_raw_0_2_bf16[1];
                    out[(row_hi_0 * 34816 + 17408 + canonical_col_0_2) / 2] = h_raw_0_2_bf16[3];
                }
                float h_raw_0_3[8];
                h_raw_0_3[0] = accum[24] * sa_lo_0 * sw[12];
                h_raw_0_3[2] = accum[26] * sa_hi_0 * sw[12];
                h_raw_0_3[4] = accum[28] * sa_lo_0 * sw[14];
                h_raw_0_3[6] = accum[30] * sa_hi_0 * sw[14];
                h_raw_0_3[1] = accum[25] * sa_lo_0 * sw[13];
                h_raw_0_3[3] = accum[27] * sa_hi_0 * sw[13];
                h_raw_0_3[5] = accum[29] * sa_lo_0 * sw[15];
                h_raw_0_3[7] = accum[31] * sa_hi_0 * sw[15];
                uint32_t h_raw_0_3_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_3[_lp*2 + 0], h_raw_0_3[_lp*2+1 + 0]));
                    h_raw_0_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_0_3 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24;
                if (row_lo_0 < M) {
                    out[(row_lo_0 * 34816 + canonical_col_0_3) / 2] = h_raw_0_3_bf16[0];
                    out[(row_lo_0 * 34816 + 17408 + canonical_col_0_3) / 2] = h_raw_0_3_bf16[2];
                }
                if (row_hi_0 < M) {
                    out[(row_hi_0 * 34816 + canonical_col_0_3) / 2] = h_raw_0_3_bf16[1];
                    out[(row_hi_0 * 34816 + 17408 + canonical_col_0_3) / 2] = h_raw_0_3_bf16[3];
                }
                int row_lo_1 = tile_m * 128 + warp_m * 64 + 16 + (lane >> 2);
                int row_hi_1 = row_lo_1 + 8;
                float sa_lo_1 = ((row_lo_1 < M) ? act_scale[row_lo_1] : 0.0f);
                float sa_hi_1 = ((row_hi_1 < M) ? act_scale[row_hi_1] : 0.0f);
                float h_raw_1_0[8];
                h_raw_1_0[0] = accum[32] * sa_lo_1 * sw[0];
                h_raw_1_0[2] = accum[34] * sa_hi_1 * sw[0];
                h_raw_1_0[4] = accum[36] * sa_lo_1 * sw[2];
                h_raw_1_0[6] = accum[38] * sa_hi_1 * sw[2];
                h_raw_1_0[1] = accum[33] * sa_lo_1 * sw[1];
                h_raw_1_0[3] = accum[35] * sa_hi_1 * sw[1];
                h_raw_1_0[5] = accum[37] * sa_lo_1 * sw[3];
                h_raw_1_0[7] = accum[39] * sa_hi_1 * sw[3];
                uint32_t h_raw_1_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_0[_lp*2 + 0], h_raw_1_0[_lp*2+1 + 0]));
                    h_raw_1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_1_0 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2;
                if (row_lo_1 < M) {
                    out[(row_lo_1 * 34816 + canonical_col_1_0) / 2] = h_raw_1_0_bf16[0];
                    out[(row_lo_1 * 34816 + 17408 + canonical_col_1_0) / 2] = h_raw_1_0_bf16[2];
                }
                if (row_hi_1 < M) {
                    out[(row_hi_1 * 34816 + canonical_col_1_0) / 2] = h_raw_1_0_bf16[1];
                    out[(row_hi_1 * 34816 + 17408 + canonical_col_1_0) / 2] = h_raw_1_0_bf16[3];
                }
                float h_raw_1_1[8];
                h_raw_1_1[0] = accum[40] * sa_lo_1 * sw[4];
                h_raw_1_1[2] = accum[42] * sa_hi_1 * sw[4];
                h_raw_1_1[4] = accum[44] * sa_lo_1 * sw[6];
                h_raw_1_1[6] = accum[46] * sa_hi_1 * sw[6];
                h_raw_1_1[1] = accum[41] * sa_lo_1 * sw[5];
                h_raw_1_1[3] = accum[43] * sa_hi_1 * sw[5];
                h_raw_1_1[5] = accum[45] * sa_lo_1 * sw[7];
                h_raw_1_1[7] = accum[47] * sa_hi_1 * sw[7];
                uint32_t h_raw_1_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_1[_lp*2 + 0], h_raw_1_1[_lp*2+1 + 0]));
                    h_raw_1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_1_1 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8;
                if (row_lo_1 < M) {
                    out[(row_lo_1 * 34816 + canonical_col_1_1) / 2] = h_raw_1_1_bf16[0];
                    out[(row_lo_1 * 34816 + 17408 + canonical_col_1_1) / 2] = h_raw_1_1_bf16[2];
                }
                if (row_hi_1 < M) {
                    out[(row_hi_1 * 34816 + canonical_col_1_1) / 2] = h_raw_1_1_bf16[1];
                    out[(row_hi_1 * 34816 + 17408 + canonical_col_1_1) / 2] = h_raw_1_1_bf16[3];
                }
                float h_raw_1_2[8];
                h_raw_1_2[0] = accum[48] * sa_lo_1 * sw[8];
                h_raw_1_2[2] = accum[50] * sa_hi_1 * sw[8];
                h_raw_1_2[4] = accum[52] * sa_lo_1 * sw[10];
                h_raw_1_2[6] = accum[54] * sa_hi_1 * sw[10];
                h_raw_1_2[1] = accum[49] * sa_lo_1 * sw[9];
                h_raw_1_2[3] = accum[51] * sa_hi_1 * sw[9];
                h_raw_1_2[5] = accum[53] * sa_lo_1 * sw[11];
                h_raw_1_2[7] = accum[55] * sa_hi_1 * sw[11];
                uint32_t h_raw_1_2_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_2[_lp*2 + 0], h_raw_1_2[_lp*2+1 + 0]));
                    h_raw_1_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_1_2 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16;
                if (row_lo_1 < M) {
                    out[(row_lo_1 * 34816 + canonical_col_1_2) / 2] = h_raw_1_2_bf16[0];
                    out[(row_lo_1 * 34816 + 17408 + canonical_col_1_2) / 2] = h_raw_1_2_bf16[2];
                }
                if (row_hi_1 < M) {
                    out[(row_hi_1 * 34816 + canonical_col_1_2) / 2] = h_raw_1_2_bf16[1];
                    out[(row_hi_1 * 34816 + 17408 + canonical_col_1_2) / 2] = h_raw_1_2_bf16[3];
                }
                float h_raw_1_3[8];
                h_raw_1_3[0] = accum[56] * sa_lo_1 * sw[12];
                h_raw_1_3[2] = accum[58] * sa_hi_1 * sw[12];
                h_raw_1_3[4] = accum[60] * sa_lo_1 * sw[14];
                h_raw_1_3[6] = accum[62] * sa_hi_1 * sw[14];
                h_raw_1_3[1] = accum[57] * sa_lo_1 * sw[13];
                h_raw_1_3[3] = accum[59] * sa_hi_1 * sw[13];
                h_raw_1_3[5] = accum[61] * sa_lo_1 * sw[15];
                h_raw_1_3[7] = accum[63] * sa_hi_1 * sw[15];
                uint32_t h_raw_1_3_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_3[_lp*2 + 0], h_raw_1_3[_lp*2+1 + 0]));
                    h_raw_1_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_1_3 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24;
                if (row_lo_1 < M) {
                    out[(row_lo_1 * 34816 + canonical_col_1_3) / 2] = h_raw_1_3_bf16[0];
                    out[(row_lo_1 * 34816 + 17408 + canonical_col_1_3) / 2] = h_raw_1_3_bf16[2];
                }
                if (row_hi_1 < M) {
                    out[(row_hi_1 * 34816 + canonical_col_1_3) / 2] = h_raw_1_3_bf16[1];
                    out[(row_hi_1 * 34816 + 17408 + canonical_col_1_3) / 2] = h_raw_1_3_bf16[3];
                }
                int row_lo_2 = tile_m * 128 + warp_m * 64 + 32 + (lane >> 2);
                int row_hi_2 = row_lo_2 + 8;
                float sa_lo_2 = ((row_lo_2 < M) ? act_scale[row_lo_2] : 0.0f);
                float sa_hi_2 = ((row_hi_2 < M) ? act_scale[row_hi_2] : 0.0f);
                float h_raw_2_0[8];
                h_raw_2_0[0] = accum[64] * sa_lo_2 * sw[0];
                h_raw_2_0[2] = accum[66] * sa_hi_2 * sw[0];
                h_raw_2_0[4] = accum[68] * sa_lo_2 * sw[2];
                h_raw_2_0[6] = accum[70] * sa_hi_2 * sw[2];
                h_raw_2_0[1] = accum[65] * sa_lo_2 * sw[1];
                h_raw_2_0[3] = accum[67] * sa_hi_2 * sw[1];
                h_raw_2_0[5] = accum[69] * sa_lo_2 * sw[3];
                h_raw_2_0[7] = accum[71] * sa_hi_2 * sw[3];
                uint32_t h_raw_2_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_0[_lp*2 + 0], h_raw_2_0[_lp*2+1 + 0]));
                    h_raw_2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_2_0 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2;
                if (row_lo_2 < M) {
                    out[(row_lo_2 * 34816 + canonical_col_2_0) / 2] = h_raw_2_0_bf16[0];
                    out[(row_lo_2 * 34816 + 17408 + canonical_col_2_0) / 2] = h_raw_2_0_bf16[2];
                }
                if (row_hi_2 < M) {
                    out[(row_hi_2 * 34816 + canonical_col_2_0) / 2] = h_raw_2_0_bf16[1];
                    out[(row_hi_2 * 34816 + 17408 + canonical_col_2_0) / 2] = h_raw_2_0_bf16[3];
                }
                float h_raw_2_1[8];
                h_raw_2_1[0] = accum[72] * sa_lo_2 * sw[4];
                h_raw_2_1[2] = accum[74] * sa_hi_2 * sw[4];
                h_raw_2_1[4] = accum[76] * sa_lo_2 * sw[6];
                h_raw_2_1[6] = accum[78] * sa_hi_2 * sw[6];
                h_raw_2_1[1] = accum[73] * sa_lo_2 * sw[5];
                h_raw_2_1[3] = accum[75] * sa_hi_2 * sw[5];
                h_raw_2_1[5] = accum[77] * sa_lo_2 * sw[7];
                h_raw_2_1[7] = accum[79] * sa_hi_2 * sw[7];
                uint32_t h_raw_2_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_1[_lp*2 + 0], h_raw_2_1[_lp*2+1 + 0]));
                    h_raw_2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_2_1 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8;
                if (row_lo_2 < M) {
                    out[(row_lo_2 * 34816 + canonical_col_2_1) / 2] = h_raw_2_1_bf16[0];
                    out[(row_lo_2 * 34816 + 17408 + canonical_col_2_1) / 2] = h_raw_2_1_bf16[2];
                }
                if (row_hi_2 < M) {
                    out[(row_hi_2 * 34816 + canonical_col_2_1) / 2] = h_raw_2_1_bf16[1];
                    out[(row_hi_2 * 34816 + 17408 + canonical_col_2_1) / 2] = h_raw_2_1_bf16[3];
                }
                float h_raw_2_2[8];
                h_raw_2_2[0] = accum[80] * sa_lo_2 * sw[8];
                h_raw_2_2[2] = accum[82] * sa_hi_2 * sw[8];
                h_raw_2_2[4] = accum[84] * sa_lo_2 * sw[10];
                h_raw_2_2[6] = accum[86] * sa_hi_2 * sw[10];
                h_raw_2_2[1] = accum[81] * sa_lo_2 * sw[9];
                h_raw_2_2[3] = accum[83] * sa_hi_2 * sw[9];
                h_raw_2_2[5] = accum[85] * sa_lo_2 * sw[11];
                h_raw_2_2[7] = accum[87] * sa_hi_2 * sw[11];
                uint32_t h_raw_2_2_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_2[_lp*2 + 0], h_raw_2_2[_lp*2+1 + 0]));
                    h_raw_2_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_2_2 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16;
                if (row_lo_2 < M) {
                    out[(row_lo_2 * 34816 + canonical_col_2_2) / 2] = h_raw_2_2_bf16[0];
                    out[(row_lo_2 * 34816 + 17408 + canonical_col_2_2) / 2] = h_raw_2_2_bf16[2];
                }
                if (row_hi_2 < M) {
                    out[(row_hi_2 * 34816 + canonical_col_2_2) / 2] = h_raw_2_2_bf16[1];
                    out[(row_hi_2 * 34816 + 17408 + canonical_col_2_2) / 2] = h_raw_2_2_bf16[3];
                }
                float h_raw_2_3[8];
                h_raw_2_3[0] = accum[88] * sa_lo_2 * sw[12];
                h_raw_2_3[2] = accum[90] * sa_hi_2 * sw[12];
                h_raw_2_3[4] = accum[92] * sa_lo_2 * sw[14];
                h_raw_2_3[6] = accum[94] * sa_hi_2 * sw[14];
                h_raw_2_3[1] = accum[89] * sa_lo_2 * sw[13];
                h_raw_2_3[3] = accum[91] * sa_hi_2 * sw[13];
                h_raw_2_3[5] = accum[93] * sa_lo_2 * sw[15];
                h_raw_2_3[7] = accum[95] * sa_hi_2 * sw[15];
                uint32_t h_raw_2_3_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_3[_lp*2 + 0], h_raw_2_3[_lp*2+1 + 0]));
                    h_raw_2_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_2_3 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24;
                if (row_lo_2 < M) {
                    out[(row_lo_2 * 34816 + canonical_col_2_3) / 2] = h_raw_2_3_bf16[0];
                    out[(row_lo_2 * 34816 + 17408 + canonical_col_2_3) / 2] = h_raw_2_3_bf16[2];
                }
                if (row_hi_2 < M) {
                    out[(row_hi_2 * 34816 + canonical_col_2_3) / 2] = h_raw_2_3_bf16[1];
                    out[(row_hi_2 * 34816 + 17408 + canonical_col_2_3) / 2] = h_raw_2_3_bf16[3];
                }
                int row_lo_3 = tile_m * 128 + warp_m * 64 + 48 + (lane >> 2);
                int row_hi_3 = row_lo_3 + 8;
                float sa_lo_3 = ((row_lo_3 < M) ? act_scale[row_lo_3] : 0.0f);
                float sa_hi_3 = ((row_hi_3 < M) ? act_scale[row_hi_3] : 0.0f);
                float h_raw_3_0[8];
                h_raw_3_0[0] = accum[96] * sa_lo_3 * sw[0];
                h_raw_3_0[2] = accum[98] * sa_hi_3 * sw[0];
                h_raw_3_0[4] = accum[100] * sa_lo_3 * sw[2];
                h_raw_3_0[6] = accum[102] * sa_hi_3 * sw[2];
                h_raw_3_0[1] = accum[97] * sa_lo_3 * sw[1];
                h_raw_3_0[3] = accum[99] * sa_hi_3 * sw[1];
                h_raw_3_0[5] = accum[101] * sa_lo_3 * sw[3];
                h_raw_3_0[7] = accum[103] * sa_hi_3 * sw[3];
                uint32_t h_raw_3_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_0[_lp*2 + 0], h_raw_3_0[_lp*2+1 + 0]));
                    h_raw_3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_3_0 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2;
                if (row_lo_3 < M) {
                    out[(row_lo_3 * 34816 + canonical_col_3_0) / 2] = h_raw_3_0_bf16[0];
                    out[(row_lo_3 * 34816 + 17408 + canonical_col_3_0) / 2] = h_raw_3_0_bf16[2];
                }
                if (row_hi_3 < M) {
                    out[(row_hi_3 * 34816 + canonical_col_3_0) / 2] = h_raw_3_0_bf16[1];
                    out[(row_hi_3 * 34816 + 17408 + canonical_col_3_0) / 2] = h_raw_3_0_bf16[3];
                }
                float h_raw_3_1[8];
                h_raw_3_1[0] = accum[104] * sa_lo_3 * sw[4];
                h_raw_3_1[2] = accum[106] * sa_hi_3 * sw[4];
                h_raw_3_1[4] = accum[108] * sa_lo_3 * sw[6];
                h_raw_3_1[6] = accum[110] * sa_hi_3 * sw[6];
                h_raw_3_1[1] = accum[105] * sa_lo_3 * sw[5];
                h_raw_3_1[3] = accum[107] * sa_hi_3 * sw[5];
                h_raw_3_1[5] = accum[109] * sa_lo_3 * sw[7];
                h_raw_3_1[7] = accum[111] * sa_hi_3 * sw[7];
                uint32_t h_raw_3_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_1[_lp*2 + 0], h_raw_3_1[_lp*2+1 + 0]));
                    h_raw_3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_3_1 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8;
                if (row_lo_3 < M) {
                    out[(row_lo_3 * 34816 + canonical_col_3_1) / 2] = h_raw_3_1_bf16[0];
                    out[(row_lo_3 * 34816 + 17408 + canonical_col_3_1) / 2] = h_raw_3_1_bf16[2];
                }
                if (row_hi_3 < M) {
                    out[(row_hi_3 * 34816 + canonical_col_3_1) / 2] = h_raw_3_1_bf16[1];
                    out[(row_hi_3 * 34816 + 17408 + canonical_col_3_1) / 2] = h_raw_3_1_bf16[3];
                }
                float h_raw_3_2[8];
                h_raw_3_2[0] = accum[112] * sa_lo_3 * sw[8];
                h_raw_3_2[2] = accum[114] * sa_hi_3 * sw[8];
                h_raw_3_2[4] = accum[116] * sa_lo_3 * sw[10];
                h_raw_3_2[6] = accum[118] * sa_hi_3 * sw[10];
                h_raw_3_2[1] = accum[113] * sa_lo_3 * sw[9];
                h_raw_3_2[3] = accum[115] * sa_hi_3 * sw[9];
                h_raw_3_2[5] = accum[117] * sa_lo_3 * sw[11];
                h_raw_3_2[7] = accum[119] * sa_hi_3 * sw[11];
                uint32_t h_raw_3_2_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_2[_lp*2 + 0], h_raw_3_2[_lp*2+1 + 0]));
                    h_raw_3_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_3_2 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16;
                if (row_lo_3 < M) {
                    out[(row_lo_3 * 34816 + canonical_col_3_2) / 2] = h_raw_3_2_bf16[0];
                    out[(row_lo_3 * 34816 + 17408 + canonical_col_3_2) / 2] = h_raw_3_2_bf16[2];
                }
                if (row_hi_3 < M) {
                    out[(row_hi_3 * 34816 + canonical_col_3_2) / 2] = h_raw_3_2_bf16[1];
                    out[(row_hi_3 * 34816 + 17408 + canonical_col_3_2) / 2] = h_raw_3_2_bf16[3];
                }
                float h_raw_3_3[8];
                h_raw_3_3[0] = accum[120] * sa_lo_3 * sw[12];
                h_raw_3_3[2] = accum[122] * sa_hi_3 * sw[12];
                h_raw_3_3[4] = accum[124] * sa_lo_3 * sw[14];
                h_raw_3_3[6] = accum[126] * sa_hi_3 * sw[14];
                h_raw_3_3[1] = accum[121] * sa_lo_3 * sw[13];
                h_raw_3_3[3] = accum[123] * sa_hi_3 * sw[13];
                h_raw_3_3[5] = accum[125] * sa_lo_3 * sw[15];
                h_raw_3_3[7] = accum[127] * sa_hi_3 * sw[15];
                uint32_t h_raw_3_3_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_3[_lp*2 + 0], h_raw_3_3[_lp*2+1 + 0]));
                    h_raw_3_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_3_3 = tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24;
                if (row_lo_3 < M) {
                    out[(row_lo_3 * 34816 + canonical_col_3_3) / 2] = h_raw_3_3_bf16[0];
                    out[(row_lo_3 * 34816 + 17408 + canonical_col_3_3) / 2] = h_raw_3_3_bf16[2];
                }
                if (row_hi_3 < M) {
                    out[(row_hi_3 * 34816 + canonical_col_3_3) / 2] = h_raw_3_3_bf16[1];
                    out[(row_hi_3 * 34816 + 17408 + canonical_col_3_3) / 2] = h_raw_3_3_bf16[3];
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 1 && warp <= 7) {
        { // mma_main
            unsigned int mma_stage_1 = 0;
            unsigned int mma_kk_1 = 0;
            int warp_m_1 = warp / 4;
            int warp_n_1 = warp % 4;
            float accum_1[128];
            unsigned int a_frag_1[16];
            unsigned int b_frag_1[16];
            unsigned int _phase_ab_full_1 = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 136) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 136) * (GROUP_M * 136)) % ((num_m_tiles - tile_1 / (GROUP_M * 136) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 136) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 136) * (GROUP_M * 136)) / ((num_m_tiles - tile_1 / (GROUP_M * 136) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 136) * GROUP_M : GROUP_M);
                accum_1[0] = 0.0f;
                accum_1[1] = 0.0f;
                accum_1[2] = 0.0f;
                accum_1[3] = 0.0f;
                accum_1[4] = 0.0f;
                accum_1[5] = 0.0f;
                accum_1[6] = 0.0f;
                accum_1[7] = 0.0f;
                accum_1[8] = 0.0f;
                accum_1[9] = 0.0f;
                accum_1[10] = 0.0f;
                accum_1[11] = 0.0f;
                accum_1[12] = 0.0f;
                accum_1[13] = 0.0f;
                accum_1[14] = 0.0f;
                accum_1[15] = 0.0f;
                accum_1[16] = 0.0f;
                accum_1[17] = 0.0f;
                accum_1[18] = 0.0f;
                accum_1[19] = 0.0f;
                accum_1[20] = 0.0f;
                accum_1[21] = 0.0f;
                accum_1[22] = 0.0f;
                accum_1[23] = 0.0f;
                accum_1[24] = 0.0f;
                accum_1[25] = 0.0f;
                accum_1[26] = 0.0f;
                accum_1[27] = 0.0f;
                accum_1[28] = 0.0f;
                accum_1[29] = 0.0f;
                accum_1[30] = 0.0f;
                accum_1[31] = 0.0f;
                accum_1[32] = 0.0f;
                accum_1[33] = 0.0f;
                accum_1[34] = 0.0f;
                accum_1[35] = 0.0f;
                accum_1[36] = 0.0f;
                accum_1[37] = 0.0f;
                accum_1[38] = 0.0f;
                accum_1[39] = 0.0f;
                accum_1[40] = 0.0f;
                accum_1[41] = 0.0f;
                accum_1[42] = 0.0f;
                accum_1[43] = 0.0f;
                accum_1[44] = 0.0f;
                accum_1[45] = 0.0f;
                accum_1[46] = 0.0f;
                accum_1[47] = 0.0f;
                accum_1[48] = 0.0f;
                accum_1[49] = 0.0f;
                accum_1[50] = 0.0f;
                accum_1[51] = 0.0f;
                accum_1[52] = 0.0f;
                accum_1[53] = 0.0f;
                accum_1[54] = 0.0f;
                accum_1[55] = 0.0f;
                accum_1[56] = 0.0f;
                accum_1[57] = 0.0f;
                accum_1[58] = 0.0f;
                accum_1[59] = 0.0f;
                accum_1[60] = 0.0f;
                accum_1[61] = 0.0f;
                accum_1[62] = 0.0f;
                accum_1[63] = 0.0f;
                accum_1[64] = 0.0f;
                accum_1[65] = 0.0f;
                accum_1[66] = 0.0f;
                accum_1[67] = 0.0f;
                accum_1[68] = 0.0f;
                accum_1[69] = 0.0f;
                accum_1[70] = 0.0f;
                accum_1[71] = 0.0f;
                accum_1[72] = 0.0f;
                accum_1[73] = 0.0f;
                accum_1[74] = 0.0f;
                accum_1[75] = 0.0f;
                accum_1[76] = 0.0f;
                accum_1[77] = 0.0f;
                accum_1[78] = 0.0f;
                accum_1[79] = 0.0f;
                accum_1[80] = 0.0f;
                accum_1[81] = 0.0f;
                accum_1[82] = 0.0f;
                accum_1[83] = 0.0f;
                accum_1[84] = 0.0f;
                accum_1[85] = 0.0f;
                accum_1[86] = 0.0f;
                accum_1[87] = 0.0f;
                accum_1[88] = 0.0f;
                accum_1[89] = 0.0f;
                accum_1[90] = 0.0f;
                accum_1[91] = 0.0f;
                accum_1[92] = 0.0f;
                accum_1[93] = 0.0f;
                accum_1[94] = 0.0f;
                accum_1[95] = 0.0f;
                accum_1[96] = 0.0f;
                accum_1[97] = 0.0f;
                accum_1[98] = 0.0f;
                accum_1[99] = 0.0f;
                accum_1[100] = 0.0f;
                accum_1[101] = 0.0f;
                accum_1[102] = 0.0f;
                accum_1[103] = 0.0f;
                accum_1[104] = 0.0f;
                accum_1[105] = 0.0f;
                accum_1[106] = 0.0f;
                accum_1[107] = 0.0f;
                accum_1[108] = 0.0f;
                accum_1[109] = 0.0f;
                accum_1[110] = 0.0f;
                accum_1[111] = 0.0f;
                accum_1[112] = 0.0f;
                accum_1[113] = 0.0f;
                accum_1[114] = 0.0f;
                accum_1[115] = 0.0f;
                accum_1[116] = 0.0f;
                accum_1[117] = 0.0f;
                accum_1[118] = 0.0f;
                accum_1[119] = 0.0f;
                accum_1[120] = 0.0f;
                accum_1[121] = 0.0f;
                accum_1[122] = 0.0f;
                accum_1[123] = 0.0f;
                accum_1[124] = 0.0f;
                accum_1[125] = 0.0f;
                accum_1[126] = 0.0f;
                accum_1[127] = 0.0f;
                #pragma unroll 1
                for (int k_tile_1 = 0; k_tile_1 < 80; k_tile_1++) {
                    mbarrier_wait(ab_full_addr + (mma_stage_1) * 8, _phase_ab_full_1);
                    for (int k_step_1 = 0; k_step_1 < 2; k_step_1++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(A_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_m_1 * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[4]), "=r"(a_frag_1[5]), "=r"(a_frag_1[6]), "=r"(a_frag_1[7])
                            : "r"(A_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_m_1 * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[8]), "=r"(a_frag_1[9]), "=r"(a_frag_1[10]), "=r"(a_frag_1[11])
                            : "r"(A_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_m_1 * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[12]), "=r"(a_frag_1[13]), "=r"(a_frag_1[14]), "=r"(a_frag_1[15])
                            : "r"(A_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_m_1 * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1]), "=r"(b_frag_1[2]), "=r"(b_frag_1[3])
                            : "r"(B_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_n_1 * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[4]), "=r"(b_frag_1[5]), "=r"(b_frag_1[6]), "=r"(b_frag_1[7])
                            : "r"(B_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_n_1 * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[8]), "=r"(b_frag_1[9]), "=r"(b_frag_1[10]), "=r"(b_frag_1[11])
                            : "r"(B_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_n_1 * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[12]), "=r"(b_frag_1[13]), "=r"(b_frag_1[14]), "=r"(b_frag_1[15])
                            : "r"(B_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_n_1 * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[0]), "+f"(accum_1[1]), "+f"(accum_1[2]), "+f"(accum_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[4]), "+f"(accum_1[(4) + 1]), "+f"(accum_1[(4) + 2]), "+f"(accum_1[(4) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[8]), "+f"(accum_1[(8) + 1]), "+f"(accum_1[(8) + 2]), "+f"(accum_1[(8) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[12]), "+f"(accum_1[(12) + 1]), "+f"(accum_1[(12) + 2]), "+f"(accum_1[(12) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[16]), "+f"(accum_1[(16) + 1]), "+f"(accum_1[(16) + 2]), "+f"(accum_1[(16) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[20]), "+f"(accum_1[(20) + 1]), "+f"(accum_1[(20) + 2]), "+f"(accum_1[(20) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[24]), "+f"(accum_1[(24) + 1]), "+f"(accum_1[(24) + 2]), "+f"(accum_1[(24) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[28]), "+f"(accum_1[(28) + 1]), "+f"(accum_1[(28) + 2]), "+f"(accum_1[(28) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[32]), "+f"(accum_1[(32) + 1]), "+f"(accum_1[(32) + 2]), "+f"(accum_1[(32) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[36]), "+f"(accum_1[(36) + 1]), "+f"(accum_1[(36) + 2]), "+f"(accum_1[(36) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[40]), "+f"(accum_1[(40) + 1]), "+f"(accum_1[(40) + 2]), "+f"(accum_1[(40) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[44]), "+f"(accum_1[(44) + 1]), "+f"(accum_1[(44) + 2]), "+f"(accum_1[(44) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[48]), "+f"(accum_1[(48) + 1]), "+f"(accum_1[(48) + 2]), "+f"(accum_1[(48) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[52]), "+f"(accum_1[(52) + 1]), "+f"(accum_1[(52) + 2]), "+f"(accum_1[(52) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[56]), "+f"(accum_1[(56) + 1]), "+f"(accum_1[(56) + 2]), "+f"(accum_1[(56) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[60]), "+f"(accum_1[(60) + 1]), "+f"(accum_1[(60) + 2]), "+f"(accum_1[(60) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[64]), "+f"(accum_1[(64) + 1]), "+f"(accum_1[(64) + 2]), "+f"(accum_1[(64) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[68]), "+f"(accum_1[(68) + 1]), "+f"(accum_1[(68) + 2]), "+f"(accum_1[(68) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[72]), "+f"(accum_1[(72) + 1]), "+f"(accum_1[(72) + 2]), "+f"(accum_1[(72) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[76]), "+f"(accum_1[(76) + 1]), "+f"(accum_1[(76) + 2]), "+f"(accum_1[(76) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[80]), "+f"(accum_1[(80) + 1]), "+f"(accum_1[(80) + 2]), "+f"(accum_1[(80) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[84]), "+f"(accum_1[(84) + 1]), "+f"(accum_1[(84) + 2]), "+f"(accum_1[(84) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[88]), "+f"(accum_1[(88) + 1]), "+f"(accum_1[(88) + 2]), "+f"(accum_1[(88) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[92]), "+f"(accum_1[(92) + 1]), "+f"(accum_1[(92) + 2]), "+f"(accum_1[(92) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[96]), "+f"(accum_1[(96) + 1]), "+f"(accum_1[(96) + 2]), "+f"(accum_1[(96) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[100]), "+f"(accum_1[(100) + 1]), "+f"(accum_1[(100) + 2]), "+f"(accum_1[(100) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[104]), "+f"(accum_1[(104) + 1]), "+f"(accum_1[(104) + 2]), "+f"(accum_1[(104) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[108]), "+f"(accum_1[(108) + 1]), "+f"(accum_1[(108) + 2]), "+f"(accum_1[(108) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[112]), "+f"(accum_1[(112) + 1]), "+f"(accum_1[(112) + 2]), "+f"(accum_1[(112) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[116]), "+f"(accum_1[(116) + 1]), "+f"(accum_1[(116) + 2]), "+f"(accum_1[(116) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[120]), "+f"(accum_1[(120) + 1]), "+f"(accum_1[(120) + 2]), "+f"(accum_1[(120) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[124]), "+f"(accum_1[(124) + 1]), "+f"(accum_1[(124) + 2]), "+f"(accum_1[(124) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]));
                    }
                    __syncwarp();
                    if (lane == 0) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage_1) * 8);
                    }
                    mma_stage_1 += 1;
                    if (mma_stage_1 == 4) { mma_stage_1 = 0; _phase_ab_full_1 ^= 1; }
                    mma_kk_1 = mma_kk_1 + 1;
                }
                float sw_1[16];
                sw_1[0] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2];
                sw_1[2] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 8];
                sw_1[1] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 1];
                sw_1[3] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 1 + 8];
                sw_1[4] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 16];
                sw_1[6] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 16 + 8];
                sw_1[5] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 16 + 1];
                sw_1[7] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 16 + 1 + 8];
                sw_1[8] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 32];
                sw_1[10] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 32 + 8];
                sw_1[9] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 32 + 1];
                sw_1[11] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 32 + 1 + 8];
                sw_1[12] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 48];
                sw_1[14] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 48 + 8];
                sw_1[13] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 48 + 1];
                sw_1[15] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 48 + 1 + 8];
                int row_lo_0_1 = tile_m_1 * 128 + warp_m_1 * 64 + (lane >> 2);
                int row_hi_0_1 = row_lo_0_1 + 8;
                float sa_lo_0_1 = ((row_lo_0_1 < M) ? act_scale[row_lo_0_1] : 0.0f);
                float sa_hi_0_1 = ((row_hi_0_1 < M) ? act_scale[row_hi_0_1] : 0.0f);
                float h_raw_0_0_1[8];
                h_raw_0_0_1[0] = accum_1[0] * sa_lo_0_1 * sw_1[0];
                h_raw_0_0_1[2] = accum_1[2] * sa_hi_0_1 * sw_1[0];
                h_raw_0_0_1[4] = accum_1[4] * sa_lo_0_1 * sw_1[2];
                h_raw_0_0_1[6] = accum_1[6] * sa_hi_0_1 * sw_1[2];
                h_raw_0_0_1[1] = accum_1[1] * sa_lo_0_1 * sw_1[1];
                h_raw_0_0_1[3] = accum_1[3] * sa_hi_0_1 * sw_1[1];
                h_raw_0_0_1[5] = accum_1[5] * sa_lo_0_1 * sw_1[3];
                h_raw_0_0_1[7] = accum_1[7] * sa_hi_0_1 * sw_1[3];
                uint32_t h_raw_0_0_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_0_1[_lp*2 + 0], h_raw_0_0_1[_lp*2+1 + 0]));
                    h_raw_0_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_0_0_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2;
                if (row_lo_0_1 < M) {
                    out[(row_lo_0_1 * 34816 + canonical_col_0_0_1) / 2] = h_raw_0_0_bf16_1[0];
                    out[(row_lo_0_1 * 34816 + 17408 + canonical_col_0_0_1) / 2] = h_raw_0_0_bf16_1[2];
                }
                if (row_hi_0_1 < M) {
                    out[(row_hi_0_1 * 34816 + canonical_col_0_0_1) / 2] = h_raw_0_0_bf16_1[1];
                    out[(row_hi_0_1 * 34816 + 17408 + canonical_col_0_0_1) / 2] = h_raw_0_0_bf16_1[3];
                }
                float h_raw_0_1_1[8];
                h_raw_0_1_1[0] = accum_1[8] * sa_lo_0_1 * sw_1[4];
                h_raw_0_1_1[2] = accum_1[10] * sa_hi_0_1 * sw_1[4];
                h_raw_0_1_1[4] = accum_1[12] * sa_lo_0_1 * sw_1[6];
                h_raw_0_1_1[6] = accum_1[14] * sa_hi_0_1 * sw_1[6];
                h_raw_0_1_1[1] = accum_1[9] * sa_lo_0_1 * sw_1[5];
                h_raw_0_1_1[3] = accum_1[11] * sa_hi_0_1 * sw_1[5];
                h_raw_0_1_1[5] = accum_1[13] * sa_lo_0_1 * sw_1[7];
                h_raw_0_1_1[7] = accum_1[15] * sa_hi_0_1 * sw_1[7];
                uint32_t h_raw_0_1_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_1_1[_lp*2 + 0], h_raw_0_1_1[_lp*2+1 + 0]));
                    h_raw_0_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_0_1_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8;
                if (row_lo_0_1 < M) {
                    out[(row_lo_0_1 * 34816 + canonical_col_0_1_1) / 2] = h_raw_0_1_bf16_1[0];
                    out[(row_lo_0_1 * 34816 + 17408 + canonical_col_0_1_1) / 2] = h_raw_0_1_bf16_1[2];
                }
                if (row_hi_0_1 < M) {
                    out[(row_hi_0_1 * 34816 + canonical_col_0_1_1) / 2] = h_raw_0_1_bf16_1[1];
                    out[(row_hi_0_1 * 34816 + 17408 + canonical_col_0_1_1) / 2] = h_raw_0_1_bf16_1[3];
                }
                float h_raw_0_2_1[8];
                h_raw_0_2_1[0] = accum_1[16] * sa_lo_0_1 * sw_1[8];
                h_raw_0_2_1[2] = accum_1[18] * sa_hi_0_1 * sw_1[8];
                h_raw_0_2_1[4] = accum_1[20] * sa_lo_0_1 * sw_1[10];
                h_raw_0_2_1[6] = accum_1[22] * sa_hi_0_1 * sw_1[10];
                h_raw_0_2_1[1] = accum_1[17] * sa_lo_0_1 * sw_1[9];
                h_raw_0_2_1[3] = accum_1[19] * sa_hi_0_1 * sw_1[9];
                h_raw_0_2_1[5] = accum_1[21] * sa_lo_0_1 * sw_1[11];
                h_raw_0_2_1[7] = accum_1[23] * sa_hi_0_1 * sw_1[11];
                uint32_t h_raw_0_2_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_2_1[_lp*2 + 0], h_raw_0_2_1[_lp*2+1 + 0]));
                    h_raw_0_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_0_2_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16;
                if (row_lo_0_1 < M) {
                    out[(row_lo_0_1 * 34816 + canonical_col_0_2_1) / 2] = h_raw_0_2_bf16_1[0];
                    out[(row_lo_0_1 * 34816 + 17408 + canonical_col_0_2_1) / 2] = h_raw_0_2_bf16_1[2];
                }
                if (row_hi_0_1 < M) {
                    out[(row_hi_0_1 * 34816 + canonical_col_0_2_1) / 2] = h_raw_0_2_bf16_1[1];
                    out[(row_hi_0_1 * 34816 + 17408 + canonical_col_0_2_1) / 2] = h_raw_0_2_bf16_1[3];
                }
                float h_raw_0_3_1[8];
                h_raw_0_3_1[0] = accum_1[24] * sa_lo_0_1 * sw_1[12];
                h_raw_0_3_1[2] = accum_1[26] * sa_hi_0_1 * sw_1[12];
                h_raw_0_3_1[4] = accum_1[28] * sa_lo_0_1 * sw_1[14];
                h_raw_0_3_1[6] = accum_1[30] * sa_hi_0_1 * sw_1[14];
                h_raw_0_3_1[1] = accum_1[25] * sa_lo_0_1 * sw_1[13];
                h_raw_0_3_1[3] = accum_1[27] * sa_hi_0_1 * sw_1[13];
                h_raw_0_3_1[5] = accum_1[29] * sa_lo_0_1 * sw_1[15];
                h_raw_0_3_1[7] = accum_1[31] * sa_hi_0_1 * sw_1[15];
                uint32_t h_raw_0_3_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_3_1[_lp*2 + 0], h_raw_0_3_1[_lp*2+1 + 0]));
                    h_raw_0_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_0_3_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24;
                if (row_lo_0_1 < M) {
                    out[(row_lo_0_1 * 34816 + canonical_col_0_3_1) / 2] = h_raw_0_3_bf16_1[0];
                    out[(row_lo_0_1 * 34816 + 17408 + canonical_col_0_3_1) / 2] = h_raw_0_3_bf16_1[2];
                }
                if (row_hi_0_1 < M) {
                    out[(row_hi_0_1 * 34816 + canonical_col_0_3_1) / 2] = h_raw_0_3_bf16_1[1];
                    out[(row_hi_0_1 * 34816 + 17408 + canonical_col_0_3_1) / 2] = h_raw_0_3_bf16_1[3];
                }
                int row_lo_1_1 = tile_m_1 * 128 + warp_m_1 * 64 + 16 + (lane >> 2);
                int row_hi_1_1 = row_lo_1_1 + 8;
                float sa_lo_1_1 = ((row_lo_1_1 < M) ? act_scale[row_lo_1_1] : 0.0f);
                float sa_hi_1_1 = ((row_hi_1_1 < M) ? act_scale[row_hi_1_1] : 0.0f);
                float h_raw_1_0_1[8];
                h_raw_1_0_1[0] = accum_1[32] * sa_lo_1_1 * sw_1[0];
                h_raw_1_0_1[2] = accum_1[34] * sa_hi_1_1 * sw_1[0];
                h_raw_1_0_1[4] = accum_1[36] * sa_lo_1_1 * sw_1[2];
                h_raw_1_0_1[6] = accum_1[38] * sa_hi_1_1 * sw_1[2];
                h_raw_1_0_1[1] = accum_1[33] * sa_lo_1_1 * sw_1[1];
                h_raw_1_0_1[3] = accum_1[35] * sa_hi_1_1 * sw_1[1];
                h_raw_1_0_1[5] = accum_1[37] * sa_lo_1_1 * sw_1[3];
                h_raw_1_0_1[7] = accum_1[39] * sa_hi_1_1 * sw_1[3];
                uint32_t h_raw_1_0_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_0_1[_lp*2 + 0], h_raw_1_0_1[_lp*2+1 + 0]));
                    h_raw_1_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_1_0_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2;
                if (row_lo_1_1 < M) {
                    out[(row_lo_1_1 * 34816 + canonical_col_1_0_1) / 2] = h_raw_1_0_bf16_1[0];
                    out[(row_lo_1_1 * 34816 + 17408 + canonical_col_1_0_1) / 2] = h_raw_1_0_bf16_1[2];
                }
                if (row_hi_1_1 < M) {
                    out[(row_hi_1_1 * 34816 + canonical_col_1_0_1) / 2] = h_raw_1_0_bf16_1[1];
                    out[(row_hi_1_1 * 34816 + 17408 + canonical_col_1_0_1) / 2] = h_raw_1_0_bf16_1[3];
                }
                float h_raw_1_1_1[8];
                h_raw_1_1_1[0] = accum_1[40] * sa_lo_1_1 * sw_1[4];
                h_raw_1_1_1[2] = accum_1[42] * sa_hi_1_1 * sw_1[4];
                h_raw_1_1_1[4] = accum_1[44] * sa_lo_1_1 * sw_1[6];
                h_raw_1_1_1[6] = accum_1[46] * sa_hi_1_1 * sw_1[6];
                h_raw_1_1_1[1] = accum_1[41] * sa_lo_1_1 * sw_1[5];
                h_raw_1_1_1[3] = accum_1[43] * sa_hi_1_1 * sw_1[5];
                h_raw_1_1_1[5] = accum_1[45] * sa_lo_1_1 * sw_1[7];
                h_raw_1_1_1[7] = accum_1[47] * sa_hi_1_1 * sw_1[7];
                uint32_t h_raw_1_1_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_1_1[_lp*2 + 0], h_raw_1_1_1[_lp*2+1 + 0]));
                    h_raw_1_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_1_1_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8;
                if (row_lo_1_1 < M) {
                    out[(row_lo_1_1 * 34816 + canonical_col_1_1_1) / 2] = h_raw_1_1_bf16_1[0];
                    out[(row_lo_1_1 * 34816 + 17408 + canonical_col_1_1_1) / 2] = h_raw_1_1_bf16_1[2];
                }
                if (row_hi_1_1 < M) {
                    out[(row_hi_1_1 * 34816 + canonical_col_1_1_1) / 2] = h_raw_1_1_bf16_1[1];
                    out[(row_hi_1_1 * 34816 + 17408 + canonical_col_1_1_1) / 2] = h_raw_1_1_bf16_1[3];
                }
                float h_raw_1_2_1[8];
                h_raw_1_2_1[0] = accum_1[48] * sa_lo_1_1 * sw_1[8];
                h_raw_1_2_1[2] = accum_1[50] * sa_hi_1_1 * sw_1[8];
                h_raw_1_2_1[4] = accum_1[52] * sa_lo_1_1 * sw_1[10];
                h_raw_1_2_1[6] = accum_1[54] * sa_hi_1_1 * sw_1[10];
                h_raw_1_2_1[1] = accum_1[49] * sa_lo_1_1 * sw_1[9];
                h_raw_1_2_1[3] = accum_1[51] * sa_hi_1_1 * sw_1[9];
                h_raw_1_2_1[5] = accum_1[53] * sa_lo_1_1 * sw_1[11];
                h_raw_1_2_1[7] = accum_1[55] * sa_hi_1_1 * sw_1[11];
                uint32_t h_raw_1_2_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_2_1[_lp*2 + 0], h_raw_1_2_1[_lp*2+1 + 0]));
                    h_raw_1_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_1_2_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16;
                if (row_lo_1_1 < M) {
                    out[(row_lo_1_1 * 34816 + canonical_col_1_2_1) / 2] = h_raw_1_2_bf16_1[0];
                    out[(row_lo_1_1 * 34816 + 17408 + canonical_col_1_2_1) / 2] = h_raw_1_2_bf16_1[2];
                }
                if (row_hi_1_1 < M) {
                    out[(row_hi_1_1 * 34816 + canonical_col_1_2_1) / 2] = h_raw_1_2_bf16_1[1];
                    out[(row_hi_1_1 * 34816 + 17408 + canonical_col_1_2_1) / 2] = h_raw_1_2_bf16_1[3];
                }
                float h_raw_1_3_1[8];
                h_raw_1_3_1[0] = accum_1[56] * sa_lo_1_1 * sw_1[12];
                h_raw_1_3_1[2] = accum_1[58] * sa_hi_1_1 * sw_1[12];
                h_raw_1_3_1[4] = accum_1[60] * sa_lo_1_1 * sw_1[14];
                h_raw_1_3_1[6] = accum_1[62] * sa_hi_1_1 * sw_1[14];
                h_raw_1_3_1[1] = accum_1[57] * sa_lo_1_1 * sw_1[13];
                h_raw_1_3_1[3] = accum_1[59] * sa_hi_1_1 * sw_1[13];
                h_raw_1_3_1[5] = accum_1[61] * sa_lo_1_1 * sw_1[15];
                h_raw_1_3_1[7] = accum_1[63] * sa_hi_1_1 * sw_1[15];
                uint32_t h_raw_1_3_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_3_1[_lp*2 + 0], h_raw_1_3_1[_lp*2+1 + 0]));
                    h_raw_1_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_1_3_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24;
                if (row_lo_1_1 < M) {
                    out[(row_lo_1_1 * 34816 + canonical_col_1_3_1) / 2] = h_raw_1_3_bf16_1[0];
                    out[(row_lo_1_1 * 34816 + 17408 + canonical_col_1_3_1) / 2] = h_raw_1_3_bf16_1[2];
                }
                if (row_hi_1_1 < M) {
                    out[(row_hi_1_1 * 34816 + canonical_col_1_3_1) / 2] = h_raw_1_3_bf16_1[1];
                    out[(row_hi_1_1 * 34816 + 17408 + canonical_col_1_3_1) / 2] = h_raw_1_3_bf16_1[3];
                }
                int row_lo_2_1 = tile_m_1 * 128 + warp_m_1 * 64 + 32 + (lane >> 2);
                int row_hi_2_1 = row_lo_2_1 + 8;
                float sa_lo_2_1 = ((row_lo_2_1 < M) ? act_scale[row_lo_2_1] : 0.0f);
                float sa_hi_2_1 = ((row_hi_2_1 < M) ? act_scale[row_hi_2_1] : 0.0f);
                float h_raw_2_0_1[8];
                h_raw_2_0_1[0] = accum_1[64] * sa_lo_2_1 * sw_1[0];
                h_raw_2_0_1[2] = accum_1[66] * sa_hi_2_1 * sw_1[0];
                h_raw_2_0_1[4] = accum_1[68] * sa_lo_2_1 * sw_1[2];
                h_raw_2_0_1[6] = accum_1[70] * sa_hi_2_1 * sw_1[2];
                h_raw_2_0_1[1] = accum_1[65] * sa_lo_2_1 * sw_1[1];
                h_raw_2_0_1[3] = accum_1[67] * sa_hi_2_1 * sw_1[1];
                h_raw_2_0_1[5] = accum_1[69] * sa_lo_2_1 * sw_1[3];
                h_raw_2_0_1[7] = accum_1[71] * sa_hi_2_1 * sw_1[3];
                uint32_t h_raw_2_0_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_0_1[_lp*2 + 0], h_raw_2_0_1[_lp*2+1 + 0]));
                    h_raw_2_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_2_0_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2;
                if (row_lo_2_1 < M) {
                    out[(row_lo_2_1 * 34816 + canonical_col_2_0_1) / 2] = h_raw_2_0_bf16_1[0];
                    out[(row_lo_2_1 * 34816 + 17408 + canonical_col_2_0_1) / 2] = h_raw_2_0_bf16_1[2];
                }
                if (row_hi_2_1 < M) {
                    out[(row_hi_2_1 * 34816 + canonical_col_2_0_1) / 2] = h_raw_2_0_bf16_1[1];
                    out[(row_hi_2_1 * 34816 + 17408 + canonical_col_2_0_1) / 2] = h_raw_2_0_bf16_1[3];
                }
                float h_raw_2_1_1[8];
                h_raw_2_1_1[0] = accum_1[72] * sa_lo_2_1 * sw_1[4];
                h_raw_2_1_1[2] = accum_1[74] * sa_hi_2_1 * sw_1[4];
                h_raw_2_1_1[4] = accum_1[76] * sa_lo_2_1 * sw_1[6];
                h_raw_2_1_1[6] = accum_1[78] * sa_hi_2_1 * sw_1[6];
                h_raw_2_1_1[1] = accum_1[73] * sa_lo_2_1 * sw_1[5];
                h_raw_2_1_1[3] = accum_1[75] * sa_hi_2_1 * sw_1[5];
                h_raw_2_1_1[5] = accum_1[77] * sa_lo_2_1 * sw_1[7];
                h_raw_2_1_1[7] = accum_1[79] * sa_hi_2_1 * sw_1[7];
                uint32_t h_raw_2_1_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_1_1[_lp*2 + 0], h_raw_2_1_1[_lp*2+1 + 0]));
                    h_raw_2_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_2_1_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8;
                if (row_lo_2_1 < M) {
                    out[(row_lo_2_1 * 34816 + canonical_col_2_1_1) / 2] = h_raw_2_1_bf16_1[0];
                    out[(row_lo_2_1 * 34816 + 17408 + canonical_col_2_1_1) / 2] = h_raw_2_1_bf16_1[2];
                }
                if (row_hi_2_1 < M) {
                    out[(row_hi_2_1 * 34816 + canonical_col_2_1_1) / 2] = h_raw_2_1_bf16_1[1];
                    out[(row_hi_2_1 * 34816 + 17408 + canonical_col_2_1_1) / 2] = h_raw_2_1_bf16_1[3];
                }
                float h_raw_2_2_1[8];
                h_raw_2_2_1[0] = accum_1[80] * sa_lo_2_1 * sw_1[8];
                h_raw_2_2_1[2] = accum_1[82] * sa_hi_2_1 * sw_1[8];
                h_raw_2_2_1[4] = accum_1[84] * sa_lo_2_1 * sw_1[10];
                h_raw_2_2_1[6] = accum_1[86] * sa_hi_2_1 * sw_1[10];
                h_raw_2_2_1[1] = accum_1[81] * sa_lo_2_1 * sw_1[9];
                h_raw_2_2_1[3] = accum_1[83] * sa_hi_2_1 * sw_1[9];
                h_raw_2_2_1[5] = accum_1[85] * sa_lo_2_1 * sw_1[11];
                h_raw_2_2_1[7] = accum_1[87] * sa_hi_2_1 * sw_1[11];
                uint32_t h_raw_2_2_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_2_1[_lp*2 + 0], h_raw_2_2_1[_lp*2+1 + 0]));
                    h_raw_2_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_2_2_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16;
                if (row_lo_2_1 < M) {
                    out[(row_lo_2_1 * 34816 + canonical_col_2_2_1) / 2] = h_raw_2_2_bf16_1[0];
                    out[(row_lo_2_1 * 34816 + 17408 + canonical_col_2_2_1) / 2] = h_raw_2_2_bf16_1[2];
                }
                if (row_hi_2_1 < M) {
                    out[(row_hi_2_1 * 34816 + canonical_col_2_2_1) / 2] = h_raw_2_2_bf16_1[1];
                    out[(row_hi_2_1 * 34816 + 17408 + canonical_col_2_2_1) / 2] = h_raw_2_2_bf16_1[3];
                }
                float h_raw_2_3_1[8];
                h_raw_2_3_1[0] = accum_1[88] * sa_lo_2_1 * sw_1[12];
                h_raw_2_3_1[2] = accum_1[90] * sa_hi_2_1 * sw_1[12];
                h_raw_2_3_1[4] = accum_1[92] * sa_lo_2_1 * sw_1[14];
                h_raw_2_3_1[6] = accum_1[94] * sa_hi_2_1 * sw_1[14];
                h_raw_2_3_1[1] = accum_1[89] * sa_lo_2_1 * sw_1[13];
                h_raw_2_3_1[3] = accum_1[91] * sa_hi_2_1 * sw_1[13];
                h_raw_2_3_1[5] = accum_1[93] * sa_lo_2_1 * sw_1[15];
                h_raw_2_3_1[7] = accum_1[95] * sa_hi_2_1 * sw_1[15];
                uint32_t h_raw_2_3_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_3_1[_lp*2 + 0], h_raw_2_3_1[_lp*2+1 + 0]));
                    h_raw_2_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_2_3_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24;
                if (row_lo_2_1 < M) {
                    out[(row_lo_2_1 * 34816 + canonical_col_2_3_1) / 2] = h_raw_2_3_bf16_1[0];
                    out[(row_lo_2_1 * 34816 + 17408 + canonical_col_2_3_1) / 2] = h_raw_2_3_bf16_1[2];
                }
                if (row_hi_2_1 < M) {
                    out[(row_hi_2_1 * 34816 + canonical_col_2_3_1) / 2] = h_raw_2_3_bf16_1[1];
                    out[(row_hi_2_1 * 34816 + 17408 + canonical_col_2_3_1) / 2] = h_raw_2_3_bf16_1[3];
                }
                int row_lo_3_1 = tile_m_1 * 128 + warp_m_1 * 64 + 48 + (lane >> 2);
                int row_hi_3_1 = row_lo_3_1 + 8;
                float sa_lo_3_1 = ((row_lo_3_1 < M) ? act_scale[row_lo_3_1] : 0.0f);
                float sa_hi_3_1 = ((row_hi_3_1 < M) ? act_scale[row_hi_3_1] : 0.0f);
                float h_raw_3_0_1[8];
                h_raw_3_0_1[0] = accum_1[96] * sa_lo_3_1 * sw_1[0];
                h_raw_3_0_1[2] = accum_1[98] * sa_hi_3_1 * sw_1[0];
                h_raw_3_0_1[4] = accum_1[100] * sa_lo_3_1 * sw_1[2];
                h_raw_3_0_1[6] = accum_1[102] * sa_hi_3_1 * sw_1[2];
                h_raw_3_0_1[1] = accum_1[97] * sa_lo_3_1 * sw_1[1];
                h_raw_3_0_1[3] = accum_1[99] * sa_hi_3_1 * sw_1[1];
                h_raw_3_0_1[5] = accum_1[101] * sa_lo_3_1 * sw_1[3];
                h_raw_3_0_1[7] = accum_1[103] * sa_hi_3_1 * sw_1[3];
                uint32_t h_raw_3_0_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_0_1[_lp*2 + 0], h_raw_3_0_1[_lp*2+1 + 0]));
                    h_raw_3_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_3_0_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2;
                if (row_lo_3_1 < M) {
                    out[(row_lo_3_1 * 34816 + canonical_col_3_0_1) / 2] = h_raw_3_0_bf16_1[0];
                    out[(row_lo_3_1 * 34816 + 17408 + canonical_col_3_0_1) / 2] = h_raw_3_0_bf16_1[2];
                }
                if (row_hi_3_1 < M) {
                    out[(row_hi_3_1 * 34816 + canonical_col_3_0_1) / 2] = h_raw_3_0_bf16_1[1];
                    out[(row_hi_3_1 * 34816 + 17408 + canonical_col_3_0_1) / 2] = h_raw_3_0_bf16_1[3];
                }
                float h_raw_3_1_1[8];
                h_raw_3_1_1[0] = accum_1[104] * sa_lo_3_1 * sw_1[4];
                h_raw_3_1_1[2] = accum_1[106] * sa_hi_3_1 * sw_1[4];
                h_raw_3_1_1[4] = accum_1[108] * sa_lo_3_1 * sw_1[6];
                h_raw_3_1_1[6] = accum_1[110] * sa_hi_3_1 * sw_1[6];
                h_raw_3_1_1[1] = accum_1[105] * sa_lo_3_1 * sw_1[5];
                h_raw_3_1_1[3] = accum_1[107] * sa_hi_3_1 * sw_1[5];
                h_raw_3_1_1[5] = accum_1[109] * sa_lo_3_1 * sw_1[7];
                h_raw_3_1_1[7] = accum_1[111] * sa_hi_3_1 * sw_1[7];
                uint32_t h_raw_3_1_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_1_1[_lp*2 + 0], h_raw_3_1_1[_lp*2+1 + 0]));
                    h_raw_3_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_3_1_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8;
                if (row_lo_3_1 < M) {
                    out[(row_lo_3_1 * 34816 + canonical_col_3_1_1) / 2] = h_raw_3_1_bf16_1[0];
                    out[(row_lo_3_1 * 34816 + 17408 + canonical_col_3_1_1) / 2] = h_raw_3_1_bf16_1[2];
                }
                if (row_hi_3_1 < M) {
                    out[(row_hi_3_1 * 34816 + canonical_col_3_1_1) / 2] = h_raw_3_1_bf16_1[1];
                    out[(row_hi_3_1 * 34816 + 17408 + canonical_col_3_1_1) / 2] = h_raw_3_1_bf16_1[3];
                }
                float h_raw_3_2_1[8];
                h_raw_3_2_1[0] = accum_1[112] * sa_lo_3_1 * sw_1[8];
                h_raw_3_2_1[2] = accum_1[114] * sa_hi_3_1 * sw_1[8];
                h_raw_3_2_1[4] = accum_1[116] * sa_lo_3_1 * sw_1[10];
                h_raw_3_2_1[6] = accum_1[118] * sa_hi_3_1 * sw_1[10];
                h_raw_3_2_1[1] = accum_1[113] * sa_lo_3_1 * sw_1[9];
                h_raw_3_2_1[3] = accum_1[115] * sa_hi_3_1 * sw_1[9];
                h_raw_3_2_1[5] = accum_1[117] * sa_lo_3_1 * sw_1[11];
                h_raw_3_2_1[7] = accum_1[119] * sa_hi_3_1 * sw_1[11];
                uint32_t h_raw_3_2_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_2_1[_lp*2 + 0], h_raw_3_2_1[_lp*2+1 + 0]));
                    h_raw_3_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_3_2_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16;
                if (row_lo_3_1 < M) {
                    out[(row_lo_3_1 * 34816 + canonical_col_3_2_1) / 2] = h_raw_3_2_bf16_1[0];
                    out[(row_lo_3_1 * 34816 + 17408 + canonical_col_3_2_1) / 2] = h_raw_3_2_bf16_1[2];
                }
                if (row_hi_3_1 < M) {
                    out[(row_hi_3_1 * 34816 + canonical_col_3_2_1) / 2] = h_raw_3_2_bf16_1[1];
                    out[(row_hi_3_1 * 34816 + 17408 + canonical_col_3_2_1) / 2] = h_raw_3_2_bf16_1[3];
                }
                float h_raw_3_3_1[8];
                h_raw_3_3_1[0] = accum_1[120] * sa_lo_3_1 * sw_1[12];
                h_raw_3_3_1[2] = accum_1[122] * sa_hi_3_1 * sw_1[12];
                h_raw_3_3_1[4] = accum_1[124] * sa_lo_3_1 * sw_1[14];
                h_raw_3_3_1[6] = accum_1[126] * sa_hi_3_1 * sw_1[14];
                h_raw_3_3_1[1] = accum_1[121] * sa_lo_3_1 * sw_1[13];
                h_raw_3_3_1[3] = accum_1[123] * sa_hi_3_1 * sw_1[13];
                h_raw_3_3_1[5] = accum_1[125] * sa_lo_3_1 * sw_1[15];
                h_raw_3_3_1[7] = accum_1[127] * sa_hi_3_1 * sw_1[15];
                uint32_t h_raw_3_3_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_3_1[_lp*2 + 0], h_raw_3_3_1[_lp*2+1 + 0]));
                    h_raw_3_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                // PR5521-derived canonical BF16 gate/up store; original BF16 RN above.
                const int canonical_col_3_3_1 = tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24;
                if (row_lo_3_1 < M) {
                    out[(row_lo_3_1 * 34816 + canonical_col_3_3_1) / 2] = h_raw_3_3_bf16_1[0];
                    out[(row_lo_3_1 * 34816 + 17408 + canonical_col_3_3_1) / 2] = h_raw_3_3_bf16_1[2];
                }
                if (row_hi_3_1 < M) {
                    out[(row_hi_3_1 * 34816 + canonical_col_3_3_1) / 2] = h_raw_3_3_bf16_1[1];
                    out[(row_hi_3_1 * 34816 + 17408 + canonical_col_3_3_1) / 2] = h_raw_3_3_bf16_1[3];
                }
            }
        }
    }

    // Cleanup
}

}  // namespace pro6000d_pr5521_fp8_derived

#include "pr5521_abi.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
std::mutex config_mutex;
std::vector<int> configured_devices;

int fail(const char* what) { last_error = what; return 1; }

int encode_byte_tile(CUtensorMap* descriptor, const void* base,
                     uint64_t inner_bytes, uint64_t rows,
                     uint32_t box_inner, uint32_t box_rows) {
    uint64_t global_dim[2] = {inner_bytes, rows};
    uint64_t global_strides[1] = {inner_bytes};
    uint32_t box_dim[2] = {box_inner, box_rows};
    uint32_t element_strides[2] = {1, 1};
    CUresult result = cuTensorMapEncodeTiled(
        descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base),
        global_dim, global_strides, box_dim, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return result == CUDA_SUCCESS ? 0 : fail("cuTensorMapEncodeTiled failed");
}
}  // namespace

extern "C" PR5521_ABI int recovery_pr5521_abi_version(void) { return 1; }
extern "C" PR5521_ABI const char* recovery_pr5521_last_error(void) {
    return last_error.c_str();
}

extern "C" PR5521_ABI int recovery_pr5521_launch(
    void* out, const void* a, const void* b, const float* scale_a,
    const float* scale_b, int m, int n, int k, unsigned long long stream_bits) {
    last_error.clear();
    if (!out || !a || !b || !scale_a || !scale_b ||
        (m != 2741 && m != 16160) || n != 34816 || k != 5120)
        return fail("pointer or fixed shape guard failed");
    const void* pointers[] = {out, a, b, scale_a, scale_b};
    for (const void* ptr : pointers) {
        if ((reinterpret_cast<uintptr_t>(ptr) & 15u) != 0)
            return fail("device pointer is not 16-byte aligned");
    }
    int device = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return fail(cudaGetErrorString(status));
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) return fail(cudaGetErrorString(status));
    if (properties.major != 12 || properties.minor != 0)
        return fail("requires SM120");
    {
        std::lock_guard<std::mutex> lock(config_mutex);
        if (std::find(configured_devices.begin(), configured_devices.end(), device)
            == configured_devices.end()) {
            status = cudaFuncSetAttribute(
                pro6000d_pr5521_fp8_derived::kernel_h3_fc1_swiglu_gemm,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 99328);
            if (status != cudaSuccess) return fail(cudaGetErrorString(status));
            configured_devices.push_back(device);
        }
    }
    CUtensorMap a_map{}, b_map{}, unused_map{};
    if (encode_byte_tile(&a_map, a, 5120, m, 64, 128) != 0 ||
        encode_byte_tile(&b_map, b, 5120, 34816, 64, 256) != 0)
        return 1;
    const int num_m_tiles = (m + 127) / 128;
    const int total_tiles = num_m_tiles * 136;
    const int grid = std::min(total_tiles, properties.multiProcessorCount);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_bits);
    pro6000d_pr5521_fp8_derived::kernel_h3_fc1_swiglu_gemm
        <<<grid, 256, 99328, stream>>>(
            a_map, b_map, unused_map, unused_map,
            const_cast<float*>(scale_a), const_cast<float*>(scale_b),
            static_cast<unsigned int*>(out), m, num_m_tiles, total_tiles, 1.0f);
    status = cudaGetLastError();
    if (status != cudaSuccess) return fail(cudaGetErrorString(status));
    return 0;
}
