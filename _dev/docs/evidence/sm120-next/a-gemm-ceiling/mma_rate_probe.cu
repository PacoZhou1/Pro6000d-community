// Raw tensor-core issue-rate probe for sm_120a (RTX PRO 6000 Blackwell).
//
// Measures register-resident throughput of three FP8 mma.sync forms:
//   plain : mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//   kind  : mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
//           (spelling used by CUTLASS SM120_16x8x32_TN)
//   bs    : mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32
//           .row.col.f32.e4m3.e4m3.f32.ue8m0, scale regs = 0x7f7f7f7f (2^0)
// Default forms: plain,bs. One JSON line per form on stdout; diagnostics on stderr.
//
// NOT YET RUN. The block-scale asm operand syntax was written from memory of
// CUTLASS mma_sm120.hpp and could not be assembled on the authoring machine
// (no nvcc). If ptxas rejects it, rebuild with -DBS_VARIANT=1 (PTX ISA spelling,
// no braces around scale operands) and record which variant assembled.
//
// Build:  nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -o mma_rate_probe \
//              mma_rate_probe.cu -lnvidia-ml
// Usage:  ./mma_rate_probe [--forms plain,bs] [--iters N] [--reps R] [--verify]
#include <cuda_runtime.h>
#include <dlfcn.h>
// nvml.h is absent from the fixed image: minimal ABI declarations, resolved with dlopen at run time.
typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;
#define NVML_SUCCESS 0
#define NVML_CLOCK_SM 1

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2);} } while (0)

#ifndef BS_VARIANT
#define BS_VARIANT 0
#endif

constexpr int kChains = 8;          // independent accumulator tiles per warp
constexpr int kWarps = 16;          // 4 warps per SM sub-partition
constexpr int kThreads = kWarps * 32;
constexpr uint32_t kScaleOne = 0x7f7f7f7fu;  // UE8M0 127 == 2^0 in every byte
constexpr double kFlopPerMma = 2.0 * 16 * 8 * 32;

enum Form { PLAIN = 0, KIND = 1, BS = 2 };
static const char* kFormName[] = {"plain", "kind", "bs"};
static const char* kFormInstr[] = {
  "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
  "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32",
  "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0"};

struct Frag { uint32_t a[4]; uint32_t b[2]; };

template <int F>
__device__ __forceinline__ void mma_op(float (&c)[4], const Frag& f) {
  if constexpr (F == PLAIN) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(f.a[0]), "r"(f.a[1]), "r"(f.a[2]), "r"(f.a[3]), "r"(f.b[0]), "r"(f.b[1]));
  } else if constexpr (F == KIND) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(f.a[0]), "r"(f.a[1]), "r"(f.a[2]), "r"(f.a[3]), "r"(f.b[0]), "r"(f.b[1]));
  } else {
    const uint32_t sfa = kScaleOne, sfb = kScaleOne;
    const uint16_t bid = 0, tid = 0;
#if BS_VARIANT == 0
    // CUTLASS-style spelling (braces around scale operands).
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(f.a[0]), "r"(f.a[1]), "r"(f.a[2]), "r"(f.a[3]), "r"(f.b[0]), "r"(f.b[1]),
          "r"(sfa), "h"(bid), "h"(tid), "r"(sfb), "h"(bid), "h"(tid));
#else
    // PTX ISA spelling (qualifier order per ISA, no braces around scale data).
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "%10, {%11,%12}, %13, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(f.a[0]), "r"(f.a[1]), "r"(f.a[2]), "r"(f.a[3]), "r"(f.b[0]), "r"(f.b[1]),
          "r"(sfa), "h"(bid), "h"(tid), "r"(sfb), "h"(bid), "h"(tid));
#endif
  }
}

// Deterministic finite E4M3 byte: never 0x7f/0xff (NaN); exponent field 2..5 keeps
// sums far from overflow over millions of accumulations.
__host__ __device__ __forceinline__ uint32_t e4m3_byte(uint32_t h) {
  uint32_t sign = (h >> 7) & 1u, exp = 2u + ((h >> 3) & 3u), man = h & 7u;
  return (sign << 7) | (exp << 3) | man;
}
__host__ __device__ __forceinline__ uint32_t hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}
__device__ __forceinline__ uint32_t pack4(uint32_t seed) {
  uint32_t h = hash32(seed), r = 0;
  for (int i = 0; i < 4; ++i) r |= e4m3_byte(hash32(h + i * 0x9e3779b9u)) << (8 * i);
  return r;
}
__device__ __forceinline__ Frag make_frag(uint32_t lane_seed) {
  Frag f;
  for (int i = 0; i < 4; ++i) f.a[i] = pack4(lane_seed * 8u + i);
  for (int i = 0; i < 2; ++i) f.b[i] = pack4(lane_seed * 8u + 4u + i);
  return f;
}

template <int F>
__global__ void __launch_bounds__(kThreads, 1)
rate_kernel(int iters, uint32_t seed, float* sink) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const Frag f = make_frag(seed + tid);
  float c[kChains][4];
  #pragma unroll
  for (int j = 0; j < kChains; ++j)
    #pragma unroll
    for (int i = 0; i < 4; ++i) c[j][i] = 0.f;
  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int j = 0; j < kChains; ++j) mma_op<F>(c[j], f);
  }
  float s = 0.f;
  #pragma unroll
  for (int j = 0; j < kChains; ++j)
    #pragma unroll
    for (int i = 0; i < 4; ++i) s += c[j][i];
  sink[tid] = s;  // unconditional store keeps the loop observable
}

// One warp, deterministic fragments, `iters` accumulations in one chain; dumps accumulators.
template <int F>
__global__ void verify_kernel(int iters, float* out) {
  const Frag f = make_frag(0x1234u + threadIdx.x);
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  for (int it = 0; it < iters; ++it) mma_op<F>(c, f);
  for (int i = 0; i < 4; ++i) out[threadIdx.x * 4 + i] = c[i];
}

template <int F> void launch_rate(int blocks, int iters, uint32_t seed, float* sink, size_t smem) {
  rate_kernel<F><<<blocks, kThreads, smem>>>(iters, seed, sink);
}
static void launch_form(int form, int blocks, int iters, uint32_t seed, float* sink, size_t smem) {
  if (form == PLAIN) launch_rate<PLAIN>(blocks, iters, seed, sink, smem);
  else if (form == KIND) launch_rate<KIND>(blocks, iters, seed, sink, smem);
  else launch_rate<BS>(blocks, iters, seed, sink, smem);
}
static void launch_verify(int form, int iters, float* out) {
  if (form == PLAIN) verify_kernel<PLAIN><<<1, 32>>>(iters, out);
  else if (form == KIND) verify_kernel<KIND><<<1, 32>>>(iters, out);
  else verify_kernel<BS><<<1, 32>>>(iters, out);
}
static cudaFuncAttributes form_attr(int form) {
  cudaFuncAttributes a{};
  if (form == PLAIN) CK(cudaFuncGetAttributes(&a, rate_kernel<PLAIN>));
  else if (form == KIND) CK(cudaFuncGetAttributes(&a, rate_kernel<KIND>));
  else CK(cudaFuncGetAttributes(&a, rate_kernel<BS>));
  return a;
}
static void set_smem_attr(int form, int bytes) {
  if (form == PLAIN) CK(cudaFuncSetAttribute(rate_kernel<PLAIN>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  else if (form == KIND) CK(cudaFuncSetAttribute(rate_kernel<KIND>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  else CK(cudaFuncSetAttribute(rate_kernel<BS>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

// ---- NVML SM clock sampling -------------------------------------------------
struct Nvml {
  nvmlDevice_t dev{};
  bool ok = false;
  void* lib = nullptr;
  nvmlReturn_t (*p_init)() = nullptr;
  nvmlReturn_t (*p_shutdown)() = nullptr;
  nvmlReturn_t (*p_bus)(const char*, nvmlDevice_t*) = nullptr;
  nvmlReturn_t (*p_clk)(nvmlDevice_t, int, unsigned*) = nullptr;
  bool init(int cuda_dev) {
    lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!lib) return false;
    p_init = (nvmlReturn_t(*)())dlsym(lib, "nvmlInit_v2");
    p_shutdown = (nvmlReturn_t(*)())dlsym(lib, "nvmlShutdown");
    p_bus = (nvmlReturn_t(*)(const char*, nvmlDevice_t*))dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2");
    p_clk = (nvmlReturn_t(*)(nvmlDevice_t, int, unsigned*))dlsym(lib, "nvmlDeviceGetClockInfo");
    if (!p_init || !p_shutdown || !p_bus || !p_clk) return false;
    if (p_init() != NVML_SUCCESS) return false;
    char bus[64];
    if (cudaDeviceGetPCIBusId(bus, sizeof bus, cuda_dev) != cudaSuccess) return false;
    if (p_bus(bus, &dev) != NVML_SUCCESS) return false;
    return ok = true;
  }
  int sm_mhz() const {
    unsigned v = 0;
    if (!ok || p_clk(dev, NVML_CLOCK_SM, &v) != NVML_SUCCESS) return -1;
    return (int)v;
  }
};

struct ClockSampler {
  const Nvml& nv;
  std::atomic<bool> stop{false};
  std::vector<int> samples;
  std::thread th;
  explicit ClockSampler(const Nvml& n) : nv(n) {}
  void start() {
    th = std::thread([this] {
      while (!stop.load()) {
        int v = nv.sm_mhz();
        if (v > 0) samples.push_back(v);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }
  void finish() { stop = true; if (th.joinable()) th.join(); }
};

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}

static int run_verify(const std::vector<int>& forms) {
  float* d; CK(cudaMalloc(&d, 128 * sizeof(float)));
  std::vector<std::vector<float>> res;
  for (int form : forms) {
    CK(cudaMemset(d, 0, 128 * sizeof(float)));
    launch_verify(form, 64, d);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> h(128); CK(cudaMemcpy(h.data(), d, 512, cudaMemcpyDeviceToHost));
    res.push_back(h);
  }
  bool nonzero = false, finite = true;
  for (float x : res[0]) { nonzero |= (x != 0.f); finite &= (x == x) && (x - x == 0.f); }
  bool identical = true;
  std::string detail;
  for (size_t k = 1; k < res.size(); ++k) {
    bool same = std::memcmp(res[0].data(), res[k].data(), 512) == 0;
    identical &= same;
    detail += std::string(detail.empty() ? "" : ",") + "\"" + kFormName[forms[k]] + "_vs_" + kFormName[forms[0]] + "\":" + (same ? "true" : "false");
  }
  printf("{\"mode\":\"verify\",\"forms\":%zu,\"iters\":64,\"identical\":%s,\"nonzero\":%s,\"finite\":%s,\"detail\":{%s}}\n",
         forms.size(), identical ? "true" : "false", nonzero ? "true" : "false", finite ? "true" : "false", detail.c_str());
  return identical && nonzero && finite ? 0 : 3;
}

int main(int argc, char** argv) {
  std::string forms_arg = "plain,bs";
  long long iters = 6000000;  // per warp per chain; ~1-3 s at expected rates
  int reps = 3, device = 0;
  bool verify = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1);} return argv[++i]; };
    if (a == "--forms") forms_arg = next();
    else if (a == "--iters") iters = atoll(next());
    else if (a == "--reps") reps = atoi(next());
    else if (a == "--device") device = atoi(next());
    else if (a == "--verify") verify = true;
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
  }
  std::vector<int> forms;
  for (size_t p = 0; p <= forms_arg.size();) {
    size_t q = forms_arg.find(',', p); if (q == std::string::npos) q = forms_arg.size();
    std::string t = forms_arg.substr(p, q - p);
    if (t == "plain") forms.push_back(PLAIN); else if (t == "kind") forms.push_back(KIND);
    else if (t == "bs") forms.push_back(BS); else if (!t.empty()) { fprintf(stderr, "bad form %s\n", t.c_str()); return 1; }
    p = q + 1;
  }
  CK(cudaSetDevice(device));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, device));
  if (prop.major != 12) fprintf(stderr, "warning: compute capability %d.%d, expected 12.x\n", prop.major, prop.minor);
  if (verify) return run_verify(forms);

  const int sms = prop.multiProcessorCount;
  Nvml nv; bool nvml_ok = nv.init(device);
  if (!nvml_ok) fprintf(stderr, "warning: NVML unavailable, clocks will be null\n");
  float* sink; CK(cudaMalloc(&sink, (size_t)sms * kThreads * sizeof(float)));
  // 64 KiB dynamic smem forces exactly one block per SM (SM has ~100 KiB).
  const int smem = 64 * 1024;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  for (int form : forms) {
    set_smem_attr(form, smem);
    cudaFuncAttributes attr = form_attr(form);
    launch_form(form, sms, (int)std::max(1LL, iters / 20), 1u, sink, smem);  // warm-up
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<double> tflops, ms_runs, mhz_runs;
    std::vector<int> before, after, during_all;
    for (int r = 0; r < reps; ++r) {
      ClockSampler sampler(nv);
      before.push_back(nv.sm_mhz());
      sampler.start();
      CK(cudaEventRecord(e0));
      launch_form(form, sms, (int)iters, 1u, sink, smem);
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      sampler.finish();
      after.push_back(nv.sm_mhz());
      CK(cudaGetLastError());
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      double mmas = (double)sms * kWarps * kChains * (double)iters;
      tflops.push_back(mmas * kFlopPerMma / (ms * 1e-3) / 1e12);
      ms_runs.push_back(ms);
      double mean = 0; for (int v : sampler.samples) mean += v;
      mhz_runs.push_back(sampler.samples.empty() ? -1.0 : mean / sampler.samples.size());
      during_all.insert(during_all.end(), sampler.samples.begin(), sampler.samples.end());
    }
    double ms_med = median(ms_runs), tf_med = median(tflops), mhz_med = median(mhz_runs);
    int dmax = during_all.empty() ? -1 : *std::max_element(during_all.begin(), during_all.end());
    int dmin = during_all.empty() ? -1 : *std::min_element(during_all.begin(), during_all.end());
    printf("{\"form\":\"%s\",\"instruction\":\"%s\",\"bs_variant\":%d,\"tflops\":%.2f,\"tflops_runs\":[",
           kFormName[form], kFormInstr[form], BS_VARIANT, tf_med);
    for (size_t i = 0; i < tflops.size(); ++i) printf("%s%.2f", i ? "," : "", tflops[i]);
    printf("],\"elapsed_ms\":%.3f,\"elapsed_ms_runs\":[", ms_med);
    for (size_t i = 0; i < ms_runs.size(); ++i) printf("%s%.3f", i ? "," : "", ms_runs[i]);
    printf("],\"iterations\":%lld,\"chains\":%d,\"warps_per_sm\":%d,\"sm_count\":%d,\"reps\":%d,", iters, kChains, kWarps, sms, reps);
    printf("\"regs_per_thread\":%d,\"local_bytes\":%zu,\"static_smem\":%zu,", attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes);
    printf("\"sm_clock_mhz_before\":%d,\"sm_clock_mhz_after\":%d,", before.empty() ? -1 : before.front(), after.empty() ? -1 : after.back());
    printf("\"sm_clock_mhz_during_mean\":%.1f,\"sm_clock_mhz_during_min\":%d,\"sm_clock_mhz_during_max\":%d,\"sm_clock_samples\":%zu,",
           mhz_med, dmin, dmax, during_all.size());
    // Average of before/after/during means (spec); during mean is the meaningful one.
    double avg = 0; int n = 0;
    for (int v : before) if (v > 0) { avg += v; ++n; }
    for (int v : after) if (v > 0) { avg += v; ++n; }
    for (double v : mhz_runs) if (v > 0) { avg += v; ++n; }
    printf("\"sm_clock_mhz_avg\":%.1f,", n ? avg / n : -1.0);
    // cycles per mma per SM-subpartition-equivalent: total SM cycles / (mma per SM)
    double cyc_per_sm_mma = mhz_med > 0 ? (ms_med * 1e-3 * mhz_med * 1e6) / ((double)kWarps * kChains * iters) : -1.0;
    printf("\"sm_cycles_per_mma_per_sm\":%.4f,\"nvml\":%s,\"device\":\"%s\",\"cc\":\"%d.%d\"}\n",
           cyc_per_sm_mma, nvml_ok ? "true" : "false", prop.name, prop.major, prop.minor);
    fflush(stdout);
  }
  if (nvml_ok) nv.p_shutdown();
  return 0;
}
