#pragma once

// Experimental CUTLASS EVT for paired [gate128, up128] N tiles.  This header
// must be built with the exact generated SF128 visitor patch from
// build_candidate.py.  It does not modify the installed CUTLASS headers.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp"

namespace cutlass::epilogue::fusion {

struct PairedGateUpFp8 : FusionOperation {
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementCompute = float;
  using ElementBlockScaleFactor = float;
  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  static constexpr int SFVecSize = 128;
  static constexpr bool IsBlockScaleSupported = true;
  using GmemLayoutTagScalefactor = cutlass::layout::RowMajor;
};

// The base store visitor performs the per-128 reduction and FP8 conversion.
// Before its reduction, both corresponding FP32 GEMM accumulators are rounded
// to BF16, then the exact upstream FP32 SiLU/product is evaluated.  The
// products replace only the gate half; the up half is zeroed.  A following
// pack kernel removes the zero half and maps the native CUTLASS scale layout
// to the vLLM [M,136] column-major scale layout.
template <class EpilogueTile, class CtaTileMNK, int FragmentSize>
struct PairedGateUpStore :
    Sm120BlockScaleFactorRowStore<
        128, EpilogueTile, CtaTileMNK, FragmentSize,
        cutlass::float_e4m3_t, float, float> {
  using Base = Sm120BlockScaleFactorRowStore<
      128, EpilogueTile, CtaTileMNK, FragmentSize,
      cutlass::float_e4m3_t, float, float>;
  using Base::Base;
  using typename Base::Arguments;
  using typename Base::Params;
  using typename Base::SharedStorage;

  template <class BaseCallbacks>
  struct ConsumerStoreCallbacks : BaseCallbacks {
    CUTLASS_DEVICE explicit ConsumerStoreCallbacks(BaseCallbacks&& base)
        : BaseCallbacks(cute::move(base)) {}

    template <class SmemTensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(
        SmemTensor&& smem_buffer, SyncFn const& sync_fn,
        int epi_m, int epi_n, bool is_last_iteration,
        VTensor visit_results) {
      static_assert(cute::size<1>(EpilogueTile{}) == 256,
                    "paired gate/up requires a 256-column epilogue tile");
      static_assert(cute::size<2>(visit_results) % 2 == 0,
                    "paired accumulator fragments must split evenly");
      constexpr int frags_per_half = cute::size<2>(visit_results) / 2;

      // CUTLASS's 16x8 SM120 accumulator mapping assigns corresponding
      // gate/up N fragments to the same thread at offset frags_per_half.
      // The numerical probe must verify this invariant against canonical
      // GEMM output for every supported M before the op is eligible for use.
      CUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < cute::size<1>(visit_results); ++row) {
        CUTLASS_PRAGMA_UNROLL
        for (int frag = 0; frag < frags_per_half; ++frag) {
          auto gate = visit_results(0, row, frag);
          auto up = visit_results(0, row, frag + frags_per_half);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < FragmentSize; ++j) {
            const float g = static_cast<float>(cutlass::bfloat16_t(gate[j]));
            const float u = static_cast<float>(cutlass::bfloat16_t(up[j]));
            const float sigmoid = 1.0f / (1.0f + expf(-g));
            const float product = (g * sigmoid) * u;
            visit_results(0, row, frag)[j] = product;
            visit_results(0, row, frag + frags_per_half)[j] = 0.0f;
          }
        }
      }
      BaseCallbacks::reduce(cute::forward<SmemTensor>(smem_buffer), sync_fn,
                            epi_m, epi_n, is_last_iteration, visit_results);
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      ConsumerStoreArgs<Args...> const& args) {
    auto base = Base::template get_consumer_store_callbacks<ReferenceSrc>(args);
    return ConsumerStoreCallbacks<decltype(base)>(cute::move(base));
  }
};

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC,
          bool DelayTmaStore, class CtaTileMNK, class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm120TmaWarpSpecialized<
        StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    PairedGateUpFp8, CtaTileMNK, EpilogueTile>
    : Sm90EVT<PairedGateUpStore<EpilogueTile, CtaTileMNK, FragmentSize>,
              Sm90AccFetch> {
  using Impl = Sm90EVT<
      PairedGateUpStore<EpilogueTile, CtaTileMNK, FragmentSize>,
      Sm90AccFetch>;
  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion
