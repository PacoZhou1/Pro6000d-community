// CPU-only probe of the epilogue's register-to-shared copy retile. This is
// closer to PairedGateUpStore::reduce than the MMA partition_C probe because
// the visitor receives fragments after thread_r2s.retile_S(accumulators).
#include "standalone_dispatch.cuh"
#include <cstdio>
#include <initializer_list>
#include <type_traits>

template <class Epi, class Mma>
auto make_epilogue_r2s(Mma const& mma) {
  auto copy_c = cute::make_tiled_copy_C_atom(typename Epi::CopyAtomC{}, mma);
  if constexpr (std::is_void_v<typename Epi::CopyOpR2R>) {
    return cute::make_tiled_copy_S(
        cute::Copy_Atom<typename Epi::CopyOpR2S, cutlass::float_e4m3_t>{}, copy_c);
  } else {
    auto r2r = cute::make_tiled_copy_S(
        cute::Copy_Atom<typename Epi::CopyOpR2R, float>{}, copy_c);
    return cute::make_tiled_copy_D(
        cute::Copy_Atom<typename Epi::CopyOpR2S, cutlass::float_e4m3_t>{}, r2r);
  }
}

int main() {
  using Gemm = typename vllm::sm120_blockwise_fp8_config_paired_quant<cutlass::float_e4m3_t>::Gemm;
  using Mma = typename Gemm::CollectiveMainloop::TiledMma;
  using Epi = typename Gemm::CollectiveEpilogue;
  Mma mma;
  auto copy_r2s = make_epilogue_r2s<Epi>(mma);
  auto coords = cute::make_identity_tensor(
      cute::make_shape(cute::Int<64>{}, cute::Int<256>{}));
  int owner[64][256];
  int retiled_index[64][256];
  for (auto &row : owner) for (int &item : row) item = -1;
  for (auto &row : retiled_index) for (int &item : row) item = -1;
  int duplicates = 0;
  for (int tid = 0; tid < int(cute::size(mma)); ++tid) {
    auto mma_fragment = mma.get_thread_slice(tid).partition_C(coords);
    auto epi_fragment = copy_r2s.get_slice(tid).retile_S(mma_fragment);
    for (int i = 0; i < int(cute::size(epi_fragment)); ++i) {
      auto position = epi_fragment(i);
      int row = int(cute::get<0>(position));
      int col = int(cute::get<1>(position));
      if (row < 0 || row >= 64 || col < 0 || col >= 256) continue;
      if (owner[row][col] != -1) ++duplicates;
      owner[row][col] = tid;
      retiled_index[row][col] = i;
    }
  }
  int checked = 0, missing = 0, same_thread = 0, half_offset = 0;
  for (int row = 0; row < 64; ++row) for (int col = 0; col < 128; ++col) {
    int gate_tid = owner[row][col], up_tid = owner[row][col + 128];
    if (gate_tid < 0 || up_tid < 0) { ++missing; continue; }
    ++checked;
    if (gate_tid == up_tid) {
      ++same_thread;
      auto mma_fragment = mma.get_thread_slice(gate_tid).partition_C(coords);
      auto epi_fragment = copy_r2s.get_slice(gate_tid).retile_S(mma_fragment);
      if (retiled_index[row][col + 128] - retiled_index[row][col] ==
          int(cute::size(epi_fragment)) / 2) ++half_offset;
    }
  }
  std::printf("{\"checked_pairs\":%d,\"missing_pairs\":%d,"
              "\"duplicate_coordinates\":%d,\"same_thread_pairs\":%d,"
              "\"flat_half_offset_pairs\":%d,\"row_group_warp_masks\":[",
              checked, missing, duplicates, same_thread, half_offset);
  bool first = true;
  for (int row : {0, 1, 16, 32, 48}) for (int group = 0; group < 2; ++group) {
    unsigned mask = 0;
    for (int col = group * 128; col < (group + 1) * 128; ++col)
      if (owner[row][col] >= 0) mask |= 1u << (owner[row][col] / 32);
    std::printf("%s{\"row\":%d,\"group\":%d,\"mask\":%u}",
                first ? "" : ",", row, group, mask);
    first = false;
  }
  std::puts("]}");
  return duplicates == 0 && missing == 0 ? 0 : 1;
}
