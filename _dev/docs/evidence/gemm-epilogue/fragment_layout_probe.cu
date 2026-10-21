// CPU-only CuTe coordinate probe. Compile against one generated N256 BF16
// dispatch directory; running the executable does not initialize CUDA.
// It reports whether a thread owns both n and n+128 accumulator coordinates
// and whether those coordinates differ by half its flattened C fragment.
#include "standalone_dispatch.cuh"
#include <cstdio>
#include <initializer_list>

int main() {
  using Gemm = typename vllm::sm120_blockwise_fp8_config_paired_n256<cutlass::bfloat16_t>::Gemm;
  using TiledMma = typename Gemm::CollectiveMainloop::TiledMma;
  TiledMma mma;
  auto coords = cute::make_identity_tensor(
      cute::make_shape(cute::Int<64>{}, cute::Int<256>{}));
  int owner[64][256];
  int flat_index[64][256];
  int duplicate = 0;
  int missing = 0;
  for (auto &row : owner) for (int &item : row) item = -1;
  for (auto &row : flat_index) for (int &item : row) item = -1;
  for (int tid = 0; tid < int(cute::size(mma)); ++tid) {
    auto fragment = mma.get_thread_slice(tid).partition_C(coords);
    for (int i = 0; i < int(cute::size(fragment)); ++i) {
      auto position = fragment(i);
      int row = int(cute::get<0>(position));
      int col = int(cute::get<1>(position));
      if (row < 0 || row >= 64 || col < 0 || col >= 256) continue;
      if (owner[row][col] != -1) ++duplicate;
      owner[row][col] = tid;
      flat_index[row][col] = i;
    }
  }
  int same_thread = 0;
  int half_offset = 0;
  int checked = 0;
  for (int row = 0; row < 64; ++row) {
    for (int col = 0; col < 128; ++col) {
      if (owner[row][col] < 0 || owner[row][col + 128] < 0) {
        ++missing;
        continue;
      }
      ++checked;
      if (owner[row][col] == owner[row][col + 128]) {
        ++same_thread;
        auto fragment = mma.get_thread_slice(owner[row][col]).partition_C(coords);
        if (flat_index[row][col + 128] - flat_index[row][col] ==
            int(cute::size(fragment)) / 2) ++half_offset;
      }
    }
  }
  std::printf("{\"mma_threads\":%d,\"duplicate_coordinates\":%d,"
              "\"missing_pairs\":%d,\"checked_pairs\":%d,"
              "\"same_thread_pairs\":%d,\"flat_half_offset_pairs\":%d,"
              "\"row0_examples\":[",
              int(cute::size(mma)), duplicate, missing, checked, same_thread, half_offset);
  for (int col : {0, 1, 8, 64, 127}) {
    std::printf("%s{\"n\":%d,\"gate_thread\":%d,\"up_thread\":%d,"
                "\"gate_flat\":%d,\"up_flat\":%d}",
                col ? "," : "", col, owner[0][col], owner[0][col + 128],
                flat_index[0][col], flat_index[0][col + 128]);
  }
  std::printf("],\"row_group_warp_masks\":[");
  bool first = true;
  for (int row : {0, 1, 16, 32, 48}) {
    for (int group = 0; group < 2; ++group) {
      unsigned mask = 0;
      for (int col = group * 128; col < (group + 1) * 128; ++col) {
        if (owner[row][col] >= 0) mask |= 1u << (owner[row][col] / 32);
      }
      std::printf("%s{\"row\":%d,\"group\":%d,\"mask\":%u}",
                  first ? "" : ",", row, group, mask);
      first = false;
    }
  }
  std::puts("]}");
  return duplicate == 0 && missing == 0 ? 0 : 1;
}
