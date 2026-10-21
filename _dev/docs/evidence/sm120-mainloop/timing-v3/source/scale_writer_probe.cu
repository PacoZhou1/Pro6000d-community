// CPU-only reconstruction of the *actual row-scale destination partition*.
// Compile once against each generated variant's standalone_dispatch.cuh and
// shadow CUTLASS headers. No CUDA runtime calls, device allocation or kernel.
// See scale-writer-probe.md for command, source/predicate provenance and limits.
#include "standalone_dispatch.cuh"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

#include <cstddef>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#ifndef PROBE_NAMESPACE
#error "pass -DPROBE_NAMESPACE=<variant private namespace>"
#endif
#ifndef PROBE_LABEL
#error "pass -DPROBE_LABEL=\"control42|warp24x2|warp24n64\""
#endif
#ifndef PROBE_WRITER_MAX_WARP
#error "pass -DPROBE_WRITER_MAX_WARP=4 for v2 or 2 for v3"
#endif

namespace {
constexpr int kM = 64;
constexpr int kN = 256;
constexpr int kK = 128;
constexpr int kSf = 128;
constexpr int kThreads = 256;

using Gemm = typename PROBE_NAMESPACE::sm120_blockwise_fp8_config_paired_quant<
    cutlass::float_e4m3_t>::Gemm;
using Mma = typename Gemm::CollectiveMainloop::TiledMma;
using Epi = typename Gemm::CollectiveEpilogue;

int run() {
  using namespace cute;
  static_assert(size<0>(typename Epi::CtaTileMNK{}) == kM);
  static_assert(size<1>(typename Epi::CtaTileMNK{}) == kN);
  static_assert(size<0>(typename Epi::EpilogueTile{}) == kM);
  static_assert(size<1>(typename Epi::EpilogueTile{}) == kN);
  static_assert(PROBE_WRITER_MAX_WARP == 2 || PROBE_WRITER_MAX_WARP == 4);

  // Matches Sm120BlockScaleFactorRowStore::get_consumer_store_callbacks:
  // mSFD -> gSFD -> sm90_partition_for_epilogue<RefSrc=true>.
  using Sfd = cutlass::detail::Sm1xxBlockScaledOutputConfig<kSf>;
  auto problem = make_shape(kM, kN, kK, 1);
  auto sfd_layout = Sfd::tile_atom_to_shape_SFD(problem);
  std::vector<float> storage(size_t(cosize(sfd_layout)), 0.0f);
  auto mSFD = make_tensor(make_gmem_ptr(storage.data()), sfd_layout);
  auto gSFD = local_tile(mSFD, typename Epi::EpilogueTile{},
                         make_coord(_, _, 0));
  Mma mma;
  auto copy_c = make_tiled_copy_C_atom(typename Epi::CopyAtomC{}, mma);
  auto copy_ref = make_tiled_copy_S(
      Copy_Atom<typename Epi::CopyOpR2S, typename Epi::ElementD>{}, copy_c);

  // A full CTA tile has no M/N residue. Each logical (row, N/128 group)
  // must map to one physical SFD address, written exactly once by the
  // visitor's writer predicate after copy_aligned's AutoFilter.
  std::map<std::ptrdiff_t, std::pair<int, int>> expected;
  int expected_aliases = 0;
  for (int row = 0; row < kM; ++row) {
    for (int group = 0; group < kN / kSf; ++group) {
      auto address = &mSFD(row, group * kSf, 0) - storage.data();
      auto [it, inserted] = expected.emplace(address, std::make_pair(row, group));
      if (!inserted) ++expected_aliases;
    }
  }

  std::map<std::ptrdiff_t, int> writes;
  std::map<std::ptrdiff_t, std::set<int>> writers;
  int attempted_threads = 0;
  int filtered_slots = 0;
  for (int tid = 0; tid < kThreads; ++tid) {
    const int warp = tid / 32;
    const int lane = tid % 32;
    // The exact full-tile predicate in the pinned row-store visitor:
    // thread_idx_in_quad == 0 and warp_idx < 4 (v2) or < 2 (v3).
    if (lane % 4 != 0 || warp >= PROBE_WRITER_MAX_WARP) continue;
    ++attempted_threads;
    auto tCgSFD = cutlass::epilogue::fusion::detail::
        sm90_partition_for_epilogue<true>(
            gSFD, typename Epi::EpilogueTile{}, copy_ref, tid);
    // The runtime copy_aligned uses AutoFilter on static tensor shapes;
    // filter_zeros removes its zero-stride broadcast slots.
    auto destination = tCgSFD(_, _, _, 0, 0, 0, 0);
    auto filtered = filter_zeros(destination);
    for (int i = 0; i < int(size(filtered)); ++i) {
      auto address = &filtered(i) - storage.data();
      ++writes[address];
      writers[address].insert(tid);
      ++filtered_slots;
    }
  }

  int missing = 0, duplicate = 0, unexpected = 0;
  std::ptrdiff_t first_missing = -1, first_duplicate = -1,
                 first_unexpected = -1;
  for (auto const& [address, coordinate] : expected) {
    auto it = writes.find(address);
    int count = it == writes.end() ? 0 : it->second;
    if (count == 0) {
      ++missing;
      if (first_missing < 0) first_missing = address;
    } else if (count > 1) {
      duplicate += count - 1;
      if (first_duplicate < 0) first_duplicate = address;
    }
  }
  for (auto const& [address, count] : writes) {
    if (!expected.count(address)) {
      unexpected += count;
      if (first_unexpected < 0) first_unexpected = address;
    }
  }
  std::cout << "{\"label\":\"" << PROBE_LABEL << "\","
            << "\"writer_max_warp_exclusive\":" << PROBE_WRITER_MAX_WARP
            << ",\"cta_mn\":[64,256],\"sf_vec_n\":128,"
            << "\"expected_scale_addresses\":" << expected.size()
            << ",\"expected_layout_aliases\":" << expected_aliases
            << ",\"candidate_writer_threads\":" << attempted_threads
            << ",\"filtered_store_slots\":" << filtered_slots
            << ",\"observed_scale_addresses\":" << writes.size()
            << ",\"missing_addresses\":" << missing
            << ",\"duplicate_writes\":" << duplicate
            << ",\"unexpected_writes\":" << unexpected
            << ",\"first_missing_offset\":" << first_missing
            << ",\"first_duplicate_offset\":" << first_duplicate
            << ",\"first_unexpected_offset\":" << first_unexpected;
  if (first_duplicate >= 0) {
    std::cout << ",\"first_duplicate_tids\":[";
    bool first = true;
    for (int tid : writers[first_duplicate]) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << tid;
    }
    std::cout << ']';
  }
  const bool unique = expected.size() == 128 && expected_aliases == 0 &&
                      missing == 0 && duplicate == 0 && unexpected == 0;
  std::cout << ",\"full_tile_scale_store_unique\":"
            << (unique ? "true" : "false") << "}\n";
  // A disproved writer predicate is data, not a process failure.
  return 0;
}
}  // namespace

int main() {
  try {
    return run();
  } catch (std::exception const& e) {
    std::cerr << "scale_writer_probe error: " << e.what() << '\n';
    return 1;
  }
}
