// Host-only CuTe accumulator-ownership probe.  No CUDA runtime calls or kernel launches.
// Build/run instructions and interpretation: see preflight-review.md.
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm120.hpp"
#include "cutlass/float8.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
constexpr int kM = 64;
constexpr int kN = 256;
constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kSfN = 128;

struct Result {
  const char* name;
  int atom_m;
  int atom_n;
  int thread_count = 0;
  int out_of_bounds = 0;
  int duplicate_elements = 0;
  int missing_elements = 0;
  int gate_up_thread_mismatch = 0;
  int fragment_offset_mismatch = 0;
  int fragment_odd_threads = 0;
  int sf128_incomplete = 0;
  int sf128_xor4_pair_failures = 0;
  std::map<int, int> warp_mask_histogram;
  std::array<int, 2> row0_warp_masks{};
  std::array<std::array<int, kWarps>, 2> row0_warp_counts{};
  int first_gate_up_m = -1;
  int first_gate_up_n = -1;
  int first_fragment_thread = -1;
  int first_fragment_index = -1;
  std::array<std::vector<std::pair<int, int>>, kThreads> thread_coordinates;
};

template <int AtomM, int AtomN>
Result inspect(const char* name) {
  using namespace cute;
  // Exactly the fixed SM120 blockwise builder's M64N256K128 TiledMMA recipe:
  // rr_op_selector_sm120 + AtomLayoutMNK + Tile<min(M,128),min(N,32),32>.
  auto mma = make_tiled_mma(
      rr_op_selector_sm120<cutlass::float_e4m3_t,
                            cutlass::float_e4m3_t, float>(),
      Layout<Shape<Int<AtomM>, Int<AtomN>, _1>>{},
      Tile<_64, _32, _32>{});
  auto identity = make_identity_tensor(make_shape(Int<kM>{}, Int<kN>{}));

  Result result{name, AtomM, AtomN};
  result.thread_count = int(size(mma.get_thr_layout_vmnk()));
  if (result.thread_count != kThreads) {
    throw std::runtime_error("expected exactly eight 32-lane MMA warps");
  }
  std::vector<int> owner(kM * kN, -1);
  std::vector<int> counts(kM * kN, 0);
  std::array<std::array<std::array<std::array<int, kSfN>, kWarps>, 2>, kM>
      warp_hits{};

  for (int tid = 0; tid < result.thread_count; ++tid) {
    auto part = mma.get_slice(tid).partition_C(identity);
    for (int i = 0; i < int(size(part)); ++i) {
      auto coord = part(i);
      int m = int(get<0>(coord));
      int n = int(get<1>(coord));
      if (m < 0 || m >= kM || n < 0 || n >= kN) {
        ++result.out_of_bounds;
        continue;
      }
      int index = m * kN + n;
      if (counts[index]++ == 0) owner[index] = tid;
      result.thread_coordinates[tid].emplace_back(m, n);
      ++warp_hits[m][n / kSfN][tid / 32][n % kSfN];
    }
  }

  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      int count = counts[m * kN + n];
      if (count == 0) ++result.missing_elements;
      if (count > 1) result.duplicate_elements += count - 1;
    }
    for (int n = 0; n < kSfN; ++n) {
      int gate = owner[m * kN + n];
      int up = owner[m * kN + n + kSfN];
      if (gate < 0 || gate != up) {
        ++result.gate_up_thread_mismatch;
        if (result.first_gate_up_m < 0) {
          result.first_gate_up_m = m;
          result.first_gate_up_n = n;
        }
      }
    }
  }

  // v6 visitor pairs flattened accumulator fragments i and i + size/2.
  for (int tid = 0; tid < result.thread_count; ++tid) {
    const auto& elems = result.thread_coordinates[tid];
    if (elems.size() % 2) {
      ++result.fragment_odd_threads;
      continue;
    }
    const size_t half = elems.size() / 2;
    for (size_t i = 0; i < half; ++i) {
      if (elems[i].first != elems[i + half].first ||
          elems[i].second + kSfN != elems[i + half].second) {
        ++result.fragment_offset_mismatch;
        if (result.first_fragment_thread < 0) {
          result.first_fragment_thread = tid;
          result.first_fragment_index = int(i);
        }
      }
    }
  }

  // Every row and each 128-column SF group must have 128 unique values.
  // The v6 SF128 visitor specifically exchanges warp w with w XOR 4;
  // test whether any such pair alone covers a complete row/SF group.
  for (int m = 0; m < kM; ++m) {
    for (int group = 0; group < 2; ++group) {
      int mask = 0;
      bool complete = true;
      for (int warp = 0; warp < kWarps; ++warp) {
        int warp_count = 0;
        for (int n = 0; n < kSfN; ++n) {
          warp_count += warp_hits[m][group][warp][n];
        }
        if (warp_count) mask |= 1 << warp;
        if (m == 0) result.row0_warp_counts[group][warp] = warp_count;
      }
      if (m == 0) result.row0_warp_masks[group] = mask;
      ++result.warp_mask_histogram[mask];
      for (int n = 0; n < kSfN; ++n) {
        int total = 0;
        for (int warp = 0; warp < kWarps; ++warp) {
          total += warp_hits[m][group][warp][n];
        }
        if (total != 1) complete = false;
      }
      if (!complete) ++result.sf128_incomplete;
      bool xor4_pair_covers = false;
      for (int warp = 0; warp < 4; ++warp) {
        bool pair_complete = true;
        for (int n = 0; n < kSfN; ++n) {
          if (warp_hits[m][group][warp][n] +
                  warp_hits[m][group][warp + 4][n] != 1) {
            pair_complete = false;
            break;
          }
        }
        xor4_pair_covers |= pair_complete;
      }
      if (!xor4_pair_covers) ++result.sf128_xor4_pair_failures;
    }
  }
  return result;
}

void print_result(const Result& r, bool full) {
  const bool unique = r.out_of_bounds == 0 && r.duplicate_elements == 0 &&
                      r.missing_elements == 0;
  const bool pair = r.gate_up_thread_mismatch == 0 &&
                    r.fragment_offset_mismatch == 0 &&
                    r.fragment_odd_threads == 0;
  std::cout << "{\"variant\":\"" << r.name << "\",\"atom_layout_mn\":["
            << r.atom_m << ',' << r.atom_n << "],\"threads\":"
            << r.thread_count << ",\"tile_mnk\":[64,256,128],"
            << "\"permutation_mnk\":[64,32,32],"
            << "\"out_of_bounds\":" << r.out_of_bounds
            << ",\"duplicate_elements\":" << r.duplicate_elements
            << ",\"missing_elements\":" << r.missing_elements
            << ",\"gate_up_thread_mismatch\":"
            << r.gate_up_thread_mismatch
            << ",\"fragment_offset_mismatch\":"
            << r.fragment_offset_mismatch
            << ",\"fragment_odd_threads\":" << r.fragment_odd_threads
            << ",\"sf128_incomplete_groups\":" << r.sf128_incomplete
            << ",\"sf128_xor4_pair_failures\":"
            << r.sf128_xor4_pair_failures
            << ",\"first_gate_up_mn\":[" << r.first_gate_up_m << ','
            << r.first_gate_up_n << "],\"first_fragment_thread_index\":["
            << r.first_fragment_thread << ',' << r.first_fragment_index
            << "],\"row0_warp_masks\":[" << r.row0_warp_masks[0] << ','
            << r.row0_warp_masks[1] << "],\"row0_warp_counts\":[";
  for (int group = 0; group < 2; ++group) {
    if (group) std::cout << ',';
    std::cout << '[';
    for (int warp = 0; warp < kWarps; ++warp) {
      if (warp) std::cout << ',';
      std::cout << r.row0_warp_counts[group][warp];
    }
    std::cout << ']';
  }
  std::cout << "],\"warp_mask_histogram\":{  ";
  bool first = true;
  for (const auto& [mask, count] : r.warp_mask_histogram) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << '"' << mask << "\":" << count;
  }
  std::cout << "},\"unique_tile_ownership\":"
            << (unique ? "true" : "false")
            << ",\"gate_up_fragment_pairing\":"
            << (pair ? "true" : "false")
            << ",\"compatible_with_v6_xor4_sf128_geometry\":"
            << (r.sf128_incomplete == 0 && r.sf128_xor4_pair_failures == 0
                    ? "true" : "false");
  if (full) {
    std::cout << ",\"thread_coordinates\":[";
    for (int tid = 0; tid < r.thread_count; ++tid) {
      if (tid) std::cout << ',';
      std::cout << "{\"tid\":" << tid << ",\"warp\":" << tid / 32
                << ",\"lane\":" << tid % 32 << ",\"mn\":[";
      const auto& coordinates = r.thread_coordinates[tid];
      for (size_t i = 0; i < coordinates.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << '[' << coordinates[i].first << ','
                  << coordinates[i].second << ']';
      }
      std::cout << "]}";
    }
    std::cout << ']';
  }
  std::cout << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool full = argc == 2 && std::strcmp(argv[1], "--full") == 0;
    if (argc > 2 || (argc == 2 && !full)) {
      throw std::runtime_error("usage: ownership_probe [--full]");
    }
    print_result(inspect<4, 2>("baseline_4x2"), full);
    print_result(inspect<2, 4>("candidate_2x4"), full);
    // Mapping failures are represented in JSON rather than by process failure.
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ownership_probe error: " << e.what() << '\n';
    return 1;
  }
}
