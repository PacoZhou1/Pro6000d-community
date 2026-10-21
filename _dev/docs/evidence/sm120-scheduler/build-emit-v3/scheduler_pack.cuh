#pragma once
__global__ void pack_gateup_fp8(
    cutlass::float_e4m3_t* __restrict__ out,
    float* __restrict__ out_scale,
    const cutlass::float_e4m3_t* __restrict__ paired,
    const float* __restrict__ native_scale, int m) {
  constexpr int H = 17408;
  constexpr int PackedN = 34816;
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= int64_t(m) * H) return;
  const int row = int(idx / H);
  const int col = int(idx % H);
  const int group = col / 128;
  const int within = col % 128;
  out[idx] = paired[int64_t(row) * PackedN + group * 256 + within];
  if (within == 0) {
    using S = cutlass::detail::Sm1xxBlockScaledOutputConfig<128>;
    auto layout = S::tile_atom_to_shape_SFD(cute::make_shape(m, PackedN, 5120, 1));
    const auto native_offset = layout(cute::make_coord(row, group * 256, 0));
    out_scale[int64_t(group) * m + row] = native_scale[native_offset];
  }
}

