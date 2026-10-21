// CUTLASS 4.5.0 example 79c adapted to MXFP8 x MXFP8 with BF16 output.
// One GEMM only; no activation quantization or weight packing in timed region.
#include "mxfp8_abi.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/float8.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace pro6000d_recovery_mxfp8_v1 {
using namespace cute;
using EA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using EB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ED = cutlass::bfloat16_t;
using EC = cutlass::bfloat16_t;
using Arch = cutlass::arch::Sm120;
using Op = cutlass::arch::OpClassBlockScaledTensorOp;
using Tile = Shape<_128,_128,_128>;
using Cluster = Shape<_1,_1,_1>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch,Op,Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,
    float,float,EC,cutlass::layout::RowMajor,8,
    ED,cutlass::layout::RowMajor,8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch,Op,EA,cutlass::layout::RowMajor,16,
    EB,cutlass::layout::ColumnMajor,16,float,Tile,Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi,void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using StrideA = typename Kernel::StrideA;
using StrideB = typename Kernel::StrideB;
using StrideC = typename Kernel::StrideC;
using StrideD = typename Kernel::StrideD;
using SFConfig = typename Main::Sm1xxBlkScaledConfig;
thread_local char error[256] = {};
void set_error(const char* msg) {
  std::strncpy(error,msg,sizeof(error)-1); error[sizeof(error)-1]=0;
}
void validate(int m,int n,int k) {
  if(m<1 || m>32768 || n!=34816 || k!=5120) throw std::runtime_error("only gate_up N34816 K5120 M1..32768");
}
auto sfa_layout(int m,int n,int k) {
  return SFConfig::tile_atom_to_shape_SFA(make_shape(m,n,k,1));
}
auto sfb_layout(int m,int n,int k) {
  return SFConfig::tile_atom_to_shape_SFB(make_shape(m,n,k,1));
}
Gemm::Arguments args(void* out,const void* a,const void* b,
                     const void* sa,const void* sb,int m,int n,int k) {
  auto as = cutlass::make_cute_packed_stride(StrideA{}, {m,k,1});
  auto bs = cutlass::make_cute_packed_stride(StrideB{}, {n,k,1});
  auto cs = cutlass::make_cute_packed_stride(StrideC{}, {m,n,1});
  auto ds = cutlass::make_cute_packed_stride(StrideD{}, {m,n,1});
  return {cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,1},
    {reinterpret_cast<EA::DataType const*>(a),as,
     reinterpret_cast<EB::DataType const*>(b),bs,
     reinterpret_cast<EA::ScaleFactorType const*>(sa),sfa_layout(m,n,k),
     reinterpret_cast<EB::ScaleFactorType const*>(sb),sfb_layout(m,n,k)},
    {{1.0f,0.0f},reinterpret_cast<EC const*>(out),cs,
     reinterpret_cast<ED*>(out),ds}};
}
}
extern "C" {
int recovery_mxfp8_abi_version(){return 1;}
const char* recovery_mxfp8_last_error(){return pro6000d_recovery_mxfp8_v1::error;}
size_t recovery_mxfp8_scale_a_bytes(int m,int n,int k) try {
  using namespace pro6000d_recovery_mxfp8_v1;validate(m,n,k);
  return size(filter_zeros(sfa_layout(m,n,k)));
} catch(const std::exception& e){pro6000d_recovery_mxfp8_v1::set_error(e.what());return 0;}
size_t recovery_mxfp8_scale_b_bytes(int m,int n,int k) try {
  using namespace pro6000d_recovery_mxfp8_v1;validate(m,n,k);
  return size(filter_zeros(sfb_layout(m,n,k)));
} catch(const std::exception& e){pro6000d_recovery_mxfp8_v1::set_error(e.what());return 0;}
size_t recovery_mxfp8_workspace_bytes(int m,int n,int k) try {
  using namespace pro6000d_recovery_mxfp8_v1;validate(m,n,k);
  constexpr uintptr_t fake=0x10000;
  auto arg=args(reinterpret_cast<void*>(fake),reinterpret_cast<void const*>(fake),
                reinterpret_cast<void const*>(fake),reinterpret_cast<void const*>(fake),
                reinterpret_cast<void const*>(fake),m,n,k);
  return std::max<size_t>(1,Gemm::get_workspace_size(arg));
} catch(const std::exception& e){pro6000d_recovery_mxfp8_v1::set_error(e.what());return 0;}
int recovery_mxfp8_pack_scales(const unsigned char* logical_a,
    const unsigned char* logical_b,unsigned char* packed_a,
    unsigned char* packed_b,int m,int n,int k,size_t abytes,size_t bbytes) try {
  using namespace pro6000d_recovery_mxfp8_v1;validate(m,n,k);
  if(!logical_a||!logical_b||!packed_a||!packed_b) throw std::runtime_error("null scale pointer");
  auto la=sfa_layout(m,n,k);auto lb=sfb_layout(m,n,k);
  size_t needa=size(filter_zeros(la)),needb=size(filter_zeros(lb));
  if(abytes<needa||bbytes<needb) throw std::runtime_error("scale output buffer too small");
  std::memset(packed_a,127,abytes);std::memset(packed_b,127,bbytes);
  for(int row=0;row<m;++row)for(int group=0;group<k/32;++group)
    packed_a[size_t(la(row,group*32,0))]=logical_a[size_t(row)*(k/32)+group];
  for(int row=0;row<n;++row)for(int group=0;group<k/32;++group)
    packed_b[size_t(lb(row,group*32,0))]=logical_b[size_t(row)*(k/32)+group];
  return 0;
} catch(const std::exception& e){pro6000d_recovery_mxfp8_v1::set_error(e.what());return 1;}
int recovery_mxfp8_launch(void* out,const void* a,const void* b,const void* sa,
    const void* sb,int m,int n,int k,void* workspace,size_t workspace_bytes,
    unsigned long long stream_address) try {
  using namespace pro6000d_recovery_mxfp8_v1;validate(m,n,k);
  error[0]=0;
  if(!out||!a||!b||!sa||!sb||!workspace) throw std::runtime_error("null device pointer");
  auto arg=args(out,a,b,sa,sb,m,n,k);
  size_t need=Gemm::get_workspace_size(arg);
  if(workspace_bytes<std::max<size_t>(1,need)) throw std::runtime_error("workspace too small");
  Gemm gemm;
  auto status=gemm.can_implement(arg);
  if(status!=cutlass::Status::kSuccess) throw std::runtime_error("CUTLASS can_implement failed");
  auto stream=reinterpret_cast<cudaStream_t>(stream_address);
  status=gemm.initialize(arg,workspace,stream);
  if(status!=cutlass::Status::kSuccess) throw std::runtime_error("CUTLASS initialize failed");
  status=gemm.run(stream);
  if(status!=cutlass::Status::kSuccess) throw std::runtime_error("CUTLASS run failed");
  return 0;
} catch(const std::exception& e){pro6000d_recovery_mxfp8_v1::set_error(e.what());return 1;}
}
