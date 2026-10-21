# M64×N256 mainloop 映射预审与 host ownership 检查

本页是运行前预审；已有 `warp24-v1` **编译失败**，尚无新候选 GPU 数值或计时结果。固定 SM120 blockwise builder 用 `rr_op_selector_sm120` 的 16×8×32 MMA、`Tile<64,32,32>` permutation 和 Cooperative `AtomLayoutMNK=4×2×1` 构造 M64×N256×K128 的 `TiledMma`（`gemm-epilogue/source/cutlass/include/cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl:153–174`）。原 2×4／PermN32 使 B operand 每线程仅 8 B，而 `SM75_U32x4_LDSM_N` 要求 16 B；失败发生于 `make_tiled_copy_B` 编译期，不能视为数值或性能结果。后续分别尝试保持 PermN32 而将 B copy 改为 x2，或保持 B x4 而将 PermN 增至 64；两者都不是“只改 AtomLayout”单变量，其中 PermN64 还可能改变 accumulator ownership。1×8 尚未有 SF128 visitor 的八 warp 归约证明，当前 builder 明确 defer，不能与已验证候选并列。

## 关键风险与验收门槛

1. **资源余量很薄。** 实测设备每 SM 65,536 个寄存器、每 CTA shared opt-in 上限 101,376 B；固定 v6 M64×N256 Cooperative 为 384 线程、168 register/thread，即名义 64,512 register/CTA，shared 含 driver 102,400 B，都是单 CTA/SM（`hardware-grounded/resource-model.md:17–29,46`）。换 warp ownership 可能改变寄存器或 epilogue 实例化；需保存 ptxas register/spill、dynamic shared、实际 launch 和 occupancy，不能只看 tile 未变就推定资源相同。
2. **gate/up 对位与 SF128 归约是两个独立条件。** 固定 v6 `PairedGateUpStore` 把同线程 accumulator fragment 的前后半配为 gate/up（`build-gateup-coop-v6/paired_gateup_visitor.cuh:55–83`）。其 SF128 shadow visitor 对 4×2 强制 `WarpsPerSF=2`，使用 `warp_idx ^ 4`；现有四 warp 分支注明假设 2×4（同目录 `include/cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp:295–302,365–389`）。因此 2×4 至少要验证同线程 ±128 列配对、fragment 半区顺序、每 row/128 列唯一覆盖、以及对应的四 warp scale 归约与 scale 写回。把 `WarpsPerSF` 由 2 改 4 是**第二个独立哈希的 visitor 改动**，不能把它算作仅改 mainloop，也不能只因原分支注释提到 2×4 就宣称数值正确。编译、逐位 FP8/scale/BF16 对照及 changed-input Graph replay 均先于计时。
3. **host 检查的范围有限。** `ownership_probe.cu` 用 CuTe 配方构造 `partition_C(identity_tensor<64,256>)`，分别列出 4×2／PermN32、2×4／PermN32 和 2×4／PermN64 的每 thread/lane 持有坐标，检查整 tile 唯一性、每 `(m,n)` 与 `(m,n+128)` 是否同线程、v6 `i` 与 `i+half` fragment 对位、每个 row/SF128 组的 warp 覆盖及旧 `warp^4` 双 warp 是否能覆盖完整组。它不执行 GEMM，不构造 B operand copy，因此不会验证 `make_tiled_copy_B` 的 16 B 对齐要求；也不模拟 epilogue `visit_results` 形状、quad 内 shuffle、shared-memory 同步、scale 写回或 CUDA Graph。即使 JSON 布尔值全为 true，仍需候选真实编译、full-bit 与 Graph gate。反之，`sf128_xor4_pair_failures>0` 可直接否证**不改**旧 2-warp 归约的兼容性；四 warp 分支须另做数值验证。

## 纯 host 编译及运行（由 root 在有固定 CUDA 工具链的实验环境执行）

从仓库根目录运行以下命令；编译调用 `nvcc`，生成的程序只用 CuTe host tensor layout，不调用 CUDA runtime/device API，不启动 GPU。`--full` 的每行 JSON 附每个 `tid`/`warp`/`lane` 的完整 `(m,n)` 顺序；去掉该选项可只看摘要。现在输出三行记录，并分别保存为 ownership-v2 证据，不覆盖旧 ownership-v1。进程退出码 0 表示成功枚举，并**不**表示候选内核编译或数值通过；每个布尔值和失败计数需单独检查。若新 PermN64 CuTe 实例化失败，保留 compiler log，不放宽 CUTLASS 断言。

```bash
nvcc -std=c++17 -O0 --expt-relaxed-constexpr -arch=sm_120 \
  -I _dev/docs/evidence/gemm-epilogue/source/cutlass/include \
  _dev/docs/evidence/sm120-mainloop/ownership_probe.cu \
  -o /tmp/sm120-ownership-probe
/tmp/sm120-ownership-probe --full > /tmp/sm120-ownership-v2.jsonl
```

解释时先用已验证的 4×2 基线做 probe 自检；若基线的几何布尔值与其已知数值行为冲突，就不得用本 probe 的相应布尔值否证候选。`unique_tile_ownership`、`gate_up_fragment_pairing` 和 `row0_warp_counts` 是坐标依据；`compatible_with_v6_xor4_sf128_geometry` 仅描述旧两 warp 归约的几何条件。新四 warp visitor 仍须对 row/scale 索引、写者唯一性和输出位值单独验证。只有候选完整数值通过后，才可以将其与固定 4×2、官方 128×128 作同输入、同库哈希、同计时范围的核级对照；普通 BF16 epilogue 分解列只作诊断，不能冒充融合链或整模加速。

## ownership-v2 已得结果与后续静态复审

`ownership-v2/ownership-status.json` 固定源码 SHA `82d84617…871fb85`、固定 CUTLASS builder/header SHA，`nvcc` 编译与纯 host 运行均退出 0。逐项从完整 `ownership.jsonl` 重新枚举：4×2／PermN32、2×4／PermN32、2×4／PermN64 的 64×256 坐标均恰有一名线程持有，全部 64×128 对应 gate/up 同线程，flatten fragment 半区无偏移失配。4×2 的 128 个 SF128 row/group 均可由 `warp^4` 配对覆盖；两个 2×4 变体每组恰由 warp `{0,2,4,6}` 或 `{1,3,5,7}` 覆盖，旧双 warp `xor4` 对 128 组全失败。这支持采用四 warp 归约的**几何动机**，没有验证 kernel 编译或 visitor 写 scale 的正确性。

当前 `build_mainloop.py` 把 2×4 visitor 的 `WarpsPerSF` 从 2 改 4；既有四 warp 分支读取 `warp^2/^4/^6`，与上述参与集合相符。然而固定 visitor 的 scale 写者条件仍是 `thread_idx_in_quad==0 && warp_idx<4`（`build-gateup-coop-v6/include/cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp:427–446`）：在集合 `{0,2,4,6}` 内可能同时允许 warp0 和 warp2，在 `{1,3,5,7}` 内可能同时允许 warp1 和 warp3。`tC_cSFD`/残量谓词也许进一步消重，但 CPU `partition_C` 不能证明它；这是实际 FP8 scale 写者唯一性和竞争的重点门槛。不得只凭 ownership-v2 或 `WarpsPerSF=4` 宣告数值合法；需要逐位 scale/FP8/BF16 对照、tail/nonfinite、changed-input Graph，最好另存写者坐标诊断或并发写检测。不能在三臂计时中途改 visitor 后仍沿用旧库 SHA。

`mainloop_probe.py` 静态检查：各库以 SHA 和元数据校验 ABI／layout／资源；正式形状、特殊分布及 matched 普通 BF16 都先比逐位结果，normal 形状在计时前做改输入 Graph 重放，正式计时包含各自 GEMM+pack 或官方 GEMM+V2 quant 的完整链并将 owned-cache-disturbed 与 warm 分列。NCU 单 arm 记录独立于 Graph 性能，`paired_BF16` 仅在 NCU 标为 GEMM-only。尚未看到 GPU gate 或候选编译结果；未发现可确定令其绕过数值失败仍报 timing 的路径。记录限制：`metadata_check` 与 host ownership 只证明静态结构，不证明 scale 写回；Graph replay 测的是合成固定形状，不等于真实模型派发或 API 收益。首轮 `warp24` 的 8 B vs 16 B copy 失败应保留为失败证据，`warp24x2` 与 `warp24n64` 均是额外变量，结果须各自命名和归因。
