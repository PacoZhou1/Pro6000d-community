# SM120 M64 主循环：最小设计选项

本稿是固定源码的 CPU 设计研究，不是新的 GPU 性能结果。目标保持 E4M3 输入、FP32 block scale、官方 BF16 舍入、SiLU×up 与 per-128 FP8 输出语义。性能分母必须是官方 **GEMM+V2 quant 完整链**；普通 paired BF16 GEMM 只作主循环诊断。BF16 lm_head 不改，小 M 回退、embedding 持续运行、其他原服务保持停止。

后续构建状态：root报告原 `warp24`（2×4、PermN32、B x4）在 `make_tiled_copy_B` 编译失败，现有[预审](preflight-review.md)与[构建设计](builder-design.md)保留该失败：TiledMma分给每线程的B operand仅8B，不足x4 CopyAtom要求的16B。**不能把原单行方案当作可运行内核。**实际后继是 `warp24x2`（2×4、PermN32、B x2）和 `warp24n64`（2×4、PermN64、B x4）；probe严格校验各自metadata，不将三种源码混成一个warp24。

## 为什么先改 warp 组织

[上一轮 M16160 NCU](../sm120-scheduler/ncu-analysis.md)中，官方 M128N128 与 paired BF16 M64N256 的 shared-load wavefront 分别约 1.118×10⁹、1.620×10⁹，相差约45%；Tensor 指令数量接近，而 Tensor active 从约83%降到约69%。后者尚未做融合量化，说明不能只删 pack。wavefront 是整核计数，包含实际 copy、访存布局和其他 shared 工作；**没有证明全部增加来自 B，也没有证明访存是唯一瓶颈**。NCU 时间不作为 API 收益。

固定 builder [sm120_blockwise_mma_builder.inl:153–174](../gemm-epilogue/source/cutlass/include/cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl)选择 `PermTileM=min(M,128)`、`PermTileN=min(N,32)`；Cooperative 为 `AtomLayoutMNK=<4,2,1>`，Pingpong 为 `<2,2,1>`。硬件 atom 由 [mma_sm120.hpp:3253–3255](../gemm-epilogue/source/cutlass/include/cute/arch/mma_sm120.hpp)固定为 `SM120_16x8x32_TN`，不是可以任意换成 N16/N32 的硬件指令。A/B FP8 shared copy 都选 [sm120_common.inl:55–90](../gemm-epilogue/source/cutlass/include/cutlass/gemm/collective/builders/sm120_common.inl)的 `SM75_U32x4_LDSM_N`；K128 的 swizzle 由同文件100–114选择。主循环 [v6 shadow:606–630](../gemm-epilogue/build-gateup-coop-v6/include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp)依据同一个 TiledMma partition_A/B，make_tiled_copy_A/B 后 retile_D，因此改变 warp 分配会改变 shared→register 的读者与重复读取。

下面是**每 warp 恰好加载其唯一 operand 一次**的逻辑模型，按整 K128 block 计算；没有按 LDSM 指令、bank conflict 或 NCU wavefront 仿真。均匀分配时，`A_bytes=64×128×N_warps`，`B_bytes=256×128×M_warps`。实际 CuTe partition 的 padding、广播、copy atom 重复及编译器行为仍需布局枚举/编译/NCU验证。

| warp 布局 | A 逻辑读取 B | B 逻辑读取 B | 合计 B / K128 | 与4×2相比 |
|---|---:|---:|---:|---:|
| 4×2 | 16384 | 131072 | 147456 | 1.000× |
| 2×4 | 32768 | 65536 | 98304 | 0.667× |
| 1×8 | 65536 | 32768 | 98304 | 0.667× |

没有改变全局输入单 CTA 的唯一 payload `(64+256)×128=40960 B`，也没有直接减少 DRAM 或 TMA 流量。第一项将更重的 B 重复读取换成更小的 A 重复读取，是与本形状相符的假设。

## 排序与实施边界

**1．M64N256 Cooperative：4×2→2×4，保持 PermN32，B copy x4→x2（warp24x2）。**warp补丁在上述 builder157–158，再在240只对目标tile选 `SM75_U32x2_LDSM_N`，其他 tile（尤其官方 M128N128 control）保持原样。x2/x4的固定copy映射见 [copy_traits_sm75.hpp:59–90](../gemm-epilogue/source/cutlass/include/cute/atom/copy_traits_sm75.hpp)：目的每thread分别2/4个32-bit值。该补丁用于匹配新8B B partition的覆盖，**不是把原路径读取字节免费砍半**；copy次数与wavefront仍需测量。消费者仍256 threads、CTA仍384 threads；[Cooperative kernel:372](../gemm-epilogue/build-gateup-coop-v6/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp)要求256 MMA threads，2×4满足该人数条件，不能擅自移除 assertion。tile/stages/TMA payload 不变；均摊 final+tmp accumulator 仍为64+64 FP32/thread，寄存器和指令实际结果不能从人数推定。

关键不是只改这一行后直接跑：SM120 epilogue builder [sm120_builder.inl:371–372](../gemm-epilogue/source/cutlass/include/cutlass/epilogue/collective/builders/sm120_builder.inl)只定义 CopyAtomC，并未另建固定4×2 TiledMma。实际 [sm90_epilogue_tma_warpspecialized.hpp:586–610](../gemm-epilogue/source/cutlass/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp)用**主循环传入的** TiledMma 构建 tiled copy，retile accumulators，682–695再把该 copy/坐标传给 visitor。因此 epilogue 映射会自动随主循环变化；不能认为保持 visitor 文件即可保持数学。需枚举所有 `(row,col)` 的实际 retile：gate与up同thread、fragment半偏移、每个128输出组的lane/warp集合以及scale写者唯一性。v6 [paired visitor:53–83](../gemm-epilogue/build-gateup-coop-v6/paired_gateup_visitor.cuh)明确依赖同thread半偏移。

SF128现有 [v6 store visitor:295–301、383–388、427–432](../gemm-epilogue/build-gateup-coop-v6/include/cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp)固定两warp、`warp^4`及首warpgroup写者；2×4应按新映射另证。该源码365–376已有明确以2×4为前提的四warp分支（xor2/4/6），是可复用的最小修正路径，但其注释不替代实际CPU布局证明和GPU逐位门槛。减少B共享读者可能降低shared压力；A复制、scale布局、epilogue归约量和普通指令可能增加，性能方向尚未知。优先分别测 matched BF16 与 fused+pack。

**2．同tile另一最小修复：2×4 + PermN64，保持B x4（warp24n64）。**builder154只对目标tile把N permutation由32改64，warp布局改动与方案1相同；希望扩大B copy可覆盖的线程值域，编译器仍须证明其合法性。硬件MMA仍m16n8k32，不能称为N64硬件atom。tile/K reduction/TMA/stage预算不变，理想每warp唯一operand模型未额外下降；但实际A/B copy分片、C ownership、retile和SF归约都可能变化，因此不是方案1的单变量copy宽度对照。复用四warp SF分支须另作CPU布局证明。与warp24x2分别编译、分别数值gate；不能因一者成功推及另一者。二者都可能增加普通指令或寄存器压力。

**3．后备：M64N128 Cooperative，普通 BF16 epilogue+原V2，比较2/3 stages。**这是保持FP8输入的主循环路线，不要求继续融合。按 v6 dispatch [standalone_dispatch.cuh:155–165](../gemm-epilogue/build-gateup-coop-v6/standalone_dispatch.cuh)增加独立canonical输出配置，tile改64×128×128、去paired visitor，复用普通BF16 Auto epilogue；不能让既有 fused256类型暗中变成128。stage入口见 builder56–57、80–88、213–216。raw operand 三stage仅 `3×(64+128)×128=73728 B`，另加scale、pipeline、epilogue，应有机会装进99KiB，但必须看生成 `sizeof(SharedStorage)`，不是保证。每thread均摊 final+tmp accumulator降至32+32；64Ki regs/SM预算余量可能改善。

代价是 N CTA 数翻倍、同Mtile的A重复TMA载入翻倍；总B逻辑载入不因N拆分减少，M64相对官方M128仍多M方向工作。三stage也占更多SMEM，可能仍只有1CTA/SM。完整链继续写/读BF16，再V2量化，不能叫“免中间张量融合”。优点是最少visitor风险、与canonical官方输出可直接逐位比较；数值和全链测量通过才考虑整模。此方案不在当前control42/warp24构建中，需独立明确立项后实现。

1×8 + PermN64更后置，不列为本轮可运行方案：逻辑operand总量与2×4模型相同，但更多转到A，并可能需要八warp SF128归约；现有visitor295–296只允许1/2/4warp，不能只把常量改8继续沿用xor分支。没有完整 reduction/writer布局证明前不构建/计时。

## 硬件预算与不采用的捷径

依据 [已核硬件事实](../hardware-grounded/sm120-facts.md)与 [实际metadata](../sm120-scheduler/kernel-metadata.json)：SM120为65536个32-bit regs/SM、255/thread、100KiB shared/SM、99KiB user shared/CTA、48warps/SM；当前M64N256 mainloop83968B、fused kernel101376B（恰99KiB），额外driver1024B单列。三stage单operand即122880B，原tile不能靠StageCount3装入；M128N256两stage单operand98304B，留出的3072B不够当前scale/barrier/epilogue。显式StageCount不能豁免硬件上限。当前168regs/thread、384threadsCTA，register/shared occupancy limit实际均为1block；2×4没有增加SMEM预算，也未保证更多常驻CTA。

主循环[v6 shadow:582–584、688–719、736–755](../gemm-epilogue/build-gateup-coop-v6/include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp)同时持有tmp/final C，逐K32 MMA、每K128 rescale。固定 `(A_scale*B_scale)` 后乘tmp并累加的顺序必须保留，不能换回 `(tmp*A)*B`、调K reduction顺序或硬件MXFP8 scale格式。M64N256 Pingpong分给4warp时仅两套C即128+128 regs/thread，已超过255且尚无operand/scale；历史v3实际spill并慢约2.8–2.9倍，不作为新的最小推荐。M32N256已实测消spill仍慢，也不重复作为新方向。

更细的gate64/up64交错不是免费layout替换：B原scale覆盖128连续原列，混排后需重映射/扩展scale布局；一个N128 CTA只产出64个SiLU乘积，输出per128动态amax跨两个CTA，必须增加明确归约/第二阶段。不能缩小输出quant组到64来绕过该依赖。A/B当前x4 LDSM选择也不能仅因MMA BRegisters为2就改x2并宣称读取减半；必须通过make_tiled_copy的源/目的覆盖证明及编译，否则易错或仅增加copy次数。本轮优先不动这两处。

## 实验入场门槛

以固定官方GEMM+V2、control42与两个编译后通过数值门槛的候选fused+pack同process随机交错31对warm/owned-cache-disturbed全链比较；matched pairedBF16 inverse先做全bit诊断。固定来源/二进制SHA、输入/权重hash、每M×每臂换input bytes和scale的Graph replay、capture输出检查先于计时。正常M257/2741/2786/16160/32768及M257 zero/tiny/large/exceptional逐位检查，不放宽BF16/Q/scale门槛；正常输入比较下游官方downGEMM。M1/5/16/256只验证fallback，不记成candidate收益。NCU只作通过门槛后单臂诊断。任何未证明布局或数值错误不得进入timing；后续按用户明确要求，即使最佳合法候选仍慢于官方，也由root做限定M的隔离模型dispatch与无profiler A/B/A，验证负结果；未经整模收益证据不采用，不能由shared计数下降直接宣布整模收益。

## Probe / CPU 分析入口

[mainloop_probe.py](mainloop_probe.py)复用冻结 scheduler helper（SHA `f27f27083dec90dcf3884cd8190c876df9957c285a1cd3b97b265a4145ac67b8`）的ABI1、hash及随机配对计时，和冻结 pipeline helper（SHA `eed1a8d2fa5e97d0b0dfefcb504c6ca9d8493d1c00997788e1343aadd60c42ae`）的官方/v2 ABI、输入生成、full-bit与capture逻辑；运行前强制检查依赖文件SHA。只加载control42/warp24x2/warp24n64的新隔离库，不加载历史legacy fusion库。默认**21臂**：三种官方完整链控制，加三个库各六个raster/swizzle。也可用 `--arms`明确子集，但普通gate/timing不能省掉三种官方控制。

以下为**由root执行的实验容器命令模板**；路径和每库完整SHA须按实际build manifest填写。`--mode timing`自身先跑全部13case gate，并在5个大M对全部选臂做换输入Graph，不依赖外部“已通过”口头结论。大buffer/Graph封装在每case/每shape函数中，结束后释放并empty_cache；capture前至少8GiB free检查，未通过即报错，无自动删臂/回退。

```sh
python3 /work/sm120-mainloop/mainloop_probe.py --mode timing \
  --library control42=/work/build-control42-v2/libsm120_mainloop_control42.so \
  --library warp24x2=/work/build-warp24x2-v2/libsm120_mainloop_warp24x2.so \
  --library warp24n64=/work/build-warp24n64-v2/libsm120_mainloop_warp24n64.so \
  --library-sha control42=CONTROL42_FULL_SHA \
  --library-sha warp24x2=WARP24X2_FULL_SHA \
  --library-sha warp24n64=WARP24N64_FULL_SHA \
  --scheduler-probe /work/sm120-scheduler/scheduler_probe.py \
  --old-probe /oldwork/gemm-epilogue/pipeline_probe.py \
  --baseline-library /baseline/libtargeted_fp8.so \
  --quant-library /v2/libfp8_silu_quant_group_major.so \
  --timed-m 2741 2786 16160 32768 --repeats 31
```

单NCU窗口用 `--mode ncu --timed-m 16160 --ncu-arm warp24x2_along_m_s1`，并只提供该候选库的LABEL/path/SHA。其所选M先做官方与该臂数值、down和changed-input Graph；随后唯一NVTX `SM120_MAINLOOP:<arm>`内完整链一次。这个局部NCU门槛不能冒充13case完整gate。`paired_<variant>_<raster>_s<swizzle>`只在NCU模式作普通BF16 GEMM诊断，不含inverse/V2，输出记录明确 `complete_pipeline=false`；不进入全链性能汇总。

[analyze_mainloop.py](analyze_mainloop.py)为CPU-only，要求唯一metadata/final summary、13个指定数值case、每正常大M全部选臂的Graph、正确依赖/库/权重hash、每完整输出元素数、stage/thread/layout/copy metadata、随机顺序完整性、31配对和同gate输入hash；它汇总所有显式选臂，相对三官方控制和同scheduler control42分别报配对比率。入口：

```sh
python3 _dev/docs/evidence/sm120-mainloop/analyze_mainloop.py \
  RESULT_STDOUT_JSONL --output RESULT_ANALYSIS_JSON
```

本地CPU检查已通过py_compile、CLI，构造的gate/21臂timing/单候选NCU记录均可解析；缺case、缺Graph臂、输入scale未改变、只比较部分输出、错误copy/permutation metadata、缺库SHA、30配对、timing输入hash不一致八类反例均被拒绝。这是解析器自检，**不是GPU数值/Graph/性能通过证据**。GPU结果由root保存原始JSONL后严格分析，设计稿不预设两候选通过或加速。
