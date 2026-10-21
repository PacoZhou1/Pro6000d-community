# 从 RTX PRO 6000 的硬件约束重新选择 FP8 优化方向

更新：2026-10-14，Asia/Taipei。本轮先核查硬件、重新计算资源与数据复用，再补采代表尺寸的 NCU；没有继续改模型精度，也没有部署新算子。结论是：**不能把下一步简化成删除 pack。应先控制 GEMM 的调度顺序和数据复用，再把主循环成本与融合输出成本分开测。prefill 与 decode 使用不同的优化目标。**

上一轮已经证明 v6 整模长输入 TTFT 慢 7.09%，本轮不重写这个结果。新增诊断揭示了当时仅用 M2741 计数器无法覆盖的问题：在长 prefill 主 chunk **M16160** 上，v6 的完整流程 DRAM 总流量反而达到对照的 **2.19–2.21 倍**，并非各个尺寸都减少 DRAM。代码和原始记录在 [hardware-grounded](evidence/hardware-grounded/)。

## 1. 先锁定这张卡的可用能力

这台机器是 **RTX PRO 6000 Blackwell Workstation / GB202 / SM120**，不是仓库早期的四张 RTX6000D，也不是 B200 的 SM100。NVIDIA RTX PRO 白皮书 Table 4 给出 188 SM、752 Tensor Core、96 GB GDDR7、1792 GB/s 带宽；dense FP8、FP32 累加峰值是 **1007.6 TFLOP/s**。宣传中的约 4000 AI TOPS 是 sparse FP4 口径，不能放进当前 dense FP8 的性能模型。[官方白皮书](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf)

当前实机 NCU [设备属性](evidence/hardware-grounded/device-attributes.json)读到：每 SM 65,536 个 32-bit 寄存器、102,400 B 可用 shared memory、每 block opt-in 上限 101,376 B、48 个 warp、1536 个线程、最多24个 block；整卡 L2 为 134,217,728 B（128 MiB）。通用架构文章里的 L1/shared 合计容量，不能直接当成单个 CTA 可分配的 shared memory。官方部分调优文档与 Programming Guide 的 SM120 资源表有不一致，本轮以当前设备属性和对应 Programming Guide 表为实际约束，来源冲突保留在 [规格核查](evidence/hardware-grounded/sm120-facts.md)。

SM120 当前这条 CUTLASS 路径使用 warp MMA 与 TMA/异步搬运。不能把 SM100 的 `tcgen05`/TMEM 或 Hopper 专用 WGMMA 当成本机可直接调用的替换方案。PRO 的通用 cluster 属性、特定 TMA multicast 指令、以及固定 CUTLASS builder 是否支持某个多 CTA 配置是三个不同问题；尚未证明高效支持的 cluster/DSM 方案只保留为低优先级可行性研究，不能据此承诺提速。精确指令和版本来源见规格核查。

## 2. prefill 和 decode 的限制不同

用官方峰值与带宽得到的理想 FP8 转折点约为 **562 FLOP/byte**。下面假设单个 GEMM 的输入与权重各从 DRAM 冷读一次、BF16 输出写回一次，另计 scale；计算脚本会验证这些尺寸确实出现在此前真实模块归因中。若部分数据命中 L2，实际 DRAM 字节可以低于该单遍模型，因此表中的传输时间不是所有缓存状态下的严格下限。[计算脚本](evidence/hardware-grounded/roofline.py)、[完整结果](evidence/hardware-grounded/roofline.json)

| gate/up 尺寸 | 理想算术强度 FLOP/B | 理想计算时间下限 | 单遍冷读假设下的理论传输时间 | 对选型的意义 |
| --- | ---: | ---: | ---: | --- |
| M1 × N34816 × K5120 | 2.00 | 0.0004 ms | 0.0995 ms | 单 token 权重读取远大于算术成本 |
| M5 × N34816 × K5120 | 9.98 | 0.0018 ms | 0.0997 ms | MTP 小批次仍要关注权重复用、发射和带宽 |
| M2741 × N34816 × K5120 | 2547 | 0.970 ms | 0.214 ms | 已有大量算术复用，少写一个张量不等于更快 |
| M16160 × N34816 × K5120 | 4148 | 5.718 ms | 0.775 ms | 主循环、scale 指令、shared/L2 数据供给都重要 |

这些是**带假设的理论预算，不是实测可持续 roofline，更不是 API 延迟预测**。实际频率低于 boost、CTA 重复读取、缓存失效、同步、动态 scale 和非 Tensor 指令都会增加成本；NCU 的 Tensor pipe active 百分比也不是“实现了多少峰值 FLOPS”。

BF16 `lm_head` 的 N248320×K5120 权重约 2.54 GB，M1/5 的一次完整 DRAM 冷读理论传输时间约 1.42 ms。这里只用于解释为何全链路 decode 不会随 gate/up 局部优化同比例变快；按用户当前范围，仍保留 BF16 head，不做低比特转换。decode 后续应围绕有效 M、已接受 token/目标模型执行次数和真实权重流量判断；已有 Stream-K 微基准变快而整模变慢的记录说明，热缓存单核结果不能直接推广。

## 3. 现有融合改变了资源和复用方式

详细计算及源码落点见 [资源模型](evidence/hardware-grounded/resource-model.md)、[可重算 JSON](evidence/hardware-grounded/resource-model.json)。

| 资源/几何 | 官方 gate/up | v6 融合 |
| --- | ---: | ---: |
| CTA tile | M128×N128×K128 | M64×N256×K128 |
| block 线程 / MMA 线程 | 384 / 256 | 384 / 256 |
| 每 MMA 线程的 FP32 accumulator 下限 | 64 个 | 64 个 |
| launch registers/thread | 168 | 168 |
| 名义 CTA 寄存器数 | 64,512 | 64,512 |
| shared/CTA，含 driver 保留 | 78,848 B | 102,400 B |
| 寄存器及 shared 限制的驻留 | 1 CTA/SM | 1 CTA/SM |

两者 tile 面积相同、occupancy 上限相同，但**输入复用并不相同**：N tile 翻倍减少 A 的重复读取，M tile 减半则增加 B 的重复读取。单层 B 为 170 MiB，比全卡 L2 的128 MiB还大；这不意味着它每次都必须完整读 DRAM，但要求 CTA 调度维持足够的时间局部性。M16160 时，两者约有184/183个单 CTA/SM波次，不能把差异归因于“v6 CTA 数翻倍”。早期 M32 的 v4 才有接近翻倍的波次。

单纯把 v6 改大至 M128×N256 也不是现成方案：两个 K128 stage 的 A+B 原始数据就需要98,304 B，距离单 CTA 101,376 B opt-in上限只剩3072 B，还没容纳 scale、barrier和epilogue；此前实际构建已因此无法满足至少两 stage 的要求。固定 block-scale K128 又约束了 tile K，不能任意减成64来省 shared。

沿实际模板类型还发现一个应单独验证的变量：默认 scheduler 的 raster heuristic。M16160 时官方CTA网格为127×272，v6为253×136，未取整的两个网格本身已使默认 raster 由 AlongM 切到 AlongN；M2741 没有这个反转。后续固定源码核对纠正了早期解释：SM120 实际选择 SM100 CLC 调度器，swizzle 不扩大 physical grid；逻辑映射还取决于 swizzle，不能用名字直接判定复用或执行顺序。本轮调度对照已在[后续实验](14-sm120-scheduler-epilogue.md)完成，源码及计数边界见[映射核对](evidence/sm120-scheduler/scheduler-shape-notes.md)。

## 4. 新增 NCU：看完整流程，也看执行代价

固定原有官方库、V2量化库和v6库，不改算子数学，采集32项实际支持的指标与 LaunchStats。NCU使用kernel replay、未锁频；前三次 `cache-control=none`。成功的三次均先通过已有数值与 changed-input Graph 门槛，再捕获一次基线 GEMM+量化和一次候选 GEMM+pack。逐核原始值、单位、来源 SHA 与汇总见 [诊断汇总](evidence/hardware-grounded/diagnostics-summary.json)，不是线上模型的硬件计数。[捕获清单](evidence/hardware-grounded/capture-inventory.json)同时列出成功与失败尝试。

| 捕获窗口 | 完整流程时间，基线→v6 | 完整 DRAM 总量，基线→v6 | L2请求量比值 |
| --- | ---: | ---: | ---: |
| M2741 | 1.629→1.790 ms | 0.559→0.316 GB | 1.173× |
| M16160 | 9.488→11.416 ms | 3.404→7.511 GB | 1.193× |
| M16160独立再次捕获 | 9.504→11.348 ms | 3.409→7.450 GB | 1.193× |

在 M16160 首次捕获中，GEMM 的 DRAM **读取**从0.951增至6.385 GB，**写入**从1.093降至0.583 GB。少写中间结果是真的，但增加的读取超过节省的写入。总计数不能直接区分是哪个 buffer 的读取，因此“tile/调度破坏复用”是有依据的候选解释，仍需单变量对照。

同一 M16160 GEMM 中，v6 的 Tensor 指令数约为原来的0.996倍（padding差异），总 warp 指令数却增至 **1.166倍**，shared load wavefront增至 **1.450倍**；Tensor pipe active/elapsed由83.59%降至67.21%。两者保持同样一个 CTA/SM，不能只用 occupancy 解释。候选的 short-scoreboard、MIO throttle占比增加，long-scoreboard并没有同步增加；这些指标应结合 issue rate和指令量读，不能把百分比直接拆成 API 耗时。[NCU 官方指标解释](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

为排除缓存条件，额外尝试了 `cache-control=all`，但应用返回 **11**，没有取得数值完成记录或有效计数器；[失败状态与命令](evidence/hardware-grounded/ncu-m16160-cache-all/status.json)、[原始日志](evidence/hardware-grounded/ncu-m16160-cache-all/stdout.log)完整保留。不能把这次失败写成控制缓存验证成功，也没有据此诊断为模型数值错误。当前证据是两次同策略重复，不是缓存因果已经闭环。

## 5. 下一轮按可证伪假设推进

**第一步先做调度与主循环的受控拆解，不直接重写更大的融合核。**

| 顺序 | 单独改变什么 | 要验证什么 | 放弃/继续条件 |
| --- | --- | --- | --- |
| 1 | 相同M64×N256、相同数学，显式选择两种raster；先固定swizzle=8，再有依据地比较1/2/4 | 默认raster切换是否影响长chunk的L2/DRAM复用；小chunk是否回退 | DRAM下降却完整时间不降，也不算优化；不能改变精度或删掉较差尺寸 |
| 2 | 同一tile与调度，对比普通BF16 epilogue与融合FP8 epilogue | 分离tile/主循环损失与SiLU、scale归约、转换、输出同步的新增成本 | 普通epilogue已明显慢，优先改主循环/调度；仅融合慢，再改epilogue。BF16逆排等诊断成本要单列，不能伪装成完整pipeline收益 |
| 3 | 在上述解释成立后，设计直接紧凑输出及scale写回 | 去掉FP8临时读写和pack，同时控制新增寄存器、shared和非Tensor指令 | M2741的pack仅约0.12ms，单删它仍不保证赢；必须比较整段、而非只报新核时间 |
| 4 | 若单CTA受硬件资源约束仍无法竞争，再评估跨CTA合作或其他FP8后端 | 是否有本机与固定版本实际支持的复用机制 | 先做能力/编译/数值门槛；不能拿SM100论文或generic cluster属性当成已支持证据 |

每条设计都先保留 BF16 舍入与 per128 scale语义，跑原12个数值case及changed-input Graph；代表大M用随机交错的完整流程paired timing，冷热口径分开。只有微基准和资源指标有一致收益，才重新进入真实模型调用追踪及无profiler A/B/A。整模前后基线输出、MTP接受数据、冷prefix和embedding计数仍需检查。若不能赢基线，保留原核，不继续为“融合”而融合。

这轮的交付是硬件约束、补充实测与下一实现的明确判据，上表新的raster/epilogue消融和新内核**尚未执行**。没有声称已完成这些实现或获得新的模型提速。

## 6. 服务与复现

本轮只使用自有短时诊断容器，不启动原27B、9B、ASR，不停embedding。每次诊断前后均保存服务身份、启动时间、重启计数与health检查，诊断容器均自动清理；最终功能与后续状态记录见 [结束检查](evidence/hardware-grounded/final-services.json)与[延后检查](evidence/hardware-grounded/final-services-later.json)。

本机离线运行 `python3 _dev/docs/evidence/hardware-grounded/roofline.py`、`resource_model.py`、`analyze_diagnostics.py` 即可重算。GPU诊断入口为 [diagnostic_control.py](evidence/hardware-grounded/diagnostic_control.py)，只允许两个代表M和新label，固定镜像/库路径；保存确切命令、库哈希、NCU策略、exit code与服务快照。NCU二进制报告保留在服务器 `/home/llm/.local/share/qwen-sm120-hardware`，原模型和库仍在此前目录，没有重复下载或修改checkpoint。
