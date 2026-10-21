# SM120：先改善遍历，再判断融合是否值得

本轮完成了调度、epilogue拆分、硬件计数、真实模型派发和无profiler A/B/A。**没有新增值得上线的优化**：融合流程仍慢于官方；官方M16160调度在合成完整算子链约快2%，整模TTFT配对差异仅0.14%，小于前后基线漂移，因此不采用。M32768相同参数会明显退化，进一步证明必须按尺寸和缓存行为选择调度。

固定环境、显卡规格和上一轮限制见 [硬件依据](13-sm120-hardware-roadmap.md)。本轮代码、原始日志和分析均在 [sm120-scheduler](evidence/sm120-scheduler/kernel-design.md)。只测试 FP8；权重、BF16 舍入、激活量化语义和 BF16 lm_head 均不改变。

## 1. 如何把变量分开

新库 SHA256 为 `af1d814966a746ff09edc55d40196b871067d88d22ed0e50700c1f3a4eb185d6`，固定镜像 `a4a4c043…c3a6b`。三种入口分别是官方 M128N128 BF16 输出、配对布局的 M64N256 BF16 输出、M64N256 融合 FP8 输出。每种均显式接受 Heuristic / AlongM / AlongN 和 swizzle 1 / 8。实际SM120默认选择 `PersistentTileSchedulerSm100`（CLC），不能套用先前SM90的fast轴／rounded-grid推断；源码映射、名义坐标顺序与实际硬件launch顺序需要分开解释。详见[尺寸与调度源码核对](evidence/sm120-scheduler/scheduler-shape-notes.md)。

两种 M64 入口用编译期断言确认 `CollectiveMainloop` 类型完全相同，均为两级流水。它们的 epilogue 和编译资源仍不同，因此不能把时间差全解释成某一条量化指令。融合主核占满 99 KiB 用户 shared memory；paired BF16 为 91 KiB，官方为 76 KiB。三者384线程，ptxas均168寄存器/线程；官方还报告少量 spill。详见 [编译与 ABI](evidence/sm120-scheduler/kernel-design.md)。

每个完整流程都包含实际后处理：官方是 GEMM + V2 SiLU量化；融合是 GEMM + pack；配对 BF16 诊断是 GEMM + 逆排 copy + V2量化。逆排不是正式候选，不用单独 GEMM 的时间冒充完整流程。

## 2. 正确性及失败记录

[接受的 v4 原始记录](evidence/sm120-scheduler/scheduler-matrix-v4/stdout.log)通过13个数值用例；M1/5/16/256仅验证原路径及回退，候选支持范围是257–32768。六个调度配置逐位对照 BF16、FP8 bytes、FP32 scales，正常输入还比较下游 BF16。M257/2741/2786/16160的8臂共32个 changed-input CUDA Graph 见证通过，计时图输出另作逐位检查。[验收说明](evidence/sm120-scheduler/numerical-verification.md)。

保留了三类失败：首版多库同名 host TLS 符号冲突导致139，改私有namespace/internal TLS/visibility后重编；首次成功矩阵遗漏新增 M2786 的 changed-input 图验证，分析器拒绝后补齐重跑；一轮 NCU 在旧 legacy 库的预热 launch 崩溃，尚未进入目标捕获，重新隔离成仅所选臂后再采集。最后一项的调用栈不能单独证明具体 ELF 根因，不能计为有效 NCU 结果。

## 3. 完整流程：融合变快，但仍落后

以下为同一进程随机交错31对 CUDA Graph event 的中位数，单位 ms。warm 是未显式冲刷的图重放，扰缓存是在计时区外写入大于两倍 L2 的自有缓冲；不是保证冷 L2。原始 p10/p90 与逐对比值见 [分析 JSON](evidence/sm120-scheduler/scheduler-matrix-v4/analysis.json)。

| M | 缓存方式 | 官方 GEMM+V2 | 原 v6 融合 | 本轮最佳融合设置 | 最佳融合 |
|---:|---|---:|---:|---|---:|
| 2741 | warm | 1.762 | 1.931 | Heuristic/s8 | 1.939 |
| 2741 | 扰缓存 | 1.755 | 1.927 | Heuristic/s8 或 AlongM/s8 | 1.925 |
| 2786 | warm | 1.796 | 2.003 | Heuristic/s8 | 1.997 |
| 2786 | 扰缓存 | 1.782 | 1.989 | AlongM/s8 | 1.989 |
| 16160 | warm | 10.446 | 12.513 | AlongM/s1 | 11.367 |
| 16160 | 扰缓存 | 10.465 | 12.527 | AlongM/s1 | 11.403 |

长 prefill 融合延迟约降低9.2%，但相对官方仍慢约8.8%。Heuristic和显式AlongM在短M理论映射相同，微小时间差不代表另一种算法。六配置全部报告，没有只选赢的一轮。上述都是合成实际尺寸核测试，不能换算成 TTFT 或 tokens/s。

## 4. 普通 BF16 epilogue 对照

在 M16160，AlongM/s1 的官方 GEMM 约9.336 ms，而同融合主循环的 paired BF16 GEMM 约10.646 ms；paired BF16 的完整三核链为13.046 ms。其逆排 copy 单独约1.513 ms，V2量化单独约0.888 ms。不同窗口的中位数不能直接相减来分摊耗时。[四组诊断与边界](evidence/sm120-scheduler/epilogue-analysis.md)。

这证明不能把全部问题归咎于输出量化：M64N256 配置本身也需要与官方 M128N128 比较。两个配置的 mainloop 不同，而 paired BF16 与 fused 才是同主循环；不要混淆这两种对照。

## 5. 同库官方调度：只允许按已测尺寸派发

[三臂补测](evidence/sm120-scheduler/official-tuning-v2/analysis.json)在同一进程中比较旧官方H8、新库官方H8、新库官方AlongM/s1，每种M各31对并通过BF16/FP8/scale和改输入Graph。旧/新H8的配对比值约1，M16160的同库H8 / AlongM-s1配对中位比为 **1.02025（warm）/1.01928（扰缓存）**。因此该尺寸约2%的完整流程收益可由同库调度对照支持。

**不能推广成“长prefill统一s1”。** M2741/2786配对收益接近零；M32768反而从约21.95 ms退化到33.04 ms，慢约50%。warm短M存在时钟变化，分别取中位数后再相除会与逐对统计不同，结论采用逐对比值，不挑最快样本。只允许 M==16160 进入隔离模型测试，其余M继续原路径。首次补测命令错误导致argparse退出2，完整保留在official-tuning-v1；接受的是修正命令后的v2。

## 6. 硬件计数与采用决定

六组有效 NCU 捕获采用相同输入、cache-control=none、clock-control=none，逐核13个 replay pass。失败的 legacy 预热尝试单独保留，不参与下表。[原始计数与严格解析](evidence/sm120-scheduler/ncu-summary.json)。

| M16160 主核 | DRAM读 GB | DRAM写 GB | Tensor活跃 % | 执行指令十亿 | shared load wavefront十亿 |
|---|---:|---:|---:|---:|---:|
| 旧官方 M128N128 H8 | 1.008 | 1.094 | 83.56 | 3.019 | 1.118 |
| 融合 M64N256 AlongN/s8 | 6.389 | 0.583 | 67.02 | 3.519 | 1.621 |
| 融合 M64N256 AlongM/s1 | 0.261 | 0.554 | 68.11 | 3.500 | 1.622 |
| paired BF16 M64N256 AlongM/s1 | 0.260 | 1.096 | 68.89 | 3.162 | 1.620 |
| 新官方 M128N128 AlongM/s1 | 0.258 | 1.097 | 82.99 | 3.005 | 1.118 |

同一融合二进制改变遍历和swizzle，主核读取从6.389降到0.261 GB；paired BF16同样从6.415降到0.260 GB。这支持“调度组合会显著改变此输入下的实际 DRAM 流量”，比上轮单看源码推断更强。但仍不能将总计数逐字节归到权重B，或把方向与swizzle各自影响完全拆开。

访存改善后，M64两臂仍有约1.62十亿 shared load wavefront，官方约1.12十亿；Tensor活跃约68–69%，官方约83%。融合比同主循环普通BF16增加约10.7%的执行指令、减少写回，却未让整个流程超过官方。pack仍是独立kernel；这轮NCU为约0.754 ms，不能把它当成API耗时或直接从另一轮未profile的中位数扣除。

因此暂不继续盲改compact store：调度修正后剩余问题涉及主循环/线程组织，不只是临时张量大小。若继续融合，应先研究能保留官方较高计算效率的tile/warp映射，再决定如何让一个CTA拿到配对gate/up并完成128元素scale归约；不能为了融合牺牲主循环效率。SM120 99 KiB CTA shared上限使“增大tile并多加流水级”不可直接套用。native MXFP8的UE8M0缩放也不等同当前FP32 per-block scales，不能把它当作无损替换。

补采M32768的同库官方对照，两者实际grid均(256,272,1)：H8主核DRAM读6.064 GB、Tensor活跃83.60%，AlongM/s1读46.498 GB、活跃36.09%；NCU主核18.264→31.113 ms，V2量化基本相同，L2 request总数也接近。计数与无profiler中约50%退化相互支持。[M32768原始计数汇总](evidence/sm120-scheduler/ncu-large-m-summary.json)。A的FP8工作集从M16160的约78.9 MiB增到M32768的160 MiB，跨过128 MiB L2容量，是应考虑的容量／复用假设；总DRAM计数仍不能直接证明全部新增读取来自A。

## 7. 隔离整模实验与服务边界

仅M16160官方调度方案通过入场门槛。新opaque op内部按精确M和canonical权重身份决定分支，M32768等回退；不改变FP8或BF16 lm_head。隔离runtime smoke逐位/回退/改输入Graph已通过，[结果](evidence/sm120-scheduler/runtime-smoke-v1/runtime-smoke.json)。真实模型witness已确认65个注册模块仅在M16160命中，M32768等回退；21张图记录20次FX rewrite。Nsight捕获65次私有官方BF16 GEMM，65次成功direct launch关联、0个Graph node／replay关联。M16160来自同次runtime派发记录，SQLite kernel记录本身不解码M参数。[集成汇总](evidence/sm120-scheduler/model-witness-v1/witness/integration-summary.json)、[实际launch汇总](evidence/sm120-scheduler/model-witness-v1/witness/dispatch-summary.json)。这解决“实际上有没有走到新路径”，不构成性能证据。

实验保留与前轮一致的65份packed副本，额外11,589,593,600 B，约10.79 GiB；新官方kernel实际不读取这些副本。A/B/A共同固定12 GiB KV池和这份额外分配，仅为控制条件，不建议将无用副本带入部署。witness启动/首次派发观测peak allocated约63.28 GiB、reserved约65.52 GiB，是累计allocator峰值，不是每请求峰值。

无profiler A/B/A已完成。每臂12次约19K输入/1输出prefill、6次约4.8K输入/1024输出decode，每类另有1条warmup剔除；逐index请求hash一致，模型/image/参数/V2/12GiB KV池/重复权重分配一致。[原始分析](evidence/sm120-scheduler/aba-official-v1/analysis.json)、[配置与采用决定](evidence/sm120-scheduler/aba-official-v1/model-contract.json)。

| 指标 | A1官方基线 | B仅M16160新调度 | A2官方基线 |
|---|---:|---:|---:|
| 长prefill TTFT中位秒 | 2.24633 | 2.25065 | 2.26271 |
| decode中位token/s | 106.728 | 106.553 | 106.725 |

逐index以(A1+A2)/2为基线，B的TTFT缩短中位 **0.1443%**；A2相对A1的逐对漂移中位 **0.7558%**。B既没有快过A1，收益也未超过基线漂移，不能据此宣称服务提速。B第12个prefill窗口有embedding业务活动；全部12对照常保留，11个embedding安静窗口的配对差异仍仅0.1525%。B的前三个decode窗口也有embedding活动，不据这6条TPS差异归因候选。

三臂12条prefill单token正文、6条decode正文hash均一致，decode投机接受统计逐index一致；性能请求的reasoning为空，不把空hash相同当成思考质量证明。五个合成质量用例及四条开放输出探针通过，后者正常停止且有答案；这不是全业务质量验收。prompt-index bootstrap只描述这些请求的差异，没有独立重复时间窗口，不能当作排除重启／时钟漂移的因果置信区间。

**采用决定：保留实验，不上线此调度，也不采用融合版。** 后续若继续，优先改M64主循环的tile/warp映射并保持资源预算，或在原官方kernel里做shape-aware调度，必须重新过完整流程与整模门槛；不再把删pack作为唯一优化方向。

所有原27B/9B/ASR保持停止，embedding不启停。各诊断前后核对identity/start/restart和embedding功能，模型运行期间15秒采样health。最终[功能检查](evidence/sm120-scheduler/final-services.json)与[延后复查](evidence/sm120-scheduler/final-services-later.json)均返回1024维有限向量，embedding自2026-10-06启动、restart_count=1均未变；[运行清单](evidence/sm120-scheduler/final-running-inventory.json)确认只保留embedding模型容器，自有实验容器均已清理。这能证明采样点与检查点状态，不能保证每个业务请求零延迟影响。生产实现尚未替换。

固定版本、构建、数值门槛、NCU、真实派发及A/B/A的复现入口见[复现说明](evidence/sm120-scheduler/reproduction.md)。独立重算整模结果见[模型核验](evidence/sm120-scheduler/model-review.md)，全轮检查见[独立评审](evidence/sm120-scheduler/final-review.md)。
