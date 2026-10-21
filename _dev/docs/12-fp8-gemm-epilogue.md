# FP8 GEMM → SiLU + block 量化 epilogue 实验

更新：2026-10-14，Asia/Taipei。v4、v6的所测数值/Graph、完整算子计时、NCU、正常模式模型派发及无profiler A1/B/A2均已完成，自有临时容器清理、结束功能与延后状态、embedding健康采样已核对。本轮没有上线候选。

本轮已经实现不物化全宽 BF16 GEMM 输出的两阶段候选。v3 修正 block scale 的乘法顺序后，所测输出与官方链逐位一致，但实际大 M 的完整算子流程耗时约为基线的 2.8–2.9 倍；v4 清除实测寄存器 spill 后，大 M 仍耗时约 1.5–1.6 倍。v5 的普通 Cooperative GEMM 正确、融合输出错误；v6 修正其SF128 warp归约后所测逐位门槛通过，但大M算子仍慢约10%–20%。**v4、v6的整模prefill对照也变慢，本轮不采用任何候选，不覆盖生产。**

v4 已在真实模型命中：新无 profiler A1/B/A2 对照的约19K长输入首文本时间为 **2.25023 / 2.76992 / 2.26305 s**，12个配对请求相对两次基线均值的 TTFT **慢22.75%**；普通 decode 生成速率约 **106.75 / 106.73 / 106.74 tok/s**，基本不变。六个decode正文hash三臂相同，合成质量检查通过，但性能没有通过，v4 不采用。详见第6节。

v6同样实际命中，其独立A1/B/A2长输入首文本时间为 **2.25125 / 2.41533 / 2.26091 s**，配对TTFT **慢7.09%**。decode速率为 **104.56 / 106.68 / 106.66 tok/s**，但两次基线本身6/6正文hash不同、MTP接受数据也变动；B与A2的hash和接受数据一致，不能将B相对A1的速率变化认作算子增益或整模逐位一致。v6也不采用。

范围限 **FP8**：保留 BF16 `lm_head`，不推进 FP4/NVFP4、head 改型或 embedding 算子。本机为 **1×RTX PRO 6000 Blackwell Workstation，96GB、600W、SM120**；实验固定 vLLM 0.31.0 官方镜像及其 CUTLASS 源码。原 27B、9B、ASR 按用户当前要求保持停用，待明确要求再启动；embedding 保持运行。此次服务策略与上轮“实验后自动恢复”的历史策略不同。

## 1. 要消除的是哪段中间数据

固定 FP8 MLP 链是：block FP8 gate/up GEMM → BF16 gate/up → FP32 SiLU(gate)×up → 每 128 列动态 FP8 量化 → FP8 down GEMM。前轮 V2 已融合 SiLU 与量化，本轮以 **官方 GEMM + V2 量化**为基线，不能把前轮 V2 已有收益算成本轮收益。

| 张量 | 本轮固定 dtype / 布局 |
| --- | --- |
| A、A scale | E4M3 `[M,5120]` 连续；FP32 `[M,40]`，stride `(1,M)` |
| canonical B、B scale | E4M3 `[34816,5120]` 行主；FP32 `[272,40]` 行主，每 128N×128K 一块 |
| 原 GEMM D | BF16 `[M,34816]` 连续，前 17408 列 gate、后 17408 列 up |
| 下游输入 Q、scale | E4M3 `[M,17408]` 连续；FP32 `[M,136]`，stride `(1,M)` |

GEMM accumulator 为 FP32，但官方链先舍入为 BF16，再将 gate/up 转回 FP32计算激活。候选必须保留这道舍入，直接对 FP32 accumulator 做 SiLU 不满足原语义。已保存的真实 eager 模块/trace归因包含M=16160、2741；本轮另测M=2786大M形状。小 M=1/5 以及 M=256 保留官方链回退。布局和固定源码边界见 [数据流调查](evidence/gemm-epilogue/dataflow.md)、[来源清单](evidence/gemm-epilogue/source-manifest.json)。

单层 BF16 D 的一次逻辑写加一次逻辑读为 `M×34816×2×2` 字节：

| M | BF16 中间 D 大小 | 原链一次写 + 一次读 |
| ---: | ---: | ---: |
| 2741 | 190,861,312 B | 381,722,624 B，约 364.04 MiB |
| 16160 | 1,125,253,120 B | 2,250,506,240 B，约 2.096 GiB |

这是逻辑张量交通量，不是实测 DRAM 节省，也不能据此预测 TTFT。缓存、临时张量、额外权重读取和 spill 都会改变实际交通量。

## 2. 实现：融合 epilogue，再 pack

canonical 布局的匹配 gate/up 列相距 17408，原 128N CTA 拿不到同组的两半。候选在启动时复制权重，每个输出组排列为 `[gate128,up128]`，对应 B scale 的 128 行组做同样置换；A、K 分组及 checkpoint 量化值保持原语义。256N CTA 因而拥有一组完整的 128 对 gate/up。

实际实现仍有两个 CUDA kernel：

1. GEMM 的自定义 epilogue 将 accumulator 分别舍入为 BF16，做 FP32 SiLU×up、128 列 max/scale 和 E4M3 转换；写 paired FP8 暂存 `[M,34816]`，每个 256 列组前 128 列有效、后 128 列置零，并写 CUTLASS 原生布局 scale 暂存。
2. pack kernel 去掉零半组，输出连续 `[M,17408]` FP8，并把原生 scale 转成下游 `(1,M)` 列主布局。

因此它删除了原 BF16 中间 D 的物化，但仍保留 FP8/scale 中间写读及 pack。本文所有计时和 NCU 对照都包含完整两阶段，不能称为“单核完成整个流程”。输出、两个暂存及 workspace 由调用方持有，stream 0 合法，launch 内不做 GPU 分配。ABI 版本为 2；查询暂存/workspace 大小和错误状态的接口见 [候选构建器](evidence/gemm-epilogue/build_candidate.py)、[适配器](evidence/gemm-epilogue/candidate_adapter.py)。

单层 packed B/B scale 新增 **178,301,440 B**。实际模型启动登记 64 份主模型与 1 份 MTP，共 **11,589,593,600 B，约 10.79 GiB**，见 [累计启动清单](evidence/gemm-epilogue/model-witness-v4-fixed/startup-pack-616-2970a1082f59-summary.json)。canonical 副本保留供回退，A/B 两臂都持有相同 packed 副本、固定 12 GiB KV 预算；这保证实验条件公平，但不能消除部署显存成本。单层合成 probe 的一次预排 event 为 v3 **421.70 µs**、v4 **430.66 µs**，分别单独记录，未计入稳态算子时间；不是完整模型启动耗时。

## 3. 数值门槛和迭代结果

先要求预分配基线逐位复现固定官方 GEMM BF16、官方量化 FP8 字节与 FP32 scale 位；再要求候选最终 Q/scale 及使用它们的官方 down GEMM BF16 全量逐位相同。普通形状还检查真实 CUDA Graph capture 后原位换输入，重放与新鲜官方计算一致。任何失败都禁止后续性能阶段，未使用误差容限替代逐位门槛。

基线默认 7 个 M（16160、2741、2786、256、257、1、5）与 M257 的 zero/tiny/large/exceptional 四组全部通过。特殊组在 FP8 A 或 A scale 注入选定 NaN/±Inf/最大有限值，检验明确边界，**不是穷举所有有限 FP8 输入或全部特殊值的证明**。官方最终 clamp 可以把 NaN 变成 +448，所以最终 FP8 全有限也不证明内部计算全有限。

| 版本 | 实际结构与变化 | 本轮已测结果 / 取舍 |
| --- | --- | --- |
| v2 | M64N256 Pingpong，64×256 epilogue | M257 normal 有 53 个 FP8、2 个 scale 差异；large 35 个 FP8、scale 无差；拒绝计时 |
| v3 | v2 加官方 block-scale 乘法顺序 | 12 个数值 case 全过，包括 M32768 上界及换输入 Graph；流程明显变慢 |
| v4 | M32N256，32×256 epilogue，保留 v3 乘法顺序 | 主记录 8 个 case、补充 4 个 case 全过，覆盖所需 M、特殊值与 Graph；spill 消失，流程仍变慢 |
| v5 | 尝试 M64N256 Cooperative，跨过原 M≥128 支持断言 | 普通BF16诊断全等、融合输出错误；原失败库无候选计时，SF128错误后由v6修正 |
| v6 | v5的Cooperative SF128归约改为两个匹配warp，保留正确gate/up配对 | 12个数值case及普通形状换输入Graph全过；大M完整算子仍慢，真实模型配对prefill慢7.09%，不采用 |

v2 的差异已通过同 tile 普通 BF16 GEMM 隔离：M257 normal 有 **791/8,947,712** 个 BF16 bit 差异，随后喂 V2 或官方量化，完整复现候选的 53 个 FP8、2 个 scale 差异；五种输入下候选都与同 tile BF16 再量化逐位一致。固定 CUTLASS 的 N128 路径先计算 `A_scale*B_scale` 再乘 MMA 结果，N256 通用路径却按 `(tmp_accum*A_scale)*B_scale` 计算。括号顺序改变 FP32 舍入及溢出，足以改变后面的 BF16 舍入。v3 在自有构建目录生成 shadow header，恢复官方表达式顺序，未覆盖安装的 CUTLASS。

该定位也解释了“scale 相同仍错字节”：normal 的 53 个差异中 33 个在 scale 逐位相同的组；tiny 虽有 840 个 BF16 差异，却被最终量化遮蔽。特殊组 row4/column3349 的官方 gate/up 仍有限，同 tile N256 输出已为 -Inf/-Inf；两边 scale 都为 +Inf，最终分别得到 0 与 +448。数值与源码证据见 [逐位诊断记录](evidence/gemm-epilogue/candidate-v2-diagnostic.jsonl)、[独立审查](evidence/gemm-epilogue/numerical-diagnostic-review.md)、[scale 顺序补丁说明](evidence/gemm-epilogue/scale-order-v3.md)。

v5 的失败与此不同：M257 普通 Cooperative BF16 诊断在五种输入均复现官方 BF16，其后官方/V2量化也逐位一致；融合 normal 输出仍有 **2,701,283 个 FP8、23,251 个 scale** 差异。普通 BF16 的正确性不能证明更换 warp 组织后的融合访客配对、归约和 scale 映射正确。见 [v5 诊断](evidence/gemm-epilogue/candidate-coop-v5-diagnostic.jsonl)、[完整失败记录](evidence/gemm-epilogue/candidate-coop-v5-full.jsonl)，这些失败记录保持原样，未改门槛。

后续identity访客跳过SiLU仍发生大面积错误，进一步将问题指向SF128量化/归约布局，而非SiLU本身；该诊断的数学参考只是辅助，不作为正式等价门槛。[identity记录](evidence/gemm-epilogue/identity-visitor.json)保留比较对象及库哈希。v6将Cooperative的SF128参与者从4个warp修正为2个warp，用`warp_idx ^ 4`取得同一行/组的对应warp，MMA gate/up配对本来正确。v6编译元数据和生成shadow header记录这一修改；原Cooperative M≥128断言已在自有构建中改为M≥64，仍属于跨上游支持域的实验。实际 [v6全量probe](evidence/gemm-epilogue/candidate-coop-v6-full.jsonl)的8个普通M（含小M回退与32768上界）和4个M257特殊case全部通过，FP8、scale、down GEMM均零bit差，普通case换输入Graph全部通过。这些结果覆盖所测输入，不能扩大为全输入/全支持域证明。

另外使用v6实际生成header的CPU CUTE probe检查epilogue经r2s retile后的64×256 tile，而非只观察MMA分片：**8192对gate/up全部同线程，且up寄存器索引恰为半片偏移，缺失/重复坐标均为0**。记录列出的行/组warp mask为17/34/68/136，各恰有2个warp，支持`warp^4`配对，见 [retile映射](evidence/gemm-epilogue/epilogue-layout-v6.json)。早期 [coop layout编译失败](evidence/gemm-epilogue/epilogue-layout-coop.stderr)是误用不含`paired_quant`类型的BF16 debug header，不能算候选数值失败；换为正确v6 header后该CPU检查编译、运行成功。CPU映射证据仍不替代上述实际GPU全bit检查。

## 4. 无 profiler 的完整算子计时

以下为合成 FP8 输入和权重、整段 Graph replay 的 CUDA Event 中位数，单位 **ms**；每格同时包含 GEMM 与量化/pack。warm 是重复同一输入，权重并不都能放进 L2；cold 在 event 前填充自有 >2×L2 缓冲，只近似驱逐缓存，填充与预排不计入时间。每一对测量随机交错两臂次序，常规记录 31 对，保留原始样本。

| 版本、M | warm 基线 / 候选 | cold 基线 / 候选 | warm 配对基线÷候选中位数 | 对数 |
| --- | ---: | ---: | ---: | ---: |
| v3，257 | 0.246 / 0.627 | 0.256 / 0.635 | 0.393 | 31 |
| v3，2741 | 1.756 / 5.110 | 1.757 / 5.113 | 0.344 | 31 |
| v3，2786 | 1.769 / 5.186 | 1.769 / 5.190 | 0.341 | 31 |
| v3，16160 | 10.366 / 29.372 | 10.383 / 29.400 | 0.353 | 31 |
| v4，257 | 0.233 / 0.287 | 0.250 / 0.295 | 0.830 | 31 |
| v4，2741 | 1.675 / 2.474 | 1.745 / 2.562 | 0.676 | 31 |
| v4，2786，补充窗口 | 1.589 / 2.446 | 1.602 / 2.459 | 0.650 | **9** |
| v4，16160 | 10.380 / 16.950 | 10.376 / 16.941 | 0.612 | 31 |
| v6，257 | 0.254 / 0.254 | 0.258 / 0.250 | 1.006 | 31 |
| v6，2741 | 1.777 / 1.950 | 1.800 / 1.971 | 0.909 | 31 |
| v6，2786 | 1.802 / 2.015 | 1.802 / 2.014 | 0.894 | 31 |
| v6，16160 | 10.509 / 12.606 | 10.525 / 12.623 | 0.833 | 31 |

配对比值小于 1 表示候选慢。各版在不同窗口运行，应以各自同窗口基线判断，不能把跨窗口的绝对时间变化归因给代码。v4 的 M2786 来自明确标注的 9 对补充记录，不能伪称全表均为 31 对；M1/5/256 的候选臂实际走同一基线回退，不属于融合速度结果。M32768 的上界检查也通过，完整数据保留于 [v3](evidence/gemm-epilogue/candidate-v3-full.jsonl)、[v4 主记录](evidence/gemm-epilogue/candidate-m32-v4.jsonl)、[v4 补充记录](evidence/gemm-epilogue/candidate-m32-v4-extra.jsonl)、[v6](evidence/gemm-epilogue/candidate-coop-v6-full.jsonl)。v6的M257接近持平，但代表性大M仍慢；这个单窗口边界结果不能代替大prefill或整模加速。

## 5. NCU：删掉 BF16 不等于更快

固定 GPU 的 NCU 2026.3 实际指标名是 `dram__bytes_op_read.sum` / `dram__bytes_op_write.sum`，旧 `dram__bytes_read/write.sum` 不可用。指标选择、查询环境及来源见 [指标清单](evidence/gemm-epilogue/ncu-metric-selection.json)。初次查询因 `ERR_NVGPUCTRPERM` 失败，随后只给自有 profiling 容器 `SYS_ADMIN`，未改变宿主驱动设置。

M2741 各自一个完整 pipeline 窗口：基线 IDs0/1 为 GEMM+V2量化，候选 IDs2/3 为 fused GEMM+pack，每臂两 kernel。下面是跨整臂求和，DRAM/L2 单位为十进制 MB；比值的分母都是同窗口的一次完整基线 pipeline。

| 窗口 / 臂 | DRAM 读 MB | DRAM 写 MB | 总 DRAM MB | L2 请求 MB | kernel 时长和 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 窗口基线 | 384.369 | 176.106 | 560.476 | 8,414.428 | 1.621 |
| v3 候选 | 243.689 | 244.287 | 487.976 | 37,858.608 | 4.891 |
| v4 窗口基线 | 384.430 | 175.011 | 559.440 | 8,414.469 | 1.628 |
| v4 候选 | 244.276 | 71.422 | 315.698 | 17,545.542 | 2.462 |
| v6 窗口基线 | 383.768 | 175.743 | 559.512 | 8,413.071 | 1.634 |
| v6 候选 | 243.622 | 71.325 | 314.947 | 9,865.578 | 1.806 |

v3 的候选/基线 DRAM 总量为 **0.871×**，L2请求为 **4.499×**，时长和却为 **3.017×**；v4 分别为 **0.564×、2.085×、1.512×**；v6为 **0.563×、1.173×、1.105×**。即使 DRAM 交通量下降，吞吐也没有跟着改善。原始范围与求和见 [v3 NCU](evidence/gemm-epilogue/ncu-v3-m2741/summary.json)、[v4 NCU](evidence/gemm-epilogue/ncu-m32-v4-m2741/summary.json)、[v6 NCU](evidence/gemm-epilogue/ncu-coop-v6-m2741/summary.json)。

资源诊断给出了更具体的机制：基线、v3、v4 GEMM 都报 REG168；静态 stack 分别为 8、568、0 字节。`LOCAL:0` 这一静态字段不能排除运行时 spill。M2741 NCU 的 v3 GEMM 各产生 **440,143,872 个 local load/store sector**，按 32 B/sector 各约 14.08 GB 请求；v4 两项都为 0。v3 的大量 local 请求与 stack 支持 spill 诊断，M32 v4 确实清除了这个实测问题。

但 v4 的 GEMM L2 请求仍为 **17.43 GB**，基线为 **8.16 GB**；Grid 从基线 `(22,272)` 变为 `(86,136)`，CTA 数约翻倍。v4 GEMM DRAM 读约 **193.54 MB**，与基线 **193.53 MB** 几乎相同，却用 2.342 ms 对比 1.493 ms。更小 M tile 增加 CTA 工作和缓存请求，是实测事实；这些数据没有单独量化每一种余下成本。GEMM 的 SM throughput 指标也从基线 82.12% 降至 v4 42.65%，不能将其或占用率直接换算为整模速度。

v6的Cooperative GEMM同样为 **0个local load/store sector**，完整流程仍慢：GEMM **1.685248 ms** 对比基线 **1.498016 ms**，pack **0.120384 ms** 对比原V2量化 **0.135712 ms**。pack省下的时间没有抵消GEMM增加的时间。GEMM SM throughput为 **67.46%** 对比基线 **81.96%**，活跃warp占比为 **20.90% / 20.82%**；两个占比接近，不能把剩余时长简单归为“占用率太低”。v6的总L2请求已接近基线但仍高17.26%，具体剩余成本未被这些计数器完全拆开。以上都是M2741该一次profile窗口的实测，没有据此预测v6整模收益。

NCU 为 kernel replay，每核记录 **8 passes**；采用 `--cache-control none --clock-control none`，缓存及 GPU 时钟未锁定，数值不能直接替代无 profiler 计时。每臂的百分比指标保留逐 kernel 解释，不相加。M16160 的 v3 NCU 尝试进程返回 **11**，没有成功的完整 pipeline 汇总，不能将 M2741 计数器外推为 M16160 实测。[失败状态](evidence/gemm-epilogue/ncu-v3-m16160/status.json)已保留。

## 6. 图集成与真实无 profiler A1/B/A2

实际 FX 图确认 `cutlass_scaled_mm` → BF16 `getitem` → V2 激活量化；BF16 还被 `sym_size(dim=0)` 使用。替换需把形状依赖改到同 M 的 A，并验证唯一数据消费者、完整布局与启动登记的 canonical B/B scale 身份，不能只凭 M/N/K 猜目标层。M≤256、未登记权重、guard不符都回退。FP8 packed副本在加载后、warmup与编译前生成；fake/meta、动态形状与 custom op 序列化也有单独检查。

[v3 runtime 合成冒烟](evidence/gemm-epilogue/runtime-smoke-v1.json)确认注册 M257 调用候选、未注册 M257 和 M5 正确回退。整模接入曾因把旧 `auto_functionalized` 六参数调用的 `eager_input_vals` 元数据带给新四参数 op 而启动失败；清理该输入元数据、只保留输出/provenance 后，固定镜像 CPU Inductor 自检与实际 v4 模型启动通过。该失败不属于候选数值结果。

[实际集成汇总](evidence/gemm-epilogue/model-witness-v4-meta-fixed/integration-summary.json)确认65个模块的权重身份、21张图中20次FX rewrite，M2741/2772/16160/32768各有65条候选 Python 首次派发记录，并保存多个小M回退。新增 packed副本为前述11,589,593,600 B；观察到累计 peak allocated **66,841,412,096 B**、peak reserved **69,883,396,096 B**，这是启动及首次派发范围的累计峰，**不是每次请求峰值**。例如M16160的一次候选持有 paired FP8暂存562,626,560 B、native scale暂存17,686,528 B、查询的CUTLASS workspace 1 B；这些资源按调用所有，不按65层固定预留大workspace。

正常模式长prefill的 [Nsight Systems witness](evidence/gemm-epilogue/model-witness-v4-meta-fixed/dispatch-summary.json)实际捕获 **130次融合GEMM + 130次pack**：M16160和2741两个chunk各65组，全部260个kernel与成功的 direct CUDA launch精确关联。该捕获 **0个graph node ID，0次匹配的CUDA Graph replay launch**。因此已证明真实模型走到候选，两阶段均实际执行；不能宣称大M候选在该服务请求中经Graph重放。合成probe的Graph正确性与模型的direct派发是两个独立口径。Python记录也不能替代kernel或replay计数。

v6也独立完成相同边界的正常模式模型检查：[集成汇总](evidence/gemm-epilogue/model-witness-v6/integration-summary.json)为65模块精确权重身份、21张图20次rewrite；[Nsight witness](evidence/gemm-epilogue/model-witness-v6/dispatch-summary.json)捕获 **130融合GEMM+130pack，全部260次成功direct关联，0次Graph replay**。所以v6实际命中已证，不能将没有发生的服务大M graph replay写成优化效果。这份profiler捕获的kernel累计时长也不充作无profiler API延迟。

接入修复后重新跑完整三臂：A1为官方GEMM+V2量化，B为通过数值门槛的v4，A2重复同一基线；三臂镜像、源码instrument清单、模型、MTP4、FP8 KV、FlashInfer、12GiB KV预算及65份packed副本一致，B库SHA256为前述`36df0b9c…748d`。并发1，固定合成性能请求逐索引payload hash相同，每臂12次约19K/1输出prefill和6次约4.8K/1024输出decode；每类各一条warmup单独剔除。早期 [a1记录](evidence/gemm-epilogue/a1/runtime.json)保留为探索证据，不混入这次重新采样的 [aba-v4分析](evidence/gemm-epilogue/aba-v4/analysis.json)。

| v4无 profiler 指标、中位数 | A1 基线 | B v4候选 | A2 基线 | 正式样本/臂 |
| --- | ---: | ---: | ---: | ---: |
| 约19K/1输出首文本时间，s | 2.250230 | 2.769921 | 2.263048 | 12 |
| 同组服务端prefill time，s | 2.234177 | 2.753120 | 2.246902 | 12 |
| 约4.8K/1024输出首文本时间，s | 0.510354 | 0.618578 | 0.512675 | 6 |
| 同组decode估计速率，tok/s | 106.74950 | 106.73051 | 106.74134 | 6 |

对每个相同索引，先以`(A1_TTFT+A2_TTFT)/2`作为两次基线均值，再计算相对B的减幅。12组长输入减幅中位数 **-22.7454%**，即候选首文本时间**慢22.75%**；12个索引全部变慢，范围约慢22.42%–22.99%。这是完整三臂配对统计，不是只拿B除以一次A1的约23%。约4.8K请求的首文本时间也配对慢 **20.95%**，但后续小M decode回退，生成速率基本不变。

该时段三臂全部正式请求有效、SSE正常结束，每个metrics窗口恰有1个目标服务成功请求；prefill和decode均无prefix命中、无preemption。约19K/1输出没有发生MTP draft，不能拿它解释接受率；6个decode请求的draft轮数、draft token和accepted token在三臂**逐索引相同**，每个请求accepted/draft轮数的中位数均为 **2.059638 token/轮**，每轮配置4个draft token。这轮没有观察到接受率变化。

12/12长输入配对、6/6decode配对的embedding前后计数器均无变化且无缺失，所以embedding静默子集与全体统计一致。该证据只覆盖快照间已完成请求的计数变化，不能根据两端运行gauge为0排除所有瞬时活动，也不构成embedding延迟测量。

6/6个decode请求的**非空正文**hash在A1/B/A2相同；reasoning均为空，其相同空hash不计作思考质量证据。12个单token prefill输出hash相同只证明该一个token。三臂每臂5项质量检查和4项开放输出检查都通过，均`finish_reason=stop`且有最终正文。五项quality是固定脚本的同名案例各自通过，原始记录没有保存payload hash，不能证明所捕获请求字节跨臂一致；修正后的分析明确记为`payload_hash_unavailable`/null，未将多个空值当成hash相同。四项开放输出则有完整、相同的跨臂payload hash，开放算术的两次思考输出为非空，正文和reasoning hash在三臂一致。长检索cold无cache命中，重复检索各命中42,016 token，证明这项合成prefix复用仍工作；它没有证明全面cache或业务质量无损。

这次固定请求对照未测真实外部工具执行、MoBase业务整链路、并发吞吐或独立重复时间窗口。分析脚本的配对bootstrap只重采样prompt索引，其区间是描述性的，不包含重启/时段波动，也不是因果置信区间。正常模式真实派发与同索引内容/接受率一致，使“候选已实际执行且该组prefill变慢”的结论可核验；当前性能结果仍应限定在这组固定栈、固定请求和采样时段，不能外推到其他负载。v4 据此拒绝采用。

v6的独立无profiler三臂已完成，见 [aba-v6分析](evidence/gemm-epilogue/aba-v6/analysis.json)。镜像、运行参数、12GiB KV预算、65份packed副本和共用运行时源码一致；A1/A2不调用挂载的`libgateup.so`，该未使用库仍为v4 SHA，B实际使用冻结的v6 SHA。原始三臂manifest除这个候选库外没有hash差异，不能机械要求未执行库与B候选库相同。

| v6无 profiler 指标、中位数 | A1 基线 | B v6候选 | A2 基线 | 正式样本/臂 |
| --- | ---: | ---: | ---: | ---: |
| 约19K/1输出首文本时间，s | 2.251251 | 2.415334 | 2.260911 | 12 |
| 同组服务端prefill time，s | 2.234124 | 2.399108 | 2.244156 | 12 |
| 约4.8K/1024输出首文本时间，s | 0.511320 | 0.541800 | 0.512516 | 6 |
| 同组decode估计速率，tok/s | 104.56239 | 106.67972 | 106.66333 | 6 |
| decode accepted token / draft轮，中位数 | 2.004399 | 2.059638 | 2.059638 | 6 |

沿用同索引两次基线均值作为分母，12组长输入TTFT减幅中位数 **-7.0872%**，即慢 **7.09%**；全部变慢，范围慢6.96%–7.21%。约4.8K输入首文本时间配对慢 **5.89%**。三臂正式性能payload逐索引相同，全部请求有效、metrics窗口各1个目标成功请求，无prefix命中与preemption；prefill没有MTP draft，12/12prefill和6/6decode配对embedding计数静默、无缺失。快照和bootstrap的限制与v4相同，没有独立重复时段。

v6的decode须保留输出漂移边界：分析汇总显示A1/A2 **6/6正文hash不同**，A1/B也6/6不同；本报告另对 [B原始decode JSONL](evidence/gemm-epilogue/aba-v6/b/decode.jsonl) 与 [A2原始JSONL](evidence/gemm-epilogue/aba-v6/a2/decode.jsonl)逐索引核对，所存正文hash **6/6相同**。MTP draft轮/token/accepted token增量亦为B与A2逐索引6/6相同，而A1与A2六个索引都变化。速率和接受数据因而不能仅归因于候选；既不能宣称decode增益，也不能把合成算子bitexact扩大为整模bitexact。reasoning均为空，其相同hash依旧不计作非空思考质量。

v6三臂的每臂5项合成质量与4项开放输出检查均通过、正常stop并有最终正文；五项quality同样只证固定脚本同名案例各自通过，未保存payload hash，不宣称捕获输入字节一致。四项开放输出的有效payload hash跨臂相同，开放算术的两次非空reasoning及正文hash三臂一致。长prefill单token正文hash12/12相同只覆盖一个token。这些有限检查未发现该测试的事实/格式问题，但不是更广模型或MoBase业务质量无损证明。实际大M候选派发已证，配对prefill仍慢，v6据此不采用；前述decode漂移保持原样记录，不调整数值门槛或删掉A1结果。

## 7. 服务策略与本轮结束状态

本轮按用户明确要求暂停原 27B、9B、ASR，**保持停用，等待用户明确要求启动**。embedding 不在暂停范围。已保存 [原始状态](evidence/gemm-epilogue/initial-state.json)、[停用次序](evidence/gemm-epilogue/pause-sequence.json)、[停用后检查](evidence/gemm-epilogue/service-state-after-pause.json)与[继续实验前检查](evidence/gemm-epilogue/resume-service-check.json)：原容器身份/镜像/命令/重启计数保持；embedding 的容器、启动时间、重启计数与原始状态相同，health 200，实际返回1024维有限向量。这些是对应时点的功能快照，不是持续零干扰或延迟 SLA 证明。

已在16:18（UTC+8）保存 [清理记录](evidence/gemm-epilogue/owned-cleanup.json)：只移除 **11个已经停用的本轮自有实验容器**，各自profiles保留；不操作原服务容器。随后 [结束服务检查](evidence/gemm-epilogue/final-service-check.json)确认原27B/9B/ASR仍停用，容器/镜像/命令与重启计数保持原记录；embedding保持原容器、启动时间和重启计数，health及实际embedding请求均200，返回1024维有限向量。该结束快照已通过，不把停用中的服务写成已恢复。

在结束功能检查后 **96.61秒**， [延后服务检查](evidence/gemm-epilogue/final-service-check-later.json)仍全部符合该政策：原27B/9B/ASR停用且身份/计数保持，embedding运行、health200、启动时间和重启计数保持。不会在用户未要求时自动恢复原三个服务。

[embedding监控汇总](evidence/gemm-epilogue/embedding-monitor-summary.json)核对 [原始采样](evidence/gemm-epilogue/embedding-monitor.jsonl)：**274次样本全部running、health200**，覆盖约 **136.57分钟**，最大采样间隔 **30.892秒**；只出现原容器ID、原启动时间及原重启计数1。该计数1是实验前已有记录，本轮未增加。周期快照、实际vector功能和延后状态互相补充，但仍不是连续请求级可用性证明，也未测embedding延迟或给出零干扰SLA。

## 8. 重现方法与适用边界

代码和证据集中于 [gemm-epilogue](evidence/gemm-epilogue/)。固定镜像 digest 为 `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`，vLLM 提交为 `db9527a46873454610df6dbedf79a36d6bf1a7f6`；不能用“同为0.31”代替源码锁定。v3 binary SHA256 为 `b2eb8ecd01ab0f324c800bca9038cad17672336d563fa2dde203cfe1bf91975f`，v4 为 `36df0b9c1d864cf3c65654c7162a65f0bc84b7489ab0e139928919972aca748d`，v6为 `e4039ddc95fa3b56f02743357b7a7193156dc500c5d8804bb77ebd3af85450d6`，见 [v6构建记录](evidence/gemm-epilogue/candidate-build-coop-v6.log)。构建只改自有 shadow include，不修改生产镜像或checkpoint。

重现顺序是：核对固定源码哈希并构建 → 先 `--baseline-only` 复现官方 → 完整候选逐位/Graph门槛 → 同窗口完整pipeline warm/cold → 一次完整pipeline NCU窗口 → 隔离临时模型真实派发与无profiler A1/B/A2。原服务生命周期由主线程控制；数值probe不启停服务，A/B/A运行器只启停本轮自有临时实验容器。性能请求发往隔离实验API。

以下是已实测 v4 库在固定镜像、GPU保留窗口内的数值/计时入口，路径为实验容器挂载路径，不是 macOS本地GPU命令：

```sh
python3 /work/gemm-epilogue/pipeline_probe.py \
  --baseline-library /baseline/libtargeted_fp8.so \
  --quant-library /v2/libfp8_silu_quant_group_major.so \
  --candidate-adapter /work/gemm-epilogue/candidate_adapter.py \
  --candidate-library /work/build-gateup-candidate-m32-v4/libgateup_quant_fp8.so \
  --candidate-sha256 36df0b9c1d864cf3c65654c7162a65f0bc84b7489ab0e139928919972aca748d \
  --m 257 2741 16160 32768 --repeats 31
```

该命令还运行默认四组特殊M257；M2786与小M需按原补充记录另跑。v6同接口换用`/work/build-gateup-coop-v6/libgateup_quant_fp8.so`与对应SHA256，并指定`--m 16160 2741 2786 256 257 1 5 32768 --repeats 31`，对应其12个case记录。数值诊断用 `--diagnostic-gemm-library` 配普通同 tile BF16 probe 和SHA，不能把编译成FP8 D的可行性库当BF16诊断。NCU 加 `--ncu --m 2741`，使用 [NCU运行器](evidence/gemm-epilogue/ncu_pipeline_control.py)记录完整命令、权限、replay/cache/clock策略；用 [求和脚本](evidence/gemm-epilogue/summarize_ncu_pipeline.py)按已从 NVTX 窗口确认的 kernel IDs0/1 与2/3汇总。

适用条件限 SM120、当前 dense gate/up 的 FP8 block布局、无额外BF16数据消费者、`256<M≤32768`以及精确匹配的目标模块；TP、offload、LoRA、其他模型/布局、其他架构、更大M和更广业务均未证明可用。逐位合成检查与小范围API质量不能代替模型级业务验收；微基准或DRAM下降也不能代替TTFT、吞吐与并发结果。v3/v4/v6均因性能未过保留为实验，v5因数值未过保留为失败定位材料；两轮真实模型prefill对照均未获收益，本轮不覆盖生产。
