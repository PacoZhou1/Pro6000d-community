# RTX PRO 6000 定向 GEMM：归因、计数器与候选筛选（2026-10-14）

本轮已把固定 vLLM 0.31.0 的真实 MTP4 请求中，若干 FP8 GEMM 和 BF16 GEMM 对到模块及矩阵尺寸；在相同尺寸的**合成单算子**上测量了硬件计数器和三种 FP8 候选。大 M `gate_up_proj` 的 tile 改动及 Split-N 均未显示可采纳的收益。`down_proj` 小 M 的 Stream-K 在 warm 重放下更快，在近似 cold 条件下更慢；三臂整模 A/B 和真实 CUDA Graph 重放取证已完成，**未显示稳定整模收益**。本轮没有将任何候选部署到原服务。

## 固定环境与三种不同分母

诊断模型为 Qwen3.6-27B-FP8，MTP4、FP8 KV、FlashInfer、单卡 RTX PRO 6000 Blackwell Workstation Edition。固定镜像 `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`，vLLM 0.31.0 提交 `db9527a46873454610df6dbedf79a36d6bf1a7f6`。固定 CUTLASS dispatch 头 `scaled_mm_blockwise_sm120_fp8_dispatch.cuh` SHA-256 为 `8a9ee5dc8edb96bc4619883a6248eb15211fd37db2d3fc02de25c5b02ee5af41`，由[定向实现脚本](evidence/gemm-targeted/targeted_kernel.py)校验，并记录在[构建清单](evidence/gemm-targeted/build/build-manifest.json)；本地提取的其他源码哈希见[源码清单](evidence/gemm-targeted/source-manifest.json)。准确启动配置和常规 CUDA Graph 路径见[上一轮分析](08-gpu-profiling.md)及其 [runtime](evidence/profiling/mtp4-runtime.json)。

下文的百分比有不同分母，不能互相加总，也不能当作请求延迟：

| 证据 | 分母及用途 | 限制 |
|---|---|---|
| 正常 CUDA Graph 的 Nsight Systems | 每份捕获中**所有选定 GPU kernel 的 duration 之和**；按原始 kernel 名推断 FP8/BF16 家族 | 不包含 CPU/网络、调度间隙；名称不能独自证明模块。 |
| `--enforce-eager` 的 Nsight Systems + `GEMM_ATTRIB` NVTX | 每份 eager 捕获中**所有选定 GPU kernel 的 duration 之和**；NVTX → 同线程 CUDA API → `(globalPid, correlationId)` 精确关联到模块、M/N/K | 仅诊断模块归属；不能把 eager 的毫秒或份额替换为正常 graph 的时延、模块份额。 |
| Nsight Compute 与 CUDA event 合成探针 | 一次隔离 GEMM 或两次 Split-N GEMM 的计数/时间；操作数沿用实测形状和 layout，但值为合成值 | 多遍 replay、缓存、内存分配和负载组合不同于整模；不是 API TTFT、tok/s 或质量。 |

正常 graph 的两份 MTP4 decode 捕获中，FP8 名称家族均约占 **62.03%**，BF16 名称 GEMM 约占 **29.50%**；两份 prefill 的 FP8 名称家族约占 **68.49%**。这是同一固定栈的名称分类结果，原始核名、次数与分母见 [MTP4 Nsight 汇总](evidence/profiling/mtp4-nsys-summary.json)。它只给优化优先级，不给模块归因。

## 真实请求中的模块与尺寸

归因实验另起同镜像临时服务并开启 `--enforce-eager`，保留真实 MTP4 请求：prefill 为 18,901 prompt / 1 completion，decode 为 4,783 prompt / 256 completion。前者捕获内有 `execute_context_1(16160)` 与 `execute_context_1(2741)` 两段；后者有 87 个 `execute_context_0(0)_generation_1(5)` stage。分析器只在同线程 stage **完整覆盖** GEMM NVTX 范围时填写 `enclosing_stage`，所以 `lm_head` 等 stage 外调用仍可有精确模块归因，stage 字段为 `null`。元数据来自[实际 trace 命中的 934 种 module+M/N/K 签名](evidence/gemm-targeted/module-signatures.json)，排除了只在启动/warmup JSONL 中出现、没有与这两份 SQLite 相交的签名；缺失元数据数为 0。

| 模块和计算 | 捕获内实测 M×N×K | 操作数与输出 | 归因含义 |
|---|---|---|---|
| target `language_model.model.layers.*.mlp.gate_up_proj` | prefill 16,160 / 2,741 × 34,816 × 5,120；decode 5 × 34,816 × 5,120 | A、权重 B：FP8 e4m3；block scales：FP32；输出 BF16 | 大 M tile 候选的目标。 |
| target `...mlp.down_proj` | prefill 16,160 / 2,741 × 5,120 × 17,408；decode 5 × 5,120 × 17,408 | A、B：FP8 e4m3；scales：FP32；输出 BF16 | 小 M Stream-K 候选的目标。 |
| `language_model.lm_head` / `ParallelLMHead` | decode 1 与 5 × 248,320 × 5,120；prefill 捕获也命中 M=1 | 输入 `[M,5120]`、权重 `[248320,5120]` 均 BF16，输出 `[M,248320]` BF16 | 已证实为 BF16 大 GEMM 的主要模块，不能把正常 graph 的同名核耗时直接全记到它。 |

表中 K/N 顺序为 GEMM 的 M×N×K；权重张量按固定源码存为 `[N,K]`，`gate_up` 例如 M=5 时 A stride `[5120,1]`、B stride `[5120,1]`，A scale `[5,40]` stride `[1,5]`，B scale `[272,40]` stride `[40,1]`。完整 shape、stride、dtype 和各层 prefix 在[模块签名](evidence/gemm-targeted/module-signatures.json)。

eager prefill 捕获有 7,363 个 GPU kernel、duration 合计 2,653.946 ms；其中 712 个 / 1,504.836 ms 通过 correlation 归因。target 64 层 `gate_up` 的 M=16,160、2,741 两组分别为 64 个 / 585.331 ms 与 64 个 / 102.776 ms；`down_proj` 分别为 64 个 / 285.391 ms 与 64 个 / 52.297 ms。eager decode 捕获有 254,913 个 GPU kernel、合计 2,689.095 ms；其中 33,147 个 / 2,258.281 ms 归因。target `gate_up` M=5 为 5,568 个 / 657.049 ms，`down_proj` M=5 为 5,568 个 / 352.241 ms。这些是整份 **eager 捕获 kernel duration 的求和**，不是每请求的 wall time；未归因的 kernel 保留在分母中。[prefill 归因](evidence/gemm-targeted/mtp4-prefill-attribution.json)、[decode 归因](evidence/gemm-targeted/mtp4-decode-attribution.json)。

decode 中显式 BF16 命名 GEMM 共 9,483 个 / 728.651 ms，全部被关联。`lm_head` M=1 有 348 个 / 550.075 ms，M=5 有 87 个 / 136.762 ms；精确数值合计 **686.836 ms，占该 eager 捕获全部 GPU kernel duration 的 25.54%，占 BF16 GEMM duration 的 94.26%**。其他 BF16 包括 `mtp.fc` 约 23.838 ms 和 target `linear_attn.in_proj_ba` 约 17.977 ms。故“BF16 大 GEMM 主要属于 `lm_head`”有 eager 相关性与实际形状证据；正常 graph 中约 29.50% 的 BF16 名称家族仍只按 kernel 名分类，不能直接称为已精确测到 `lm_head` 的 graph 份额。分析器报告 CUDA API **记录数**与唯一 `(pid,correlationId)` 键数，未把 Runtime/Driver 记录数或未匹配 kernel 当成独立 launch。[分析器](evidence/gemm-targeted/analyze_attribution.py)。

## 合成计数器与候选实验

尝试对带真实模型的 eager 容器直接用 Nsight Compute replay 时，容器报告 `OOMKilled=true`、退出码 9，**没有整模硬件计数器**；[失败记录](evidence/gemm-targeted/ncu-model-failure.json)保留。后续改用对应实测尺寸的独立合成算子，内存上限 24 GiB，NCU 每核 8 pass。这限制了关于真实模型 cache/DRAM 的结论。

实际安装并使用的工具为 **Nsight Compute 2026.3.0.13**，见[安装记录](evidence/gemm-targeted/ncu-install.json)。合成 `lm_head` BF16 M=1/5：NCU 计时约 1.594/1.581 ms，DRAM sustained peak 比率约 93.64%/94.70%，L2 命中率约 4.99%/12.72%，SM throughput 约 13.94%/23.50%。这提示**该合成大词表 GEMM**在此操作数与 replay 条件下更接近 DRAM 限制，不证明正常服务中的 `lm_head` 带宽瓶颈。合成 FP8 `gate_up` M=16,160/2,741：DRAM 比率约 14.36%/13.71%，L2 命中约 96.70%/96.47%，SM throughput 约 88.45%/86.31%；小 M=5 的 DRAM 比率则约 91.31%。合成 `down_proj` M=16,160 的 DRAM 比率约 7.63%、L2 命中约 98.00%、SM throughput 约 86.97%；M=5 常规计数的 DRAM 比率约 85.45%、L2 命中约 20.33%，另有显式 `--cache-control all` 的 M=5 采样。小 grid 的 active-cycle 派生值在原始采样中出现超过 100% 的异常，不能据此解释 Tensor pipe 利用率或候选收益。相反的缓存/吞吐特征说明不能用一个形状概括整个 FP8 家族，更不能将反复使用同一合成权重的 L2 命中外推到 64 层真实权重流。[八组 NCU 指标摘要](evidence/gemm-targeted/counters/summary.json)、[实际测量口径](evidence/gemm-targeted/counter-method.md)。

首个候选只在 `M>256,K=5120,N=34816` 的独立 SM120 block-FP8 dispatch 使用 64×128×128 Pingpong tile，其他 M 回落固定路径。合成 `gate_up` M=2,741 的 warm 中位数从基线 **1,647.3 μs 升至 1,813.2 μs**；M=16,160 从 **9,298.9 μs 升至 10,886.1 μs**。小 M 的候选实际回落，不能记为 tile 的性能证据。基线独立库全输出与固定 vLLM 算子 bit-equal；候选检查了全输出有限性和抽样行/N block 对比，但未做全模型质量验证。[tile 数值与计时](evidence/gemm-targeted/micro-1/results.jsonl)、[首个候选构建](evidence/gemm-targeted/build/build-manifest.json)。

第二个候选把 N=34,816 拆成两个 N=17,408 GEMM。合成 M=2,741 的 warm `original_official` **1,617.7 μs**，Split-N 两 GEMM 加 `torch.cat` 的 `split_official_auto_e2e` **1,762.6 μs**；M=16,160 分别为 **9,159.5 μs** 与 **10,303.9 μs**。仅测两 GEMM、不含拼接的结果不能替代端到端算子比较。两个拆分路径的 swizzle、输出分配与缓存行为不同；所测 6 个 M 的结果只支持“当前 Split-N 实现无可采纳收益”，不证明所有 Split-N 策略都慢。数值检查为合成全输出与固定算子相等。[Split-N 数值与计时](evidence/gemm-targeted/micro-splitn/results.jsonl)、[探针](evidence/gemm-targeted/splitn_probe.py)。

第三候选把小 M `down_proj` 的调度换为 Stream-K，目标 `M≤16,K=17408,N=5120`，固定 tile 128×32×128 和原 scale；库 SHA-256 `ecad960defeeffa1d6fb1faae4998650a07fd2404fe8acf569f5dc6f159529bd`。合成探针对 M=1、5、16 的实验路径实际启用，M=17 回落固定路径。固定独立库的全输出与 vLLM 算子 bit-equal；Stream-K 的 M=1 与基线 bit-equal，M=5 全输出 NRMSE **9.63×10⁻⁸**、最大绝对差 **1.53×10⁻⁵**，M=16 全输出 NRMSE **3.62×10⁻⁵**、最大绝对差 **0.0078125**。四个 M 的合成检查均 `pass=true`，这并非模型质量验收。[构建状态](evidence/gemm-targeted/build/down-streamk-build.json)、[数值与计时原始结果](evidence/gemm-targeted/micro-streamk/results.jsonl)、[候选探针](evidence/gemm-targeted/down_streamk_probe.py)。

| 合成 `down_proj` | 固定 warm → Stream-K warm，中位数 | 固定 cold → Stream-K cold，中位数 |
|---|---:|---:|
| M=1 | 36.54 → 23.16 μs | 74.72 → 78.85 μs |
| M=5，真实 decode 尺寸 | 36.65 → 26.64 μs | 75.78 → 79.34 μs |
| M=16 | 36.56 → 22.53 μs | 74.72 → 79.84 μs |

warm 是固定操作数的 CUDA Graph 重放；cold 是由本进程 >L2 缓冲区刷新做的**近似**缓存驱逐，并非真实 64 层权重流。M=5 的 warm 单算子 duration 约降 27.3%，cold 约升 4.7%；由此只能说它对缓存状态敏感，不能宣布整模收益。M=17 不走候选，不能算入加速样本。[执行状态与 embedding 前后检查](evidence/gemm-targeted/micro-streamk/status.json)。64×32×128 tile 曾实际试编译并触发 CUTLASS 的 M scale granularity 静态断言；32×32×128 **没有实际编译**，只是因相同粒度约束被排除，不能称为第二次编译失败或性能结果。

### 无 profiler 的整模 API 对照

采用固定 CUTLASS A1 → **修正版 Stream-K v2** B → 再次固定 CUTLASS A2 的顺序。三臂均完成 prefill、普通 decode、thinking 各 1 次 warmup + 3 次正式请求；各 12/12 请求有效，五项短质量探针与四项开放输出探针各自全通过。三类正式请求的 payload SHA-256 均按 case/index 一一相同，prefix-cache hits 为 0，测量窗口内 embedding 请求计数未变；三个容器使用相同固定镜像及服务参数，B 另外加载实验 custom op。[三臂统计与逐请求配对](evidence/gemm-targeted/api-summary.json)、[A1 性能](evidence/gemm-targeted/api-baseline/performance.jsonl)、[B 性能](evidence/gemm-targeted/api-streamk-v2/performance.jsonl)、[A2 性能](evidence/gemm-targeted/api-baseline-repeat/performance.jsonl)。

| 指标（各臂 n=3，中位数） | A1：固定 CUTLASS | B：Stream-K v2 | A2：固定 CUTLASS |
|---|---:|---:|---:|
| prefill 首文本时间 | 2.25355 s | 2.24537 s | 2.26051 s |
| 普通 decode 估计速率 | 108.99 tok/s | 104.91 tok/s | 107.34 tok/s |
| thinking decode 估计速率 | 129.90 tok/s | 121.49 tok/s | 121.08 tok/s |

B 的 prefill 首文本时间比 A1 低约 0.36%、比 A2 低约 0.67%，绝对差只有约 8–15 ms；普通 decode 速率分别比 A1/A2 低 **3.75%/2.27%**。thinking 速率比 A1 低 **6.47%**，但比 A2 **高 0.34%**，接近这三次样本的波动，不能概括为所有 decode 都退化。速率是客户端对完整请求的估计值，不是 NCU 单核时间。三臂 prefill 正文哈希逐项相同；普通 decode 正文、thinking reasoning 在 B 与 A1 对比均 **0/3 相同**，而 **A1 与 A2 两次固定基线之间也均 0/3 相同**。各请求的 MTP accepted-per-draft 也有变化；相同输入未产生相同生成工作，因此不能将长生成的速率差严格归因于 Stream-K kernel，更不能据此断言模型质量退化。短质量与开放输出探针全通过，但不等于长生成逐 token 等价。[A2 质量](evidence/gemm-targeted/api-baseline-repeat/quality.jsonl)、[B 质量](evidence/gemm-targeted/api-streamk-v2/quality.jsonl)、[A2 开放输出](evidence/gemm-targeted/api-baseline-repeat/open-output.jsonl)。

旧 `api-streamk` guard-v1 虽跑完请求，其 compiled graph 中实验 op 为 **0 节点**，日志也没有实验 launch；这是一条**无效候选臂**，不纳入上表。[guard 缺失证据](evidence/gemm-targeted/api-streamk/runtime-guard-miss.json)。v2 的保存版 compiled graph 中有 **64 个** `torch.ops.pro6000d_exp.down_streamk` 调用节点、原 CUTLASS 调用节点为 192 个；独立运行时 smoke 的 8 项检查均通过，包括对新输入的合成 CUDA Graph replay。[compiled graph 证据](evidence/gemm-targeted/api-streamk-v2/compiled-graph-proof.json)、[运行时 smoke](evidence/gemm-targeted/runtime-smoke-v3/results.jsonl)。

随后另起相同 v2 补丁、正常图模式的 Nsight Systems 诊断服务，在首文本之后捕获 256-token decode。实际捕获到 **5,395 次 Stream-K kernel**，每次都带非零 `graphNodeId`，且按同一进程/correlation ID 关联到成功的 `cudaGraphLaunch`；共 166 个不同图 launch，kernel grid 为 **188×1×1**。这证明候选确实参与模型图重放，未把启动或 capture 日志当作 replay 证据。[重放证明](evidence/gemm-targeted/streamk-replay/replay-proof.json)、[真实请求](evidence/gemm-targeted/streamk-replay/requests.jsonl)、[关联脚本](evidence/gemm-targeted/analyze_replay.py)。

补丁按 M/N/K 和布局分流，因而也覆盖同尺寸的 `mtp.layers.0.mlp.down_proj` M=5；不是只按 target module prefix 过滤。该 MTP 形状在独立 eager 归因中确有 87 次调用。主干编译图的 64 节点数只描述主干，不能当作整份 trace 中候选调用数的分母。短 trace 的累计 350.729 ms 只作诊断，不能与另一模式、不同步数的 eager `down_proj` 累计时间相减来算提速。

A1/B/A2 未显示稳定的整模收益，当前不采纳 Stream-K。真实核迹确认了 replay，但有输出和接受率差异的请求速率仍不能还原为纯 kernel 因果效应。thinking 的 1024 个输出 token 在三臂中均为 reasoning、没有最终正文，也不能当作完成任务速率。原服务已恢复，见下方恢复记录。

## 服务恢复与保留产物

10 月 14 日 11:36–11:39（Asia/Taipei），按 **27B → 9B → ASR** 顺序启动同一组原容器，逐项健康后才启动下一项。27B 保持原 vLLM 0.27.1、MTP2、显存预算 0.45；9B/ASR 预算仍为 0.22/0.16。四项服务的容器 ID、镜像和脱敏启动参数均与本轮初始快照一致，未安装实验补丁。[顺序恢复](evidence/gemm-targeted/restore-sequence.json)、[初始状态](evidence/gemm-targeted/initial-state.json)。

实际 API 验证全部通过：27B、9B 正常结束并答出 17×23=391；ASR 的 5 秒合成音频转写与预期匹配；embedding 返回 1024 维有限向量。11:44 的复查距功能检查约 **309.58 秒**，四项 health/models 均为 200，启动时间与重启计数未再变化。这只证明恢复后的功能和短窗口稳定性，没有重测四服务共存性能或完整 MoBase 链路。[功能验证](evidence/gemm-targeted/restoration.json)、[稳定性复查](evidence/gemm-targeted/restoration-stability.json)。

**Embedding 始终未停止。** 初始、各实验前后和最终快照的 ID、启动时间 `2026-10-06T09:31:08.313413609Z`、重启计数 1 均一致。服务器本地监控在 10:03:22–11:44:26 记录了 **203 次**健康采样，全部为 200 且身份一致，最大采样间隔 30.65 秒。它在 SSH 断线期间仍继续记录；周期采样不能证明每一瞬间的可用性或共享 GPU 对请求延迟零影响。[监控摘要](evidence/gemm-targeted/embedding-monitor-summary.json)、[逐条记录](evidence/gemm-targeted/embedding-monitor.jsonl)。外部 SSH/8002 曾超时，现已恢复，但现有证据不能断定是连接频繁导致。[连接观测](evidence/gemm-targeted/connectivity-interruption.json)。

本轮 7 个临时诊断/对照容器已删除，独立探针容器运行后自动删除，现场剩余 `codex-gemm-*` 容器为 0。[清理记录](evidence/gemm-targeted/cleanup.json)。实验代码、脱敏结果与源码哈希保存在本目录；大体积原始 trace/SQLite 在本地忽略目录中保留并列入[原始产物清单](evidence/gemm-targeted/raw-artifact-manifest.json)。服务器上的工具、编译库、原始捕获保留于 `/home/llm/.local/share/qwen-gemm-targeted/`，模型沿用 `/home/llm/models`，本轮没有下载或转换权重。

## 复现入口与采用门槛

本地可用[归因分析器](evidence/gemm-targeted/analyze_attribution.py)重新分析已保存 SQLite；[模块签名汇总](evidence/gemm-targeted/module-signatures.json)列出实际 trace 与原 JSONL 的 SHA。独立探针为 [tile](evidence/gemm-targeted/standalone_fp8_probe.py)、[Split-N](evidence/gemm-targeted/splitn_probe.py)、[Stream-K](evidence/gemm-targeted/down_streamk_probe.py)。对应的 `micro-1/status.json`、`micro-splitn/status.json`、`micro-streamk/status.json` 保存了执行参数、退出状态和 embedding 前后 ID、StartedAt、RestartCount、health；三个窗口均未观察到 embedding 重启或健康失败，但不能推出其请求延迟完全未受共享 GPU 影响。模型 API 单臂复现入口为[run_api_arm.py](evidence/gemm-targeted/run_api_arm.py)，对照仅采用有效 v2 臂。

当前决策是**保留固定 CUTLASS 路径**。任何候选进入服务前须先通过完整数值/布局与并发安全检查，再在固定栈、相同输入、无 profiler 条件下测量整模 TTFT 和 decode tok/s，并记录输出一致性及 embedding 状态。单核 NCU 时间、eager 模块份额或合成 CUDA event 中位数都不构成整模提速证据。

## 后续算子开发取舍

维持官方 vLLM `v0.31.0 / db9527a46873454610df6dbedf79a36d6bf1a7f6` 为开发基线。本轮固定源已包含小 M swap-AB 与按权重大小选择 swizzle 的路径，不能把重写这些已有分流说成新增优化。

- **Prefill**：代表性大 M FP8 kernel 的 Tensor pipe active 比率约 88%，DRAM 比率较低；这里的 active 比率不是实得 FLOPS。先验证不同量化格式或真正减少计算/中间读写的实现，不继续盲扫已验证变慢的 tile/Split-N。
- **Decode**：代表性 gate/up 与 BF16 head 的显存吞吐较高，单纯改 BF16 tile 的空间有限。`lm_head` 权重一次读取约 2.543 GB（BF16，248320×5120×2 字节）；更值得单独验证 head 低精度路径、减少 head 调用/权重读取次数，或整体低比特权重格式。它们会改变精度或推理流程，必须另做模型质量与实际后端验收，当前没有测得此类收益。
- **当前采用决定**：三种实验候选均不替换原服务。本轮完成的是定位、实现和实测筛选，未交付生产加速补丁；仍不能保证社区的 130/200 tok/s。
