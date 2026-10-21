# Qwen3.6-27B：GPU 热点与首次算子实验

测试日期：2026-10-14。机器为 **1 × RTX PRO 6000 Blackwell Workstation 96GB、600W、SM120**。这是用户服务器的结果，与仓库首页四张 RTX 6000D 的 GLM/MiMo 成绩不可直接比较。

后续已完成真实 GEMM 尺寸与模块归因、合成硬件计数器、三种定向候选及整模 A/B/A；新结果与最新服务恢复记录见[定向 GEMM 报告](09-targeted-gemm.md)。本页保留当时采样的口径和结果。

## 已确认的热点

六份真实 Nsight Systems 采样一致指向 **矩阵乘 GEMM**。本轮首先选择 FP8 blockwise GEMM 做实现替换实验。GDN 已有融合实现，在这些窗口中占比较小，当前优先级低于 GEMM。

下表是两次独立采样占比的算术平均。**分母为每份捕获窗口内全部 GPU kernel duration 的总和；类别按原始 kernel 名称推断。不是 HTTP 请求延迟占比。** 各行只展示主要类别，未列满 100%。

| kernel 类别 | MTP4 prefill | MTP4 普通 decode | DFlash15 思考 decode |
|---|---:|---:|---:|
| FP8 GEMM | 68.49% | 62.03% | 64.10% |
| BF16 命名 GEMM | 1.30% | 29.50% | 21.54% |
| 全注意力 | 14.18% | 2.35% | 2.97% |
| GDN 与卷积相关融合 kernel | 4.86% | 2.52% | 5.20% |
| activation / FP8 quant | 6.18% | 1.74% | 1.96% |

证据：[MTP4 汇总](evidence/profiling/mtp4-nsys-summary.json)、[DFlash15 汇总](evidence/profiling/dflash15-nsys-summary.json)；完整原始名称、调用次数、平均耗时与分类规则见同目录各 `*-analysis.json`。FP8/BF16 分类先识别 FP8，再识别 BF16，避免把 FP8 GEMM 名称中的 BF16 输出类型重复计入。

| 捕获窗口 | 输入 / 请求输出 token | kernel 数 | kernel 耗时总和 |
|---|---|---:|---:|
| prefill 1 / 2 | 18,901 / 1；18,946 / 1 | 3,673 / 3,673 | 2,266.68 / 2,272.79 ms |
| MTP4 decode 1 / 2 | 约 4,800 / 256 | 90,058 / 91,143 | 2,344.26 / 2,372.72 ms |
| DFlash15 thinking 1 / 2 | 4,792 / 256 | 84,000 / 94,500 | 1,482.65 / 1,668.05 ms |

decode 的 256 是**整个请求输出量**，不是捕获窗口对应的 token 数。思考场景全部 256 token 为 reasoning，没有最终正文；这里评估生成阶段，不代表任务完成速度。

### 具体落在哪些 kernel

- prefill 最大单个名称是 `cutlass_3x_gemm_fp8_blockwise` 的 128×128×128 tile 路径，首次累计约 1,551.23 ms / 520 次，约占 68.44%。这是第一优化对象。
- MTP4 decode 最大名称是同一 FP8 blockwise 家族的 SM120 swap-AB 128×32×128 路径，首次累计 1,454.25 ms / 22,576 次，平均约 64.42 μs。
- MTP4 decode 另有两个较大的 `cutlass_80_wmma_tensorop_bf16_*gemm*` 名称，合计约 27.87%。这提示 BF16 投影也值得继续定位；**当前没有完整模块归因，不能称为已证实的 lm_head 或 draft 开销**。
- GDN decode 已出现 `gdn_decode_post_conv_mtp_kernel` 等融合 kernel；把现有融合路径重写一遍，潜在收益上限小于先处理 GEMM。

GEMM 家族覆盖 MLP、其他投影及可能的 draft 计算，不能把上述占比统称为 MLP 占比。尚未使用 Nsight Compute 测量 DRAM 吞吐、SM 利用率或 occupancy，所以目前只能确定耗时热点，不能据此断言全部受显存带宽限制。

## 采样方式与边界

固定 vLLM 0.31.0 / `db9527a46873454610df6dbedf79a36d6bf1a7f6`，FP8 权重、FP8 KV、FlashInfer attention、V2 runner、单卡、并发 1、显存比例 0.70；Nsight Systems 2026.5.1 使用 CUDA graph node tracing。准确镜像、启动参数和后端日志保存在 [MTP4 runtime](evidence/profiling/mtp4-runtime.json)、[DFlash runtime](evidence/profiling/dflash15-runtime.json)。

每类先预热，再采两次新请求。prefill 在请求前开启捕获，只输出 1 token；NVTX 显示 16,160 加 2,741 / 2,786 token 的两段 context。decode 在收到首段 SSE 文本后调用 `/start_profile`，因此只观察随后一段稳定生成；两组分别只有 `context_0 / generation_1(5)` 和 `context_0 / generation_1(16)`，没有 prefill context。

`/start_profile` HTTP 调用耗时约 0.46–0.84 秒，真实捕获起点在控制调用过程中；不能将 HTTP 返回时刻直接当成开始时间。所有采样响应完整，固定输出 token 与服务端计数吻合，prefix cache hits 为 0，窗口前后 embedding 请求计数未变。客户端证据：[MTP4](evidence/profiling/mtp4-capture-requests.jsonl)、[DFlash15](evidence/profiling/dflash15-capture-requests.jsonl)。

NVTX → 同线程 CUDA API → correlation ID 可精确关联的 kernel 耗时覆盖：prefill 约 96.8%，MTP4 decode 约 66.6%，DFlash15 约 78.3%。未关联部分保留在总体 kernel 排名中，不按时间重叠强行分配给 target/draft。GPU interval union / 首末 GPU 事件跨度仅描述窗口内活动覆盖，不能称为整机 GPU 利用率。

分析由 [analyze_nsys.py](evidence/profiling/analyze_nsys.py) 离线重算。原始 `.nsys-rep` 和 SQLite 留在服务器 `/home/llm/.local/share/qwen-profiling/{mtp4,dflash15}/`，路径、字节数与 SHA256 见 [raw-manifest.json](evidence/profiling/raw-manifest.json)。本地 SQLite 在 `evidence/profiling/raw/`，Git 忽略二进制数据。

## 首次优化实验：FP8 GEMM

原镜像已经选择 CUTLASS，强制 `cutlass` 本身不会提速。候选采用 **B12X 1.5.0 的 SM120 block-FP8 GEMM**，通过 vLLM 的 `linear_backend_per_quant.fp8_block_w8a8` 切换。此轮是集成现成算子实现，尚未编写新的 CUDA kernel。

版本选择有一个实际约束：vLLM 0.31 官方 extra 固定 B12X 1.3.0，它要求 CuTe DSL 4.6.2；当前 FlashInfer CUDA 13 extra 要求 DSL ≥4.7。B12X 1.5.0 要求当前已装的 4.7.1，因此派生镜像只增加该包、保持 torch/FlashInfer/DSL 不变。**1.5.0 超出该 vLLM tag 官方 pin，属于本轮验证的实验组合。** 依赖与源码链接见 [backend-source-notes.md](evidence/profiling/backend-source-notes.md)，镜像与 wheel 哈希见 [b12x-image.json](evidence/profiling/b12x-image.json)。

同一个派生镜像分别强制 CUTLASS 与 B12X；对照完全关闭 profiler，使用同一组确定性新输入，每类 3 次正式请求。prefill 约 19K 输入 / 1 输出；普通和思考 decode 约 4.8K 输入 / 1,024 输出。由此测得的速度与前面的短窗口 profile 各自报告，不能混用。

### 整模 A/B 与复测

| 指标（每格 n=3，中位数） | CUTLASS | B12X | 相对变化 |
|---|---:|---:|---:|
| 约 19K 冷输入 TTFT | 2.225 s | 2.662 s | **延迟 +19.65%** |
| 普通 decode | 109.08 tok/s | 109.41 tok/s | +0.30% |
| 思考 token decode | 130.03 tok/s | 130.95 tok/s | +0.71% |

**不采用全量 B12X 替换：prefill 明显退化，decode 没有可观的整模收益。** 日志确认两臂分别选择 `CutlassFp8BlockScaledMMKernel` / `B12xFp8BlockScaledMMKernel`，不是静默落回同一实现；见[实际 dispatch](evidence/profiling/backend-selection.json)。

两臂九组正式请求的 payload、正文、reasoning 哈希分别 **9/9 相同**；均完整生成指定 token，cache hits 为 0，无抢占，观测窗口内没有 embedding 计数变化。两臂各 5 项结构化/长前缀提取和 4 项开放回答检查全部通过，共 18 项；这些是合成 API 检查，不代表 MoBase 整链路或广泛业务质量验收。思考性能组 1,024 token 全为 reasoning，不是完成整项任务的速度。

随后重新启动相同 CUTLASS 容器，清空运行时前缀缓存做 A/B/A 复测：prefill 2.241 s（n=3，相对首轮 +0.75%），普通 decode 109.086 tok/s（n=3，+0.003%）。思考第 1 个复测窗口发生一次 **18-token embedding 请求**，保留原记录并从无干扰复测对照剔除；相同第 2/3 个输入的首轮和复测分别为 131.735 / 131.744 tok/s（各 n=2，+0.006%）。不能把不同输入子集的中位数直接比较。复测九组输入与输出哈希仍全部一致。

原始性能与质量记录、每个输入的范围、全部污染计数及排除理由见[可重算 A/B 汇总](evidence/profiling/ab-summary.json)。没有锁定 GPU 时钟，也未随机化测试顺序；小样本结果是描述性测量，不宣称 +0.3% / +0.7% 具有统计显著性。算子实现替换之外，本轮不变更模型或权重精度。

### 独立 GEMM 数值与微基准

[gemm_probe.py](evidence/profiling/gemm_probe.py) 使用两种实际 MLP 尺寸、随机生成的相同 FP8 操作数与 scale，分别调用固定 vLLM 的 CUTLASS 路径和 B12X 路径。M 为矩阵行数，投机验证阶段包含多行，不能只测 M=1。

| K / N | M | CUTLASS | B12X |
|---|---:|---:|---:|
| 5,120 / 34,816 | 1 | 114.03 μs | 111.25 μs |
| 5,120 / 34,816 | 5 | 114.57 μs | 114.95 μs |
| 5,120 / 34,816 | 16 | 115.50 μs | 116.86 μs |
| 17,408 / 5,120 | 1 | 34.82 μs | 28.14 μs |
| 17,408 / 5,120 | 5 | 34.84 μs | 39.65 μs |
| 17,408 / 5,120 | 16 | 34.12 μs | 56.95 μs |

12/12 实现与形状组合的数值检查通过：全输出 finite，前 512 列对反量化后 FP32 GEMM 的 NRMSE <2%、cosine ≥0.999；FP32 参考关闭 TF32。**这只验证这些随机输入的算子数值，不证明模型质量，也不覆盖大 M prefill 的单算子性能。** 计时为预热后 CUDA graph replay，排除量化与 Python dispatch，反复使用相同权重会受 L2 缓存影响，不能直接换算模型 tok/s。

这也说明 B12X 并非所有形状都更快：down projection 的 M=1 有收益，M=5/16 则退化。是否值得用于整模，必须以下面的真实 API 对照为准。[原始数值与计时](evidence/profiling/gemm-probe.jsonl)

首次独立脚本在 `QuantFP8` 初始化时因缺少 vLLM 配置上下文退出，尚未运行 GEMM；改为调用固定源码同一底层量化 helper 后通过。保留 [首次失败日志](evidence/profiling/gemm-probe-attempt1.log)，不将脚本初始化错误归因于 B12X kernel。

## 服务恢复与保留产物

所有临时模型容器已停止并删除，27B → 健康 → 9B → 健康 → ASR 按顺序恢复。四个原容器的 ID、镜像、启动参数均与本轮开始时一致：生产仍是 3.6 / vLLM0.27.1 / MTP2，27B 显存预算 0.45；本轮没有将候选后端部署到生产，也没有修改 MoBase。

27B / 9B 的真实算术响应正确且正常结束；embedding 返回 1,024 维有限非零向量；ASR 对 5 秒合成音频转写正确。四项 health / models 均为 200。[功能验收](evidence/profiling/restoration-functional.json)

embedding **全程未停止**：容器 ID、启动时间 `2026-10-06T09:31:08.313413609Z`、RestartCount=1 与初始记录一致；28.5 分钟内 58 次健康采样全部为 200。约 142 秒后的恢复复查中，四个服务的 StartedAt / RestartCount 均未再变。27B 在本次手动启动后 RestartCount 为 0，初始记录为 1，报告保留此差异，不把它当作上轮故障从未发生。这个复查是短期启动稳定性检查，不是持续压测。[恢复汇总](evidence/profiling/restoration-summary.json)、[稳定性快照](evidence/profiling/restoration-stability.json)、[清理记录](evidence/profiling/temporary-cleanup.json)

原始 profile、Nsight 用户目录安装和 B12X 实验镜像保留，便于下一轮复现；没有重新下载模型。模型仍在原来的 ModelScope 下载目录 `/home/llm/models`。仓库只提交 `_dev/docs` 内的文档、脚本与脱敏文本证据，原始二进制 trace 不进 Git。

## 下一步算子开发范围

1. **优先 FP8 GEMM。** 在固定 vLLM 0.31 源码上补逐层 M/N/K、layout 与硬件计数器，针对 prefill 大 M 和 MTP 验证多行分别调 SM120 tile / scheduling。已有 CUTLASS 是有效基线，不能预设换库就更快。
2. **再定位 BF16 大 GEMM 的模块。** 两个大 kernel 在 MTP4 decode 占约 27.9%，但现有 NVTX 不能证明它们就是 lm_head。确认调用与实际尺寸后再选择实现，避免改错路径。
3. **暂不重写 GDN。** 当前窗口占比较小且已有融合实现。也不把只有 M=1 更快的微基准直接推广到 MTP4 / DFlash15。

本轮完成了热点定位和一项可复现的算子后端实验，结论为保留 CUTLASS。尚未完成自研 CUDA 补丁、Nsight Compute 带宽/occupancy 分析、长达 262K 的容量验证、并发吞吐或 MoBase 全链路质量验收；这些不计入本轮成果。
