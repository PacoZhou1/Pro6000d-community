# 16. 下一轮计划：先测 FP8 GEMM 硬件上限，再决定 MXFP8；decode 并行做草稿词表裁剪

日期：2026-10-14，Asia/Taipei。本页是**任务说明**，不是实验报告；所有数字要么引自 01–15 号文档，要么引自文末列出的外部来源，均已标明。本轮目标只有两个：①用一天时间测出这张卡 FP8 GEMM 真正能到多少，决定 GEMM 线是否继续；②在 decode 上验证一条不改精度、不改权重的无损加速路径。

## 0. 为什么换方向

过去六轮（07–15 号文档）整模结果：B12X 替换 prefill 慢 19.65%；tile / Split-N / Stream-K 均否；V2 量化核 prefill 快 1.41%（单时段，小于基线漂移的两倍）；epilogue 融合 v4 慢 22.75%、v6 慢 7.09%；调度顺序快 0.14%；M64 warp 重排慢 2.63%。所有融合候选的 Tensor Core 活跃度都停在 68%，官方 83–84%（14、15 号文档）。

两个此前没有被当作目标的事实：

| 事实 | 数据 | 来源 |
|---|---|---|
| 官方 CUTLASS blockwise FP8 GEMM 在 M=16160 只到标称峰值的 61% | 2×16160×34816×5120 = 5.761 TFLOP ÷ 9.336 ms ≈ **617 TFLOP/s**，峰值 1007.6 | 14 号文档第 4 节；13 号文档 |
| 同一款卡上别人的 FP8 kernel 到 73% | FlashInfer 为 MiniMax-H3 写的 SM120 融合 FP8 kernel 在 RTX PRO 6000 实测 **732–741 TFLOP/s** | [FlashInfer #5521](https://github.com/flashinfer-ai/flashinfer/pull/5521)、[#5524](https://github.com/flashinfer-ai/flashinfer/pull/5524) |
| cuBLASLt 在 5090 上到 76% | `torch._scaled_mm` ≈ 640 TFLOP/s，该卡 FP16 累加峰值 838 | [FlashInfer #5963](https://github.com/flashinfer-ai/flashinfer/issues/5963) |
| decode 每步读 5 次 lm_head | eager 归因：M=1 调用 348 次 / 87 步 = 每步 4 次草稿，M=5 每步 1 次验证；每次约 1.58 ms；合计约 7.9 ms/步，占 eager decode kernel 时间 25.5% | 09 号文档第 2 节 |

官方路径为什么只有 61%，固定源码里能看到（`evidence/gemm-epilogue/source/cutlass/include/`）：

- `sm120_blockwise_mma_builder.inl:171–178`：通过 `rr_op_selector_sm120` 选择 `SM120_16x8x32_TN`，即**普通** `mma.sync.aligned.kind::f8f6f4` 指令，并断言“非 block-scaled builder 只支持 F8F6F4 MMA”。
- `sm120_mma_tma_blockwise_scaling.hpp:676–694`：每个 K=128 块结束后在 CUDA 核上执行 `accum(i) += tmp_accum(i) * scale_ab`，即**软件缩放**，与 Tensor Core 指令争用发射槽。
- 为此每线程同时保存 `accum` 与 `tmp_accum` 两份 FP32 累加器。13 号文档第 3 节已算出这是 tile 放不大、流水级加不上、v3 寄存器溢出的直接原因。

因此余量不在“让 M64 追上 M128”，而在官方路径自身的 61% 与同卡已证明的 73–76% 之间，以及 decode 里每步 4 次草稿 lm_head 读取。

## 1. 范围与不变约束

- 只做 **FP8**。权重与激活精度不低于 FP8；不做 NVFP4 / FP4。
- `lm_head` 权重保持 BF16、数值不变。任务 D 允许为草稿步骤持有一份 **按行切片的 BF16 副本**，验证步骤仍用完整 `lm_head`；这不是精度改动。
- 固定栈不变：vLLM 0.31.0 镜像 `sha256:a4a4c043…c3a6b`，提交 `db9527a4…`，Qwen3.6-27B-FP8 既有权重目录。需要其他版本做对照时另起隔离镜像并记录，不覆盖基线。
- 模型与数据只从 ModelScope 下载，沿用 `/home/llm/models`；下载后生成分片 SHA256 清单。
- 服务策略沿用 15 号文档：embedding 全程不停，每次 GPU 实验前后记录其 ID / 启动时间 / 重启计数 / health 并做一次实际向量请求；原 27B / 9B / ASR 保持停止，等用户明确要求再启动；自有实验容器用完即删。
- 证据放在 `_dev/docs/evidence/sm120-next/`，子目录 `a-gemm-ceiling/`、`b-mxfp8/`、`d-draft-vocab/`、`e-ddtree-notes/`；报告写成 `17-sm120-next-round.md`。
- 延续既有口径：profiler 下的时间不作性能结论；性能对照用无 profiler、同进程随机交错、31 对的 CUDA Event 中位数，warm / 扰缓存分开；整模用逐 index 同 payload 的 A1/B/A2，以 `(A1+A2)/2` 为基线；负结果如实保留；百分比标明分母。
- **新增要求：所有微基准必须记录 SM 时钟。** 15 号文档已证明一次 2.6% 的“收益”几乎全是时钟从 2.262 升到 2.316 GHz 造成的。优先尝试 `nvidia-smi -lgc` 锁频（需 root，记录是否成功）；锁不了则每次测量同时采集 NVML 时钟，并报告周期数（时间 × 平均时钟）而不只报毫秒。

## 2. 任务 A：这张卡 FP8 GEMM 的真实上限（优先级最高，预算一天）

目的：得到“好的 FP8 kernel 在本卡、本形状能到多少 TFLOP/s”的硬数字。它决定 GEMM 线是否继续，所有后续任务以它为判据。不写任何新 kernel。

### A1. 裸 MMA 吞吐：普通形式 vs block-scaled 形式

写一个最小 CUDA 程序，在寄存器上循环发射两种指令，各测吞吐：

- 普通：`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`（当前 CUTLASS 路径使用的 `kind::f8f6f4` 等价形式）。
- block-scaled：`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0`，scale 寄存器填 `0x7f7f7f7f`（UE8M0 的 127 = 2⁰，数学结果不变）。

背景：GeForce GB202（5090）上普通形式只有半速，实测 510 vs 1014 TFLOP/s；RTX PRO 白皮书附录 A 称 PRO 卡 FP32 累加全速，本地 `evidence/hardware-grounded/sm120-facts.md:17–18` 也是这么记的，但**没有人在 PRO 6000 上实测过**，该 issue 作者明确征求 PRO 6000 的 A/B。两种结果都有用：若 PRO 卡全速，排除一个假设；若也是半速，这是 2 倍级别的发现，后续所有 FP8 kernel 都应改用 block-scaled 形式。参考实现见 [FlashInfer #5963](https://github.com/flashinfer-ai/flashinfer/issues/5963) 中的测试方法。

输出：两种形式的 TFLOP/s、SM 时钟、寄存器数、SASS 中的指令名（`QMMA.16832.F32.E4M3.E4M3` vs `QMMA.SF.16832…E8`）。

### A2. cuBLASLt 参照：`torch._scaled_mm`

在固定镜像的 torch 2.13.0+cu130 中，对真实形状测 cuBLASLt 的 FP8 GEMM：

- 形状：gate_up `(M, N=34816, K=5120)`、down `(M, N=5120, K=17408)`，M ∈ {2741, 2786, 16160, 32768}。
- scale 模式按 cuBLAS 在 sm_120 的实际支持逐一尝试并记录接受 / 拒绝：tensor-wise FP32；MXFP8（VEC32 UE8M0）；1×128 / 128×128 block（cuBLAS 13.4 文档列为 Hopper 模式，sm_120 是否接受未知）。sm_120 可能只接受 TN 布局，见 [PyTorch #198126](https://github.com/pytorch/pytorch/issues/198126)。
- 操作数用随机 FP8；本任务只测速度，不测数值。
- 与固定 CUTLASS blockwise 路径（`_C.cutlass_scaled_mm`）在**同一进程、同一窗口**随机交错配对，31 对，warm / 扰缓存各一组；同时报告时钟。

输出：每个形状一行，列出 cuBLASLt 各 scale 模式与 CUTLASS 的毫秒、TFLOP/s、时钟、配对比值。

### A3. 可选：社区 SM120 FP8 GEMM

仅在 A1 或 A2 显示比 CUTLASS 快 10% 以上时做，否则跳过。候选（均未验证，需固定 SHA 并先过随机输入数值检查）：eous 的 SM120 DeepGEMM 移植（`mma.sync` 实现，[Docker Hub](https://hub.docker.com/r/eous/vllm-sm120)）、QuACK 的 `gemm_sm120.py`、aidxhxr 的 5090 kernels（[#5963](https://github.com/flashinfer-ai/flashinfer/issues/5963) 中提及）。只做单算子对照，不接入模型。

### A 的判据

- 产出一张表：本卡各形状“可达到的最高 FP8 TFLOP/s”及其来源（指令形式 / cuBLASLt 模式 / 社区 kernel）。
- **若最高值相对 CUTLASS 的 617 不足 +10%，GEMM 线整体结束**，本轮只继续任务 D。
- 若 ≥ +10%，按来源进入 B（block-scaled / MXFP8 路径）或 A3 的接入评估。

## 3. 任务 B（条件启动）：MXFP8，让硬件做缩放

前提：A 显示 block-scaled 指令或 cuBLASLt MXFP8 模式有 ≥10% 的单算子优势。

原理：`kind::mxf8f6f4` 在 MMA 内部应用 scale（每 32 个 K 元素一个 UE8M0 scale），不需要 `accum += tmp_accum × scale` 的软件补乘，也不需要第二份累加器。**可验证的假设**：寄存器压力减半后，M128×N128 以上的 tile 或三级流水能放进 99 KiB shared / 168 寄存器预算。这要用 ptxas 资源报告和 NCU 证明，不能默认成立。

### B0. 合成单算子（不需要真实权重）

- 用 CUTLASS 的 SM120 block-scaled builder（`kind::mxf8f6f4`，scale_vec 1X）或 cuBLASLt MXFP8 模式，随机 MXFP8 操作数，形状同 A2。
- 与 CUTLASS blockwise 同窗口配对计时；记录 ptxas 寄存器 / spill / shared、NCU 的 Tensor 活跃度与指令构成，对照 15 号文档官方 83.55% 的基准。
- 判据：代表性大 M（16160）配对 ≥ +10% 才进 B1。

### B1. 真实权重与数值

- 从 ModelScope 下载 **Qwen3.6-27B（BF16 原始权重）**，固定 revision，生成分片 SHA256 清单。现有 block-128 FP8 权重不能无损转成 MXFP8，必须从 BF16 重新量化。
- 用 ModelOpt 量化为 MXFP8（W8A8，scale 每 32 元素 UE8M0）。社区已有 Qwen3.8-27B 的同类 checkpoint 以 `--quantization modelopt_mxfp8` 在 vLLM 运行（[示例](https://friendli.ai/models/talumbau/Qwen3.8-27B-mxfp8)），可参照其配置，但其质量数据不适用于 3.6。
- 逐层数值门槛：固定一组真实 hidden states，比较 BF16 参考、现有 block-128 FP8、MXFP8 三者的 gate_up / down 输出与最终 logits（余弦、NRMSE、top-1 一致率、KL）。MXFP8 相对 BF16 的误差不得明显大于现有 block-128 FP8 相对 BF16 的误差；具体阈值在测量前写死并记录。
- vLLM 0.31 能否加载 `modelopt_mxfp8` 并在 SM120 派发到硬件 block-scaled kernel，要用启动日志和 Nsight kernel 名证实。SGLang 在 SM120 上 MXFP8 已能跑通、CUTLASS 版 dense GEMM 有人认领（[SGLang #19637](https://github.com/sgl-project/sglang/issues/19637)）；vLLM 侧 SM120 MXFP8 dense 路径未找到确认。若 0.31 不支持，记录为阻塞项，可另起隔离镜像验证，不改基线。

### B2. 整模 A/B/A

只有 B0、B1 都过才做。A 臂为现有 block-128 FP8，B 臂为 MXFP8；KV dtype、MTP、显存预算、请求一致。质量门槛在既有 5 项合成 + 4 项开放输出之上，加长上下文检索、思考模式正常结束、JSON / 工具调用，以及 B1 的逐层 KL。性能判据：prefill 配对 TTFT 改善须大于两次基线漂移的两倍，且在两个独立时段重复。

## 4. 任务 D：草稿头词表裁剪（与 A/B 独立并行）

### 依据

09 号文档 eager 归因：每个 decode 步，`lm_head`（248,320 × 5,120 BF16，2.54 GB）被读 5 次——4 次草稿（M=1）+ 1 次验证（M=5），每次约 1.58 ms，NCU 显示 DRAM 吞吐已达峰值 94%（带宽受限）。按 15 号文档 decode 106.7 tok/s、每轮接受 2.06 + 1 bonus 推算，每步约 28.7 ms，其中 lm_head 约 7.9 ms（27%）。

FR-Spec（[论文](https://arxiv.org/html/2502.14856v1)）的做法：草稿只在高频 token 子集上计算 logits，验证仍用完整词表。贪婪验证下最终输出等于目标模型的贪婪输出，输出分布不变。若子集取 32,768 个 token（词表的 13%），4 次草稿读取从 4 × 2.54 GB 降到 4 × 0.34 GB，每步省约 5.5 ms，**decode 上限约 +20%**，实际取决于接受率下降幅度。论文在 EAGLE-2 上报告 lm_head 计算减少 75%、总体 1.12×；本模型 lm_head 占比更高（27%），所以估算更高，但必须实测。

SGLang 以 `--speculative-token-map` 实现，仅支持 EAGLE-2（[文档](https://docs.sglang.io/advanced_features/speculative_decoding.html)）；vLLM 没有找到对应实现；Qwen 内置 MTP 共享 `lm_head`，需要自己改草稿路径。有后续论文指出固定子集会漏掉上下文相关的低频 token（[SpecVocab](https://arxiv.org/html/2602.13836v1)），因此接受率是核心风险指标。

### D0. 先在正常图模式确认（半天）

eager 的 5 次/步要在正常 CUDA Graph 模式下复核：对 MTP4 decode 做一次 Nsight 诊断捕获，统计每步 lm_head 形状的 kernel 次数与时长（可用 09 号文档的 NVTX 关联方法），确认草稿步确实各做一次 M=1 全词表投影，而不是别的实现。若不是，本任务停止并记录。

### D1. 词频表

- 用 Qwen3.6 tokenizer 统计一个通用语料（中文 + 英文 + 代码，公开数据集，记录来源与 SHA；不用业务数据）的 token 频率，生成 16K / 32K / 64K 三档高频子集，保存为 token id 列表及其 SHA256。
- 另外单独统计思考（reasoning）文本的频率分布是否与普通文本不同；若差异大，为思考模式准备独立子集。

### D2. 实现

在固定 vLLM 0.31 的 MTP 草稿路径（`qwen3_5_mtp.py` 与 `LogitsProcessor` 的草稿调用）上，以 opt-in 方式：

- 加载时按子集行索引切出 `lm_head_draft = lm_head[subset]`（BF16，32K 档约 0.34 GB），原 `lm_head` 不动。
- 草稿步用 `lm_head_draft` 计算 logits，argmax（或采样）后把子集内索引映射回完整词表 id；验证步与 bonus token 仍走完整 `lm_head` 与原有校验逻辑。
- guard：仅草稿步、仅目标模型、仅注册的 head 权重身份；任何不匹配回退原路径并计数。CUDA Graph 捕获、换输入回放、fake/meta 与序列化检查与前几轮相同。

### D3. 正确性门槛

- 贪婪解码下，固定 6 条 4.8K/1024 请求与 4 条开放输出，候选的最终正文与**同配置 MTP4 基线**逐 token 比较；预期一致。若出现差异，先定位是否来自 batch 形状导致的数值差（07 号文档已记录 MTP2/3/4 之间存在此类差异），再判断。
- 接受率：报告 accepted / draft 与每轮接受长度，对照基线；三档子集各测。
- 既有 5 项合成质量、4 项开放输出、思考模式正常结束。

### D4. 性能 A/B/A

无 profiler，同 payload，A1/B/A2，各 6 条普通 decode + 6 条思考 decode（4.8K/1024），每类 warmup 剔除。报告 tok/s、每步时间、accepted/draft、embedding 计数、payload hash。另附一次诊断捕获证明草稿步 lm_head 字节数实际下降。

判据：32K 档若 decode 改善 < 5% 或接受率下降超过改善幅度，任务结束并保留负结果；≥ 10% 且输出一致，进入跨时段复测与思考模式。

## 5. 任务 E：DDTree 可行性笔记（只调研，不实现，不占 GPU）

树验证（DDTree：从 DFlash 草稿分布建树，一次前向验证整棵树）在 MLX 移植上比纯 DFlash 再快 77%，vLLM 只有实验性 fork（[DDTree 论文](https://arxiv.org/abs/2604.12989)、[oMLX PR](https://github.com/jundot/omlx/pull/805)、[AEON fork](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DDTree)）。本轮只写一页笔记：vLLM 0.31 V2 runner 做树状验证需要哪些改动（树 attention mask、GDN 递归状态的分支回滚、KV 分页），Luce 为 llama.cpp 写的 3 个 SSM 回滚 kernel 对应到 vLLM 是什么，预计工作量与风险。供决定下下轮是否立项。

## 6. 本轮明确不做

- M64×N256 融合主循环的任何变体，包括 15 号文档末尾提出的 M64N128 两级 / 三级流水。原因：A 的结果出来之前，它的天花板（Tensor 活跃 68% vs 官方 84%）已经说明方向不对；且其权重重排副本 10.79 GiB 与生产 KV 池 11.15 GiB 同量级，没有部署路径。
- 大 M 换 tile、Split-N、Stream-K、删 pack：09、12、14 号文档已否。
- GDN、attention 重写：prefill 占比合计不到 20%。
- NVFP4 / FP4、`lm_head` 降精度：范围外。
- 任何未锁频或未记录时钟的微基准结论。

## 7. 顺序与时间盒

| 顺序 | 任务 | 预算 | 判据 |
|---|---|---|---|
| 第 1 天 | A1 + A2；D0 + D1 并行 | 1 天 | A 给出上限表；D0 确认 5 次/步 |
| 第 2 天起 | 按 A 结果：≥ +10% 进 B0；否则 GEMM 线关闭 | B0 一天 | B0 ≥ +10% 才进 B1 |
| 与 B 并行 | D2 → D3 → D4 | 两天 | D4 ≥ +10% 且输出一致才做跨时段复测 |
| 任意空档 | E 笔记 | 半天 | 一页文档 |

每个任务的退出条件在开始前写进 `request.md`；没过判据的任务直接结束，不追加变体。

## 8. 外部来源

- [FlashInfer #5963：SM120 FP8 MMA 半速与 block-scaled 指令对比、5090 实测](https://github.com/flashinfer-ai/flashinfer/issues/5963)
- [FlashInfer #5521 / #5524：MiniMax-H3 SM120 融合 FP8 kernel，RTX PRO 6000 实测 732–741 TFLOP/s](https://github.com/flashinfer-ai/flashinfer/pull/5521)
- [Colfax：SM12x block-scaled GEMM 教程（mma.sync 路径、tile 配置、约 60% 峰值）](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)
- [SGLang #19637：SM120 性能优化计划，含 MXFP8 状态与 Qwen3.6-27B-FP8 后端对照](https://github.com/sgl-project/sglang/issues/19637)
- [cuBLAS 13.4 文档：FP8 block scaling 模式](https://docs.nvidia.com/cuda/cublas/index.html)
- [PyTorch #198126：sm_120 上 MXFP8 scaled_mm 仅支持 TN 布局](https://github.com/pytorch/pytorch/issues/198126)
- [Qwen3.8-27B MXFP8 社区 checkpoint（vLLM modelopt_mxfp8）](https://friendli.ai/models/talumbau/Qwen3.8-27B-mxfp8)
- [FR-Spec 论文](https://arxiv.org/html/2502.14856v1)、[SGLang FR-Spec 文档](https://docs.sglang.io/advanced_features/speculative_decoding.html)、[SpecVocab：固定子集的局限](https://arxiv.org/html/2602.13836v1)
- [DDTree 论文](https://arxiv.org/abs/2604.12989)、[Luce DFlash + DDTree 实测](https://rits.shanghai.nyu.edu/ai/luce-dflash-brings-2x-speculative-decoding-to-qwen3-6-27b-on-a-single-rtx-3090/)
- [lastloop-ai：Qwen3.6 27B on RTX PRO 6000 社区配置与速度](https://github.com/lastloop-ai/vllm-blackwell-guide)
- [eous/vllm-sm120 镜像（SM120 DeepGEMM 移植）](https://hub.docker.com/r/eous/vllm-sm120)
