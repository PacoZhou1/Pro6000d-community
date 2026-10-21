# 下一轮推理优化路线：仅 FP8

日期：2026-10-14，Asia/Taipei。本页按用户最新指示更新：**只推进现有 Qwen3.6-27B-FP8 路径，不排查历史 NVFP4 思维链循环，不做 FP4，也不改 BF16 `lm_head`**。此前本页提出的“先定位 NVFP4 循环”、FP8/NVFP4 四组对照、head-only FP8 和按 FP4 尺寸切后端，均已被本次范围决定取代，**不是当前实验的前置条件**。用户报告的历史循环现象仍只是未复现的记录，不能据此归因于精度或当前 FP8 路径；旧实验与背景见[投机解码](07-specdecode-experiment.md)、[定向 GEMM](09-targeted-gemm.md)。

## 当前候选与优先级

1. **评估已有融合核的 V2 线程组织。** 固定 vLLM 0.31.0 的 `silu_and_mul_per_block_quant_kernel` 已把 SiLU、乘法和每 128 元素 FP8 block 量化融合。最终选中的 **V2 group-major** 源将每个 128 列组交给一个 warp，不是重新实现缺失的融合；V1 是对照版本。两份正常 Graph prefill 捕获中原核名称组各调用 136 次、累计 97.827/98.173 ms，占**所有 GPU kernel duration 之和**的 4.316%/4.319%；这是整个核零成本消失的算术上限，非 API 加速承诺。候选只命中布局匹配的 `M≥256`，小 M decode 按原核回退。[逐核口径](evidence/fp8-fusion/profile-opportunities.md)、[选中源码](evidence/fp8-fusion/fused_silu_quant_group_major.cu)。
2. **数值与真实大 M 执行方式已经核实。** V2 通过受测 FP8 字节/scale 位模式、异常值、小 M 回退和五个尺寸的独立单核 Graph 复用；合成 M2741、M16160 单核 warm/cold 计时优于同次原核。真实模型长 prefill 捕获有 **130 次候选直接 CUDA launch**、6 次 M1 官方回退；B 编译缓存 21 个 Python 文件引用新 op。候选 graphNodeId 全为空，说明该配置的大 M prefill 走**直接 launch**；整模大 M Graph replay 不适用，毋须留作下一轮验收债。合成 M16160 的单核 NCU 显示 active-warp barrier stall 13.64%→0%、DRAM 吞吐占峰值 66.38%→93.52%，只支持机制解释，不是 API 收益测量。[数值与单核](evidence/fp8-fusion/micro-results.md)、[真实派发](evidence/fp8-fusion/integration-results.md)、[计数器](evidence/fp8-fusion/counter-summary.json)。
3. **完成的一次无 profiler A/B/A 只支持候选信号。** A1/B/A2 各有 12 个长 prefill、6 个普通 decode 有效请求。prefill TTFT 中位 **2.268213/2.244258/2.284222 s**；同 index 的 B 对 `(A1+A2)/2` 减少中位 **1.4144%**，但 A1–A2 自身中位漂移约 **0.706%**，无独立重复时段。普通 decode **106.735803/106.731730/106.705445 tok/s**，基本不变；同 index 非空正文 hash 三臂全同，固定质量各 5/5、开放输出各 4/4。短思考算术正常结束，**未测思考 decode 速率**。这轮有小幅 prefill 收益的候选证据，尚不能作为跨时段或生产收益承诺；原服务已恢复并通过功能和后续稳定性检查，embedding 未停，原 0.45 四服务预算和 MoBase 长业务未测。[当前报告](11-fp8-fusion.md)、[三臂逐样本](evidence/fp8-fusion/api-comparison.json)。
4. **若本候选无稳定整模收益，再单独研究独立 FP8 动态量化。** `per_token_group_quant_8bit_register_kernel` 在两份 prefill 中占 GPU kernel-duration 总和 1.566%/1.565%，在两份 decode 中占 1.394%/1.395%。它是**另一个**算子和未来假设；先证明张量依赖及是否能少一次读写，不能把它的份额算到本轮 SiLU 候选。已存在的 residual/RMSNorm/FP8 quant、sigmoid/mul/quant 等融合也不得重复列为“待启用”。

本轮不设 FP4 或 BF16 head 分支，也不把历史思维链循环故障作为 FP8 kernel 实验的阻塞项。V2 **保留为候选，不覆盖生产**。若继续探索 FP8 周边，先检查上游 GEMM 到现有融合量化核的 BF16 中间写读与张量依赖；epilogue 融合目前只是未来假设，没有可行性或整模收益证据。FlashInfer + MTP4 的服务日志提示**不支持 fused multi-step draft decode，草稿步骤之间回退重建 attention metadata**；这是另一个未量化耗时占比的系统假设，不纳入本核收益。[A1 日志](evidence/fp8-fusion/a1/server.log)。投机解码、MoBase 前缀与调用链、生产容量/并发仍是独立工作；旧思考组固定 1024 输出全为 reasoning，其 tok/s 不表示任务正常结束或答案质量。

<details>
<summary>旧版路线原文（仅供追溯；已被 2026-10-14 FP8-only 指令取代）</summary>

以下为旧版原文，不作为当前工作顺序或待办。

# 下一轮推理优化路线

日期：2026-10-14，Asia/Taipei。依据本机[定向 GEMM 实验](09-targeted-gemm.md)、[投机解码对照](07-specdecode-experiment.md)、[MoBase 审计](04-mobase-integration.md)，并复核下文上游资料。本页是建议和待验证假设；本次没有连接服务器、改配置或运行新 GPU 实验。

**用户补充后的调整：**此前使用 **官方 NVFP4 权重 + vLLM** 时曾随机出现思维链反复循环、迟迟不输出最终答案。具体 checkpoint/revision、vLLM 版本、MTP/KV/缓存配置及同题 FP8 对照尚不明确；这是用户报告的历史故障，尚未由本轮复现，不能直接归因于 4 bit 精度。NVFP4 改列为先复现质量问题的路线。当前主线保留已有 FP8 权重和 BF16 head，先验证投机解码、缓存与调用链；head-only FP8 也需先过长思考/终止质量检查，不作为低风险默认优化。

## 优先级

| 顺序 | 工作 | 主要改善对象 | 当前判断 |
|---|---|---|---|
| 前置 | 思维链循环复现、正常完成的业务基线与固定输入回放 | 先判断质量，再归因收益 | 加入偶发循环率和最终答案完成率；本轮短质量探针不足以排除此故障。 |
| 1 | 保留 FP8/BF16 head，分解投机解码每轮成本并做业务调优 | 每次 target 校验兑现的有效 token | 已测 MTP4 普通输出、DFlash15 思考各有优势；需检查多轮/缓存命中后的质量，不把思考 token 速率当任务完成速度。 |
| 条件实验 | 仅 `lm_head` 的 FP8 路径 | decode 中大词表投影的权重读取 | 改动范围集中但直接影响 logits，需先过长思考与终止质量门槛，尚无收益。 |
| 故障排查 | 同模型 NVFP4 与实际后端对照 | 定位历史随机思维链循环 | 能复现、归因且通过质量回归后才恢复为性能候选。 |
| 并行 | MoBase 前缀复用、输入体量和串行调用 | 首正文时间、完整任务时间 | 有长输入、多次 Agent/工具/Generation 的本地代码与样本线索；实际收益需走用户入口测。 |
| 最后 | 生产显存预算与并发调度 | 四服务共存时的 p95、吞吐与稳定性 | 实验预算 0.70 的结果不能直接当作原 0.45 预算的上线结果。 |

## 先补：思维链循环的定位与验收

记录原始流中 reasoning/content 是否持续重复、是否生成思考结束与终止 token、最终正文是否有效，以及 finish_reason、实际长度、接受 token/轮、缓存命中、请求顺序。区分持续生成循环、正常但很长的推理，以及只有解析/展示层没有显示正文。现阶段不能断定是哪一种底层原因。

在同一基座模型、同一引擎提交和相同题目/采样/模板下，先做 **FP8 / NVFP4 × MTP 关闭 / 开启** 四组对照；KV 精度、后端等条件分别固定并记录。随后分别补冷请求、前缀复用、长历史和连续多请求；若失败再单独切 KV、kernel backend 或 graph 模式，不能一次改多个变量。采样可变时需要多次重复和失败样本，固定 seed 也不保证不同执行路径逐 token 一致。

待排查假设包括量化数值及 scale、MTP 校验/状态回退、混合注意力缓存复用、采样/模板/终止配置。上游 [vLLM issue #55357](https://github.com/vllm-project/vllm/issues/55357) 的作者报告过 NVFP4 + MTP 下思维链重复及接受率归零；它是不同的 Qwen3.8-Flash-Next 模型、TP2 和特定 nightly，作者的状态异常推断也不是本机根因证明，只提示应将 MTP/缓存交互纳入对照。

验收以**有效最终答案完成率、循环发生率、达到长度上限却无答案的比例、完成任务时间**为主，tok/s 次之。达到 `max_tokens` 的截断不能算正常完成；增加 repetition/presence penalty、缩短预算或关闭 thinking 都是独立行为变量，不能代替根因修复。可以设置有限预算和超时控制故障成本，但要将触发记为异常/未完成。此前 thinking 组的 1024 token 全为 reasoning，只支持 token 速率观察，不支持无循环、正常结束或完整思考质量结论。

## 1. 条件性的算子开发：只动 lm_head

实测 BF16 head 为 `[M,5120] × [248320,5120]ᵀ`，M=1/5。BF16 权重大小为 2,542,796,800 字节，约 2.543 GB；改为 FP8 时原始权重元素占用约减半，另有 scale 和工作区。这是存储量估算，不等于调用耗时减半。该 head 占本次 eager decode 所有 GPU kernel duration 约 25.54%，不代表正常图模式或 HTTP 延迟份额。

若进入低精度 head 实验，先实现显式可开关、可回退的 head-only FP8 原型，其他层、MTP 配置和 KV 格式不变。启动时准备权重，量化激活、GEMM、输出及临时分配全部计入实际耗时。分别测 M=1、M=5 与真实并发下的 M，覆盖冷权重流与 CUDA Graph。先复用已有 SM120 FP8 GEMM；只有它的实际 head 尺寸仍有明确浪费时，再开发专用核。正常完成与思维链循环回归不通过时不进入性能采用阶段。

固定 0.31 提取源码中的 `qwen3_5.py` 与 `qwen3_5_mtp.py` 已向 `ParallelLMHead` 传入 `quant_config`；但 `Fp8Config.get_quant_method` 没有 `ParallelLMHead` 分支。因此还需核对分派、FP8 权重及 scale 注册/加载、vocab padding、target/draft 的 head 共享关系，以及 `LogitsProcessor` 的真实路径。[源码指纹](evidence/gemm-targeted/source-manifest.json)。上游 [issue #40999](https://github.com/vllm-project/vllm/issues/40999) 记录过相关加载问题，截至本次读取已因长期无活动关闭；它基于旧版，不能整段照搬，尤其不能重复修本机已存在的 `quant_config` 传递。

质量验证用固定真实 hidden states 比 BF16/FP8 logits 的误差、分布差异和 top-token 一致率，再测中文、数字、JSON、工具调用、长文本、正常停止与 MTP 接受率。数值误差小不是业务质量保证。先保持原权重不覆盖，实验转换结果只在隔离产物中保存。

## 2. 质量问题解决后再评估：NVFP4 与按尺寸选择后端

优先在同一代 3.6 上比较 FP8/NVFP4，保持 vLLM、输入、MTP、KV 与容量条件一致；如 ModelScope 只有其他候选，需为模型变化另补同模型 FP8 对照。沿用用户要求：下载只走 ModelScope，仍在 `/home/llm/models`。

逐层确认真实量化格式、scale 和 kernel：有的 checkpoint 只是混合精度，不能把全部权重都当 4 bit，也不能把 W4A16 与 W4A4 当同一执行方式。[官方 Qwen3.6 配方](https://recipes.vllm.ai/Qwen/Qwen3.6-27B) 已列出 NVFP4 路线，但不同硬件的实现和配置仍需本机验证。

上游 [PR #55170](https://github.com/vllm-project/vllm/pull/55170) 已合并 SM120/121 上适用 checkpoint 的 W4A4 优先策略；作者同时指出 W4A4 的 prefill 与 W4A16 的小 batch decode 存在取舍。进一步的[共享权重、动态 W4A4/W4A16 PR #54614](https://github.com/vllm-project/vllm/pull/54614) 截至本次读取仍 Open，其公布速度来自特定后端组合、3.8、Max-Q 且关闭 MTP，不能直接套用到本机 MTP4 的 M=5。

可做的开发是：先拿两个固定后端的可靠整模基线，再研究按 M 分派，并确认权重布局共享、无需每次重新打包、图模式下真实命中。优先复用上游已有工作。本机尚未验证动态路径，不能当成 0.31 的现成参数直接打开。

## 3. 投机解码：优化每轮有效产出

核心指标是 `有效输出 token /（draft + target 校验 + head/采样 + 状态处理耗时）`。提高接受比例或加长草稿本身不保证更快。

先分解 MTP4 中 M=1/M=5 的 head 调用及 draft、target、GDN 状态回退时间，检查有无冗余操作。自回归 draft 步之间有依赖，不能直接把多次 head 合并或删除来改跑分；融合 logits/采样也需保留采样、惩罚、工具约束等语义，并核验固定源码是否已有相同融合。

按普通问答、中文检索、代码、工具调用、完整思考任务分别比较 MTP 深度和匹配的 DFlash。已有思考组输出只有 reasoning，下一轮必须包含正常结束的最终答案、任务成功率和总完成时间。3.8 的 DFlash2/DSpark 可作为独立路线，但没有必要将换模型、换量化和换草稿同时进行。

## 4. Prefill 与 MoBase：减少重复工作

冷输入需要计算的部分，可研究保持当前精度的量化/激活融合，先确认固定实现尚未覆盖且能减少中间读写；NVFP4 路线等质量问题解决后再评估。本轮已测变慢的 FP8 tile、Split-N 优先级下降。GDN 在现有采样中份额较小，暂不作为第一重写目标。

重复长输入先测实际前缀命中：稳定 system prompt 和工具 schema 的顺序，把语义允许的动态内容放在稳定部分之后，检查多轮上下文和检索结果是否造成大段重复计算。前缀缓存已经开启，工作是提高实际复用，不是再开一次开关。它主要跳过共享输入计算，不直接提升每个新 token 的 decode 速度。[vLLM APC 文档](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) 还说明 hybrid/Mamba 状态的 block 边界约束；新共享前缀 checkpoint 选项有模型、MTP 和版本条件，须核对固定 0.31 实现后再决定是否测试。

从 MoBase 用户入口记录：排队、prefill、首 reasoning、首工具调用、工具耗时、首正文和总完成时间。再判断能否并行无依赖工具、减少重复模型调用、精简重复上下文。保持检索覆盖、引用和工具准确率；关闭思考或削减回答质量不能算同能力提速。路由简单任务到现有 9B 属于单独的产品策略实验，不混入 27B 算子收益。

## 建议的下一轮闭环

先采集能正常完成的业务风格请求和历史故障线索，建立循环/终止回归；保留 FP8 权重、BF16 head，验证 MTP4、缓存复用和 MoBase 调用链。在质量基线成立后，低精度 head 才作为独立分支，做数值、冷/热算子计时与无 profiler 整模 A/B/A。每个候选记录实际 dispatch、输出和接受率变化，并在多个时段重复，不能继续只凭每场景 n=3 就认定小幅收益。

如 head 收益有限或质量不过关，保留 BF16 head，不自动转 NVFP4。NVFP4 必须先完成历史故障的排查与质量验证。胜出方案回到原 27B 0.45 显存预算做 1/2/4/8 并发、真实长输入和四服务共存验证；分别报告单用户速度、总吞吐、p95 和 embedding 延迟。Embedding 保持运行。开发基线继续固定官方 vLLM 0.31.0 的已记录提交；需要新依赖时另开对照，不能覆盖原基线。

已有 MTP4 候选可先进入生产预算与 MoBase 验收，不必等待新算子：它在此前受控普通输出实验中优于 MTP2，但还没有证明原四服务共存预算下同样受益或排除偶发循环。以完整任务正确结束作为采用前提，GPU 测量串行安排，避免相互干扰。

</details>
