# DDTree 可行性笔记（任务 E，只调研）

日期 2026-10-14。未使用 GPU / SSH / 凭据。标注约定：**[源码]** = 已在 vLLM 固定提交 `db9527a4…` 的 GitHub raw 文件中核对；**[声称]** = 论文/博客/PR 自述，未复现；**[推断]** = 本文判断。vLLM 路径均指该提交。

## 1. DDTree 相对 DFlash 链式验证改了什么

- DFlash 一次前向对 `[bonus, mask×15]` 输出每个未来位置的边缘分布，取各位置 top-1 排成**一条链**交给目标验证；链上任一位置错，其后全部作废。[源码] 本机 V2 路径正是如此：`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` 每请求 bonus+N mask，`rejection_sampler_utils.py:_rejection_kernel`（约 557–666 行）读 `draft_sampled[logit_idx+1]`，按线性顺序逐位接受，首个拒绝处重采样。
- DDTree 用同一份 DFlash 草稿分布（无需第二个模型、无需重训），以各位置边缘概率的路径乘积为分数，用最佳优先堆在**节点预算 B**（不含根）内建前缀闭合的树；树按深度展平成序列，position id=深度，**树 attention mask** 让每个节点只看 KV 前缀、自己的祖先和自己；目标一次前向打分全部节点，沿树走一遍按目标自身规则（贪婪/采样）取最长匹配路径，第一个不匹配的目标 token 作下一个 bonus，然后把 KV **压缩到被接受路径**。[声称，arXiv 2604.12989 HTML]
- 代价：验证 token 数从 16 变成 1+B，目标前向不再是 M≈16 的窄 GEMM；B 大时 decode 从带宽受限滑向算力受限。论文 Fig.3 称 τ 随 B 单调升而加速比在 B≈256–512 见顶。[声称]

## 2. 已报告的加速比及条件（逐条不可外推到本机）

| 来源 | 条件 | 结果 | 状态 |
|---|---|---|---|
| DDTree 论文 Table 1 | 8×H200 分片评测，Qwen3-4B/8B、Qwen3-Coder-30B-A3B（全是纯注意力模型，**无 GDN**），bf16，block 16，B∈{16..1024} 取每格最优，T=0，max 2048 新 token，目标用 PyTorch SDPA（FA2 不支持树 mask），batch 未说明 | 例：4B AIME24 5.56×→7.27×，τ 7.54→10.37；8B MATH-500 5.56×→7.52×；30B-A3B HumanEval 6.09×→8.22×；SWE-bench Lite 2.77×→4.38×，τ 3.61→5.71；60 个组合全胜 DFlash。T=1：4B AIME24 3.50×→5.31× | 声称；倍数相对自回归；预算按数据集择优，有挑选 |
| Luce DFlash+DDTree 博客 | 单张 RTX 3090 24GB，llama.cpp 分支，目标 Q4_K_M、草稿 BF16，**B=22**，Qwen3.5-27B | 峰值 207.6 tok/s vs AR 38.0（5.46×）；HumanEval 均值 129.5（3.43×）；Qwen3.6-27B 仅称 “up to 2×”，发布当日草稿 HumanEval 约 78 tok/s、接受长度 5.05（同草稿在 3.5 上 9.18）。页面无 DDTree 对链式的对照数字 | 声称；Q4 量化、非 vLLM |
| oMLX PR #805 | MLX，Qwen3.5-4B-4bit / 27B-4bit，B∈{2,4,6,8}，192 token，机器未写；此 PR 2026-10-07 被机器人因不活跃关闭，未合并 | 27B 代码 18.2→19.5 tok/s（1.07× vs DFlash，AR 16.6）；27B 散文 7.5→13.4（1.78×，但仍低于 AR 15.9）；4B 散文 1.77×。**本任务说明里的“再快 77%”来自 4B/27B 散文的小基数，不是代码/长输出** | 声称；基数低，预算≤8 |
| AEON vLLM fork | DGX Spark GB10 sm_121a，NVFP4，Qwen3.6 变体，k=15 | 作者明说：**DDTree 目前不比平坦 DFlash 快**，DFlash 仍是生产赢家；非平坦分支提交与基线质量不等价；M53 探针平均接受长度约 2.7、位置 0 之后分支接受多为 0% | 声称；是最接近本机栈的负面证据 |

读法：学术 5–8× 来自 H200 上小/MoE 全注意力模型；落到 Qwen3.6-27B 这类混合模型，已公开的实测只有 Luce（Q4、3090）。Luce 同页给出 Qwen3.6 草稿接受长度仅 5.05（对比 3.5 的 9.18）[声称]，说明 3.6 的 DFlash 草稿质量低于论文设定，树的增益空间未知。本机自测 DFlash15 每轮接受+bonus 中位数为 3.93（思考）/2.62（不思考）（07 号文档，实测）——比 Luce 的 5.05 还低，树的上限更紧。

## 3. 在 vLLM 0.31 V2 上要补什么，以及固定提交里已有什么

| 环节 | 需要 | 固定提交现状 |
|---|---|---|
| 树 attention mask + 分页 KV | 目标 attention 支持任意祖先 mask；SM120 上 FlashInfer/XQA/FA2 均无树 mask | **[源码] 无 `tree_attn.py`**；`vllm/v1/attention/backends/` 仅有 flash_attn、flashinfer、flex_attention、triton_attn 等，未检索到 tree 后端；`config/speculative.py` 仅提到 “non-tree speculation”。vLLM 论坛称树模式经 `--speculative-token-tree`、静态树，且旧写作称 tree attention 只在 V1 EAGLE 上实验——本提交已找不到。可选手段：flex_attention（编译慢、非 graph-safe 快路径，AEON 自述“graph-safe 不等于快”）或自写祖先 mask 的 Triton/CUDA kernel 并兼容分页 KV（AEON 自述仍缺）。 |
| GDN 递归状态分支 | 48 层 GDN：每个树节点的状态必须由**父节点**而非前一个展平 token 推出；接受后只提交接受路径 | **[源码] 只支持链**：`gdn_attn.py` 元数据 `spec_state_indices_tensor [batch, num_spec+1]`、`num_accepted_tokens`、`uniform_spec_sequence_length`，每个 spec token 一个状态槽，接受 n 个后读第 n 个槽——隐含“展平序 = 因果序”。树需改成按父指针索引并在 FLA 递归核（`model_executor/layers/fla/ops/`）与 causal-conv1d 中支持“从父槽读、写自己的槽”，这是最大工作量。`recoverssm_metadata.py` 的 `commit_recoverssm_state(num_accepted_tokens)` 也只按个数。 |
| KV 槽 | 节点按展平序占槽；接受后把被接受路径的 K/V 搬到连续位置并释放其余 | V2 `_prepare_dflash_inputs_kernel` 为链顺序写 slot_mapping。树需要提交后的 KV gather/compaction kernel 与 block table 回写，且与 prefix cache（混合模型已有“未识别草稿 KV 组”告警，07 号文档）交互；AEON 建议树模式关 prefix cache。[推断] |
| 树的拒绝采样 | 沿树走：每层在兄弟间做多候选验证（贪婪：匹配任一子节点；采样：需残差分布多分支修正） | **[源码] 线性**：`rejection_sampler_utils.py:_rejection_kernel` 与 `rejection_sampler.py:_verify` 的输入是 `draft_sampled`+`cu_num_logits`，无父指针。贪婪树验证改动较小（比较各子节点），采样需新算法。 |
| CUDA graph 形状 | 树大小固定（预算 B 固定、不足补空/掩码节点）才能保持 FULL graph；动态树要分桶 | 现有 `max_num_seqs*(num_spec+1)` 固定形状缓冲（`gdn_attn.py`），B 固定即可沿用；每步树形状变化需把 mask/父指针做成 graph 内输入张量。[推断] |
| 草稿侧 | 取 top-K 与建树（O(B log B)）放 GPU，避免 D2H 同步 | `dflash/speculator.py` 已在 GPU 上采样 draft；建树需新 kernel 或接受每步一次同步。[推断] |

另：vLLM RFC #47572 / PR #47576（ReplaySSM，缓存 SSM 输入而非状态，用于混合模型投机解码）以及论文 TreeWY（arXiv 2608.20961，GDN 混合模型投机验证）、SGLang 侧 DFlash-TfM “无需回滚的树验证 GDN kernel”——这些仅见于搜索摘要，**未打开核实**，是下一步最该读的上游线索。

## 4. Luce 的三个 SSM 回滚 kernel 与对应关系

页面只给出名字，**未说明各自作用**；以下是依名称的 [推断]，立项前需读其源码（llama.cpp 分支）：

- `ggml_ssm_conv_tree`：树感知的短卷积——每个节点的卷积窗口取自其**祖先链**的前 3 个输入，而非展平序前驱。对应 vLLM `causal_conv1d` 的 spec 路径（需按父指针取窗口）。
- `ggml_gated_delta_net_tree`：GDN 递归按树展开，每个节点从父状态更新并写入自己的状态副本。对应 FLA `fused_recurrent_gated_delta_rule` 的 spec 模式，把“前一个 token”换成“父节点索引”。
- `ggml_gated_delta_net_tree_persist`：推测为接受路径确定后，把被接受叶的状态“持久化”为新的提交状态（取代回滚/重算）。对应 vLLM 的 `num_accepted_tokens` 选槽 + `commit_recoverssm_state`，需改为“按接受叶节点索引选槽”。

映射思路：vLLM 已经是“每节点一槽、事后选槽”，**不需要回滚重算**，所以结构上比 llama.cpp 更接近；缺的是父指针语义。状态规模：Qwen3.6-27B 每 GDN 层每节点状态为 H×dk×dv 个 fp（具体维度 [待核]），B=22~64 个槽×48 层×并发，显存需另估。[推断]

## 5. 工作量与风险排序（[推断]，人周为粗估）

1. GDN/conv 父指针递归核 + 状态槽提交（高风险，约 3–5 周）：数值与基线需逐 token 对照；AEON 的树/GDN 质量问题正出在这里。
2. 树 mask attention on 分页 KV，SM120 FlashInfer/XQA 之外的路径（高风险，约 2–4 周）：官方后端无支持；flex_attention 可能破 CUDA graph 或太慢，FA2 不支持（论文自己退回 SDPA）。
3. KV 压缩/槽管理 + prefix cache 交互（中，约 1–2 周）。
4. 树拒绝采样（贪婪：低，约 3 天；采样：中，约 1–2 周）。
5. 建树与 graph 固定形状（中，约 1 周）。
6. 性能不确定性（最大不确定项）：本机接受+bonus 仅 2.6–3.9；B 增大后 decode 验证不再是 M≈16 的带宽受限 GEMM，FP8 GEMM 效率（任务 A 的 617 TFLOP/s 上限）与 DFlash 草稿本身的耗时会成为约束。

合计约 2–3 人月量级，且需要 SM120 attention 内核工作（与本项目其他方向重叠度低）。

## 6. 结论：目前 NO-GO（不立项下下轮实现），保留条件式 GO

理由：固定提交无任何树基础设施；已公开的混合模型/GDN 实测中，最接近的 vLLM fork 作者自己说树不比 DFlash 快；Luce 的 2× 是 Q4 量化 3090 数字；本机 DFlash 接受长度偏低，上限收益未知。

先做的零 GPU / 低成本预检（**条件式 GO 的前提**，全部满足才立项）：
1. 用本机已有 DFlash15 的草稿 logits 做**离线 oracle 估算**：在 07 号文档同类 payload 上记录每步各位置 top-K 分布与目标真实 token，离线计算在 B=16/32/64 下 DDTree 的期望接受长度 τ_tree；要求 τ_tree/τ_chain ≥ 1.4（思考负载，τ_chain≈3.9），否则止步。
2. 实测目标在 M=1+B（B=32/64）下单步前向相对 M=16 的耗时比（FP8，SM120，CUDA graph）；要求比值 ≤ τ_tree/τ_chain 的 70%，否则净收益为负。
3. 读完 PR #47576、TreeWY、SGLang DFlash-TfM kernel 与 Luce 三个 kernel 源码，确认 GDN 树递归可在 FLA 核上以父指针实现，且无需整层回滚。
4. 确认存在可用于分页 KV 的祖先 mask 注意力方案（上游合并或可移植的 Triton kernel），并在 SM120 上通过数值检查。
5. 任务 A 已确定 FP8 GEMM 上限，任务 D（词表裁剪）结果已出——二者决定 decode 剩余余量，避免同一瓶颈重复投入。

仅当 1+2 通过（零实现成本即可判定）才进入 3–5；任一不过则关闭。

## 7. 来源

- [DDTree 论文 arXiv 2604.12989](https://arxiv.org/abs/2604.12989)（HTML 版，数字见第 2 节）
- [Luce DFlash+DDTree 博客](https://rits.shanghai.nyu.edu/ai/luce-dflash-brings-2x-speculative-decoding-to-qwen3-6-27b-on-a-single-rtx-3090/)
- [oMLX PR #805](https://github.com/jundot/omlx/pull/805)、[AEON DDTree fork](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DDTree)
- vLLM 固定提交：[gdn_attn.py](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/gdn_attn.py)、[rejection_sampler_utils.py](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py)、[dflash/speculator.py](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py)、[recoverssm_metadata.py](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/recoverssm_metadata.py)
- 未核实线索（仅搜索摘要）：vLLM RFC #47572 / PR #47576、[TreeWY](https://arxiv.org/pdf/2608.20961)、[vLLM 论坛 draft tree](https://discuss.vllm.ai/t/what-is-the-recommended-way-to-support-dynamic-pruning-for-speculative-decoding-draft-trees/2630)
