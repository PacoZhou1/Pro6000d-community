# Qwen 27B 优化路线与算子开发边界

日期：2026-10-13。依据：本次服务器实查、[基线报告](01-server-baseline.md)、[项目审查](02-project-audit.md)、[上游调研](03-upstream-research.md)、[MoBase 调用链](04-mobase-integration.md)。本文件中的待测方案不是已经实现的性能收益。

**2026-10-14 后续更新：**下文早期列出的 GEMM 尺寸/BF16 归因、三种定向候选和整模 A/B/A 已完成，见[定向实测](09-targeted-gemm.md)。按新证据调整后的后续优先级见[下一轮路线](10-next-inference-roadmap.md)；本页保留早期计划，不能把其中“下一轮”当作仍未执行的任务。

## 当前算子开发判断（2026-10-14更新）

已完成六份真实 GPU trace：19K 冷 prefill、MTP4 普通 decode、DFlash15 思考 decode 各两份。按捕获窗口内全部 GPU kernel 耗时求和，FP8 GEMM 分别约占 **68.49%、62.03%、64.10%**；decode 的 BF16 GEMM 另占约 29.50% / 21.54%。GDN 与卷积合计约 2.5–5.2%，本轮应优先 GEMM。[完整热点、口径及实验](08-gpu-profiling.md)

首个算子实验已经落地：同依赖隔离镜像中，将 FP8 block GEMM 从 CUTLASS 切到 B12X 1.5.0，完成真实随机算子数值检查与关闭 profiler 的整模 API A/B。全量切换使 prefill 首 token 约 2.225→2.662 秒，decode 中位数基本不变，因此不采用该全量替换。此轮集成已有算子，**未编写新的 CUDA kernel**，也没有把失败候选算作提速成果。

下一轮自研先对头部 FP8 GEMM 记录逐层 M/N/K、layout 和硬件计数器，再研究 SM120 tile / scheduling；不能仅根据高占比断定是带宽瓶颈。MTP 验证包含多行，单测 M=1 会误导，已有微基准在 M=5/16 观察到 B12X 退化。BF16 大 GEMM 值得做模块归因，但当前不能证实全部属于 lm_head 或 draft。GDN、量化融合不作为首选重写目标。

报告保留原始 Nsight 数据路径与哈希、按名称归类的 kernel 排名、NVTX 关联缺口、无 profiler 性能和数值检查。客户四卡 GLM/MiMo 补丁作为技术参考，不整文件覆盖本次单卡 dense Qwen 实现。

## 1. 建议顺序

1. **先保持 Qwen3.6-27B-FP8 权重，对比官方 vLLM 0.27.1 和 0.31.0。** 当前已有 FP8、MTP2、prefix cache、chunked prefill；不能把“打开这些开关”写成新优化。0.31 有针对 SM120 的 GDN prefill 和 MTP 更新，适合做新开发基线。
2. **保留 embedding，停止其他干扰服务做测试。** 用户明确要求 embedding 不停。本轮临时容器在 `127.0.0.1:18000`，原模型容器保留；不得把测试结果称为完全独占 GPU。
3. **确认实际后端，优先验证更强投机解码。** 同样的参数不保证同样的kernel；检查GDN、attention、graph mode和接受token/轮。在已有MTP2对照上，先测MTP3/4及匹配3.6的外部DFlash。社区同卡FP8对照提供了较大的潜在收益线索，但旧栈、计时与题目不同，须重新验证。[补充研究](06-community-priorities.md)
4. **再单独比较3.8 / NVFP4及其匹配草稿。** 官方3.8-FP8已测得与3.6相近，进一步考察质量、任务耗时及DFlash2/DSpark；NVFP4用于验证精度和带宽的权衡。不能同时换版本、模型、精度和草稿，再宣称某一个改动带来全部收益。
5. **仅对 profile 证明的热点写算子。** 优先围绕 dense MLP、GDN、small-M launch/state 管理；不搬 GLM/MiMo 的 MoE route 和 sparse MLA 补丁。

## 2. vLLM 源码基线

建议固定 **v0.31.0 / `db9527a46873454610df6dbedf79a36d6bf1a7f6`**。这是源码开发起点；是否作为生产版本由本机 A/B 和 MoBase 质量验收决定。[官方 release](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)

```bash
git clone --branch v0.31.0 --single-branch https://github.com/vllm-project/vllm.git vllm-qwen-sm120
git -C vllm-qwen-sm120 rev-parse HEAD
git -C vllm-qwen-sm120 switch -c codex/qwen27b-sm120
```

先使用官方镜像建立性能参考，再在相同 CUDA / torch / FlashInfer 依赖下构建源码。固定 tag、完整 commit、平台镜像 digest、权重 revision、配置、补丁 hash；不要只记录 `nightly`。原仓库 0.26/0.29 的整个 Python 文件覆盖方式不适合这次移植。

已量化的 Qwen block-FP8 checkpoint 继续使用其自身 FP8 配置。0.31 对“在线把 BF16 量化成 FP8”的参数调整，不能被误解为必须把现有 block-FP8 改成 per-tensor。[配置依据见上游报告](03-upstream-research.md)

## 3. 实验矩阵

所有比较保留 TP1、相同权重、上下文限制、KV dtype、采样、embedding 服务状态。先 C1；并发 4/8 是后续吞吐实验，不能用总吞吐替代每用户速度。

| 编号 | 一次变化 | 重点指标 | 继续条件 |
| --- | --- | --- | --- |
| A | 0.27.1 → 0.31.0，仍 FP8/MTP2 | 短/长 TTFT、decode 估计、服务端 prefill、接受率 | 冷缓存与长输出可重复，无错误/明显质量退化 |
| B | GDN `flashinfer` / `triton` | 长输入 prefill、recurrent state 正确性 | 日志证实实际 dispatch，长上下文误差可接受 |
| C | 当前 FP8 linear 的 auto / cutlass / marlin | small-M decode 与大-M prefill | 确认 weight/activation scale 语义及输出质量 |
| D | MTP off / 1 / 2 / 3 / 4 | accepted/drafted、TTFT、TPOT、QPS | 按真实代码/中文/工具题选，不能只用重复文本 |
| E | 官方可选 B12X attention / linear | Graph mode、decode、长输入 | 确认包版本和固定 tag 实際 enum；B12X attention用block64/128，同block FlashInfer对照 |
| F | Qwen3.8-27B-FP8 | 完成率、工具正确率、总任务时间 | 质量至少不劣于现有 3.6，单纯换名字不承诺加速 |
| G | NVIDIA 3.8-27B-NVFP4 | 带宽、MLP 延迟、TTFT、业务质量 | 混合 FP8/NVFP4 层逐一确认，量化回归通过 |
| H | 3.6-FP8 MTP → 匹配DFlash；3.8另测匹配DFlash2/DSpark | accepted tokens/轮、草稿与校验耗时、TTFT、decode、状态内存 | 相同KV/backend下补MTP对照，支持所用block长度，中文/工具/长历史质量通过 |

A/F已测；追加实验已完成D中的MTP2/3/4及H中的3.6-DFlash7/15对照，详见07文档。B/C/E尚待实际profile决定，G仍作为独立精度变量。上表编号用于引用，不代表当前执行优先级。DFlash与DFlash2结果不可混用；已有社区200+报告不是本机目标保证。

0.31 的 FlashInfer XQA 快路径还有构建/下载、head_dim、KV 精度等 guard。若日志仍退到 PIECEWISE，应先查实际 dispatch 与依赖是否可用；不能强开 FULL 后忽略 backend 的兼容检查。B12X 在固定 tag 的 enum/可选包需以源码和镜像核验；attention实测须记录解析后的block size与KV容量，B12X要求64/128，并为同一block size补FlashInfer对照。详细说明见上游报告。

本轮已完成0.27.1的MTP2/off对照：1024输出中位90.73 vs 50.48 tok/s。无MTP启动时捕获了FULL decode graph，整体生成仍更慢，当前应保留MTP2。追加实验已补0.31同栈MTP2/3/4，普通输出以MTP4更好；gamma1与0.31的MTP关闭对照仍未测，不能宣布某个gamma全局最佳。

0.31同3.6权重/MTP2的首轮启动已确认FlashInfer GDN prefill、XQA decode与FULL decode图捕获；图捕获不证明每个请求实际replay。固定语料n=3显示长输入TTFT和4.8K/1024输出略有改善，但0.7K/256输出decode及总时反而变慢；本轮不能把0.31称为所有场景更快，也不能把变化拆算给某一个kernel。镜像包含FlashInfer、Transformers、CUTLASS DSL升级，旧版首轮采样时还有后台镜像下载；无下载干扰的回测已完成，1024输出为89.71 tok/s。0.31长输入和旧版回测长输出各有一个窗口与短embedding请求重叠，后续严格同负载复测需补足样本。仍保留MoBase业务质量门槛。[原始证据与口径](01-server-baseline.md)、[后端与警告](03-upstream-research.md#72-本机首轮v031运行验证)。

`max-model-len=262144` 是允许的上限，不是每个短请求都计算 262K token。减少它主要改变容量/内存/启动规划；不能直接承诺短请求会快。`gpu-memory-utilization` 也不是算力占比或 GPU 隔离机制。当前服务显存必须按实际进程与峰值规划，不能只把各容器的百分比相加。

## 4. 算子研发清单

先记录运行时 `(M,N,K)`、dtype、权重/activation scale shape、stride、SM 数、CUDA Graph 模式。MTP2 的一次校验可能包含多 token，不能把所有 decode kernel 当作 M=1。

对 Qwen27B TP1，dense MLP 主形状为 gate/up `(N=34816,K=5120)`、down `(N=5120,K=17408)`。先 profile small-M 的权重读取、scale 加载、split-K reduction 与 launch 开销；再考虑融合 dynamic quant、SwiGLU、残差/归一化。GDN 包含状态更新，正确性必须同时比输出和 final state。full-attention 只有 16 层，不能因为它容易写就默认是 64 层模型的主要瓶颈。

验收分两层：

- **算子层：**同一随机种子、不同长短序列和 batch、尾块、连续 prefill→decode、MTP reject/accept、FP32 reference、dtype 容差；测 warm/cold L2、CUDA events、graph replay，排查 NaN/Inf 与 race。
- **整模层：**保持 API usage/stop/工具 JSON/思考分离正确，测端到端 TTFT 和 TPOT。算子快 10% 不等于整模快 10%；按 profile 占比估算上限，再以真实结果决定是否保留。

只有在当前栈的 profile 仍显示这些算子是主要开销时才投资自研。SGLang 相关 PR 已出现“isolated GEMM 更快但单请求 serving 不升反降”的一手证据，见[上游研究第 5 节](03-upstream-research.md)。

硬件带宽也给预期提供约束。[NVIDIA 工作站卡规格](https://www.nvidia.com/en-gb/products/workstations/professional-desktop-gpus/rtx-pro-6000/)为1792 GB/s。仅作量级估算，若每次普通自回归step要读约27–31GB权重，忽略全部其他开销的带宽上限约58–66 step/s；真实工作集不等于模型文件总大小，必须用profile确认。实测无MTP约50.5 tok/s，与带宽受限的假设相容，但不是证明。MTP一次target校验可接受多token，能超过上述“每step一个token”的简化估算；NVFP4则减少部分权重流量。这也是不应承诺只改一个FP8算子就从90直接到200 tok/s的原因。

## 5. MoBase 验收与恢复

MoBase 主 Agent 开思考并绑定工具，最终 generation 另发关思考请求；真实输入可达到数万字符。合成直连速度不能直接证明用户首次正文更快。至少补测：短问答、真实风格合成检索、工具调用 JSON、长历史、多轮思考、引用/表格、正常 stop，以及有图片时的多模态路径。

计时分开记录首 reasoning、首 content、首 tool_calls、MoBase 首 answer_chunk 和总耗时；embedding 请求用量与延迟同时作为保护指标。0.27回测与0.31的FP8 attention启动日志都提示q/k/v/prob scaling部分缺失或取1.0，合成API冒烟通过不足以排除精度问题；上线前核对scale语义并完成长上下文、思考、工具和多模态业务质量回归。禁止用关闭思考或减少输出预算掩盖业务能力下降。

临时测试容器不覆盖原容器，模型目录只读挂载，使用原权重不进行转换。回退为停止临时容器并启动原 `vllm-qwen-27b`，核对 `/health`、`/v1/models`、聊天与工具 smoke。9B/ASR 在用户授权的测试期间暂停；ASR 原有显存启动失败需单独解决，不能不检查资源就重启到失败循环。首次基线恢复见[基线报告](01-server-baseline.md)，追加实验的最新结束状态及一次启动重试见[07文档](07-specdecode-experiment.md#服务恢复与启动问题)。

本轮未执行完整业务回归、未承诺 130/200 tok/s、已用临时localhost容器测试3.8-FP8，未替换生产模型或测试NVFP4、未编写生产算子补丁。后续排序由本次实际 A/B 结果调整。

用户追加的ASR恢复已单独落地：原0.27.1/3.6保持，仅27B显存预算0.50→0.45；KV池11.15GiB/318743 tokens，单262K上下文保留、长上下文并发余量减少。四服务健康、ASR真实合成音频转写通过，旧0.50容器停止保留可回退。该容量调整不计入前面的性能A/B收益，详见[最终状态](01-server-baseline.md#11-用户追加缩减27b-cache恢复asr)。
