# Claude 中断后的 FP8 实验接手

2026-10-15，America/Los_Angeles。恢复范围为核对 Claude 已做的 A/D/P/F 实验，补齐 F1/F2 接入验证，以及实际构建和测试 CUTLASS MXFP8、PR5521 派生 GEMM。所有改动在 `_dev/docs`，不修改模型权重或正式服务。原 27B、9B、ASR 继续停止，embedding 保持运行。

## 接手时已核实的工作

已对收到的 7,437 个文件建立 SHA256 清单，并从服务器取回 F 实验遗漏的原始日志、数值结果和服务记录。原始文件保留，新工作使用独立 `takeover` 与 `b-mxfp8-recovery` 目录。接手时 GPU 仅有 embedding，实际健康检查为 200，返回 1,024 维有限向量；容器身份、启动时间和重启计数未变。见 [初始状态](evidence/sm120-next/takeover/services-initial.json)、[收到文件的清单](evidence/sm120-next/takeover/received-artifacts.json)。

| 已做方向 | 原始证据支持的结论 | 不能据此声称的结果 |
| --- | --- | --- |
| A：FP8 GEMM ceiling | M16160、N34816、K5120，原 CUTLASS block128 约 9.307 ms；cuBLAS rowwise 约 8.737 ms；cuBLAS MXFP8 约 9.718 ms | 没有测过的 CUTLASS MXFP8 或新 rowwise 实现也必然更慢；所有 GEMM 方向均无空间 |
| D：draft 词表 64K | 两个夹持窗口的 12 组普通/思考固定输出，配对改善约 13.55% 和 13.72%；固定 greedy 校验通过 | 全部业务质量无损，或正式服务已经启用。目标模型验证端仍保留完整 BF16 head；D5/D6 仍是计划 |
| P：prefill / V2 | V2 两轮长 prefill TTFT 改善约 1.41% / 1.44%；本组 decode 一致 | 小幅改善可直接推广到正式四服务预算、长上下文与真实业务 |
| F：GDN 两段融合 | 直接算子数值测试 45/45；F1/F2 局部函数明显变快；组合长 prefill 约改善 1.63% | 组合保持生成一致：B 的 6 个长 decode 正文 hash 全部与 A 不同 |

详细复算与限制见 [A/D/P 恢复审计](evidence/sm120-next/recovery-existing-audit.md)和 [F 恢复审计](evidence/sm120-next/f-fusion/recovery-audit.md)。D 的成绩属于特定投机解码实验，不是本轮新内核的成绩。

## F1/F2：先诊断接入，再谈采用

F1 融合 gated RMSNorm 与 per128 FP8 输入量化；F2 融合 causal conv、激活与 post-conv 整理，状态更新仍有独立 kernel。组合 B 的 prefill 配对改善为 1.6301%；F1 单独 BN 为 1.0371%，F2 单独 BC 为 0.5715%。BN/BC 在 A2 后顺次执行，没有各自前后夹持基线；BC 的改善与 A1→A2 约 0.58% 漂移同量级。

| 臂 | 六个正式 decode 与 A1 正文相同 | MTP accepted 总数 | accepted 逐项与 A1 相同 |
| --- | ---: | ---: | ---: |
| A2 | 6/6 | 4,178 | 6/6 |
| B：F1+F2 | 0/6 | 4,105 | 0/6 |
| BN：仅 F1 | 6/6 | 4,176 | 0/6 |
| BC：仅 F2 | 6/6 | 4,178 | 6/6 |

组合 B 已经带新的 splitting op，不能用“旧版没修编译边界”解释分叉。历史 WV 的 192 次 F1 实际输入检查零差异，但其编译边界为旧版，也不能替 B 作证。单独核的逐位相同不自动证明整个包装器、状态更新、编译图和投机接受过程相同。

恢复版诊断将官方 conv/post 与候选从相同真实输入和初始状态运行，每次比较完整 q/k/v/g/beta 与全部 conv_state；F1 同时比较 FP8 字节和 scale 位。遇到差异立即终止并保留证据。诊断加入 clone、同步和参考计算，其耗时不作为性能成绩。见 [诊断 runtime 与边界](evidence/sm120-next/takeover/f-diagnostic/README.md)。

源码审查另发现明确缺陷：F1 的 `md=None` warmup 路径没有填充 q/s，却继续读它们做 GEMM。固定上游实现仅预热后返回，原 forward 会正常量化零初始化 core 输出；实验包装器直接消费未初始化缓冲区。该缺陷需要修，但尚不能认定它造成历史真实请求分叉。对“绕过 packed 入口就丢失 MTP 专用路径”的猜测，固定源码不支持：packed 入口最终调用相同的 unpacked 方法，后者仍有 fused MTP 分支。见 [固定上游源码及 SHA](evidence/sm120-next/takeover/source/source.json)。

## 新 GEMM 对照的准确含义

新比较固定为 `A[M,5120] × W[34816,5120]ᵀ → BF16[M,34816]`，M 为 2,741 和 16,160，输入全部为随机数据。block128、MXFP8 K32 幂二 scale、rowwise 是三种不同数值表示，各自与自己的参考比较；不要求三臂输出相同，也不把这种单算子速度当成原权重可直接替换的证据。

[FlashInfer #5521](https://github.com/flashinfer-ai/flashinfer/pull/5521) 是 MiniMax-H3 特定尺寸的融合 FC1/SwiGLU 核，并非可以直接调用的通用 Qwen rowwise GEMM。因此这里明确新增一个 **PR5521 派生 rowwise GEMM**：保留 FP8 MMA/TMA 主循环，修改实际 K/N，输出 canonical gate/up BF16，去掉原模型专用的前后融合。上游的 732–741 TFLOP/s 不能移作本机该算子的成绩。源版本、派生修改与构建记录见 [来源与实验方案](evidence/sm120-next/b-mxfp8-recovery/research.md)及 [派生核说明](evidence/sm120-next/b-mxfp8-recovery/pr5521_derivative.md)。

验收顺序为：编译 → 每种 scale 的数值参考 → 派生核完整矩阵（包括尾块） → 改输入后 CUDA Graph 重放 → 31 组随机配对计时。计时单独记录 GPU 时钟/功耗；保留 embedding，所以不锁全卡频率。warm 与写入自有大于两倍 L2 缓冲区的扰缓存条件分开报告；扰缓存不等于证明完全冷缓存。

输入量化、MX scale 排列、rowwise 权重预排不在 GEMM GPU 计时区间。Graph 时间也不含 host TMA descriptor 编码。即使某臂超过 10% 门槛，也只是支持后续权重转换和质量调查；本轮不转换权重。原文“约 6% scale 代价”是估算，不能作为本机 profiler 已测出的固定占比；相同缩放格式也不能单凭调度变化就保证逐位相同。

## F1/F2 新整模结果

真实模型组合诊断已完成：F1 2,112 次、F2 2,448 次检查零差异，6 个正式 decode 的正文与逐项 draft/accepted 都与 A1/A2/BC 相同。该诊断仍使用旧 warmup 分支，因此不能把恢复一致归功于后来的补丁；插入参考计算、clone 和同步改变了运行安排。前后 embedding 功能与 18 次周期健康检查全部通过。见 [诊断实测分析](evidence/sm120-next/takeover/fdiag-results.md)。

补丁仅在 `md=None` 分支加入官方 per128 FP8 量化，填充 q/s；融合数学、权重、MTP4 与 BF16 head 不变。实际 CUDA poison probe 在 M=1、256、32768 均验证：旧分支留下全部预填 NaN，新分支 FP8 字节与 FP32 scale 位均与官方一致且有限。这是抽取函数的输出契约测试，packed warmup 使用明确的无状态 stub，不冒充真实预热或图集成测试。整模集成另由下述无诊断 A/B/A 验证。见 [完整 probe 结果](evidence/sm120-next/takeover/null-v1/null-metadata-poison.json)。

三臂使用相同冻结 runtime，仅 `PRO6000D_F_ENABLE` 从空值变为 `norm,conv` 再变回空值；相同镜像、模型、MTP4、FP8 KV 与自动 KV 分配，均为 30.75 GiB / 848,709 tokens。每臂独立预热，正式 prefill 12 次、decode 6 次。

| 指标 | A1：V2 对照 | B：V2 + 修复后的 F1/F2 | A2：V2 对照 |
| --- | ---: | ---: | ---: |
| 18,939 输入 / 1 输出，TTFT 中位数 | 2.212220 s | **2.189054 s** | 2.230441 s |
| 4,787 输入 / 1,024 输出，decode 中位数 | 106.6206 tok/s | 106.6227 tok/s | 106.6184 tok/s |
| 6 条 decode 正文相同 | 6/6 | 6/6 | 6/6 |
| 同例 MTP drafts / accepted 相同 | 6/6 | 6/6 | 6/6 |

长 prefill 按同请求相对前后基线均值计算，耗时下降中位数 **1.4896%（约 32.89 ms）**；12 条均快于各自前后基线。A1→A2 的配对漂移为 +0.8228%，所以这里只确认本窗口的小幅收益。decode 配对速度变化 **+0.0076%**，没有可用的 decode 加速。三臂全部 18 个正式窗口的 embedding 指标无变化；健康监控 53 次全为 200，6 次前后实际向量请求均通过。

12 条 prefill 与 6 条 decode 的输入 hash、输出正文均三臂一致；5 项合成质量检查各自通过，但旧脚本没有保存这些项的输入 hash，因此不声称其输入字节经过比对。4 项开放式检查的输入、正文与 reasoning hash 均复核一致。它们不是完整 MoBase 业务、长上下文、并发或循环风险验收。严格复算见 [新 A/B/A 分析](evidence/sm120-next/takeover/ffix-results.md)及 [逐请求计算](evidence/sm120-next/takeover/ffix-v1/analysis-independent.json)。

**结论：保留为待业务验证的 prefill 候选，未部署。** 已修正可复现的未初始化输出缺陷，本次无诊断整模也保持所测输出和接受工作量一致；但旧分支在带诊断运行中同样恢复一致，不能宣称历史 B 分叉的唯一根因已确定。

## GEMM 实测与取舍

两个新 GEMM 均已通过实际 CUDA 编译、数值及改输入 Graph 验收。首轮 M16160 的同配对加速比为 cuBLAS rowwise 1.0619、PR 派生 1.0487、CUTLASS MXFP8 0.9689；未到 1.10。PR 与 cuBLAS rowwise 在两种 M 的全矩阵 BF16 比较均逐位一致，但不表示它们与原 block128 模型等价。第一轮有频率相位变化，第二轮在计时前增加至少 15 秒四臂轮转预热，单独保留两轮结果。见 [首轮复算](evidence/sm120-next/b-mxfp8-recovery/results-review.md)。

第二轮从相同随机 seed、输入和库版本开始，两个尺寸各 31 组同组随机顺序。下表加速比取**每组官方耗时 / 候选耗时的中位数**，不是两列中位数相除；大于 1 表示候选快。

| M | 实现 | warm 中位耗时 | warm 配对加速比 | 扰缓存配对加速比 |
| ---: | --- | ---: | ---: | ---: |
| 2,741 | 官方 block128 | 1.61398 ms | 1.0000 | 1.0000 |
| 2,741 | cuBLAS rowwise | 1.55312 ms | 1.0374 | 1.0377 |
| 2,741 | CUTLASS MXFP8 | 1.69757 ms | **0.9503** | **0.9471** |
| 2,741 | PR5521 派生 rowwise | 1.55466 ms | 1.0377 | 1.0361 |
| 16,160 | 官方 block128 | 9.51210 ms | 1.0000 | 1.0000 |
| 16,160 | cuBLAS rowwise | 8.91405 ms | **1.0670** | **1.0667** |
| 16,160 | CUTLASS MXFP8 | 9.77795 ms | **0.9726** | **0.9724** |
| 16,160 | PR5521 派生 rowwise | 9.07670 ms | 1.0478 | 1.0484 |

持续预热后两尺寸 warm 官方吞吐均约 **605–606 TFLOP/s**，实际时钟和功耗随负载变化，未锁频。初次短窗口更高的绝对吞吐不能直接外推持续运行；两轮均没有任一候选达到 1.10 门槛。增加预热并不意味着各样本频率完全一致，NVML 只是观测记录，不是周期归一化。

**这两个新实现没有给出足够收益支持换 FP8 缩放格式。** 保留原 block128 权重；不开展模型重校准、不引入 MXFP8/rowwise 权重变更。这个结论限定于本轮实际 tile、两个 prefill 尺寸与计时边界，不宣布所有 GEMM 方向已无空间。第二轮逐项数值、Graph、功耗及原始样本见 [稳态补测分析](evidence/sm120-next/b-mxfp8-recovery/steady-results-review.md)和 [结果 JSON](evidence/sm120-next/b-mxfp8-recovery/steady-summary.json)。

后续在用户现有 FP8 范围内，价值排序是：先对保留的 F1/F2 小幅 prefill 候选补真实业务与更长运行验证；decode 则以已有 draft 词表实验为线索，保持目标模型完整 BF16 head，另做接受率、完整回答及 MoBase 工具链验证。D5/D6 等历史计划没有在此次恢复中执行，也没有将合成数字作为上线承诺。

## 连接与服务状态

按用户要求，连接固定走本机代理，只建立一个持久 SSH 主连接，后续命令和传输复用。子命令设置 `ProxyCommand=false`，主连接不可用时失败，不自动新建 TCP 连接。本机新增一个绑定已有代理组的临时 loopback SOCKS listener，未改持久配置或全局选路。最终从代理控制接口复核：目标地址仅一条连接、实际链路为既有 `PROXY` 组，SSH master 已持续约 30 分钟。见 [连接策略](evidence/sm120-next/takeover/connection-policy.json)和 [实时连接记录](PUBLICATION.md#publication-boundaries)。

最终两次真实 embedding 请求间隔 **37.34 秒**，均返回 200、1,024 维有限向量；容器 ID、镜像、命令、启动时间与 restart count 均与初始记录一致。27B、9B、ASR 继续停止。所有本轮实验容器均已清理，GPU 仅剩原 embedding 的 EngineCore（约 5,604 MiB）。见 [第一次检查](evidence/sm120-next/takeover/services-final-1.json)、[第二次检查](evidence/sm120-next/takeover/services-final-2.json)、[最终清单](evidence/sm120-next/takeover/inventory-final.json)。

## 复查与复现

本地不需 GPU 即可复算整模数据：

```sh
python3 _dev/docs/evidence/sm120-next/takeover/analyze_ffix_aba.py \
  --root _dev/docs/evidence/sm120-next/takeover/ffix-v1 \
  --output /tmp/ffix-recomputed.json
```

GEMM 构建使用固定镜像与源 SHA，两个 builder、完整编译命令/日志和 generated CUDA 均保留在 [新 GEMM 实验目录](evidence/sm120-next/b-mxfp8-recovery/) 与 [构建记录](evidence/sm120-next/takeover/b-mxfp8-recovery/)。可执行 `.so` 不进仓库。先按 [研究与构建说明](evidence/sm120-next/b-mxfp8-recovery/research.md)重建，再按 [补测参数](evidence/sm120-next/b-mxfp8-recovery/steady-diff.md)运行；输入量化与权重排列仍在计时外。

服务器上的已冻结控制器位于 `/home/llm/.local/share/qwen-sm120-takeover`，模型控制器沿用 `/home/llm/.local/share/qwen-fp8-fusion`。重新测量必须用新的 `ffix-vN` / `gemm-vN` 输出目录，单 GPU 串行，先检查 embedding 和已有实验；控制器只清理自己创建的确切容器名。复用现有 SSH ControlPath，禁止自动回退新连接。此次恢复没有修改生产模型配置或启动原三个服务。
