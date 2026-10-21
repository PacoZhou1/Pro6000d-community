# Qwen 27B 推理优化调研

**公开资料入口（2026-10-21）：**[结果、复现命令与发布边界](PUBLICATION.md)。以下“当前”服务状态均为 2026-10-15 的历史检查，本次资料发布没有连接或操作线上服务。历史各轮的停服/恢复指令不是当前操作要求。

当前服务已恢复：27B在原8000换成新FP8组合镜像，9B/embedding/ASR仍用原8001/8002/8003，密钥均保持原值。四项真实请求通过，embedding未停；生产27B用10GiB cache保留262K上下文。[生产切换结果](20-production-switch.md)。以下为前轮测速及恢复历史。

最新：已固定组合镜像 `pro6000d/qwen36-fp8:20261015-r1` 并保留运行。本轮普通118.17、思考133.64 token/s，19K首token2.164秒；对最初未优化的v0.31/MTP2，普通decode参考约+25.4%，prefill基本持平；相对首次v0.31/MTP2（BF16 KV/FA2）思考测试，思考整体配置参考约+43.8%。思考未复现历史64K单项148.10，差异如实保留。16项自然结束/格式检查通过，embedding正常，原其他服务继续停止。详见[组合镜像实测与部署](19-fp8-combined-release.md)。

以下为前轮记录。

更新：2026-10-15，America/Los_Angeles。已接手并补齐 Claude 中断的 FP8 实验：修复 F1 预热分支未初始化输出，F1/F2 新整模 A/B/A 的长 prefill TTFT 配对改善 **1.49%**，decode 基本不变，所测正文和 MTP 接受计数一致。实际构建 CUTLASS MXFP8 与 PR5521 派生 rowwise GEMM，预热复测均未达 **1.10×** 门槛，保留原 FP8 权重。候选未部署；最终两次 embedding 真实请求通过，原 **27B、9B、ASR 保持停止**，临时实验容器已清理。详见 [接手结果与证据](18-claude-fp8-takeover.md)。连接现通过本机代理复用单一 SSH 长连接。

前轮完成 **SM120 M64主循环warp组织、官方完整链与整模A/B/A**。2×4新布局在M16160同调度微基准比旧融合核快约2.6%，但四种尺寸仍慢于官方；整模TTFT **2.309秒**，前后官方基线 **2.243／2.257秒**，配对退化 **2.63%**，**不采用**。NCU确认shared load wavefront减少约31%，但没有减少DRAM/L2流量或提高融合核Tensor活跃率；采样频率不同，不能把时间差解释成已证明的周期减少。见[主循环实验与整模结果](15-sm120-m64-mainloop.md)。最终两次实际embedding请求通过，容器身份、启动时间和重启计数不变；原 **27B、9B、ASR保持停止，等用户明确要求再启动**，临时实验容器已清理。

前轮已完成SM120调度及配对epilogue实验：M16160融合链比原v6快约9.2%，仍慢于官方约8.8%；官方同库调度整模配对TTFT仅改善0.14%，小于基线漂移，两种方案均未采用。见[调度实验](14-sm120-scheduler-epilogue.md)。

前一轮已修复实验模型启动，完成 **FP8 GEMM → SiLU/量化 epilogue 融合**的数值检查、真实模型调用追踪、NCU 及两组无 profiler A1/B/A2。v6 长输入 TTFT 为 **2.415 秒**，前后基线 **2.251 / 2.261 秒**；12 个配对请求的退化中位数 **7.09%**，早期 v4 退化 **22.75%**，**不接入生产**。当轮 11 个临时实验容器已清理。见[融合整模报告](12-fp8-gemm-epilogue.md)。

上一轮已有 SiLU + block FP8 量化核的 V2 线程组织改动，在当时单次 A/B/A 中的 prefill 配对 TTFT 改善中位为 1.4144%，只保留为候选；本轮以官方 GEMM + V2 量化核为对照。历史服务恢复记录仅表示当轮结束状态，不代表当前正在运行。见[V2 实验](11-fp8-fusion.md)。

正常 graph profiling 中，FP8 GEMM 名称家族占 prefill / MTP4 decode / DFlash15 思考 decode 的 GPU kernel 累计时间约 **68.5% / 62.0% / 64.1%**。后续 eager 归因定位到 FP8 gate/up、down 和 BF16 `lm_head` 的真实尺寸；`lm_head` 占该 eager decode 捕获的 BF16 GEMM 时间约 **94.26%**。这两个统计口径不能互换。

定向实验中，prefill tile 与 Split-N 均变慢；小 M down 的 Stream-K 在热缓存微基准中约快 27%，但真实整模普通 decode 为 **104.91 tok/s**，低于前后两次固定基线 **108.99 / 107.34 tok/s**。三臂生成内容也有差异，因此不作纯算子因果归因；当前没有可采纳的生产加速补丁。当轮已证明候选实际参与 CUDA Graph 重放；当轮结束曾恢复全部原服务，embedding 未停。见[定向 GEMM 结果](09-targeted-gemm.md)；上一轮结果见[GPU profiling 报告](08-gpu-profiling.md)。

## 建议

**当前范围更新：**用户已明确只推进 FP8 优化，保留 BF16 `lm_head`；历史 NVFP4 思维链循环定位、FP4 和 head-only FP8 均不在本轮范围。本轮已按[SM120 硬件路线](13-sm120-hardware-roadmap.md)完成受控实现，最新结果见[Claude FP8 实验接手](18-claude-fp8-takeover.md)；前序工作见[融合整模实验](12-fp8-gemm-epilogue.md)、[V2 量化核实验](11-fp8-fusion.md)。下文关于 NVFP4/head 的建议为历史记录，不作为本轮前置条件；旧固定 token 思考速率也不代表正常完成。

**算子开发以官方 vLLM `v0.31.0 / db9527a46873454610df6dbedf79a36d6bf1a7f6` 为基线，保留MTP2对照。单纯升级框架或把3.6换成3.8，没有在本机测出130/200 tok/s。** 补查社区并完成追加受控实验后，普通输出优先以MTP4作候选，较长思考输出以DFlash15作候选；后续仅在 FP8 范围内依据 profile 选择优化方向。[官方release](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)、[社区实测与口径核查](06-community-priorities.md)、[本机MTP/DFlash对照](07-specdecode-experiment.md)

服务器实际是 **1×RTX PRO 6000 Blackwell Workstation，96GB、600W、SM120**；本仓库历史项目为4×RTX6000D、84GB/卡、350W/卡。GLM/MiMo的速度不能直接作为单卡dense Qwen27B的目标保证。

## 本机实测

并发1、冷前缀、合成固定user语料，每格n=3中位数；embedding始终运行，多数采样窗口未观察到embedding活动，但两个窗口记录到短embedding请求，见下述限制。decode是由真实usage tokens和SSE文本时间估算，未把chunk数当token数。

| 模型 / 框架 | MTP | 约4.8K输入 / 1024输出 decode | 约19K输入 / 1输出 TTFT |
|---|---|---:|---:|
| 3.6-FP8 / 0.27.1，首轮及下载结束后复测 | 2 | **90.7 / 89.7 tok/s** | **2.378 / 2.403秒** |
| 3.6-FP8 / 0.27.1 | 关闭 | 50.5 tok/s | 2.236秒 |
| 3.6-FP8 / 0.31.0 | 2 | **94.2 tok/s** | **2.179秒** |
| 3.8-FP8 / 0.31.0 | 2 | **93.4 tok/s** | **2.190秒** |

0.31的长输入首token和长输出有所改善，但短约0.7K/256输出decode从旧版98.4/100.2降到92.6 tok/s，不能概括为所有场景更快。3.8与3.6在同框架上接近，换代应主要考察业务质量、思考长度及工具成功率。

新版日志确认GDN prefill由Triton/FLA转为FlashInfer，attention走SM120 XQA，并捕获FULL decode图；这是整个运行栈升级，不能把全部差异归给某一个kernel。无MTP虽然也捕获FULL图，长输出仍明显更慢，当前应保留MTP2。

0.31的19K输入第3次窗口记录到170-token embedding请求；旧版回测1024输出第1次窗口记录到57-token embedding请求。表格保留原n=3；剔除后各余n=2，分别为TTFT 2.167秒、decode 89.32 tok/s，未改变本轮结论方向，但不能称完全无干扰。

这些是约11–20秒的固定1024输出请求，未做30秒持续decode、并发吞吐、完整MoBase质量验收或kernel profile。不同模型/后端可能生成不同文本；系统UUID前缀不同。完整口径、逐请求JSONL、复测和日志见[服务器基线报告](01-server-baseline.md)。

## 追加投机解码实验

同一Qwen3.6-FP8、vLLM0.31/V2、约4.8K输入/1024输出，C1，每格n=3中位数：

| KV / attention | 方法 | 普通输出 decode | 思考 token decode |
| --- | --- | ---: | ---: |
| BF16 / FlashAttention2 | MTP2 | 86.44 tok/s | 92.93 tok/s |
| BF16 / FlashAttention2 | MTP4 | 104.65 tok/s | 109.80 tok/s |
| BF16 / FlashAttention2 | DFlash15 | 89.60 tok/s | 135.07 tok/s |
| FP8 / FlashInfer | MTP4 | **107.73 tok/s** | 124.60 tok/s |
| FP8 / FlashInfer | DFlash15 | 94.82 tok/s | **138.34 tok/s** |

同BF16栈，MTP4普通输出约+21%，DFlash15思考约+45%；同FP8/FlashInfer栈，DFlash15相对MTP4思考约+11%，普通输出反而约−12%。**思考组1024输出全部是reasoning，没有最终正文，不能当作完整任务提速。** 19K输入TTFT仍约2.2秒，prefill需另查profile。

这轮显存预算0.70，输入跨臂完全一致；与上表旧实验的预算、运行器、KV/后端等条件不同，应按各自同栈比较。多token输出hash存在差异，合成质量检查通过不等于业务质量无损。原四服务预算下的候选性能、262K容量、并发和MoBase整链路尚未验证。全部变体、正常结束检查、接受率与原始记录见[追加实验报告](07-specdecode-experiment.md)。

## 优化优先级

下列为早期调研优先级记录；涉及 NVFP4、head 低精度的内容已被用户后续的 FP8-only 范围覆盖。最新实验结论和继续开发的切入点见[Claude FP8 实验接手](18-claude-fp8-takeover.md)。

1. **处理资源与调用链。** ASR原先因显存不足反复重启，累计超过4万次；按用户追加要求，已通过缩减27B cache恢复。MoBase主Agent开思考、调用工具且多步串行，最终另做generation；模型直连tps不是用户首次正文延迟。
2. **以0.31做开发和后续实验。** 保留现有FP8/MTP2对照，MTP3/4已测；下一步优先核验MTP4的共存预算与真实业务，再看GDN与linear后端。不要整套覆盖本仓库GLM/MiMo的旧fork文件。
3. **DFlash按思考负载验证，再单独比较NVFP4。** 同卡3.6-FP8社区对照报告MTP2约93.3、DFlash约171–198 tok/s，但使用旧定制栈、长代码输出和不同计时口径；本机追加实验已测到思考约135–138 tok/s，普通输出仅约90–95 tok/s，未复现普遍200 tok/s。NVFP4是另一个减少权重流量的候选，需检查混合精度层的实际dispatch与质量；不能把两个改动混在一起归因。3.8路线另测匹配的DFlash2/DSpark。[具体依据](06-community-priorities.md)
4. **自研以 profile 为依据。** FP8 真实尺寸与 BF16 模块归因已完成。代表性合成大 M FP8 的 Tensor pipe active 比率约 88%，小 M gate/up 和 BF16 head 的 DRAM 吞吐较高；下一步更值得验证减少权重读取、head 低精度或更低比特权重。整模硬件计数器因内存不足未取到，合成计数器不能直接证明服务瓶颈。已测三种候选均不采用，GDN 优先级仍较低。见[定向实现、实测及取舍](09-targeted-gemm.md)。

0.31的三个合成API冒烟（算术、JSON、tool_call）均通过，3.8也通过；FP8 scaling日志仍提示部分scale=1.0，长上下文和业务质量必须另验收后再正式切换。

## 模型来源与服务状态

按用户要求，模型下载统一走 **ModelScope**，沿用 `/home/llm/models`。本轮复用了现有3.6和3.8；3.8的66个权重分片全部与ModelScope官方SHA256匹配，没有重复下载。vLLM0.31运行镜像已获取并锁定官方一致的amd64 digest。

以下为此前已完成的 cache 调整和恢复记录；当前服务状态以上方最新说明为准。按用户当时的追加要求，生产27B的gpu-memory-utilization从0.50降至0.45，KV池约15.92→11.15GiB，仍保留262K上下文；原0.50容器停止保留，可回退。ASR已恢复，5秒合成音频转写200、约0.9秒完成。27B/9B/embedding/ASR四项health均200，embedding全程未停。

追加投机解码实验结束后也已恢复同一组原容器。27B恢复首次因cache显存测算失败自动重试一次，之后与另外三项在32.65分钟复查中均未新增重启；该历史记录见[投机解码实验恢复记录](07-specdecode-experiment.md#服务恢复与启动问题)。10月14日此前的定向 GEMM 实验后，四项实际功能检查通过，约309.58秒复查无新增重启；embedding启动时间与原重启计数保持不变，203次周期健康采样全为200。见[当轮恢复证据](09-targeted-gemm.md#服务恢复与保留产物)。

第一张表的性能是在改cache前测得；调整后的四服务共存状态只做了功能验证，未冒充新性能基线。生产仍为3.6/0.27.1，仅改变显存预算，没有修改MoBase代码、转换权重或安装实验算子补丁。cache调整与回退方法见[基线报告第11节](01-server-baseline.md#11-用户追加缩减27b-cache恢复asr)。

## 文档与证据

1. [服务器配置、基线、对照与最终状态](01-server-baseline.md)
2. [当前客户项目哪些能借鉴、哪些不能照搬](02-project-audit.md)
3. [上游项目、Qwen3.8、精确版本与一手来源](03-upstream-research.md)
4. [MoBase调用链、思考/工具/流式与验收](04-mobase-integration.md)
5. [实验矩阵、算子开发与回退](05-optimization-plan.md)
6. [论坛和社区实测：最大优化空间与下一轮优先级](06-community-priorities.md)
7. [Qwen3.6：MTP2/3/4与DFlash7/15受控实验](07-specdecode-experiment.md)
8. [GPU profiling：prefill / decode 热点与 FP8 GEMM 实验](08-gpu-profiling.md)
9. [定向 GEMM：真实尺寸、BF16 lm_head、硬件计数器与三种算子候选](09-targeted-gemm.md)
10. [下一轮路线：仅 FP8，先验证已有 SiLU + block 量化核的线程组织](10-next-inference-roadmap.md)
11. [FP8 SiLU + block 量化核实验：V2 数值、真实派发、API 对照与待验证项](11-fp8-fusion.md)
12. [FP8 GEMM epilogue：启动修复、v4/v6 融合、NCU 与整模 A/B/A](12-fp8-gemm-epilogue.md)
13. [SM120 硬件路线：资源约束、长 prefill DRAM 放大与受控优化设计](13-sm120-hardware-roadmap.md)
14. [SM120 调度与 epilogue 对照：硬件计数、实际模型派发与 A/B/A](14-sm120-scheduler-epilogue.md)
15. [SM120 M64主循环：warp布局、官方完整链、NCU与整模A/B/A](15-sm120-m64-mainloop.md)
16. [SM120 下一轮计划（历史计划，含尚未采用的路线）](16-sm120-next-round-plan.md)
17. [剩余实验计划（历史计划）](17-remaining-experiments-plan.md)
18. [Claude FP8 实验接手：F1/F2 修复与整模、新 MXFP8/rowwise GEMM 实测](18-claude-fp8-takeover.md)
19. [最终 FP8 组合镜像、实测和配置差异](19-fp8-combined-release.md)
20. [四服务恢复与生产 27B 切换记录](20-production-switch.md)

`evidence/`保存脱敏配置、原始JSONL、模型指纹和复测脚本；不含登录密码、API key或业务请求正文。子agent使用Sol完成项目、MoBase、上游和独立审查；主agent核对现场、测量与最终结论。
