# 社区实测：最大的优化空间在哪里

核查日期：2026-10-13。以下是社区作者公开的结果及代码核查，**不是本机新增实测**。本次仅补充研究和文档，没有改动服务器。

## 1. 判断：优先验证更强的投机解码

对这台单卡、低并发、dense 27B 的 decode，当前最值得验证的高收益方向是 **外部并行草稿模型 DFlash / DFlash2**，其次是 **NVFP4 与实际 linear 后端**。单换3.8或微调一个FP8 GEMM的优先级下降。这个排序是综合社区对照和本机baseline作出的判断，尚不是本机性能结论。

本机3.6-FP8无MTP约50.5 tok/s，MTP2约90，说明“一次target校验兑现多个token”已经有效。下一步重点是提高每轮有效产出，同时控制草稿、校验和GDN状态管理的耗时。接受率百分比不能单独决定胜负：草稿更长时，接受比例可能下降，但每轮接受的token数仍可上升。

## 2. 最接近本机的公开对照

作者 gisenberg 在 **RTX PRO 6000 Blackwell 96GB、Qwen3.6-27B-FP8** 上公开了以下结果；已逐项读取其固定commit下的summary JSON。

| 方案 | 作者报告的平均输出速率 | 相对同组MTP2 |
| --- | ---: | ---: |
| 不使用投机解码 | 48.3 tok/s | 0.52× |
| 原生MTP2 | 93.3 tok/s | 1.00× |
| DFlash，k=7 | 170.8 tok/s | 1.83× |
| DFlash，k=11 | 185.8 tok/s | 1.99× |
| DFlash，k=15 | 197.5 tok/s | 2.12× |

[作者报告](https://github.com/gisenberg/local-model-eval/blob/0ac6602be3477faaf82194a4b7599dc853d64a13/results/QWEN36_RTXPRO6000.md)；[公开结果及来源指纹](evidence/community-source-check.json)。作者正文中“k7比MTP2快2.5倍”的说法与表格不符，此处按170.8÷93.3计算。

需要保留的测试条件：4道代码题，每题3次，开启思考、temperature=0.3、输出上限15K，使用旧的定制vLLM 0.20.1开发构建、BF16 KV和开启的prefix cache。**这是短输入、长代码输出的线索，不能直接外推MoBase长历史、工具调用或多人并发。** [编排脚本](https://github.com/gisenberg/local-model-eval/blob/0ac6602be3477faaf82194a4b7599dc853d64a13/tools/qwen36_27b_mtp_sweep.py)

其计时脚本用非流式POST的`completion_tokens / 完整请求耗时`，包括prefill；本机此前采用SSE首末输出估计decode，二者口径不同。固定题重复且未显式清空前缀缓存。质量分数使用模型自己生成的测试，best-of-3满分也不足以证明业务质量不下降。[计时及评分脚本](https://github.com/gisenberg/local-model-eval/blob/0ac6602be3477faaf82194a4b7599dc853d64a13/tools/nvfp4_qwen36_27b_bench.py)

另一个RTX PRO 6000 Max-Q用户的Spec-Bench帖子也报告DFlash在低并发下有较大收益，但其约2.5–3.3倍是相对**各框架不启用投机解码**的基线，不能再乘到我们已经开启MTP2的约90 tok/s上。[作者的论坛测试](https://www.reddit.com/r/LocalLLaMA/comments/1v22hu9/benchmarked_every_specdecode_method_on_qwen3627b/)

## 3. 如何看待200+的配置分享

e-accelerate的3.8/NVFP4/DFlash2/Triton案例实际用 **RTX PRO 5000 48GB**。267.8 tok/s来自关闭思考、greedy、1024输出上限，仅一轮三道代码题；拿它与开启思考、512输出上限的EAGLE 132.8相比，不能归因成DFlash2加速2倍。同为开启思考的表内对照是159.7对132.8，约1.20倍。相同block16下，Triton相对FlashInfer的收益约8%，也远小于标题数字暗示的差距。[固定版本报告](https://github.com/e-accelerate/Qwen3.8-27B-DFlash2-Triton-Blackwell/blob/a8521a0c626d53d737a46068debbcf13ce3100f0/README.md)

脚本正确读取SSE usage，排除首token等待，但没有公开逐请求原始结果、实际输入/输出长度及缓存命中。强制block16超过草稿训练时的block8；少量题目的输出检查不够替代质量回归。[bench.py](https://github.com/e-accelerate/Qwen3.8-27B-DFlash2-Triton-Blackwell/blob/a8521a0c626d53d737a46068debbcf13ce3100f0/bench.py)。其参考的[MiaAI 96GB配方](https://github.com/MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark)也把“240+”注明为根据草稿数据估算、需要本机复测。

这些资料证明路线值得试，不能证明本机、当前四服务共存预算或长上下文下能稳定200+。配置允许262K上下文，也不等于速度是在262K输入上测得。

## 4. NVFP4和自研算子放在哪里

**NVFP4要看真正执行的kernel。** vLLM官方PR修正了SM120/121的默认选择：对适用checkpoint优先W4A4，避免落到weight-only W4A16。作者在PRO 6000 Max-Q、3.8-NVFP4、带32K缓存前缀的单请求上报告TTFT 1455→926ms、输出74.81→92.63 tok/s；这说明后端选择能显著影响结果，不能把“权重变4bit”直接当成速度保证。该修复已合并；后续以0.31为起点检查实际dispatch。[vLLM PR #55170](https://github.com/vllm-project/vllm/pull/55170)

**自研small-M kernel需要整模验证。** SGLang一个尚未合并的PRO 6000测试PR报告，FP8 GEMM加权耗时减少约6.6–8.9%，但128输入/512输出、C1整模从103.34降至102.08 tok/s；部分并发场景才有收益。该测试本身仍有抖动和正确性待验收项，足以提醒我们不能只看kernel微基准。[SGLang PR #39978](https://github.com/sgl-project/sglang/pull/39978)

若profile支持自研，先看dense MLP gate/up与down的small-M权重读取，再看GDN在多token校验时的状态更新、保存和回退；具体形状见[算子清单](05-optimization-plan.md#4-算子研发清单)。

prefill另行评估：外部草稿主要针对decode。本机框架升级后约19K冷输入TTFT只从约2.40降至2.18秒。长输入应继续测GDN prefill、量化linear及chunk调度；重复上下文应核实真实前缀缓存命中。MoBase的首次正文还受串行工具步骤、思考与最终generation影响。

## 5. 下一轮实验顺序

1. **保留现有FP8/MTP2对照，先验证外部DFlash。** 3.6使用匹配的草稿权重；同时用MTP3/4做低成本对照。先确认固定vLLM版本的hybrid模型、KV dtype、图模式兼容性；若需更换KV/backend，补相同配置的MTP对照，不能把差异全记给草稿算法。
2. **再单独加入NVFP4。** 使用对应模型和草稿，核对W4A4/W4A16及混合FP8层；测量性能与业务质量。模型下载仍只走ModelScope，存入原模型目录；公开HF链接仅作资料来源，不代表另行下载。
3. **3.8路线比较DFlash2、DSpark与MTP。** DFlash与DFlash2不是同一套权重；不要强行互换。先采用草稿模型支持的block长度，再研究扩大block。vLLM 0.30已加入DFlash2/DSpark相关支持；SGLang也有官方配方，不能说只有某个私人fork能用。[vLLM 0.30发布](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)、[SGLang配方](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
4. **胜出配置再profile和做MoBase回归。** 固定输入、思考模式、采样和输出预算，记录实际token数、finish_reason、冷/热缓存、TTFT、decode、接受token/轮、显存与并发。短代码、中文检索、长历史和工具调用分开统计。embedding继续保留，ASR容量按用户要求规划；不能照抄社区0.88/0.92显存占用到四服务共存环境。

算子源码基线仍为vLLM v0.31.0。此次变化是把“更强投机解码”提前为主要实验方向，并未建议照搬社区旧构建、关闭思考换跑分，或承诺本机达到特定tps。


## 后续实测

本机已按上述优先级完成Qwen3.6-FP8的MTP2/3/4与DFlash7/15对照，另测FP8 KV/FlashInfer组合。普通输出以MTP4更好，较长思考以DFlash15更好；没有测出所有场景200 tok/s，也没有发现投机解码改善冷输入prefill。条件、口径和原始证据见[追加实验报告](07-specdecode-experiment.md)。
