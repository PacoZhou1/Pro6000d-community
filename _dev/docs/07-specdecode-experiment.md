# Qwen3.6 FP8：MTP与DFlash本机对照

日期：2026-10-13。本轮依据用户“继续下一步”的要求，执行[社区调研](06-community-priorities.md)的首个优先实验。已完成7种配置、1次MTP2末尾复测，并恢复原四服务。

## 范围与起始状态

单卡RTX PRO 6000 Blackwell Workstation 96GB，保留embedding运行；仅在测试期间暂停27B、9B、ASR。起始四服务health均200且没有排队/运行请求。生产27B仍为0.27.1/3.6-FP8/MTP2，显存预算0.45；本轮不改变原容器配置。证据：[initial-state.json](evidence/specdecode/initial-state.json)。

本轮先测FP8下的MTP2/3/4与DFlash，不同时引入NVFP4、3.8或自研kernel。临时服务绑定服务器localhost:18000；原容器停止保留。

## 固定运行栈与公平对照

- vLLM v0.31.0，镜像ID `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`。
- 现有3.6-FP8权重只读挂载；TP1、max-model-len262144、max-num-seqs16。
- 首轮统一`VLLM_USE_V2_MODEL_RUNNER=1`、`--kv-cache-dtype auto`、`--attention-backend FLASH_ATTN`、max-num-batched-tokens32768、显存预算0.70。
- 保留prefix cache，但每个label以固定且跨实验臂相同的首前缀避免同一实验臂内缓存命中；逐请求记录缓存与embedding活动。

BF16 KV/FlashAttention/预算等与上一轮FP8 KV/FlashInfer配置不同，因此以本轮同栈、同payload的MTP与DFlash作为方法对照；不能直接拿旧约90–94 tok/s作单变量基线。不同方法的多token输出存在差异，测得的是各配置完成固定输出请求的速度，不能当作相同逐token轨迹的纯kernel耗时比较。另补FP8 KV/FlashInfer的MTP4控制组，以免把后端和KV变化全部归因给DFlash。

## 草稿来源与兼容性

草稿：[ModelScope z-lab/Qwen3.6-27B-DFlash](https://www.modelscope.cn/models/z-lab/Qwen3.6-27B-DFlash/summary)，固定revision `fd20fcba71887d77ff81735911bb5935a9af5aa5`，下载至原模型根目录下的`Qwen3.6-27B-DFlash`。公开元数据的模型权重3,460,432,504字节，SHA256 `e0c050b34798d32728a164d2c3f1681746ff85c11945701b0205b654e2f1fdbe`；config SHA256 `fcdec9ee2da902d24e69ba3fc666e50a0aa723147689ffc900f425db0381bc54`。本地文件校验另行留存；全部下载请求从ModelScope发起。

草稿7个文件均已通过下载后SHA256校验，见[draft-verification.json](evidence/specdecode/draft-verification.json)。现有目标3.6-FP8也已按官方ModelScope固定revision逐文件校验：79个文件匹配，仅缺少不影响推理的.gitattributes；66个权重分片全部匹配。见[官方清单](evidence/specdecode/target-modelscope-manifest.json)、[本机核验](evidence/specdecode/target-file-verification.json)。

该草稿为BF16、5层，四层滑窗和一层全注意力、block_size16，匹配Qwen3.6-27B；DFlash k15为首选验证点。它与3.8的DFlash2不同，不能互换。

vLLM 0.31固定源码明确混合滑窗DFlash需要V2 runner；V2同时有MTP与DFlash路径，兼容性清单未排除Qwen3.6 hybrid/GDN。源码条件满足仍需实际启动验证。[草稿实现](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/models/qwen3_dflash.py)、[V2配置](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/vllm.py)、[官方草稿说明](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)

## 测量方法

使用[本轮客户端](evidence/specdecode/benchmark_client.py)，延续上一轮固定合成user语料和SSE usage计数，每个请求保存完整计时与服务端指标差值；不是用SSE chunk数充当token数。C1，每个正式格n=3，每类形状预热排除；短/中输入256输出、长输入1输出、约4.8K输入1024输出，并增加开启思考的约4.8K/1024对照。首UUID由label确定，同label跨实验臂的完整payload和user corpus hash均应相同；每臂新建服务，不复用其他臂缓存。

decode仍是`(completion_tokens-1)/(末次文本时间-首次文本时间)`估计。投机解码首次chunk可能包含多个token，需保留这一偏差说明；同时记录端到端输出速率、TTFT和服务端指标供交叉判断。固定输出使用ignore_eos，因此性能用例不替代正常结束、工具JSON或业务质量验收。

启动、下载和捕获脚本保存在[evidence/specdecode](evidence/specdecode)。凭据只在服务器内存中读取，不写入仓库；运行时日志仅保留脱敏诊断。

## BF16 KV / FlashAttention2对照

每格n=3中位数；8次运行每次22个完整请求（含预热/观察），共176个请求，跨臂同label的payload hash全部一致。每次15个正式采样，共120个，均为完整固定输出、独立指标窗口、0 prefix-cache命中，且未观察到embedding计数活动；仍未锁定GPU时钟或功率，也未做随机化交叉顺序。

| BF16 KV / FlashAttention2 / V2 | 不思考：约4.8K/1024 decode | 思考：约4.8K/1024 decode | 约19K/1 TTFT |
| --- | ---: | ---: | ---: |
| MTP2 | 86.44 tok/s | 92.93 tok/s | 2.213秒 |
| MTP3 | 94.10 tok/s | 102.24 tok/s | 2.228秒 |
| MTP4 | 104.65 tok/s | 109.80 tok/s | 2.235秒 |
| DFlash15 | 89.60 tok/s | 135.07 tok/s | 2.203秒 |
| DFlash7 | 93.16 tok/s | 119.78 tok/s | 2.206秒 |

MTP4相对MTP2的不思考decode提升21.1%，思考提升18.1%。DFlash15相对MTP2在此组思考请求提升约45.3%，不思考仅约3.6%。**1024输出的思考组全部仍是reasoning token，未生成最终正文；这不是整项任务完成时间提升45.3%。**其每轮接受token加bonus的中位数分别为3.93和2.62，MTP2分别为2.58和2.39；说明收益随负载变化，不能直接套用社区长代码题的200 tok/s。[可重算汇总](evidence/specdecode/summary.json)、[MTP2原始请求](evidence/specdecode/paired-mtp2-bf16.jsonl)、[DFlash15原始请求](evidence/specdecode/paired-dflash15-bf16.jsonl)

DFlash15通过3项聊天/JSON/工具调用冒烟和5项合成检查。44,518-token输入检索首次6.179秒，重复请求命中43,200 token缓存、0.515秒，两次答案均正确。这只是有限用例的真实API验证，不是MoBase业务质量验收。[冒烟](evidence/specdecode/smoke-dflash15-bf16.jsonl)、[合成检查](evidence/specdecode/quality-dflash15-bf16.jsonl)

MTP3与MTP2的22个完整输出hash全部相同；MTP4、DFlash7/15各自与MTP2仅5/22相同（3个正式单token prefill和2个预热），正式256/1024-token解码请求为0/12相同。DFlash15 FP8/FlashInfer为4/22，正式长输出同样0/12相同。因此差异不只出现在DFlash，不能把这项计数当作质量通过率。虽然输入与greedy采样相同，不同校验形状/数值路径可能产生差异，目前没有完成逐token等价性诊断，**不得宣称更改草稿长度或DFlash与基线逐字等价、质量无损**。vLLM固定版本也说明数值精度限制会造成此类差异，但这不能证明本机的具体根因。[官方说明](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/features/speculative_decoding/README.md#lossless-guarantees-of-speculative-decoding)

MTP3/4、DFlash7/15均通过3项API冒烟和5项合成检查。DFlash7另通过4次无JSON grammar、无ignore_eos、正常stop的开放式中文与思考请求，两种题各重复两次；同配置重复输出一致。检查内容为合成事实摘要与小算术，不能替代MoBase的真实多轮工具、检索和视觉回归。

启动日志另有“未识别草稿KV组”warning。固定源码下该告警不会关闭DFlash，但会使prefix缓存保守地按投机组处理所有KV组，影响尾块命中边界。实际draft/accepted计数增长，重复前缀用例成功，仍保留更长上下文、多轮缓存与并发回归要求。[本机日志](evidence/specdecode/runtime-dflash15-bf16.json)、[vLLM缓存回退规则](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/kv_cache_coordinator.py#L104)


## 末尾复测和正常结束验证

全部候选测完后重启同配置MTP2，再测22个请求：不思考86.33 tok/s、思考92.89 tok/s，对比首轮86.44/92.93仅约−0.13%/−0.04%；19K TTFT 2.246秒，对比2.213约+1.50%。payload和输出hash均22/22相同，未出现明显decode基线漂移。复测运行时直接记录了V2环境变量。SSH复用连接曾短暂延迟，但服务器新连接可用，远端loopback计时和22个完整响应均保留；不把SSH等待时间作为模型响应耗时。[复测](evidence/specdecode/paired-mtp2-bf16-repeat.jsonl)、[运行时](evidence/specdecode/runtime-mtp2-bf16-repeat.json)

MTP2复测、DFlash7 BF16、MTP4 FP8/FI、DFlash15 FP8/FI，各做2种无JSON约束、无ignore_eos的题，每种重复两次，共16次均正常stop且事实检查通过。两种题的最终正文跨配置完全一致；思考题的reasoning相对MTP2分别从第819/786/391个字符（0-based）开始不同。完整合成输出保存，可复核第一处字符差异；没有记录逐token logits，所以这不是逐token数值等价性证明。[核查汇总](evidence/specdecode/validation-summary.json)、[开放式验证脚本](evidence/specdecode/open_output_probe.py)

各次运行共24个聊天/JSON/工具冒烟请求通过；除初次MTP2外，其余7次运行另做35个合成质量请求全部通过，初次MTP2的对应质量检查由末尾复测补足。上述开放式检查只覆盖表述和简单算术；DFlash15 BF16和MTP3/4 BF16的开放式请求没有单独补跑，不能宣称所有配置均通过所有类别。

## FP8 KV / FlashInfer组合与运行要点

同组请求和全局参数一致，DFlash另指定草稿FlashInfer。每格n=3中位数。

| FP8 KV / FlashInfer / V2 | 不思考：约4.8K/1024 decode | 思考：约4.8K/1024 decode | 约19K/1 TTFT |
| --- | ---: | ---: | ---: |
| MTP4 | **107.73 tok/s** | 124.60 tok/s | 2.259秒 |
| DFlash15 | 94.82 tok/s | **138.34 tok/s** | 2.239秒 |

DFlash15相对同栈MTP4：思考+11.0%，不思考−12.0%；不是全面更快。两者均通过3项冒烟、5项质量检查、4次正常stop请求。BF16与FP8两组同时更换了KV dtype和attention backend，不能把差值单独称为FP8 KV或FlashInfer的收益。没有测FP8/FlashInfer下的MTP2，所以不把138.34与BF16/FA2的92.93做单变量比较。

运行要点已通过本机启动验证：

- 混合滑窗草稿使用`VLLM_USE_V2_MODEL_RUNNER=1`，固定vLLM0.31镜像；目标FP8权重和草稿只读挂载。
- `--attention-backend FLASHINFER`只指定target；DFlash草稿需在`--speculative-config`内另写`"attention_backend":"FLASHINFER"`，草稿KV dtype在省略时继承全局配置。日志确认实际后端。
- block_size16草稿并不强制k15，k7已真实启动并完成相同请求；k7在本机未取得全面优势。
- 每个新实验臂使用独立容器编译缓存，避免不同draft长度共享旧编译产物；重启复测仅复用同配置缓存。
- FP8 attention日志仍有q/k/v/prob scale缺失或使用1.0的告警。有限合成检查通过，不等于长上下文精度已经充分验收。
- DFlash日志提示多模态输入走text-only draft；本轮没有验证图片任务。

复现参数和每次启动实况见[run_variant.py](evidence/specdecode/run_variant.py)、[控制脚本](evidence/specdecode/experiment_control.py)、[FP8 DFlash运行记录](evidence/specdecode/runtime-dflash15-fp8-fi.json)。脚本会停止自己创建的临时容器，生产服务须按恢复流程单独启动。


## 这轮结果如何改变优化顺序

1. **普通生成先用MTP4做候选。** 同BF16栈86.44→104.65 tok/s约+21%；FP8/FlashInfer组合107.73 tok/s。MTP3增幅较小，但这批固定请求与MTP2的输出hash全部相同，可作为更保守的实验选项，这不保证其他请求也完全等价。
2. **思考负载再用DFlash15做候选。** 同BF16栈92.93→135.07 tok/s约+45%；FP8/FlashInfer下相对已经优化的MTP4再提升约11%。必须量MoBase整次任务的首正文、总耗时、工具成功率与思考长度，不能把reasoning吞吐直接等同用户体验。
3. **prefill要另找优化点。** 本轮19K冷输入首token都在2.20–2.26秒左右，未发现投机解码改善prefill。真实缓存重复前缀能显著减少重复输入耗时，但那是cache命中收益。后续10月14日已完成GPU profile，FP8 GEMM约占prefill kernel耗时68.5%；首次B12X后端替换未带来整模收益，见[后续profiling报告](08-gpu-profiling.md)。
4. **算子开发版本仍选官方vLLM0.31.0固定提交。** 已在这张SM120卡上实测MTP与DFlash路径。以固定镜像、源码和当前FP8基线为对照，逐项改变一个因素；不整套替换客户4卡GLM/MiMo分支的模型实现。3.8和NVFP4不在本轮变量中，本轮结果不能推导它们的收益。

本轮候选在临时服务显存预算0.70下测试；原四服务共存是27B 0.45、9B 0.22、ASR 0.16、embedding 0.05。DFlash还要放入约3.46GB草稿权重及草稿状态，**没有验证四服务共存时仍能保留262K上下文或达到表中速度**。当前可交付的是可复核的实验结果与候选排序，正式MoBase调用链、视觉、并发、长上下文精度、峰值显存和稳定性仍需专门回归。


## 服务恢复与启动问题

七个本轮临时容器均已删除，ModelScope草稿权重留在原模型目录供后续复现；原27B/9B/ASR容器的ID、镜像和启动参数保持初始值。原27B仍为vLLM0.27.1、Qwen3.6-FP8、MTP2、显存预算0.45；没有把实验候选配置直接切到MoBase。[临时容器清理](evidence/specdecode/temporary-cleanup.json)

恢复过程出现一次真实失败，不能写成零故障恢复：27B与9B并行启动时，27B首次日志给出KV可用显存−2.8GiB，随后因无cache blocks退出；自动重试在9B已经运行后成功，测得KV约11.17GiB、320,232 tokens。27B重启计数由0变1；其余三项没有新增重启。并行初始化干扰显存测算是符合日志的解释，尚未做专门因果实验。恢复脚本据此改为27B健康后才启动9B，再由调用方启动ASR，避免重叠的初始化测量。保留[首次验证失败](evidence/specdecode/restoration-functional.json)、[启动诊断](evidence/specdecode/restoration-27b-diagnostics.json)、[恢复事件](evidence/specdecode/restore-startup.json)。

embedding的容器ID、2026-10-06T09:31:08.313413609Z启动时间和原重启计数1均未变。终检在15:42 UTC完成：四服务health/models均200，27B和9B实际算术回答均正常stop且正确，embedding返回1024维有限非零向量，ASR对5秒合成音频转写正确。相对15:09 UTC的首次恢复快照，32.65分钟后四服务StartedAt/RestartCount均不变，27B保持1、9B/ASR保持0、embedding保持原1；这是重启稳定性核查，不是持续压测。见[最终功能与稳定性证据](evidence/specdecode/restoration-final.json)、[验证脚本](evidence/specdecode/restore_verify.py)。新顺序恢复代码仅经静态检查，没有再次停服演练；实际这次恢复仍包含上面记录的并行启动失败。


## 复现入口

这些是本服务器专用实验脚本，需具备现有四容器、已核验的模型目录、固定0.31镜像和SSH连接。先上传本目录下客户端/控制脚本到GPU主机`/tmp/qwen-spec-experiment/`，保存原状态并在请求空闲时执行`experiment_control.py pause`；它不会停止embedding。`SSH_TARGET`、`SSH_SOCKET`、`SSH_PORT`指向当前有效SSH会话；不把密码或API key写入命令。输出stem必须使用新名字，防止覆盖旧证据。

```bash
python3 _dev/docs/evidence/specdecode/run_variant.py \
  --ssh-target "$SSH_TARGET" --socket "$SSH_SOCKET" --port "$SSH_PORT" \
  --name codex-spec-mtp4-fp8-fi-new --method mtp --draft-tokens 4 \
  --kv fp8 --attention FLASHINFER \
  --output-stem _dev/docs/evidence/specdecode/mtp4-fp8-fi-new

python3 _dev/docs/evidence/specdecode/run_variant.py \
  --ssh-target "$SSH_TARGET" --socket "$SSH_SOCKET" --port "$SSH_PORT" \
  --name codex-spec-dflash15-fp8-fi-new --method dflash --draft-tokens 15 \
  --draft-path /home/llm/models/Qwen3.6-27B-DFlash \
  --kv fp8 --attention FLASHINFER \
  --spec-extra '{"attention_backend":"FLASHINFER"}' \
  --output-stem _dev/docs/evidence/specdecode/dflash15-fp8-fi-new
```

每臂结束自动捕获脱敏runtime并停止该临时容器，任何失败也会尝试停止；必须检查停止结果。测试后在GPU主机执行`experiment_control.py restore`，等待原27B、9B依次health成功，再`docker start vllm-qwen3-asr`并等待ASR health；运行`restore_verify.py`验证四项实际功能。删除自己创建且已停止的临时容器，保留模型与证据。该恢复动作与实验runner分开，runner不自动恢复生产。

本地`python3 _dev/docs/evidence/specdecode/summarize.py`可由原始JSONL重算性能中位数、接受率和embedding窗口。质量与开放式输出原文使用合成内容，不含业务对话。
