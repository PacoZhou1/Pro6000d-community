# 当前服务器与 Qwen 27B 性能基线

日期：2026-10-13。状态：已完成初始共享GPU现场、v1两组观察、v2固定语料的v0.27.1/MTP2与关闭MTP对照，以及旧版API冒烟；v0.31同权重对照、3.8-FP8与API冒烟已完成，旧版复测与原服务恢复也已完成。

本报告由已保存、脱敏的采集数据生成，不包含服务器公网 IP、认证信息或业务问答。统计读取本地文件，没有追加线上请求。初始环境与之后实验环境必须区分：用户后来授权停止其他大模型但保留 embedding，主执行者已停止9B、ASR与27B，然后重新启动原27B；受控结果应以随后独立证据文件为准，不能回写为本节初始状态。

## 1. 证据与时间

| 文件 | 内容与时间（UTC） |
|---|---|
| [server-inventory.json](evidence/server-inventory.json) | 12:06:21，GPU、容器参数、版本、models与初始metrics |
| [runtime-diagnostics.json](evidence/runtime-diagnostics.json) | 12:06:58，容器状态、精选运行/错误日志、安装包与模型config |
| [startup-backends.json](evidence/startup-backends.json) | 当前 selected_startup_logs 为空；仅有GPU进程内存快照，不能证明实际内核backend |
| [baseline-loopback.jsonl](evidence/baseline-loopback.jsonl) | 12:09:15–12:09:48，GPU主机loopback请求与前后metrics差值，11行 |
| [controlled-startup-backends.json](evidence/controlled-startup-backends.json) | 停9B/ASR并重启原27B后的实际backend、显存与CUDA graph启动记录 |
| [baseline-controlled.jsonl](evidence/baseline-controlled.jsonl) | 原vLLM仅27B+embedding环境，另一组随机合成语料；不是matched A/B |
| [paired-v027.jsonl](evidence/paired-v027.jsonl) | 12:41:37–12:42:约47，v2固定user corpus与独立前缀，14次请求，新增3次1024输出 |
| [smoke-v027.jsonl](evidence/smoke-v027.jsonl) | 12:43:34–12:43:35，算术、JSON、tool_call合成API冒烟，3项通过 |
| [benchmark-provenance.json](evidence/benchmark-provenance.json) | 12:42:15，CPU/内存、模型配置与tokenizer/template文件hash、GPU瞬时状态 |
| [benchmark_client.py](evidence/benchmark_client.py) | 采集脚本与计时口径；使用合成输入，无业务数据 |

北京时间/台北时间为 UTC+8。源码参数、模型config、运行日志与请求metrics分别是不同证据；未采集到的实际backend不能由“nightly”“FP8”或MTP配置反推。

## 2. 初始线上现场

GPU 是 **1×NVIDIA RTX PRO 6000 Blackwell Workstation Edition**，计算能力12.0；总显存97,887 MiB（约95.59 GiB），采集时已用93,504 MiB，功率上限600 W，驱动595.71.05。这不是仓库README中4×RTX6000D的机器，已有四卡GLM/MiMo成绩不能作为本机Qwen的前后基准。采集瞬时GPU利用率0%，只描述该时刻，不证明整个采样期间独占或空闲。

后续provenance确认CPU为AMD Ryzen9 9950X，16核32线程、单socket/NUMA；名义内存64GB，Linux可见64,763,473,920bytes（约60.32GiB）。这套硬件信息来自本次服务器现场，不沿用旧四卡机的Xeon描述。

| 初始容器 | 镜像 | GPU-memory-utilization配置 | 上下文/并发配置 | 观察 |
|---|---|---:|---|---|
| vllm-qwen-27b | vllm/vllm-openai:v0.27.1 | 0.50 | 262144 / 16 | running，restart_count=1，主问答服务 |
| vllm-qwen3.5-9b | nightly（固定image_id记录于原始证据） | 0.22 | 131072 / 8 | 同GPU另一模型 |
| vllm-embedding | 同9B nightly镜像 | 0.05 | 8192 | Qwen3-Embedding-0.6B、pooling；后续实验保留 |
| vllm-qwen3-asr | vllm-openai-audio:nightly | 0.16 | 65536 / 3 | 启动显存检查失败并持续重启 |

这些配置比例合计0.93，但比例不是实时显存使用计量。GPU进程快照有三项EngineCore：5,368、65,918、22,182 MiB；未保存PID→容器完整对应关系，因此不把65,918 MiB直接归为某个模型，也不据此断言0.50失效。

ASR restart_count从40445到40447。错误日志明确显示启动时 **free 3.12 GiB < desired 15.2 GiB（0.16×94.97 GiB）**，随后 Engine core initialization failed。这是ASR启动资源不足的直接证据；Docker瞬时Status=running、OOMKilled=false没有否定持续初始化失败。当前证据只支持“共享GPU上存在资源配置问题与重启活动”，不能量化ASR拖慢Qwen的幅度，也不能声称停ASR即可解决prefill/decode瓶颈。

主27B容器读取到的运行包为 vLLM0.27.1、PyTorch2.13.0+cu130、Transformers5.15.0、FlashInfer0.6.16.post3。镜像ID为 `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`。服务model ID是 `qwen3.6-27b-fp8`；挂载模型config的architectures为 `Qwen3_5ForConditionalGeneration`、model_type=qwen3_5，文本层64层、每4层一次full attention、1层MTP。这里报告服务别名和文件元数据，不据别名验证权重发布revision或原始模型身份。

主27B启动参数：TP1、FP8量化、FP8 KV cache、chunked prefill、prefix caching、trust_remote_code、reasoning parser=qwen3、auto tool choice、tool parser=qwen3_coder、MTP num_speculative_tokens=2。FP8 config为dynamic activation、weight block128×128。没有显式 max_num_batched_tokens；最初的startup快照未给出实际backend；后续重启日志已确认attention/GDN路径，见第5节。FP8 linear具体dispatch尚需profile，不能假定已启用最优SM120内核。

## 3. 共享 GPU 的 localhost 基线

原始11行包括1次预热（32输出tokens，不计入以下统计）、9次正式不思考请求、1次思考观察。正式请求均并发1、temperature0、ignore_eos=true、stream=true；短/中输入固定256输出tokens，长输入固定1输出token用于prefill。所有输出 finish_reason=length；这来自强制固定长度，不代表真实业务回答被正常完整生成。

合成word_count是输入生成参数，不是token数；真实prompt_tokens来自usage。每轮前缀不同，实际输入token数略有变化，以下给出范围。短、中、长各3次交错执行；只报告中位数和min–max，不以n=3估计稳定p95。

| 请求组 | 输入tokens范围 | 输出tokens | 次数 | 首文本秒，中位数（范围） | 总秒，中位数 | 客户端decode估计tok/s，中位数（范围） | 服务端prefill估计tok/s，中位数（范围） |
|---|---:|---:|---:|---|---:|---|---|
| short_decode | 666–678 | 256 | 3 | 0.1367（0.1146–0.1408） | 2.8902 | **91.9（88.0–93.9）** | 5,054（4,976–6,170） |
| medium_decode | 4785–4823 | 256 | 3 | 0.6961（0.6278–0.6994） | 3.5080 | **88.5（88.3–91.9）** | 6,975（6,932–7,793） |
| long_prefill | 18924–18935 | 1 | 3 | 2.4118（2.2919–2.4262） | 2.4119 | 不适用 | **7,927（7,884–8,341）** |

正式9次均成功返回usage，prompt_tokens_total/generation_tokens_total差值与usage一致，prefix_cache_hits增量均0。所有11个metrics窗口的 `single_request_metrics_window=true`：主27B running_before=waiting_before=running_after=0，request_success与对应request计时count增量为1。由此能将这些主服务metrics差值用于单请求估算；它不排除同GPU的9B/embedding活动或ASR启动开销。

思考观察只有1次：4809输入、256输出tokens，首文本0.6314s，总3.2685s，reasoning1093字符、content0、first_content=null。输出预算被思考用完，不能用它与不思考短答比较“用户收到答案更快”，也不能推断完整工具调用业务的延迟。

## 4. 公式与精度

令P为usage.prompt_tokens，N为usage.completion_tokens，t_first/t_last为客户端首/末非空content或reasoning分片的monotonic时间。

- 客户端decode估计：`(N−1)/(t_last−t_first)`，N>1。91.9/88.5就是此口径，不是总tokens/整请求秒。MTP可以把多个tokens合成一片，首片可能不止1token，所以这只是估计；SSE块数不等于token数（256token请求只有102–110个文本分片）。
- 服务端prefill估计：`P/Δvllm:request_prefill_time_seconds_sum`，要求窗口只有该请求、prefill_count增量为1、tokens差值匹配。它是服务端所定义prefill阶段的平均速率，含该阶段调度/执行成本，不等价于某个kernel微基准。
- 服务端TTFT：`Δvllm:time_to_first_token_seconds_sum`（count增量为1）；客户端首文本含HTTP/SSE解析和loopback传输，不能直接等同服务端TTFT。
- 队列秒：`Δvllm:request_queue_time_seconds_sum`。本次正式请求约8.2–14.2微秒，说明此组没有可观测主服务排队；不能外推生产忙时排队情况。
- MTP接受率：`Δaccepted_tokens/Δdraft_tokens`，与日志里的每位置接受率/平均接受长度分别报告。最初此组正式decode样本约67.9%–76.2%，单凭接受率不足以确定MTP是否更快；后续已补开/关对照，见第7节。

prefix cache hits=0意味着这一组没有缓存命中，适合观察冷前缀输入成本；prefix caching已配置开启不能当作这些请求受益的证明。初始日志的累计prefix hit rate约74%描述历史业务窗口，与本组cachehits0并不矛盾。日志里的“Avg generation throughput”是时间窗口聚合，也不能直接拿来替代单请求decode公式。

## 5. 原vLLM重启后的受控资源观察

按用户授权停止9B、ASR与原27B后，重启原v0.27.1的27B并保留embedding。启动日志现在给出了实际backend：文本attention使用 **FLASHINFER**、prefill/decode query dtype=bfloat16、KV=float8_e4m3fn、decode_backend=flashinfer-native、arch=sm120；**GDN prefill 使用 Triton/FLA**（`requested=auto, head_k_dim=128`），ViT attention是FLASH_ATTN，top-p/top-k采样使用FlashInfer。**FlashInferBackend配合spec-decode不支持FULL_AND_PIECEWISE，vLLM自动改为PIECEWISE CUDA graph**。这是实际限制的证据，但没有full graph对照，不能直接量化其性能损失。

同日志确认max_num_batched_tokens=8192，模型载入28.95 GiB，可用KV15.92 GiB，KV容量455,773 tokens，CUDA graph pool0.18 GiB。重启后的27B EngineCore进程为51,416 MiB；初始快照中大进程为65,918 MiB。这是两个时点的实测差异，涉及工作历史、缓存/allocator及进程重启，不足以证明显存泄漏；启动memory预算与峰值/长期进程显存不能混作一个概念。另一个保留的EngineCore为5,368 MiB。

受控文件也有1次预热、9次不思考测试、1次思考观察。此轮v1采集器每次随机生成prompt，与初始组token长度接近但具体内容不同；因此下表是两组描述统计，**不是相同body的配对A/B**，不计算“停止其他模型带来X%提升”的因果结论。

| 原v0.27.1仅27B+embedding | 输入tokens范围 | 次数 | 首文本中位秒 | 客户端decode估计中位tok/s（范围） | 服务端prefill估计中位tok/s（范围） |
|---|---:|---:|---:|---|---|
| short_decode，256输出 | 669–677 | 3 | 0.1377 | 96.6（94.6–97.7） | 5,018（4,976–6,083） |
| medium_decode，256输出 | 4789–4801 | 3 | 0.7119 | 90.9（87.6–91.9） | 6,799（6,772–7,624） |
| long_prefill，1输出 | 18905–18946 | 3 | 2.4796 | 不适用 | 7,685（7,680–8,212） |

此组同样全部single_request_metrics_window=true、cachehits0；短/中decode估计略高，长输入prefill没有提升。内容不匹配且每格n=3，现有证据不支持主模型慢全由ASR/9B造成，也不支持宣称资源隔离无效果。思考观察仅1次，同样单列，不计入正式不思考基线。

## 6. v2固定语料旧版基线

v2将user corpus固定，并为每个用例保留独立首前缀以测冷缓存；记录corpus_sha256与payload_sha256。不同轮次语料hash可能不同，比较版本时必须按相同label核对user corpus hash，首前缀可独立变换。主服务忙时拒绝强行测；每次采样另记embedding前后metrics，区分“服务仍运行”与“采样时发生GPU工作”。该文件14次请求含预热1次、思考观察1次、短/中/长各3次、持续1024输出3次；正式不思考统计仍每格n=3。

| v0.27.1 / v2，仅27B+embedding | 输入tokens范围 | 输出tokens | 次数 | 首文本秒，中位数（范围） | 总秒中位数 | 客户端decode估计tok/s，中位数（范围） | 服务端prefill估计tok/s，中位数（范围） |
|---|---:|---:|---:|---|---:|---|---|
| short_decode | 671–675 | 256 | 3 | 0.1388（0.1356–0.1440） | 2.7356 | **98.369（97.219–98.397）** | 4,980（4,830–5,094） |
| medium_decode | 4798–4811 | 256 | 3 | 0.6173（0.5365–0.6864） | 3.3692 | **93.276（85.715–95.053）** | 7,882（7,069–9,077） |
| long_prefill | 18899–18952 | 1 | 3 | **2.3778（2.2565–2.3964）** | 2.3779 | 不适用 | **8,018（7,945–8,463）** |
| sustained_4k_decode | 4780–4805 | 1024 | 3 | 0.5416（0.5397–0.6272） | 11.9019 | **90.734（88.151–92.192）** | 8,946（7,755–8,997） |

所有14次stream_done、complete_fixed_output与single_request_metrics_window均true，prefix cache hit增量0。embedding每个窗口所有`*_total`计数无变化，前后running/waiting均0；这支持“未观察到embedding请求活动”，仍不等同锁定GPU/CPU整个运行环境。短/中/持续生成依旧使用ignore_eos=true与length结束，思考观察不得混入正常正文或工具调用速度比较。

采样期间官方镜像image pull在后台进行。主执行者现场vmstat观察CPU93–94%idle、wa=0，但该观察不是全程CPU/I/O隔离证明，也没有锁定GPU clocks/power或温度。12:42:15单点GPU状态为P1、SM2857MHz、显存13365MHz、406.79W/600W、55°C、利用率96%、显存56,935MiB。候选运行后可重复旧版控制组，检查时间顺序、后台下载与热状态影响；当前不能将v2与早先v1差异全归因优化。

API冒烟3/3通过：算术与JSON使用正常EOS（stop），工具调用以tool_calls结束。它只验证合成API基本能力，不是MoBase端到端、真实工具执行或模型质量验收。

文件来源指纹保存于provenance：

| 文件 | SHA256 |
|---|---|
| config.json | `885e6830f8d6883fefd63e3608c267452da2e6ce353a3494f42a7aa3d70c8434` |
| tokenizer_config.json | `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b` |
| tokenizer.json | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| chat_template.jinja | `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` |
| model.safetensors.index.json | `6d19a4e607604c1ac631f810a56e6084c892b4cb0251c530c6a24fc877f9fb4b` |

权重分片数66、MTP parameter keys=22；这些配置/索引指纹不是66个权重文件的完整内容hash，也不是官方权重revision核验。当前文件足够核对候选是否复用同一挂载配置/tokenizer/template；权重来源身份仍需单独记录。

## 7. 关闭 MTP 的同权重对照

保持v0.27.1镜像、同一只读模型挂载、TP1、FP8权重/KV、maxlen/maxseq/gpu-memory-utilization和embedding不变，只移除`--speculative-config`。临时容器在同一localhost端口串行运行。证据：[paired-v027-nomtp.jsonl](evidence/paired-v027-nomtp.jsonl)、[runtime-v027-nomtp.json](evidence/runtime-v027-nomtp.json)、[可重算汇总](evidence/summary.json)。每个label的user corpus与MTP2臂一致；首个system UUID依然不同以避免前缀命中，不能称请求字节完全相同。

| 请求组 | MTP2 decode估计tok/s | 无MTP decode估计tok/s | MTP2 TTFT秒 | 无MTP TTFT秒 |
|---|---:|---:|---:|---:|
| 约0.7K输入 / 256输出 | 98.37 | 50.76 | 0.139 | 0.109 |
| 约4.8K输入 / 256输出 | 93.28 | 50.54 | 0.617 | 0.510 |
| 约19K输入 / 1输出 | — | — | 2.378 | 2.236 |
| 约4.8K输入 / 1024输出 | **90.73** | **50.48** | 0.542 | 0.520 |

各格n=3中位数。无MTP的14行也全部完整输出、单请求窗口匹配、cache hits0，embedding计数无变化且running为0。启动日志确认无MTP臂为PIECEWISE=7、FULL=5的CUDA graph，GDN prefill仍为Triton/FLA；MTP2臂只有PIECEWISE。因此，虽在启动时捕获FULL图，该组合仍未弥补去掉MTP的代价；本次约4.8K/1024组合里MTP2约为无MTP的1.80倍。**建议保留现有MTP2，不为FULL graph单独关闭它。**

MTP2长输出组接受率中位约71.7%，平均接受长度约2.43；无MTP组没有draft计数。两组仍未锁频、样本有限且后台有镜像下载，不能把1.80倍作为所有业务/并发/上下文的固定收益。无MTP首token略早，说明prefill/首步与持续输出的目标存在取舍，应按MoBase真实输出长度决定。

## 8. 官方 v0.31.0，同一3.6-FP8权重

[paired-v031.jsonl](evidence/paired-v031.jsonl) 使用同一只读模型目录与启动参数，变化为整个官方vLLM镜像。镜像下载已结束、3.8权重核验已结束才发请求，embedding仍在，其中long_prefill_r3窗口记录到1次170-token embedding请求，其余窗口未观察到embedding计数变化；每个label的user corpus与旧版相同，system UUID不同。不是只换一个kernel：vLLM、FlashInfer、Transformers、执行器、默认dispatch等一起变化，因而报告为**运行栈组合升级**。

| 请求组，C1、每格n=3 | 0.27.1 TTFT秒 | 0.31.0 TTFT秒 | 0.27.1 decode估计tok/s | 0.31.0 decode估计tok/s |
|---|---:|---:|---:|---:|
| 约0.7K / 256 | 0.139 | 0.085 | 98.37 | 92.63 |
| 约4.8K / 256 | 0.617 | 0.494 | 93.28 | 97.88 |
| 约19K / 1 | 2.378 | 2.179 | — | — |
| 约4.8K / 1024 | 0.542 | 0.497 | 90.73 | 94.23 |

约19K服务端prefill从8018到8764 tok/s。这一小样本里新版首文本延迟更短，但decode并非每格都提高；不能写成“全面提速”或承诺130/200 tok/s。旧版组有后台镜像下载，复测结果另列。

[runtime-v031.json](evidence/runtime-v031.json)、[graph-v031.json](evidence/graph-v031.json)确认：

- vLLM0.31.0、torch2.13.0+cu130、Transformers5.17.0、FlashInfer python/cubin0.7.0.post1、CUTLASS DSL4.7.1；B12X未安装。
- GDN prefill实际为FlashInfer，head_k_dim128；GDN decode为cuda；文本attention实际为FlashInfer XQA、query BF16、KV FP8、SM120。
- FULL decode graph捕获成功；仍同时存在PIECEWISE图，不能称所有执行都变为FULL。
- hybrid cache manager的attention逻辑block为1600，XQA底层JIT页面为64，两者不能混写；KV容量440878 tokens（约15.36GiB）。旧版455773 tokens；相同gpu-memory-utilization不保证相同KV容量。
- 日志提示q/prob与KV部分scale回落到1.0，必须做长上下文质量回归；三个简单API冒烟通过不等于校准/质量验收。

[smoke-v031.jsonl](evidence/smoke-v031.jsonl)中算术、JSON、tool_call三项通过，工具仅生成调用参数，未执行任何实际工具。14个性能请求均完整、单请求metrics匹配、cache hits0。初始化和预热不计入统计；仍需结合JIT日志识别首遇shape开销，样本数不足以声称p95改善。

## 9. Qwen3.8与复测

复用 `/home/llm/models/Qwen3.8-27B-FP8`，本地66个权重分片的SHA256全部匹配[ModelScope官方文件](evidence/qwen38-modelscope-verification.json)，没有下载新权重。使用同一0.31镜像、FP8 KV、MTP2与其他参数，仅改模型目录和served-model-name。user corpus逐label一致；模型、tokenizer和模板随权重版本一起变化，生成内容也可能不同，不能归因为某个算子变化。

| 同0.31/MTP2，每格n=3 | 3.6 TTFT秒 | 3.8 TTFT秒 | 3.6 decode估计tok/s | 3.8 decode估计tok/s |
|---|---:|---:|---:|---:|
| 约0.7K / 256 | 0.085 | 0.086 | 92.63 | 96.06 |
| 约4.8K / 256 | 0.494 | 0.497 | 97.88 | 92.17 |
| 约19K / 1 | 2.179 | 2.190 | — | — |
| 约4.8K / 1024 | 0.497 | 0.500 | 94.23 | 93.39 |

3.8长输出总时中位11.454秒，接受率约69.6%、平均接受长度2.39；约19K serving prefill为8700 tok/s。本次没有证据支持“换3.8就大幅加速”。[paired-v031-qwen38.jsonl](evidence/paired-v031-qwen38.jsonl)14行全部完整、单请求metrics匹配、cache hits0；embedding窗口计数无变化。[API冒烟](evidence/smoke-v031-qwen38.jsonl)三项通过，仍不构成3.8业务质量优于3.6的结论。

实际新后端、包版本与缓存参数见 [runtime-v031-qwen38.json](evidence/runtime-v031-qwen38.json)。

### 下载结束后的旧版回测

[paired-v027-repeat.jsonl](evidence/paired-v027-repeat.jsonl)在0.31/3.8测试结束后重启旧测试容器、重新预热完成。没有镜像下载或权重哈希扫描，仍保持embedding。14/14请求完整且主模型metrics匹配，corpus与其他组一致、cache hits0；sustained_4k_decode_r1窗口有1次57-token embedding请求，其余窗口未观察到embedding计数变化。

| 0.27.1/MTP2复测，n=3 | TTFT中位秒 | decode估计tok/s | 原0.27.1首轮decode估计tok/s |
|---|---:|---:|---:|
| 约0.7K / 256 | 0.139 | 100.23 | 98.37 |
| 约4.8K / 256 | 0.691 | 91.25 | 93.28 |
| 约19K / 1 | 2.403 | — | — |
| 约4.8K / 1024 | 0.702 | **89.71** | **90.73** |

因此新版长输出94.23比旧版两轮89.71–90.73略高，约19K TTFT2.179也低于旧版2.378/2.403。短输入新版decode92.63低于旧版98.37/100.23。首步尤其4.8K TTFT有波动；未锁频、非随机交叉顺序、每组仅n=3，不能给出稳定百分位或把差额全部分配给单一kernel。模型版本比较仍是3.6/3.8同新版框架的独立对照。

### 保留embedding的实际负载与敏感性

五组v2的70个窗口中，有两次观察到embedding请求计数变化：0.31/3.6的long_prefill_r3为1次170-token请求，0.27.1回测的sustained_4k_decode_r1为1次57-token请求。主模型单请求metrics匹配不等于同GPU没有其他工作。上述各表保留原n=3，并明确披露这两个窗口；若剔除，0.31长输入TTFT为2.1668秒（n=2），旧版回测1024输出decode为89.3188 tok/s（n=2），结论方向未改变。不能以两个无活动窗口声称可靠百分位或统计显著性。

逐窗口检查与剔除统计见 [embedding-overlap-audit.json](evidence/embedding-overlap-audit.json)。初始v1未记录embedding指标，不能被这次v2检查追溯证明为空闲。

### 基准测试结束后的恢复（调整cache前）

[post-benchmark-restoration.json](evidence/post-benchmark-restoration.json)确认原`vllm-qwen-27b`（0.27.1）与原9B均已恢复，embedding保持原启动时间`2026-10-06T09:31:08.313413609Z`，未在本任务中重启。8000/8001/8002的health均200，models返回各自原服务模型；embedding一次合成输入返回1024维向量。GPU现场已用显存75,192MiB（约73.4GiB），利用率瞬时0%。

[smoke-restored.jsonl](evidence/smoke-restored.jsonl)聊天、JSON与tool_call三项通过。ASR保持停止，避免恢复既有失败循环，当时restart_count=40461；未调整其配置。4个本次创建的测试容器已移除，0.31镜像保留以供后续复现。此时原模型权重/原容器配置、MoBase代码未更改，生产未切换至3.8或0.31。随后用户要求削减27B cache恢复ASR，最终配置见下一节。

## 10. 结论边界

合成localhost请求不包含公网/隧道、MoBase长prompt多节点/工具、SSE正文过滤，也不构成完全独占GPU或完整业务SLA。ASR持续重启是配置问题，但“它就是主模型慢的主因”无对照证据。本轮服务恢复后仍需MoBase真实路径和质量验收，才能决定正式升级。

## 11. 用户追加：缩减27B cache，恢复ASR

用户在基准完成后明确要求通过削减27B cache给ASR留显存。仅将原0.27.1的`--gpu-memory-utilization`从0.50降至0.45；仍为原3.6-FP8、FP8 KV、MTP2、max-model-len262144、max-num-seqs16，模型目录和其他Config/HostConfig保持一致。它减少进程的显存预算，主要压缩KV池；不是把计算能力限制到45%。

原0.50容器停止并保留为 `vllm-qwen-27b-cache50-backup-20261013`，新0.45容器沿用原名与端口；相同image ID已校验。[变更记录](evidence/cache-budget-change.json)。基准表均是本次改cache前的0.50配置，不能改写成调整cache后四服务共存的性能。

[runtime-cache45.json](evidence/runtime-cache45.json)确认KV池11.15GiB、容量318743 tokens，262144-token请求的理论最大并发由此前约1.74x降为1.22x。单个262K上下文仍可容纳，但长上下文并发余量减少；max-num-seqs16不代表能同时跑16个满262K请求。相对原受控启动的15.92GiB KV池，本次少分配约4.77GiB。

原ASR保持其原镜像/参数，启动后health200。[asr-transcription-smoke.json](evidence/asr-transcription-smoke.json)使用Mac系统TTS产生的5秒合成英文WAV，接口返回200，约0.903秒完成，并正确识别“Hello. This is an audio test. One, two, three.”；这是单条功能冒烟，不是ASR吞吐或中文质量测试。[ASR运行记录](evidence/runtime-asr.json)保存实际版本与启动日志。

[smoke-cache45.jsonl](evidence/smoke-cache45.jsonl)确认27B聊天/JSON/tool_call三项仍通过。[final-state.json](evidence/final-state.json)为用户追加要求后的最终状态：27B、9B、embedding、ASR全部running且health200；embedding原启动时间未变。原/新27B Config仅Cmd变化，HostConfig零差异，镜像ID相同，Cmd差异仅0.50→0.45。生产仍为原3.6/0.27.1；框架、权重、MTP与MoBase代码未改。

如需回退，先停ASR，再停止新27B并改名，将 `vllm-qwen-27b-cache50-backup-20261013` 改回 `vllm-qwen-27b` 后启动；不要同时启动两个27B容器。停止状态的旧容器保留原0.50配置和编译缓存，不参与当前GPU占用。

[追加稳定性检查](evidence/asr-stability-check.json)在ASR启动约5分钟后再次确认四项health200，ASR与新27B的restart_count均0；这是短时恢复验证，不代表长期稳定性保证。
