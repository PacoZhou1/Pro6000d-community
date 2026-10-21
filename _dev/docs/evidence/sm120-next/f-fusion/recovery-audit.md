# F-fusion 中断恢复审计

2026-10-15。本稿只读本地源码与已保存结果，未连接服务器、初始化CUDA、修改旧文件或操作服务。结论由JSONL独立复算；CHANGES的既有说法与profiler摘要单独标明证据等级。主agent负责恢复任务状态与后续服务器实验。

说明：前半记录首次扫描时的本地证据状态。root随后把漏同步原件恢复到新的 `../takeover/f-received`，本稿末尾补充了已补齐结论；原件与旧结果均未覆盖。

## 当前可确认的结果

五臂各自保存了13条prefill（含1条warmup）、7条decode（含1条warmup）、5条quality、4条open-output。20个客户端子进程均exit0。实际runtime.json确认相同image、model mount、启动cmd、MTP4、FP8 KV与V2量化库SHA；**没有**前几轮使用的显式12GiB KV参数，不可套用旧实验的内存公平性结论。实际自动分配KV规模、JIT/clock证据尚未在本地保存。

| 臂 | 实际环境 `PRO6000D_F_ENABLE` | 意义 | prefill TTFT median s | decode TTFT median s | decode TPS median |
|---|---|---|---:|---:|---:|
| a1 | unset | V2-only前基线 | 2.224715160 | 0.505563391 | 106.6079185 |
| b | norm,conv | F1+F2 | 2.195258502 | 0.500534355 | 104.5532335 |
| a2 | unset | V2-only后基线 | 2.238360925 | 0.507495328 | 106.6063712 |
| bn | norm | F1-only隔离臂 | 2.209184248 | 0.502598892 | 106.1009676 |
| bc | conv | F2-only隔离臂 | 2.218196749 | 0.505093924 | 106.6152774 |

来源为 [归档原始 ABA](../takeover/f-received/aba/) 各臂原始JSONL，剔除 `warmup=true`。prefill的每请求配对改善公式为 `100×(mean(A1_i,A2_i)-B_i)/mean(A1_i,A2_i)`，以下均再取配对中位数，未用“中位数之比”替代。

| 候选 | prefill全12对改善 | embedding静默配对改善 / n | decode TTFT改善 / n | decode静默配对改善 / n |
|---|---:|---:|---:|---:|
| b | 1.630061% | 1.587383% / 9 | 1.253853% / 6 | 1.244088% / 5 |
| bn | 1.037093% | 1.053723% / 8 | 0.734382% / 6 | 0.734382% / 4 |
| bc | 0.571508% | 0.561328% / 9 | 0.352841% / 6 | 0.351123% / 5 |

这里的静默指三臂对应请求的embedding before/after所有已存指标均不变，不证明采样间无瞬时活动。A2相对A1的逐请求TTFT漂移中位数为prefill **+0.575798%**、decode **+0.411575%**。实际启动顺序是A1→B→A2→BN→BC（runtime StartedAt为06:28:28、06:32:49、06:37:12、06:42:25、06:46:46 UTC）。所以BN/BC相对同一A1/A2的计算仅作探索性对照，**没有各自的前后夹持基线，不能称两轮独立ABA**。约0.57%的BC长prefill改善与这轮基线漂移同量级；需要新夹持窗口才能判断收益是否稳定。

## 输出与工作量：不能只看吞吐

所有五臂prefill/decode逐索引payload SHA均与A1相同，包括warmup。所有正式请求valid、流正常结束且finish=length；每个metrics窗口一个目标请求。正式prefill/decode均无prefix cache hits。以下hash比较排除warmup；reasoning全为空的相同hash不计作质量证明。

| 对A1比较 | prefill单token正文相同 | decode正文相同 | 六条decode accepted计数逐条相同 | decode accepted总数 |
|---|---:|---:|---:|---:|
| a2 | 12/12 | 6/6 | 6/6 | 4178 |
| b | 12/12 | **0/6** | **0/6** | 4105 |
| bn | 12/12 | 6/6 | **0/6** | 4176 |
| bc | 12/12 | 6/6 | 6/6 | 4178 |

A1 accepted总数亦4178。B的正文/接受工作量不同，decode TPS约低1.93%不能直接归因于融合kernel；BN虽正文一致，接受计数仍不同，decode工作量也非严格相同。BC在这6个decode上保持正文和accepted计数一致，是更干净的后续隔离对象，但一次窗口仍不够证明加速。正式decode原始文件只保存正文hash，不保存完整token序列、首个分叉token或逐层logits；无法从这些hash判断B为何不同、差异是否发生在prefill、recurrent state、MTP或其他整合路径。

五臂quality均5/5 pass；这些历史记录均没有payload SHA，只能确认固定脚本的同名案例各自通过，不能称捕获输入bytes一致。open-output均4/4 pass、同索引payload hash一致，且其正文与reasoning hash在五臂间均4/4相同。它们支持该组开放式合成检查，不抵销B长decode差异，也不构成完整业务质量或整模逐位证明。

[results/aba-comparison.json](results/aba-comparison.json)仅汇总A1/B/A2，长prefill全配对1.630061%、静默9对1.587383%与本次复算一致；BN/BC未包含在原汇总。现有 [scripts/analyze_aba.py:310附近](scripts/analyze_aba.py)把三臂quality缺失的 `None` hash直接比较相等，未像较新analyzer明确标记`payload_hash_unavailable`；其文字“quality/open-output payloads相同”需要限定，不能照抄当验收。

## F1/F2实际实现范围

**F1不是更换GEMM tile。** [f_kernels.py:24–129](../takeover/f-received/src/f_kernels.py)把GDN gated RMSNorm与后续per128 FP8输入量化融合，直接按z真实stride16384读取，避开z reshape copy。先按官方表达式得FP32 norm/gate值，**舍入BF16后**做amax、scale和FP8转换；scale和量化除法用 `libdevice.div_rn`。源码另加载一次x并使用evict_last，约束FP8 store向量为8B，意图保持官方归约布局。当前源码hash为 `1b51a6d934e214190fcb586b217d1f890d096a699fdc06a83c97f0f2d61c8dd6`，与B/BN/BC runtime环境一致。

真实接入在 [sitecustomize.py:82–185](../takeover/f-received/instr/sitecustomize.py)，新增opaque `vllm::pro6000d_gdn_core_norm_quant`，重写GDN forward：先算core，再fused norm+quant，直接调用原FP8 out_proj CUTLASS GEMM。融合守卫含SM120、BF16、128维、M>=256、真实token数与M一致、非fused spec-decode分支及layout约束。**即使未融合，其forward/opaque边界也已被替换**；fallback显式调用原core+norm再官方量化，不能认为只有大prefill kernel变动。MTP可能走不同core入口、compiler边界/初始化也不同，均只是需要核验的解释方向，不是已定位的decode根因。

源码末尾给 `CompilationConfig._attention_ops` 增加新opaque splitting op，防止跨层编译piece合并。此改动对图划分有实际设计意义；必须取回各臂sitecustomize快照/编译输出确定它何时加入。`md is None`路径（94–96）只运行原core并return，未填充q/s，但新forward后续仍以q/s进out_proj；这是warmup/profile路径契约的具体审查点，不是已证明真实请求错误。prewarm异常会被捕获为日志，并不阻止启动，所以必须恢复日志确认实际是否成功，不能从服务启动exit0推定无首次JIT。

**F2有两次launch，不是一核完成全部状态处理。** [f_kernels.py:135–253](../takeover/f-received/src/f_kernels.py)将width4 causal conv+silu与post-conv q/k l2norm、v拆分及g/beta合并；卷积结果先按官方BF16舍入，另一个state-update kernel写最后3个输入token。入口仅单序列prefill、无decode/spec、无bias、silu、非APC分支。接入通过 [sitecustomize.py:190–250](../takeover/f-received/instr/sitecustomize.py)中 `pending` host字典：conv包装器延后执行并返回stand-in，post包装器消费，未匹配则补执行官方conv。此状态传递、fallback、slot/initial-state、真实conv_state stride及capture路径都需要runtime级验证；直接kernel数值测试不覆盖完整包装器。

## 已有profiler摘要支持什么

本地 [nsys-w0-vs-wc.txt](results/nsys-w0-vs-wc.txt)记录96次F1、96次F2+96次state-update，原norm/conv/post和部分量化/copy调用减少；原candidate的一个Inductor norm/quant合成kernel总时长从30.45ms增到51.31ms。后来的 [splitfix摘要](results/nsys-w0-vs-wc-splitfix.txt)该项变为29.48ms，整窗口kernel时长和从2251.4→2203.9ms、kernel span2271.8→2223.7ms；原candidate时长和2233.4ms、span2255.7ms。

这些是**文本摘要**，目前缺对应nsys-rep、SQLite、命令/status、snapshot与runtime/server.log，不能独立核验窗口是否仅一次请求、direct/Graph关联或新splitting修改的版本。不能用2203.9ms推算无profiler TTFT收益，也不能认定当前ABA恰为splitfix前/后版本。`kernel_totals.py`把整份数据库中的kernel直接聚合并缩短/合并名字，没有请求范围筛选或launch关联验收；恢复时须补窗口与来源校验。

## 本地缺件与续接顺序

1. **先取回原始证据，不重跑覆盖。** 主机任务根为 `/tmp/qwen-sm120-next-f/`，源码和脚本指定位置为 `src/`、`instr/`、`bin/`、`logs/`、`aba/{a1,b,a2,bn,bc,w,wv,w0,wc}`。由主agent核对并恢复numeric.json及stdout/stderr、micro JSON、diag/quant semantics结果、server/control/run_api日志、gpu-clock.csv、control-cmd.json、initial snapshot、DONE标记、nsys原件/SQLite、Triton/Inductor相关源码与hash。当前本地results无numeric或micro输出；CHANGES中的“实测逐位一致”不等于可复核原始记录。各臂共享同一可变host instr目录，runtime仅pin了f_kernels，不pin sitecustomize/V2源码；不能根据同一mount路径证明全臂源码相同。
2. **补严格CPU分析，保存新文件。** 对五臂验证case/index/输出长度、payload、同image/cmd/env差异、embedding静默子集、全部content/reasoning/accepted/draft关系；质量hash缺失标unknown。BN/BC明确探索性，不能替换既有ABC summary或伪称各自fresh ABA。核actual KV bytes/可用内存与source版本是否同等。
3. **补正确性门的失败传播与runtime覆盖。** `test_numeric.py`按源码预期45个记录：F1 21、F2 22、两项Graph；当前脚本最后只输出all_pass/ALL PASS或SOME FAIL，**数值失败仍可能exit0**，不能只看容器exit。Graph检查是直接kernel调用、若干换输入与参考比较，没有显式保存output_changed；未证明runtime wrappers和每个实际model M。需要逐项核原始all_pass，后续新runner若有fail应非零退出。特别恢复wv在真实core输入上的F1 VERIFY日志；现有VERIFY仅记累计不一致，不主动raise，无F2全状态等价日志。
4. **微基准证据还不够闭环。** `bench_micro.py`虽随机交错、默认31轮，但允许更小轮数，只保存中位/最小改善而丢弃原始samples/orders；无同进程数值gate绑定或source/inputhash。所谓cold为固定256MiB flush，未query/证实固定cold-L2；F2两臂共用可变state，输出buffer与预分配策略也不同。取回结果后只能按原口径解释，后续用新目录补raw、独立state/outputs、明确setup/allocator/capture范围、正常/特殊数值及Graph门，再评估算子收益。
5. **若用户继续整模验证，先选隔离对象并冻结版本。** BC目前输出/accepted最干净，但收益与漂移同量级；BN/B存在不同层次MTP/文本差异。先确定splitfix源码与实际模型派发，诊断组合B首分叉及状态，再做fresh、夹持的单项/组合对照。保留MTP4/FP8KV、BF16 lm_head与原FP8语义；不要用无profilerdecode TPS单独宣布加速。若修改源码，旧5臂与旧profiler证据全部保留，不沿旧目录重新写结果。

## 服务与退出状态的边界

五臂services-before/after均passed；embedding-monitor分别26/27/26/26/26条，共131条，全部running、无error，各臂ID/start/restart不变。该monitor只读容器状态，**没有逐sample HTTP health**；前后verify另外执行health200+1024维有限vector检查。A2/BN的部分API窗口embedding计数非静默，已在上表剔除相应配对。最后BC的services-after显示27B、9B、ASR停止，embedding运行，其他非目标服务状态按初始snapshot保留。这是历史采样证据，不能当作恢复时刻的实时服务状态；本次审计未操作它们。

可续接点是“恢复F-fusion原始门槛与splitfix版本 → 对BN/BC补严格分析 → 定位B长decode差异/adapter状态 → 选定版本后fresh隔离验证”，不是立即采用1.63%或2.1%数字。当前证据不足以宣布任一F候选上线。

## 原始证据补齐后的更新

新原件位于 [takeover/f-received](../takeover/f-received/)，由root取回。下列 raw/source 链接优先指向该新归档；只存在旧目录的 profiler 文本与原分析脚本仍保留原链接。前文“尚未保存/缺件”是首次扫描历史状态，已被本节补齐内容取代。后续实际诊断与 fresh 修复对照见 [fdiag 结果](../takeover/fdiag-results.md) 和 [ffix fresh ABA](../takeover/ffix-results.md)。[numeric.json](../takeover/f-received/results/numeric.json)实际为 **45/45 ok、all_pass=true**，与源码预期case数一致；现在可确认所测直接kernel/回退/Graph数值门通过，不再仅依赖CHANGES文字。

实际 [micro.json](../takeover/f-received/results/micro.json) 为41轮，数值如下；仍只有聚合统计，没有原始samples/orders，因此不能独立重算分布与成对中位数：

| 操作 / M | warm官方µs | warm候选µs | warm配对改善 | 标为cold的配对改善 |
|---|---:|---:|---:|---:|
| F1 /16160 | 842.2080 | 338.9120 | 59.75355% | 61.15802% |
| F2 /16160 | 1096.9280 | 492.1600 | 55.12783% | 56.30671% |
| F1 /2741 | 58.9120 | 41.3760 | 30.86216% | 31.00970% |
| F2 /2741 | 179.6160 | 84.4160 | 53.71013% | 55.16551% |

“cold”仍是旧脚本固定256MiB flush口径，不是已证明固定cold-L2。上述改善仅为那两个函数范围的事件计时，不是完整模型收益。

**B不是pre-splitfix配置。** 恢复的各臂 `server.log:20` engine compilation_config给出：

| 臂 | splitting_ops包含 `vllm::pro6000d_gdn_core_norm_quant` |
|---|---|
| a1 / a2 / bc / w / wv / w0 | 否 |
| b / bn / wc | **是** |

因此可排除“B尚未加splitting op”这个具体解释，不能用它给B的文本分叉开脱。现有src/instr的sitecustomize均为 `ce01159d779ea193f7a86235aae1a147258b64db9c172f7a6772bf51c3b3380c`，kernel为前述1b51…8dd6；但还不能用最终host文件hash补造各历史臂完整source manifest。

`aba/wv/server.log`实际有192条F1 in-situ VERIFY，最后一条为484行：calls192、累计FP8 byte mismatch0、scale bit mismatch0，M2741。它支持该WV运行中的真实core→norm/quant结果，但WV的splitting配置是旧版，不能直接替代B/BN当前图划分下的组合验证。

root已授权新建独立 [takeover/f-diagnostic](../takeover/f-diagnostic/)，不修改已恢复源码。它保留FK数学与post-splitfix接入，新增每个真实eligible F2调用的原conv_state完整clone参考：官方conv+post与候选分别从相同初态出发，全bit比较q/k/v/g/beta及整个conv_state，记录每call的shape/stride/计数与首差位置bits，无张量/业务正文。F1共用原VERIFY，并改为任何差异立即raise。该版本为同步诊断，若F2触发CUDA capture明确记录并报错；没有静默略过或把eager成功当原Graph证明。CPU AST/mock检查已完成，实际GPU及组合输出定位由root执行，诊断运行时间不得作为性能数据。
