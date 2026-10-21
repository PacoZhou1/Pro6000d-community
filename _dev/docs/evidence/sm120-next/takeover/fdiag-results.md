# Fdiag-v1 实际结果与无诊断 Bfix 对照计划

本稿为独立本地复算，读取 [fdiag-v1](fdiag-v1/) 和 [f-received/aba](f-received/aba/) 的原始JSONL、启动/服务/来源记录。未连接服务器、运行GPU、改变服务或修改任何已冻结源码。该run由root执行，controller记录2026-10-15 10:24:46至10:29:13 UTC、state=completed；四类API子进程均exit0。

**结论：这次插入参考、clone和同步后的组合运行中，所观察的F1/F2局部数学逐位通过，且decode正文和speculative工作量与历史A1/A2/BC一致。它没有定位历史B的根因，也不构成修复或性能通过。**

## 来源与实际配置

runtime为固定image `a4a4c043…c3a6b`、相同模型/启动cmd、MTP4、FP8 KV、FlashInfer；`PRO6000D_F_ENABLE=norm,conv`、F1 VERIFY与F2 VERIFY_CONV均为1。启动日志20行的splitting_ops包含新opaque op，`enforce_eager=False`，没有 `--enforce-eager`，也没有profiler启动。日志163/210行记录正常的CUDA graph capture完成，实际graph pool约0.25GiB；不是全模型eager运行。

sources与本地冻结诊断目录完全匹配：

| 文件 | SHA256 |
|---|---|
| f_kernels.py | `1b51a6d934e214190fcb586b217d1f890d096a699fdc06a83c97f0f2d61c8dd6` |
| sitecustomize.py | `e85188ff9ca8579acef309a15597a39072027279fd97879c0ae37bccbf4121db` |
| f_diagnostic.py | `37780728f8494127a63172fdea5e5cc3dc8edb94ac1a769bd5010680fed260d0` |
| V2 entry / V2 library | `8cdace70…0753b63` / `087b7caa…933b3f` |

核验对象为 [source-manifest.json](fdiag-v1/source-manifest.json)、[runtime.json](fdiag-v1/runtime.json)与diagnostic本地源。当前site仍包含旧md=None未写q/s分支，**没有**使用 `f-mdnone-fix`。因此“无需先修mdNone也能在另一执行安排下得到一致文本”是实际观察；它既不能否定该确定缺陷，也不能证明缺陷与历史B无关。

KV日志164–167、215行仍为30.75GiB、848709 tokens、四组各1616 blocks，与历史五臂相同。不存在这轮已观察KV token容量改变的证据。

## 严格局部数值覆盖

### F1

[server.log](fdiag-v1/server.log)提取到 **2112条** `P3 VERIFY norm+quant in-situ`。calls恰为1…2112，均无跳号、重复或累计值回退；每条累计FP8 byte/scale bit mismatch均为0。最后一条calls2112、M2502。所有观察长度见下表。

### F2

[prof/f2-diagnostic-516.jsonl](fdiag-v1/prof/f2-diagnostic-516.jsonl)为 **2448条**，同一PID516，call恰为1…2448，全部schema为f2-in-situ-v1、pass=true、cumulative_failed_calls=0，无capture_not_supported或其他error记录。每条六项顺序恰为q/k/v/g/beta/conv_state，均dtype/shape一致、mismatches0。

独立检查各比较的elements：q/k各`L×2048`，v为`L×6144`，g/beta各`L×48`，state为`586×10240×7=42004480`。所有条目完整覆盖，合计 **265095055872个元素比较**，没有以切片或只比较active state代替完整state。原synthetic state为3列，而真实state为7列，本次完整比较包括额外保留列。

| 实际token长度 | F1 calls | F2 calls |
|---|---:|---:|
|42 /64 /92|0|各48|
|79 /120|0|各96|
|1616|336|336|
|2502|96|96|
|2728,2730,2736,2738,2740,2752,2763,2767,2769,2770,2773,2779,2800|各48|各48|
|3164,3165,3166,3171,3177,3184,3189|各48|各48|
|9696|48|48|
|16160|624|624|
|32320|48|48|
|合计|2112|2448|

F1与F2的差额336次恰为F2观察到的M<256调用，与F1 M>=256守卫相符。F2每条metadata为num_prefills1/num_decodes0，state shape恒为`[586,10240,7]`、stride恒为`[1654784,1,10240]`，x形状为`[10240,L]`。记录中有16个不同state data pointers，但没有layer_name，**不能只靠这些计数证明逐层身份覆盖或把指针数当层数**。覆盖仅限被包装器实际命中并留下记录的调用，不涵盖未命中fallback、MTP decode、所有形状/输入域或整个GDN ssm state。

此F2诊断使用克隆初始conv_state给官方conv+post，再用真实state运行候选；full-state一致仅说明在该次实际输入/初态/顺序下两条局部链一致。F1参考使用已经由当前core路径算出的同一core_out，因此仍未与A的整层GDN core/ssm状态、out_proj输出逐位对照。

同步verifier若在eligible F2调用时发现CUDA capture会明确失败；这次无此记录，说明这些进入Python诊断的调用在检查时不处于capture。**这不是Nsight的逐kernel direct/replay关联证明**；不能把全模型graph配置等同于F1/F2已通过真实Graph replay审计，也不能把未进入诊断的其它模型路径算进覆盖。

## 实际请求与输出

prefill为13条（1warmup+12正式），decode为7条（1+6）。全部valid、stream_done、finish=length；每metrics窗口成功请求数为1，prompt/generated计数与usage一致。正式prefill/decode都没有prefix cache hits，embedding before/after指标均不变。连同warmup，payload hash与历史A1分别13/13、7/7相同；正文hash也13/13、7/7相同。

六条正式decode均1024 completion tokens、空reasoning。对照如下；数对为`drafts / accepted`，不是由文本长度推算。

|index / prompt tokens|A1=A2=BC=fdiag|BN|历史B|
|---|---|---|---|
|1 /4787|328 /698|330 /696|321 /702|
|2 /4793|310 /713|311 /712|357 /666|
|3 /4800|349 /677|347 /679|337 /686|
|4 /4781|289 /734|290 /733|345 /682|
|5 /4780|342 /681|343 /680|341 /685|
|6 /4805|351 /675|350 /676|342 /684|

fdiag的正文对A1/A2/BN/BC均6/6相同，对历史B为0/6；drafts与accepted对A1/A2/BC均6/6相同，对BN/B均0/6相同。accepted总数4178，和A1/A2/BC一致，区别于BN4176、B4105。因此本次不仅恢复了最终正文，也恢复了这组speculative计数；但没有保存逐token logits/首分叉位置，无法据此解释历史差异如何产生。

quality 5/5各自通过；历史quality记录没有payload hash，**不要把缺失content_sha256的None相等当输出一致**。open-output 4/4通过，有完整payload hash，和历史A1四条payload、正文、reasoning hash均一致。它们是固定合成检查，不是业务或整模bitexact证明。

## 服务保留证据

services-before/after均passed：原27B/9B/ASR维持停止，embedding维持运行，身份、启动时间、restart_count匹配。两端embedding health200、实际embedding API返回1024维有限vector。monitor为18条，全部health200/running，ID/start/restart不变，最大相邻间隔 **15.021342秒**，首尾跨度255.177191秒。该采样不证明无瞬时延迟影响；本次18个正式性能请求的embedding指标窗口静默，也不使同步诊断时间成为性能结果。

controller completed及API exit0已留证；当前这些文件未单独给出“owned container removed”字段，不以服务保留检查代替该清理事实。如需最终恢复证明，root应另存自有容器盘点。

## 为什么尚未定位或修复根因

fdiag在F2前插入官方链与full-state clone，之后逐张量比较、CPU同步和JSON日志；F1也加入真实参考、比较和同步。这些改变会影响分配/地址复用、工作排队、时间和cache状态，可能掩盖、改变或暴露原路径问题。**“开诊断后恢复一致”只能证明现象对执行安排敏感，不能证明某个特定race、未初始化读、Triton错误或MTP算法就是根因。**

本轮没有移除融合、没有修mdNone，却已恢复匹配，意味着不能将这次恢复归功于尚未运行的最小补丁。也不能由2448次局部F2零差异宣称所有组合及无诊断版本安全。所有时延/TPS因主动参考与同步均不用于加速结论。

## Bfix无诊断对照计划（由root决定执行）

1. **仅用已冻结的 `f-mdnone-fix`。** site SHA `854043107312ee708f4107e35362d6096ce270cc33d4a50293d1f2026df1bbcc`；kernel与V2原样，唯一逻辑差异为md=None补官方量化写q/s。保留norm,conv与原模型cmd/MTP4/FP8KV/graph配置。F_VERIFY、F_VERIFY_CONV均unset，且不加载f_diagnostic；冻结所有source/env/命令及V2实际库SHA，保证没有clone/reference/日志同步漏入。新目录新结果，不覆盖历史B或fdiag。
2. **首先看正确性与可比工作量。** 保留同一脚本及全部请求（warmup单列），逐索引payload hash、usage、finish、请求窗口、embedding活动；比较12prefill与6decode正文hash，以及每条drafts/accepted。quality缺hash标unavailable，open-output须核payload及正常final body。固定image CPU/null-MD poison测试若还未做，作为独立验收补齐，不把服务启动成功替代q/s全写证明。
3. **结果分流。** 若文本和accepted均恢复，记录为“Bfix在该无诊断窗口与历史基线一致”，随后fresh A1/Bfix/A2再确认；不能从一次独立Bfix对旧A断言因果或性能。若正文相同但accepted仍BN式不同，decode工作量尚未固定，先不作纯算子加速结论。若正文仍分叉，最小补丁没有消除现象，保持拒绝采用，转到整层GDN输入/活跃conv+ssm状态及输出projection的受控对照；不要再以局部VERIFY为理由放宽门槛。
4. **性能另门。** 只有无诊断输出/状态证据足够后，fresh夹持ABA用相同KV分配与source、相同请求/embedding窗口，报告paired TTFT相对baseline drift与独立窗口限制；不要把fdiag time、旧BN/BC探索性数字或旧nsys时长并入。无论结果如何，原服务停止/embedding保留政策不变，不自动上线。

主要原始证据SHA：F2 JSONL `fa8833492662cfbf1b2efb1c44cb50cd38b7c8b9c8273cc6ca1a938745b95600`；server.log `b8bac7302fb0ee8180a6875970b44ef98812fd355ed0b065e8b405b5724e3363`；decode.jsonl `cc7d47d9f7ca650426888954fbc9d820c535d874682abd58bd0e6fffbd4e3356`；runtime.json `e18f40b3a022754e660a1c6944983da6a016839f463b8bde43094241c6f6fa51`。
