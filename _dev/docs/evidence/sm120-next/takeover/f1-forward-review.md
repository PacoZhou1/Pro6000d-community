# F1 forward / opaque / MTP：本地复核与最小修复草案

只读固定源码、已恢复运行记录；未联网、SSH、GPU或服务操作。已运行的 `f-diagnostic` 不修改。新修复单独放在 `f-mdnone-fix`，由root决定运行。这是契约修复与差异定位，不预先认定B长decode分叉根因。

## 已确认与必须撤回的假设

固定官方文件为 [source/qwen_gdn_linear_attn.py](source/qwen_gdn_linear_attn.py)，来自vLLM提交 `db9527a46873454610df6dbedf79a36d6bf1a7f6`（root已核官方来源并保存）：
`https://raw.githubusercontent.com/vllm-project/vllm/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`。

- 官方895–929：BF16/enable_fused条件命中后创建zero core_out，调用packed opaque，最后走 `self.out_proj(core_out.flatten(-2))`，由原FP8 linear执行其输入量化。
- 官方1766–1796：metadata存在时，packed方法只是相同的qkv/z切分、split_ba，再调用 `_forward_core_fused_norm`。它没有额外单独的packed MTP分支。
- 官方1849–1891：`_forward_core_fused_norm` 本身检查 `_can_use_fused_gdn_mtp_decode` 和 `num_prefills==0`，命中则1684–1729的spec decode+norm路径，最终调用1731–1764的 `ops.fused_gdn_decode_post_conv_mtp`；否则普通 `_forward_core` 再gated RMSNorm。

因此，初步提过的“F1 fallback直接用unpacked方法，可能绕过packed MTP快路”在拿到固定源码后**不成立，应撤回**。F1在正常metadata fallback时按同样的切分调用同一个方法，原MTP选择仍在。没有证据证明它变成了另一套MTP状态算法。三臂/五臂日志中的“FlashInfer不支持fused multi-step draft，重建attention metadata”是外层speculator日志，在所有臂都有；不能把它混同于GDN局部fused decode选择。

## 明确的md=None契约缺陷

恢复的F1 [sitecustomize.py:94–96](f-received/instr/sitecustomize.py)在md=None时调用官方packed方法后直接return。官方1778–1780只prewarm并return；不会填q/s，也不会改变传入zero core_out。新forward在172–173把q/s分配为torch.empty，181却不经输入量化而直接 `kern.apply_block_scaled_mm(A=q,As=s,...)`。q/s并未传给官方packed方法，因此这个分支中**必然缺少对新输出的写入**，不是依赖具体CUDA数值的猜测。

原forward虽然core_out仍为零，最后仍经正常out_proj的输入量化，因而没有这个未初始化q/s读取。新runtime的最小草案是在null-MD分支补与现有fallback完全相同的官方group quant：

```diff
 if md is None:
     self._forward_core_fused_norm_packed(...)
+    torch.ops._C.per_token_group_fp8_quant(
+        core_attn_out.flatten(-2), q, s, 128, 1e-10,
+        fp8_min, fp8_max, False, True, False)
     return
```

已复制出 [f-mdnone-fix/sitecustomize.py](f-mdnone-fix/sitecustomize.py)，只改此分支，F1/F2 kernel、正常metadata融合/回退、MTP条件、split-op与量化数学均不变；不包含`.so`。使用同样V2及F_ENABLE开关。没有把“packed fallback替换”或prewarm改法一起合入，以便隔离一个确定缺陷。

这不等于证明decode根因：md=None主要用于profile/warmup。仍需实际分支命中、输出读取与后续执行证据，才能判断它是否影响真实捕获、状态、采样或只影响启动。若独立修复后B文本仍不同，应继续定位；若恢复一致，说明与该变更相关，还需fresh重复与具体执行链证据。不能因一次恢复一致便宣称整个双融合数值安全。

## 已知组合现象及解释边界

原五臂decode逐请求 `(drafts,accepted)`：

| index | A1=A2=BC | BN | B |
|---:|---|---|---|
|1|328,698|330,696|321,702|
|2|310,713|311,712|357,666|
|3|349,677|347,679|337,686|
|4|289,734|290,733|345,682|
|5|342,681|343,680|341,685|
|6|351,675|350,676|342,684|

A1/A2/BN/BC的正文hash六条逐条相同，B六条都不同；BN已改变speculative工作量，说明它并非“只在大prefill换一条kernel而其他执行工作完全相同”的证据。B与BN都使用splitfix，BC不装F1新forward/opaque。这个2×2观察支持把F1接入边界作为重点，但不能仅由组合分叉推断F2有race或F1归约错误。非线性argmax、recurrent state、graph/allocator路径和真实输入都可能放大或隐藏局部差异，现有hash无法区分它们。

重要反证：五臂server日志都报KV cache30.75GiB、848709 tokens，四组各1616 blocks，actual CUDA graph pool约0.25GiB；没有证据表明B分叉来自KV token容量不同。B/BN estimated graph memory0.85GiB，对照/BC0.86GiB，只是估计口径，不是已证明数值差异。API decode payload hash一致、temperature0、prefix hits0；不能凭输出不同就归为普通随机采样。长decode只存全文hash，尚无首个分叉token/logits。

## 现有VERIFY的覆盖空洞

F1 VERIFY在 `_forward_core` 已运行之后，以**当前分支产生的同一core_out**作为norm+quant参考输入。即使FP8/scale都一致，也没有对照官方整个GDN core、更新前后conv/ssm state或out_proj的输出。旧WV192次零差异还来自pre-splitfix图。新的F2诊断从同一conv_state和x比较局部conv+post，若通过，也不证明其输入state与A基线相同。

故现阶段正确推理是“已测局部数学保持一致；完整前后文和图路径仍待对照”，不是“局部零差异所以整模文本应该相同”。

## 最小验收与分流

1. **null-MD poison门。** 在隔离GPU单层/官方上下文中把core_out初始化为零，把q和s分别poison为非零bytes/NaN，设置md=None，调用新opaque后与“官方packed warmup + 官方quant(zero)”逐位比较；要求q全部被定义、scale为官方eps/448位模式，输入尺寸至少含1/5/256/16160。再测真实CPU fake/schema、Inductor和启动profile/capture，确保只修契约且没有新图错误。本地AST抽取+metadata-only mock已复现旧分支q/s均不写，并验证草案两输出都调用quant写入；这不是CUDA数值验收。
2. **先读正在跑的fdiag-v1。** 对其真实F2 q/k/v/g/beta/全state和F1 q/scale逐调用结果按L/call核覆盖。遇capture不支持或服务失败必须保留为未完成；不能跳过后称全过。若有first mismatch，优先取该步必要输入/slot/state/上游路线作受控复现；不要盲目用整模hash调kernel。
3. **同输入/同state的GDN边界对照。** 下一步应在新诊断目录对一个真实层调用保存并恢复**活跃slot**的conv/ssm state，分别走官方packed+输入quant与F1替换完整core+norm+quant；比FP8/scale、core输出、所有被改state，必要时下游out_proj BF16。不能连续在同一可变state上跑两次当成对照；先确认固定源码的kv_cache/metadata状态语义再写该诊断。重点含prefill、MTP draft非spec小M、target verifier spec分支、padded M与num_actual_tokens不等。当前F2/F1局部VERIFY不能替代此门。
4. **纯adapter控制。** 可另建版本只把F1融合数学换回 `gdn_norm_quant_reference + q.copy_/s.copy_`，保留新forward/opaque/split-op/F2不变，作为“融合kernel vs接入”诊断；这与mdNone修复分开跑，避免一次改变两个因素。若仍有BN accepted或B文本差异，应优先看包围这一步的state/编译/输出projection，而不能继续只调F1除法。
5. **正式复测。** root选定修复版本后冻结source/runtime/env，用同样MTP4/FP8KV、prompt和无profilerfresh基线/候选窗口复测正文hash及逐case drafts/accepted。先诊断输出一致性，再解释性能；若做MTP0对照须baseline/candidate同改且明确是额外诊断，不代表原MTP4证明。不得把同步fdiag时延用于加速结论。

另有明确的启动覆盖问题：F1 prewarm在f_kernels.py:268创建FP32 weight，真实model eligibility日志weight为BF16；B/BN虽然打印pre-compiled成功，随后首个真实请求仍出现 `_gdn_norm_quant_kernel` JIT warning。F2在B/BC也有首次JIT。F1的weight dtype至少提供了一个具体specialization不匹配点，但不能据此断言全部JIT原因；应该记录实际Triton cache key/launch参数后补覆盖。该改动暂不和mdNone修复合并，原性能warmup请求也已独立剔除。
