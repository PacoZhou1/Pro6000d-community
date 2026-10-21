# FP8 周边算子机会：现有 MTP4 CUDA Graph 捕获的离线拆分

本页只推进 **FP8** 路径，不分析或尝试 FP4，也不改 BF16 `lm_head`。输入是固定 vLLM 0.31.0、RTX PRO 6000 Blackwell 的四份真实 MTP4 Nsight Systems 2026.5.1 SQLite：两份约 19K prompt / 1 completion 的 prefill 与两份首文本之后开启的 steady decode。原始 SQLite、已有分析 JSON 及 eager 归因 JSON 的 SHA-256、完整原始 kernel 名、次数、纳秒求和、匹配规则都保存在[机器可读结果](profile-opportunities.json)。本轮只在本地读取已有文件；现有证据目录没有可独立使用的 PyTorch Chrome trace，因此不声称 CPU op 或 Python 调度的逐算子耗时。

**分母：**下表每个百分比均是该捕获内目标 GPU kernel 的 duration 之和，除以**同一捕获所有 GPU kernel duration 之和**；r1/r2 分别计算，未把两次合成一个请求。prefill 总和分别为 2,266.680/2,272.789 ms、3,673 个 kernel；decode 总和分别为 2,344.261/2,372.721 ms、90,058/91,143 个 kernel。GPU memcpy/memset 单列，未放进 kernel 分母。duration 求和既不是 GPU 活动区间 union，也不是 HTTP wall time 或吞吐增益。decode 的 NVTX 以 `execute_context_0(0)_generation_1(5)` 为主；它包含 target、MTP draft/verify 及采样，不能只凭名字把一个 kernel 分给某个模块。`GEMM_ATTRIB` eager 归因证明真实 `gate_up`/`down` 的 M/N/K，但它**不能把 eager 耗时和模块归属直接转成正常 CUDA Graph 的周边 kernel 归属**。[prefill 归因](../gemm-targeted/mtp4-prefill-attribution.json)、[decode 归因](../gemm-targeted/mtp4-decode-attribution.json)。

| 原始名称规则形成的互斥组 | prefill r1/r2：次数；耗时 ms；占 kernel 总和 | decode r1/r2：次数；耗时 ms；占 kernel 总和 |
|---|---|---|
| 独立 `per_token_group_quant_8bit_register_kernel` | 408/408；35.495/35.565；1.566%/1.565% | 16,932/17,136；32.671/33.092；1.394%/1.395% |
| 已融合 `silu_and_mul_per_block_quant_kernel` | 136/136；97.827/98.173；4.316%/4.319% | 5,644/5,712；6.707/6.785；0.286%/0.286% |
| 已融合 residual/RMSNorm/FP8 quant 的 Triton 名称 | 176/176；30.553/30.634；1.348%/1.348% | 7,304/7,392；12.083/12.234；0.515%/0.516% |
| 已融合 sigmoid/mul/FP8 quant 的 Triton 名称 | 40/40；6.640/6.639；0.293%/0.292% | 1,660/1,680；1.015/1.028；0.043%/0.043% |
| 已融合 residual/RMSNorm、**不含 FP8 quant** 的 Triton 名称 | 106/106；26.486/26.788；1.168%/1.179% | 4,399/4,452；8.517/8.622；0.363%/0.363% |
| 独立 `layer_norm_fwd_kernel` | 96/96；19.176/19.185；0.846%/0.844% | 捕获中未匹配此确切名称 |
| 已融合 QK RMSNorm/RoPE/gate | 40/40；10.707/10.782；0.472%/0.474% | 1,660/1,680；3.487/3.527；0.149%/0.149% |
| 明确命名的 GPU copy/index kernel | 812/812；18.438/18.373；0.813%/0.808% | 1,909/1,932；3.087/3.121；0.132%/0.132% |
| 采样或 MTP acceptance 的名称组 | 16/16；0.083/0.084；0.004%/0.004% | 913/924；4.923/5.029；0.210%/0.212% |

匹配方式为 SQLite `CUPTI_ACTIVITY_KIND_KERNEL` 的 `demangledName → shortName → mangledName`（首个能解析的 StringIds 名称），以 `end-start` 计时，并以有先后顺序的确切子串/名称规则放入互斥组。结果中保留所有 raw name。上表的“copy/index”是命名组合，**不是** PCIe/GPU memcpy：单列的 CUPTI GPU memcpy 为 prefill 164 次 / 11.849/11.848 ms、decode 2,742/2,775 次 / 1.567/1.589 ms；copyKind 代码未被解释为传输方向。`reshape_and_cache_flash_kernel` 是另一个 KV cache 写入组，prefill 约 2.05 ms，decode 约 4.73–4.79 ms，亦未混入 FP8 动态激活量化。采样组包括 argmax、resample/rejection 等名称，模块用途仍是名称推断。

## 候选顺序与可证明的上限

**第一步：定位仍独立的动态 FP8 量化能否和相邻算子合并。** `per_token_group_quant_8bit_register_kernel` 是两类请求都存在的独立 launch：prefill 每份 408 次，decode 每份超过 1.69 万次。若它们的 GPU duration 奇迹般全部消失且没有替代成本，捕获内 kernel-duration 总和最多下降 **1.565%–1.566%（prefill）/ 1.394%–1.395%（decode）**。这是严格按已观察时间算的*乐观算术上限*，不是预计提速；真实数据依赖、scale 计算、精度和布局可能使融合成本高于收益。先用算子级输入/输出关系确定哪一批独立量化接在未融合 norm、投影或其他消费者之后，再决定能否避免中间 BF16 写回。正常 graph 的核名与时间邻近本身不足以证明共享张量。

**第二步：只研究现有融合核与 FP8 GEMM 之间还剩下的中间读写。** `silu_and_mul_per_block_quant_kernel` 已把 SiLU、乘法、block-FP8 量化合在同一核，并非“漏做 SiLU+量化融合”；它的 prefill GPU 耗时约 **98 ms / 4.32%**，高于独立量化。若能在 `gate_up` GEMM 产出到 `down_proj` GEMM 消费的路径中进一步移除整个独立核，4.32% 才是其**零替代成本的捕获 kernel-duration 上限**；由于 gate/up 双分支激活、block scale 与下游 FP8 输入布局，不能预设一个普通 GEMM epilogue 就可实现，也不能把上限当成 API 收益。decode 对应仅约 **0.286%**，优化优先级明显低于 prefill。eager 实测 target `gate_up` 为 prefill M=16,160/2,741、decode M=5，N=34,816、K=5,120；`down_proj` 为 N=5,120、K=17,408。[实测 module+M/N/K/stride](../gemm-targeted/module-signatures.json)。

**避免重复已有融合。** Triton 名称显示 `fused_add_rms_norm_per_token_group_fp8_quant_permute` 已将 residual/RMSNorm、量化和 permute 合并；`_fused_qk_rmsnorm_rope_gate_kernel` 已合 QK norm/RoPE/gate；普通 residual/RMSNorm 也已有融合名称。把这些组再列成“待做的 norm+quant、RoPE+norm 融合”没有证据。独立 `layer_norm_fwd_kernel` 在 prefill 约 19.18 ms / 0.85%，但核名不足以确认它是否沿 FP8 MLP 路径；需先找到张量依赖和模块归属，不能直接视作可与 FP8 quant 合并。

**拷贝、采样暂排在后面。** 明确命名的 GPU copy/index 核在 prefill 约 0.81%、decode 约 0.13%；decode 的采样/MTP acceptance 约 0.21%。这些有减少 launch 的潜力，但正常 CUDA Graph 已批量 replay 众多 kernel，删除 GPU kernel 数不能等同减少同数量 CPU launch。必须先证实数据依赖、同步边界及并发安全，再测无 profiler 同输入 API。对 FP8 的四个名称组（独立量化、已融合 SiLU+量化、已融合 norm+量化、已融合 mul+量化）即使**全部 GPU 耗时归零**，prefill 的上限也只有 **7.523%–7.524%**、decode **2.238%–2.240%** 的*捕获 kernel-duration 总和*；其中大部分已经是融合实现，不是现成可回收的独立 launch 时间。

下一步只应在隔离实验中补张量依赖证据，并对单个候选做数值、不同 M/缓存状态及无 profiler 整模 A/B。上述上限不涵盖 CPU 调度、真实请求 wall time、性能波动或 BF16 `lm_head`；因此本页不承诺任何端到端收益，也不提出 FP4 更改。
