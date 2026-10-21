# FP8 SiLU + block 量化核线程组织实验

状态：**V2 数值、单核、真实模型直接派发、A1/B/A2 API 与单核硬件计数验证完成；原服务已恢复并通过功能及后续稳定性检查**。日期：2026-10-14，Asia/Taipei。本轮只评估现有 FP8 路径的一个核，不做 FP4/NVFP4 故障定位，不改 BF16 `lm_head`、模型权重或 embedding 服务。

## 固定环境与已知热点

目标硬件为 RTX PRO 6000 Blackwell Workstation Edition 96 GB、SM120。固定实验镜像 digest 为 `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`，vLLM 0.31.0 源码提交为 `db9527a46873454610df6dbedf79a36d6bf1a7f6`。本机 Tensor Core 已运行 CUTLASS FP8 GEMM；此候选优化其前面的 CUDA 数据准备，不是开启新的 Tensor Core。NVDEC/NVENC 是视频编解码器，不负责文本 token decode；SM120 不能直接套用 SM100 专属 Tensor Memory/`tcgen05` 路径。[硬件适用边界](evidence/fp8-fusion/hardware-capabilities.md)、[固定源码清单](evidence/gemm-targeted/source-manifest.json)。

正常 CUDA Graph 的两份长 prefill Nsight 捕获中，原始名称 `silu_and_mul_per_block_quant_kernel` 各 136 次、GPU duration 合计 97.827/98.173 ms，分别占捕获内**全部 GPU kernel duration 求和**的 4.316%/4.319%。对应两份 steady decode 分别 5,644/5,712 次、6.707/6.785 ms、0.286%/0.286%。这些是核名称推断的组、各捕获独立的时间求和；不是 wall time、GPU busy union 或整模加速比例，也不把 eager 模块归因直接移植到正常 Graph。[原始输入哈希、名称规则与分母](evidence/fp8-fusion/profile-opportunities.json)。

原 vLLM 核已将 SiLU、gate/up 乘法和每 128 元素 FP8 block 量化放在同一 kernel。本轮最终选中 **V2 group-major/group-fast CTA 顺序**：每个 128 列组由一个 warp 处理，以 warp 归约计算 scale，调整线程组织与同步；它**没有新增 SiLU 与量化之间的融合**。选中源是[fused_silu_quant_group_major.cu](evidence/fp8-fusion/fused_silu_quant_group_major.cu)，库 SHA-256 为 `087b7caa92c4742ece3875b6dd1421e25360b834ce4106f7fa166f90bc933b3f`；V1 `fused_silu_quant.cu` 只作已测对照。[复现清单](evidence/fp8-fusion/reproduction-manifest.json)。隔离运行时守卫只对 SM120、连续 BF16 输入 `[M,34816]`、`M≥256`、FP8 输出 `[M,17408]`、FP32 scales `[M,136]` 且 stride `(1,M)`、group size 128、无 `scale_ub` 的调用选候选。其他调用回退官方 `_C.silu_and_mul_per_block_quant`，因此小 M decode 是正确性护栏而非预期收益点。[运行时守卫](evidence/fp8-fusion/runtime/sitecustomize.py)。

独立的 `per_token_group_quant_8bit_register_kernel` 是另一组热点，prefill 的占比为 1.566%/1.565%，decode 为 1.394%/1.395%；本轮候选不修改它。已有 residual/RMSNorm/量化等融合也不是本轮新功能。[机会边界](evidence/fp8-fusion/profile-opportunities.md)。

## 已完成的数值、单核和真实模型派发

V1 与 V2 在固定镜像 SM120 上各通过 12 组输入的 FP8 输出**逐字节**和 FP32 scale **逐 bit**对照；覆盖实际尺寸、`M<256` 回退及 NaN/Inf/零/小值/有限溢出。五个 M 的独立单核 CUDA Graph 在捕获后更换输入，结果与重新执行的官方核一致。V2 的 14 项运行时/FX/Inductor 守卫检查也通过，包括先执行 M5 再执行 M256，目标 M256 未落回官方 op。这证明受测算子的数值与独立图兼容性，不能推广为所有输入的形式证明。[数值与单核记录](evidence/fp8-fusion/micro-results.md)、[原始与汇总](evidence/fp8-fusion/micro-comparison.json)。

V2 合成单核图重放中，M=2741 的 warm/cold 中位分别为 **146.72/176.13 µs**，同次官方 **206.91/228.32 µs**；M=16160 为 **894.59/917.50 µs**，官方 **1223.07/1255.42 µs**。每项 31 次，减少约 29.1%/22.9% 与 26.9%/26.9%。这些是单核、合成输入和 256 MiB 缓存冲刷近似；不能称为整模 TTFT 收益。[micro-v2 原始记录](evidence/fp8-fusion/micro-v2.jsonl)。

真实 FP8 模型、MTP4、FP8 KV、FlashInfer 和 0.70 显存预算的长 prefill Nsight 诊断中，V2 候选核 **130 次**（M=16160 与 2741 各 65 次），另有 6 次 M1 使用官方回退。130 次全部匹配成功的**直接 CUDA launch**，**0 次具有 graphNodeId**。这已明确该配置下大 M prefill 的实际执行方式是直接发射；**整模大 M Graph replay 不适用，不把它留作人为验收债**。独立单核 Graph 兼容性只是另一项测试。该捕获候选核 duration 合计 66.858301 ms、全部 3673 核合计 2283.011888 ms，均为单次诊断捕获的求和，不用于 A/B API 改善归因。[派发核对](evidence/fp8-fusion/integration-results.md)、[correlation 证据](evidence/fp8-fusion/witness/dispatch-proof.json)。B 臂的编译缓存有 **21 个**生成 Python 文件引用新 op，是编译保留证据；实际发射仍由上述 Nsight 证明。[编译缓存审计](evidence/fp8-fusion/b-cache-audit.json)。

## 无 profiler API：A1/B/A2

同一固定 payload 构造器按 index 配对；每臂 12 个约 19K 输入/1 输出的 prefill 请求和 6 个约 4.8K 输入/1024 输出的普通 decode 请求，另有每类 1 次 warmup，测量请求均 valid。A1/B/A2 同 index payload SHA 全同；A1–A2 与 A1–B 的 prefill **12/12**、decode **6/6 非空正文 hash 相同**。两类 reasoning hash 均为空串 hash，不能算“相同推理文本”的证据。三臂质量合成检查各 **5/5**，开放输出各 **4/4** 且全部正常 `stop`、有最终正文，共 27/27；其中短算术可正常结束，**没有测思考 decode 速率或广泛业务质量**。54 个测量请求的 embedding 前后累计计数未见活动，prefix cache hits 均为 0；前后快照无法排除两次读取之间瞬时在途活动。[三臂逐样本、输入哈希及计数](evidence/fp8-fusion/api-comparison.json)。

| 无 profiler 指标（每臂中位数） | 官方 A1 | 候选 B | 官方 A2 | 当前解释 |
|---|---:|---:|---:|---|
| 长 prefill 首文本时间，12 请求 | 2.268213 s | 2.244258 s | 2.284222 s | 逐 index B 对 `(A1+A2)/2` 的 TTFT 减少中位 **1.4144%**，12 对范围 1.3697%–1.5054%；A1–A2 自身中位漂移约 0.706%。 |
| 服务端 `request_prefill_time_seconds_sum`，12 请求 | 2.251270 s | 2.228187 s | 2.268387 s | 各窗口增量的中位数，与客户端 TTFT 是不同计时口径。 |
| 普通 decode 有效输出速率，6 请求 | 106.735803 tok/s | 106.731730 tok/s | 106.705445 tok/s | 基本不变；小 M 原核回退。 |
| decode accepted/draft 中位 | 2.059638 | 2.059638 | 2.059638 | 按各窗口计数相除后取中位；不把它称为完整质量证明。 |
| 合成质量检查 | 5/5 | 5/5 | 5/5 | 仅固定合成题。 |
| 开放输出正常结束、有最终正文 | 4/4 | 4/4 | 4/4 | 包含短思考算术；不代表长思考或 MoBase 业务质量。 |
| 同 index 非空正文 hash | 对照基准 | prefill 12/12、decode 6/6 与 A1 相同 | prefill 12/12、decode 6/6 与 A1 相同 | 三臂 reasoning 均为空，不能算非空推理一致。 |

配对百分比按每个 index 的 `100 × (((A1+A2)/2−B) / ((A1+A2)/2))` 计算，再取 12 对的中位数；它并非三臂中位数直接相除。[离线分析器](evidence/fp8-fusion/analyze_api.py)已逐 index 核验输入、目标服务窗口及 embedding 指标，并保留各臂 min/max、所有请求、warmup 排除及计数。A1/B/A2 是**一次顺序时段**，A1–A2 已有约 0.706% 漂移；描述性 bootstrap 只重采样 prompt index，没有独立重复时段的置信度。小幅 prefill 改善可称这次受控测试的候选信号，不能宣称跨时段稳定或生产收益。

## 单核硬件计数器：机制线索

合成 M=16160 单核 NCU 计数中，官方/V2 的 `gpu__time_duration.sum` 为 **1.225792/0.858336 ms**，DRAM 吞吐占峰值持续值 **66.379%/93.516%**；active-warp 的 barrier stall 比例 **13.6408%/0%**。V2 更少等待归约同步，且该受测核更接近带宽限制。NCU 只覆盖每版 **一个选定 launch、各 9 次 replay pass**，缓存和时钟未固定；stall 百分比不是整核 wall-time 份额，这些计数**只用于机制诊断，不是 API benchmark**。[计数汇总与源哈希](evidence/fp8-fusion/counter-summary.json)、[原始 CSV](evidence/fp8-fusion/quant-counters.csv)。进一步减少上游 GEMM 的 BF16 中间写读或研究 epilogue 融合只能列为未来张量依赖假设；当前没有可行性或整模收益证据。

## 尚待证明与采用决定

- **大 M 执行方式已厘清。** 本配置真实模型长 prefill 是直接 CUDA launch；不再要求一个本路径不使用的整模大 M Graph replay 作为验收条件。
- **跨时段、原生产预算和真实业务：尚未验证。** 现有 A/B/A 只有一次顺序时段、0.70 实验预算；原 0.45 四服务共存、并发、MoBase 长业务与长思考质量未测。
- **服务恢复：已通过。** 原 27B、9B、ASR 使用同一容器、镜像和命令恢复，四接口实际功能检查及 316.63 秒后的稳定性快照通过。embedding 全程未停，74 个监测点身份、启动时间和重启计数不变且健康 200；测试容器无残留。[恢复证据](evidence/fp8-fusion/restoration-results.md)。

采用决定：**保留 V2 作为这次受控实验有小幅 prefill 收益的候选，不覆盖生产**。跨时段、原 0.45 预算四服务共存及更广 MoBase/质量测试尚未完成，不足以支持生产采纳。既有 Tensor Core FP8 GEMM、BF16 head 与独立动态量化核不在本候选改动范围内。

独立 Sol 审查已从原始记录重算关键统计并核对源码、编译产物及恢复证据；唯一发现的工具帮助措辞已修正，实测版源码哈希仍保留。[独立审查记录](evidence/fp8-fusion/final-review.md)。
