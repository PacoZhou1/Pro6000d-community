# FP8 gate/up GEMM 到激活量化的数据流与图边界

2026-10-14；只读固定源码、现有 eager 归因和既有编译证据。本文是实现接口调查，**没有运行候选 GPU 核，也没有证明 CUTLASS epilogue 可编译、等价或加速**。范围为 RTX PRO 6000（SM120）、固定镜像 `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b` 中的 vLLM `db9527a46873454610df6dbedf79a36d6bf1a7f6`、单卡 Qwen3.6-27B-FP8 + MTP。BF16 `lm_head` 和 embedding 不在改动范围。

## 真实链和布局

[固定 Qwen3.5 模型源码](../gemm-targeted/source/vllm/model_executor/models/qwen3_5.py)把文本 MLP 设为 `Qwen3NextMLP`；[Qwen3Next 固定源码](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/models/qwen3_next.py)表明它是 `Qwen2MoeMLP` 别名。[固定 `Qwen2MoeMLP.forward`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/models/qwen2_moe.py)为 `gate_up_proj(x)` → `act_fn(gate_up)` → `down_proj(out)`；该 Python 链中没有第二个 `gate_up` 消费者。Qwen3.5 的 `packed_modules_mapping` 把 checkpoint `gate_proj`、`up_proj` 装入 `gate_up_proj`，两半在输出维上先 gate 后 up；MTP 使用同种 decoder/MLP 结构。这个源码事实**不代替**实际 FX 图的单消费者检查。

[真实 eager 模块/核归因](../gemm-targeted/module-signatures.json)精确关联的主模型 `mlp.gate_up_proj` 输入/输出如下；`M=2741/16160` 出现在 prefill，`M=5` 出现在 decode，尺寸不能泛化到其他请求或配置。

| 张量 | dtype、shape、stride / 语义 |
|---|---|
| A、A scale | E4M3 `[M,5120]` 连续；FP32 `[M,40]` stride `(1,M)`，每 128 K 一块 |
| canonical B、B scale | E4M3 `[34816,5120]` stride `(5120,1)`；FP32 `[272,40]` stride `(40,1)`，每 128 N × 128 K 一块 |
| `gate_up` GEMM D | BF16 `[M,34816]` 连续；前 17408 列 gate，后 17408 列 up |
| SiLU×up 后量化 | E4M3 `[M,17408]` 连续；FP32 scale `[M,136]` stride `(1,M)`，每 128 输出列一组，作为 `down_proj` 输入 |

`Fp8LinearMethod.create_weights` 注册合并层 `weight` 与 `weight_scale_inv`；其 `process_weights_after_loading` 再调用后端 postprocess。[固定 CUTLASS 包装](../gemm-targeted/source/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py)把 B/B scale 转置交给 `cutlass_scaled_mm`。固定 [SM120 dispatch](../gemm-targeted/source/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh)的 accumulator 为 FP32，D 类型为 BF16，使用 `LinearCombination`，已有 TMA mainloop；此路径尚无目标 SiLU×up+per128 E4M3 输出量化 epilogue。[现有 activation-quant pass](../fp8-fusion/source/vllm/compilation/passes/fusion/act_quant_fusion.py)则把 `SiluAndMul` 与 `per_token_group_fp8_quant` 合成 `auto_functionalized` opaque op，保留量化的 FP32 scale 转置布局；它没有跨过 CUTLASS GEMM 消除 BF16 D。数值比较必须先把每个 gate/up FP32 accumulator **舍入为 BF16**，再执行原 SiLU、乘法、max/scale 和 E4M3 转换；直接从 FP32 accumulator 激活不是同一计算。

## 权重预排、生命周期与实验成本

独立 CUDA 核设计的前提是使每个 128 列输出组的 gate/up 在同一个 256N tile 内配对。对 `g∈[0,136)`，预排 B 的输出组 `2g` 复制 canonical gate 行 `[128g:128(g+1)]`，`2g+1` 复制 canonical up 行 `[17408+128g:17408+128(g+1)]`；B scale 的 272 个 N 组做**相同置换**，K 的 40 组顺序不变。预排只是逐行/scale 置换，绝不能重新量化 checkpoint。候选 ABI 暂定 `epilogue_gateup_fp8(out_fp8, out_scale_f32, A_fp8, B_packed_fp8, As_f32, Bs_packed_f32, M, workspace, workspace_bytes, stream)`；仅 `M>256,K=5120,N=34816`，输出布局如上。256N tile 和跨 subtile epilogue 的可行性仍由独立编译/数值实验判定，不能由 ABI 推断成功。

**本轮实验保留 canonical B/B scale，并另持有 packed 副本。**因此 `M<=256`、guard 不符或图未匹配时，仍走原 GEMM + 上轮 V2 融合激活量化路径，不必改小 M 核。预排应在模型权重载入、量化后处理完成后，且任何 warmup/编译/真实请求之前，一次性在启动阶段生成；不能在首请求懒生成。固定 vLLM [loader](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/model_loader/base_loader.py)顺序是 `load_weights` → `finalize_layerwise_processing`（适用时）→ `process_weights_after_loading` → `model.eval()`；[后处理遍历](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/model_loader/utils.py)逐模块运行 quant method，可能涉及 CPU offload/已预处理权重。因此 opt-in 只能在目标 `gate_up_proj` 的最终 runtime FP8 权重、scale 布局已验证后复制，并需验证权重重载、offload、TP、MTP 注册及重复初始化的幂等性；现阶段不指定未经验证的 Python hook。

单层副本 B 为 `34816×5120=178,257,920` B，scale 为 `272×40×4=43,520` B。按本模型 64 层主模型 + 1 层 MTP 的 65 份副本，合计 **11,589,593,600 B ≈10.79 GiB**，还未计分配器、预排临时缓冲、CUDA Graph 和 workspace。A 基线与 B 候选应**同样在启动期预排并持有副本**，A 不调用 packed 数据，以对齐常驻显存压力；两臂还须固定并记录同一 KV cache 容量及实际可用显存。固定 vLLM [CLI](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/engine/arg_utils.py#L1262-L1277)确有 `--kv-cache-memory-bytes`，其 [CacheConfig 语义](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/cache.py#L229-L236)为每 GPU 指定字节数且非空时忽略 `gpu_memory_utilization`；实际可用值和能否保留所需上下文仍须实验测定，不能仅把旧 0.45 比例转成假定值。单独报告约 10.79 GiB 的部署成本；实验公平性不等于生产免费。

## 编译图替换边界

首选在 vLLM 的 compilation/fusion pass 中、现有 `SiluMulBlockQuantPattern` 已产生量化 opaque op 后，识别 `cutlass_scaled_mm` 的 BF16 D（允许经可证明无数据拷贝/无语义变化的 `view`）→ `auto_functionalized(silu_and_mul_per_block_quant)` → `(FP8, scale)` 的实际 FX 子图，并替换为**一个**有 fake/meta 与 CUDA 实现的 opaque op。需要以实际导出图确认 pass 顺序和节点签名，不按 Python 层 MLP 名称直接绕过编译。matcher 应严格验证目标 layer 身份（主模型/MTP）、FP8 block backend、完整 dtype/shape/stride、无 bias、scale group=128/转置布局、B 与 packed 副本对应、BF16 D 单消费者、无别名/额外输出；`M>256` 才调用候选。任何不满足条件的图保持原链与 canonical B；优化命中失败要计数并显式报告，不能把回退性能写成候选收益。

这里存在一个必须实证的边界：FX 节点常只见算子和参数，未必保存 Python module prefix。仅靠 `M/N/K` 或 B 指针猜层身份不充分。需在固定图导出/生成 Python 中验证 B 参数身份可以映射到已预排的对应 layer，或引入显式且不会使旧路径错误的标签/缓冲；再检查每个目标 layer 的命中数量。现有 [V2 custom op 编译证据](../fp8-fusion/)只能说明激活量化 opaque op 能进入小 M compiled graph，**不能证明**新 GEMM→量化跨算子匹配或大 M 一定 graph replay；前轮大 prefill 是 direct launch。保留 MTP 与 CUDA Graph 小 M 回退，尤其不能因为替换而改动 `lm_head`。

## 理论流量及证据边界

若真正不物化 BF16 `[M,34816]`，每层省去该中间张量的一次**逻辑写+读**，即 `2×2×34816×M=139264M` B：M=16160 为 2,250,506,240 B（2.096 GiB）；M=2741 为 381,722,624 B（364.04 MiB）；M=5 为 696,320 B，但小 M 本方案回退，不省这部分。若某请求恰好每层各执行一次 M=16160 和 M=2741，65 层的算术总量是 171,094,876,160 B（159.35 GiB），**不是**已观测 DRAM、也不是 TTFT 预测。缓存、store policy、重算、额外 packed 权重读取、FP8/scale 输出和下游 `down_proj` 读都改变实际 HBM 交通量。应由目标 GEMM+量化**整段**的 NCU DRAM/L2 bytes 和无 profiler A1/B/A2 延迟交叉验证，而非拿逻辑字节直接乘带宽。

相邻的 `SiluAndMul+FP8 quant` 已有 vLLM pass 和上轮 V2 候选，不能再次宣称消掉一个原本独立的 SiLU launch。更容易的旁路机会是继续优化现有 FP8 激活量化核的线程组织，但它已经被上轮单核和整模测试约束；本方案的增量目标明确是**删除 GEMM BF16 中间 D 的物化**。要证明这个增量，需看到单个 fused CUDA launch、旧 GEMM/量化 launch 消失、没有全宽 BF16 中间分配，并对照原链/V2 的 FP8 字节和 FP32 scale；再做同显存/KV、同栈 A1/B/A2 的真实模型指标及质量。`design-review.md` 列有编译、资源、数值和实验门槛，本页不预判它们通过。

## 官方能力参照

[NVIDIA CUTLASS 自定义 epilogue 教程](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/001_gemm_with_fused_epilogue.html)展示了 epilogue 访客树和元素级运算的一般能力；[CUTLASS Blackwell 功能页](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)区分 SM120 与 SM100 的调度/硬件能力；[NVIDIA CUTLASS 3.9 发布讨论](https://github.com/NVIDIA/cutlass/discussions/2262)提及 SM120 epilogue 示例。它们不提供本镜像固定 CUTLASS 中“block FP8 GEMM + BF16 中间舍入 + gate/up 配对 SiLU + per128 E4M3/FP32 scale”的现成可替换核，也不能据 SM100 的 TMEM/tcgen05 路径推断 SM120 可用。实际实现须以固定 [提取源码清单](source-manifest.json)和本镜像编译/数值结果为准。
