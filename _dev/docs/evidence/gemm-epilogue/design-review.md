# FP8 gate/up GEMM → 量化：独立设计审查（实现前）

审查时间：2026-10-14。只读固定源码与前轮实测；未连接服务器、编译候选或运行 GPU。本页列出可证伪的实现门槛，**不代表融合已经可编译、数值正确或加速**。范围限 RTX PRO 6000 SM120、固定 vLLM 0.31 镜像中的 FP8 路径；FP8 权重及 BF16 `lm_head` 不变，embedding 持续运行。

## 固定路径与尺寸

- [真实 eager 模块/核关联](../gemm-targeted/module-signatures.json)记录主模型 `mlp.gate_up_proj` 的 FP8 block CUTLASS GEMM 为 `M=2741/16160, N=34816, K=5120`，A 为 FP8 `[M,5120]`、scale FP32 `[M,40]` stride `(1,M)`，权重 B 为 FP8 `[34816,5120]`、block scale FP32 `[272,40]`，输出 BF16 `[M,34816]`。`mlp.down_proj` 接收 FP8 `[M,17408]`、scale FP32 `[M,136]` stride `(1,M)`，其权重 scale 为 `[40,136]`；decode 小 M 另有路径。记录是 eager 归因，不自动证明正常编译图里的张量唯一消费者。
- [Qwen3.5 固定模型](../gemm-targeted/source/vllm/model_executor/models/qwen3_5.py)将文本模型 MLP 选为 `Qwen3NextMLP`，并把 `gate_proj/up_proj` 打包为 `gate_up_proj`；[固定 CUTLASS 包装](../gemm-targeted/source/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py)把 B 转置传给 `cutlass_scaled_mm`，同时传递 A/B scale。不能把两半当作独立无 scale 的 BF16 GEMM。
- [固定 SM120 CUTLASS](../gemm-targeted/source/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh)的 M>256 默认 tile 为 `128×128×128`，A/B block scale 粒度 `(1,128,128)`；`LinearCombination` 将 FP32 accumulator 以 round-to-nearest 写为 BF16 D。[原量化核](../fp8-fusion/source/upstream/fused_silu_mul_block_quant.cu)从 `[gate(17408) || up(17408)]` 的 BF16 输入取同 row、同 128 列组，先转 FP32 算 SiLU/乘法，再求组内最大值、计算 FP32 scale，写 E4M3 输出及可转置的 scale。原 [编译 pass](../fp8-fusion/source/vllm/compilation/passes/fusion/act_quant_fusion.py)只把 SiLU/乘法与后续量化合成一个 op；它不跨越前面的 `cutlass_scaled_mm` opaque op 消除 GEMM 输出。

## 必须先解决的设计问题

| 风险 | 为什么当前方案不能直接假定成立 | 必要证据/通过条件 |
|---|---|---|
| **P0：gate/up 两半跨 CTA** | 原 N=34816 的 gate 与 up 对应列相距 17408，128N CTA 只拥有其中一半；量化的每个 FP8 group 又需要同 row 的 128 对乘积及一次组内 scale。两个 CTA 没有可依赖的普通 epilogue 同步。 | 若采用离线按 `[gate128,up128]` 交错重排权重及 B scale、改 256N tile，给出每个 `(row,group)` 唯一 CTA 和 128 对元素映射，证明 CUTLASS 实际 epilogue 分片同时可取得两半；不能仅因 CTA tileN=256 就假定 `EpilogueTileAuto` 的寄存器/共享内存片段保留两半。若需额外全局暂存/第二 kernel，须改称分阶段实现并重新计算流量。 |
| **P0：block scaling 与 BF16 舍入** | B 的 `[272,40]` 每 128N、128K 一块，重排 N 的同时必须逐块同步重排 B scale，且不改变 A scale、K 分组和原 weight checkpoint。原 GEMM 先写 BF16；SiLU 从 BF16 的 gate/up 而不是未舍入 FP32 accumulator 读取。 | 在原权重与重排权重上逐元素核对 GEMM BF16 逻辑输出（反置换后）；再核对最终 FP8 **字节**、FP32 scale **位模式**与官方链和前轮 V2 的对应输出。测试实际 M=2741/16160、边界 255/256/257、小 M、尾块、零/NaN/±Inf/有限溢出，并显式记录 `expf`、量化饱和、最小 scale 和转置 scale 地址。若 tile 变更导致 BF16 不逐位一致，应先定位 K 累加/scale/舍入差异，再评估是否允许任何质量门槛调整，不能悄悄放宽。 |
| **P0：是否真的删掉中间张量** | 只把官方 GEMM 与前轮 V2 量化包进一个 Python/custom op，若仍物化 BF16 `[M,34816]`，就没有消除目标读写。[现有 pass](../fp8-fusion/source/vllm/compilation/passes/fusion/act_quant_fusion.py)不会自动跨 opaque CUTLASS op 把它删除。正常编译图也可能有额外消费者、别名或 debug/LoRA 分支。 | 对目标模型实际 normal compiled FX/生成代码检查 GEMM 输出的使用者、别名和活跃期：只允许预期的激活/量化消费者；证明候选不分配/写入该全量 BF16 中间 D，同时确有 FP8 `[M,17408]` 与 scale `[M,136]` 输入 `down_proj`。有其他 consumer 时 guard 回退或保留对应输出，不能断开语义。 |
| **P1：CUTLASS 输出布局/资源可行性** | 标准调度器以 `N=34816`、原 D stride 建立 epilogue；目标只写一半宽度的 FP8 D 和独立 FP32 scale，不能直接把原 `ElementD=BF16` 换成 FP8 指针。256N 配对会改变寄存器、shared memory、TMA 描述与 occupancy，可能无法编译或更慢。 | 在固定 CUDA/CUTLASS 源编译出独立符号；实测资源占用、launch 状态和同形状单核耗时。明确新 D/scale 的 TMA/stride、每个 group 恰好一次写、不同 CTA 无重叠；CUDA Graph capture/replay 换输入及多 stream 并发不依赖全局可变 scratch。编译失败或资源退化须如实记录。 |
| **P1：只命中目标层** | `M,N,K` 形状、scale stride 和硬件架构不足以证明调用一定是目标 MLP；小 M、其他 layer 或模型可能撞形状。 | 以目标模块/量化方法/完整 dtype、shape、stride、group size、scale 布局、无 bias 等 guard 加 opt-in；在正常 `torch.compile` 图与真实模型捕获里确认目标命中、非目标和小 M 官方回退，检查 fake/meta、auto_functionalized、CUDA Graph 生命周期和流安全。保持原 9B/ASR/embedding 代码路径不变。 |

## 流量与速度怎样证明

M=16160 时完整 BF16 `[M,34816]` 是 **1,125,253,120 字节**；一次写加一次读的**逻辑**交通量是约 2.2505 GB（2.096 GiB）。M=2741 时对应约 190.9 MB，读写合计约 381.7 MB。缓存命中、TMA 写法和权重流会改变真实 DRAM 字节；不能把这些逻辑量直接写成节省的 HBM 流量。FP8 输出、FP32 scale、权重、A/B scale 与下游 `down_proj` 的读写仍存在；若逐请求重排权重或复制中间缓冲，也可能抵消收益。权重/B scale 应尽可能在模型装载时一次性生成并记录额外显存及启动成本，若做不到须纳入正式 A/B。

先以同 M/K/N、同 FP8 输入与权重做官方链、前轮 V2、真正 fused epilogue 三方单核对照；warm/cold 分开，固定样本数、缓存处理和时钟口径。NCU 至少同时比较目标核与该 GEMM→量化**整段**的 `dram__bytes_read.sum`、`dram__bytes_write.sum`、L2/SM 指标、kernel 时长和 launch 数，说明重放次数与分母；单个 kernel 的字节下降不等于整段下降。Nsight Systems 要把候选 launch 与真实模型模块关联，并确认旧 BF16 物化核/量化核是否消失。最终使用**无 profiler**、逐索引同 payload 的 A1（官方链）/B（候选）/A2（官方链）及如需的 V2 同栈臂，测长 prefill TTFT、服务端 prefill、普通 decode/思考输出速率、prefix cache 和 accepted/draft；逐请求保存原始指标、内容与非空 reasoning hash、正常结束的合成/业务质量及 embedding 前后计数。若质量、数值或整模速度未过门槛，保留负结果，不把单核改善称整模收益。

服务实验仅由 root 在隔离窗口执行：embedding 不停；按用户最新要求，原 27B/9B/ASR 保持停用，直到用户明确要求启动，不在本轮结束时自动恢复。结束时应记录这三项的停用状态、embedding 身份/启动/重启/健康持续观测及临时容器清理。本设计审查只提出验证点，不代表候选的数值或整模实验已经通过。
