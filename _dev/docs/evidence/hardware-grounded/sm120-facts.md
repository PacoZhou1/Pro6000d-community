# RTX PRO 6000 Workstation SM120：FP8 GEMM 的硬件事实与预算

核查日期：2026-10-14。目标是本机 **RTX PRO 6000 Blackwell Workstation Edition，GB202，CC 12.0 / SM120，96GB、600W、188 个 SM**。本文只整理 NVIDIA、CUDA、CUTLASS 一手资料、读取本轮已有设备属性，以及做 CPU 算术推导；没有新增 GPU 测试。当前项目仅优化 FP8，BF16 `lm_head` 保持原样，embedding 继续运行。

规格、合法指令、资源预算必须先于 tile / pipeline / epilogue 选择。删除一次 pack 或 BF16 中间写读，只减少某部分逻辑流量；它不能保证 MMA 利用率、寄存器压力、缓存重复读取或全链路时间改善。实际 NCU 与 A/B/A 结果由[主报告](../../13-sm120-hardware-roadmap.md)记录，本文不把规格估算写成实测。

## 1. 型号与峰值口径

[NVIDIA CUDA GPU 列表](https://developer.nvidia.com/cuda/gpus)将 RTX PRO 6000 Workstation 列为 **12.0**，B200 / GB200 为 **10.0**，B300 / GB300 为 **10.3**。这些都叫 Blackwell，但架构专属指令和资源不能互换。

[RTX PRO Blackwell Architecture v1.1](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf) Appendix A、Table 4，印刷页 46–48（PDF 页索引 45–47），最右列才是此处的 Workstation SKU。下表保留其原始精度与累加口径：

| 项目 | Workstation 规格 | 用途与限制 |
|---|---:|---|
| 启用 SM / Tensor Core | 188 / 752 | 不能用完整 GB202 方框图的 192 SM 计算本卡并行度 |
| Boost clock | 2617 MHz | 峰值基于 Boost；持续时钟未知 |
| FP8 Tensor、FP32 accumulate | **1007.6 / 2015.2 TFLOP/s** | 前者 dense；后者启用 sparsity 的有效峰值 |
| FP8 Tensor、FP16 accumulate | 1007.6 / 2015.2 TFLOP/s | 此 SKU 公布的 FP32 累加峰值没有再减半 |
| BF16 Tensor、FP32 accumulate | 503.8 / 1007.6 TFLOP/s | BF16 head 应用 BF16 口径 |
| FP4 Tensor、FP32 accumulate | 2015.2 / 4030.4 TFLOP/s | 仅解释广告数字，不是本项目计算能力 |
| L2 | 131072 KB | 本轮设备属性确认为 134217728 B，即 128 MiB |

原表标签含 “Peak FP8 Tensor TFLOPS with FP32 Accumulate”；脚注说明峰值按 Boost 计算、斜线后数值使用 sparsity。dense FP8 roofline 因而应取 **1007.6×10¹² FLOP/s**，FMA 按两个 FLOP 计数。不能取 2015.2，更不能取约 4000 AI TOPS。任意零值不等于硬件要求的结构化稀疏布局及 metadata；当前权重和 GEMM 没有稀疏转换。

[Workstation 官方 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf)第 3 页确认 **96GB GDDR7、512-bit、1792 GB/s、600W**，其 AI TOPS 脚注明确属于 FP4 sparsity。1792 GB/s = **1.792 TB/s**，是十进制理论带宽；不是模型可持续带宽，也不是 HBM 带宽。不要误取 Max-Q 的 300W / 2280MHz 列。

## 2. 本机可用资源与官方文档冲突

本轮[设备属性原始来源摘要](device-attributes.json)来自已完成的 NCU capture，目标型号、CC 12.0、188 SM 已实际确认。资源设计以这些实际属性为主，并与 [CUDA Programming Guide 当前 Tables 30–32](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications)交叉核对。表中的 K / KB 均为 1024 单位。

| 限制 | SM120 本机 / 文档 | SM100 / SM103 的区别 |
|---|---:|---|
| warp 大小 | 32 threads | 同样 32 |
| 每 SM 常驻 warp / thread 上限 | **48 / 1536** | 64 / 2048 |
| 每 SM 常驻 CTA 上限 | **24** | 32 |
| 每 CTA thread 上限 | 1024 | 同样 1024 |
| 每 SM 32-bit registers | **65536 = 256 KiB** | 同样 64K，不能因 datacenter 型号便假定翻倍 |
| 每 CTA / 每 thread registers 上限 | 65536 / 255 | 本机 255 亦见上一轮 NCU 属性；分配粒度仍影响 occupancy |
| 每 SM 可分配 shared memory | **102400 B = 100 KiB** | 228 KiB |
| 每 CTA opt-in shared memory | **101376 B = 99 KiB** | 227 KiB |
| unified L1 / shared 容量 | 128 KiB | 256 KiB；统一池不是全部可分配 SMEM |

超过 48 KiB 的 CTA shared allocation 需动态 shared memory 和显式 opt-in，例如 `cudaFuncAttributeMaxDynamicSharedMemorySize`。SM120 可选 shared carveout 为 0、8、16、32、64、100 KiB；还要计入 CUTLASS 的 pipeline、barrier、scale 与 epilogue 存储，不能把 128 KiB 统一池当成 128 KiB CTA scratch。

**冲突保留：**[CUDA 13.0 Programming Guide 的技术规格表](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#features-and-technical-specifications)给出 24 CTA / SM、100 KiB shared / SM、99 KiB / CTA；[CUDA 13.0 Blackwell Tuning Guide §198.51.100.10](https://docs.nvidia.com/cuda/archive/13.0.0/blackwell-tuning-guide/index.html#occupancy)却写 CC12.0 为 32 CTA / SM、128KB shared / SM，同时仍写 99KB / CTA。本文没有把两个口径拼接；本机 actual attributes 与 Programming Guide 表一致。官方冲突的修订原因未确认。

## 3. SM120 合法 MMA、TMA 与不可移植的路径

| 能力 | 已确认的口径 | 本项目约束 |
|---|---|---|
| FP8 Tensor MMA | SM120 使用 warp-level `mma.sync`，A/B 与 FP32 accumulator 的 fragment 在 registers | 不是 SM100 的 TMEM accumulator；扩大 tile 会扩大寄存器存活集合 |
| E4M3 × E4M3 → FP32 | CUTLASS 的 `SM120_16x8x32_TN<…,float>` 包装 `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32` | A 为 M×K row-major；canonical B[N,K] row-major 等价于 MMA 视角的 K×N column-major |
| FP32 累加精度 | PTX 指定至少 single precision；累加顺序、rounding、subnormal 处理未指定 | FP32 accumulator 不保证改 tile、K 分组、scale 乘法结合顺序后逐位相同 |
| TMA / async copy | CC12.x 支持 TMA、硬件 `memcpy_async`、异步 barrier | TMA 是搬运能力，不意味着有异步 `tcgen05` Tensor MMA |
| TMEM / `tcgen05.mma` | PTX 的目标名单为 SM100/相关 10.x family 与 SM110 family，没有 SM120 | 无受支持的 SM120 `tcgen05` / TMEM accumulator 编程路径；此结论不推断物理晶体管是否存在 |
| Hopper `wgmma.mma_async` | PTX 标为 `sm_90a` 专属 | CUTLASS 的“4/8 MMA warps”调度名称不能当作 WGMMA 指令支持 |

依据：[CUTLASS 4.2.1 SM120 功能说明](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)、[官方 SM120 MMA 包装源码](https://raw.githubusercontent.com/NVIDIA/cutlass/v4.2.1/include/cute/arch/mma_sm120.hpp)及 [PTX ISA 9.0：mma](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)、[tcgen05.mma](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tcgen05-mma)、[wgmma.mma_async](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)。普通 E4M3/E5M2 MMA 的早期路径要求 SM89+；新 `.kind` / `.block_scale` 路径需相应 SM120a 或 SM120f family target。本项目使用 `sm_120a`。`compute_100f` 和 `compute_120f` 属于不同 family，不能互换。

CUTLASS 官方公开实现的 Pingpong 是两组各 4 个 MMA warp、工作于不同输出 tile；Cooperative 是同一 tile 的 8 个 MMA warp。它们是软件调度方式。公开版本只作为架构能力和预算依据，不能自动替代本项目镜像里锁定的 CUTLASS 源码。[SM120 blockwise builder](https://raw.githubusercontent.com/NVIDIA/cutlass/v4.2.1/include/cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl)也明确把 epilogue carveout 与 pipeline storage 从 stage 可用容量扣除。

原生 MX block-scaled MMA 的 scale 编码、粒度与布局有自己的要求，例如 UE8M0。当前项目 **FP8 + 任意 FP32 scale、每 128 元素分组**属于既定语义，不能把它直接等同于硬件 MXFP8，也不能假定改成原生 scale 编码仍 bitexact。FP32 software rescale、BF16 round、SiLU、max reduction、scale、FP8 convert 都需要单独预算，Tensor TFLOP/s 不给出这些指令的吞吐保证。

### TMA 对齐与 cluster 的边界

[CUDA 13.0 Driver API：`cuTensorMapEncodeTiled`](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)要求 global base 至少 16-byte 对齐，global strides 是 16-byte 的倍数；无 interleave 时 box 最内层字节数也须为 16 的倍数。这里 K=5120 的 FP8 行 stride 满足这一必要条件；scale `[M,40]` 的 column-major stride `(1,M)` 若直接编码为二维 FP32 tensor，跨列 stride 为 `4M`，M2741 时为 **10964 B**，不满足 16-byte stride 条件。可考虑其他映射、padding 或普通 load，但这不是“scale 不能用任何 TMA”的证明。异步 copy 还必须遵守 completion barrier / fence 协议。

**cluster 与 multicast 不等同：**本轮 actual `cluster_launch=1`、`multicast_supported=0`，记录本身已声明后者不是所有 TMA multicast 路径的判定。CUDA feature table 把 CC12.x 的 cluster / DSM / TMA 列为支持；CUTLASS 4.2.1 SM120 文档则针对 **GeForce** 写无 multicast、cluster 固定 1×1×1。因此不能据 GeForce 描述断言 RTX PRO 完全不支持 cluster，也不能据 generic cluster 属性套用 B200 的 CTA-pair / 高效 multicast GEMM。当前工程使用单 CTA 路径；RTX PRO 上具体 TMA multicast 指令的合法性、性能及多 CTA 调度收益，本文未做新验证。

## 4. 对本项目形状的直接计算

以下是 **CPU 算术推导**。FMA 计 2 FLOP，FP8 dense 峰值 `P=1007.6e12`，理论带宽 `BW=1792e9 B/s`，其交点 `P/BW=562.28 FLOP/B`。BF16 Tensor 对应交点为 281.14 FLOP/B。所谓单遍逻辑流量采用 `s×(MK+NK)+2MN`，FP8 输入 `s=1`、BF16 输入 `s=2`、输出 BF16；省略 scale、重复读取、缓存、激活、量化、pack 与 spill。它是筛选设计方向的理想模型，**不是实测 DRAM 字节数，也不是所有缓存状态下的严格 DRAM 下界**。

| GEMM | M | 计算量 GFLOP | 单遍逻辑算术强度 FLOP/B | `2MNK/P` 理想计算时间 ms |
|---|---:|---:|---:|---:|
| gate/up N34816 K5120，FP8 | 1 | 0.3565 | 2.00 | 0.000354 |
| 同上 | 5 | 1.7826 | 9.98 | 0.001769 |
| 同上 | 2741 | 977.2099 | 2550.44 | 0.969839 |
| 同上 | 16160 | 5761.2960 | 4156.03 | 5.717840 |
| down N5120 K17408，FP8 | 1 | 0.1783 | 2.00 | 0.000177 |
| 同上 | 5 | 0.8913 | 9.98 | 0.000885 |
| 同上 | 2741 | 488.6050 | 2962.82 | 0.484920 |
| 同上 | 16160 | 2880.6480 | 5375.14 | 2.858920 |
| lm_head N248320 K5120，BF16 | 1 | 2.5428 | 1.00 | 0.005047 |
| 同上 | 5 | 12.7140 | 5.00 | 0.025236 |

head 形状与 dtype 可核对现有 [BF16 probe](../gemm-targeted/bf16_counter_probe.py)。这里不把 prefill 的全部 M2741/16160 自动套到只输出选定 token logits 的 head；也不提出改变 BF16 head 的方案。

| 权重 | 不含 scale 字节数 | 与 128 MiB L2 比较 | 假定完整从 DRAM 读一次、只按理论 BW 的时间 |
|---|---:|---|---:|
| gate/up FP8 | 178257920 B = 170 MiB | 大于 L2 | 0.099474 ms |
| down FP8 | 89128960 B = 85 MiB | 小于 L2，但与全链路数据竞争 | 0.049737 ms |
| head BF16 | 2542796800 B ≈ 2.368 GiB | 远大于 L2 | 1.418971 ms |

**推导含义：**M1/5 的 FP8 GEMM 远低于理想交点，优先检查权重读取、launch、M 尾部利用率与 SM 覆盖；大 M 的理想强度在交点之上，tile / pipeline / accumulator / software rescale 应先检验，不能仅因减少 epilogue 字节就认定受益。实际重复读取可能显著改变 roofline 分类。head 的小 M 依然要遍历大权重，gate/up 的改善不能自动成为 decode 全链路改善。

### CTA 覆盖

假定无 split-K / Stream-K、一个 CTA 一个输出 tile，CTA 数为 `ceil(M/TM)×ceil(N/TN)`。这是形状推导，不是核函数实际调度证据。

| gate/up tile TM×TN | M1 / M5 | M2741 | M16160 |
|---|---:|---:|---:|
| 128×128 | 272 | 5984 | 34544 |
| 64×256 | 136 | 5848 | 34408 |
| 32×256 | 136 | 11696 | 68680 |

N256 小 M 只有 136 CTA，少于 188 SM；M 尾部也会造成大量无效 tile 行。down 若 Ntile128，小 M 只有 40 CTA；Ntile64 也只有 80。head Ntile128 为 1940 CTA，不能把它与 down 的欠覆盖混为一谈。以上说明 M1/5 和 M2741/16160 需要不同的设计判断。

### Shared memory 与 registers 预算

Ktile128、两级 pipeline，**仅 FP8 A/B operands**需要 `2×(TM+TN)×128` B：

| tile | operand SMEM | 距 99 KiB CTA 上限的剩余空间 |
|---|---:|---:|
| 128×128 | 64 KiB | 35 KiB |
| 64×256 | 80 KiB | 19 KiB |
| 128×256 | 96 KiB | 3 KiB |

这还没有加 scale、padding、barrier、pipeline bookkeeping 或 epilogue scratch。合法 stage count 必须按实际 `SharedStorage` 和 epilogue carveout 计算；不能因理论 operand 总数未超上限就认定可编译或能双 CTA 常驻。

64×256 tile 有 16384 个 FP32 accumulator。若平均分给 4 个 MMA warp，为每线程 128 个 accumulator；8 个 MMA warp 则为 64 个。若 blockwise scaling 使 final accumulator 与临时 block accumulator **同时完整存活**，前者将达到 `128+128=256`，已经超过每线程 255 上限，尚未计算 operand、scale、地址与 SiLU 变量。这是条件资源预算，不能代替编译器 live range / 寄存器报告，但能提前排除无条件扩大 tile 的假设。

更少 spill 也不等于更高性能：例如 384 threads × 168 registers ≈ 64512 registers，接近 65536 / SM，按均匀分配估算只能容纳一个 CTA。分配粒度、warp specialization 的非均匀寄存器策略和 shared 限制还需编译/occupancy 实测；低 occupancy 本身也不是 Tensor pipeline 不饱和的证明。SiLU、归约、scale 和转换消耗普通指令与同步资源，不能由 dense FP8 峰值折算成“免费 epilogue”。

## 5. 确认、未知与后续证据要求

- **确认：**本机 SM120 / 188 SM / 128 MiB L2、100 KiB shared / SM、99 KiB / CTA、24 CTA / SM，以及官方 dense FP8 FP32-accumulate 峰值口径；合法核心路径是寄存器 `mma.sync` + 合规异步搬运。
- **未确认：**模型状态下可持续 Tensor 峰值、DRAM / L2 带宽、SiLU / reduction 吞吐、不同 scale 布局的搬运效率、具体 TMA multicast 路径与收益。当前文档没有执行这些 GPU probe。
- **不可由规格推出：**改变 tile 后 full-bit 等价、消 spill 后加速、删 pack 后加速、降低 DRAM 字节后整模加速。都要保持原 FP8 bytes / scale bits / 下游 GEMM 及实际全链路验收。
- **版本边界：**CUDA 13.0 archive 用于与项目工具链对照，当前 Programming Guide 用于核对更新后的表；CUTLASS 4.2.1 公开源码用于架构能力说明。最终工程实现仍以项目固定源码、编译元数据与实机证据为准。

本文的硬件事实要求先约束合法指令、99 KiB CTA SMEM、65536 registers / SM、tile 并行度和数值顺序，再评估融合收益。不会将 SM100/SM103 的 TMEM、CTA-pair、HBM 或稀疏 FP4 广告峰值转嫁到本机 dense FP8 路径。
