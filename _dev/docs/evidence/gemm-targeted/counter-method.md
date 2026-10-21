# SM120 GEMM 归因与硬件计数器：原测量方案与执行差异

本页主体保留执行前方案；实际执行结果以 [09-targeted-gemm.md](../../09-targeted-gemm.md) 和 `counters/summary.json` 为准。实际安装 **Nsight Compute 2026.3.0.13**（见 `ncu-install.json`），没有安装下面调研时列出的更新版本。整模 NCU replay 因宿主内存不足 OOM，故改为真实归因尺寸/布局的合成单算子计数；这不是整模权重的计数。所有成功单算子采样为 8 passes。额外 down-M5-cold 案例显式 `--cache-control all`，其他案例为 none；均 `--clock-control none`。down 小 grid 的 active-cycle 派生指标出现超过 100% 的值，按多 pass 短 kernel 的不稳定计数处理，不据此断言利用率。

以下是原始方案。前轮 [`08-gpu-profiling.md`](../../08-gpu-profiling.md) 用 Nsight Systems 确认：FP8 命名 GEMM 约占捕获 kernel 耗时的 62%–68%；普通 MTP4 decode 另有约 29.5% BF16 命名 GEMM，但尚不能定位为 `lm_head`。本轮先查真实矩阵尺寸与模块，再测少量代表性 kernel 的硬件瓶颈，最后才选 tile/scheduling 或其他实现改动。所有 Nsight Compute 时间均是诊断值，整模提速仍由**无 profiler**、同输入 API A/B 决定。

## 工具与权限

- NVIDIA [正式下载入口](https://developer.nvidia.com/tools-overview/nsight-compute/get-started) 当前 Linux x86_64 按钮对应 **Nsight Compute 2026.3 Update 1（2026.3.1.2）** 的 `.run` 包；[发布历史](https://developer.nvidia.com/nsight-compute-history) 标明它配 CUDA 13.4 Update 1。下载会转向带时效 token 的 NVIDIA CDN，应从入口获取，不保存 token。官方列 Blackwell GB20x 支持；当前服务器驱动 595.71.05 低于页面推荐的 Linux 615.71，但页面注明旧驱动仍受支持。是否在这张 SM120 卡上可采全套所需 counter，须以本机 `ncu --query-metrics` 和单 kernel smoke 确认，不先改驱动。[系统要求](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html)、[GPU 支持表](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/gpu-support.html)。
- root 已只读看到宿主 `RmProfilingAdminOnly: 1`。NVIDIA [权限指南](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) 允许在 Linux 容器对**本次隔离实验容器**授予 `--cap-add=SYS_ADMIN` 后以该容器内的 `ncu` 访问计数器；宿主权限仍须满足实际驱动/容器检查。若报 `ERR_NVGPUCTRPERM`，记录失败，不更改 `/proc/driver/nvidia/params`、`modprobe.d`、设备 ACL 或重载驱动，也不停止 embedding。先对单个短合成 kernel 验证权限，成功后再碰模型请求。
- Nsight Compute 默认可能调整 GPU 时钟、冲刷缓存并多遍 replay；正式诊断显式使用 `--clock-control none --cache-control none`，限制 kernel 名和 `--launch-count`，并记录 replay pass 数。这样避免 NCU 主动锁时钟/清空整卡缓存，却不能消除多遍 replay、串行化和共享资源干扰。[Profiling Guide：replay、时钟、cache 与串行化](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)、[CLI 选项](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)。

## 先归因尺寸，再量 counter

1. 在固定 vLLM 0.31 的**临时镜像**中，对 FP8 `Fp8LinearMethod`/block-scaled GEMM 调用点记录真实输入/权重 shape、scale layout、M/N/K、模块全名、阶段（prefill、普通 decode、MTP draft/verify、DFlash thinking）、调用次序与对应请求 ID；记录在隔离窗口并脱敏。不能仅凭上一轮合成微基准的 `M=1/5/16` 或 GPU kernel tile 名推断整模实际 M；长 prefill 的 chunk 和验证批次可能改变 M。
2. 对 BF16 大 GEMM 加模块级 NVTX/shape 记录，用 Nsight Systems 的同线程 CUDA API→correlation ID/图节点证据将 `cutlass_80_*bf16*gemm*` 与具体调用对齐。先区分 target/draft、投影、`ParallelLMHead`/logits；若 CUDA graph 捕获/重放让映射不唯一，另做仅用于归因的 eager/小窗口对照，并明确其性能不能代替原 graph 路径。没有确认前继续写“BF16 命名 GEMM”，不写成已证实 `lm_head`。
3. 由真实调用分层选**少数**代表 kernel：长 prefill 的 FP8 blockwise、普通 decode/投机验证的 FP8、归因后的一种 BF16 大 GEMM。每个保留 M/N/K、layout、grid、核名、图模式、cache 状态、启动顺序和 embedding 计数；若为同名多次调用，不能只靠 `-k` 混合不同尺寸。先用 Nsight Systems/日志定位调用，再用 NCU 的 `--kernel-id`、NVTX 范围或可核对的 launch skip/count 精确抽样。[NCU 过滤参数](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)。

## 最小 counter 分组与判读

在目标容器先执行 `ncu --version`、`ncu --query-metrics --query-metrics-mode all`，确认**这张卡**确有下列完整名称及后缀；缺失就用同义的官方 section（`SpeedOfLight`、`MemoryWorkloadAnalysis`、`Occupancy`、`ComputeWorkloadAnalysis`）或所列 `breakdown:` 指标，不造值、不把 n/a 当 0。单次只选足以回答一个问题的最小组，优先压到一个 replay pass；记录工具报告的 pass 数。[Metrics Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)、[Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)。

| 诊断问题 | 候选最小指标（须本机查询） | 解释边界 |
|---|---|---|
| 核是否真正更快、SM 与内存谁接近上限 | `gpu__time_duration.sum`、`sm__throughput.avg.pct_of_peak_sustained_elapsed`、`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 同一 kernel、同 replay/cache/clock 策略比较；不能与未 profile 的 API TTFT/tok/s 混算。 |
| DRAM/L2 是否吃紧，权重是否实际走显存 | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`、`dram__bytes_read.sum`、`lts__t_sectors.avg.pct_of_peak_sustained_elapsed`、`lts__t_sector_hit_rate.pct` | L2/DRAM 是全卡共享单元；embedding 或其他 GPU 活动可污染。高 L2 命中/低 DRAM 不等于真实模型权重流式访问也如此。 |
| 低利用率是否由并行度不足或 Tensor pipe 限制 | `sm__warps_active.avg.pct_of_peak_sustained_active`、SM Tensor pipe/`sm__ops_path_tensor_*` 可用项 | occupancy 是活动 warp 比率，不是 Tensor Core 利用率；SM120 的 pipe 名称/可用性需查 `--query-metrics`，再看 Compute Workload/throughput breakdown。 |

先用 top-level `.pct_of_peak_sustained_elapsed` 判接近持续峰值的单元，再用该 throughput 的 `breakdown:` 与实测字节数核实；低 DRAM 加低 SM 不应直接定性为“显存带宽瓶颈”。NVIDIA 指出小 kernel、多 pass 和其他 GPU 活动会使共享 L2/DRAM 计数与 hit rate 不稳定；变更后仍以**绝对 kernel duration 和无 profiler 整模请求**为性能判据。[官方 triage 方法](https://docs.nvidia.com/nsight-compute/ComputeTriage/)。

## embedding 不停时的采样边界

只暂停非 embedding 的原 27B/9B/ASR，保留 embedding 原容器和固定 `StartedAt`。在单个本地 loopback 临时容器里运行带 `SYS_ADMIN` 的 NCU；只对一个经确认的 kernel 调极少数 counter、一次短请求，并设置超时。前、中、后持续查 embedding health、ID/StartedAt/RestartCount、请求/Token 单调计数与 running/waiting。若任何计数变化，保留原记录并剔出“无干扰”硬件归因；即使计数未变，也只称“该窗口未观测到 embedding 请求”，不保证后台 GPU 完全空闲。NCU 即便只追踪临时进程，GPU L2/DRAM 与 embedding 仍共享，故多 pass/replay 可能影响其延迟；不能通过停止 embedding 来换干净数据。[NVIDIA 对异步共享资源和多 pass 的说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。

**微基准与真实服务分开。** 上轮 `gemm_probe.py` 反复 replay 相同随机 FP8 权重，容量/复用模式可能让权重留在 L2，且只覆盖 M=1/5/16、GEMM 本身。实际模型以多层不同权重、chunked prefill、attention/GDN、MTP 验证和调度交替运行，显存与缓存行为可能相反。先用真实模型请求选尺寸和瓶颈，再在等尺寸/等 layout 微基准调实现，最终回到冷前缀、cache hits=0、无 profiler API A/B；微基准上的单形状收益不得写成整模提速。
