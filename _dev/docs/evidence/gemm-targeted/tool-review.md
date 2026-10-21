# GEMM 定向工具静态独立审查（2026-10-14）

以下保留断线期间的原始静态审查，不代表最终执行状态。后续处理见本文末尾。

范围仅为本地 `attribution_patch_used.py`、`analyze_attribution.py`、`targeted_kernel.py`、`standalone_fp8_probe.py` 及固定 vLLM 源码的静态核对。服务器断线；未连接服务器、编译 CUDA 候选、运行 Nsight 或取得本轮真实 GPU 结果。本审查**不是**归因和提速的完成验收。

## 待修与使用边界

1. **P2，stage 字段可能被错误地称为“包含 GEMM”。** `analyze_attribution.py:147-152` 只以 GEMM NVTX 的起点在同线程 stage 范围内选择 `enclosing_stage`，没有检查 stage 的结束时间是否覆盖 GEMM 的结束时间。一个 `[100,200]` stage 和 `[150,250]` GEMM 会被标为“enclosing”，但前者没有包含后者。kernel 的 CUDA API correlation 归因不受影响。建议检查两个端点，或者把字段及文案明确改成“覆盖 GEMM 起点的 stage”；添加交叉结束的合成 fixture。
2. **P2，长 module prefix 会产生无法解析的 NVTX 标签。** `attribution_patch_used.py:131-137` 对完整标签直接截取前 240 字符，可能截掉尾部 `:M...:N...:K...`；`analyze_attribution.py:21-23,134-138` 随后将其列为 malformed，相关 kernel 因而不归因。建议只截 prefix，保留尺寸尾段，并在真实 trace 验证 `malformed_gemm_labels` 为空。若固定模型实际 prefix 都短，此风险不会触发；当前不能用静态审查断言触发次数。
3. **P2，仅单线程独立探针安全；不能把 C ABI 直接并发集成。** `targeted_kernel.py:101-129,149-178` 用进程内 `inline` 全局变量传 workspace、字节数和 stream，两条主机线程并发调用会互相覆盖；同一 workspace 若在多个未同步 stream 上复用也会竞争。文件第26行已声明单线程，当前独立探针第98-110行按顺序调用且同当前 stream，因此这是集成边界，不是对现有单线程测量的否定。若进入服务需将上下文作为调用参数局部传递，或先落实串行和 stream 隔离；CUDA graph 捕获后也必须保持输入、输出及 workspace tensor 在 replay 期间存活。
4. **P3，大 SQLite 的内存负载尚未验证。** `analyze_attribution.py:35-41,113-126,161-170` 将 kernel、NVTX 和每种 CUDA API 全表读取为 Python 字典列表；`shortest_matches` 第57-85行已是排序加双堆，不再有每个点线性扫所有范围的问题，但输入仍可能消耗远高于 SQLite 文件大小的内存。真实 19K eager trace 前先记录 SQLite 文件大小和可用 RAM；必要时通过 SQL 按 PID/时间筛选或流式迭代。不要把 O((R+Q) log R) 的匹配算法误当成整体有界内存保证。

## 已核对、仍须上机验证

- `analyze_attribution.py --self-test` 本地通过。另用一组纯内存 SQLite fixture 复核：Runtime 和 Driver 各一条同 `(pid,correlationId)` 的 API 记录时，最新版分别报告 **2 条 API record、1 个唯一 correlation key、1 个归因 kernel**，没有把 kernel 重复累计。报告时不能把 API record 数称为独立 GEMM launch 数。没有真实 Nsight SQLite，仍需检查 `PROCESSES.globalPid`、未归因占比、歧义 key、malformed label 和实际 NVTX worker PID。
- `attribution_patch_used.py` 的四类固定方法签名与 `source/` 内固定 vLLM 代码一致；`Fp8LinearMethod.apply` 通过 ContextVar 传 prefix 给 `CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm`，该方法内含实际 `ops.cutlass_scaled_mm` 调用。它只为 `--enforce-eager` 的真实捕获提供模块关联；CUDA graph 正常 replay 不能由这些 Python hook 逐次归因。
- `targeted_kernel.py` 的 SHA 固定 dispatch header 保留了原有 `M<=64` swap-AB、`65..256` Pingpong 和其余 default 路径；实验分支只改 `M>256,K=5120,N=34816` 的调度配置。此 K/N 仍须由真实归因 trace 确认是目标模型热点，不能由代码中的常量反推。固定 [CUTLASS caller](https://raw.githubusercontent.com/vllm-project/vllm/db9527a46873454610df6dbedf79a36d6bf1a7f6/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh) 也默认构造 `KernelHardwareInfo`、检查 `can_implement`、计算 workspace 大小、调用 `op.run`；独立 adapter 预分配 workspace，故输出相等门槛必要，但不能由此声称性能路径与 Python operator 完全相同。
- `standalone_fp8_probe.py:113-132,185-189` 的 A scales 使用固定 [Python CUTLASS backend](./source/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py) 相同的 column-major 量化；B scales 的 `[N/128,K/128]` 连续存储在传给 operator 时转置，布局与独立 driver 描述一致。固定 [SM120 `.cu`](https://raw.githubusercontent.com/vllm-project/vllm/db9527a46873454610df6dbedf79a36d6bf1a7f6/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu) 在 `b.size(1)*b.size(0) > l2CacheSize` 时取 swizzle 8，否则取 1；FP8 每元素 1 字节，因此 probe 的 `K*N > L2` 与其一致。执行时还须记录实际 L2 值与 swizzle。
- 探针第194-227行要求 variant 0 的**全输出**与固定 operator bit-equal；candidate 的实验大 M 只检查全输出有限性，加上 1–3 token 行及三个 N 块的 FP32/variant 0 数值误差。该门槛能拦住布局与明显精度错误，不能证明全部 candidate 输出或模型质量无差异。第232-256行的 warm graph replay 和 >L2 flush 的 cold 近似是两种微基准；同一合成权重反复测量，即使总权重大于 L2，也不能替代真实模型多层权重流式读取及完整请求 A/B。

## 上机前判定

本地四个脚本 AST 解析通过，分析器自测通过；上述 P2 应修或在报告中明确风险边界。只有固定镜像候选实际编译、variant 0 全输出 exact gate、candidate 数值门槛、单 kernel Nsight Compute 计数、无 profiler 的同栈 A/B，以及 embedding 连续存活和服务恢复证据全部取得后，才能评价算子是否可用或是否提速。当前这些结果均未取得。

## 后续执行状态（主线程补记）

- stage 已改为完整区间包含，并重跑真实 SQLite；实际 trace malformed/ambiguous 均为零。长 prefix 修正在新 attribution_patch.py，实际捕获仍保留不可变 attribution_patch_used.py。
- 实际 SQLite 分析成功，大小与哈希见 raw-artifact-manifest.json。
- 服务实验采用独立 output/scratch、host launch lock 和 record_stream；最终 runtime-smoke-v3 的真实 GPU 检查通过。
- 三个算子候选、八组合成 NCU、有效 A1/B/A2 API 与真实 Graph replay 已完成。guard-v1 未命中已排除，v2 正向证明见 streamk-replay/replay-proof.json。
- 结果及限制见 [最终报告](../../09-targeted-gemm.md)。全任务独立审查另行执行，不能把本静态记录当作该审查。
