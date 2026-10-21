# 任务 A 结果：FP8 GEMM 硬件上限（RTX PRO 6000 Blackwell，SM120）

日期 2026-10-14。主机 /tmp/qwen-sm120-next-a/（固定镜像 sha256:a4a4c043…c3a6b，nvcc 13.0.88，torch 2.13.0+cu130，vLLM 0.31.0，驱动 595.71.05，功耗上限 600 W）。无 sudo，未锁频，只记录 NVML 时钟。原始数据全部在本目录；对脚本的每处改动见 `../CHANGES.md`。

## 结论（先说）

1. **A1：PRO 6000 上普通 FP8 MMA 不是半速。** 普通形式、`kind::f8f6f4` 形式、block-scaled（UE8M0 0x7f7f7f7f）三者都是约 1000 TFLOP/s（标称 1007.6），约 4.0 SM 周期/条 mma/SM，三者之间差别在 1.5% 以内。5090 上“普通形式半速”的假设在本卡上被排除；改用 block-scaled 形式不会带来 2 倍。
2. **A2：M=16160 gate_up 上，cuBLASLt 最高 674 TFLOP/s（tensor-wise），仅为同次运行 CUTLASS（619）的 ×1.090，为 617 的 ×1.093，均低于 ×1.10 门槛。按计划规则 GEMM 线关闭**，本轮只继续任务 D。
   - 而且这个最高值来自 tensor-wise 缩放（整个张量一个 scale），不等价于生产用的 1×128 / 128×128 块缩放，数值上不是可替换方案；能保持块缩放语义的 cuBLASLt 模式（MXFP8 VEC32）比 CUTLASS 慢 4%。
   - 块缩放（DeepSeek 式 1x128 / 128x128）在本栈 sm_120 上 cuBLASLt 不接受（见下），所以“拿 cuBLASLt 吃下 blockwise”这条路不存在。
3. 注意，需要保留的限制：裸 MMA 能到 ~1000 TFLOP/s、GEMM 实测只有 ~620–670，差距主要不在 MMA 速率：GEMM 运行时 SM 时钟约 1850 MHz（功耗达 600 W 上限），而裸 MMA 探针时钟 2590 MHz。这是 power-limited 下的 GEMM，不是 MMA 发射速率受限。

## A1：裸 MMA 吞吐

构建：`-gencode arch=compute_120a,code=sm_120a`，BS_VARIANT=0（CUTLASS 式括号拼写）一次通过，无需 BS_VARIANT=1。注册 42/42/43，无 spill。

SASS（`sass-mma-by-function.txt`，用镜像内 pip 的 nvdisasm）：
- plain 与 `kind::f8f6f4`：同一条 `QMMA.16832.F32.E4M3.E4M3`（两种拼写编出同一指令，同速符合预期）。
- block-scaled：`QMMA.SF.16832.F32.E4M3.E4M3.E8`（另一条不同的指令，scale 寄存器 R40 两个操作数；“URZ”为指令尾部操作数）。
- 因此“两种形式编出不同指令”成立，A1 有意义。

`--verify`（64 次累加，一个 warp）：三者累加结果逐位相同（`identical:true, nonzero, finite`），UE8M0 127 的 scale 保持数学结果。

吞吐（每次 3 重复取中位数；每 SM 1 block × 16 warp × 8 链，188 SM，6,000,000 次迭代）：

| 形式 | 运行1 TFLOP/s | 运行2 TFLOP/s | 运行内各重复 | 运行时 SM 时钟均值 MHz | 周期/mma/SM |
|---|---|---|---|---|---|
| plain | 997.91 | 1000.35 | 1009/998/997；1005/1000/994 | 2593 / 2592 | 4.00 / 3.99 |
| kind::f8f6f4 | 997.98 | 999.82 | 997–1002；999–1000 | 2599 / 2602 | 4.01 / 4.01 |
| block-scaled | 992.89 | 983.97 | 983–993；978–985 | 2585 / 2549 | 4.01 / 3.99 |

- plain / bs ≈ 1.00–1.01，不存在半速。block-scaled 略低 0.5–1.6%，对应时钟略低（2549–2585 vs 2593 MHz），周期数相同。
- 运行1 的 plain 首次重复 “before” 时钟 180 MHz 为空闲起点，运行内 min=180 是爬频采样；三次重复中位数不受影响。第二次运行是时钟/热漂移复核，两次一致。

## A2：torch._scaled_mm / F.scaled_mm vs vLLM CUTLASS blockwise

每个 M 单独一次容器运行（M=32768 峰值显存 17.9 GB，空闲时他人占 5.4 GB；远低于上限）。31 对，同进程随机交错，warm 与 cold（>2×L2 写缓冲）各一组，每个 (shape,M) 先 1.5 s 不计时爬频预热，逐样本取 ±8 ms 填充窗口内的 NVML SM 时钟。比值 = CUTLASS_ms / arm_ms（>1 表示 arm 更快）。随机 FP8 操作数，仅测速度，不做数值结论。

### scale 模式接受/拒绝（所有 shape、所有 M 一致）

| arm | 结果 |
|---|---|
| cutlass_blockwise（vLLM 参照） | 接受 |
| cublas_tensorwise（`_scaled_mm`，FP32 标量） | 接受 |
| cublas_rowwise（`_scaled_mm`，(M,1)/(1,N)） | 接受 |
| cublas_mxfp8_vec32（`_scaled_mm`，e8m0 swizzled） | 接受 |
| cublas_mxfp8_nn_functional（`F.scaled_mm`，BlockWise1x32 + SWIZZLE_32_4_4，猜测签名居然可用） | 接受（与 vec32 同速） |
| cublas_blk_1x128_128x128_colmajorB（`_scaled_mm`） | 拒绝：`CUBLAS_STATUS_NOT_SUPPORTED` 于 `cublasLtMatmulAlgoGetHeuristic` |
| cublas_blk_1x128_128x128_rowmajorB | 拒绝：Invalid scaling configuration（128x128 的 B scale 需 (40,272) 且 near-inner-dim-major） |
| cublas_blk_1x128_1x128 / _contigB | 拒绝：前者 Invalid scaling configuration，后者 `CUBLAS_STATUS_NOT_SUPPORTED` |
| cublas_v2_blk_1x128_128x128_{colmajor,rowmajor}B（`F.scaled_mm`） | 拒绝：“DeepSeek style (1x128, 128x128) scaling only supported in CUDA for SM90” |
| cublas_v2_blk_1x128_1x128_{colmajor,contig}B（`F.scaled_mm`） | 同上拒绝（torch 把该组合当作 DeepSeek 式，仅 SM90） |

注：我没有穷尽所有 scale 布局组合；以上是尝试过的全部。torch 的错误文本说明 128×128 块 scale 形状应为 (40,272)；v1 的 1x128/128x128 用 colmajor（outer-dim-major）布局被 cuBLAS 启发式直接拒绝，rowmajor 布局被 torch 校验拒绝。完整异常原文在 `scaled_mm_m*.jsonl` 的 `attempt` 行（`exception` 字段）。

### M=16160（决策形状）

gate_up（N=34816, K=5120），warm / cold：

| arm | ms (warm) | TFLOP/s (warm) | 平均 SM MHz | 周期 (百万, warm) | 配对比值 warm | ms (cold) | TFLOP/s (cold) | 配对比值 cold |
|---|---|---|---|---|---|---|---|---|
| CUTLASS blockwise | 9.307 | 619 | 1850 | 17.214 | 1 | 9.282 | 621 | 1 |
| cuBLAS tensor-wise | 8.542 | **674** | 1850 | 15.799 | **1.090** | 8.525 | 676 | 1.089 |
| cuBLAS row-wise | 8.737 | 659 | 1850 | 16.160 | 1.066 | 8.736 | 659 | 1.062 |
| cuBLAS MXFP8 vec32 | 9.718 | 593 | 1850 | 17.976 | 0.958 | 9.683 | 595 | 0.959 |
| cuBLAS MXFP8 F.scaled_mm | 9.725 | 592 | 1850 | 17.988 | 0.957 | 9.684 | 595 | 0.959 |

down（N=5120, K=17408）warm：CUTLASS 4.518 ms / 638 TFLOP/s；tensor-wise 4.648 / 620（0.972）；row-wise 4.329 / 665（1.044）；MXFP8 4.809 / 599（0.940）。cold 与此一致（row-wise 667，1.045）。

### 其余 M（warm；ms / TFLOP/s / 平均 MHz / 配对比值），gate_up

| M | CUTLASS | tensor-wise | row-wise | MXFP8 vec32 |
|---|---|---|---|---|
| 2741 | 1.617 / 604 / 1852 | 1.495 / 654 / 1852 / 1.083 | 1.554 / 629 / 1.041 | 1.704 / 574 / 0.951 |
| 2786 | 1.626 / 611 / 1845 | 1.505 / 660 / 1845 / 1.079 | 1.560 / 637 / 1.041 | 1.718 / 578 / 0.944 |
| 32768 | 19.916 / 587 / 2006 | 17.594 / 664 / 2004 / 1.132 | 30.216 / 387 / **0.659**（异常慢） | 20.025 / 583 / 0.994 |

down warm：M=2741 CUTLASS 607 TFLOP/s，tensor-wise 609（1.003），row-wise 632（1.040），MXFP8 582（0.958）；M=2786 CUTLASS 614，tensor-wise 615（1.001），row-wise 640（1.042），MXFP8 587（0.959）；M=32768 CUTLASS 631，tensor-wise 669（1.060），row-wise 659（1.044），MXFP8 587（0.931）。cold 组与 warm 差在 2% 内，完整表见 jsonl 的 `summary` 行。

所有 shape 的最佳 cuBLASLt arm 相对 CUTLASS 是 +0% 到 +13%（最高：M=32768 gate_up tensor-wise ×1.132，但这是长 M 且是 tensor-wise；该 M 不在生产 prefill 的主要区间）。周期数（时间 × 平均时钟）与毫秒的比值一致，说明这些差别不是时钟漂移造成：同一 (shape,M) 内各 arm 交错测量，时钟相同。

### 判据（计划 A 节）

- M=16160 gate_up：最高 674 TFLOP/s（tensor-wise）。vs 同次运行 CUTLASS 619：**×1.090**；vs 14 号文档 617：**×1.093**。均 < ×1.10。脚本 `decision` 行 `gemm_line_closes: true`（warm 与 cold，gate_up 与 down 均为 true；down 最高 row-wise ×1.044）。
- **决定：GEMM 线关闭。** 不进 B（MXFP8，cuBLASLt MXFP8 比 CUTLASS 慢 4–7%，且 A1 排除了“普通形式半速”）；不做 A3（无 ≥10% 的单算子优势；唯一接近的是 tensor-wise，需丢掉块缩放精度，不可替换）。
- 边界提醒：×1.090 距离门槛仅 1 个百分点；但①它是 tensor-wise 缩放的上界，不是 blockwise 的等价实现；②所有保持块缩放语义的 cuBLASLt 路径都 ≤ CUTLASS；③FlashInfer #5521 的 732–741 TFLOP/s 融合 kernel 属于 A3 候选，未测（按计划，A1/A2 未显示 ≥10%，所以跳过）。

## 失败、限制与诚实声明

- 构建失败三次后成功（全部日志保留）：①镜像无 nvml.h 且无 libnvidia-ml stub → 改为 dlopen（`build-v0-nvmlh-failed.log`）；②`-arch=sm_120a` 在 nvcc 13.0 下 PTX 以 compute_120 汇编，`.kind::f8f6f4`/block_scale 报“not supported on .target sm_120” → 改 `-gencode arch=compute_120a,code=sm_120a`（`build-v0-archflag-failed.log`、`build-v1-archflag-failed.log`）；③镜像无 nvdisasm，首次 `cuobjdump -sass` 为空，改 PATH 后得到 SASS。
- 未锁频（无 sudo）；仅记录 NVML 时钟。GEMM 运行时 SM 时钟 1805–2006 MHz，功耗达 600 W 上限；裸 MMA 探针 2550–2600 MHz。不同 M 的时钟不同（M=2741/2786 的 down 为 ~2000 MHz，M=16160 的 down 为 ~1840 MHz），所以不同 M 间的绝对 TFLOP/s 不宜直接比较，同一 (shape,M) 内的配对比值才是比较口径。
- 逐样本时钟取 ±8 ms 填充窗口的 NVML 轮询均值（样本窗口比单次 1–20 ms 的重放更宽，是平滑值，不是瞬时值）；首次冒烟运行因窗口无样本而得到 195 MHz，已修正并弃用该运行。
- nvidia-smi 的 1 秒轮询 CSV 仅 15 行/次（SSH/网络间歇中断不影响主机上的后台任务，但数据稀疏）；只用于佐证功耗达 600 W。
- 只测速度；随机 FP8 操作数；tensor-wise / row-wise 的“更快”不含任何精度含义。
- B 操作数是 `w.t()`（TN）；未测 NN。`F.scaled_mm` 的块缩放签名是否还有别的有效调用方式未穷尽（报错文本明确说该组合在 CUDA 上仅 SM90）。
- 测试期间 SSH 链路反复超时（宿主/网络问题，非本任务造成），一次中断约 15 分钟；所有长任务均在主机后台运行，结果未受影响。
- 嵌入服务：前后均通过；见 `embedding-before.txt` / `embedding-after.txt`、`services-before.json` / `services-after.json`。容器 ID `4224a2b8…`，StartedAt 2026-10-06T09:31:08.313413609Z，RestartCount 1（实验前后一致，该计数是实验前就有的），before/after 的实际 embedding 请求返回 1024 维、有限、范数 1.0。27B/9B/ASR/备份容器状态未变（均为已停止）。
- 主机上本实验容器已全部删除（`docker-ps-after.txt` 中无 codex- 容器）；/tmp/qwen-sm120-next-a/ 保留（含 root 属主的二进制与 sass）。
