# 任务 A：FP8 GEMM 硬件上限（A1 裸 MMA 吞吐 + A2 cuBLASLt 对照）

**状态：尚未运行（not yet run）。** 本目录所有文件均在无 NVIDIA GPU、无 nvcc 的 Mac 上编写，仅做过 `python3 -m py_compile` 与纯 CPU 的汇总逻辑单测；CUDA 源码、`run_a.sh` 的容器流程、`torch._scaled_mm` 的各 scale 模式全部未经实机验证。下面的“预期输出”是格式说明，不是结果。

计划来源：`../../16-sm120-next-round-plan.md` 第 2 节。不写任何新 GEMM kernel。

## 文件

| 文件 | 作用 |
|---|---|
| `mma_rate_probe.cu` | A1：寄存器常驻循环测三种 FP8 mma 的吞吐（plain / `kind::f8f6f4` / block-scaled UE8M0 `0x7f7f7f7f`），含 `--verify` |
| `scaled_mm_ceiling.py` | A2：`torch._scaled_mm` 各 scale 模式 vs vLLM CUTLASS blockwise，同进程随机交错 31 对，warm / 扰缓存，逐样本 NVML 时钟 |
| `run_a.sh` | root 操作员在 GPU 主机上的完整序列（构建、探针两次、A2、时钟/功耗快照、服务快照） |

## 设计要点

- A1：每 SM 1 个 block（64 KiB 动态 smem 强制），16 warp，每 warp 8 条独立累加链；grid = 运行时查询的 SM 数（预期 188）。每次 mma 为 2*16*8*32 = 8192 FLOP/warp。输入为确定性、有限的 E4M3 随机字节。JSON 含 TFLOP/s（中位数及每次重复）、耗时、迭代数、寄存器数/local 字节、NVML SM 时钟（前/后/运行中均值/最小/最大/平均）以及“每 SM 每条 mma 的周期数”。
- `--forms` 默认 `plain,bs`；`run_a.sh` 里加了 `kind`（CUTLASS `SM120_16x8x32_TN` 的拼写）以确认它与 plain 同速。
- block-scaled 内联汇编的操作数写法与 FlashInfer #5963 描述一致（助记符顺序、`{sfa}, {0,0}, {sfb}, {0,0}`、scale 必须放寄存器）。大括号细节来自对 CUTLASS 的记忆，未能汇编验证；若 ptxas 报错，用 `-DBS_VARIANT=1`（PTX ISA 拼写）重编，`run_a.sh` 会自动回退并把使用的变体写入 `bs_variant.txt`。
- A2 逐 arm 尝试并记录接受/拒绝（`kind":"attempt"`，含异常原文）：
  `cublas_tensorwise`、`cublas_rowwise`、`cublas_mxfp8_vec32`（float8_e8m0fnu、swizzled 1-D scale）、`cublas_blk_1x128_128x128_{colmajor,rowmajor}B`、`cublas_blk_1x128_1x128`、以及猜测签名的 `cublas_mxfp8_nn_functional`（仅当 `torch.nn.functional.scaled_mm/ScalingType/SwizzleType` 存在）。B 操作数始终是 `w.t()`（K,N 列主序，即 TN），与 CUTLASS 路径共用同一份随机权重。
- CUTLASS 参照：`vllm._custom_ops.cutlass_scaled_mm(a, w.T, out_dtype=bf16, scale_a=[M,K/128] stride(1,M), scale_b=[N/128,K/128].T)`，与既有探针（`evidence/gemm-epilogue/pipeline_probe.py`）相同。
- 计时：每次 arm 单独 CUDA graph；每对内所有 arm 随机排列；cold 模式在计时事件之外写 >2×L2（至少 256 MiB）缓冲。每个样本取其窗口内 NVML 轮询（约 2 ms）的 SM 时钟均值；汇总同时给 `median_cycles = ms × 平均时钟`。随机种子与脚本 SHA256 写入 meta。比值定义：`paired_cutlass_over_arm = cutlass_ms / arm_ms`，>1 表示该 arm 更快。
- 仅测速度，随机 FP8 操作数，不做数值结论。

## 精确命令

在 GPU 主机上（root），目录同步到主机后：

```bash
sudo ./run_a.sh                       # 标签默认 a-<UTC 时间>
sudo LOCK_CLOCKS=1 ./run_a.sh a-lock  # 另外尝试 nvidia-smi -lgc；失败只记录，不中止
```

结果在 `/home/llm/.local/share/qwen-sm120-next/<label>/`，之后拷回本目录。

`run_a.sh` 内部顺序：服务快照（复用 `verify_service_state.py --functional`，字段与既有 `final-services.json` 相同）→ 后台 15 s 一次的 embedding 状态监控 → `nvidia-smi -q -d CLOCK,POWER`（前/中/后）→ 容器内构建 → `--verify` → 探针运行两次 → `scaled_mm_ceiling.py` → 服务快照。只用镜像 `sha256:a4a4c043…c3a6b`，容器名 `codex-sm120-next-<label>-*`，`--network none`、24g 内存、3 CPU；不启动/停止/重启任何模型服务，只在退出时删除自己的容器；启动前若已有 `codex-(fp8|gemm|sm120)-*` 容器则拒绝运行。

等价的手动构建与 SASS 检查（容器内，`/work` 为本目录）：

```bash
nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -Xptxas -v \
     -o mma_rate_probe mma_rate_probe.cu -lnvidia-ml
#   若找不到 libnvidia-ml.so：加 -L/usr/local/cuda/lib64/stubs（仅链接期；运行期用驱动注入的真实库，容器需 NVIDIA_DRIVER_CAPABILITIES 含 utility）
cuobjdump -sass mma_rate_probe | grep QMMA
./mma_rate_probe --verify --forms plain,kind,bs
./mma_rate_probe --forms plain,kind,bs [--iters 6000000] [--reps 3]
python3 scaled_mm_ceiling.py --out scaled_mm_ceiling.jsonl --pairs 31 [--m 16160] [--shapes gate_up]
```

## 预期输出

- `build.stdout` / `sass-mma-forms.txt`：SASS 中的指令名。预期 plain 为 `QMMA.16832.F32.E4M3.E4M3`，block-scaled 为带 `.SF`/`E8` 后缀的 `QMMA.SF.16832…`（与计划 A1 一致；若实际名称不同，以实测为准并记录）。`-Xptxas -v` 的寄存器数在 `build-v*.log`。
- `mma-verify.jsonl`：一行 `{"mode":"verify","identical":true,...}`，退出码 0；`identical:false` 表示 `0x7f7f7f7f` 缩放没有保持数学结果，后续吞吐数字作废。
- `mma-rate-run{1,2}.jsonl`：每种形式一行（含 `tflops`、`elapsed_ms`、`iterations`、`sm_count`、`sm_clock_mhz_*`）。两次运行之差反映漂移。解读：峰值标称 1007.6 TFLOP/s；若 plain 约为 bs 的一半，则 PRO 6000 也是半速（2 倍级发现）；若相同，则排除该假设。
- `scaled_mm_ceiling.jsonl`：`meta`（环境、API 探测、lock-clocks 结果）、`attempt`（accepted / 异常原文）、`sample`（逐样本毫秒与 SM 时钟）、`summary`（每 shape×M×{warm,cold}×arm：中位毫秒、TFLOP/s、时钟、周期、配对比值）、`decision`。

## 判据（计划 A 节）

取 M=16160 的 gate_up，比较“本卡实测最高 FP8 TFLOP/s（A1 的 mma 上限仅作天花板参考；可达到的 GEMM 以 A2 各 arm 为准）”与 CUTLASS 的 617 TFLOP/s（14 号文档）：

- **若最高值 < 617 × 1.10（约 679 TFLOP/s），GEMM 线整体关闭**，本轮只继续任务 D。`scaled_mm_ceiling.py` 的 `decision` 行用同次运行的 CUTLASS 实测与 617 两个基准各判一次，`gemm_line_closes` 要求两者都不足 +10%。
- 若 ≥ +10%：按来源进入 B（block-scaled / MXFP8）或 A3（社区 kernel，需固定 SHA）。
- 所有结论须带时钟：用 `median_cycles` 与 `sm_clock_mhz_*` 排除 15 号文档记录过的时钟漂移假象。

## 操作员首先要检查的未知项

1. `nvcc` 能否汇编 block-scaled 内联汇编（看 `bs_variant.txt` 与 `build-v0.log`）。
2. `mma-verify.jsonl` 的 `identical` 与 SASS 里是否真的出现两种不同的 QMMA 变体（若 plain 与 bs 编出同一指令，A1 无意义）。
3. torch 2.13 的 `_scaled_mm` 是否接受 e8m0 MXFP8 scale、1x128/128x128 blockwise 以及哪种 scale 张量布局（看 `attempt` 行的异常原文；`cublas_mxfp8_nn_functional` 的调用签名是猜的，失败不代表 torch 不支持）；`meta.api` 记录了相关符号是否存在。
4. 容器内 NVML 是否可用（`meta.nvml_error`、探针 stderr 里的 “NVML unavailable”）。不可用则时钟为 null，应在宿主用 `nvidia-smi` 快照补充或锁频重跑。
5. M=32768 时每个 arm 的 bf16 输出约 2.3 GB，graph 池按 arm 数累加（最多约 8 个 arm）；如显存不足，用 `--m` 拆开跑。
6. 容器中 `nvidia-smi -lgc` 一般无效；锁频由 `run_a.sh` 在宿主执行（`LOCK_CLOCKS=1`），成败写入 `lock-clocks.txt`。
