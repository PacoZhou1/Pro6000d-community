# PR #5521 FP8 GEMM 派生臂（仅实验）

这是从固定 `source/flashinfer_pr5521_fc1.cu`（SHA256 `e9b3c49e10b63ecd439737738e7ea0cbd713dcab9fddcc89ef1a025b221cba63`）的 FP8 namespace 派生的**新核**，不是直接调用 FlashInfer PR 的 API，也不是可复用的通用 GEMM。原 PR 在另一模型/形状上报告的约 732 TFLOP/s 不能移作本臂成绩。本地仅完成源码生成及静态布局检查；GPU 编译、数值、CUDA Graph、计时均待主线验收。

`pr5521_build.py` 只截取固定 FP8 GEMM namespace，保留原 persistent CTA、4-stage TMA pipeline、mma.sync FP8 主循环及 `accum × act_scale × weight_scale` 的次序。将 K=5376 的 84 个 64 元素 tile 改为 K=5120 的 80 个；将输出 112 个 tile 改为 136 个，并保留 `accum[84]`、`accum[112]` 等寄存器下标原样。每组 `h_raw[8]` 仍先按原代码 BF16 round，然后直接按 gate/up canonical 顺序写 `[M,34816]` BF16；原 norm/AdaLN 量化和 SwiGLU 不进入派生核。生成器固定上游文件 SHA、五类尺寸替换的确切次数、32 组 epilogue 和所有 `accum[]` 引用，任何不匹配会停止生成。

## ABI 与输入布局

见 `pr5521_abi.h`：`recovery_pr5521_launch(out, A, B_packed, scale_a, scale_b, M, 34816, 5120, stream)`。只支持 M=2741 或 16160。A 是 E4M3 `[M,5120]`；B 是 E4M3 `[34816,5120]`，每 16 行为 8 gate + 8 up。对于 canonical 半宽列 `c ∈ [0,17408)`，`p = 16*(c//8) + c%8 + 8*is_up`；`B_packed[p,k] = B_canonical[c+is_up*17408,k]`，`scale_b[p] = scale_b_canonical[c+is_up*17408]`。`scale_a` 是逐行 FP32 `[M]`；这与生产的 block-128 权重缩放**不是同一数值语义**。预排 B 和 scale、准备输入都在计时外，计时输出仅 GEMM 的 BF16 gate/up，不含激活/量化或下游层。

调用时 host 端编码两个 TMA descriptor；无设备分配、无同步。只围绕 GPU kernel 的 CUDA Event 计时不包括 descriptor 编码，CPU 墙钟包括。第一次调用还配置动态 shared memory，应先预热再计时。GPU 图捕获/换输入回放需单独实测，不因此静态设计而推定可用。host 错误返回非零并用 `recovery_pr5521_last_error()` 取诊断；异步 GPU 错误需调用者同步所在 stream 后检查。

## 本地生成与远端验收门槛

本地仅执行：

```bash
python3 _dev/docs/evidence/sm120-next/b-mxfp8-recovery/pr5521_build.py
```

会产出 `build-emit/pr5521_derived.cu`、`pr5521_abi.h` 与 `pr5521_manifest.json`，不碰 GPU。在具备 CUDA 13/SM120 的**隔离实验环境**中可由主线加 `--build --nvcc /path/to/nvcc` 编译 `libpr5521_derived.so`；manifest 分开记录源/派生/DSO SHA 与未验证状态。编译成功也仅代表 DSO 存在。必须先与同一 A、canonical B/scale 的参考 GEMM 比较整张 BF16 gate/up（含 M tail），检查 NaN/Inf、写入覆盖和换输入图回放，再与同形状 CUTLASS/MXFP8 进行无 profiler、公平 warm/扰缓存比较。若数值或图门槛失败，只保留失败证据，不引用性能。

`--build` 固定带 `--resource-usage -lineinfo`，把 nvcc 的 stdout/stderr 合入 `build-emit/pr5521_compile.log`。成功和失败均写 manifest，记录完整命令、nvcc/build exit code 与 log SHA；失败时脚本非零退出。构建前移除同名旧 DSO，防止失败后误认旧库为新产物。本地用 `/usr/bin/false` 代替 nvcc 做了 CPU-only 失败路径检查：退出 1、manifest 与空日志 SHA 一致，未生成 DSO；尚未调用真正 nvcc。

## 固定源码到派生核的人工映射检查

- 生成时锁定原源 SHA `e9b3c49e…`；K 主循环原 `load_g/84` 的余数表达式 4 处、其余 `load_g/84` 20 处、两个 `k_tile<84` 循环，全部改为 80；生成文件再查无旧 `load_g/84`。`accum[84]`、`accum[112]` 属寄存器片段索引，未改；生成器比较原/派生全部 770 个 `accum[]` 引用顺序。
- 原 `GROUP_M*112` tile 映射 36 处均改 136；N=34816 正好有 136 个 256-row B tile，K=5120 正好有 80 个 64-byte TMA tile。B、scale 同用 `p=16*(c//8)+c%8+8*is_up` 预排；本地枚举证明 34816 个 packed 行和 17408 个输出半宽列各覆盖一次。
- 原 32 组 `h_raw` 各由 8 个 `accum×scale_a×scale_b` 浮点结果生成，先保留原 BF16 RN 包装为四个 word：`[gate lo] [gate hi] [up lo] [up hi]`。派生输出分别写 `out[row, col]` 和 `out[row, 17408+col]`；共 128 个 word 存储语句。原 14336 宽度与 SwiGLU 指数/倒数调用均不再出现在执行段。
- tile M=128。M=2741 是 21 个满 tile + 尾部 53 行（总 22 tile）；M=16160 是 126 个满 tile + 尾部 32 行（总 127 tile）。A TMA descriptor 只声明真实 M 行，越界位置使用原源的 zero-fill；所有派生写入保留 `row<M` 条件。A 与输出**无需物理 padding**。这些是源码/地址映射检查，不替代 GPU race、数值与 Graph 验证。
