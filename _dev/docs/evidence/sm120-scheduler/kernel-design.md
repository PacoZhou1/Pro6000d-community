# SM120 scheduler：隔离构建与诊断 ABI

本轮仅控制 **Heuristic / AlongM / AlongN raster 和 swizzle 1 / 8**。保留固定 v6 的 M64N256K128 Cooperative、FP8/FP32 scale 数学、BF16 舍入和 SiLU/量化；没有改 cluster、精度或 K reduction 顺序。新库是 probe 专用，不是模型 runtime 替换库。内核构建由 Sol 在本地开发，实际服务器编译、GPU 数值、Graph、计时与 NCU 由 root 执行。

## 构建入口与来源

root 的固定 image 挂载：旧实验目录只读 `/oldwork`，新实验目录 `/work` 可写。

```sh
python /work/sm120-scheduler/build_scheduler.py \
  --v6-source-dir /work/v6-source \
  --cutlass-include /usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include \
  --build-dir /work/build-v3
```

需上传本目录 [build_scheduler.py](build_scheduler.py)、[scheduler_driver.cu](scheduler_driver.cu)、[scheduler_abi.h](scheduler_abi.h)、[manifest-template.json](manifest-template.json)。旧服务器 v6 目录没有本地抽取时生成的 `source-manifest.json`，首轮因此在编译前失败。root 将旧源原样复制到新可写 `/work/v6-source`，附上本地锁定的 manifest 后构建；旧目录不修改。没有 Python 运行时依赖；builder 只读标准库并执行 nvcc，不初始化 CUDA。`--emit-only` 只验证 source hash 并生成源码，不表示编译或 GPU 通过。

输出为 `/work/build-v3/libsm120_scheduler.so`、`compile.log`、`build-manifest.json` 以及生成的 standalone/shadow headers。编译失败也保留 exit code、log 和 manifest；不继续作为有效候选。nvcc 使用 `-O3 -arch=sm_120a`，不开 `--use_fast_math`；`-lineinfo --resource-usage` 为 NCU 与 ptxas 资源证据保留信息。

builder 校验 v6 manifest 与每个原始文件的硬编码 SHA：visitor、mainloop scale-order shadow、Cooperative M64 support-domain shadow、dispatch、adapter、driver。v6 二进制 lineage 为 `e4039ddc95fa3b56f02743357b7a7193156dc500c5d8804bb77ebd3af85450d6`；本轮库哈希必须另算，不能沿用旧 SHA。原始 CUTLASS scheduler 与三个未修改 upstream header 亦按固定 image SHA 校验。安装目录和旧证据均不修改。

## 三个入口的控制边界

| 入口 | 权重布局 / 输出 | 保留的主循环与数学 | 变化及需单列成本 |
|---|---|---|---|
| `sm120_official_bf16` | canonical B/Bs → canonical BF16 `[M,34816]` | 原官方 M128N128K128，Auto/Cooperative 配置；N128 scale grouping 不改 | scheduler 参数可调；后续 V2 量化组成官方全 pipeline |
| `sm120_paired_bf16` | paired B/Bs → paired BF16 `[M,34816]` | M64N256K128 Cooperative，**与 fused exact same CollectiveMainloop type** | 普通 BF16 epilogue Auto；逆排与后续 V2 量化分别测量 |
| `sm120_fused_fp8` | paired B/Bs → Q FP8 `[M,17408]`、scale FP32 `[M,136]` column-major | 固定 v6 的 Cooperative mainloop、BF16 round、FP32 SiLU×up、V2 quant 语义 | GEMM 自定义 epilogue + 原 v6 pack，共两次 kernel launch |

paired BF16 的 stage count 直接取 fused mainloop 的实际 `DispatchPolicy::Stages`，以 `static_assert(std::is_same_v<...>)` 要求整个 `CollectiveMainloop` 类型一致，防止 BF16 epilogue 的自动 carveout 悄悄改变 stage 数。若普通 BF16 的 storage 无法装下该配置，构建应失败，不能改 stage 后仍宣称 same-mainloop。两个 epilogue 的 shared storage、tile、输出字节数仍不同；它们的时间差用于受控诊断，并不能全部归因于某条 SiLU 或量化指令。

新代码在 caller 中显式设置 `scheduler.raster_order` 和 `scheduler.max_swizzle_size`。raster 不进行整数 cast，而是逐项映射固定 header 的 enum。输入非法直接失败。GPU 工作区/stream 用私有 `pro6000d_scheduler_v1::c3x` 的 `static thread_local` 状态传递；raster 没有 host 全局 setter，每次 launch 都进入 kernel 参数，因此 capture 后的 Graph 不依赖后续 setter 状态。

### 多库 host 符号隔离修复

初始隔离构建仍复制了旧 `vllm::c3x` 名称。真实首轮 probe 在 M1 的旧基线首个 GEMM 中退出 139，尚未调用新 M64 GEMM，也未输出任何 numerics。root 的 faulthandler 定位旧 `Baseline.gemm`；[readelf 证据](symbol-collision-v1.json)显示新库同名 workspace / workspace_bytes / stream 是 **TLS UNIQUE DEFAULT**，旧 baseline 是 **OBJECT UNIQUE DEFAULT**。不能依赖 `RTLD_LOCAL` 消除这种 GNU unique 符号冲突。

修复只涉及 host 隔离：standalone 的 namespace 改为 `pro6000d_scheduler_v1`；5 个 c3x state 改成 internal-linkage `static thread_local`；私有 visitor 类型改名；host 编译启用 `-fvisibility=hidden -fvisibility-inlines-hidden -fno-gnu-unique`，仅 `scheduler_abi.h` 的 C ABI 明确 default visibility。[修复版 readelf](symbol-isolation-v3.json)已确认 c3x 状态为 LOCAL TLS、unsafe c3x symbols 为空，公开入口保留专用 C ABI。此符号检查不能代替重新运行最小 case 与全量 GPU 门槛。

本地文本核对将新 identifier 还原后，dispatch / visitor 与首版生成文本完全相同；adapter 除 namespace / state linkage 外等于 v6；原 pack 函数体 SHA 与三个 shadow header SHA 未变。此核对只证明修改范围；修复版编译与 ELF 隔离已实际通过，数值和 Graph 仍以独立 GPU probe 为准。ABI1 参数与数学不变，二进制 hash 已重新锁定。

`sm120_scheduler_metadata()` 是纯 host JSON，记录 shape、enum、两 M64 mainloop 类型匹配，以及每臂 compile-time stages、block/MMA thread 数、kernel/mainloop/epilogue `SharedStorage` 字节数。ptxas 的实际寄存器、spill 见 compile.log；运行时 occupancy / kernel duration 需 NCU，不由 compile-time metadata 推断。

## 版本化 C ABI 与 ctypes

完整原型以 [scheduler_abi.h](scheduler_abi.h) 为唯一接口合同。专用 ABI 为 **1**，不导出任何旧 `gateup_quant_*` symbol，旧 ABI2 adapter 不可能误用本库。

```python
import ctypes as C
V, I, Z, U = C.c_void_p, C.c_int, C.c_size_t, C.c_ulonglong

lib.sm120_scheduler_abi_version.argtypes = []
lib.sm120_scheduler_abi_version.restype = I
assert lib.sm120_scheduler_abi_version() == 1
for name in ("sm120_scheduler_last_error", "sm120_scheduler_metadata"):
    getattr(lib, name).argtypes = []
    getattr(lib, name).restype = C.c_char_p
for prefix in ("official_bf16", "paired_bf16", "fused_fp8"):
    query = getattr(lib, f"sm120_{prefix}_workspace_bytes")
    query.argtypes, query.restype = [I, I, I], Z  # M, raster, swizzle
lib.sm120_fused_fp8_temp_scale_bytes.argtypes = [I]
lib.sm120_fused_fp8_temp_scale_bytes.restype = Z
for prefix in ("official_bf16", "paired_bf16"):
    launch = getattr(lib, f"sm120_{prefix}")
    launch.argtypes = [V]*5 + [I]*3 + [V, Z, U]
    launch.restype = I
lib.sm120_fused_fp8.argtypes = [V]*8 + [I]*3 + [V, Z, U]
lib.sm120_fused_fp8.restype = I
```

- `raster`: **0=Heuristic，1=AlongM，2=AlongN**；`swizzle`: **1 或 8**。不要按其他 CUTLASS enum 的底层整数猜值。
- launch 三个整数顺序均为 **M, raster, swizzle**，固定 N34816 / K5120 不再传入。
- BF16 指针顺序：**output, A, B, As, Bs**。fused 指针顺序：**out_fp8, out_scales, temp_fp8, temp_scales, A, Bpacked, As, Bspacked**。
- 末尾均为 **workspace, workspace_bytes, stream_address**；stream 是 `unsigned long long`，**0 是合法默认 stream**。
- 支持域 **257≤M≤32768**。小 M 在 probe 中显式走官方 fallback，不调用这三个固定大 M 配置。
- launch `0` 只表示异步发射成功；`1` 表示失败并写 last_error。GPU 异步错误仍由 caller 的同步/Graph 验证捕获。没有静默 fallback。
- workspace query / temp-scale query 都是纯 host；非法参数返回 0。合法 workspace query 至少返回 1 byte；query 与 launch 使用完全相同 shape/scheduler。query 不验证 CUDA 指针地址。

## Buffer 准备与布局

所有输入/输出/workspace 由 caller 预分配，launch 不分配 GPU memory。各臂可变 buffer 不重叠；保留 canonical 与 packed 两份权重的 setup / 额外内存成本，两臂同等持有。所有用于 TMA 的地址/stride 需满足固定 CUTLASS 的对齐和 layout 要求，C ABI 不接收任意 stride。

| Buffer | dtype / shape / stride |
|---|---|
| A | E4M3 `[M,5120]` contiguous |
| As | FP32 `[M,40]` column-major，stride `(1,M)` |
| canonical B / Bs | E4M3 `[34816,5120]` row-major；FP32 `[272,40]` row-major |
| Bpacked / Bspacked | 同形状 contiguous，行组按 `gate128,up128` 交错 |
| BF16 output | `[M,34816]` row-major；official canonical、paired 交错 |
| fused out_fp8 | E4M3 `[M,17408]` contiguous |
| fused out_scales | FP32 `[M,136]`，stride `(1,M)` |
| fused temp_fp8 | E4M3 `[M,34816]` contiguous；每 256 列前 128 有效、后 128 置 0 |
| fused temp_scales | native SF128 layout；用 byte query 预分配，并持有 storage |
| workspace | 按每个入口与 scheduler query 返回 bytes，独立 uint8 buffer |

预排与 inverse 示意（实际 GPU 操作由 probe/root 执行）：

```python
Bpacked = B.view(2, 136, 128, 5120).permute(1, 0, 2, 3).contiguous().view(34816, 5120)
Bspacked = Bs.view(2, 136, 40).permute(1, 0, 2).contiguous().view(272, 40)
Dcanonical = Dpaired.view(M, 136, 2, 128).permute(0, 2, 1, 3).contiguous().view(M, 34816)
```

inverse 会产生实际 GPU copy / allocation（除非 probe 改用预分配 out-copy 实现），必须留出缓冲并单列代价；不能只计 paired GEMM 就当作完整 BF16 pipeline。普通 BF16 诊断可分别比较 kernel-only，以及 **GEMM + inverse + V2 quant**；候选正式门槛仍是 **fused GEMM + pack** 对 **官方 GEMM + V2 quant**。

## 本地验收与待 root 完成部分

本地已通过固定 source SHA 校验、生成锚点计数与 `--emit-only`。root 首版实际 nvcc 编译成功，证据为 [build-manifest.json](build/build-manifest.json) 与 [compile.log](build/compile.log)，库 SHA 为 `671e83890f77687e1a8a984ae1b64766b096bd94437f8fdade6963bb5e282400`。生成源/三个 shadow header 与首版本地 emit 的 SHA 一致；编译通过 exact-mainloop `static_assert`，paired BF16 kernel name 明确为 `StageCount<2>`。该库随后出现上述 host 符号问题，不能作为合格 GPU 候选。namespace 隔离修复版需另外编译/验收并保存新 SHA。

| 实际 ptxas 编译结果 | registers / thread | stack frame B | spill stores / loads B |
|---|---:|---:|---:|
| fused M64N256 | 168 | 0 | 0 / 0 |
| paired BF16 M64N256 | 168 | 0 | 0 / 0 |
| official BF16 M128N128 | 168 | 8 | **12 / 20** |
| 原 v6 pack | 16 | 0 | 0 / 0 |

修复版 root 已实际编译成功，见 [v3 manifest](build-v3/build-manifest.json)、[v3 compile.log](build-v3/compile.log)，SHA 为 **`af1d814966a746ff09edc55d40196b871067d88d22ed0e50700c1f3a4eb185d6`**。编译资源与上表相同。pack body SHA 仍为 `78b22a384534cddc9f2d2babeb9f266cf10da2a714b5ef5c4e05f326a41b12e2`；三份 shadow header 仍是固定 v6 哈希。

root 实际读取的 [host metadata](kernel-metadata.json)为：

| 入口 | stages | block / MMA threads | mainloop SharedStorage B | epilogue SharedStorage B | kernel SharedStorage B |
|---|---:|---:|---:|---:|---:|
| fused M64N256 | 2 | 384 / 256 | 83968 | 18432 | **101376** |
| paired BF16 M64N256 | 2 | 384 / 256 | 83968 | 9216 | 93184 |
| official BF16 M128N128 | 2 | 384 / 256 | 68608 | 9216 | 77824 |

fused 的用户 shared storage 恰为 99 KiB CTA 上限。三个 `sizeof` 口径不是直接相加得到 kernel storage；driver 保留的 shared memory 与用户 storage 需在 NCU 记录中分列。compiler spill 的静态字节数也不能当作每次执行的 local 流量，官方 control 不能标成零 spill。build manifest 的 `numerical_gate=not_run` / `graph_gate=not_run` 仅表示构建器不运行 GPU；独立 probe 结果是后续门槛来源。

全部 6 个 raster/swizzle 组合逐项做普通 BF16 full-bit、FP8 bytes / scale bits、下游官方 down GEMM 与 changed-input Graph。普通 BF16 paired 先 inverse 再对官方 canonical 比较；不得放宽现有数值门槛。任何失败配置禁止进入 winner timings。

只有通过数值与 Graph 的配置才进入随机交错、成对、31 次、完整 pipeline 的 warm/cold 计时。不同 epilogue 输出逻辑字节、inverse 和 pack 成本明确保留；NCU 总 DRAM 计数不是 buffer 级归因。新库尚未证明模型增益，模型入场由 root 按实际门槛决定。
