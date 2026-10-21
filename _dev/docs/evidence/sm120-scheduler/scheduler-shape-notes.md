# 两个大 M 的官方 scheduler：源码映射与解释边界

**M16160 与 M32768 的官方 M128N128 Heuristic 都是 AlongM。**AlongM/s1 在前者约快 2%，在后者约慢 50%；这不是 Heuristic 在这两个尺寸之间换了方向。源码能确认物理 grid、swizzle 和逻辑 tile 映射，不能据此确认缓存失效根因或硬件实际执行先后。

## 固定源码与旧解释纠正

最小完整 header 已逐字节保存于 [scheduler-source](scheduler-source/source-manifest.json)，保留原版权与依赖声明：

- [tile_scheduler.hpp](scheduler-source/tile_scheduler.hpp)，361–383 行：`Sm120 + void/PersistentScheduler` 实际选择 **`PersistentTileSchedulerSm100`**。
- [tile_scheduler_params.h](scheduler-source/tile_scheduler_params.h)，1881–1904、1919–1926 行：该 CLC params 复用 Sm90 的 `get_rasterization_order`，用**未按 swizzle 向上取整**的 tile 数；s1 禁用 remap，s8 直接使用 divisor=8。
- [sm100_tile_scheduler.hpp](scheduler-source/sm100_tile_scheduler.hpp)，205–248 行：physical grid 是完整 CTA grid；AlongN 将 x/y 转置。770–798 行：在可被 swizzle 整除的内部区域重排，残余区域保持原坐标，最后按 AlongN 转置逻辑 M/N。
- 同文件 368–369、439–471 行：初始 block ID 和后续 CLC cancellation 返回的 ID 都经过同一映射；实际任务动态领取，**不是静态 188 个常驻 CTA 的线性扫描**。类名带 Sm100 不代表 SM120 使用 SM100 的 MMA/TMEM。

这与旧 [resource_model.py](../hardware-grounded/resource_model.py) 151–156、174–177 行的 Sm90 算术模型有区别：`scheduler_rounded_mn_ctas`、`scheduler_minor_dimension` 不能当作本机 CLC 的实际 launch grid / fast axis。旧模型的 tile 数、逻辑 operand 字节与资源预算可保留；source mapping 现已标为 `legacy_sm90_*`，另用 `clc_*` 字段记录实际 grid、选定方向、直接 swizzle divisor、residual region 和映射来源。不要重写原始 NCU 数据来迎合模型。

本轮同步纠正记录（行号为修正前位置）：

- [13 主文](../../13-sm120-hardware-roadmap.md)原第 48 行：去掉“经过 swizzle 向上取整后”作为方向反转依据。官方 127×272 与 v6 253×136 **本身**已使 Heuristic 从 AlongM 变 AlongN；该结论仍成立，但实际 CLC 没有按 s8 扩大 physical grid。
- [旧 resource-model.md](../hardware-grounded/resource-model.md)原 181–191 行：将 `PersistentTileSchedulerSm90`、rounded grid 及无条件的 minor/fast-changing 解释替换为上述 Sm100 CLC 路径，不能只补 footer。
- [14 主文](../../14-sm120-scheduler-epilogue.md)原第 9 行的“AlongM 指 N 连续、AlongN 指 M 连续”需删去无条件表述。**s1** 时物理 x 邻居在 AlongM 中改变逻辑 M，在 AlongN 中改变逻辑 N；**s8 内区**每组 8 个 x 邻居改变另一个逻辑维度。名称和 swizzle 必须一起解释。

## 两个尺寸的实际参数

官方 tile=(128,128,128)、cluster=(1,1,1)、N=34816、K=5120，逻辑 tile 数为 `(ceil(M/128),272)`。Heuristic 调用的规则为 `tiles_N > tiles_M ? AlongM : AlongN`，这两个尺寸不触发 grid-y overflow 回退。

| M | 逻辑 M×N tiles | H8 实际方向 / divisor | H8 与 AlongM/s1 physical grid | CTA 数 | s8 residual |
|---|---|---|---|---:|---|
| 16160 | 127×272 | AlongM / 8 | (127,272,1) | 34544 | x=120…126，不重排 |
| 32768 | 256×272 | AlongM / 8 | (256,272,1) | 69632 | 无；两维均整除 8 |

M16160 grid 有真实 [旧官方 NCU](ncu-m16160-official/raw.csv)和 [新官方 AlongM/s1 NCU](ncu-m16160-official-along-m-s1/raw.csv)佐证，都是 `(127,272,1)`，不是 rounded `(128,272,1)`。新增 M32768 的 [H8 CSV](ncu-m32768-official-heuristic-s8/raw.csv)与 [AlongM/s1 CSV](ncu-m32768-official-along-m-s1/raw.csv)也都确认 `(256,272,1)`；不只是源码推导。

同样，M16160 的 M64N256 AlongN/s8 physical grid 实测为 **(136,253,1)**，AlongM/s1 为 **(253,136,1)**，CTA 都是 34408；见 [AlongN CSV](ncu-m16160-v6-along-n-s8-isolated/raw.csv)、[AlongM CSV](ncu-m16160-v6-along-m-s1/raw.csv)。没有按 s8 把 253 扩至 256 个 launch block。

## 邻近物理 ID 如何映射到逻辑 M/N

以下均是 cluster=1 的**坐标映射**，按名义物理 x 邻居列出；不代表 CUDA/CLC 的实测时间顺序，也不能据此断言哪些 CTA 同时占用哪些 SM。

AlongM/s1 无 remap：`(m,n)=(x,y)`。在 `y=0`，x=0…7 映为 `(0,0),(1,0),…,(7,0)`，邻居使用相同 B tile、不同 A tile。

AlongM/s8 的内部区域令 `q=floor(grid.x/8)`：

```text
m = floor(x/8) + q*(y mod 8)
n = (x mod 8) + 8*floor(y/8)
条件：x < 8*floor(grid.x/8)，y < 8*floor(grid.y/8)
其他坐标不 remap；AlongN 则在 remap 后交换逻辑 m/n。
```

因此两个尺寸在 y=0、x=0…7 都映为 `(0,0),(0,1),…,(0,7)`：同一个 A tile，8 个 B tile。x=8…15 映为 `(1,0)…(1,7)`。y=1 的首组分别是 M16160 的 `(15,0)…(15,7)` 与 M32768 的 `(32,0)…(32,7)`。M16160 的 x=120…126 绕过 remap；M32768 整个 grid 都在内部区域。

代码注释画了 serpentine 顺序，但固定版本实际表达式没有按子网格奇偶翻转；以上按实际表达式计算，没有按示意图补出代码不存在的反转。可运行 [verify_mapping.py](scheduler-source/verify_mapping.py)先检查三份固定 header SHA，再按这些表达式枚举；[mapping-verification.json](scheduler-source/mapping-verification.json)记录 **6/6 通过**：两个官方尺寸的 H8/s1，以及 M16160 的 v6 AlongN/s8 / AlongM/s1，均为合法逻辑 tile 的一一置换并与已存 NCU physical grid 相同。每组保存完整枚举摘要 hash、首批名义邻居和 raw CSV SHA。

```sh
python _dev/docs/evidence/sm120-scheduler/scheduler-source/verify_mapping.py
```

这是纯 CPU 复写表达式的验证，不是调用编译后的 C++ scheduler，也不是新增 GPU 测试；它不能证明实际 CLC 执行先后或缓存命中。

## 与已有计时相连：可支持什么

[同进程官方三臂 31 对结果](official-tuning-v2/analysis.json)比较 old H8、新库 H8、新库 AlongM/s1，完整链均为 GEMM+V2 quant，并通过其数值及 changed-input Graph 门槛：

| M / 模式 | 新 H8 median ms | 新 AlongM/s1 median ms | M1/H8 成对时间比 |
|---|---:|---:|---:|
| 16160 / warm | 10.431584 | 10.226848 | 0.98015 |
| 16160 / 扰缓存 | 10.430624 | 10.233600 | 0.98108 |
| 32768 / warm | 21.953632 | 33.038433 | 1.50554 |
| 32768 / 扰缓存 | 21.984257 | 33.079296 | 1.50401 |

新增 [M32768 严格 NCU 汇总](ncu-large-m-summary.json)两组均通过 capture/export/source/hash/数值/Graph/window 检查，A/As fingerprint 相同。H8 resolves AlongM，因此它们的**有效调度参数仅 swizzle 8→1**，没有 tile、mainloop、raster 方向或 grid 变更：

| M32768 capture | GEMM ms | GEMM DRAM read GB | GEMM L2 request GB | Tensor active % | 完整 GEMM+quant ms | 完整 L2 request GB |
|---|---:|---:|---:|---:|---:|---:|
| Heuristic/s8 → AlongM/s8 | 18.264320 | 6.063568 | 95.039189 | 83.598 | 20.036896 | 98.034951 |
| 显式 AlongM/s1 | 31.113472 | 46.498232 | 95.036751 | 36.094 | 32.887232 | 98.031940 |

这是总 L2 request 近似不变、DRAM read 与 Tensor active 巨变的实际 memory-hierarchy 证据，与无 profiler 大 M 回归方向一致。它仍是每臂一次 kernel-replay capture，none cache / clock control；profiler 时间不是 API 时间，Tensor active 不是 peak-FLOP 利用率。

**分配到 A 的解释仍是假设：**唯一 FP8 A 的 footprint 从 **78.90625 MiB** 增至 **160 MiB**，As 从 **2.46582 MiB**增至 **5 MiB**；后者的 A 单独已经超过本卡 128 MiB L2。AlongM/s1 的名义布局沿 M 扫描完整 A 后进入下个 N tile；s8 内区则让同 A tile 在邻近 ID 中先服务 8 个 N tile。若实际领取/缓存行为足够接近此局部性，s1 在更大 A 中可能丢失跨 N 的 A 复用，而 s8 保留更短距离的复用。这与新计数一致，但**不能证明额外 DRAM 来自 A、不能保证其阈值、也不能排除其他动态调度/缓存/频率因素**；总 read counter 无 buffer attribution。

因此不能按“大 M 开 AlongM/s1”设置通用阈值：这两个大 M 的收益方向已经相反；当前数据只支持对固定形状有门槛的选择。后续扩展必须重新验证尺寸、尾部、资源与完整流程计时，不能把仅有两个点的 L2 footprint 假设变成 dispatch 规则；更不能把算子约 2% 换算成整模收益。
