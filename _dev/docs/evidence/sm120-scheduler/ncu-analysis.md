# M16160 scheduler / epilogue：NCU 证据

这六个 capture 表明：**AlongM/s1 的融合 pipeline 总 DRAM 计数显著低于 AlongN/s8，但在 NCU 中仍慢于两种官方完整 pipeline。**同 tile 的普通 BF16 GEMM 已有相近的 Tensor active 和大部分主循环计数，不能把 M64N256 与 M128N128 的全部差异归因于 SiLU、量化或 pack。本文仅解释 profiler 计数；不据此宣称 API、prefill、decode 或整模收益。

## 验收与可重现入口

[analyze_ncu.py](analyze_ncu.py)只在 CPU 读取已保存的数据。六个成功 job 由 [ncu-accepted-jobs.json](ncu-accepted-jobs.json)显式选择；[ncu-summary.json](ncu-summary.json)为 **6/6 usable、全部证据检查通过**，以严格模式执行并返回 0：

```sh
python _dev/docs/evidence/sm120-scheduler/analyze_ncu.py \
  --jobs _dev/docs/evidence/sm120-scheduler/ncu-accepted-jobs.json \
  --output _dev/docs/evidence/sm120-scheduler/ncu-summary.json
```

验收联查：capture / export 都退出 0、容器清理、服务检查、全部 source snapshot SHA、CLI / metadata / compiled manifest 的库 SHA、full-bit 数值及 changed-input Graph、唯一 `profiled_once` / NVTX / `call_count=1`、CSV units 行、连续唯一 ID、实际 **1 或 2** 个 kernel 与对应角色。kernel 名称可能缩短 namespace，成员身份主要来自冻结命令与 window 证据，名称只作交叉核对。请求计数有缺失或未知单位即拒绝。CPU 反例验证多余 kernel、损坏 DRAM 单位、错误 ID 均被拒绝，不能将这些反例检查说成 GPU 测试。

七次尝试另存 [ncu-attempts.json](ncu-attempts.json)、[ncu-inventory.json](ncu-inventory.json)。原 [AlongN/s8 失败](ncu-m16160-v6-along-n-s8/status.json)为 **exit 11、无导出/计数、`not_known` root cause**；stderr 的 traceback 在 legacy launch 中，但没有把未知根因自动归到某一历史符号问题。它未通过数值/NVTX capture 验收，不能拿去作计数比率。成功重跑为 [AlongN/s8 isolated](ncu-m16160-v6-along-n-s8-isolated/status.json)，保留了失败目录。

失败盘点可使用 `--allow-incomplete`，其 JSON 的 `all_jobs_usable` 仍为 false。六成功汇总使用严格模式，**没有**以该开关替代最终通过。

## 实际计数

M=16160、N=34816、K=5120，FP8 输入与 FP32 scale；所有成功 capture 的 A/As fingerprint 相同，为 `e8c9023113f4e6b803d7326a17d03af5cd1d991ad88bda3b453bd065f1b95714`。scheduler 库统一为 `af1d814966a746ff09edc55d40196b871067d88d22ed0e50700c1f3a4eb185d6`。cache-control / clock-control 都为 none；这些协议一致不代表缓存、温度或频率状态严格相同。

单位：时间 ms；字节数十进制 GB。CSV 格式的小数精度会限制归一化精度，表格按可读位数舍入。旧官方为 Heuristic/s8；新官方为同 M128N128 的 AlongM/s1。两个 paired BF16 行仅计 GEMM，**不含 inverse 或 V2 quant**。

| capture | kernels / 范围 | GEMM ms | 后续 kernel ms | 总 ms | DRAM read GB | DRAM write GB | DRAM total GB | L2 request GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [旧官方](ncu-m16160-official/raw.csv) | 2 / GEMM+V2 quant | 8.730720 | 0.851584 | 9.582304 | 2.133568 | 1.331335 | 3.464903 | 48.611514 |
| [融合 AlongN/s8](ncu-m16160-v6-along-n-s8-isolated/raw.csv) | 2 / GEMM+pack | 10.693408 | 0.754624 | 11.448032 | 6.687935 | 0.826984 | 7.514919 | 57.979029 |
| [融合 AlongM/s1](ncu-m16160-v6-along-m-s1/raw.csv) | 2 / GEMM+pack | 9.701280 | 0.754208 | 10.455488 | 0.559800 | 0.797720 | 1.357520 | 58.003499 |
| [paired BF16 AlongM/s1](ncu-m16160-paired-along-m-s1/raw.csv) | 1 / GEMM-only | 9.729376 | — | 9.729376 | 0.260286 | 1.095685 | 1.355971 | 57.859705 |
| [paired BF16 AlongN/s8](ncu-m16160-paired-along-n-s8/raw.csv) | 1 / GEMM-only | 10.629120 | — | 10.629120 | 6.415156 | 1.119590 | 7.534746 | 57.821394 |
| [新官方 AlongM/s1](ncu-m16160-official-along-m-s1/raw.csv) | 2 / GEMM+V2 quant | 8.513664 | 0.855008 | 9.368672 | 1.382811 | 1.337539 | 2.720350 | 48.606109 |

完整 pipeline 的 profiler 比率（分母旧官方）：融合 AlongN/s8 的 duration 为 **1.1947×**、DRAM total **2.1689×**；融合 AlongM/s1 的 duration 为 **1.0911×**、DRAM total **0.3918×**；新官方 AlongM/s1 为 **0.9777× / 0.7851×**。融合 AlongM/s1 相对新官方的 duration 为 **1.1160×**、DRAM total 为 **0.4990×**。这些是同形状、同输入 hash、相同 declared cache/clock policy 下的单次 capture 比率。

### GEMM 计数与资源

下表只摘 GEMM；完整每-kernel instruction pipes、shared wavefront / bank-conflict、stall、local sectors、LaunchStats 与 device attributes 全部在 [ncu-summary.json](ncu-summary.json)。指令/wavefront 按十亿计；Tensor active 与 warp occupancy 均保留其 NCU 原始百分比口径。

| GEMM | Tensor active % | total inst ×10⁹ | Tensor inst ×10⁹ | FMA inst ×10⁹ | ALU inst ×10⁹ | shared-load wavefront ×10⁹ | warp occupancy % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧官方 M128N128 | 83.558 | 3.019 | 0.707461 | 0.787 | 0.309 | 1.118 | 20.834 |
| 新官方 M128N128 AlongM/s1 | 82.993 | 3.005 | 0.707461 | 0.782 | 0.302 | 1.118 | 20.874 |
| paired BF16 M64N256 AlongM/s1 | 68.895 | 3.162 | 0.704676 | 0.773 | 0.315 | 1.620 | 20.900 |
| paired BF16 M64N256 AlongN/s8 | 69.093 | 3.181 | 0.704676 | 0.773 | 0.315 | 1.620 | 20.803 |
| fused M64N256 AlongM/s1 | 68.112 | 3.500 | 0.704676 | 0.904 | 0.414 | 1.622 | 20.811 |
| fused M64N256 AlongN/s8 | 67.024 | 3.519 | 0.704676 | 0.904 | 0.414 | 1.621 | 20.880 |

三类 GEMM 均 384 threads / CTA、168 registers / thread；user dynamic shared 分别为官方 **77824 B**、paired **93184 B**、fused **101376 B**，每 CTA 另有 driver **1024 B**。NCU 的 register / shared occupancy limits 均为 1 block；用户 storage 与 driver storage 不能重复相加或混同。编译器 spill 与 NCU local sector 计数分别留证，不能用静态 spill 字节替代执行流量。

在同 M64N256 与同 scheduler 参数下，paired 和 fused 的 Tensor instruction 总数相同，普通 BF16 路径也已有约 69% Tensor active 和约 1.62×10⁹ shared-load wavefront。fused 相比 paired 多出普通计算指令与不同输出/资源布局，支持“融合额外工作可见”的诊断；但 BF16 与 FP8 epilogue 的 store 字节和 shared storage 同时不同，不能把时间差完整解释为 SiLU/量化成本。Tensor active 是 active cycles 指标，不能直接当作 Boost 峰值 FLOP 利用率。

## 解释边界

AlongM/s1 与 AlongN/s8 的 M64 路径均出现大幅不同的 DRAM read、而 L2 request 总数接近：该现象在普通 BF16 和 fused 中都存在。它支持继续把 tile、raster/swizzle 和 memory hierarchy 放在同一设计中审查；**两个参数同时变化且只有单次 capture，不能单独证明 raster 或 swizzle 的因果，也不能把总 read 数指定为 B 权重重复读取。**

从完整窗口看，减低 DRAM total 并未在本轮 NCU 中让融合超过官方。pack 仍约 0.754 ms，而 M64N256 普通 BF16 GEMM 本身也没有达到 M128N128 的 profiler 时间与 Tensor active。因此只删 pack 的假设不能由这些计数得到支持；下一步仍以独立无 profiler 完整 pipeline 测量作性能门槛。

新 isolated capture 只加载选定 arm，metadata 明确 `numerical_arms` / `excluded_arms`；旧成功官方 capture 保留旧快照中额外 legacy 检查。解析器逐 job 校验自己的冻结 source 和实际检查，未机械要求所有调用进程完全相同。每次 NCU 的数值/Graph 验证只覆盖该 capture 声明的形状和输入；不能扩成全 FP8 输入域或整模逐位相同。

NCU replay 时间、stall pct 和总 DRAM/L2 都不等于 API wall time；stall pct 也不能相加成 API 时间份额。默认 none 缓存策略不构成固定 cold cache，单次窗口没有重复样本方差。这些 synthetic 算子计数不构成模型入场或上线证明，BF16 inverse / V2 量化的拆分与无 profiler 结果需看独立证据。
