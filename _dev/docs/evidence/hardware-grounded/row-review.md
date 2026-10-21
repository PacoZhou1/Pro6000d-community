# SM120 硬件优先调查：01–03 中期独立核查

审查时间：2026-10-14。独立 Sol 子 agent 仅本地读取原始请求、当前 13 号报告、固定 CUTLASS 源码、CPU 模型脚本、NCU CSV/状态及 NVIDIA 一手资料；没有 SSH、GPU、服务操作。本文件不作整轮最终验收。

## 判断

截至本次快照，01–03 的核心声明有来源和原始计数支撑，**未发现阻断性、可证伪缺口**。特别是 M16160 的 v6 DRAM 放大是真实 NCU 汇总，不是把 M2741 的节流结论外推；raster 方向翻转有固定源码依据，但因果关系仍被正确标为待实验假设。下面一处术语建议收紧，不构成阻断。

## 逐项证据

1. **设备与峰值。** [NVIDIA RTX PRO 白皮书 Table 4](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf)的 Workstation 最右列给出 188 SM、96GB GDDR7、1792GB/s、dense FP8/FP32-accumulate **1007.6 TFLOP/s**，斜线后的 2015.2 属 sparsity；约 4030 的 FP4 广告值不适用于本任务。[NVIDIA CUDA GPU 列表](https://developer.nvidia.com/cuda/gpus)区分 Workstation CC12.0 与 B200/GB200 CC10.0。`device-attributes.json` 指向 `ncu-m2741/raw.csv` 的 SHA 一致，实际记录为 188 SM、65536 registers/SM、102400B shared/SM、101376B opt-in shared/block、48 warp/1536 thread/24 block 每 SM、128MiB L2。`sm120-facts.md` 同时披露 CUDA 文档的 24 与 32 block/SM、100 与 128KiB shared/SM 冲突，并以本机属性和 Programming Guide 表作为设计约束；没有混用 SM100 的 TMEM/`tcgen05`。cluster_launch=1 与 generic multicast_supported=0 没有被冒称为 RTX PRO TMA multicast 的最终能力判定。
2. **资源与形状算术。** `resource_model.py` 从保存的旧版 NCU kernel 行提取 launch 资源；重新核对 `resource-model.json`：官方 M128N128、v6 M64N256 的面积都是 16384，8 个 MMA warp/256 线程意味着每线程至少 64 个 FP32 accumulator；两者均 384 总线程、168 registers/thread，名义 64512 registers/CTA。官方/v6 shared 分别 78848/102400B，NCU 两种 occupancy resource limit 都为 1 CTA/SM；不能因此推得相同时长。两 stage 原始 FP8 A+B 数据分别是 65536/81920B；M128N256 的两 stage 单 operands 已为 98304B，离 101376B opt-in 仅 3072B，不足以直接宣称新的大 tile 可行。M2741 CTA 数 5984/5848 与真实 NCU grid 相同，M16160 为 34544/34408；逻辑 A/B 重复请求与实测 L2/DRAM 在文档中分开。`roofline.py` 对实际模块尺寸文件做 SHA/shape 检查，dense FP8 理想峰值与带宽给出的交点 562.28 FLOP/B、数值表均可从公式复算。
3. **raster 的实际语义。** 固定 `standalone_dispatch.cuh` 只设 `max_swizzle_size=8`，没有显式 raster；固定 `tile_scheduler_params.h` 的 `get_log_swizzle_size` 及 `get_rasterization_order` 对 M2741 的官方/v6 网格分别得 (24,272)/(48,136)，都是 AlongM；M16160 得 (128,272) AlongM / (256,136) AlongN。`sm90_tile_scheduler.hpp` 索引映射表明 AlongM 的 minor/快速变化坐标是 N、AlongN 的是 M。报告没有把名称直译为权重复用方向，也没有把翻转当成 DRAM 放大的既定原因。
4. **NCU 分母、单位、失败。** `ncu-m2741/raw.csv`、`ncu-m16160/raw.csv`、`ncu-m16160-repeat/raw.csv` 均为顺序 ID0 官方 GEMM、ID1 官方 V2 量化、ID2 v6 GEMM、ID3 v6 pack；status 记录 exit/export=0、自有容器清理，命令为同镜像/固定库、`--clock-control none --cache-control none`。原始 CSV 的 DRAM 在 M2741 用 Mbyte、M16160 用 Gbyte；`analyze_diagnostics.py` 转十进制 byte 后的两核求和与报告吻合：M2741 基线/v6 **0.55894/0.31551 GB、1.62938/1.78992ms**；M16160 **3.40441/7.51066 GB、9.48816/11.41584ms**；独立同策略再采 **3.40941/7.44995 GB、9.50448/11.34848ms**。M16160 GEMM DRAM 读 **0.95077→6.38496 GB**；但计数只涵盖该合成尺寸的整段两核 NCU kernel replay，不能按 buffer 归因、不能充作真实 API 时延。`ncu-m16160-cache-all/status.json` 退出 11、无有效 CSV，报告如实披露失败。总 warp 指令、tensor 指令、shared wavefront、issue 与 stall 仅按各自单位/分母比较；stall 百分比没有被拆成时间份额。

## 一处非阻断的表述收紧

`roofline.py` 的 `ideal_dram_floor_ms` 和主报告表头“理想 DRAM 时间下限”，实际计算前提是**该单次 GEMM 的 A、B 至少各从 DRAM 读取一次、D 写一次且按广告峰值带宽传输**。热 L2 命中时，这不是无条件的 DRAM 时间下界。`sm120-facts.md` 已明确“不适用于所有缓存状态的严格下界”，主报告正文也列出一次读写假设；建议将表头写为“单遍冷读假设下的理论传输时间”，避免读者跳过前提将其用作实际内存瓶颈判据。无需改变已算数值。

余下审查边界：本次没有重新运行 GPU/NCU，未验证未实现的 raster/swizzle 消融、cluster multicast 或新的 epilogue；01–03 后续若代码、表格或原始数据再改，应在最终 REVIEW 中复核最终版本。
