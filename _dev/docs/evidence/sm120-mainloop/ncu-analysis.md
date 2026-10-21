# SM120 mainloop 五臂 NCU 机制诊断

只基于五份原始 NCU CSV（[官方](ncu-official-v1/raw.csv)、[control fused](ncu-control-fused-v1/raw.csv)、[control BF16](ncu-control-paired-v1/raw.csv)、[candidate fused](ncu-candidate-fused-v1/raw.csv)、[candidate BF16](ncu-candidate-paired-v1/raw.csv)）和可重算的 [逐项解析](ncu-analysis.json)。[analyze_ncu.py](analyze_ncu.py) 锁定固定 `../sm120-scheduler/analyze_ncu.py` 的 SHA，复用其单位归一化，再要求本轮 `analyze_mainloop.py` 的合成数值／改输入 CUDA Graph／单次 NVTX 窗口门槛通过。五份作业的 capture、export、服务保护均成功，原始 CSV 精确对应官方两 kernel、control fused 两 kernel、control BF16 一 kernel、candidate fused 两 kernel、candidate BF16 一 kernel；每个 kernel 的 CSV 范围都带选定 `SM120_MAINLOOP` NVTX。输入指纹同为 `e8c90231…1b95714`，weight 指纹、cache/clock 选项相同；源码快照、build manifest、库 SHA 与 `.ncu-rep` 均逐项记录。这里的 candidate 是 `warp24n64_v3`（2×4 AtomLayout **同时**改 PermN64），control 是 4×2/PermN32；两者 raster AlongM、swizzle 1，官方是固定原 H8。不能把两项布局变化拆成单变量贡献。

| M16160 单次 NCU 范围 | kernel | NCU 时长 ms | DRAM 读/写 GB | L2 请求 GB | shared 读 wavefront（亿） | shared 读 bank conflict（万） | Tensor 指令（亿） | 寄存器/线程；shared/CTA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原官方 BF16 GEMM＋V2 quant | 2 | 9.651 | 2.084/1.336 | 48.609 | GEMM 11.179 | GEMM 78.83 | GEMM 7.075 | GEMM 168；78,848 B |
| 4×2 fused GEMM＋pack | 2 | 10.494 | 0.539/0.797 | 58.003 | GEMM 16.224 | GEMM 206.14 | GEMM 7.047 | GEMM 168；102,400 B |
| 2×4/PermN64 fused GEMM＋pack | 2 | 10.271 | 0.539/0.797 | 58.004 | GEMM 11.205 | GEMM 118.06 | GEMM 7.047 | GEMM 168；102,400 B |
| 4×2 普通 BF16 GEMM 诊断 | 1 | 9.782 | 0.246/1.096 | 57.860 | 16.200 | 74.98 | 7.047 | 168；94,208 B |
| 2×4/PermN64 普通 BF16 GEMM 诊断 | 1 | 9.477 | 0.248/1.096 | 57.865 | 11.132 | 39.93 | 7.047 | 168；94,208 B |

表中 GB 为十进制，按 CSV 已四舍五入的值归一化；shared 读、bank conflict、Tensor 指令和资源列只指对应 GEMM kernel，不把 pack/quant 的 0 或其他计数混入。shared/CTA 包含 1,024 B driver 保留；fused 动态 shared 为 101,376 B，BF16 诊断为 93,184 B。完整原始计数还保留 shared 写、local sectors、所有已选 stall 百分比、grid/block 与每个 kernel 的全量 metric map，在 JSON 的 `jobs[].kernels[]` 下。

同范围候选对 control：fused GEMM 的 shared 读 wavefront 为 0.691×、shared 读 bank conflict 0.573×，普通 BF16 GEMM 分别为 0.687×、0.533×；两对 Tensor 指令数均完全相同。shared 写 wavefront 在 fused 为 9,914,064→11,010,675、BF16 为 10,480,517→10,466,751；shared 写 bank conflict 分别为 4,560→115 和 1,672,069→1,658,303。fused 完整链 DRAM 总字节比为 0.9998×、L2 请求字节比为 1.00001×；因此这些数据支持“shared 访问形态改变”，没有证据称 B weight DRAM 字节减少，也没有提供 buffer 级归因。MIO-throttle 样本百分比在 fused GEMM 为 2.156%→0.332%、BF16 GEMM 为 1.269%→0.479%，它们是 warp 状态指标，**不是**耗时占比。NCU 单次 replay 的完整链 10.494→10.271 ms、BF16 诊断 9.782→9.477 ms 只作同配置核级观察；计时会受 profiler replay、cache/clock 和抽样波动影响，不能替代未 profiler 的 31 对完整链结果，更不能称 API 或整模收益。BF16 诊断仅一颗 GEMM，不能和 fused/官方完整链直接比速度。

尤其不能把 NCU 内 fused GEMM 的 9.738464→9.516352 ms（−2.28%）直接归因于流水周期缩短：该指标同时报告 `sm__cycles_elapsed.avg.per_second` 从 2.262197→2.316136 **GHz**（+2.38%）。以“时长 ms × 频率 GHz”粗估，分别约 22.0303 与 22.0412 **百万 cycles**，后者约 +0.05%；Tensor active 为 68.0567%→68.0232%，`smsp__inst_executed.sum` 还从 3,499,803,845 增至 3,548,780,577（+1.40%）。这是把 NCU 已归一化、可能来自不同 replay pass 的时长和平均频率相乘，**只能作近似一致性检查，不能作精确周期或因果分解**；更不能据此把 shared wavefront 降低等同于端到端周期收益。[未 profiler 的 31 对完整链计时](timing-v3/analysis.json)在 M16160、同 AlongM/s1 候选相对 control 的 paired 中位为 warm 1.02574×、扰缓存 1.02587×，是该次状态下约 2.6% 的实际观测；它仍无法区分动态频率／功耗状态与每请求周期数的贡献。不需要为此修改全局时钟设置。

本地重算：

```bash
python3 _dev/docs/evidence/sm120-mainloop/analyze_ncu.py \
  --root _dev/docs/evidence/sm120-mainloop \
  --output /tmp/sm120-mainloop-ncu-review.json
```
