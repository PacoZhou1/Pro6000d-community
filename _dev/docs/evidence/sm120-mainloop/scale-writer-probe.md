# Row-scale 写者地址纯 host 诊断设计

`scale_writer_probe.cu` 已在固定 CUDA 环境作为**纯 host 程序**完成五组编译与运行。它比 `ownership_probe.cu` 多重建了固定 CUTLASS epilogue 的 scale 地址路径：从 `Sm1xxBlockScaledOutputConfig<128>::tile_atom_to_shape_SFD` 的真实 SFD layout，按固定 `Sm90TmaWarpSpecialized` 的 `make_tiled_copy_C_atom`、`make_tiled_copy_S(CopyOpR2S)` 和 `sm90_partition_for_epilogue<true>` 得到每个线程的 `tCgSFD`。枚举的是 `copy_aligned` 的静态 `AutoFilter` 消去零步长广播后的实际目的地址；检查一个完整 64×256 CTA tile 的 64×2 个 scale 地址，每址是否恰好有一个写者。

源对应 `gemm-epilogue/source/cutlass/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp:682–699` 与 `gemm-epilogue/build-gateup-coop-v6/include/cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp:473–497,427–447`。宏 `PROBE_WRITER_MAX_WARP` 手工固定为真实 shadow header 的 predicate：control42/v2 是 4，v3 是 2；构建记录必须附被包含的生成头、shadow visitor 和 probe 的 SHA。该宏只在完整 tile 的残量判断恒真时等效；边缘 CTA、实际 device `copy_aligned`/同步、以及 FP8 scale 数值都不由纯 host 检查证明。

在 root 的固定 CUDA 实验环境，**每个 variant 分开**对其已生成的 build include、`standalone_dispatch.cuh`、固定 CUTLASS include 编译，且不要覆写旧 v2/v3 编译产物。例如 v3 的 warp24x2：

```bash
nvcc -std=c++17 -O0 --expt-relaxed-constexpr -arch=sm_120a \
  -DPROBE_NAMESPACE=pro6000d_mainloop_warp24x2_v3 \
  '-DPROBE_LABEL="warp24x2_v3"' -DPROBE_WRITER_MAX_WARP=2 \
  -I /work/build-warp24x2-v3/include -I /work/build-warp24x2-v3 \
  -I /usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include \
  -I /usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/tools/util/include \
  /work/sm120-mainloop/scale_writer_probe.cu \
  -o /results/scale-writer-warp24x2-v3
/results/scale-writer-warp24x2-v3 > /results/scale-writer-warp24x2-v3.json
```

对照组用 `control42-v2` 生成头、`PROBE_NAMESPACE=pro6000d_mainloop_control42`、`PROBE_WRITER_MAX_WARP=4`；2×4 v2 仍用 4，2×4 v3 的私有 namespace 带 `_v3` 后缀且 writer 上限为 2。最稳妥是直接读取每个 `build-manifest.json` 的 `private_namespace`，如 `run_scale_writer.py` 所做。不同 build 的头和 namespace 不能交叉。

实际 [writer-v1 原始状态](writer-v1/writer-status.json)与各组独立 JSON 均已落盘：五组 `nvcc` 编译、纯 host 运行退出码均为 0；记录的源码 SHA `efcd9c12…1e5dd1f6` 与当前 `scale_writer_probe.cu` 一致，五个 raw JSON 与汇总逐字段相同。control42 的 128 个预期地址各写一次；两个 2×4 v2 均有 256 次写入、128 次重复，首个重复地址由线程 0 和 64 写；两个 2×4 v3 均为 128 次写入、无重复且 128 地址全覆盖。各组均无预期地址缺失、无意外地址。这独立于 `partition_C`，说明把 v3 写者范围收窄为 warp 0/1 修正了**完整 CTA tile 的目的地址几何**。它仍不能证明边缘 CTA、实际 device `copy_aligned`/同步及长期无竞争；v3 的 GPU 数值与改输入 Graph 门槛由另一个实验执行，不可把本 host 结果当成其替代。计时结论也须等完整链原始记录，不能从 writer 几何推出收益。若将来 host AutoFilter 与实际 device copy 发现差异，最小补证是给隔离 visitor 增加仅诊断用的 `(CTA,row,group,tid,address)` 输出或 per-address 原子计数，并单独哈希、不进入计时。
