# 16/17 号遗留实验的本地恢复审计

日期：2026-10-15。只读取本地计划、汇总、分析与部分原始 JSON；未连接服务器、运行 GPU 或改变服务。本页区分已有实测与下一步计划；[16](../../16-sm120-next-round-plan.md)、[17](../../17-remaining-experiments-plan.md) 是历史计划，不能当作新的执行授权。服务器目前无实验负载的状态由主线另行核查，本页没有独立远程确认。

| 方向 | 现有证据支持的状态 | 不能据此声称 |
|---|---|---|
| A：FP8 GEMM 上限 | [A 汇总](a-gemm-ceiling/results/summary.md)记录实际 A1 裸 MMA 两种 SASS 形式约 984–1000 TFLOP/s；A2 在决策形状 M=16160 gate_up 上，随机 FP8 单算子、同进程 31 对、未锁频，tensor-wise cuBLASLt 674 对 blockwise CUTLASS 619 TFLOP/s（×1.090），保持可比块缩放的 MXFP8 约 593。按 16 号预设的 10% 判据，停止本轮 GEMM 线有依据。 | 这不是所有 FP8 kernel 的硬件最高性能或生产权重可替代方案；tensor-wise 缩放语义不同，M=32768 gate_up 有 ×1.132 但不是本轮主决策形状，社区 A3 未测。不同负载时钟不同，裸 MMA 结果不能直接当整模上限。 |
| D：64K 草稿词表 | 从两份 [D4 首轮](d-draft-vocab/results/host/logs/d4_analysis_64.json)与[复测](d-draft-vocab/results-retest/logs/d4_analysis.json)的每 index `tps` 重算，B/((A1+A2)/2)−1 的 12 请求合并中位数分别 **+13.5495%**、**+13.7243%**，与原摘要相符；normal 各 6、thinking 各 6，warmup 排除。对应请求的跨臂 payload hash 一致。两轮 [D3 首轮](d-draft-vocab/results/host/logs/d3_compare.json)、[D3 复测](d-draft-vocab/results-retest/logs/d3_compare.json)均为 64K vs 基线每 pass 16/16 token 一致；[额外比较](d-draft-vocab/results-retest/logs/x_compare_64k.json)的 24 条新合成请求在非流式与流式各 24/24 一致。 | “无损 +13.5%”只成立于上述固定 FP8 模型、MTP4、greedy、单卡隔离、合成/公开语料词表和该请求集合；不是 MoBase 业务质量或任意输入/采样/长期运行保证，也尚未部署生产。额外 24 条均 `ignore_eos` 定长；长输出自然 EOS 与工具调用未覆盖。首轮 32K D3 仅 4/16、复测 16/16，根因未明。设备端 fallback 计数器未接线，STRICT 与日志仅为间接证据。 |
| P1：前缀复用 | [P 汇总 E1](p-prefill/summary.md#e1p4-分块规则--p1-前缀复用)及 [E1 分析](p-prefill/analysis/e1-analysis.md)显示重复请求 S1 的 19K/44K TTFT 从 2.33/6.39 s 到约 0.377/0.383 s，命中 16160/42016；S2 是否命中强烈依赖先前落下快照的请求顺序；S3 第二轮有效，第三轮复用没有继续增长。 | 严格逐 token 正确性并未全面通过：S2 多数命中请求与冷路径不同，MTP4 S3 第三轮 19K 分叉。`summary.md` 把分叉写成“完全由 chunk 边界造成”，现有日志、冷路径换分块的噪声底与近并列 logprob 支持**关联/合理解释**，但不足以排除其它贡献或证明业务质量可接受；须保留未过 gate 的判定。 |
| P2：注意力 | [P 汇总 E2](p-prefill/summary.md#e2注意力后端)、[E2 分析](p-prefill/analysis/e2-analysis.md)记录 FlashInfer nsys BF16 Q/FP8 KV 的 34 个 prefill kernel 合计约 326 ms；FLASH_ATTN 在固定 FP8 KV 配置下启动失败；TRITON_ATTN 可启动但 19K/44K/64K/128K 配对 TTFT 慢 16.9%/33.8%/43.0%/63.7%，16 条 greedy 对照只有 2/16 输出相同。 | “FlashInfer 唯一可用后端”文字过强：Triton 可用但慢且输出不同。准确表述应是“所测三种中唯一通过当前速度/输出选择门槛的后端”。未做 NCU；219 TFLOP/s 是由模型 FLOP 公式与 nsys 时长估算，不是实测硬件利用率；FP8 QK、新 tile 尚未试。 |
| P3：小算子融合 | [P 汇总 E3](p-prefill/summary.md#e3v2-siluquant-跨时段复测)记录 V2 SiLU+quant 第二时段 12 条 19K prefill 配对 TTFT −1.443%，与先前 −1.414% 接近，decode 约不变。另有 [F 源码/变更](f-fusion/CHANGES.md)及 [F A/B/A](f-fusion/results/aba-comparison.json)：新 norm/conv 融合候选已经存在并有单时段 API 数据，非“尚未开始”。 | V2 仍只可列候选；未做生产 0.45 显存共存、广泛业务质量。F 的 A/B/A 中有 embedding 活动窗口，且摘要承认臂间输出 hash 差异、无跨时段重复；不能直接把 F 结果算成可部署加速。本页未对 F 原始逐请求证据做完整复算。 |
| P4：分块 | [P 汇总 E1](p-prefill/summary.md#e1p4-分块规则--p1-前缀复用)的启动日志、只读 `SPLITPROBE` 与源码规则支持 MTP4 的 1616 token 块对齐及 EAGLE 少一个块，19K 首块 16160；无前缀缓存时单块。ABAB 同 prompt 19K/44K 冷 TTFT 单块约快 1.415%/1.062%。 | 关闭前缀缓存只是机制诊断，会丢掉命中收益；不应推荐直接用作生产优化。部分精确边界与 MTP off 条件依赖当前 vLLM 版本。 |
| D5 / D6 | [17 号计划第 6、8 节](../../17-remaining-experiments-plan.md#6-decode-收尾小项)只有“64K+MTP5/6”和“64K+DFlash15”的拟议条目；本地 `sm120-next`（排除下载 venv）的文件名与现有 D/P 汇总没有 D5/D6 运行、输出或完成标记。 | 不能写成已启动、已测或已获授权。此为本地证据缺失判断，不声称远端绝对不存在。 |

复测时段的 [D4 JSON](d-draft-vocab/results-retest/logs/d4_analysis.json)中，A1 thinking 第 3 个正式请求窗口有一次 embedding 11-token 请求，不能说全部窗口无干扰。仅作为敏感性分析，剔除该配对后合并中位变为 **+14.0714%（n=11）**，不改变正向观察；原 +13.7243%（n=12）应保留并披露。embedding 前后身份/启动/重启与功能检查是当时的服务证据，不代表此刻的远端状态。

## 可恢复清单（后续须重新确认授权与现场状态）

1. 以本页和原始证据作为进度基准；先核对当前服务器服务/容器、embedding 健康与资源占用。原 27B、9B、ASR 依用户当前要求保持停止，embedding 不停；本页不触发任何启动。
2. 若继续 P1，先决定是否值得做长上下文质量回归；S2/S3 第三轮已违反原定逐 token 门槛，不能在未解释/评估前宣称可用。若继续 P2，保持当前 FlashInfer，新的 FP8 QK/tile 改动须单独数值和整模配对验证。
3. 若继续 P3，V2 与 F 是不同候选；先独立审 F 的数值、输出差异和 embedding 窗口，且要跨时段复测，不能把 V2 的 −1.4% 自动加到 F 上。
4. D64K 可作为下一轮候选基准。新的 D5/D6 需单独确定实验设计、模型适配、质量门槛与现场隔离；不要把 17 号计划当执行许可。32K 分叉、采样/长自然停止/业务质量、设备回退计数仍是已知缺口。

本审计没有重新执行所有客户端或 profiler；离线重算了 D4 中位数，抽核 D3/E2 的 JSON 与历史汇总，并检查 D5/D6 本地文件证据。所有速度百分比均限于对应请求和实验环境。
