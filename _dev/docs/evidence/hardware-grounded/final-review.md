# SM120 硬件优先调查：独立愿景评审

- review_agent_mode: direct-spawn-agent
- review_independence: strong
- review_actual_model: gpt-6.1-sol
- user_model_override: 用户明确指定 Sol 子 agent；本审查实际使用 Sol，覆盖 skill 的默认同模型偏好。
- claim_ledger: claims.json
- claim_coverage: 5/5
- claim_coverage_status: complete
- review_result: vision_met
- handoff_humanized: false（humanizer-zh 未安装，保留独立审查者生成的中文草稿）
- source_ref: `request.md`、同前缀 CSV 01–05、`claims.json`、`log.md`。
- reviewed_commits: `4fdf3e0`, `0316cfa`, `415ca85`, `94dd25e`, `c64241a`。
- review_scope: 本地只读源码、文档、原始 CSV/JSON/日志及已有服务快照；没有 SSH、GPU、服务操作，也没有新实验。

## 结论

**vision_met，仅指本轮“从 RTX PRO 6000 SM120 规格和实际资源出发，补诊断并提出可证伪下一步”的调查目标。** 本轮未实现新 raster/epilogue 变体，未得到生产提速补丁，也没有把上一轮 v6 的负整模结果改写成收益。独立重算确认：合成真实形状 NCU 的 M2741 完整两核 DRAM 是基线/v6 0.559/0.316 GB；M16160 首次为 3.404/7.511 GB，独立同策略复测为 3.409/7.450 GB，v6 约 2.19–2.21 倍。报告将调度/复用作为待验证解释，而非已证因果。最新服务策略下，原 27B、9B、ASR 在结束及延后检查时保持停止，embedding 保持原身份且两次功能检查通过。

没有发现需要补 issue 的可证伪差距：`gaps: []`。未完成的新调度消融和新 kernel 属于报告明确列出的**下一轮工作**，不是本轮已做承诺或本轮失败。

## 五项声明的证据对账

| 项 | source_ref | evidence_ref 与独立检查 | 结论 |
| --- | --- | --- | --- |
| 01 硬件规格与能力口径 | `request.md`; CSV 01 | `sm120-facts.md`、`device-attributes.json`、`ncu-m2741/raw.csv`。NVIDIA RTX PRO 白皮书 Table 4 的 Workstation 列为 188 SM、1792 GB/s、dense FP8/FP32 accumulate 1007.6 TFLOP/s；sparse FP8 2015.2、sparse FP4 4030.4 不混用。设备属性源 CSV 的 SHA 匹配，实机为 CC12.0、每 SM 65536 registers、102400B shared、48 warp/1536 thread/24 CTA、整卡 128MiB L2。`sm120-facts.md` 披露 CUDA 两份官方文档冲突、SM100 `tcgen05`/TMEM 不可移植，并将 cluster launch 与具体 TMA multicast 可行性分开。 | 覆盖；官方峰值不是持续实测速率 |
| 02 资源、实际形状与算术模型 | `request.md`; CSV 02 | `resource_model.py`, `resource-model.json`, `resource-model.md`, `roofline.py`, `roofline.json`；脚本核对旧真实模块归因文件 SHA 和相应 M/N/K。官方 M128N128 与 v6 M64N256 各 16384 输出元素、每 256 MMA 线程至少 64 个 FP32 accumulator，384 线程×168 registers=64512 registers/CTA；NCU shared 分别 78848/102400B，均被报告为 1 CTA/SM。M2741 真实 NCU grid 与模型 5984/5848 对齐；M16160 网格为 34544/34408。固定 CUTLASS scheduler 源码的 swizzle/raster 规则对应 M16160 官方 AlongM、v6 AlongN；AlongM 的快速变化坐标实际为 N。逻辑 A/B 请求、L2、DRAM 分列，未将 raster 翻转当作流量放大的已证原因。冷读模型表头已按上次中期反馈改成条件传输时间；FP8 理想交点 562.28 FLOP/B 可复算。 | 覆盖；模型不等于缓存/吞吐实测 |
| 03 定向 NCU 与失败边界 | `request.md`; CSV 03 | `diagnostic_control.py`, `analyze_diagnostics.py`, `capture-inventory.json`、三个成功目录的 `raw.csv`/`status.json` 及失败 `ncu-m16160-cache-all/status.json`。逐个核对捕获清单的 status SHA、raw CSV 与汇总 SHA、四核 ID0/1 官方 GEMM+V2量化、ID2/3 v6 GEMM+pack。M2741 CSV 用 Mbyte，M16160 用 Gbyte；转十进制 byte 后分别得到完整两核时间 1.629→1.790ms、9.488→11.416ms、9.504→11.348ms，DRAM 0.559→0.316GB、3.404→7.511GB、3.409→7.450GB。三次命令均 `cache-control=none`、未锁频；额外 `cache-control=all` 返回 11、无有效计数和数值完成，报告未拿它证明缓存因果。指令、shared wavefront、issue、stall 的单位/分母与原始 CSV 一致，stall 百分比未冒充 API 时间份额。 | 覆盖；仅合成真实形状的 NCU kernel replay，不是线上模型计数 |
| 04 下一轮硬件优先决策 | `request.md`; CSV 04 | `13-sm120-hardware-roadmap.md`, `README.md`。报告利用 M2741 与 M16160 相反的 DRAM 趋势、同 1 CTA/SM 的资源边界、M1/5 与大 M 的不同理论预算，先排同 tile 的 raster/swizzle 单变量试验，再分离普通 BF16 epilogue 与融合输出成本，最后才考虑直接紧凑写回/cluster。每一步给数值、完整流程、DRAM/L2 与无 profiler 整模准入/否决条件；明确这些变体尚未执行、不预言速度。 | 覆盖；优先级是可证伪设计而非效果保证 |
| 05 服务约束与清理 | 最新 `request.md`; CSV 05 | `hardware-before-services.json`、四次 NCU 各前后快照、`final-services.json`、`final-services-later.json` 共 11 份：全部 `passed`，四服务 ID/重启计数一致，三个原模型均 stopped，embedding 均 running。结束与 418.426882 秒后的实际 embedding API 均 200、1024 维、有限值，health200；原启动时间 `2026-10-06T09:31:08.313413609Z` 与原重启计数 1 未变。四个诊断 status 均记录自有容器已不存在；`verification.json` 的 11 快照/3 有效采集/1 失败与原文件一致。 | 覆盖；时点/功能证据，不宣称连续可用性或零干扰 SLA |

## 推断和证据的边界

- 公开规格的 1007.6 TFLOP/s 和 1792 GB/s 只是 Boost 条件下广告峰值；`roofline.py` 的 A/B 各一次冷读假设不适用于所有热缓存状态。GPU Tensor pipe busy 百分比不是已达峰值比例。
- NCU 每臂两核时长之和属于 profiler kernel replay，cache 未控制为固定冷态、时钟未锁；M16160 的额外 DRAM 读无法从总计数拆给 B、A、scale 或临时张量。两次同政策重复支持差异存在，不能证明 raster 或缓存驱逐造成差异。
- `cache-control=all` 尝试退出 11，不是第四个成功样本；失败仍保留。尚无 raster/swizzle 单变量、同 tile BF16/融合 epilogue 消融、cluster multicast 可行性或新模型 A/B/A；报告把这些列为后续门槛。
- 服务结论来自 11 个保存的时点和末尾两次功能测试。虽原 embedding 身份、启动时间、重启计数未变，仍没有连续请求级监控或延迟测量；未借此证明零影响。
- 仓库实际只增加调查脚本、派生模型和文档；没有改生产数据流、模型精度、BF16 `lm_head`、checkpoint 或原三服务的停用政策。

## 差距

无。上列实验缺项是经过明确标注的下一轮验证，不是本轮“硬件先行调查与决策”被替代验证伪装通过。
