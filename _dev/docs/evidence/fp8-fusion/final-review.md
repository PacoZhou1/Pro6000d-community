# FP8 算子实验独立审查

- review_agent_mode: direct-spawn-agent
- review_independence: strong
- review_actual_model: gpt-6.1-sol（按用户指定）
- claim_coverage: 7/7
- review_result: vision_met

审查只使用本地原始记录、源码、git 版本和离线重算；未连接 GPU 服务器、重跑实验或修改实现。原始请求是完成 FP8 范围内的热点判断、靶向算子试验与实测对比，保持 embedding，最后恢复原服务；没有要求证明生产提速。七项记录均有相应证据。第一轮发现 `run_api_arm.py --witness` 的帮助误称 Graph replay 证明；`247f0bd` 已将其改为 real-kernel dispatch capture，并在 `reproduction-manifest.json` 同时保留实测版 SHA `37292f6c...`（git `04cbce5`）和当前 SHA `ca251e75...`。两版哈希均从文件独立重算匹配，改动只有帮助文案。现无剩余可证伪差距。

关键复核：

- 四份正常图捕获的热点比例按**每份全部 GPU kernel duration 求和**计算，已融合 SiLU+block-FP8 量化核在两份长 prefill 中约占 4.32%，decode 约 0.286%；这不是 HTTP 墙钟占比。原 vLLM 已有 SiLU/乘法/量化融合，本候选改 warp/CTA 线程组织。[热点源数据](profile-opportunities.json)、[源码固定记录](installed-source-manifest.json)。
- 本地所选 V2 CUDA 源 SHA `a245bb14...`、库 SHA `087b7caa...` 与构建清单、微基准和真实容器记录一致。两版各 12 组保存的 GPU 数值对照均无 FP8 字节或 scale 位错误，五种 M 的单核图改输入重放通过；V2 14 项运行时/guard smoke 通过。该测试集不等于所有输入的形式证明。[原始微基准](micro-v2.jsonl)、[运行时 smoke](runtime-smoke-v2.jsonl)、[运行时 guard](runtime/sitecustomize.py)。
- 对保留的 `witness/capture.sqlite` 运行离线分析器得到与 `witness/dispatch-proof.json` 相同结果：真实长 prefill 中候选 130 次、M1 官方回退 6 次；候选 130 次均匹配成功的直接 CUDA launch，`graphNodeId` 为 0。候选 duration 求和 66.858301 ms、全部 3673 个核求和 2283.011888 ms，仅是该诊断窗口，不是请求耗时或整模 Graph replay。[派发说明](integration-results.md)。
- 三臂原始 JSONL 各 12 个正式 prefill、6 个正式 decode，同 index payload SHA 一致。长 prefill 首文本中位 A1/B/A2 为 2.268213/2.244258/2.284222 s；B 相对同 index 两次 A 均值的改善中位 1.4144%（12 对范围 1.3697%–1.5054%）。decode 速率中位 106.7358/106.7317/106.7054 token/s，基本不变。54 个正式请求窗口的 embedding 计数快照均完整且无已观测增量；前后快照不能排除瞬时活动。正文非空 hash 配对 prefill 12/12、decode 6/6 相同；reasoning 全空，已被当前 `analyze_api.py` 排除出“有意义相等”。三臂各 5 个结构化、4 个正常结束开放题，共 27/27 固定合成质量检查通过。[API 汇总](api-comparison.json)、[配置对比](configuration-comparison.json)。
- 合成 M16160 NCU 的官方/候选 `gpu__time_duration.sum` 为 1.225792/0.858336 ms，DRAM 吞吐占持续峰值为 66.379%/93.516%；单版单核各 9 次 replay pass，时钟和缓存未控。`.ncu-repz` 与导出 CSV SHA 均与 [计数摘要](counter-summary.json) 相符。这里的 replay 是 NCU 单核采样，不能用于宣称请求级收益。
- 初始、功能恢复与 316.629131 s 后快照逐项对比，四服务容器 ID、镜像、命令和重启数均保持原状态，功能与健康检查通过。74 条 embedding 监测均 health=200、原 ID/启动时间/重启数相同；自有临时容器剩余 0。周期采样不提供毫秒级可用性保证。[恢复汇总](restoration-summary.json)、[原始记录](restoration.json)、[监测](embedding-monitor.jsonl)。

采用边界：候选只在隔离容器 0.70 显存预算的一次顺序 A1/B/A2 中显示小幅长 prefill 改善；单核合成收益更大，但不可外推。未验证跨时段、原生产 0.45 预算四服务共存、真实 MoBase 端到端或广泛长思考质量。生产保持原 vLLM 0.27.1/MTP2，候选未部署。
