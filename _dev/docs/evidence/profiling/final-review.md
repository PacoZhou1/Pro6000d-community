# 2026-10-14 Qwen GPU profiling：独立愿景审查

- Review agent: direct Sol subagent；独立性：strong。本轮只读审查仓库、原始 JSONL、六份本地 Nsight SQLite 和恢复证据；没有重连 GPU 主机或重新执行推理。
- Source: `_dev/docs/.mission/20261014-qwen-profiling/request.md`；唯一任务状态源为同目录 `20261014-qwen-profiling.csv`。审查时六项普通工作已由 `db481ba`、`07239c1`、`e16338c`、`c6cc715`、`d50178c`、`433aeb2` 提交。
- Result: **vision_met**。`gaps_found: none`；`limited_review: false`。没有可证伪的未兑现原始目标。此判断限于本轮请求：真实热点定位、一项现成算子后端实验、负结果如实保留、embedding 不停和原服务恢复；不把自研 CUDA、MoBase 端到端质量或生产提速纳入已完成声明。
- Claim coverage: **6/6**。下表逐项核对，并不直接采信 ledger 的 `validated` 字段。

| 项 | 原始目标和交付声明 | 独立核验及证据 | 判断 |
|---|---|---|---|
| 01 | 固定 0.31 profiler/tooling，并记录原容器和 embedding 起点 | `initial-state.json` 记录四原容器和 embedding 启动时间；`nsight-install.json` 记录 2026.5.1 安装、SHA256 与真实单 kernel smoke。该 smoke 仅证实工具可采，不被报告冒充模型 profile。 | covered |
| 02 | 真实约 19K 冷 prefill trace、热点耗时份额和无 profiler 对照 | `mtp4-prefill-r1/r2-analysis.json`、本地 `raw/mtp4-capture.1/2.sqlite`、`raw-manifest.json`：两次各 3,673 kernels、2,266.68/2,272.79 ms kernel 总耗时，输入 18,901/18,946、输出 1；原始 SQLite 的字节数及 SHA256 与 manifest 一致。`cutlass-control-performance.jsonl` 的三次正式无 profiler prefill 中位 TTFT 2.224842 s，不能混作 trace latency。 | covered |
| 03 | 普通及思考 decode 的真实 trace，与 prefill 分开且披露归因不确定性 | `mtp4-nsys-summary.json`、`dflash15-nsys-summary.json` 及四份 decode SQLite：每类两次，约 4.8K 输入/256 请求输出；NVTX 只有 generation 范围、无 prefill context。六份 kernel share 分母均为**各捕获窗口全部选中 GPU kernel duration 总和**，不是 HTTP latency。API correlation 时长覆盖 prefill 约 96.8%、普通 decode 约 66.6%、思考 decode 约 78.3%；未匹配部分仍进入总排名，但未强配 target/draft。GPU interval union/观测事件跨度未被报告写成设备利用率。 | covered |
| 04 | 从 trace 选热点，做真实后端/算子替换及数值、API A/B；保留失败 | 六份 trace 的 FP8 命名 GEMM 约占 prefill 68.49%、普通 decode 62.03%、思考 decode 64.10%。`backend-selection.json` 和两份 runtime 显示相同派生镜像、除 `linear_backend_per_quant.fp8_block_w8a8` 外启动参数一致，分别真正派发 CUTLASS/B12X。`gemm-probe.jsonl` 12/12 合成算子数值检查通过，FP32 参考仅前 512 输出列，全输出检查 finite；首次初始化失败保存在 `gemm-probe-attempt1.log`。原始 A/B 各 3 warmup+9 formal，9/9 同 payload、正文、reasoning 哈希，18 项合成 API 检查通过。中位 prefill 2.224842→2.662015 s（+19.65%），普通 decode 109.082→109.413 tok/s（+0.30%），思考 decode 130.032→130.953（+0.71%）；无可部署的整模净加速。A/B/A thinking #1 有一次 18-token embedding 请求，原记录保留；`ab-summary.json` 将其从 clean 复测剔除，并用相同 #2/#3 初始输入比较 131.735468→131.743659 tok/s（+0.0062%）。 | covered |
| 05 | 清理临时容器，按序恢复 27B/9B/ASR，embedding 不停，四 API 验收 | `temporary-cleanup.json` 列出五个本轮容器均删除、无 owned 容器残留；`restoration-functional.json` 四模型 health/models 为 200，27B/9B 算术、ASR 5 秒样本转写、embedding 1,024 维向量均通过。`restoration-stability.json` 约 142.06 秒后四项再次 200，StartedAt/RestartCount 自功能检查以来未变。`embedding-monitor.jsonl` 原始 58 行均 running/health200，同 ID、StartedAt、RestartCount；`restoration-summary.json` 与这些值一致。四原容器 ID/镜像/命令一致；27B RestartCount 从初始 1 变为本次启动后 0，此差异已在报告公开，不误称计数完全不变。 | covered |
| 06 | 交付原始 trace 来源/校验、份额、实验结果和限制 | `raw-manifest.json` 的六份 SQLite SHA256 与本地忽略二进制逐一相符；远端 `.nsys-rep` 的路径/大小/SHA 有清单，本轮未重连服务器复查。`08-gpu-profiling.md` 的 FP8/BF16/attention/GDN 表值可从分析 JSON 复算，明确 name-based 类别、sampled kernel duration 分母、256-token 捕获与 1,024-token 无 profiler API 性能实验的区别；明确 B12X 1.5.0 超出官方 pin、实验未写自研 CUDA、未部署到生产。 | covered |

审查中的独立复算：六份本地 SQLite 的 kernel 数、duration sum、union、observed span 与分析 JSON/汇总一致；报告表格的命名 FP8/BF16 GEMM 份额按每组两次捕获算术平均正确。A/B 三臂各 12 请求的 usage 与服务端 token/success 增量一致；`decode_estimate_tps = (completion_tokens - 1) / (last_text_s - first_text_s)` 可复算。两臂所有窗口 embedding counters 未变；A/B/A 仅 thinking #1 正式窗口有上述增量。质量与开放回答的**实际保存 message** 而非空 hash 逐项相同。B12X 微基准是重复相同合成权重的 CUDA graph GEMM 计时，不可推导整体模型 tok/s。

非阻断的复现细节：现版 `analyze_nsys.py` 重算 MTP4 的四份 JSON 时，`raw_kernel_ranking[*].category_rule` 有旧版 `gdn/gated-delta/mamba/linear-attn` 与新版加上 `delta-rule` 的描述文字差异；剔除此说明字段后四份 JSON 完全相同，所有耗时/类别/占比不变。DFlash15 两份以 `--top 0` 重算与存档完全相同。因而不要把 MTP4 的整文件 `cmp` 不相等误报为测量差异。

验证边界：没有重新下载模型、重启服务、跑 GPU 或访问 MoBase；生产恢复据原始快照、功能请求和短期稳定性记录判断，不能延伸为长期压测或真实业务质量保证。简单凭据模式扫描未在版本化 profiling 文本中发现明文 Bearer、`sk-`、`hf_` 或非脱敏 `--api-key` 值；本地 SQLite 为 Git 忽略，前期对其 metadata/StringIds 已做同类扫描。本轮无 `source_ref/evidence_ref/why_it_matters/suggested_followup_issue` 四元反证 gap，因此没有要求新增 follow-up issue。
