# FP8 GEMM epilogue：独立愿景评审

- review_agent_mode: direct-spawn-agent
- review_independence: strong
- review_actual_model: gpt-6.1-sol
- user_model_override: 用户明确指定子 agent 使用 Sol，覆盖 skill 的默认同模型偏好；实际使用 Sol。
- claim_ledger: claims.json
- claim_coverage: 7/7
- claim_coverage_status: complete
- review_result: vision_met
- handoff_humanized: false（本机未安装 humanizer-zh，按 skill 保留审查者生成的草稿）
- source_ref: `request.md`、同名前缀 CSV 的 01–07 及 REVIEW-01、`claims.json`、`log.md`。
- reviewed_commits: `00c812e`, `e79356c`, `a0aac87`, `3f9cce2`, `d632e78`, `a9a7168`, `80799c5`。

## 结论

原请求是查清 FP8 gate/up GEMM 至量化之间的中间读写，开发可行的针对性候选并拿真实算子与整模对照；它没有要求候选必须加速。已实现隔离候选、通过限定输入的逐位数值门槛，证明正常模式真实模型派发，并得到两轮无 profiler A1/B/A2 的**负性能结果**。v6 的 12 个同索引约 19K 输入请求，B 相对 A1/A2 均值的首文本时间配对中位数慢 7.0872%；v4 慢 22.7454%。候选未进入生产。最新用户指令要求原 27B、9B、ASR 保持停用、embedding 继续运行；结束和延后快照与此一致。未发现需要后续 issue 的可证伪差距；`gaps: []`。

`vision_met` 仅表示本轮**实验与实测目标**完成，不表示可部署加速、全域逐位等价、MoBase 业务质量或长期服务 SLA 已证明。

## 七项承诺与独立核验

| 项 | source_ref | evidence_ref 与核验 | 判断 |
| --- | --- | --- | --- |
| 01 真实数据流和尺寸 | `request.md`; CSV 01 | `audit-results.md`, `dataflow.md`, `source-manifest.json`, `fx-proof-v2/summary.json`：真实链为 block FP8 GEMM → BF16 `[M,34816]` → V2 SiLU/量化；21 张实际 FX 图中有 20 个匹配片段、65 个 gate/up 权重组。M2741 的 BF16 中间张量逻辑写加读为 381,722,624 B，报告明确未当作 DRAM 实测。 | 覆盖 |
| 02 有边界的候选实现 | `request.md`; CSV 02 | `runtime/packing.py`, `runtime/fx_matcher.py`, `runtime/fx_rewrite.py`, `runtime/candidate_op.py`, `candidate-design.md`, `candidate-build-coop-v6.log`：gate/up 每 128 列成对预排，BF16 舍入后才激活、量化；两核写 FP8 临时与最终输出，未物化 BF16 D。仅 opt-in、精确权重/scale 存储身份和结构匹配、SM120 与 `256<M≤32768` 走候选，其余回退官方 GEMM+V2。原 BF16 `lm_head` 路径未改。v6 本地 `.so` SHA256 `e4039ddc95fa3b56f02743357b7a7193156dc500c5d8804bb77ebd3af85450d6` 与构建、probe、模型 manifest 相符。 | 覆盖；只适用于实验固定域 |
| 03 数值、Graph、完整算子和硬件计数 | `request.md`; CSV 03 | `candidate-coop-v6-full.jsonl` 有 12 个数值 case；官方 BF16、FP8 字节、scale 位和下游 BF16 GEMM 对照均零差，普通候选形状换输入后合成 CUDA Graph replay 通过。v2/v5 失败原始记录保留。v6 M2741 完整 GEMM+pack warm 中位 1.950ms，对同窗口官方 GEMM+V2 的 1.777ms；`ncu-coop-v6-m2741/summary.json` 两核求和 DRAM 559,511,552→314,947,328 B，但 kernel 时长和 1.634→1.806ms。NCU 是合成 replay 的 8-pass 窗口，绝非 API 延迟。预排单层成本单列、整模预排墙钟未单独测量。 | 覆盖；负结果与测量边界正确披露 |
| 04 真实整模命中 | `request.md`; CSV 04 | `model-witness-v6/integration-summary.json`：65 模块精确身份，21 图/20 rewrite；`dispatch-summary.json` 与本地 `capture.sqlite` SHA 相符，130 个 Cooperative fused GEMM +130 个 pack，260 个 correlation 各有一次成功 direct `cudaLaunchKernel`；Graph node/replay 命中 0。小 M fallback 另有记录，报告没有把合成 Graph 兼容性冒称为服务端 Graph replay。v4 冻结 runtime 与 witness manifest 另存。 | 覆盖 |
| 05 无 profiler 的真实请求对照 | `request.md`; CSV 05 | 重新从 `aba-v4/`、`aba-v6/` 三臂 JSONL 计算：每臂每版 12 个正式 prefill、6 个正式 decode，另各剔 1 个 warmup；逐索引性能 payload SHA 相同。v4 首文本时间中位数 2.250230/2.769921/2.263048s，配对慢 22.7454%；v6 2.251251/2.415334/2.260911s，配对慢 7.0872%。正式窗口目标请求恰 1 次成功、无 prefix 命中/preemption，嵌入服务成功计数在每个前后快照不变。每臂 5 项同名合成质量及 4 项开放输出正常通过；五项质量请求**无**保存 payload hash，修订后的 `analyze_api.py`/`analysis.json` 明记 `payload_hash_unavailable`/null；四项开放请求 hash 一致。v6 decode A1/B 和 A1/A2 的正文 0/6 相同，B/A2 为 6/6，相应 MTP 计数也变；报告未把 decode 波动归为候选收益。 | 覆盖；请求和输出证据等级准确 |
| 06 服务生命周期 | 最新 `request.md`; CSV 06 | `initial-state.json`, `owned-cleanup.json`, `final-service-check.json`, `final-service-check-later.json`, `embedding-monitor.jsonl`：11 个已停用且本轮自有的实验容器被清理，原容器不在清理集合。结束时 27B/9B/ASR 为 stopped，身份、镜像、命令和重启计数匹配；embedding 原 ID、2026-10-06 启动时间和原重启计数 1 保持，health200，实际向量 API 200、1024 维且有限。96.61 秒后原服务状态与 embedding health 仍符合政策。原监控 JSONL SHA 与汇总相符，274 次约 136.57 分钟采样全部运行/200、最大间隔 30.892 秒。 | 覆盖；周期采样非连续可用性保证 |
| 07 报告与取舍 | `request.md`; CSV 07 | `12-fp8-gemm-epilogue.md`, `README.md` 准确区分逻辑 BF16 流量、合成完整算子、NCU、Nsight 真实 direct launch 与无 profiler API；公开 v4/v6 负收益、v6 decode 基线漂移、失败版本、固定栈/单并发/合成请求限制及不采用决定。报告和 README 均按用户最新要求写原三服务保持停用、embedding 运行。 | 覆盖 |

## 证据等级、局限与保密核对

- 合成逐位门槛只覆盖所测 12 case 和固定 BF16/FP8 布局；v6 对固定 CUTLASS Cooperative M64 支持断言做实验性 shadow 放宽。不能推为任意输入、架构或版本的通用正确性。
- NCU 的 DRAM 下降与 L2/时长并存；固定窗口无 profiler 微基准和整模配对均未得到大 M 加速。额外 packed 权重为 11,589,593,600 B；A/B 均预排保证比较公平，但整模预排墙钟、并发压力、长期稳定和 MoBase 业务 E2E 尚未测。
- v6 decode 两次基线自身输出与 MTP 接受计数不一致；只能确认这轮候选在长 prefill 更慢，不能从 decode 速率推断其算子效果。质量探针只验证固定有限案例，未证明业务质量无损。
- 结束服务结论限最终功能快照、96.61 秒后 health 快照和 274 个周期样本；不推断连续零故障或 embedding 延迟无干扰。原始状态文件的 `--api-key` 对应值为 `<REDACTED>`；评审文件不抄录凭据。
- review_process_limit: 本次仅本地只读复核源码、提交、JSONL/JSON、manifest、已有 SQLite 与二进制哈希；未重跑 GPU、NCU 或远端服务。先前 02–06 行的独立静态/数值/原始数据检查沿用，最终重点复核新报告与边界。该限制不降低已保存原始记录对本轮目标的支持等级。

## 差距

无。性能未提升是已测出的取舍，不是原用户实验目标的缺口；生产部署、真实 MoBase 全链路与长期质量保障均未被本轮声明为已完成。
