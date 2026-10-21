# SM120 调度与 epilogue 实验：独立愿景评审

- review_agent_mode: direct-spawn-agent
- review_independence: strong
- review_actual_model: gpt-6.1-sol
- user_model_override: 用户明确指定子 agent 使用 Sol；实际使用 Sol，覆盖 skill 默认的同主模型偏好。
- claim_ledger: claims.json
- claim_coverage: 6/6
- claim_coverage_status: complete
- review_result: vision_met
- handoff_humanized: false（humanizer-zh 不可用，交工单保留独立评审者草稿）
- source_ref: `request.md`、同前缀 CSV 全部行、`claims.json`、前轮 `13-sm120-hardware-roadmap.md`。
- reviewed_commits: `4a6e0b1`, `22f7b06`, `7509c7b`, `af986b2`, `555fc91`, `f8ab2fe`, `81fe894`。
- review_scope: 本地读取源码、固定构建、原始 JSONL/CSV/NCU 文本与保存的 SQLite、git diff、服务快照；离线重算；无 SSH、GPU 或服务操作。

## 结论

**vision_met，限于本轮获准的“受控调度与 epilogue 实现、数值/Graph 门槛、硬件诊断及条件整模试验”。** 结果是负采用决定，不是生产加速：融合链在 M16160 调度后比自身旧版快约 9.2%，仍比官方慢约 8.8%；官方同库 AlongM/s1 在该合成完整链约快 2%，但整模 A/B/A 的 prefill 配对差异仅 0.1443%，低于 A1→A2 基线漂移，且 M32768 同设置约慢 50%。报告据此不采用候选，也没有更改生产 FP8、BF16 `lm_head` 或原服务配置。

`gaps: []`。下列限制属于已披露的实验边界，不构成本轮原请求未兑现：没有跨独立时间窗的服务收益证据、没有真实 MoBase 端到端质量试验、NCU 总流量不能按 A/B buffer 归因、embedding 采样不能证明零时延影响。

## 六项声明逐项核验

| 项 | source_ref | evidence_ref 与独立核验 | 判断 |
| --- | --- | --- | --- |
| 01 固定源码构建同 tile 调度与普通 BF16 epilogue | `request.md`; CSV 01; 13 号路线第 5 节 | `build-v3/build-manifest.json`、`build-v3/compile.log`、`scheduler_driver.cu`、`scheduler_abi.h`、`symbol-isolation-v3.json`。编译 exit 0，库 SHA `af1d8149…85d6`；M64 paired BF16 与 fused 的 `CollectiveMainloop` 静态同类型、两级流水，官方 M128N128 控制独立。v1 同名 TLS 导致 139，v3 私有 namespace/internal TLS 与只导出 C ABI，未把失败记为数值通过。数学、pack 主体的固定来源与生成文件 SHA 有清单。 | 覆盖；三种 epilogue 资源不同，不能把所有差异归于输出算术。 |
| 02 逐位数值与改输入 Graph 门槛 | `request.md`; CSV 02 | `scheduler-matrix-v4/stdout.log`、`analysis.json`、`scheduler_probe.py`、`numerical-verification.md`。13 个用例通过：M1/5/16/256 是官方/回退，支持域 M257 起；六配置的 BF16、FP8 字节、FP32 scale 全量逐位核对，M257 的 zero/tiny/large/exceptional 分布通过。M257/2741/2786/16160 的八臂改输入 Graph 共 32 见证通过，timing graph 输出再次核对。边界分布的下游 BF16未单独运行，但 FP8/scale 输入逐位相同；正常分布有显式下游 BF16比较，报告如实限定。首次 v3 漏尾块 Graph 被分析器拒绝，v4 才纳入。 | 覆盖；没有把小 M 回退或历史失败冒称候选成功。 |
| 03 同输入随机配对的完整流程 | `request.md`; CSV 03 | `scheduler-matrix-v4/stdout.log` 和四个 `epilogue-*/stdout.log`、`official-tuning-v2/stdout.log`；此前中期 review 已对原 JSONL 重算，与各 `analysis.json` 一致。主矩阵 M2741/2786/16160 八臂，各 warm/扰缓存 31 对；同库官方三臂补测 M2741/2786/16160/32768，各 31 对且全量数值/Graph 通过。M16160 H8/AlongM-s1 配对比 1.02025/1.01928；M32768 比约 0.664，明显反向。缓存扰动是在计时外写自有缓冲，不是固定冷缓存。 | 覆盖；合成 Graph 时间与服务 TTFT 分开。 |
| 04 NCU、CLC 映射与 M32768 反证 | `request.md`; CSV 04、04B | `ncu-summary.json` 6/6 严格可用，`ncu-large-m-summary.json` 2/2 可用；每个成功窗口保存原 CSV、单位、source/库 SHA、唯一 NVTX/profiled_once、数值/Graph。M32768 同输入 H8 与 AlongM/s1 GEMM grid 均 `(256,272,1)`，主核 18.264→31.113 ms、DRAM read 6.064→46.498 GB、Tensor active 83.60→36.09%，L2 request 约 95.04 GB。`scheduler-source/verify_mapping.py` 离线重跑 6/6，与固定 header SHA、实测 physical grid 相符，确认 SM120 默认 `PersistentTileSchedulerSm100` CLC；旧 `resource_model.py` 的 SM90 rounded-grid 推断已标历史字段、13/14 主文已纠正。旧 NCU exit 11 且无计数，单独留在 inventory，不进入成功比值。 | 覆盖；每臂单次 profiler capture，不能将总 DRAM 指定给 A，或把 stall/Tensor active 当作 API 时间/峰值 FLOP 利用率。 |
| 05 条件整模准入与拒绝上线 | `request.md`; CSV 05 | `runtime-smoke-v1/runtime-smoke.json` 的目标/回退逐位与改输入 Graph 通过；`model-witness-v1/witness/` 保存 65 个注册模块 M16160 候选、65 个 M32768 回退、21 图/20 改写。独立查询保存的 `dispatch.sqlite` 得 65 个私有官方 BF16 kernel、65 个成功 direct `cudaLaunchKernel` 关联、0 Graph node；同次 dispatch 记录给出 M，SQLite 自身不解码 M。`aba-official-v1` 每臂 1 warmup + 12 正式 prefill、1+6 decode，同 index payload SHA 全同；原 JSONL 重算 A1/B/A2 TTFT 2.246333/2.250655/2.262705 s、配对改善 0.144349%，B 第 12 prefill 有 embedding 活动，剔后 n11 为 0.152455%，A1→A2 漂移约 0.756%（逐对）。decode TPS 106.728/106.553/106.725，无收益。`model-contract.json` 为 `not_adopted`，与报告一致。 | 覆盖；真实性与负采用决定相称，不称服务提速。 |
| 06 文档、复现与受保护服务 | `request.md`; CSV 06 | `14-sm120-scheduler-epilogue.md`、`README.md`、`reproduction.md` 给固定镜像/库/SHA、CPU 重算、GPU 隔离步骤及失败边界。三臂正式性能正文 12/12 单 token 和 6/6 decode 同 hash，MTP 接受计数逐 index 一致；性能 reasoning 为空，开放思考有非空 reasoning 且四个开放用例三臂同 payload/hash，五个合成质量用例通过但旧质量请求无 payload SHA。`final-services.json` 和 `final-services-later.json` 两次 embedding 功能请求 200、1024 维有限值，ID/2026-10-06 启动时间/restart_count=1 不变，原 27B/9B/ASR 仍停；`final-running-inventory.json` 不见自有实验容器，ABA 的 53 个 embedding 采样均健康同身份。 | 覆盖；服务结论是采样点与功能检查，不是连续 SLA。 |

## 科学口径与剩余限制

- 官方 M16160 同库约 2% 是合成算子完整 GEMM+V2 链的随机配对结果；不能与整模 TTFT 直接相加或宣称生产提速。M32768 两种 H8/AlongM-s1 实际 raster 同为 AlongM，仅 swizzle 8→1；本轮修正了过去以 SM90 rounded grid/fast axis 解释 SM120 的错误。名义坐标置换不等于 CLC 实际领取顺序；A footprint 跨 L2 是合理假设，非按 buffer 的流量证明。
- 六组 M16160 与两组 M32768 NCU 均为选定合成算子、单次 kernel-replay、cache/clock none。核级 profiler 时长与原 Graph event、无 profiler API 时长是不同口径；失败尝试没有混入有效计数。
- A1/B/A2 一段序列有基线漂移和 B 的 embedding 计数变化。所有 12 对已保留并另列安静窗口敏感性，prompt-index bootstrap 不能排除时间窗或重启变异。性能固定 token 的 reasoning 为空；同名合成质量脚本没有捕获 payload 字节；未验证真实业务或长期服务质量。
- Witness 原始 `.nsys-rep`/`.sqlite` 本机 SHA 与 export status 相符，但被 Git ignore；文本派生结果可审，新的 checkout 要从记录的远端路径保留原始二进制才能再次从 SQLite 重算 launch 关联。本审查没有新增 GPU/服务实验。

## 差距

无可证伪的本轮未闭环差距。继续修改 tile/warp、解释 A/B 逐 buffer 流量或争取跨时间稳定服务收益，是后续研究，不是本轮批准任务的必需正结果。
