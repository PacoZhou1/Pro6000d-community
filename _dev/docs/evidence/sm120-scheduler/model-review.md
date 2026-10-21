# SM120 官方调度模型实验：中期独立核验

状态：只读、离线中期审查，**不是最终 mission 评审**；未连接服务器或运行 GPU。证据为 `model-witness-v1/witness/` 的 dispatch JSON/SQLite、`aba-official-v1/` 三臂原始 JSONL、运行清单、服务快照和 embedding 监测。已从原始记录独立重算下述数量、配对指标和 SHA；不以 `analysis.json` 的结论代替原始记录。

## 模型分支确实命中，但执行方式须说准

- `model-witness-v1/witness/startup-pack-617-*-summary.json` 记录 64 层主模型加 1 层 MTP，重复 pack 共 65 份、11,589,593,600 B（约 10.79 GiB）；`integration-summary.json` 的 21 个真实 FX 图、20 个改写以及注册身份核验相符。运行时日志中的 M16160 有 65 条 `official_along_m_s1_m16160`，M32768 有 65 条 `canonical_v2_fallback`。这里的 Python dispatch 记录是首次执行/捕获的日志，不计请求或 Graph 重放次数。
- 独立查询 `model-witness-v1/witness/dispatch.sqlite`：有 **65 个**含私有 `pro6000d_scheduler_v1` 的官方 BF16 GEMM kernel，均为 grid `(127,272,1)`，均可按 correlationId 匹配成功的直接 `cudaLaunchKernel`；**0 个**带 Graph node ID。65 个 kernel 的设备执行时长相加为 637.435625 ms，这不是完整请求耗时。可认定真实模型正常 prefill 路径执行了候选，但不能称其为服务 CUDA Graph replay。M16160 的形状来自同次运行的注册/dispatch 记录，SQLite 的 kernel 名与网格本身不能解码 GEMM 参数。

## 无 profiler A1/B/A2：核级小收益未转化为可信服务收益

三臂 `a1`/`b`/`a2` 使用同一镜像 SHA、模型挂载、V2 量化库 SHA、命令参数和显式 12 GiB KV 池；候选臂另挂相同源码树及 pinned 官方调度库，并启用 `PRO6000D_OFFICIAL_MODE=candidate`。每臂 prefill 为 1 次 warmup + **12 次正式**约 19K 输入、1 token 输出，decode 为 1 次 warmup + **6 次正式**约 4.8K 输入、1024 token 输出；四种 probe 的进程 exit code 都是 0。原始 12 个 JSONL 文件 SHA 与 `analysis.json` 的 source SHA 全部相同；同 index 的 prefill/decode payload SHA 三臂全同。暖身请求排除在正式统计外，目标服务 metrics 每窗口只有 1 个成功请求，前缀缓存命中增量为零。

| 指标，正式请求中位数 | A1 原路径 | B 候选 | A2 原路径 |
| --- | ---: | ---: | ---: |
| prefill 首文本时间 | 2.246333 s | 2.250655 s | 2.262705 s |
| decode 估计吞吐 | 106.728 tok/s | 106.553 tok/s | 106.725 tok/s |

同 index 以 `(A1+A2)/2` 为基线的 **12 对** prefill 首文本时间降低中位数是 **0.144349%**；B 的第 12 窗有 embedding 请求计数 +9、输入 token +54，剔除此配对后安静窗口 **n=11** 的中位数是 **0.152455%**。但是 A1→A2 的 prefill 中位数本身增加 **16.372 ms / 0.729%**，远大于配对点估计，且三臂只是同一个连续时间序列，没有独立重复的时间窗口；该 0.14–0.15% 不能声称整模或线上可部署提速。原 `analysis.json` 的 prompt-index bootstrap 也只描述这组提示差异，不是跨时间因果置信区间。

B 的 decode 正式请求第 1–3 窗有 embedding 完成请求计数分别 +19/+17/+33、输入 token +113/+113/+197；六个正式请求的吞吐中位数比 A1/A2 略低，不能称 decode 收益。以三臂首文本时间作配对，B 的六对中位数反而慢 **0.532806%**；剔除三窗后 n=3 仍慢 **0.372496%**。安静窗口的判断只基于请求前后计数差，不能证明窗口中没有短暂并发或 embedding 完全空闲。

## 输出、质量与服务边界

- 原始正式性能请求中，三臂 **12/12 prefill 单 token 正文**、**6/6 decode 正文**逐 index 哈希相同；decode 的 MTP 接受计数逐 index 三臂相同。性能请求的 reasoning 哈希均为*空字符串*，所以不能当作非空思考内容一致的证据。
- 五个同名合成质量用例及四个开放输出用例每臂均通过；开放输出的四个 payload SHA 完整且三臂一致，内容与 reasoning 也一致，思考算术用例有非空 reasoning。历史质量 JSONL 没有 payload SHA，只能证实同名脚本结果和保存的输出一致，不能断言三臂质量请求字节完全相同。以上不覆盖真实 MoBase 端到端业务或长期质量。
- `aba-official-v1/embedding-monitor.jsonl` 的 **53 个采样**均显示相同 embedding 容器 ID、2026-10-06 启动时间、restart_count=1、running 和 health 200；起终服务快照的功能 embedding 请求为 200、1024 维有限向量。原 27B/9B/ASR 三个服务在起终快照保持停止，符合用户要求。监测只覆盖采样时点，不证明零延迟影响或全天可用。三臂临时容器在 `sequence.json` 中均记录测量后移除。

模型集成、数值及隔离门槛已有本轮证据支持；**M16160 同库合成链约 2% 的优势没有在这次整模 API 对比中形成可区分于漂移的收益**。后续若继续探索，应保留 `M==16160` 精确 guard，先解释 M32768 调度退化与 NCU 资源/时延，再做独立时间窗口重复；不应将该实验候选推广到生产。
