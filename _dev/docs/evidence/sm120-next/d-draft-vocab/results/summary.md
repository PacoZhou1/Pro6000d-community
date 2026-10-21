# Task D 结果：草稿头词表裁剪（FR-Spec 式），Qwen3.6-27B-FP8 MTP4

日期 2026-10-14（主机 UTC）。固定镜像 sha256:a4a4c043…c3a6b，V2 runner，fp8 KV，FLASHINFER，MTP4，kv-cache 12 GiB，其他与历史口径一致。补丁未改动，未 commit。原始数据：`host/`（logs、各 arm 的 runs/*，含 gpu-clock.csv、embedding-monitor、server.log、d3.json、perf.jsonl）。

## 1. Smoke / 诊断捕获
- 日志：`draft-vocab enabled: 32768 rows (320.0 MiB)`，无 `disabled`；PIECEWISE+FULL 图与 speculator 的 prefill/decode 图均已捕获（draft head 在捕获前构建）；STRICT=1 下无回退。设备端计数器未接线（设计中已说明），故“零回退”依据是静态 guard 通过 + STRICT 不抛错。
- Nsight（32K 候选，256 token decode，92 个 decode 步，两次 capture 结果一致）：每步 4 个 M=1 草稿头 kernel 由 1576 us（cutlass wmma，grid 8x1940）变为 **217.7 us（cuBLAS gemv，grid 8192）**，约缩小 7.2 倍（≈1.54 TB/s，带宽受限比例保持）；验证头 M=5 仍是 1 个/步、1575.4 us、grid 8x970，**未变**。每步头部合计 7.87 ms → 2.45 ms。见 `host/runs/smoke-nsys-32k/d0_capture{1,2}.json`。

## 2. D3 正确性（贪婪，6 normal + 6 thinking 4.8K/1024 + 4 开放输出，每臂 2 pass）
基线自一致：pass1 vs pass2 16/16 token 完全一致。

| 子集 | 与基线逐 token 一致 | 接受率 accepted/draft（全部） | 平均接受长度 | normal / thinking / open 接受长度 |
|---|---|---|---|---|
| 基线 | — | 0.5945 | 2.378 | 2.116 / 2.605 / 3.040 |
| 16K | **16/16** | 0.4417 | 1.767 | 1.558 / 1.933 / 2.419 |
| 32K | **4/16**（仅 4 条短开放输出一致；12 条 1024-token 请求均分叉） | 0.4720 | 1.888 | 1.658 / 2.075 / 2.604 |
| 64K | **16/16** | 0.5573 | 2.229 | 1.984 / 2.449 / 2.764 |

（pass1 与 pass2 的接受率完全相同；接受长度 = accepted/drafts，不含 bonus。）开放输出均正常 `stop`，thinking 在开放题中正常结束；其余为 ignore_eos 的 length 终止，与基线一致。

### 32K 分叉诊断（未改 gate）
- 子集文件检查：32768 个 id，无重复、升序、范围 0..248076 < 248320、EOS 248044/248046 均在内，SHA 与 coverage.json 一致，启动日志 SHA 匹配；形状与 16K/64K 路径相同。未发现映射/排序/越界/dtype 问题，故没有补丁修复，未重跑 32K D3。
- 首个分叉位置（候选 vs 基线，token 序号）：normal 59/12/15/3/62/203，thinking 127/193/158/103/19/118，均为长输出的早期。
- 在基线服务上做 teacher forcing（prompt + 基线前缀，取 top-10 logprob，`runs/tf-base/tf.jsonl`）：12 处分叉点基线 token 与候选 token 的 logprob 差仅 0.00–0.38 nat，且为 0.125 的整数倍（bf16 logit 量化步长）。基线 token 在 TF 下仍为 top-1 的只有 3/12，候选 token 为 top-1 的 8/12，1/12 两者皆非，1/12 完全持平。即这些分叉点都是 1–3 个 bf16 ulp 的近平局，基线自己的 prefill 与逐步 decode 数值也不一致。这与 07 号文档记录的“形状/接受模式相关的数值差”一致，但**不能证明**这就是 32K 分叉的原因。
- 未解之处：为何只有 32K 分叉而 16K、64K 全同。同一批 6+6 请求在 D4 路径（流式，profile_client）里，32K 的正文 SHA 与 A1/A2 12/12 完全一致，D3 路径（非流式 + return_token_ids）则分叉；两条路径请求体除 return_token_ids/stream 外相同。因此分叉依赖请求路径/调度时序，机制未定位。结论：**按 D3 token 级 gate，32K 不通过**；16K、64K 通过。

## 3. D4 A1/B/A2（无 profiler，同 payload，每臂 6 normal + 6 thinking，warmup 剔除，基线 = (A1+A2)/2）
| B 臂子集 | normal tok/s A1 / B / A2 | thinking tok/s A1 / B / A2 | 配对变化 normal / thinking / 合并(中位) | 接受长度 normal A→B；thinking A→B | A1 vs A2 漂移 | 输出等价 | 过 gate? |
|---|---|---|---|---|---|---|---|
| 32K | 106.75 / 117.19 / 106.77 | 130.63 / 147.09 / 131.12 | +9.77% / +13.93% / **+12.32%**（均值 +12.76%） | 2.12→1.83；2.59→2.30 | 0.02% / 0.37% | D4 路径正文 12/12 同基线；D3 token 级 4/16 | **否**（D3 失败） |
| 64K | 106.78 / 119.18 / 106.76 | 130.62 / 147.96 / 130.64 | +12.30% / +14.01% / **+13.55%**（均值 +13.81%，逐条范围 normal 11.2–17.8%，thinking 12.5–16.0%） | 2.12→1.98；2.59→2.45 | 0.01% / 0.01% | D3 16/16；D4 12/12 同 | **是** |
| 16K | 106.76 / 108.75 / 106.78 | 130.64 / 132.91 / 131.13 | +3.26% / +2.53% / **+2.98%** | 2.12→1.56；2.59→1.94 | 0.01% / 0.37% | D3 16/16；D4 12/12 同 | 否（<5%） |

tok/s 为各臂逐条 decode_estimate_tps 的中位数；配对变化为逐 index B/((A1+A2)/2)-1 的中位数。估算每步时间（ms）：基线 29.2 / 27.5（normal/thinking）；32K 24.1 / 22.4；64K 25.0 / 23.3；16K 约 23.5 / 22.1，但 16K 的接受长度降幅吃掉了几乎全部节省（跌破 README 的保本线）。16K 臂对 normal 的逐条范围 −0.95% 到 +12.99%，噪声较大。
SM 时钟（nvidia-smi 500 ms，利用率>30% 样本）：所有 D3/D4 臂中位 2842 MHz，范围约 2002–2887 MHz，功耗中位 466–487 W，温度最高 82 C；各臂之间无系统性差异（B 臂功耗略低 ≈8 W）。未尝试 `-lgc` 锁频（无 root）。
Embedding：每个 arm 前后检查（ID/StartedAt/RestartCount 不变、health 200、1024 维有限向量）全部通过，监控线程每 10 s 记录；详见各 runs/*/services-*.json、embedding-monitor.jsonl。部分臂的 perf.jsonl 中 embedding 侧 vllm 计数在测量窗口内有变化（d4-a1、d4-a2、d4-b 的若干条，d4-16-a2），为其他客户端（如 one-api）的正常流量，非本实验触发；各臂中位数不受影响，但属于未隔离的环境噪声。

## 4. 决策（按计划规则）
- 32K：性能 +12.3%，但 D3 token 级等价不通过（4/16）→ 不能进入下一步，原因待查（见 2）。
- **64K：+13.55%（≥10%），D3 16/16 逐 token 一致，D4 正文一致，A1/A2 漂移 ≤0.37% → 满足“≥10% 且输出一致”，建议进入跨时段复测**（独立第二时段，并补 thinking 模式与更长/更多样的输出，以及上面 32K 路径相关分叉的原因，因为 64K 的 16/16 也可能只是该分叉模式没触发）。
- 16K：+3.0% <5%，接受率降幅过大，结束，保留负结果。
- 对照 README 估算：零损失上限 +19%（64K）实测 +13.6%，接受长度 2.12→1.98 的损失与 coverage-product 模型预测的方向一致。

## 5. 失败项与限制
- 一个 32K 的 D3 失败（上述），机制未定位；teacher forcing 仅 12 点，且 TF（prefill 路径）与 decode 数值不同，仅作旁证。
- 只有同一天的单时段，无跨时段重复；语料是通用语料，zh 覆盖率 32K 为 88.6%，64K 为 95.6%。
- 子集为单一通用子集；reasoning 专用子集未测。5 项合成质量项未单独跑（沿用 D3 的开放输出与 4.8K 请求），文本质量只做了与基线逐 token 比较。
- 草稿头设备计数器未接线，“无回退”依据 STRICT 模式与日志。
- d0 分析中 M=1 草稿 kernel 的“形状”依旧是由 kernel 名/grid 推断。
- 每次 GPU 实验使用 ssh 单连接；一次 SSH 通道中断后按要求等待 10 分钟后重连，主机上的批处理脚本不受影响。

## 6. 主机残留
`/tmp/qwen-sm120-next-d/`（src、inputs、runs、logs、results*.tgz、smoke-nsys 的 .nsys-rep/.sqlite 约 数十 MB）保留；无 codex-next-d-* 容器；`/home/llm/models` 未改动；未触碰非实验容器。
