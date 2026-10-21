# F1/F2 null-metadata 修复后的 fresh A/B/A

本稿于 2026-10-15 从本地原始文件独立复算；没有连接服务器、运行 GPU 或操作服务。root 执行的完整窗口为 UTC 10:42:10–10:55:12，三臂控制器与各臂四个 API 子进程均成功退出。来源：[控制器状态](ffix-v1/status.json)、[独立分析 JSON](ffix-v1/analysis-independent.json)、[CPU 分析器](analyze_ffix_aba.py)。

本次 **B 的长 prefill 配对 TTFT 中位减少 1.4896%**；decode TPS 配对差仅 **+0.0076%**，不构成有意义的 decode 增益。正式 decode 六条在三臂间正文哈希、drafts 与 accepted 计数均逐例相同。它说明修复版在这个 fresh 窗口通过所存 API 对照，**不证明历史 B 分叉的唯一原因已找到，也不证明整模逐位相同**。

## 版本与实验范围

A1/B/A2 均使用同一 `f-mdnone-fix` 只读 instrument mount；运行环境唯一差异为 `PRO6000D_F_ENABLE='' / 'norm,conv' / ''`。三臂镜像、模型 mount、启动命令、其余环境完全一致。镜像为 `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`；MTP=4、FP8 权重/FP8 KV、FLASHINFER、max batched tokens=32768、GPU memory utilization=0.7；没有显式 12GiB KV 参数，不能借用此前不同实验的 KV 预算。BF16 lm_head 沿用原模型路径，本轮没有更换 GEMM 数学。

固定 [source-manifest](ffix-v1/source-manifest.json) 记录 site `85404310…f1bbcc`、F kernel `1b51a6d9…c8dd6`、V2 entry `8cdace70…53b63`、V2 lib `087b7caa…33b3f`。新 site 相对旧 `ce01159d…b3380c` 只在 opaque custom op 的 `md is None` 分支补官方量化调用；详见 [最小补丁说明](f-mdnone-fix/README.md)。本轮没有启用 F1 VERIFY、F2 reference/clones/sync 或 diagnostic helper。

三臂实际日志均为 KV 30.75GiB、848709 tokens，Graph capture pool 先 0.84GiB、后 0.25GiB。A1/A2 使用 compile cache 目录前缀 `ab689653c5`，B 为 `4540126b00`；B splitting ops 包含新 opaque op，存在 F1/F2 fused 首次执行 marker，norm eligibility 的真实 weight dtype 为 BF16。这是接入版本和图边界确实变化的证据，不是每次融合调用的 direct/replay witness。B 有 prewarm 成功日志，但不能据此声称所有真实 dtype/shape 专门化均已提前编译。

原始请求：[A1 prefill](ffix-v1/a1/prefill.jsonl)、[B prefill](ffix-v1/b/prefill.jsonl)、[A2 prefill](ffix-v1/a2/prefill.jsonl)；[A1 decode](ffix-v1/a1/decode.jsonl)、[B decode](ffix-v1/b/decode.jsonl)、[A2 decode](ffix-v1/a2/decode.jsonl)。每臂 13/7 条分别含一条 warmup，下表只用正式 12/6 条。warmup 的 payload/content 亦三臂相同。

## 逐请求性能与输出

TTFT 定义是客户端 `first_text_s`。每个索引以 `baseline_i=(A1_i+A2_i)/2`，减幅为 `100×(1-B_i/baseline_i)`，最后取配对中位数；没有用三个中位数的比值替代。prefill 每请求只输出一 token，不是持续生成吞吐。

| 指标 | A1 median | B median | A2 median | 配对结果 |
|---|---:|---:|---:|---:|
| 长 prefill TTFT s | 2.212219954 | 2.189053771 | 2.230441273 | 减少 1.489599706% |
| decode 请求 TTFT s | 0.503787323 | 0.499012402 | 0.506550583 | 减少 1.207493360% |
| decode TPS | 106.620630125 | 106.622735072 | 106.618371086 | +0.007629111% |

prefill 配对节约中位 32.8897ms，12/12 的 B 均快于对应 A1 和 A2；减幅范围 1.4324%–1.5788%。A2/A1 的逐对 TTFT 漂移中位为 prefill **+0.822799%**、decode **+0.575060%**，应与收益一并报告。这是一次顺序夹持窗口，没有随机重启重复与置信区间，不能视 12 个索引为 12 轮独立部署。

| prefill index | A1 TTFT s | B TTFT s | A2 TTFT s | 配对减幅 |
|---:|---:|---:|---:|---:|
| 1 | 2.186495 | 2.163368 | 2.203877 | 1.4495% |
| 2 | 2.191542 | 2.168938 | 2.209842 | 1.4429% |
| 3 | 2.197637 | 2.173323 | 2.214910 | 1.4935% |
| 4 | 2.200374 | 2.176817 | 2.218919 | 1.4857% |
| 5 | 2.204519 | 2.181936 | 2.222769 | 1.4324% |
| 6 | 2.210270 | 2.186748 | 2.228606 | 1.4729% |
| 7 | 2.214170 | 2.191360 | 2.232277 | 1.4332% |
| 8 | 2.220432 | 2.195444 | 2.238300 | 1.5216% |
| 9 | 2.225858 | 2.199109 | 2.240307 | 1.5214% |
| 10 | 2.225455 | 2.200708 | 2.245213 | 1.5490% |
| 11 | 2.233928 | 2.206879 | 2.250633 | 1.5788% |
| 12 | 2.235364 | 2.210622 | 2.254443 | 1.5271% |

12 条正式 prefill 的 payload SHA、prompt tokens 与单 token 正文哈希均三臂对应相同；正文 hash 前缀均为 `e6a20027a916`。正式 prefill 没有 MTP draft/accepted。

| decode index | A1 TTFT s | B TTFT s | A2 TTFT s | 三臂同 drafts / accepted | 三臂同正文 SHA 前12位 |
|---:|---:|---:|---:|---:|---|
| 1 | 0.500465 | 0.496761 | 0.504071 | 328 / 698 | `d4af778f3c7e` |
| 2 | 0.501959 | 0.497985 | 0.505098 | 310 / 713 | `c825d81e597f` |
| 3 | 0.503061 | 0.498984 | 0.506599 | 349 / 677 | `e74f68609bb6` |
| 4 | 0.504514 | 0.499041 | 0.506503 | 289 / 734 | `d2d8da034f0e` |
| 5 | 0.505155 | 0.499841 | 0.507253 | 342 / 681 | `8d7128252de8` |
| 6 | 0.506866 | 0.501614 | 0.509525 | 351 / 675 | `b39641371b68` |

每臂正式 decode 的 drafts 总计 1969、draft tokens 7876、accepted 4178，六条 completion 均 1024 tokens；payload SHA 和 prompt tokens 逐索引相同。`per_pos accepted` 与总 accepted 相同，draft tokens=4×drafts；不是仅比较 accepted 总和而忽略逐例差异。reasoning tokens 均为零，空 reasoning hash 相同不计作质量证明。完整 SHA 及逐请求数值保存在独立分析 JSON，正文未在此展开。

TPS 按原客户端公式 `(completion_tokens-1)/(last_text_s-first_text_s)` 重算，与原字段逐条一致；它包含 MTP 调度与分批交付的客户端观察范围，不能当单 kernel 吞吐。

| decode index | A1 TPS | B TPS | A2 TPS | 配对 TPS 差 |
|---:|---:|---:|---:|---:|
| 1 | 108.833970 | 108.828709 | 108.832675 | -0.00424% |
| 2 | 115.144280 | 115.154235 | 115.145184 | +0.00825% |
| 3 | 102.306184 | 102.303550 | 102.307750 | -0.00334% |
| 4 | 123.506445 | 123.520810 | 123.517870 | +0.00700% |
| 5 | 104.407290 | 104.416761 | 104.404068 | +0.01061% |
| 6 | 101.717263 | 101.732531 | 101.725960 | +0.01074% |

## 缓存、质量与 embedding

所有正式请求均 valid、stream_done、finish=length，目标模型每个 metrics 窗口 request_success=1；prompt/generation token 计数与 usage 一致，prefix hits、external prefix hits、cached prompt tokens、preemptions 均为零。正式 prefill 12/12 配对及 decode 6/6 配对在三臂中都满足 embedding 请求/输入/生成计数不变，且前后 running/waiting 为零，因此静默子集结果与全量相同。这个条件只约束两次 scrape 的已存计数，不能证明其间绝无瞬时 GPU 工作。

各臂五项 quality 同名案例均通过，但原 raw **没有 payload SHA**，跨臂 quality 输入身份标记为 `null/unavailable`，不把缺失值相等称为一致。四项 open-output 均通过且正常 stop；对已保存 payload 的 canonical JSON 和正文/reasoning 重新计算哈希，三臂 4/4 输入与输出相同。上述是固定合成 API 检查，不是 MoBase 端到端或广泛模型质量评估。

53 个 [embedding monitor 样本](ffix-v1/embedding-monitor.jsonl) 全部 HTTP 200、running=true，容器 ID、启动时间与 restart_count 不变，最大采样间隔 **15.068669s**。三臂共六个 before/after 服务检查均保持原 27B/9B/ASR 停止、embedding 运行，实际 embedding 请求返回 1024 维有限向量；身份与启动/重启字段亦独立比较一致。这些是该窗口状态证据，不代表持续无间隔监控。

## null-metadata 契约的独立 CUDA poison 检查

root 在 fresh A/B/A 后串行执行了 [probe](verify_null_metadata.py)；[null-v1 状态](null-v1/status.json) 为 completed/exit0（UTC 10:59:50–10:59:58），[结果 JSON](null-v1/null-metadata-poison.json) passed=true。固定设备为 RTX PRO 6000 Blackwell Workstation，torch 2.13.0+cu130、vLLM 0.31.0。probe hash `ca4bdd67…e64b61`、原/新/官方三源 hash 与冻结版本一致。

这是 **AST 抽取原版与修复版同一 custom op 函数体** 的窄测试：只去掉框架 decorator/annotation，保留 guards；AST 确认两函数除新增官方量化语句外完全相同。`self._forward_core_fused_norm_packed` 被明确替换为 **无状态、无 CUDA、无 tensor 写入的 stub**；固定官方源码的实际 null-metadata 分支只 prewarm+return。投影输入是 stub 不读取的零元素 CUDA placeholder，core 是真实 BF16 CUDA 零张量 `[M,48,128]`，q/s 预填 FP8 NaN byte `0x7f` / FP32 NaN。参考是官方 CUDA `per_token_group_fp8_quant` 对同一零 core 的输出。

| M | FP8 元素数 | scale 元素数 | 旧 q/s poison 未写 | 修复版 FP8 byte差 / scale bit差 | 修复版非有限 q / s |
|---:|---:|---:|---|---:|---:|
| 1 | 6144 | 48 | 全部保留 | 0 / 0 | 0 / 0 |
| 256 | 1572864 | 12288 | 全部保留 | 0 / 0 | 0 / 0 |
| 32768 | 201326592 | 1572864 | 全部保留 | 0 / 0 | 0 / 0 |

三例 core 仍为零，scale 为列主 stride `(1,M)`。旧分支留下所有 poison，修复分支与官方全 bit 相同且有限，支持“补齐 null-metadata q/s 输出契约”这个具体修复。**没有运行真实 packed warmup、没有加载整模、没有测 decorator/custom op registration 或 CUDA Graph**，因此不能把此 probe 称为模型 E2E 或图正确性证据；整模行为由前述 fresh A/B/A 另证。

## 因果边界与复现

历史 B 曾是正文 0/6 相同，BN 正文相同但每例 accepted 不同；实际 splitfix 配置已确认，不能再用“B 还没 splitting op”解释。此前带 reference/clones/sync 的 fdiag 也恢复了文本与 accepted，但显著改变执行过程。现在没有 diagnostics 的最小 mdNone 修复版本，在一个 fresh A/B/A 窗口恢复三臂所存输出与 MTP 计数一致。这组证据与明确的输出契约 bug 相容，**仍未通过旧/新版本在同一 fresh 配置下反向切换复现等实验唯一隔离历史分叉因果**，也没有首分叉 token/logits/逐层状态轨迹。不要把“发现真实 bug”“窄契约测试通过”“这轮 fresh API 对照通过”合并成“历史分叉唯一根因已证明”。

CPU 复算（不访问服务器/GPU）：

```sh
python3 _dev/docs/evidence/sm120-next/takeover/analyze_ffix_aba.py \
  --root _dev/docs/evidence/sm120-next/takeover/ffix-v1 \
  --output /tmp/ffix-analysis.json
```

分析器要求三臂和 controller completed、固定 manifest、运行配置等价、全部请求/输出 hash 与计数门、服务与 monitor 门；异常会非零退出。输出附原始文件与分析器 SHA。CUDA poison 的原执行命令、container/status/服务记录保留在 [null-v1](null-v1/)；原冻结 runtime 未在本次审阅中更改。
