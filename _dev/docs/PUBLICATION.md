# Qwen3.6-27B FP8：结果、源码与复现

本分支公开 2026-10-13 至 10-15 的 Qwen27B 推理优化工作：调研、基线、profiling、CUDA/Triton 实验、整模测试脚本、选定原始结果和最终组合镜像的构建材料。2026-10-21 整理发布，没有重新跑 GPU 测试或操作线上服务。

测试机器为 **单卡 RTX PRO 6000 Blackwell Workstation，96GB、600W、SM120**。它与仓库首页的四卡 RTX 6000D 机器不同，成绩不能互换。模型保持原 Qwen3.6-27B FP8 权重及完整 BF16 目标输出头，没有采用 FP4 或重新量化权重。

## 最终实测

并发 1，vLLM 0.31 / V2 runner、MTP4、FlashInfer / FP8 KV，显式 12GiB KV 池，embedding 保持运行且计数在性能窗口内无变化。性能窗口没有 profiler、没有前缀缓存命中；每组剔除一次预热。

| 项目 | 最终中位数 | 次数 | 与最初 v0.31 的关系 |
| --- | ---: | ---: | --- |
| 普通 decode，约 4.8K 输入 / 1024 输出 | **118.17 tok/s** | 6 | 初始 MTP2 为 94.23，跨轮参考 **+25.4%** |
| 思考 token decode，同上 | **133.64 tok/s** | 6 | 初始 BF16 KV / FA2 / MTP2 为 92.93，整体配置参考 +43.8% |
| 约 18.9K 输入 / 1 输出 TTFT | **2.164 秒** | 12 | 初始为 2.179 秒，基本持平 |
| 同组服务端 prefill | **8,804 tok/s** | 12 | prompt tokens / 服务端 prefill 时间，不能拿 decode TPS 比较 |

收益是整个配置包的历史参考，没有把单项收益相加，也没有在最终组合轮重跑旧版。初始 FP8 KV / MTP2 的思考基准缺失；同为 FP8 KV / FlashInfer 的早期 MTP4 思考为 124.60 tok/s，最终比它约 +7.3%。初始 prefill 剔除一次 embedding 活动后的中位数是 2.1668 秒，仍支持“基本持平”。

性能脚本以 `ignore_eos` 固定输出长度，思考 token 速度不等于完整业务完成速度。另有普通、思考、JSON、工具参数四类合成用例，流式 / 非流式各两次，**16/16 通过自然结束及格式检查**；工具仅生成参数，没有执行工具。一次自然思考样例为 1871 思考 + 737 正文 token。最初 2048-token 预算截断的失败记录也保留了；修订仅提高该测试预算到 8192。这些小样例不能证明所有业务输入的质量或无循环。

完整限制及逐请求来源见 [组合发布报告](19-fp8-combined-release.md)、[原始性能数据](evidence/fp8-release-20261015/runs/measure-v2/b/)、[组合分析](evidence/fp8-release-20261015/combined-analysis.json) 和 [历史基准](evidence/fp8-release-20261015/historical-baseline.json)。随后四服务共存使用 10GiB KV 池，只做了功能与稳定性检查；以上性能不是四服务共存的重新测量。

## 采用了什么，哪些没有奏效

最终组合包含 [V2 SiLU → block FP8 量化](evidence/fp8-fusion/fused_silu_quant_group_major.cu)、[F1/F2 GDN 融合和预热输出契约修复](evidence/sm120-next/takeover/f-mdnone-fix/)、[固定 65,536-token 草稿词表](evidence/fp8-release-20261015/received/src/patched/)，并使用 MTP4。目标模型的完整 BF16 `lm_head` 不变；草稿子集会改变提议和接受工作量，不能把 decode 差异全部归给 kernel。

正常图执行的 profiling 中，FP8 GEMM 名称家族占 prefill / MTP4 decode 的 GPU kernel 累计时间约 68.5% / 62.0%；eager 归因中的 BF16 GEMM 大头是目标 `lm_head`。两个捕获口径不同，累计 kernel 时间也不是整条请求的墙钟时间。见 [profiling](08-gpu-profiling.md) 和 [实际 GEMM 尺寸与归因](09-targeted-gemm.md)。

| 实验 | 实测发现 | 结论 |
| --- | --- | --- |
| FP8 GEMM → SiLU / 量化 epilogue 融合 | v6 整模 TTFT 配对退化 7.09%；更早 v4 退化 22.75% | 未采用 |
| M64 tile / warp / 调度优化 | shared load wavefront 减少仍未转化成整模收益；主循环候选 TTFT 退化 2.63% | 未采用 |
| 小 M down Stream-K | 热缓存微基准快约 27%，整模 decode 仍低于前后基线 | 未采用 |
| MXFP8、rowwise GEMM | 实际构建并复测，未通过预设 1.10× 门槛 | 没有转权重 |
| F1/F2 修复后的整模 A/B/A | 该窗口 prefill 配对 TTFT 改善 1.49%，decode 基本不变 | 纳入组合，收益不累加 |
| 64K 草稿词表与最终组合 | 历史单项思考 148.10，组合只到 133.64；六条 reasoning 均不同，接受工作量也不同 | 如实保留，不能称所有场景最优 |

算子链、数值检查、实际派发、硬件计数器与无 profiler 的整模测量分别保存。没有用微基准成绩替代整模收益，也没有把合成形状的计数器当作整模瓶颈的直接证明。

## 从哪里开始复现

先在仓库根目录重算已发布的原始数据；这一步只读本地文件，不连接服务器：

```sh
python3 _dev/docs/evidence/fp8-release-20261015/clients/analyze_combined_only.py \
  --root _dev/docs/evidence/fp8-release-20261015/runs/measure-v2/b \
  --acceptance _dev/docs/evidence/fp8-release-20261015/runs/smoke-v2/b/acceptance.jsonl \
  --output /tmp/qwen27b-combined-review.json
```

预期为 12 prefill + 6 普通 decode + 6 思考性能记录有效，16 项自然结束检查通过。输出文件必须不存在，分析器不会覆盖原结果。最终轮仅测组合版；[五臂分析器](evidence/fp8-release-20261015/clients/analyze_five_arm.py) 是保留的实验工具，不代表已经完成了五臂最终测试。

测试入口和主要实现：

- [固定性能客户端](evidence/profiling/profile_client.py)、[自然结束 / SSE / JSON / 工具参数客户端](evidence/fp8-release-20261015/clients/natural_probe.py)。
- [SiLU 量化实现与实验](evidence/fp8-fusion/README.md)、[GEMM epilogue 实验](12-fp8-gemm-epilogue.md)、[主循环实验](15-sm120-m64-mainloop.md)。
- [F1/F2 修复与验证](18-claude-fp8-takeover.md)、[草稿词表实验](evidence/sm120-next/d-draft-vocab/README.md)、[GEMM 上限实验](evidence/sm120-next/a-gemm-ceiling/README.md)。
- [镜像集成](evidence/fp8-release-20261015/image/integration.md)、[冻结源码锁](evidence/fp8-release-20261015/image/source-lock.json)、[历史镜像记录](evidence/fp8-release-20261015/release-lock.json)。

### 构建组合镜像

先生成全新的构建目录，检查所有冻结输入的 SHA、草稿 ID 子集及 Linux ELF：

```sh
python3 _dev/docs/evidence/fp8-release-20261015/image/prepare_context.py \
  --output /tmp/qwen27b-public-context
```

该步仅需 Python / 文件系统，不需要 Docker 或 GPU。本仓库包含构建所需的约 1MiB 自研量化扩展，同时附有 CUDA 源码、原构建命令和 SHA。扩展为 Linux x86_64 / SM120，不能在 macOS 直接加载。其余编译产物没有纳入发布。

在 Linux 构建机上缓存好固定 vLLM 基础镜像，再运行：

```sh
docker build --network=none --pull=false \
  -t pro6000d/qwen36-fp8:source-build /tmp/qwen27b-public-context
```

基础镜像固定为 vLLM 0.31、提交 `db9527a46873454610df6dbedf79a36d6bf1a7f6`，完整 digest 见源码锁。构建不下载依赖、不编译 CUDA、不加载模型，并执行八种开关组合的 CPU 安装检查。公开版的路径脱敏会改变部分打包 / provenance 文件，重建镜像的 ID 不保证等于历史 `20261015-r1`；该历史 tag 也不是公开镜像仓库的可拉取承诺。

权重按原实验从 ModelScope 准备到自己的模型目录，再显式启动隔离测试端点：

```sh
MODEL_DIR=/path/to/Qwen3.6-27B-FP8 \
PRO6000D_IMAGE=pro6000d/qwen36-fp8:source-build \
bash _dev/docs/evidence/fp8-release-20261015/deployment.example.sh
```

该示例使用 loopback `18000`、12GiB KV 池，拒绝覆盖同名容器。三个独立开关为 `PRO6000D_RELEASE_V2`、`PRO6000D_RELEASE_F`、`PRO6000D_RELEASE_DRAFT64K`，取值 `0/1`，默认全开。改变开关需要新进程。原始实验 controller 是特定服务器的实验记录，其中有容器启停 / 清理逻辑；复现前应按自己的环境检查，优先使用以上公开入口。

## 文档与证据目录

[总索引](README.md) 保留 01–20 全套报告：01–07 为调研、基线与投机解码；08–09 为 profiling / 实际 GEMM；10–15 为量化融合、epilogue 和硬件约束实验；16–17 是历史计划，不能当作已完成结果；18 为中断实验接手；19 为最终组合；20 为生产切换的历史验收。

`evidence/` 保留各轮源码、脚本、补丁、来源锁、摘要、逐请求记录、必要日志及数值 / Graph / 派发检查。草稿词表和 prefill 实验中原先被 `runs/` 忽略的必要数据也收录。重复的大量逐调用 witness 只保留汇总和关键记录；原始 profiler 数据库、权重、校准语料、下载目录、虚拟环境不发布。

<a id="publication-boundaries"></a>

## Publication boundaries

线上 IP、SSH 用户目录及本机个人路径改为 RFC5737 示例地址和通用目录；凭据、私有连接诊断、代理配置、个人工具设置不发布。示例地址不能用于访问原服务器。保留的是合成测试输入 / 输出，含部分 reasoning；不包含客户业务语料。

[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) 逐文件记录原始 SHA 与公开 SHA，以及筛选 / 脱敏规则。历史报告内的 artifact hash 可能指脱敏前文件，不应拿来验证公开元数据的字节；验证本次文件用公开清单，验证数值实现用独立的镜像源码锁。本文及公开导航调整也记录在清单中。

上游源码保留原版权头；vLLM / FlashInfer 为 Apache-2.0，CUTLASS 为 BSD-3-Clause，具体来源在实验 manifest 与 [第三方声明](../../THIRD_PARTY_NOTICES.md)。模型、CUDA 和其他依赖仍适用其各自条款。

## 本次发布校验

本机已通过冻结输入校验及构建上下文生成、24 条性能原始记录与 16 项验收的离线重算、21 项现有 CPU 自检、496 个 Python 文件语法检查、21 个 shell 文件语法检查和本地 Markdown 链接检查。独立复核未发现原服务器地址或真实凭据。详见 [校验记录](publication-validation.json)。本次没有构建 Docker 镜像、运行 GPU 或重新验证线上服务。
