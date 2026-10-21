# Qwen3.6 FP8 组合镜像实测与固定版本

> 最新状态：用户随后授权生产切换，新27B现已接回原8000及原密钥，9B/ASR已恢复，embedding未重启。当前部署见[服务恢复与生产切换](20-production-switch.md)。下文服务状态描述保留为本轮12GiB隔离测速时的历史记录。

2026-10-15。按用户最终要求，只测我们的组合镜像，取消旧版/新版原版的重复对照。组合服务已保留运行，embedding 保持运行，原27B、9B、ASR继续停止。

## 实际结果

单卡 RTX PRO 6000 Blackwell 96GB，C1，原Qwen3.6-27B FP8权重、完整BF16目标输出头，vLLM0.31/V2 runner、MTP4、FlashInfer、FP8 KV，显式12GiB KV池。每组排除一次预热，性能窗口无profiler、无前缀命中，embedding计数无变化。

| 测试 | 有效次数 | 中位数 | 范围 |
| --- | ---: | ---: | ---: |
| 约18.9K输入、1输出，首token延迟 | 12 | **2.164秒** | 2.137–2.188秒 |
| 约4.8K输入、1024输出，普通decode | 6 | **118.17 token/s** | 112.83–123.39 |
| 约4.8K输入、1024输出，思考decode | 6 | **133.64 token/s** | 129.30–139.97 |

按用户指定，以**最初未做我们优化的官方vLLM0.31/MTP2**为主基准，不计入从0.27升级到0.31的收益。

| 项目 | 最初v0.31/MTP2 | 当前组合版 | 参考变化 |
| --- | ---: | ---: | ---: |
| 约4.8K输入/1024输出，普通decode | 94.23 token/s | 118.17 token/s | **+25.4%** |
| 约19K输入首token延迟 | 2.179秒 | 2.164秒 | 降低0.70%，基本持平 |
| 思考decode | 92.93 token/s（最早思考测试） | 133.64 token/s | **+43.8%**（整体配置参考） |

思考基准补取首次官方v0.31/MTP2实测：92.93 token/s。该次模型仍为FP8，但使用BF16 KV/FlashAttention2；当前使用FP8 KV/FlashInfer/MTP4加自研组合。因此+43.8%是整个配置包的跨轮参考，不是相同KV/后端下的纯算子收益。同为FP8 KV/FlashInfer的早期MTP4思考记录是124.60 token/s，当前比它约+7.3%。最初FP8 KV/MTP2的思考结果仍无记录，不能补造。来源见[首次投机解码测试](07-specdecode-experiment.md)。

历史0.31的三次prefill中有一次embedding活动；剔除后两次中位数为2.1668秒，对当前约降低0.15%，同样应判断为基本持平。普通decode的历史三次窗口没有embedding活动。配置从初始MTP2变为MTP4，并加入自研融合和64K草稿词表；该比较表示整个优化包的跨轮结果，不是单一算子的收益。原始来源见[最初0.31测试记录](01-server-baseline.md#8-官方-v0310同一36-fp8权重)。没有追加服务器测试，也没有把单项收益相加。

**思考组没有复现历史64K草稿单项的148.10 token/s，当前低约9.8%。** 核查确认这六条请求、客户端和token工作量相同，但reasoning输出6/6不同，接受token总数从历史4364变为4245、草拟token总数从7128变为7600。不能据此把差异归因于某个kernel，也不能称完整组合在每个场景都最优。按用户收敛和立即交付要求，本轮不再追加实验。

性能客户端用ignore_eos固定1024输出；思考token速度不等于完整业务任务完成速度。另做的普通/思考、JSON、工具参数（仅生成，不执行工具），流式/非流式各重复两次，**16/16通过自然结束和格式检查**。思考样例共2608token，其中1871思考、737正文；没有触发重复循环检测。首次2048token预算截断已保留，不能隐去；修订只增加该用例预算至8192。小样例不等于全面质量回归，不保证所有输入无循环，重启间措辞也可能变化。

## 固定镜像和服务

- 镜像：`pro6000d/qwen36-fp8:20261015-r1`
- 不可变ID：`sha256:3835800bf3d91cd3cb9225e3877428cc565f3e4dd7243bfe911759d828f59fb5`
- 保留运行的容器：`pro6000d-qwen36-fp8-release`
- 当前服务器内入口：`http://127.0.0.1:18000`。**没有替换MoBase原入口**，当前restart策略为no。
- 模型只读挂载：`/home/llm/models/Qwen3.6-27B-FP8`。
- 镜像内含：V2 SiLU→FP8量化融合、修复后的F1/F2融合、固定64K草稿词表。三者均有实际调用日志；不依赖宿主临时instrument挂载，不改变原FP8权重或目标BF16输出头。
- 镜像默认三项都开启。`PRO6000D_RELEASE_V2`、`PRO6000D_RELEASE_F`、`PRO6000D_RELEASE_DRAFT64K` 各可设0关闭；全关仍保留V2 runner。开关变更需要重建容器，现有服务没有动态切换。

最终两次检查相隔约109秒：组合服务健康和真实JSON请求均通过、容器ID/启动时间不变、重启0；embedding健康和1024维真实向量请求均通过，原ID/启动时间/重启次数不变。原27B、9B、ASR仍停止，没有临时测试容器。未改全局GPU频率、未推送镜像仓库。全程复用本机代理上的同一SSH主连接。

## 使用与复现

已有服务正在运行，无需再次启动。查询：

```sh
curl http://127.0.0.1:18000/health
docker logs --tail 50 pro6000d-qwen36-fp8-release
```

完整启动参数见[启动示例](evidence/fp8-release-20261015/deployment.example.sh)，仅用于需要重新创建时；它拒绝覆盖同名容器。后续如需关闭某项优化，可仅停止并重命名该组合容器，再设置对应开关执行示例；不要操作embedding或启动原来的三个服务。三个开关的CPU组合检查已通过，未把本轮未做的GPU全关对照称为已验证。

固定构建上下文保留在服务器 `/home/llm/.local/share/qwen-fp8-release-20261015/context-r1`，归档为同目录 `context-r1.tgz`。镜像已在服务器Docker中保留，模型权重不打入镜像。构建、源哈希和开关说明见[集成说明](evidence/fp8-release-20261015/image/integration.md)；其中“本地CPU完成”是打包子任务当时的状态，后续实际构建/整模结果以本报告为准。

## 证据

- [固定版本与服务信息](evidence/fp8-release-20261015/release-lock.json)、[Dockerfile](evidence/fp8-release-20261015/image/Dockerfile)、[实际构建结果](evidence/fp8-release-20261015/build-r1/status.json)
- [完整数据分析](evidence/fp8-release-20261015/combined-analysis.json)、[原始性能数据](evidence/fp8-release-20261015/runs/measure-v2/b/)、[实际调用日志](evidence/fp8-release-20261015/runs/measure-v2/b/server.log)
- [自然结束验收](evidence/fp8-release-20261015/smoke-audit.json)、[思考历史口径核查](evidence/fp8-release-20261015/clients/thinking-history-audit.md)
- [历史基准与计算口径](evidence/fp8-release-20261015/historical-baseline.json)、[最后服务检查](evidence/fp8-release-20261015/final/final2/)

运行中还修正了测试控制器对刚关闭端口的检查，防止TIME_WAIT被误作仍有服务；首次启动检查失败记录保留，未因此改动模型镜像。尚未验证并发吞吐、262K实际长上下文、其他模型恢复共存预算或完整MoBase业务链路。
