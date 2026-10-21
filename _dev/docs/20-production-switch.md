# 服务恢复与新27B生产切换

2026-10-15。用户授权恢复原服务，并将27B换成固定FP8组合镜像，保留原端口和密钥。已完成。

| 服务 | 原端口 | 当前状态 | 实际验证 |
| --- | ---: | --- | --- |
| Qwen3.6-27B | 8000 | 新组合镜像运行 | 聊天算术返回391，正常stop |
| Qwen3.5-9B | 8001 | 原容器恢复 | 聊天算术返回391，正常stop |
| Embedding | 8002 | 始终运行 | 1024维有限非零向量 |
| Qwen3-ASR | 8003 | 原容器恢复 | 5秒音频转写正确 |

四项health及models均200，四项密钥与原配置在服务器内存中逐值核对一致，端口映射保持原样；无鉴权models请求均返回401。密钥没有写入文档、脚本或证据文件。27B沿用原服务名 `vllm-qwen-27b` 和served-model-name，MoBase配置无需修改。

27B镜像为 `pro6000d/qwen36-fp8:20261015-r1`，不可变ID `sha256:3835800bf3d91cd3cb9225e3877428cc565f3e4dd7243bfe911759d828f59fb5`。沿用原模型挂载、GPU0、bridge端口映射及unless-stopped重启策略，使用MTP4、V2 runner、FlashInfer和三项打包优化。

为四服务共存，生产KV cache设为 **10GiB**，仍保留 **262144** 最大上下文。实际初始化容量为 **275178 token**（262K请求并发容量1.05x），没有因减少cache下调max-model-len。四服务就绪后GPU使用89879MiB、空闲7371MiB；这只是当时快照。

27B、9B、ASR启动后重启计数均为0；embedding维持原ID、2026-10-06启动时间和原重启计数1。按27B→9B→ASR顺序启动，embedding未停；独立后续复查确认四项容器身份、启动时间、重启计数稳定。

旧27B保留为停止的 `vllm-qwen-27b-before-fp8-r1-20261015`，未删除。旧实验容器 `pro6000d-qwen36-fp8-release` 已停止，18000不再作为当前服务入口。其它非模型服务未改动。

## 性能口径

此前组合镜像实测普通118.17、思考133.64 token/s。服务端prefill时间口径中位 **8804 token/s**（12次，范围8705–8932），约19K输入首token **2.164秒**；按输入量/TTFT估算约8740输入token/s，两种口径不同。

这些性能数据来自此前 **12GiB KV、仅embedding共存** 的测试。本次恢复为 **10GiB KV、四服务共存**，已验收实际功能，没有冒充重新测过共享负载下的性能。[完整性能报告](19-fp8-combined-release.md)。

## 证据和复查

- [生产当前状态](evidence/fp8-release-20261015/production/desired-state.json)
- [启动及切换过程](evidence/fp8-release-20261015/production/20261015-r1/status.json)
- [四项真实功能验证](evidence/fp8-release-20261015/production/20261015-r1/functional.json)
- [身份、端口与鉴权复查](evidence/fp8-release-20261015/production/20261015-r1/followup.json)
- [新27B初始化容量](evidence/fp8-release-20261015/production/20261015-r1/startup-evidence.json)

`activate.py` 是本次已执行的专用切换脚本，不要重复运行；它会拒绝覆盖已有服务/备份。未来核对服务状态以本页和desired-state为准，前轮“原服务保持停止”的记录已被用户本次启动指令覆盖。回退时应先停新27B，再将旧备份恢复为原服务名，避免同端口或重复显存占用；本次没有触发回退。
