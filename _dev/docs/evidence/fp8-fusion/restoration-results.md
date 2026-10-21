# 服务恢复与 embedding 保持运行

2026-10-14 05:40:59 UTC（13:40:59 台北时间）完成四项真实功能检查，全部通过；316.63 秒后的独立快照再次通过。27B、9B、ASR 使用实验前同一个容器、镜像与命令恢复，重启计数未增加。27B 仍为原 vLLM 0.27.1、MTP2、0.45 显存预算；本轮候选没有部署到生产。

| 服务 | 恢复后的真实请求 | 结果 |
|---|---|---|
| 27B | 17 × 23 算术请求 | HTTP 200，含正确整数，正常 stop |
| 9B | 同一算术请求 | HTTP 200，含正确整数，正常 stop |
| ASR | 5 秒合成语音 | HTTP 200，归一化转写与预期相符 |
| embedding | 实际向量请求 | HTTP 200，1024 维全部有限，范数约 1 |

embedding 全程未停止。每 30 秒记录一次的 74 个观测点全部健康 200，容器 ID、镜像、启动时间 `2026-10-06T09:31:08.313413609Z`、重启计数 1 保持不变。监测覆盖 05:09:27–05:45:58 UTC，随后稳定性快照再次确认身份及健康。周期观测不是毫秒级可用性证明。

本轮拥有的 `codex-fp8-*`、`codex-gemm-*` 测试容器残留为 0。保留源码、编译产物与原始诊断文件用于复现；没有改模型权重、MoBase 配置或主机驱动。

证据：[恢复顺序](restore-sequence.json)、[四接口功能检查](restoration.json)、[后续稳定性快照](restoration-stability.json)、[embedding 监测](embedding-monitor.jsonl)、[容器清理检查](container-cleanup.json)、[汇总](restoration-summary.json)。离线复核命令：

```bash
python3 _dev/docs/evidence/fp8-fusion/analyze_restoration.py _dev/docs/evidence/fp8-fusion
```
