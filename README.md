# 四张 RTX 6000D 跑 GLM-5.3-Flash

我们最初直接使用**社区发布的原始镜像**，在自己的四张 RTX 6000D 上测得：**单路约 132 tps，8 路并发总速度只有 300 多 tps。**

优化后，单路测到约 **200 tps**，8 路并发总速度约 **648 tps**；长文本处理速度从约 **8,054** 提升到 **9,134 tps**。

*tps = 每秒生成或处理的 token 数。并发这里指 8 个请求同时运行的总速度。*

![社区原始镜像在我们的机器上，与我们优化后的速度对比](docs/final-comparison.svg)

图中只对比我们的同一套四卡机器，展示各项优化后的实测成绩。详细数值和测试设置放在[测试记录](docs/BENCHMARKS.md)。

## 我们的机器

- **显卡：**4 × RTX 6000D，84 GB/卡，350 W/卡
- **整机：**Supermicro SYS-551A-T，X13SWA-TF 主板
- **CPU：**Xeon w5-3433，16 核 32 线程
- **内存 / 硬盘：**64 GB / 2 TB NVMe
- **系统：**Ubuntu 22.04.5
- **模型：**GLM-5.3-Flash-NVFP4

[完整机器配置](docs/HARDWARE.md) · [部署与开机自启](docs/DEPLOYMENT.md)

<details>
<summary>与 4 张满血 RTX PRO 6000 96GB 的成绩对照</summary>

| 测试 | 我们优化后 | 4 张满血 RTX PRO 6000 96GB |
|---|---:|---:|
| 单路生成 | 约 200 tps | 249 tps |
| 8 路并发总生成 | 648 tps | 872 tps |
| 长文本处理 | 9,134 tps | 14,288 tps |

这里比较的是两套不同的机器。我们说的“社区原始镜像成绩”，始终指**原始镜像在我们四张 6000D 上跑出的 132 / 367 tps**，不是满血 96GB 卡的成绩。不同版本和测试设置见[测试记录](docs/BENCHMARKS.md)。

</details>

## 文件与使用

- [测试记录与数据来源](docs/BENCHMARKS.md)
- [测试脚本使用方法](docs/REPRODUCIBILITY.md)
- [源码改动说明](docs/OPTIMIZATION.md) · [向原项目提交的建议](docs/UPSTREAM.md)
- [原 HTML 跑分报告](reports/html-single/report.html) · [CSV](reports/html-single/report.csv) · [截图](reports/html-single/table.png)
- [镜像与源码版本](SOURCE_LOCK.json) · [许可证与第三方声明](THIRD_PARTY_NOTICES.md)

仓库包含部署脚本、源码改动和实测数据；不包含模型权重、镜像文件或登录凭据。

## DeepSeek V4 Flash R33

**Pro6000d社区**现已加入 DeepSeek-V4-Flash-0731 R33 的公开重建材料：固定 B12X 上游提交上的 19 行 M4 packed-route 差异、无凭据启动器、版本锁和与官方 vLLM 0.29.0 镜像的首批六轮脱敏对照。完整边界、应用步骤和测量限制见 [DeepSeek V4 Flash R33](docs/DEEPSEEK_V4_FLASH.md)。
