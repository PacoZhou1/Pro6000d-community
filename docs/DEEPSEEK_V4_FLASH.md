# DeepSeek V4 Flash R33：公开重建材料与官方镜像对照

本仓库的中文名为 **Pro6000d社区**。这一节记录四张 RTX 6000D 上 DeepSeek-V4-Flash-0731 的 R33 部署。它发布可审核的源代码差异、启动器、版本锁和脱敏测量汇总；不发布模型权重、容器镜像或 writable layer、缓存、完整 `docker inspect`、凭据、环境文件和原始问答/思维链。

## 可重建范围

R33 的唯一已确认 host bind 源码改动是 `patches/deepseek-v4/fused_moe.m4-packed-r33.patch`。该补丁针对 B12X `59d51a36a942d56a9c36265855cdc7856fa7712e`，在严格限定的 RTX 6000D、SM120、156 SM、TP4、W4A16、M4、24 routed rows、256 experts、4096 hidden、512 intermediate、top-k 6、SiLU 和 `fp4_e8m0_k32` 条件下，提前选择既有 packed route；不满足任一条件即保留上游分支。

使用 `deploy/deepseek-v4/apply-fused-moe-patch.sh /path/to/b12x` 前会校验固定提交并执行 `git apply --check`。`serve-ds4-flash.sh` 是公开的环境参数启动器；运行者必须自行提供与 `VERSION_LOCK.json` 一致、且拥有合法使用权的 vLLM/B12X 运行时和模型。

本项目使用 Apache-2.0。B12X 与 vLLM 的固定上游也采用 Apache-2.0；补丁保留其文件路径与上游归属，详见 `THIRD_PARTY_NOTICES.md`。

## 已验证的官方镜像对照（首批 6 轮）

对照使用相同的 0731 权重。官方侧为 vLLM `0.29.0`（commit `98dff2a81d747d1dba01a47f939f48c3526d4206`，image digest `sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b`），CUDA 13.0.2、FlashInfer 0.6.18、TP4 NCCL、SM120 FlashInfer sparse MLA、DeepGEMM MXFP4；PCIe 大于两卡时关闭 custom all-reduce。官方轮次关闭 speculative decoding。

三道题分别为关系、理财、学习；官方每题两轮，与历史 R33 单轮对应。官方 6 轮均产生 `done`，53 个请求均有完整 `finish` 与 `DONE` SSE，错误数为 0。

| 题目 | 工具调用：官方两轮 | 工具调用：R33历史单轮 | 正文去协议字符：官方两轮 | 正文去协议字符：R33历史单轮 |
| --- | ---: | ---: | ---: | ---: |
| 关系 | 16 / 14 | 17 | 720 / 902 | 889 |
| 理财 | 9 / 16 | 11 | 765 / 482 | 780 |
| 学习 | 16 / 19 | 14 | 662 / 648 | 573 |

官方单轮耗时为 163–307 秒，R33 历史单轮为 26–35 秒。以客户端 `(completion - 1) / (elapsed - firstToken)` 的加权估计，官方为 44.86 tok/s，R33 为 240.24 tok/s；TTFT 中位数分别为 3.708 秒和 0.784 秒。

这些是跨引擎、多组件差异的观察，不能归因到单一内核、补丁或 speculative decoding，也不构成统计等效或质量结论。官方 API 没有提供缓存字段；日志末次的 80.9% 只对应日志窗口，不能表述为六题整体缓存率。R33 三题有精确缓存记录为 87.0%。本次没有精确 prefill 指标，故未推断或比较 prefill。

官方 full20 的两并发运行已启动，但尚无结果，本文不将其列为完成验证。
