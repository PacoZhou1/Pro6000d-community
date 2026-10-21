# 定向 GEMM 证据独立复核

独立 Sol 子代理基于原请求、任务表、七项声明、提交 `f3dc981..722784d`、本目录源码和原始证据检查。结论为 `vision_met`、`claim_coverage:7/7`；没有发现可证伪的未闭环缺口。结论仅对应本轮“实际尺寸/模块归因、形状级瓶颈检查、定向候选实现与实测筛选、服务恢复”，不表示生产已提速。

- 两份真实 eager SQLite 离线重新分析与保存 JSON 完全一致。decode BF16 `lm_head` 686.836482 ms / 全部 GPU kernel duration 2689.094715 ms；其 94.26% 是 **eager BF16 GEMM** 内部份额，不是正常 graph 或 API 延迟份额。正常 graph 的核名分类只用于优先级。
- 整模 NCU OOM 记录保留；八组 NCU 是实测形状上的合成单算子，不能证明真实权重流的带宽限制。tile、Split-N、Stream-K 的 GPU 数值检查分别为 6/6、6/6、4/4；其中 Split-N 是两次官方 GEMM 加拼接的实际组合执行，并非另造 CUDA 库。
- guard-v1 编译图没有实验 op，未纳入候选速率。v2 图有 64 个实验 op；最终 smoke-v3 八项通过。原始 replay SQLite 离线重算与报告相同：5395 个实验 kernel 均带 graphNodeId，且准确关联到成功 cudaGraphLaunch。图命中只证明执行，不证明整模速度收益。
- A1/B/A2 的同场景正式 payload 一致，各臂 12/12 性能请求有效、5/5 短质量与 4/4 开放输出合成探针通过。普通 decode 中位 108.99/104.91/107.34 tok/s；B 没有稳定收益。A1 与 A2 自身也产生不同长输出，因此没有把速率差包装为纯核因果效应。
- 恢复原 27B、9B、ASR 并加上持续运行的 embedding 后，四项功能通过；309.580731 秒后复查服务身份/健康/重启未变。本地 embedding 监控 203 次、6063.905059 秒全 200、同一身份；周期采样不证明零延迟影响。七个本轮临时容器已清理。
- 本地 `experiment-manifest.json` 的 13 个源码 SHA、`raw-artifact-manifest.json` 的 19 个原始文件大小/SHA 全部匹配；原始大型二进制不随 Git 提交。对文本证据做高置信凭据赋值模式扫描未见命中，不公开请求或鉴权内容。

审查边界：仅离线复核，未连服务器、未运行 GPU、未验证 MoBase 真实业务、长时共存性能或整模 NCU。具体声明等级和七项对账见同轮 mission 的 `20261014-gemm-targeted.review.md`；用户可读的结果在 [09-targeted-gemm.md](../../09-targeted-gemm.md)。
