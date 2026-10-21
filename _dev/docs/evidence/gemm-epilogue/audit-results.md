# 实际数据流与实验状态审计

2026-10-14。本页记录本轮已经观察到的事实，候选正确性和加速结论另由后续数值及模型实验决定。

- 固定镜像、vLLM 提交和提取的 CUTLASS/Python 文件见 [source-manifest.json](source-manifest.json)。[dataflow.md](dataflow.md)解释真实 shape/stride、BF16 舍入边界、按 128 行 gate/up 交错的权重置换及理论逻辑流量。
- 修复诊断脚本的 SymInt 序列化后，临时模型已实际启动；[server.log](fx-proof-v2/server.log)保留启动与 API 日志。固定 KV 预算为 `12884901888` 字节，模型最大上下文 262144，MTP4，FP8 KV，FlashInfer。
- 实际启动保留 canonical 权重并生成 64 份主模型和 1 份 MTP 的 packed 副本，累计 **11,589,593,600 字节**，见 [累计清单](fx-proof-v2/startup-pack-516-25fb56bf5f9d-summary.json)。这是新增显存成本；A、B 都必须保留相同副本和 KV 预算。
- 实际 FX 子图出现 `_C.cutlass_scaled_mm` → BF16 `getitem` → V2 `silu_and_mul_per_block_quant`。BF16 节点同时有 `sym_size(dim=0)` 形状消费者；唯一数据消费者是激活量化。删除中间 BF16 时需把形状依赖改到同 M 的 A，不能忽略此边。
- [21 份实际图的汇总](fx-proof-v2/summary.json)只证明图连接，**不授权替换**。图中数字 module stack key 不足以证明权重身份；后续运行时以启动登记的 canonical B/Bs 存储身份联合匹配对应 packed 副本，未匹配必须回退。
- [原始状态](initial-state.json)、[停用次序](pause-sequence.json)及[停用后功能检查](service-state-after-pause.json)已保存。按用户最新要求，原 27B、9B、ASR 保持停用，直到用户明确要求启动；不再自动恢复。embedding 保持原容器、启动时间和重启次数，实际 embedding 请求返回 1024 维有限向量。

`fx-proof/server.log` 是首次诊断失败证据：JSON 写入不能序列化 SymInt；它没有得到模型测试结果。第二次 `fx-proof-v2` 已修复，并按模型加载事件分别保存主模型/MTP 清单，避免同 PID 的第二次加载覆盖第一次。

首次无 profiler 基线已保存于 [a1](a1/)。其中 12 次正式 prefill 的首文本时间中位数为 2.235234 秒，6 次正式 decode 估计速率中位数为 106.8111 tok/s；另有各一条 warmup。此时尚无候选及重复基线对照，不能据此宣布优化收益。
