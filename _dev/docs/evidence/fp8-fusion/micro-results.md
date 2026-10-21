# 真实 GPU 数值与单算子测量

两版均在固定镜像、SM120 上通过 12 个输入组合的 FP8 字节与 FP32 scale 位模式检查。五个 M 的 CUDA Graph 在捕获后改变输入，再与新执行的官方核比较。M<256 回退官方。异常值包括 NaN、±Inf、零、极小值、较大有限值及有限 BF16 乘积溢出。通过只适用于这组测试，不是全部可能输入的形式证明。

V2 的 group-fast CTA 顺序胜出：M2741 的 warm/cold 图中位数为 146.72/176.13 微秒（同次官方 206.91/228.32）；M16160 为 894.59/917.50 微秒（官方 1223.07/1255.42）。每项 31 次，范围在原始 JSONL。候选分别减少约 29.1%/22.9%、26.9%/26.9%。这是合成单核图重放，包含图发射计时开销，不代表整模收益。冷缓存通过 256MiB tensor fill 近似，不等同真实模型权重流。

V2 接入 smoke 的 14 项检查通过：auto_functionalized、先 M5 后 M256 的 FX/Inductor，以及六类合法 fallback 与目标路径。回退案例通过临时官方 packet 计数验证；目标 M256 没有调用官方 packet。后续服务检查仍须证明模型实际选中候选。

原始数据：`micro-v1.jsonl`、`micro-v2.jsonl`、`runtime-smoke-v1.jsonl`、`runtime-smoke-v2.jsonl`；汇总 `micro-comparison.json`。V1 smoke 对应提交 e5d89bf 中未扩展的脚本，V2 对应当前增加 fallback 矩阵的脚本。
