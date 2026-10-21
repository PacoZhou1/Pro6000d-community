# 候选实现静态审查

独立 Sol reviewer 核对固定 vLLM 源码、FP32 SiLU 与量化语义、shape/layout guard、fallback 和图捕获入口。已修复 default CUDA stream=0 被误拒绝及 row/scale 索引的 int32 溢出风险；现用 int64 地址计算。直接调用 CUDA Event 可能包含 Python dispatch 间隙，因此 micro 同时记录 CUDA Graph replay，对小幅差异优先使用后者。

实际 CUDA 13.0 nvcc 无 GPU 容器编译成功，源码和产物 SHA 见 `libfp8_silu_quant.so.json`。这只证明可编译，不证明数值、性能或模型集成。`runtime_smoke.py` 检查 auto_functionalized、M5→M256 FX 正向命中和 Inductor；其 GPU 结果单独保存。
