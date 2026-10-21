# FP8 fused SiLU quant：隔离实验材料

用户本轮只要求优化 FP8；不测试 FP4、不定位历史循环、不改变 BF16 `lm_head`。所有服务操作仅限已授权的三个非 embedding 模型。`initial-state.json` 记录实验前身份与健康状态，`session_control.py` 持续检查 embedding 的容器 ID、启动时间、重启次数和 health。

`profile-opportunities.json` 从四份已有真实正常图模式 SQLite 重新汇总，保留原始名称、次数、分母与文件 SHA。当前选择 **已融合** SiLU+FP8 quant 的内部线程组织作为首个候选：每个 128 元素组由一个 warp 处理，每 CTA 四个组，去掉原 128 线程 CTA 的 shared-memory 树形归约。该核在原 prefill 捕获的 kernel-duration 总和中约占 4.32%，decode 约占 0.286%；它的算子收益不能当作 HTTP 收益。

`installed-source-manifest.json` 固定了镜像中实际 Python 源码。上游 C++ 源码按同一 vLLM commit 保存并校验；原始大文件与编译产物忽略入 Git。编译和运行只在独立容器中进行，不覆盖镜像、模型权重、原服务或 MoBase。

数值门槛是与原融合核的 FP8 字节及 FP32 scale 位模式一致，覆盖真实尺寸、分派边界和异常值。通过后才验证图执行，再进行独立进程的 A1/B/A2 测量。每臂长输入 prefill 12 次、普通 decode 6 次，复用同一个固定 payload 构造器；不同 index 使用独立前缀，相同 index 跨臂一致。固定 token 性能请求与正常结束的质量请求分别记录。

本文件是实验设计，不代表候选通过数值、集成或性能验收。实际结果见 [报告](../../11-fp8-fusion.md)。
