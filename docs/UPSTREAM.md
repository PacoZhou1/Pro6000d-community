# 上游贡献建议（尚未提交）

建议先将可复现实验交给 **Local Inference Lab / R30维护者**，再讨论通用部分进入vllm-project/vllm。本次实现依赖R30分支中的GLM/B12X接口；镜像所用vLLM源码指向 [voipmonitor/vllm](https://github.com/local-inference-lab/rtx6kpro/blob/2a763eb0de595cd6a432971781dbaf1634f858bc/models/glm-5.3-flash/validation/shared-serving-r30.md)，并不等于vllm-project/vllm主线已经有同样代码。

## 拆成三个独立事项

1. **6000D硬件部署与实测报告**：可先给rtx6kpro提文档贡献，包含350W、84GB、所有测量口径、未达到目标和负面实验。收益可由同行复测，不能把不同精度/硬件混成严格A/B。
2. **小批量BF16投影dispatch**：针对准确shape/device guard、初始化编译、torch.compile/custom op和fallback提交小补丁；保留原算法作者。需在维护者当前分支重放数值测试、M1–4/大batch/其他设备fallback、C1/C8/C64以及内存证据。
3. **mHC输出生命周期**：以最小资源保留复现、显存释放及bitwise回放为bugfix提交。不要标注prefill加速：大graph真实测试没有加速。

先询问维护者对应模块的当前所有者与目标分支；本仓库的完整替换文件仅适用于锁定R30镜像，不直接覆盖更新版本。2026-09-10社区主文档已经到R32，应先确认类似问题是否已修复。环境变量、机器专用systemd与调参表通常不适合直接作为vLLM主线代码PR。

## 到vLLM主线前还缺什么

把候选拆成可独立审查的小diff，重基到维护者认可的当前commit，补通用设备/shape选择与回退测试，排除精度变化带来的混淆，提供真实回归和性能命令。按 [vLLM贡献指南](https://docs.vllm.ai/en/latest/contributing/) 完成代码规范、测试与贡献流程。当前只能说已有值得审查的代码和证据，不能承诺上游会接受。

本仓库发布不等于已提交issue或PR；尚未向这些项目发送消息。
