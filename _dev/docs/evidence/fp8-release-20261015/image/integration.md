# FP8 release 镜像集成

目标是把已测的 V2 SiLU+blockquant、F1/F2（含 mdNone q/s 契约修复）、固定 draft 65536 子集放进同一个镜像，再由 root 实测组合收益。这里只完成本地 CPU 集成与检查，没有运行 Docker、GPU、模型或服务，不能把单项历史成绩相加作为组合收益。

## 固定输入与数学边界

基础镜像固定为 `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`（0.31.0，commit `db9527a4…`）。[source-lock.json](source-lock.json) 保存完整 SHA，关键输入为：

| 输入 | 固定 SHA 前缀 | 来源 |
|---|---|---|
| F site（mdNone fix） | `854043107312` | `sm120-next/takeover/f-mdnone-fix` |
| F Triton kernels | `1b51a6d934e2` | 同上 |
| V2 site | `8cdace709c1c` | 同上 |
| V2 量化 `.so` | `087b7caa92c4` | 已实测二进制；不在镜像构建中重编 |
| draft speculator | `f97f98fdeeb8` | 取回的实际 retest 已测源码 |
| draft_vocab | `8b1416c634c8` | 同上，保留原字节及历史 docstring |
| draft subset | `a9773fbd8a79` | 固定 `subset_65536.json`，65536 个排序唯一 ID |
| 原 speculator | `da3b576318cf` | 固定镜像 pristine |
| 原 base sitecustomize | `43d81125d923` | 固定镜像原件 |

`prepare_context.py` 同时核对取回的 `draft64-retest-arm-meta.json` 中三项实际 source/subset SHA 与固定镜像标识。V2 的 `.cu`（`a245bb14…`）与实际 `build-v2.log` 一同装入 `/opt/pro6000d/reproduction`，记录 nvcc 13.0.88 / sm_120 的原构建命令与库 SHA；这些是复建材料，不会默默用重编产物替换已测库。

F/V2/draft 数学文件均原字节复制，没有改 FP8 scale、舍入、GDN/F2 状态更新或 draft 的 BF16 linear/argmax。draft 只新建 BF16 权重行切片与 ID 映射；目标模型的完整 BF16 lm_head 不改，不引入 FP4。固定子集可能改变 draft 提议和 accepted 工作量，组合模型测试必须按原始请求输出与 MTP 计数解释。

## 三个独立开关

| 环境变量 | 默认 | 关闭效果 |
|---|---:|---|
| `PRO6000D_RELEASE_V2` | `1` | 不安装 V2 FUSED_OPS 替换，使用官方量化 |
| `PRO6000D_RELEASE_F` | `1` | 不安装 F1/F2 forward/opaque/conv 包装与 splitting op |
| `PRO6000D_RELEASE_DRAFT64K` | `1` | 不安装 draft 导入钩子，直接读取基础镜像原 speculator 与完整 draft lm_head |

值只接受 `0/1`，未知 `PRO6000D_RELEASE_*` 拼写报错。三项全 `0` 是同一镜像的官方无融合/完整 draft 路径；仍执行只读源完整性检查及原 base sitecustomize，且镜像默认的 `VLLM_USE_V2_MODEL_RUNNER=1`/spawn 保留。全关仅指这三类自定义优化全关，不表示恢复基础镜像所有环境默认值。不是依靠 patched speculator 的 env fallback 模拟 OFF。

Dockerfile 还默认 `VLLM_USE_V2_MODEL_RUNNER=1`、`VLLM_WORKER_MULTIPROC_METHOD=spawn`，与已测路径一致。DRAFT64K 开启时会拒绝非 V2 runner；实际 speculator 的工厂另拒绝 `use_local_argmax_reduction=true`，因为它在 `sample_draft` 中优先于子集分支，否则会静默绕过子集。原 draft 工厂的 strict=1 守卫继续负责拒绝非 MTP、概率提议、adaptive verification、watermark、TP>1、不合适的 head dtype/quantization 等；工厂返回 None 亦明确失败。没有新增 logits 或采样数学。

对外只使用三个 release 开关。bootstrap 派生 `PRO6000D_FP8_FUSION_*`、`PRO6000D_F_ENABLE`、`PRO6000D_F_KERNELS_SHA256`、`PRO6000D_F_V2` 与 `VLLM_DRAFT_VOCAB_*` 为包内绝对路径和冻结 SHA；若用户提供的 legacy 值与派生值冲突，启动失败。release 禁用 F VERIFY/VERIFY_CONV 和 diagnostic helper。改变开关需要重启进程，不能在已经导入模型后动态切换。

`PRO6000D_F_V2=1` 在内部始终表示“执行 V2 guard 脚本以保留原 base site”，实际 V2 是否启用由 release V2 开关派生的 `PRO6000D_FP8_FUSION_ENABLE` 决定。不要从该 loader 字段误判 V2 数学是否启用。

## 加载顺序与失败策略

包全部位于 `/opt/pro6000d`；不需要宿主 `/instrument`、`/subsets`、源码或库挂载，模型权重仍由部署侧提供。build-time installer 只在 Python purelib 安装一个 `.pth`，将 `/opt/pro6000d/bootstrap` 放到搜索路径首位。保留基础镜像 `ENTRYPOINT=["vllm","serve"]`、Cmd=null、WorkingDir=/vllm-workspace，不提供自定义 entrypoint，也不改 CLI 参数。

进程启动顺序为：bootstrap sitecustomize → 检查整个 bundle manifest/固定 installed vLLM 文件 SHA/原 base site SHA → 解析三个开关 → 可选 draft 专用导入钩子 → 原字节 F site → 原字节 V2 site → 原 base sitecustomize 一次。原 base site 只是 apport handler，实际原件也归档在包中，但运行时仍执行镜像原路径。F/V2 的 `_HERE` 均指向包内 runtime 目录；原 docstring 中历史 `/instrument` 文案未被修改，部署以本说明为准。

镜像不覆盖 site-packages 中的任何 vLLM `.py`。DRAFT64K=1 时，导入钩子只将 `vllm.v1.worker.gpu.spec_decode.speculator` 与新 `draft_vocab` 指向包内冻结文件，载入后给 head 工厂加“拒绝静默绕过”的 CPU 检查。F/V2 钩子目标不同，不竞争同一个 fullname。OFF 不安装该钩子，原 speculator 字节仍由源 SHA 守卫确认。

默认 worker 使用 spawn：新 Python 进程由永久 `.pth` 自行加载同一 bootstrap，派生环境在子进程中再次严格校验，不依赖父进程临时 `sys.path` 或 host instrument mount。fork 若由上游其他 CPU 进程路径使用，会继承已注册钩子/模块状态；`activate()` 的一次性守卫避免重复注册，不能在 fork 后通过改 env 重新选版本。这里说明的是引导状态的 spawn/fork 兼容设计，不宣称 CUDA 初始化后 fork 或非默认 GPU worker 方式已实测通过；组合模型验收以 root 实际启动配置为准。

普通 Python 会吞掉 sitecustomize 的异常并继续启动。因此 bootstrap 捕获任何启动错误，打印 traceback 和 `PRO6000D_RELEASE_FATAL` 后 **exit78**，不能带着未生效补丁继续服务；后续 lazy import 的源/注册错误照常传播至 vLLM。不要用 `python -S` 或覆盖 `/opt/pro6000d`/`.pth` 绕过该机制。没有用宽泛异常降级成“优化已开启”。原 F 对不满足融合守卫的真实请求保持原本官方 fallback；原 prewarm 的失败日志行为亦保持原字节，不能把单次成功 marker 当全 dtype/shape 专门化或逐调用派发证明。

## 构建与检查

本地或 root 的构建主机先生成一个全新目录，输入必须与 source lock 精确匹配：

```sh
python3 _dev/docs/evidence/fp8-release-20261015/image/prepare_context.py \
  --output /tmp/pro6000d-release-context
```

默认路径来自本仓库；在服务器可用 `--runtime`、`--received`、`--v2-library`、`--v2-source`、`--v2-build-log` 指定已经固定的原件位置。输出目录已存在则拒绝，不会覆盖旧结果。所有检查完成后才复制 payload，输出 bundle manifest SHA；Docker build context 只含自己的 payload、Dockerfile 和 dockerignore。

root 执行 Docker 构建（本 agent 未执行）：

```sh
docker build --network=none -t pro6000d/qwen3.6-27b-fp8:20261015 \
  /tmp/pro6000d-release-context
```

固定 base 需已在构建主机缓存。构建没有 pip/apt/download、没有 CUDA 编译、没有模型加载。Dockerfile 在安装 `.pth` 前审计固定 base 源码，随后以八种开关组合启动真实 Python 做 CPU-only installation matrix，要求使用正确 sitecustomize、draft 选择路径、flags 和完整性守卫，且没有导入 torch。该构建门不是 GPU、CUDA Graph 或模型正确性门。

保留 base 默认入口时，容器参数应从 `/model` 开始；如果 controller 传 `serve /model ...`，需要沿旧方式显式 `--entrypoint vllm`，否则会重复 `serve`。

## 本地已完成和待 root 实测

[cpu_self_test.py](cpu_self_test.py) 的 [21 项 CPU 自检](cpu-self-test.json) 已通过：八组合派生、非法/未知开关、legacy 冲突、runner/diagnostic 拒绝、local-argmax/no-head 拒绝、hook 路径、synthetic payload/upstream/base SHA 损坏拒绝，以及真实 Python 子进程验证 sitecustomize 错误退出 78。fixture 不冒充固定镜像源码。真实冻结输入的 prepare_context SHA/AST/65536 ID/原二进制 ELF 与 recipe 检查亦已本地通过。

root 后续须保存最终 image ID/RepoDigest、Docker history/inspect、bundle manifest 与构建日志，验证无 instrument/subset 挂载时组合模型正常启动、实际三类优化均生效，并用同镜像 OFF/ON/OFF 或约定矩阵完成质量、输出、MTP 工作量、embedding 静默子集和完整 API 性能检查。共享的目标 BF16 head 不变；DRAFT64K=1 仍多分配约 640MiB BF16 子集副本，关闭时没有该副本，报告需如实记录，KV 预算由控制器显式对齐。

历史 mdNone q/s 输出契约 bug、窄 poison probe 通过以及 fresh ffix API 通过是分别成立的证据，不等于历史 B 分叉的唯一因果已证明。组合版本改变 draft 候选本来就可能改变 accepted；不能把不同 accepted 工作量下的 decode TPS 全部归因于 F/V2 kernel。新的组合镜像在 root 实测前不宣称最终加速或上线结果。
