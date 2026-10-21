# Pro6000d-community 项目审查：对 Qwen 27B 性能调研的可用证据

审查日期：2026-10-13。仓库快照：`d2c0ec3be319b7d8949778eefd15bb77fa693b8d`，最近提交为 2026-09-22 的 MiMo V2.6 Flash 优化材料。本审查子任务只读取本地项目与主任务采集后保存的脱敏证据并维护本文件；不直接连接 MoBase 服务器、启动容器或改变线上配置。

适用说明：已检查从 `/` 到本项目的祖先路径，未发现磁盘上的 `AGENTS.md`；遵循用户给出的 Windows 到 Apple Silicon macOS 迁移说明，使用当前 `/Users/researcher/Projects/Pro6000d-community`。源码文件及历史结果按本地快照审查，不将文档中的“当前宿主”视为 2026-10-13 的线上探测结果。

## 1. 对本次决策最重要的结论

1. **原始公开仓库没有 Qwen3.6 27B 或 Qwen3.8 的部署入口、权重配置、性能基线与专用补丁。** 已检索 README、docs、patches、MiMo 子目录及 benchmarks。本次另行采集的 Qwen 证据位于 `_dev/docs/evidence/`，不属于原始加速成绩。现有公开成绩可证明团队在固定四卡平台上的优化经验，不能证明 Qwen 27B 的预期速度，更不能单凭版本号判断换 3.8 会加速。
2. **GLM、MiMo、DeepSeek 的 MoE/MTP 结果不能当作 Qwen dense 保证。** 小批次 GEMM 文件名含 `dense`，指 MoE 模型中的部分 BF16 投影，不表示仓库验证过 dense 27B 全模型。其算子选择对模型模块、分片形状、数据类型和 M1–4 有条件限制。
3. **优先复用测量方法和可回退实验流程。** 本仓库强调实际 active 并发、冷缓存、版本锁、token 数、服务端指标、精度变化和负面实验，适合变成本次诊断框架；现成 Docker 替换文件不适合直接装进 Qwen 线上镜像。
4. **实际用户服务器只有一张 RTX PRO 6000 Blackwell Workstation 96 GB/600 W，线上 Qwen 已为 TP1。** 四卡通信、TP4 分片和设备名为6000D的专用 guard 与该单卡环境不同。当前服务器不开展 TP2/TP4 对照；应优先检查单卡 backend、MTP、batch、KV 和共存服务。用户明确要求保留 embedding，性能测试必须标为“Qwen + embedding 共存”，不能称为整卡独占。

依据：[README.md:19、24、51](../../README.md)、[docs/OPTIMIZATION.md:9–15](../../docs/OPTIMIZATION.md)、[patches/homelab_dense_fused.py:14–28](../../patches/homelab_dense_fused.py)、[mimo-v2.6-flash/README.md:20](../../mimo-v2.6-flash/README.md)。

## 2. 已记录的硬件身份与不可跨越的差异

| 维度 | 本仓库历史实测身份 | 对 MoBase/Qwen 调研的影响 |
|---|---|---|
| GPU | 4 × NVIDIA RTX 6000D，84 GB SKU，350 W/卡；SM120、156 SM、112 MiB L2；历史可用容量约 85651 MiB/卡 | “Pro6000D”名称必须与 `nvidia-smi` 产品名、SM 和容量核对；不能按 RTX PRO 6000 96 GB 预算 |
| 拓扑 | 全部 PCIe Gen5 ×16，同一 NUMA、不同 CPU 根端口 | 四卡 PCIe 通信是候选瓶颈；理论显存带宽不能代表跨卡归约带宽 |
| 理论显存带宽 | 448-bit × 历史频率读数，对应约 1.398 TB/s | 仅理论值，不是有效吞吐，也不能直接推导 tok/s |
| CPU/RAM/SSD | Xeon w5-3433，16C/32T、1 NUMA；标称 64 GB；2 TB NVMe | 部署加载、CPU offload、并发业务占用应另查；不应假设其与 MoBase 当前主机相同 |
| 软件宿主 | 历史 Ubuntu 22.04.5、驱动 580.178.04 OPEN、内核 6.8.0-138；部分旧测试实际为 6.8.0-40 | 历史环境不是当前线上事实；驱动/容器版本变化需重新验收 |
| 网络 | 正式社区口径在服务器 loopback；HTML 经认证 LAN 网关与流式桥接 | 线上 TTFT 需区分服务端排队、推理、网关与网络耗时 |

来源：[docs/HARDWARE.md:3–9](../../docs/HARDWARE.md)、[README.md:19–24](../../README.md)。以上是原始仓库的四卡实测身份，实际客户身份如下。

### 2.1 本次实际单卡硬件与 embedding 保留边界

主任务保存的 `server-inventory.json` 于 2026-10-13 12:06 UTC 观测到 **1 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition，compute capability 12.0（SM120），97887 MiB 总显存，600 W 功率上限，驱动595.71.05**。这是96 GB SKU，既不同于本仓库四张84 GB/350 W 6000D，也不同于本仓库“四张满血96 GB”的参考列。两者共享SM120并不代表设备、SM数量、shape dispatch、权重布局和软件分支相同；本仓库helper要求准确6000D设备名，该用户卡不会满足该guard。[evidence/server-inventory.json:2–3](evidence/server-inventory.json)、[patches/homelab_dense_fused.py:26–28](../../patches/homelab_dense_fused.py)。

初始采样的 Qwen 服务为 vLLM 0.27.1、TP1、FP8 权重与 KV、gpu memory utilization 0.50、max seqs16、max len262144，开启 chunked prefill/prefix cache 与 MTP2；这与原仓库 R30/TP4/MTP3 不同。[evidence/server-inventory.json:7–36、46、188](evidence/server-inventory.json)。硬件库存的93,504 MiB已用显存是多服务共存时的单次快照，不能当作27B模型自身显存，也不能用启动参数的0.50比例推断剩余显存完全可分配。

**embedding 必须保留。** 原库存列出 `Qwen3-Embedding-0.6B` pooling/embed 服务，BF16、gpu memory utilization0.05、max len8192。[evidence/server-inventory.json:107–126](evidence/server-inventory.json)。后续 `controlled-state.json` 记录 Qwen27B 与 embedding 运行、9B和ASR退出；这应标作“27B + embedding 共存、9B/ASR暂停”的实验臂。[evidence/controlled-state.json:5–24](evidence/controlled-state.json)。embedding进程常驻与有无embedding请求是不同维度：v1客户端仅读取Qwen的8000指标；v2客户端也读取embedding的8002前后指标，五组v2共70个窗口中有两次短embedding请求，其余未观测到计数变化；逐项披露见[负载审计](evidence/embedding-overlap-audit.json)。这仍不能证明embedding进程在整个实验期持续运行或整卡独占；后续结果需同时记录该服务状态及实际负载。

## 3. 三条运行时路线及版本锁

| 路线 | 模型/量化及推测配置 | 已锁定运行时 | 补丁与部署边界 |
|---|---|---|---|
| GLM R30 DMA9100 | `local-inference-lab/GLM-5.3-Flash-NVFP4`；保留 NVFP4 权重与 BF16 target head；MoE W4A16（BF16 激活）、小投影 BF16；MTP3 | vLLM `60e72555e755e094e0c0c0ddfd65917514cc2151`，版本 `0.26.1rc0+glm53.r30.vllm60e72555`；B12X `3edbcbce70f491741b82f5eab9c1b30b39447228`；torch 2.13.0、CUDA 13.3；镜像 registry digest 固定 | 四个完整 replacement 文件仅适用于该 R30；不直接覆盖主线或 R31/R32 |
| MiMo V2.6 Flash | `MiMo-V2.6-Flash-RL`；代码存在 FP8 QKV/linear 与 FP4 MoE A16 路径；三层 MTP、probabilistic draft、draft scale 1.25 | vLLM 0.29.0 / `98dff2a81d747d1dba01a47f939f48c3526d4206`；B12X `59d51a36a942d56a9c36265855cdc7856fa7712e`；QKV fix `211e252d0b4f8429f9b15fc52bdfed07782c7f70` | 九个 vLLM replacement 文件；本地镜像 image ID 已记录；公开 Dockerfile 基础镜像使用 tag，未完成干净环境重建验收 |
| DeepSeek V4 Flash R33 | `DeepSeek-V4-Flash-0731`；W4A16 packed route；TP4/DCP1；历史 K3 推测配置 | voipmonitor/vLLM `ae89131442359dc332d9c46009be3c1f8cdee0b4`，版本 `0.26.1rc0+glm53.r33.vllmae891314`；B12X `59d51a36...` | 唯一已确认 host bind 差异为 M4 packed-route patch；官方 0.29.0 是另一个对照臂，不能与 R33 本体版本混同 |

GLM 版本见 [SOURCE_LOCK.json:3–20](../../SOURCE_LOCK.json)；GLM 精度见 [docs/HARDWARE.md:9、13–18](../../docs/HARDWARE.md)。MiMo 版本见 [mimo-v2.6-flash/SOURCE_LOCK.json:2–6](../../mimo-v2.6-flash/SOURCE_LOCK.json)，运行参数见 [mimo-v2.6-flash/run.sh:11–20](../../mimo-v2.6-flash/run.sh)。MiMo 的公开锁未提供完整模型 config/revision/权重分片 hash，因此此处描述的是启动选择与已出现的代码路径，不能声称所有层的精度身份已独立认证。DeepSeek 版本见 [deploy/deepseek-v4/VERSION_LOCK.json:5–27](../../deploy/deepseek-v4/VERSION_LOCK.json)。

GLM 启动器固定 TP4/DCP1、FP8 target KV、16 GiB KV/卡、max model len 262144、max seqs 64、batch8192、capture sizes 至 256、DMA pieces4、NCCL4 channels、350 W。262144 是单请求配置上限，不是“64 个请求都能同时跑满 262K”的容量证明。[deploy/create.sh:26–32](../../deploy/create.sh)、[docs/HARDWARE.md:13–21](../../docs/HARDWARE.md)。这些值含分支特有环境变量，不应直接翻译成官方 vLLM/Qwen 配置而假设含义一致。

## 4. 性能数字实际证明了什么

| 历史结果 | 测量定义/配置 | 可用结论与限制 |
|---|---|---|
| GLM C1 132.09 → 194.64–204.59 tok/s；verifier 55.28 → 78.74 steps/s | R30、TP4/DCP1/MTP3、350 W、batch4096；最终 C1 三轮 | 输出速度受 MTP 接受率影响；verifier 更接近 target forward rate，但也不是 Qwen dense 基线 |
| GLM C8 366.66 → 648.45；C32 1350.57；C64 1934.76 tok/s | 旧 capacity profile：batch4096/KV20、context0、15 s 预热、30 s 持续窗口 | 是总输出速率，不是每用户速率；C32/C64 未测原始起点；不是 DMA9100 的完整并发复测 |
| GLM 冷 nominal32K 8054 → 9134 tok/s | DMA9100、batch8192/KV16、MTP3；12 冷样本 × 3 轮的轮中位数；实际输入约 323xx | 不是恰好 32768 tokens；不是与上述 C1/C8 同配置同次测试；当前 DMA9100 C1 单组为 187.69 tok/s、76.33 verifier |
| GLM HTML 32K prefill 9002.83、decode 192.60 tok/s | C1、128 输出、单样本、客户端扣估计网络延迟 | 不能替代冷32K三轮或持续 decode；HTML decode 随输入长度的变化不能证明长上下文普遍提升生成能力 |
| MiMo C1 145.7 → 231.7；C64 1716.5 → 2979.2 tok/s | 0.6.2、context0、30 s；最终 C1 两轮平均；并发来自此前 MTP3 profile | 最终配置未重测并发；不能将整个 +59%/+73.6% 归因于单个 CuTe 或 FP8 补丁 |
| MiMo 独占 8K/4K prefill 8731、decode 191.9、TTFT 0.938 s | 三轮平均；prefill 为输入 tokens / 首字等待 | 是包含 TTFT 边界的客户端有效输入速率；本仓库无等同口径的 MiMo 优化前 prefill 对照 |
| DeepSeek 官方 44.86 与 R33 历史 240.24 tok/s；TTFT 中位数 3.708/0.784 s | 多轮业务工具任务；客户端 `(completion−1)/(elapsed−firstToken)` 加权估计；官方关闭 speculative | 跨引擎、多组件、任务轨迹不同；没有精确 prefill；不可作为补丁加速比或统计质量等效证据 |

出处：[docs/BENCHMARKS.md:13–43](../../docs/BENCHMARKS.md)、[mimo-v2.6-flash/README.md:5–16](../../mimo-v2.6-flash/README.md)、[mimo-v2.6-flash/results.json:159–199、203–265](../../mimo-v2.6-flash/results.json)、[docs/DEEPSEEK_V4_FLASH.md:15–29](../../docs/DEEPSEEK_V4_FLASH.md)。

另一个硬件参考列使用满血 RTX PRO 6000 96 GB；并发列来自 R28.1/DCP4/LMCache/benchmark0.4.29，冷 prefill 与 C1 来自 R30。该列不能用来隔离“84 GB 与 96 GB 卡的差异”或证明同 profile 同软件 A/B。[docs/BENCHMARKS.md:22–30](../../docs/BENCHMARKS.md)。

“context0”是工具选择的初始上下文档位，不表示 HTTP 请求没有 prompt。例如 GLM C8 原结果实际平均输入 78 tokens；MiMo 平均为 75 tokens。[results/capacity/c8.json:61–73、102](../../results/capacity/c8.json)、[mimo-v2.6-flash/results.json:8–16](../../mimo-v2.6-flash/results.json)。

## 5. 补丁逐项适用性

### 5.1 GLM BF16 小投影与 mHC

`homelab_dense_fused.py` 只在开关开启、prefix 为 `.self_attn.q_b_proj` / `.self_attn.o_proj` / `.self_attn.in_proj_qkvgfab`、BF16 contiguous 权重、准确 `(N,K)` 集合 `{(4096,1536),(4096,4096),(4096,2048),(6288,4096)}`、设备名 `NVIDIA RTX 6000D` 且 SM120/156SM 时准备。warmup 仅 M1–4；执行 guard 限二维、BF16 contiguous、无 bias、M1–4。它重用已安装的 vLLM CuTe `ll_bf16`，并检查私有编译 cache 避免 replay 时编译。[patches/homelab_dense_fused.py:13–54](../../patches/homelab_dense_fused.py)。

对 Qwen 的结论：不能仅把环境变量开成 1 就期待命中；Qwen 的实际 prefix、hidden、TP 分片形状、量化 GEMM 和 batch 必须逐一核对。如果走 FP8/NVFP4/GPTQ/AWQ 等路径，也不能假设会经过这段 BF16 unquantized dispatch。即使形状偶然相同，也需在目标 Qwen 权重上重新验证数值、compile/graph replay、fallback 与端到端收益。本仓库记录的同精度投影增益仅 verifier 约 +2.15% 和 +1.51%，C8 基本持平；无法解释或保证“全模型倍数提升”。[docs/OPTIMIZATION.md:11–15](../../docs/OPTIMIZATION.md)。

`mhc-lifetime-v1.py` 针对 breakable graph 输出弱绑定和 scratch 生命周期；216 次 bitwise 输出对比属于历史数值资格证据。扩大 graph 后 prefill 约 −0.83%，日常保留 capture256。Qwen 的当前模型实现是否使用同一 mHC 模块未知；该文件不能被列为 Qwen prefill 加速方案。[docs/OPTIMIZATION.md:17–19](../../docs/OPTIMIZATION.md)、[patches/mhc-lifetime-v1.py.diff:15–33](../../patches/mhc-lifetime-v1.py.diff)。

### 5.2 MiMo 路线

MiMo CuTe helper 改用 `(4096,2048)` / `(4096,8192)`，同样限定 6000D/SM120/156SM/BF16；M1–4 外在 opaque custom op 中回退 `torch.nn.functional.linear`，注释明确涉及 vLLM 编译图重用时的 shape guard。[mimo-v2.6-flash/patches/model_executor/layers/homelab_mimo_dense.py:13–56](../../mimo-v2.6-flash/patches/model_executor/layers/homelab_mimo_dense.py)。这和 GLM helper 也不是可互换实现。

FP8 QKV tail pad 特判 `(3392,4096)` 与 6000D；B12X FP8 kernel selector 另有支持条件与 fallback。QKV 加载修复处理 MiMo 的 fused FP8 checkpoint 分片与 scales。三层 MTP、draft logits scale 1.25、reasoning/tool parser 都是独立适配面，不是“通用 dense 加速开关”。[mimo-v2.6-flash/patches/model_executor/kernels/linear/scaled_mm/b12x.py:50–54、95–105](../../mimo-v2.6-flash/patches/model_executor/kernels/linear/scaled_mm/b12x.py)、[mimo-v2.6-flash/patches/model_executor/models/mimo_v2_mtp.py:285–332](../../mimo-v2.6-flash/patches/model_executor/models/mimo_v2_mtp.py)、[mimo-v2.6-flash/patches/v1/worker/gpu/spec_decode/speculator.py:141–149](../../mimo-v2.6-flash/patches/v1/worker/gpu/spec_decode/speculator.py)、[mimo-v2.6-flash/run.sh:12、18–20](../../mimo-v2.6-flash/run.sh)。

### 5.3 DeepSeek M4 packed route

补丁要求 W4A16、num tokens=4、routed rows=24、256 experts、hidden4096、intermediate512、top-k6、SiLU、`fp4_e8m0_k32`，以及指定 6000D 设备；满足后提前选择已有 packed route。dense 模型没有这类 expert route，故不能作为 Qwen dense 的加速候选。[patches/deepseek-v4/fused_moe.m4-packed-r33.patch:7–25](../../patches/deepseek-v4/fused_moe.m4-packed-r33.patch)。应用脚本会校验 B12X HEAD 并 `git apply --check`；应保留这种锁定方式，而不是复制此补丁本身。[deploy/deepseek-v4/apply-fused-moe-patch.sh:8–16](../../deploy/deepseek-v4/apply-fused-moe-patch.sh)。

### 5.4 通信、batch、图捕获的可迁移部分

这些是可以提出实验的维度：TP、真实拓扑、NCCL/B12X 路由、batch8192/4096、KV 容量、graph sizes、功率与温度。不能把 GLM 的 `8KB` / DMA4 / NCCL4 channels / 32 MiB L2 persist 一组值视为 Qwen 的既定最优值。仓库自己记录 batch16384 更慢、强制 prefill 大消息走 PyNCCL 约 −10.5%、A/B/A 恢复，说明方向与收益需在目标 workload 上测量。[docs/OPTIMIZATION.md:5–7、21–25](../../docs/OPTIMIZATION.md)。

## 6. 复现可信度与缺口

| 项目 | 已具备 | 仍存在的限制 |
|---|---|---|
| GLM 软件身份 | registry manifest、image config、vLLM/B12X/LMCache commit、四 patch SHA | 固定 R30 fork；完整替换文件与后续主线不兼容；未发布镜像和完整私有实验历史 |
| 模型身份 | GLM config/index/tokenizer SHA | 无完整 safetensors shard hash 与已验证模型 repository revision；MiMo/DeepSeek 也不足以在本地确认完整权重身份 |
| 数值资格 | 真实 GLM TP4 权重、M1–4、M8/32 fallback、graph replay、四 rank tail、mHC 对照脚本 | 是历史资格结果；不是此阶段重跑，不是通用准确率评测，更不是 Qwen 回归验证 |
| benchmark 工具 | 公共 0.6.1 固定 commit+SHA；历史 0.6.2 除 VERSION 字符串外字节相同记录 | 无社区当时私有测试文件 hash；R28.1 是不同 0.4.29；public benchmark 文件需要再获取 |
| 冷 prefill | 12 样本×3 轮、保存 token/TTFT/cache 验证字段、检查无已有负载 | server validation 不可用时只能记未知；inputs/TTFT 不是 isolated GPU prefill kernel 时间 |
| 并发持续 decode | 检查 effective concurrency 和 max running 等于请求并发，排除 capacity limited/loop/error | duration 窗口输出，不是完成整请求时间；不覆盖真实 MoBase 长历史混合负载 |
| 冷 8K/4K | 独立 cache salt、exact token targeting、完整输出验证 | 有限波次吞吐包含 admission、prefill、排队和收尾；C64 历史最多57 active，不能等同64路持续 decode |
| 可部署包装 | GLM 脚本有 SHA/功率/同名容器拒绝；MiMo 构建校验九文件 SHA | 公开 GLM 入口仅静态/mock验收；MiMo clean build 未验收，FROM 使用 mutable tag，COPY 整文件覆盖包路径依赖基础镜像布局 |

出处：[SOURCE_LOCK.json:7–25](../../SOURCE_LOCK.json)、[docs/REPRODUCIBILITY.md:5–7、22–26、36–40](../../docs/REPRODUCIBILITY.md)、[docs/DEPLOYMENT.md:3、30、39、47](../../docs/DEPLOYMENT.md)、[qualification/qualify-dense-integration.py:14–18、25–30、42–65](../../qualification/qualify-dense-integration.py)、[qualification/qualify-fused-allranks.py:18–25、54–60](../../qualification/qualify-fused-allranks.py)、[mimo-v2.6-flash/Dockerfile:1–8](../../mimo-v2.6-flash/Dockerfile)、[mimo-v2.6-flash/README.md:28](../../mimo-v2.6-flash/README.md)。

本次本地检查：从 `patches/` 工作目录运行 `shasum -a 256 -c SHA256SUMS`，四文件均匹配；逐项用 SHA256 检查 MiMo `SOURCE_LOCK.json` 中九文件，均匹配。该检查只确认本地导出文件与锁一致，不证明 GPU 算子正确性、线上镜像身份或构建成功。仓库原包装验证也明确 `gpu_benchmarks_rerun_for_export=false`。[results/publication-validation.json:3–19](../../results/publication-validation.json)。

## 7. 对 Qwen3.6 优化 / Qwen3.8 迁移调研的执行输入

在获得单独授权的诊断阶段，先取得并脱敏记录下列事实，然后决定实验矩阵；本报告不代替远端盘点：

- MoBase 当前准确模型 ID、revision、config、dense/MoE 架构、量化权重与激活、是否多模态、推测方法与接受率。型号字符串“3.6 27B”或“3.8”不足以锁定身份。
- 准确 GPU 产品名/数量/容量/SM、拓扑、功率/温度/频率、同卡其他进程；容器 digest、实际 vLLM commit、CUDA/torch/attention/GEMM backend、已挂补丁。
- 启动参数 TP/DP/PP、KV dtype/分配、max len、batch、max seqs、chunked prefill、prefix cache、graph 模式及服务端 active/waiting；MoBase 网关与业务并发分布。
- 用户所谓“prefill/decode 太慢”的原始输入/输出 token 数、TTFT、TPOT/ITL、完成耗时；冷/热缓存、reasoning tokens、工具回合、图片 token、排队与网络边界。

建议固定同一业务样本和输出约束，至少分三层报告：①本机 loopback 的冷 prefill 与持续 decode；②本机有限 8K/4K 或真实长历史请求；③MoBase 网关业务端到端。每层保留 C1 与实际目标并发的每请求 P50/P95 TTFT/ITL、总吞吐、完成耗时、错误/取消、KV 与显存；不能用单个聚合 tps 替代用户体验。

第一轮候选宜先使用目标 Qwen 已支持的原生运行时配置，在已确认的单卡TP1和embedding保留条件下逐项比较 backend、MTP、batch 与 KV；第二轮才对热点 shape 的算子进行移植资格验证。切换 3.8 应作为独立模型候选，使用同任务质量与性能门槛，并单独记录 tokenizer、chat template、reasoning/tool parser 的差异。只有在这些身份与实测补齐后，才能判断“优化3.6”还是“换3.8”。本仓库现有结果不足以给出速度承诺。

## 8. 基线客户端版本与只读方法审查

### 8.1 旧基线实际使用的 v1

下表审查对象为归档的 **[evidence/benchmark_client_v1.py](evidence/benchmark_client_v1.py)**，对应 `baseline-loopback.jsonl` 与 `baseline-controlled.jsonl` 两组旧基线。当前 [evidence/benchmark_client.py](evidence/benchmark_client.py) 已是 v2，修正情况另见8.2，不能将下表已修复问题视为v2现状。本子任务未修改或执行任一脚本。v1保留客户端首SSE、首文本、首正文、最后文本、usage、finish reason和Prometheus增量，并用服务端success/prefill计数及token增量核对单请求窗口，这些字段能支持低风险的C1基线观察；以下边界影响旧结果的结论。

| 风险/限制 | 代码位置 | 对结果解释的要求 |
|---|---|---|
| A/B请求未匹配 | 62–68：每次新UUID进入system prompt，且UUID作为随机种子重新生成整个user prompt | 原始与controlled两臂的文本和token长度不同，MTP接受率可随内容变；三轮中几%的decode变化不足以独立归因于暂停9B/ASR。后续严格A/B使用同一固定prompt集，并用独立cache盐或可核查的零命中保持冷缓存 |
| 客户端decode仅估计 | 101–107、135–136：首/末文本chunk与`(completion−1)/(last−first)` | MTP可能一chunk输出多个tokens；首文本不必对应第一个模型token，usage也可能包含parser未展示的协议tokens。不能称其为精确token ITL、target forward速度或每token计时；应同时展示服务端decode时间与MTP计数 |
| thinking与正文等待不同 | 100–108、163：reasoning可设置first_text，first_content单独保存；thinking仅单次256tokens | first_text包括思考首字。`first_content=null`且content_chars=0时，只观测到思考，不能称用户正文在该TTFT就出现，也不能当完整业务回答质量或时延 |
| 只测短C1 | 154–163：顺序C1、256输出，long case只1输出、每档3轮 | 不是仓库15s预热+30s持续decode，也不覆盖并发容量、长输出、P95或真实MoBase混合负载。合成words档位不是精确tokens；应以每行usage为准，long约19K，不能写成32K |
| busy/metrics失败时继续 | 37–52、73–79：metrics异常仅记录unavailable，默认busy=0；最多等待30s后照常发请求 | 脚本没有强制无running/waiting的前置条件。后验`single_request_metrics_window`可识别同endpoint部分污染，但不保证整GPU无其他服务负载，尤其embedding共存 |
| 单请求窗口校验有限 | 115–141：聚合所有labels，检查success/prefill/token增量 | 当前模型单endpoint有用；不能用于多模型endpoint直接精确归属，也不能检测在窗口内启动但未完成的其他请求。after延迟0.25s并非同步flush保证；window=false或metrics缺失的样本不用于服务端时段归因 |
| 未严格验收流式完成 | 89–90、110–112、129–141：可在`[DONE]`退出，也可EOF；只记录finish，未断言usage/output/finish/完整DONE | 当前已记录行可人工核对，但未来断流、提前完成或错误SSE可能成为无效样本而继续输出。应将完成性作为采用样本的显式门槛，不能仅看打印了一行JSON |
| 缺少共存负载标签和GPU轨迹 | 123–134：结果只记录请求与Qwen指标，未记录embedding状态/负载、GPU频率功率、测试臂版本 | 报告需通过同时段外部状态证据明确“多服务共存”或“27B+embedding共存”；无法从Qwen8000的running=0推出embedding空闲或GPU独占 |

最初审阅时，`baseline-loopback.jsonl` 全11行与当时已写入的 `baseline-controlled.jsonl` 前9行均记录 `single_request_metrics_window=true`、prefix cache hits=0、running_before=0；这支持所审阅旧样本作为同Qwen endpoint无已完成请求污染的冷C1观测，但不消除上述跨服务负载、文本变化与短输出边界。完整轮次与最终数量以主任务完成后的审核为准。原始样本的thinking行只有reasoning、没有content；不得用该行证明正文首字或业务任务完成。

### 8.2 当前 v2 已修正的内容与保留边界

已逐项只读核对 [evidence/benchmark_client.py](evidence/benchmark_client.py) 的当前代码：

| 项目 | v2实现与代码位置 | 验证/解释边界 |
|---|---|---|
| 固定user corpus | `text_for(words, label)`，记录`corpus_sha256`，harness_version=2和run_tag；60–66、129–134 | 对照臂使用同label时user文本可配对；需核对实际corpus hash一致。system仍加入新的UUID，完整payload并不相同，prompt token数、MTP接受率仍可能略变 |
| 繁忙拒测 | 初始Qwen metrics不可用抛错；等待后若仍running/waiting则抛错；71–81 | 已修复v1等待30s后仍忙就继续的问题。等待中的metrics异常及after指标缺失仍应按结果字段排除，不能从缺失值推断空闲 |
| embedding计数 | loopback读取embedding服务8002的before/after指标；82、120、145、167 | 已补共存负载观测；若计数/指标可用，可检查该时窗内的embedding活动。public-local不读取embedding指标；进程常驻仍需外部状态证据，不能称整卡独占 |
| 流式完成验收字段 | 保存`stream_done`，`complete_fixed_output`要求DONE、finish=length、usage completion等于requested；87、93–95、139–140 | 已补完整输出判据。代码记录布尔值而非对false抛错；报告/聚合必须仅采用`complete_fixed_output=true`且有效single-request window的样本 |
| 更长输出 | 默认paired profile新增三轮约4K档输入、1024输出；164、178–180 | 已补比v1的256输出更长的C1请求；不是15s预热+30s固定窗口的持续decode，label中的4k是档位，实际tokens以usage为准 |

v2沿用SSE文本chunk首末计时的`client_decode_tps_estimate`（146–147），因此多token chunk与reasoning/正文边界仍适用；不能将该字段改称精确ITL或target verifier速度。冷缓存应以每个样本实际prefix cache hits增量为0核查，UUID前缀只是隔离策略，不能独立证明零命中。CPU/GPU clock未固定，温度、功率、频率与时间顺序仍可能造成漂移；配对user corpus和延长输出提高可比性，但不足以把小幅变化严格归因于某个共存服务或配置。所有后续有效loopback结果应标注“单卡TP1、Qwen27B + embedding共存”，并进一步区分embedding有活动、计数不变或指标未知。
