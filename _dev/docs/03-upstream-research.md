# Qwen 27B 在 SM120 上的上游调研与版本建议

查询日期：2026-10-13。来源为 Qwen/NVIDIA 模型仓库、vLLM/SGLang/FlashInfer 源码、正式 release 与维护者 PR；搜索结果摘要只用于找到入口，版本与合并状态均进一步通过 GitHub/Hugging Face API、固定 tag 原始源码核查。本文包含上游研究与临时容器实测；测试期间按用户授权暂停非embedding模型。新版本已做合成性能测试，尚未完成MoBase业务质量验收或替换生产部署。

## 1. 结论与当前机器边界

**建议把官方 vLLM `v0.31.0`，commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`，作为此次 Qwen 算子优化的源码基线。先保持线上 Qwen3.6-27B block-FP8 权重不变，对比版本与后端，再比较 Qwen3.8-27B-FP8 / NVIDIA NVFP4。**

线上实查为 **1 × RTX PRO 6000 Blackwell Workstation Edition，97,887 MiB，600W，compute capability 12.0，driver 595.71.05**。本次初始服务为 vLLM 0.27.1、torch 2.13.0+cu130、FlashInfer 0.6.16.post3，TP1、FP8 128×128 block weights、FP8 KV、MTP `num_speculative_tokens=2`，并启用 chunked prefill / prefix caching、max-seqs 16、max-len 262144、gpu-memory-utilization 0.50。来源见本次本地证据 [server-inventory.json](evidence/server-inventory.json)、[runtime-diagnostics.json](evidence/runtime-diagnostics.json)。同卡有其他常驻服务且 ASR 启动反复因空闲显存不足失败；共享状态下测到的是在线体验，不能作为独占 GPU 的算子性能上限。

仓库 README 中 **4 × RTX 6000D，84GB/卡，350W/卡**是此前 GLM/MiMo/DeepSeek 项目的硬件，不是本次线上机器。GLM 的约 200 tok/s、MiMo 的 231.7 tok/s 不能当作 Qwen27B 的目标保证：它们的模型结构、卡数、MTP、量化与测量脚本都不同。MiMo 本地锁定的 vLLM 0.29.0 commit `98dff2a81d747d1dba01a47f939f48c3526d4206`已与官方 tag 一致核实，可作为历史补丁参考，但不直接搬到 Qwen。

## 2. Qwen3.8 是可核实的公开权重

截至查询日，官方 `Qwen` 组织公开、非 gated 的权重与 revision 如下。`revision`锁定的是 HF 仓库，不等于本地分片已通过完整 SHA256 校验；下载后还应生成分片 hash manifest。

| 模型 | HF revision | 官方结构 / 格式 |
| --- | --- | --- |
| [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | BF16，dense FFN + GDN/full attention |
| [Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) | `e89b16ebf1988b3d6befa7de50abc2d76f26eb09` | dynamic E4M3 block FP8，128×128 weight blocks |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | BF16，同一推理架构 |
| [Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | dynamic E4M3 block FP8，128×128 weight blocks |
| [NVIDIA Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) | `0893e1606ff3d5f97a441f405d5fc541a6bdf404` | NVIDIA 混合量化导出 |
| [NVIDIA Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | `482ca0f3832238542f8f5295dde86b5f22711d80` | MLP/lm_head NVFP4；attention/GDN projections FP8 |

3.6/3.8 的 27B 版均为 `Qwen3_5ForConditionalGeneration` / `qwen3_5`，文本为 `qwen3_5_text`，不是 MoE。64 层里 48 层 Gated DeltaNet、16 层 full attention；hidden 5120、FFN intermediate 17408；GDN QK/V heads 为 16/48、head_dim 128；full attention Q/KV heads 为 24/4、head_dim 256；内置 MTP head 的 `mtp_num_hidden_layers=1`。这一个 MTP 模块可以递归 draft 多步，不能把 `num_speculative_tokens=3`称为模型里有三层独立 MTP。

HF safetensors 元数据把 BF16 VLM 全部张量合计为约 27.78B。NVIDIA NVFP4 页面的约 18B 存储 tensor count 受打包方式影响，**不表示架构变成 18B**。其模型卡明确说明混合量化、Local Hessian 校准，推理验证硬件为 GB300；不能把“Blackwell supported”自动等同于每个算子在 SM120 都有同一快路径。[NVIDIA 3.8 模型卡](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)

Qwen3.8 模型卡报告比 3.6 更好的 coding/agent 指标，因此是值得进行业务质量评测的替换候选；这不直接证明推理更快。两者主要算子形状相同，框架和精度保持一致时预期相近的原始计算代价，但 MTP 接受率、实际思考长度、工具调用成功率可能改变整项任务耗时。需在真实业务集测 token 数、完成率与耗时。[Qwen3.8 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)

## 3. 要拉哪个 vLLM tag / commit

| 用途 | 精确版本 | 查询日状态与建议 |
| --- | --- | --- |
| 线上基线 | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` | 保留原镜像 digest / 原权重和配置供 A/B |
| 历史 MiMo 对照 | `v0.29.0` / `98dff2a81d747d1dba01a47f939f48c3526d4206` | 已核实，仍无 SM12x GDN prefill 自动分支 |
| 中间变化定位 | `v0.30.0` / `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | 已含 SM12x GDN prefill 与 NVFP4 W4A4 优先 |
| **本次首选开发基线** | **`v0.31.0` / `db9527a46873454610df6dbedf79a36d6bf1a7f6`** | **2026-10-05 发布的最新正式版；先原版复现，再开发补丁** |
| 备用最新 main 快照 | `d9a543c500e9014045d7ae769df0ca7782e31870` | 2026-10-13 查询所得；滚动 main，仅遇到明确已修 bug 时评估 |

版本直接核查来源：[0.31.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)、[0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)、[tag31 API](https://api.github.com/repos/vllm-project/vllm/git/ref/tags/v0.31.0)、[tag30 API](https://api.github.com/repos/vllm-project/vllm/git/ref/tags/v0.30.0)、[tag29 API](https://api.github.com/repos/vllm-project/vllm/git/ref/tags/v0.29.0)、[tag27.1 API](https://api.github.com/repos/vllm-project/vllm/git/ref/tags/v0.27.1)、[main 固定提交](https://github.com/vllm-project/vllm/commit/d9a543c500e9014045d7ae769df0ca7782e31870)。PR merge 日期早于 release 发布不保证它在该 release：本文又读取 tag29/tag30 的 GDN 源码，确认 0.29 无而 0.30 有 SM12x 分支。

可在独立开发目录拉源码，并明确核查 HEAD（以下是建议命令，本文未在服务器执行）：

```bash
git clone --branch v0.31.0 --single-branch https://github.com/vllm-project/vllm.git vllm-qwen-sm120
git -C vllm-qwen-sm120 rev-parse HEAD
# 预期 db9527a46873454610df6dbedf79a36d6bf1a7f6
git -C vllm-qwen-sm120 switch -c codex/qwen27b-sm120
```

使用官方 `vllm/vllm-openai:v0.31.0`做第一轮框架 A/B 更容易复现；另记录平台 manifest digest、image id、CUDA runtime、torch/FlashInfer/CUTLASS DSL 版本。源码 build 应与官方依赖锁一致，不能只升级 Python `vllm`包保留旧 CUDA extensions。

v0.31.0 的源码依赖锁为 torch 2.13.0、FlashInfer python/cubin 0.7.0.post1、CUTLASS DSL `[cu13]`4.7.1；DeepGEMM 是 `vllm-project/DeepGEMM` fork，而不是把 DeepSeek 的上游任意 main 装进去。[固定 requirements](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/requirements/cuda.txt)、[DeepGEMM pin](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/cmake/external_projects/deepgemm.cmake)。FlashInfer `v0.7.0.post1`对应 commit `946200de1ae94fc93fdd0926f0a13afd1fa7f0f1`（annotated tag 已解引用）；其 jit-cache 修复由 [vLLM #59323](https://github.com/vllm-project/vllm/pull/59323)纳入。查询日已有 FlashInfer 0.7.1，但不先单独换依赖，以减少变量。

## 4. 上游已经有的实际加速路径

### 4.1 Prefill：先兑现 SM120 GDN fast path

[vLLM #55715](https://github.com/vllm-project/vllm/pull/55715)，merge `f6326f53bda46898a331c2d24500332c285d9a2b`，打开此前被平台条件挡住的 FlashInfer SM12x GDN prefill。条件为 CUDA runtime ≥13、head_k_dim=128；本次 Qwen27B 几何符合。底层 [FlashInfer #3479](https://github.com/flashinfer-ai/flashinfer/pull/3479)，merge `2ec0c9b0a27c5028c565afc7220d2b8dd90078f9`，已在 2026-06-17加入 SM120 delta-rule prefill。

PR 的 Qwen3.8 几何 kernel microbenchmark 在 SM121 报告 3.83–4.52×；RTX PRO 6000 Max-Q 的含 prefix-cache/MTP 端到端样例 TTFT 下降约 5.0–6.6%。**前者是 GDN 算子时间比，后者是特定场景请求指标，不能称为模型整体 prefill 提升4倍，更不能承诺600W工作站卡同一百分比。**原始 FlashInfer PR 的小 head/多卡分片 geometry 存在比 FLA 慢的情况，也支持必须用本机 TP1 形状验证。

v0.31固定源码可用 `--additional-config '{"gdn_prefill_backend":"flashinfer"}'`与 `"triton"`作单变量 A/B，读取日志确认 `Using FlashInfer GDN prefill kernel`而不是仅凭参数认定生效。vLLM 内置另一条 `cutedsl` GDN prefill 只面向 SM10x，和 FlashInfer 自带 SM120 CuTeDSL不是同一后端。[固定 dispatch 源码](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L93)

0.31 release 中 [#53864](https://github.com/vllm-project/vllm/pull/53864)修复的是内置 CuteDSL BF16 KKT inversion divergence；不能把它说成当前 SM120默认 FlashInfer路径的直接修复。它提醒任何 GDN fused 改动都要同时测 output 与 final recurrent state。

### 4.2 当前 FP8：dense GEMM比 MoE route 更 relevant

本模型的优先投影如下；这是 TP1 `(N,K)`，M随实际调度 token 数变化。vLLM可能把 in_proj 的逻辑拆分合并为不同调用，profile应从运行时记录实际 shapes。

| 投影 | `(N,K)` | 层数 |
| --- | --- | --- |
| MLP `gate_up_proj` | `(34816,5120)` | 64 |
| MLP `down_proj` | `(5120,17408)` | 64 |
| GDN `in_proj_qkvz` | `(16384,5120)` | 48 |
| GDN `out_proj` | `(5120,6144)` | 48 |
| full attention `qkv_proj` | `(14336,5120)` | 16 |
| full attention `o_proj` | `(5120,6144)` | 16 |

上游 [SGLang #33629](https://github.com/sgl-project/sglang/issues/33629)给出了这组 Qwen3.6-27B-FP8 SM120 几何：weight E4M3 /128×128、dynamic activation scales；MLP两项应首先测。该 issue 虽显示 Closed/completed，但最后关闭说明是 bot 因 inactivity 自动关闭，不代表优化已合并。以 PR `merged_at`及实际源码判断。

v0.31可以对比 `--linear-backend cutlass`、`marlin`、`b12x`与 auto；默认 block FP8候选优先级含 DeepGEMM/CUTLASS/B12X/Marlin/Triton，实际选择受平台、量化键、模型默认禁用规则和依赖可用性限制。不能仅看候选列表声称本机已用了 DeepGEMM。[kernel registry](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/kernels/linear/__init__.py#L457)

Marlin FP8在本版本明确处理 `kFp8Static128BlockSym`权重：它是 FP8 weight-only / A16路径，并非与 CUTLASS W8A8 完全同一 arithmetic。可以作为小M decode候选，却应同时测 prefill和质量；不要把移除旧 `VLLM_TEST_FORCE_FP8_MARLIN`环境变量误当成移除 Marlin 支持。[Marlin 实现](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/kernels/linear/scaled_mm/marlin.py)

官方 v0.31已经集成可选 B12X block-FP8 backend，要求 SM12x、activation group `(1,128)` / weight `(128,128)`、N/K为128倍数；安装入口 `vllm[b12x]`，该tag pin `b12x==1.3.0`。这比把历史MiMo完整 linear.py 覆盖到Qwen更适合做独立比较。[B12X适用检查](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/kernels/linear/scaled_mm/b12x.py#L53)、[extra依赖](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/setup.py#L1550)。仓库之前的 `voipmonitor/b12x`是 `local-inference-lab/b12x`社区fork，应单独锁SHA；“vLLM有B12X集成”不代表那些历史fork补丁全部成为官方代码。

### 4.3 NVFP4：把减少权重带宽作为单独实验

[vLLM #55170](https://github.com/vllm-project/vllm/pull/55170)，merge `13cf9e05c1eda0bfe5cbfb9344343ca2737d0723`，在SM120/121为可用W4A4 checkpoint优先选择W4A4而非W4A16。PR在RTX PRO 6000 Max-Q、Qwen3.8-27B-NVFP4、32K cached prefix +2K input/256 output的样例报告output throughput 74.81→92.63 tok/s、TTFT 1455.2→926.1 ms；作者明确说明收益集中在prefill，BS1 decode-only W4A4可能输weight-only几个百分点。不能称为decode算子固定+23.8%。

本次当前为FP8，以上不是无需换权重就能获得的收益。若质量通过，应单独测NVIDIA `482ca0f...`混合checkpoint：MLP/lm_head NVFP4，attention/GDN是FP8，强制全局linear backend可能让不同精度层走不同fallback；逐层检查。v0.31提供 `--kernel-config`中的`linear_backend_per_quant`以分别指定`fp8`与`nvfp4`。

优先测试NVFP4 `auto/flashinfer_cutlass`与`marlin`；需要额外依赖时再测`b12x/flashinfer_b12x`。`flashinfer_cutedsl`名字不等于SM120 W4A4 fast path；固定tag的FlashInfer CuTeDSL W4A4 dense实现主要面向SM10x，SM120上应检查CUTLASS/B12X或W4A16选择。[NVFP4注册顺序](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/kernels/linear/__init__.py#L551)

NVIDIA文档区分SM100与SM120：SM120 shared memory / block、warp occupancy更小，不能把SM100的tile/cluster/TMEM算法原样套用。[Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)

### 4.4 MTP：已有融合，但两步接受率仍须记录

线上日志已显示MTP accepted/drafted counters与mean acceptance length，说明并非只加载配置却不draft。后续比较必须保留每次试验的metric增量而不是长时间累积counter。

v0.29已纳入 [#52539](https://github.com/vllm-project/vllm/pull/52539)，merge `cdb8545a91be16f2b80234e0801458ad2a18cf1b`，fused GDN MTP支持HV/H=`1/2/3/4/8`，包含27B的3。v0.30还纳入 [#55369](https://github.com/vllm-project/vllm/pull/55369)，merge `b339d75a410e4e889049dff59042f385909bad67`，从multimodal wrapper的`text_config`正确读取MTP层数。v0.31 GDN source默认`VLLM_GDN_DECODE_KERNEL=cuda`并在不支持时回退，允许`triton`控制组。

建议先固定gamma2复测新旧版本，再在胜出的stack测试off/1/2/3/4；并发高时draft开销可能超过收益。报告接受率=`accepted_tokens/draft_tokens`与mean acceptance length=`1+accepted_tokens/draft_steps`，二者分母不同。SGLang的`num_draft_tokens`可能包含bonus/树节点，不能只比较数字“3”。[vLLM Qwen3.8 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)也要求从metric确认MTP有效。

DFlash2 / DSpark是额外checkpoint与不同draft算法，不能直接视作内置MTP无风险升级。[vLLM #54928](https://github.com/vllm-project/vllm/issues/54928)有Qwen3.8 SM120 greedy verifier差异调查，先通过业务质量门槛再测；无需在第一轮版本A/B加入第三种draft变量。

## 5. SGLang值得做对照，但当前优化PR尚不能当作release能力

查询日最新正式SGLang为 [v0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)，2026-10-02发布；annotated tag解引用commit **`e00930c5489053f26d86b179cee0d087f846acbb`**。它可以作为第二推理引擎对照，而非先替换在线API。

[官方Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)对RTX PRO6000有单GPU的BF16/FP8/NVFP4 + speculative部署与质量验证；cookbook固定验证版本是0.5.19，网页里的运行域不能自动外推到0.5.21。SM120优先`--attention-backend flashinfer`，`trtllm_mha`为SM100路径；GDN recurrent state pool与KV池分配需要`--mamba-full-memory-ratio`，否则显存很大也可能限制并发。其BF16 state可省显存，但应当是另一次质量测试，不能在baseline顺便改掉模型声明的FP32 state。

两项直接相关、截至查询日**未合并**的PR：

| PR | 锁定head SHA | 可参考内容与边界 |
| --- | --- | --- |
| [#39978 小M block-FP8 decode](https://github.com/sgl-project/sglang/pull/39978) | `19789d4b2c8c15a9619a1820c708e9a663afa455` | Qwen27B形状、streaming weight TMA、warp MMA、FP32 split-K；层加权isolated GEMM约-6.60–8.88%；单请求serving却-1.22%，其他case约-1.16%到+5.84%；完整model eval pending，不能作为可靠整模加速承诺 |
| [#34731 swapAB crossover](https://github.com/sgl-project/sglang/pull/34731) | `7db8f397f4e093a88d678a855d9afcc063c09445` | 主要改变M72–256，短128-token prompt的batch benchmark约+3.4–9.4%；当前maxseq16/MTP2通常不在这个M区间，优先级较低 |

PR39978目前还明确记录full CUTLASS graph racecheck在baseline和candidate都出现shared-memory write/write hazards；它有算术smoke/unit证据，但模型accuracy尚未完成。研究其策略可以，直接cherry-pick到生产或宣称已在官方release可用不成立。

当前主任务仍是vLLM：SGLang这组结果说明dense FP8小M是可优化方向，也说明kernel microbenchmark快8%不保证用户tps变快8%。

## 6. main更新的条件性使用

首轮坚持v0.31.0。如果出现hybrid prefix缓存边界复现，可考虑同tag加 [#60533](https://github.com/vllm-project/vllm/pull/60533)，merge **`d8fa463e2ba751759a3534dd3ce11af92f1eed83`**，2026-10-08，修正Mamba prompt checkpoints / EAGLE proof hit边界一致性；这是正确性条件，不是必然tps提升。

[MRV2 state-copy #60210](https://github.com/vllm-project/vllm/pull/60210)，merge `8a26869bcad1ae4ab3069a7d2aece86c17c1c695`，主要针对MRV2+PP+MTP异步slot混用。本次TP1单卡/无PP不具备它的主触发条件，不能用GB200 PP2的+28%结果支持换main。[static FP8 fusion #61108](https://github.com/vllm-project/vllm/pull/61108)，merge `12e6a585ae020cdcef9ba61aae45eec5ba29591d`，针对torch main/2.15 RC pattern matching，本次torch2.13也不应以它作为升级理由。

## 7. 可执行的最小实验矩阵

按下列顺序，每一步保留上一轮胜者，避免一口气变更模型、precision、draft、runner和cache：

1. **共享线上baseline**：当前0.27.1/FP8/MTP2，只记录c1真实体验，GPU utilization/power/memory、TTFT、output tokens、TPOT、接受率、其他服务状态；不得标为独占benchmark。
2. **维护窗口受控A/B**：同一Qwen3.6 FP8本地分片、同样采样/参数，0.27.1→0.31.0。embedding仍常驻，需核对模板实际渲染和prompt token数；cold prefix与warm prefix分开，固定gamma2，保留相同context限制与maxseq16。
3. **prefill后端**：0.31的FlashInfer GDN与Triton/FLA对照，ISL2048/8192/32768，OSL256/1024；一条8K/1K固定请求便于与其他模型比较。
4. **FP8线性后端**：auto/cutlass/marlin，可用后再b12x；M小批decode与大M prefill分别profile。全部后端保留相同weights，先记录实际kernel class。
5. **MTP步数**：off/1/2/3/4，c1/4/8/16；记录draft开销、接受率、平均接受长度。每个档位有相同真实代码/工具/中文业务prompt。
6. **Qwen3.8替换候选**：相同0.31/backend，先官方3.8-FP8，再NVIDIA3.8-NVFP4；先业务质量过关，再调gamma与context容量。
7. **条件性引擎对照**：SGLang0.5.21，使用相同target权重、同等缓存冷热、输出长度/采样、FP32 state、相同gamma语义；结果领先且工具/多模态质量通过才进入切换评估。

后续用户已授权测试时停止其他模型服务，**embedding始终保留**。因此受控A/B应注明“27B +常驻embedding”，而非完全独占GPU；升级前后保留相同embedding预算与流量条件。原GPU95GB占用时，另开一个27B试验容器大概率不能正确启动；显存fit与kernel性能应分开确认。具体维护操作与测量结果由主任务记录，本文不代替线上执行证据。

计时口径：

- TTFT包含排队、tokenize/template、prefill及首token；`input_tokens / TTFT`只能叫端到端输入处理速率代理，不能叫纯GPU prefill throughput。prefix命中的token不应被当成真正计算的输入。
- 单流decode tok/s建议用`(output_tokens-1)/(last_token_time-first_token_time)`，同时保留TPOT分布；`output_tokens / total_request_time`则是包含TTFT的端到端输出速率。
- 并发总output throughput=`sum(output_tokens)/benchmark_wall_time`，不是每用户速度；报告c1/c4/c8/c16、P50/P95 TTFT/TPOT、完成率、实际输出长度。
- MTP draft tokens不算用户生成tokens；以最终accepted输出和API usage计数。thinking/reasoning tokens与answer tokens应注明是否都计入。
- warmup/JIT/graph capture不计入稳态；每个case至少3轮、ABBA顺序、同样时钟/功率条件，保留raw JSON。算子比较用CUDA graph + CUPTI计时与cold-L2条件，保存geometry/scale dtype，不只报最快一次。

### 7.1 v0.31 参数兼容与B12X attention补充核查

本次预量化FP8权重可继续使用`--quantization fp8`：`Fp8Config.from_config`从checkpoint的`quant_method`判定serialized FP8，保留128×128 block配置。0.31移除的是旧`fp8`方法对未量化BF16权重的online quantization；那类用途才使用`fp8_per_tensor`，本次不要改成后者。[固定FP8 config实现](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/quantization/fp8.py#L158)

v0.30 release notes里的`B12X_ATTN`名称在v0.31固定registry中实际为 **`B12X`**，建议运行`--attention-backend B12X`并核查镜像`--help`。它是regular/GQA decoder paged attention，不是GLM/DS sparse MLA后端。支持SM120/121、BF16 Q、head64/128/192/256、KV `auto/bfloat16/fp8/fp8_e4m3`，page64/128；Qwen27B的head256符合。B12X实现要求实际`--block-size`为64或128；若框架自动改选，也要记录解析后的block size与KV容量，并用相同block size的FlashInfer臂对照，避免把KV分页变化归因于attention kernel。它宣告`UNIFORM_BATCH`图能力，并为`1+num_speculative_tokens`的uniform verifier预建plans，MTP2对应Q长度3。prefill/extend保持eager并不妨碍uniform decode图。[固定registry](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/registry.py#L120)、[B12X capabilities](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/b12x.py#L199)、[block-size检查](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/b12x.py#L393)、[verifier plans](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/b12x.py#L541)。依赖仍需可用`b12x==1.3.0`；源码支持范围不等于已完成本机测量。

v0.31 **FlashInfer也有SM12x XQA decode + uniform MTP图的候选路径**：head_dim范围16–256/16倍数、Qheads可被KVheads整除（本次24/4符合）、无DCP。`supports_trtllm_attention`另要求cubins已可用（有`flashinfer-cubin`包就直接通过，否则检查NVIDIA artifactory可达）、未启用batch-invariant、未显式关闭`attention_config.use_trtllm_attention`；否则图能力会退到`UNIFORM_SINGLE_TOKEN_DECODE`，MTP下仍可能降为PIECEWISE。启动时捕获FULL图仍不等于已证明每次请求实际按FULL replay。[FlashInfer graph gate](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/flashinfer.py#L971)、[artifactory/平台gate](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/utils/flashinfer.py#L592)。

优先实测：0.31/FP8/MTP2/FlashInfer默认 → 相同block size的B12X与FlashInfer对照 → 胜者去掉MTP。每轮记录实际block size、KV容量、decode kernel、GDN backend、CUDA graph mode，避免把KV分页、GDN prefill、dense linear和GQA attention的变化混作一个改动。

### 7.2 本机首轮v0.31运行验证

本机同Qwen3.6-FP8/MTP2的首轮0.31启动日志已确认 **FlashInfer GDN prefill**、FlashInfer attention `decode_backend=xqa`，JIT构建了SM120/head256/KV E4M3/page64的XQA verifier模块（spec Q长度3/2及普通Q长度1），并捕获PIECEWISE与FULL decode图；GDN decode日志为`cuda`。[运行时](evidence/runtime-v031.json)、[图与JIT日志](evidence/graph-v031.json)。这些是后端选择和图捕获证据，尚非每个请求实际kernel耗时或graph replay的profile。

[0.31固定语料](evidence/paired-v031.jsonl)与[0.27固定语料](evidence/paired-v027.jsonl)的14个label均有相同user corpus hash，全部请求完整且主模型指标窗口匹配；两者各先做1次预热，各正式档仅n=3，system UUID及prompt token数略有差异。短约0.7K/256输出的TTFT中位0.139→0.085秒，但decode估计98.37→92.63 tok/s、总时2.736→2.838秒；约4.8K/1024输出的decode估计90.73→94.23 tok/s、总时11.902→11.361秒；约19K/1输出TTFT 2.378→2.179秒。可说**此样本的长输入首字与长输出略快，短档不是全面加速**；不能把变化单独归因于GDN或FULL图，也不能据此判断MoBase任务更快。旧版首轮采样时有镜像下载；下载结束后的旧版回测已完成，4.8K/1024为89.71 tok/s、19K TTFT为2.403秒，详见[基线报告](01-server-baseline.md)。该复测支持长输入/长输出方向，但仍非充分随机交叉试验。embedding全程保留，不过0.31的`long_prefill_r3`窗口与一次170-token embedding请求重叠，旧版回测的`sustained_4k_decode_r1`与一次57-token请求重叠；对应中位数分别来自未重叠样本，仍须披露并在严格同负载复测时补足样本。

0.31镜像同时更新了FlashInfer 0.6.16.post3→0.7.0.post1、Transformers 5.15.0→5.17.0、CUTLASS DSL 4.6.0→4.7.1（torch仍为2.13.0+cu130），且启动时记录了不同的KV容量/分页规划；所以这是**版本及依赖组合**对照，不是单一算子A/B。[旧版回测运行日志](evidence/runtime-v027-repeat.json)和[0.31运行日志](evidence/runtime-v031.json)均提示checkpoint缺q scaling factor、KV scaling取1.0、q/prob scaling未校准；这不是仅凭日志可认定的新版本精度回归，也不能凭[合成API冒烟3/3](evidence/smoke-v031.jsonl)排除长上下文、思考、工具或多模态质量风险。上线前应核对实际量化scales语义，做同任务质量/数值回归；不要未验证就改填scale。

## 8. 如果选择换模型

**先考虑Qwen3.8-27B**：架构/工具模板与现有27B最接近，可以把升级质量与算子收益分开验证；同结构意味着不应期待单靠“3.8”数字就有大幅原始decode提升。

若业务允许用更少active参数换速度，可另测 [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)：官方为35B总/3B active、40层、256专家/top8+shared。它是MoE，不是27B dense的drop-in算子实验。预计工作集与FLOPs更小是结构推断，实际速度和质量仍须单卡实测。其MoE/shared-expert fusion成果不能用于声称27B也能获得同一收益。

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)是另一架构：125B主体/6B active，加51B n-gram embedding及4B MTP，GDN+QSA+gated residual；“180B/6B active”不表示只要6B显存。单卡96GB可能需要NVFP4、PLE offload等独立内存与运行域验证，工程复杂度明显更高，排在当前27B版本/精度A/B之后。

## 9. 开发优先级

最有根据的顺序为：**修正共享GPU运维与baseline口径 → 官方0.31兑现GDN快路径 → FP8 dense后端选择与MTP gamma → profile确认剩余热点 → 小M MLP GEMM/fusion → 有质量门槛的NVFP4/3.8切换**。

如果0.31之后profile仍以MLP投影为主，值得围绕`gate_up/down`的M=1/3/4/8/16/32/48/64优化weight读取与split-K、融合activation量化、SwiGLU/残差；若热点是GDN state/metadata/launch，优先减少状态copy与重复kernel launch。GLM/MiMo的稀疏MLA/MoE packed-route优化不能原样套入dense27B；只保留可验证的技术思路。

验收标准应是本机同一工作负载的TTFT/TPOT/总吞吐有稳定改善、质量/工具/长上下文回归通过，并保留旧镜像快速回滚。当前公开证据没有支撑“这台单卡必达200/300 tok/s”的固定承诺。

## 本机模型来源与下载约定

用户指定后续模型下载统一走 **ModelScope**，沿用 `/home/llm/models`。本次检查发现已有 `/home/llm/models/Qwen3.8-27B-FP8`，未重复下载或转换权重。

已通过 [ModelScope 官方 Qwen3.8-27B-FP8](https://modelscope.cn/models/Qwen/Qwen3.8-27B-FP8) 的 repository API 对本地文件逐一计算 SHA256：66个 safetensors 权重分片，以及存在的配置、tokenizer、模板等共81个文件全部与官方一致；82个仓库文件中仅缺 `.gitattributes`，不影响推理。66个权重条目的`Revision`均为 `f7115dacf0c93b80b010ab9244a35d6ef3590f2a`；部分元数据条目显示其他文件修订，不能把单个条目revision称为整个仓库的快照revision。逐文件结果见 [qwen38-modelscope-verification.json](evidence/qwen38-modelscope-verification.json)。ModelScope文件revision与本文列出的Hugging Face revision属于不同仓库，不能互换。

若后续引入NVFP4，先在ModelScope确认发布者、模型配置与分片hash；尚未确认的镜像不自动视为NVIDIA官方发布。本轮下载的是vLLM运行镜像，不是模型；已核对其amd64 manifest与官方相同，见 [image-acquisition.json](evidence/image-acquisition.json)。
