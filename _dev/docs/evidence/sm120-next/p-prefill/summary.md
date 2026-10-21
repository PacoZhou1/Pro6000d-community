# P 系列 prefill 实验汇总（E1 前缀复用与分块 / E2 注意力后端 / E3 V2 复测）

日期 2026-10-15。主机单卡 RTX PRO 6000（SM120），镜像 vLLM 0.31.0（sha256:a4a4c043…），Qwen3.6-27B-FP8，隔离容器参数同 d-draft-vocab 的 arm.py（gpu-mem 0.70、max-num-batched-tokens 32768、FP8 KV、FLASHINFER、MTP4、12 GiB KV）。脚本和改动见 [CHANGES.md](CHANGES.md)，原始数据在 `runs/`，离线汇总在 `analysis/`（`e1-analysis.md`、`e2-analysis.md`、`e3-api-comparison.json`、`e1-chunk-abab.txt`）。所有臂前后 embedding 容器 ID 4224a2b8…、StartedAt 2026-10-06T09:31:08Z、RestartCount 1 不变，各臂 services-before/after 校验通过，结束时无 codex-* 容器。

## E1：P4 分块规则 + P1 前缀复用

### 1. 16160 + 2741 的切分规则（已用源码和日志确认）
- 启动日志：`Mamba cache mode is set to 'align' ... when prefix caching is enabled`；`Setting attention block size to 1616 tokens to ensure that attention page size is >= mamba page size`（MTP 关时是 **1568**）。1616 就是缓存/快照的基本单位。
- 源码 `Scheduler._mamba_block_aligned_split`（v1/core/sched/scheduler.py）：align 模式下，一次 prefill 的 chunk 不得越过 `last_cache_position = num_tokens − num_tokens % block_size`；MTP/EAGLE 类草稿模型会再减一个 block（`use_eagle_block_drop`）。因此 18,9xx 输入：⌊18,9xx/1616⌋ = 11 个 block → 17,776，再减 1616 = **16,160**（10 个 block），剩余 2,7xx 为第二块。预算 32768 只在更长输入时起作用（先取 ⌊32768/1616⌋×1616 = 32,320，再按上述规则）。
- 用只读观察补丁（`instr/sitecustomize.py`，只打印、不改返回值）实测：MTP4 下 18,834→16160+2674；32,930→30,704+2,226；43,937→32,320+9,696+1,921；17,066→14,544+2,522（⌊17066/1616⌋=10，减 1 = 9 个 block）。MTP 关：整除规则无 eagle 减一，且不再出现 16160。
- 关闭前缀缓存（`--no-enable-prefix-caching`，mamba 模式变 none）：同样的 prompt 变成单块（18,835 与 43,934 各一步）。
- 注意：`vllm:iteration_tokens_total` 直方图不能用来数分块（两块的请求也只记 1 步），已弃用，以 SPLITPROBE 日志为准。

### 2. 分块对冷 TTFT 的影响（≈1%–1.4%，单块更快）
| 比较 | 长度 | 配对中位变化 | 说明 |
|---|---|---:|---|
| profile_client prefill，n=6，no-prefix-cache vs mean(base1, base2) | ~18.9K | −0.915%（−1.118…−0.453） | 但 base1→base2 漂移 +1.298%，不足以单独下结论 |
| 同一批 prompt 的 ABAB（a1,b1,a2,b2，每臂 18.9K 与 44K 各 6 个请求，A/B 用同一批 prompt） | 18.9K | **−1.415%**（−1.482…−1.325） | A1→A2 漂移 −0.11% |
| 同上 | 44K | **−1.062%**（−1.110…−1.004，n=6） | A1→A2 漂移 −0.06% |
结论：把 16160+2741 合成一块，冷 TTFT 约快 1.1%–1.4%，略高于计划预期的 <1%；但关闭前缀缓存会放弃所有复用，不可作为生产手段。

### 3. 前缀复用（MTP4、前缀缓存开；块 1616）
基础数据（e1-base1）：19K 冷 2.33 s；完全重复 hit 16160/18948，TTFT 0.377 s（−84%）；44K 冷 6.39 s；重复 hit 42016/43945，TTFT 0.383 s（−94%）。
**关键机制：每个请求只留下一份 Mamba 快照，位置就是它自己的“最后一个 chunk 边界”（对 MTP 是 ⌊n/块⌋×块 − 1 块）。** 所以：
- S1 完全重复：命中 = 该边界（16160 / 42016），重算 2,788 / 1,929 token。
- S4 位置扫描（种子 19K，测试 = 前 P 个 token 相同 + ~1000 新 token）：P ≤ 17775 时命中全为 0（含 16159/16160/16161、所有 1616 的倍数 ±1），P ≥ 17776 才命中 16160（P=17776、17777、18000、18900、19000 均 16160）。44K 种子：P ≤ 43631 全 0，P ≥ 43632 命中 42016。即命中粒度不是 1616，而是“种子请求自己的末尾边界”，要求 P 至少高出该边界一个 block（因 EAGLE 少匹配一个 block）。
- S2 共享前缀（19K：共享 ~15K，尾 1K/2K/4K）：命中取决于**先前是否有请求在共享区内落下快照**。种子（共享+4K 尾，快照在 16160，已越过 15K 共享区）不能被后面的请求命中；先发 1K 尾的请求会在 12928 落快照，之后 2K/4K 尾命中 12928（TTFT 0.57 / 0.83 s，对比冷 2.09 / 2.37 s）。倒序发送时第一个请求命中 0。44K（共享 40K）：命中 37168（TTFT 0.74–1.29 s 对比冷 6.0–6.5 s），同样依赖顺序。
- S3 多轮：第 2 轮命中 16160（19K）/ 42016（44K）：TTFT 0.42 / 0.44 s 对比冷 2.4 / 6.6 s；第 3 轮仍命中 16160 / 42016（decode 产生的 token 没有快照，第 3 轮 19K 要重算 3.3K token，TTFT 0.50 s）。
- MTP 关（块 1568，无 EAGLE 减一）：S1 重复命中 18816（19K）/ 43904（44K），TTFT **0.056 / 0.072 s**（MTP4 为 0.377 / 0.383 s）；S3 第 2 轮 0.091 / 0.111 s，第 3 轮 0.133 / 0.186 s；S2 命中 14112（19K）/ 39200（44K）。MTP4 因多减一个 block 且 conv 状态更大（页 1616 token），复用明显少，是代价。MTP 开/关 S2 的顺序依赖相同。
- 试验性启动参数（e1-junction：`--enable-mamba-shared-prefix-checkpoint`；e1-modeall：`--mamba-cache-mode all`）：命中完全没变化（12928）。后者日志明确“falling back to 'align' mode”（该模型不支持 all）；前者在 MTP 下未起作用（需要“前缀匹配单位小于 Mamba 块”，此处不满足）。冷 prefill TTFT 与基线无差异（2.278 / 2.286 s vs 2.251 / 2.281 s）。

### 4. 正确性门槛（greedy 256 token，hit 请求 vs 用 cache_salt 隔离的冷请求）
- cold-a 与 cold-b（以及 e1-ctrl 里隔了无关请求再重发的 cold-c）在所有 S2/S3 请求上逐 token 完全相同；e1-base1 与 e1-base2（两个会话）的 6 个冷请求输出也 6/6 相同，即同一计算路径可逐位复现。
- **S1（MTP4 与 MTP 关各 8/8）、S3（MTP4：第 2 轮 4/4 + 44K 第 3 轮 2/2 = 6/8；MTP 关：第 2、3 轮 8/8）命中输出与冷输出完全一致。** MTP4 的 S3 19K 第 3 轮（首个分歧 token 26、159，ctrl 臂 26、122）和多数 S2 命中请求**不一致**：S2 首个分歧位置在 e1-base1 为 18–195（44K 为 0–177），e1-ctrl 为 0–87，MTP 关 19K 为 0–90、44K 为 0–8（0 = 第一个生成 token 就不同，此处 top1−top2 的 logprob 差仅 0.125–0.5，属近并列）；None 表示完全一致，见 analysis/e1-analysis.md。
- 分歧与“chunk 边界不同”一一对应（e1-ctrl 的 SPLITPROBE 日志）：命中请求从 12928 续算，冷请求的 chunk 边界在 14544/16160；边界相同时必然逐位相同（S1、S3 第 2 轮、S2 的 hit=0 且分块相同的请求均 None）。有两类“hit=0 仍不同”的请求（S2 倒序首个请求）是因为它们被调度器按共享前缀边界拆成 12928+3232+…，与冷版 16160+… 不同。
- 噪声底：同样的 6 条冷 19K prompt，仅把前缀缓存关掉（单块 vs 16160+2741），6/6 输出都在第 22–52 个 token 分叉；MTP 关与开对比也在 20–52 分叉。分歧点处 top1−top2 的 logprob 差大多 ≤0.5（多数 ≤0.25），属于近并列 token 的数值抖动。
- **判定**：严格按“逐 token 一致”的门槛，S1 和 S3 第 2 轮（边界对齐的命中）通过；S2、S3 第 3 轮不通过，但分歧完全由 chunk 边界造成的数值差异解释，与冷路径换分块的噪声底同量级，没有发现状态错误的迹象（没有早期大面积崩坏）。是否可接受需要长上下文质量回归（本次未做）。

### 5. GDN 快照显存
`mamba_ssm_dtype = float32`；每层 SSM 状态 48×128×128×4 B = 3,145,728 B，conv 状态 10240×(3+4 个草稿位)×2 B = 143,360 B（MTP 关：3 列 61,440 B），与日志里“padding 0.62% / 0.13%”吻合。MTP4 下每层页 ≈ 3.31 MB，**每份快照 48 层 ≈ 158.9 MB（151.5 MiB）**；MTP 关 ≈ 154 MB。约等于 4,800 token 的全注意力 KV。每个请求只保留一份。

### 局限
S4 只在 19K / 44K 两颗种子上扫；MTP 关的 S4 只做了扫点（命中 18816 / 43904）；44K S2 命中 37168 的具体来源没有逐 chunk 追到；没有做长上下文质量回归。

## E2：注意力后端
- **FlashInfer 变体（nsys 一个 prefill 请求，`runs/e2-nsys-fi`、`analysis/nsys_an.txt`）**：34 次 `flashinfer::BatchPrefillWithPagedKVCacheKernel<causal, CTA_TILE_Q=64, …, DTypeQ=bf16, DTypeKV=float8_e4m3, DTypeO=bf16, accum=float>`，共 326 ms（捕获内全部 kernel 约 2.29 s）。即 Q 与输出为 BF16、KV 为 FP8、在 kernel 内反量化后用 BF16 算、FP32 累加；日志 `FlashInfer resolved query dtypes: prefill=bf16, decode=bf16 ... kv_cache_dtype=float8_e4m3fn`；没有走 FP8 QK。q_scale 回落 1.0 的告警对该路径不起作用。未做 NCU。
- **FLASH_ATTN：启动失败**，错误 `Selected backend FLASH_ATTN is not valid for this configuration. Reason: ['kv_cache_dtype not supported', 'FP8 KV cache requires FA3 on SM90, FA4 with head_size=512 on SM90, or FA4 on SM100']`（SM120 不支持）。按规则未改 KV dtype，跳过。
- TRITON_ATTN 可启动。冷 prefill TTFT 中位（s）：

| 长度（prompt token） | FLASHINFER A1 | TRITON_ATTN | FLASHINFER A2 | Triton 相对 mean(A1,A2)（配对中位） | A1→A2 漂移 |
|---|---:|---:|---:|---:|---:|
| ~18.8K | 2.257 | 2.646 | 2.264 | +16.93% | +0.45% |
| ~43.9K | 6.349 | 8.503 | 6.350 | +33.83% | +0.16% |
| ~63.9K | 10.587 | 15.129 | 10.564 | +43.03% | −0.08% |
| ~127.9K | 28.720 | 47.019 | 28.674 | +63.72% | −0.07% |
n：A1 与 Triton 每长度 4 个（128K 也是 4），A2 为 3（128K 为 2，按协调者要求裁剪）；配对只用共有的 index。所有请求 prefix hit = 0。
- D3 正确性（16 条，1 次 pass，greedy，对 A1）：TRITON_ATTN 仅 2/16 完全一致，其余首个分歧 token 位置 15–216；A2 对 A1 16/16 一致。
- **判定**：在 SM120 + FP8 KV + 此模型上，FlashInfer 是唯一可用且最快的后端；Triton 慢 17%–64%（随长度增长）且输出不同，不采用。FLASH_ATTN 不可用。注意力 kernel 本身 326 ms ≈ 219 TFLOP/s 的结论保持，换后端无法收回；要提高只能走 FlashInfer 内部更优 tile/FP8 QK（需另行质量验证，未做）。
- 限制：D3 按协调者要求只在 A1 与 Triton 上跑（FLASH_ATTN 无法启动）；没有对 Triton 的 kernel 级时间。

## E3：V2 SiLU+quant 跨时段复测
协议同 doc 11：首轮 `experiment_control.py` / `run_api_arm.py` 原件（主机 `qwen-fp8-fusion/`），库 SHA 087b7caa… 校验一致，B 使用 `/instrument` 守卫（日志有 “FP8 fusion candidate launched”）。每臂 12 个 ~19K prefill + 6 个 decode + 5 质量 + 4 开放输出。
| 指标（中位） | A1 | B (V2) | A2 |
|---|---:|---:|---:|
| 长 prefill TTFT（12 请求） | 2.3044 s | 2.2694 s | 2.3020 s |
| decode 速率 | 106.61 tok/s | 106.60 | 106.59 |
- 逐 index 配对 TTFT 减少（B 对 (A1+A2)/2）：**中位 1.443%**（范围 1.394%–1.560%，n=12），首轮为 1.414%；A1→A2 漂移 −0.13%。两次时段结果一致（+0.03 个百分点）。
- 正文 hash：A1 vs A2、A1 vs B 的 prefill 12/12、decode 6/6 相同（reasoning 均空）；质量合成 5/5 ×3 臂、开放输出 4/4 ×3 臂全部通过；prefix hits 为 0；embedding 计数无活动。
- **判定**：V2 的小幅 prefill 改善（约 1.4%）在第二个独立时段复现，decode 不变，输出无差异；可列为候选，但仍是小项（全模型 TTFT −1.4%），且未做 0.45 显存四服务共存与更广质量验证。

## 总体 verdict
1. 16160+2741 是 MTP 的 EAGLE 少缓存一个 block + 1616 token 块对齐的结果；合成单块只带来约 1%–1.4% 冷 TTFT，价值主要是解释 P1。
2. 前缀复用对 S1/S3 很有效（TTFT −84%~−94%，MTP 关时 −97%），对 S2 的“公共前缀+新尾巴”只在共享区内先有请求落过快照时才有效；每请求仅一份快照，没有 block 级粒度，调参（junction、mode all）在 0.31 + MTP4 上无效。MTP4 比 MTP 关复用差很多。
3. 非边界对齐的命中会因 chunk 边界不同而产生数值分歧（噪声量级），逐 token 严格门槛不过；需质量回归后再谈采用。
4. 注意力后端无替代，FlashInfer 保留。
5. V2 复测通过（−1.44%）。
