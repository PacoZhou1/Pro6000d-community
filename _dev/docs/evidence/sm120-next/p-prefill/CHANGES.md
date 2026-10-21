# CHANGES（相对 d-draft-vocab 脚本的改动）
基础：`results-retest/arm.py`、`results/scripts/{common,d3_client,d3_compare}.py`、主机 `profile_client.py`（SHA 711dc49e…，未改）。主机工作目录 `/tmp/qwen-sm120-next-p/`（新建，未触碰 `-d` 目录）。

## arm.py（逐项）
- 新增参数：`--attention-backend`（默认 FLASHINFER）、`--extra-arg=`（原样追加到 `vllm serve`）、`--no-prefix-caching`（把 `--enable-prefix-caching` 换成 `--no-enable-prefix-caching`）、`--no-spec`（去掉 `--speculative-config`，即 MTP 关）、`--instr DIR`（只读挂载到 /instr 并设 `PYTHONPATH=/instr`）、`--cargs KIND=参数串`（给某个客户端追加参数）、`--startup-deadline`。
- 新增客户端类型：`prefill`（profile_client 的 prefill 用例，n=6+1 warmup）、`pfx`、`len`、`split`、`gen`、`nsysp`（prefill 的 nsys 捕获）。
- 其余（容器命令、12 GiB `--kv-cache-memory-bytes`、gpu-mem 0.70、max-num-batched-tokens 32768、FP8 KV、监控、清理）与原版相同。
## common.py
- ROOT 改为 `/tmp/qwen-sm120-next-p`，容器前缀 `codex-next-p-`；`snapshot()` 额外忽略 `codex-fp8-*`（E3 容器）。
## 新增文件
- `hc.py`（客户端公共函数：流式 TTFT、/metrics 增量、/tokenize 校准文本长度、top-2 logprobs）、`pfx_client.py`（S1–S4 + 盐隔离冷对照）、`len_client.py`（E2 四长度冷 prefill）、`split_client.py`、`gen_client.py`、`analyze_e1.py`、`analyze_e2.py`、`e3_run.py`（沿用首轮 `experiment_control.py`/`run_api_arm.py` 原件，仅加监控与日志）、`instr/sitecustomize.py`（只记录 `Scheduler._mamba_block_aligned_split` 的入参与返回值，不改返回值；仅用于 `e1-split-*`、`e1-ctrl` 臂）、各 driver 脚本。
## 方法说明 / 偏差
- “冷对照”用请求字段 `cache_salt`（vLLM 0.31 支持，盐进入首块哈希）隔离，而不是 `/reset_prefix_cache`（该端点只在 dev 模式开放，未启用以免改服务参数）。
- 中途按协调者要求裁剪：E2 的 D3 只在 A1 和各非 FlashInfer 后端上跑，A2 不跑；128K 的 n=2、其余 n=3（A1 已按 n=4 跑完，配对只用 index 1–3；128K 为 1–2）。E1 因链式驱动已先行完整执行，未再裁剪。
- 额外加入的诊断臂：`e1-ctrl`（空闲块顺序对照 + 分块日志）、`e1-chunk-a1/b1/a2/b2`（同一批 prompt 上 “分块 vs 单块” 的 ABAB）。
