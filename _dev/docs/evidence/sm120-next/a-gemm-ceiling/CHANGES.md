# CHANGES (执行任务 A 时对原准备文件的改动记录)

- 新增 `common_host.sh`、`build_host.sh`：run_a.sh 的非 root 改编（不能 sudo、不锁频）。差异：
  - 工作目录 /tmp/qwen-sm120-next-a/{work,results}（原为 ~/.local/share/qwen-sm120-next）；
  - 容器名前缀 codex-next-a-（原 codex-sm120-next-<label>-）；
  - 不做 nvidia-smi -lgc（LOCK_CLOCKS=0），仅记录 NVML/nvidia-smi 时钟；
  - 服务快照仍复用 $OLD/work/gemm-epilogue/verify_service_state.py（只读）。
- `run_a.sh`、`mma_rate_probe.cu`、`scaled_mm_ceiling.py` 原样（以下若有修改会追加）。
- `mma_rate_probe.cu`（第一次构建失败：镜像内无 nvml.h，也无 libnvidia-ml stub，见 results/build-v0-nvmlh-failed.log）：去掉 `#include <nvml.h>`，改为内联最小 NVML ABI 声明并在运行时 `dlopen("libnvidia-ml.so.1")`（`nvmlInit_v2` / `nvmlDeviceGetHandleByPciBusId_v2` / `nvmlDeviceGetClockInfo` / `nvmlShutdown`）；构建命令去掉 `-lnvidia-ml` 与 `-L stubs`，加 `-ldl`。
- `build_host.sh`（第二次构建失败：`-arch=sm_120a` 在 nvcc 13.0.88 下 PTX 以 compute_120 汇编，ptxas 对 `.kind::f8f6f4`/block_scale 全部报 “not supported on .target sm_120”，日志 results/build-v0-archflag-failed.log / build-v1-archflag-failed.log）：改用 `-gencode arch=compute_120a,code=sm_120a`。
- 新增 `sass_host.sh`：镜像里 /usr/local/cuda/bin 无 nvdisasm，`cuobjdump -sass` 输出为空（results/ 中最初的 mma_rate_probe.sass 只有头部）；改把镜像内 pip 安装的 `/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin/nvdisasm` 所在目录加入 PATH 后重新反汇编（NVDISASM_PATH 写成文件路径无效）。
- `scaled_mm_ceiling.py`（冒烟运行 results/scaled_mm_smoke.* 暴露的问题，冒烟结果仅作调试，不用于结论）：
  1. 逐样本 SM 时钟恒为 195 MHz：图重放约 1.4 ms 短于 NVML 轮询周期，窗口无样本时回退到“最近的更早样本”，且 GPU 刚从空闲起步。改为 `window_mean` 前后各加 8 ms 填充，并在每个 (shape,M) 计时前加入 1.5 s 不计时的爬频预热。
  2. 异常文本截断由 600 改为 2500 字符（cuBLAS 提示的 128x128 scale 形状被截断）。
  3. 新增 arm：`cublas_blk_1x128_1x128_contigB`（B 的 1x128 scale 为连续 [K/128,N]），以及通过 `F.scaled_mm`（v2 API）的 `cublas_v2_blk_1x128_128x128_{colmajor,rowmajor}B`、`cublas_v2_blk_1x128_1x128_{colmajor,contig}B`。
- 新增 `probe_host.sh`（verify + 探针两次 + 按函数统计 SASS）、`scaled_host.sh`（单个 M 的 A2 运行，后台 nohup，轮询 .done 文件；每个 M 单独一次容器运行，31 对、warm+cold、两个 shape）。原因：到主机的 SSH 链路反复超时，长任务必须在主机后台运行。
- 容器以 root 写入 results/，故 mma_rate_probe、*.sass 等文件属 root（目录属 mingyue，可删/改名）。
- `run_a.sh`、`mma_rate_probe.cu` 的 PTX 内联汇编（BS_VARIANT=0）未改；BS_VARIANT=1 未被需要。
- 冒烟运行（results/smoke1-*、scaled_mm_smoke2.*）仅用于调试，不计入结论。
