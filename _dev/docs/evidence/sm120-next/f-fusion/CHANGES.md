# CHANGES（P3 小算子融合 f-fusion）

主机工作目录 `/tmp/qwen-sm120-next-f/`（新建；没有触碰其他 `/tmp/qwen-sm120-next-*`）。固定镜像 `sha256:a4a4c043…`（vLLM 0.31.0）。

## 新增文件
- `src/f_kernels.py`：两个 Triton 融合核及护栏入口。
  - F1 `_gdn_norm_quant_kernel`：把 `layer_norm_fwd_kernel`（GDN 门控 RMSNorm）的函数体逐表达式复制，尾部接 per-128 FP8 group 量化；`z` 按真实 stride（16384）读取，因此同时去掉官方链里的 `z.reshape(-1,128)` 拷贝（at::native::elementwise_kernel，每个 GDN 层一次）。
  - F2 `_conv_post_fused_kernel` + `_conv_state_update_kernel`：把 `_causal_conv1d_fwd_kernel`（width 4，silu）和 `_fused_post_conv_kernel`（q/k l2norm、v 拆分、g/beta 门控）合成一个核，仅用于“单个 prefill 序列”；官方 conv 的 state 更新（输入最后 3 个 token）由独立小核完成。
- `src/sitecustomize.py`：运行时守卫（`PRO6000D_F_ENABLE=norm,conv`；`PRO6000D_F_V2=1` 时先加载 V2 守卫 `v2_sitecustomize.py`，因此 V2 在所有臂都开启）。
- `scripts/`：`attribute.py`（nsys SQLite 归因）、`probe_quant_semantics*.py`（官方量化算子语义探测）、`test_numeric.py`（数值门）、`bench_micro.py`（微基准门）、`diag_*.py`（排错）、`f_run.py`（A1/B/A2 与 nsys witness 驱动，沿用 doc 11 的 `experiment_control.py` / `run_api_arm.py` 原件）、`run_in_image.sh`、`host_setup.sh`、`common.py`（来自 p-prefill，仅改 ROOT/PREFIX）。

## 关键实现决定（均由实测驱动）
1. 官方 CUDA 量化算子语义（探测得出）：`scale = fmaxf(absmax, 1e-10) / 448`（IEEE 除法，**不是**乘倒数）；`q = clamp(v / scale, -448, 448)`（IEEE 除法）→ e4m3 RN。融合核用 `libdevice.div_rn` 复现。注意 PyTorch 的 `tensor / 448.0` 会被实现为乘倒数，不能作为参照。
2. F1 的方差归约必须与官方 `layer_norm_fwd_kernel` 使用同一寄存器布局（`sizePerThread=[1,8]`, `threadsPerWarp=[2,16]`，1 warp）。第一版因 FP8 存储偏好 16 字节向量，Triton 把整条链改成 `[1,16]` 布局，导致 16% 的行 rstd 差 1 ulp（y 有 257/9930 万个元素不同）。修复：FP8 输出指针的 stride 参数 `do_not_specialize` 并加 `tl.multiple_of(...,[1,8])`，把存储向量限制为 8 字节；另外为归约单独加载一次 x（`evict_last`，避免被 CSE）。修复后 ttgir 布局与官方一致，y/rstd 逐位相同。
3. F2 的卷积累加顺序、BF16 乘积舍入、silu 表达式与官方逐句相同；l2norm 用同样的 `[16,128]` tile、4 warp，因此归约顺序相同。
4. 对 M（token 数）和 L 关闭 Triton 整除特化，避免真实请求长度不同引起首次 JIT 重编译落入 TTFT。
5. 守卫：F1 只在 SM120、BF16、`M>=256`、`z` stride=(*,128,1)、scales stride=(1,M) 时走融合核，否则在 op 内部运行官方链；F2 只在单序列 prefill（`num_prefills==1, num_decodes==0, 无 spec`，≥4 token，BF16，width 4，silu，无 bias，无 APC）时启用，否则运行官方 kernel。F2 用“conv 包装器记录、post-conv 包装器消费”的方式接入，不改 vLLM 源码。
