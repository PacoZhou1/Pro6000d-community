# SM120 scheduler / BF16 epilogue：中期独立核验

状态：中期只读审查，**不是最终 mission 评审**。没有 SSH、GPU、服务操作或代码改动。依据本地 `scheduler-matrix-v4` 和四个 `epilogue-*` 的原始 `stdout.log`、源码快照、`status.json`、服务快照及派生 `analysis.json`；离线重新调用现有分析器并直接抽算配对比值。

## 数值与配对门槛

- `scheduler-matrix-v4/stdout.log` 有一份 metadata、13 个 numerics、三个 timed M（2741/2786/16160）、一份成功 summary；离线重算与保存的 `analysis.json` **完全相同**。13 个 case 的比较全部逐位通过，包括 M1/5/16/256 的官方 fallback、M257/2741/2786/16160/32768 的六个融合配置，以及 M257 的 zero/tiny/large/exceptional。四个 M 各有 8 个 changed-input Graph 见证（官方、旧 v6、六个 scheduler 配置），共 32 个，均通过；先前遗漏的尾块 M2786 已纳入。三个 timed M 的 warm 和近似缓存扰动各 31 对、每对八臂同输入随机顺序，timing capture 的输出再次与新鲜官方结果逐位核对。`analyze_results.py` 的完整性/输入 seed+SHA 条件已对这批记录实际通过。
- 四个 `epilogue-*` 目录各有数值通过、三条完整链的 changed-input Graph 通过、warm 与近似缓存扰动各 31 对；离线重算的各份 `analysis.json` 与原始 JSONL 一致。它们分别报告旧官方完整链、新官方 GEMM 与完整链、paired BF16 GEMM、逆排列、V2量化和 paired 完整链；不把普通 BF16 的三核链伪装成融合候选。M16160 的三次诊断使用相同 `input_sha256=e8c90231...95714`，M2741 的诊断输入也与主矩阵相同。所有五次使用 v3 库 SHA `af1d8149...85d6`，probe metadata 的自身 SHA 与每次运行前的源码快照相符。
- 五次 `status.json` 均 exit 0、`service_after_passed=true`、自有容器已清理；前后服务快照均通过，原 27B/9B/ASR 保持停止、embedding 运行。这些实验很短，embedding monitor 每次只有 1–2 个采样且无坏样本，只能支持这些时点和末尾功能检查，不能称连续 SLA 或零干扰。

## 可支持的核级结果

| 合成完整流程，warm 31 对中位 | 旧官方 | 旧 v6 | 新 v6 Heuristic/s8 | 新 v6 AlongM/s1 | 意义 |
| --- | ---: | ---: | ---: | ---: | --- |
| M2741 | 1.762 ms | 1.931 ms | 1.939 ms | 1.985 ms | 此尺寸新 AlongM/s1 比旧官方慢，不能视为通用优化。 |
| M2786 | 1.796 ms | 2.003 ms | 1.997 ms | 2.050 ms | 尾块结论与 M2741 一致，且数值/Graph 覆盖完整。 |
| M16160 | 10.446 ms | 12.513 ms | 12.504 ms | 11.367 ms | AlongM/s1 改善了旧融合臂，但仍比官方完整链慢约 8.8%；没有可部署的跨尺寸胜出结论。 |

同一主矩阵中，旧 v6 与新 Heuristic/s8 时长很接近，支持控制臂可比；这是单次合成算子的 Graph replay，不是实际服务延迟。缓存扰动模式的方向一致，且 flush 只是大于两倍 L2 的自有写入，不是固定冷缓存。不能由这些时间反推出 DRAM 读哪个 buffer 或宣称 raster 已被证明为唯一原因。

**约 2% 的小收益口径。** 单独的 `epilogue-m16160-along-m-s1` 把旧官方完整链与新库官方 AlongM/s1 完整链在同一进程成对比较：warm 配对旧/新中位比 **1.020884**，近似缓存扰动 **1.019506**；两种模式各 31/31 对均为新臂更快，分别约 10.429→10.203 ms、10.422→10.225 ms。原始配对 warm 比值第 10/90 百分位约 1.016/1.023。可说“该固定 M 与合成输入、该次隔离核级配对里约 2%”，**不能说模型服务提速 2%**。旧官方与新官方来自不同 DSO/编译入口，直接对比不能把全部差异归给 swizzle=1；同新库 AlongM/s8 属另一独立运行，其旧/新比值约 1.000，非与 s1 严格交错的同进程消融。`AlongN/s8` 对官方反而慢，M2741 AlongM/s8 近持平。

**追加同库三臂对照，收紧归因与形状域。** `official-tuning-v2` 在同一进程、同一 v3 库下比较旧官方 Heuristic/s8、新官方 Heuristic/s8、新官方 AlongM/s1，四个 M 各有 BF16、FP8、scale 全量逐位检查、三臂改输入 Graph 与 timing capture 复核；warm/扰缓存各 31 对。原 `stdout.log` 离线重算与保存的 `analysis.json` 完全一致，结束服务与 embedding 两次采样通过。**M16160** 的新库 Heuristic/s8 对新库 AlongM/s1 配对中位比 warm **1.02025**、扰缓存 **1.01928**，两组各 **31/31 对** AlongM/s1 更快；旧官方对新库 Heuristic/s8 约 1.00085/1.00061。这使该尺寸上约 2% 的调度收益有了同库单变量支持，仍只限合成完整算子链。

反证同样明确：**M32768** 的新库 Heuristic/s8 对 AlongM/s1 配对比仅 **0.66421/0.66489**，单臂中位时间约 **21.95→33.04 ms**，两组各 31/31 对 AlongM/s1 更慢，退化约 **50%**。M2741/2786 的配对比接近 1，不能据某个单臂中位数差额声称提速。因此后续 runtime 若接入，应先严守 **M 恰为 16160** 的选择条件，不能用 `M>=16160` 或“大 M”范围代替；还要通过真实 compiled graph 命中、数值/Graph 与无 profiler 整模 A/B/A。当前没有生产收益结论。

普通 BF16 同 M64N256 的 `paired_gemm`、逆排列和量化分别计时可帮助拆成本，但 paired 完整链含额外 inverse，不能与融合 GEMM+pack 直接比较或把分项中位数相减作精确因果归因。相关数值与 Graph 门槛已通过；关于主循环/epilogue 内部流量仍需 NCU 同范围计数及后续无 profiler 整模 A/B/A。

## 隔离模型 runtime 的静态预审（尚未运行 GPU）

`model-runtime/candidate_abi.py` 的 C ABI 参数顺序与 `scheduler_abi.h` 一致，库 SHA、ABI1、N/K/架构在装载时检查；workspace 查询和启动硬限制 `M==16160`、AlongM/s1。`candidate_op.py` 又要求注册的 canonical B/Bs 存储身份与 SM120，其他合法 M 或未注册权重调用固定 CUTLASS GEMM 加原 V2 量化。候选 BF16 中间张量与 workspace 按调用分配，并在当前流上记录使用；C++ adapter 的控制字段为私有 `static thread_local`。FX 改写沿用前轮严格匹配器，仅候选臂插入新功能 op，pass UUID 带库 SHA；Meta 输出形状保留符号 M，未复制旧量化节点的 `eager_input_vals`。离线逐文件比对 frozen-runtime-v6，除候选 ABI/op、FX 改写、sitecustomize、smoke 外，匹配器、pack 注册、FX probe/分析与 CPU selfcheck 源码相同。**未发现阻断冒烟的确定静态问题**；这些检查不能替代真实 compiled graph、逐位数值、改输入 Graph replay 与服务 trace。

共同的 startup pack 会为 64/65 层额外复制约 10.6/10.8 GiB 的 gate/up 权重与 scale。A/B 两臂都做相同复制，仍须记录实际 `startup-pack` 总字节、相同 KV 池与容器启动结果；若因显存不足而改动其中一臂的池大小，这次比较不可直接归因于调度。Python 首次 dispatch/capture 记录也不能充当 CUDA Graph replay 或整模请求次数。

## 失败记录与待核验边界

- `hardware_control.py` 已从已有远端快照补回本地，SHA256 `c532ed67c6f4abcea5a57b114667b5b04783f812caf1a45c170836cce9832791` 与 `build-v1/status.json` 的源码 SHA 一致，先前 NCU 指标文件缺失意见闭环。
- `ncu-m16160-v6-along-n-s8/status.json` exit 11；stderr 的 SIGSEGV 栈落在旧 `candidate_adapter.py` 的 legacy launch，尚无目标 `profiled_once` 或有效原始计数。保存的调用栈不能单独证明具体 ELF 符号冲突机理；此失败不参与 NCU 对照或性能结论。去掉选定 NCU arm 无关的 legacy 加载后，还需重新获取完整同口径的各臂计数。
- 本批结果只证明选定合成算子的数值与 Graph replay 以及短时核级计时。尚无新配置在真实模型编译图、真实 prefill/decode API、长期 MoBase 质量、生产服务或跨重启稳定性上的收益验证；实验中的任何 2% 不可外推。
