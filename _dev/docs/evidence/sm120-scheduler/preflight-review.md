# SM120 scheduler 实验：中途静态预审

状态：预审，非最终 mission 验收。仅阅读本地设计、控制器、probe、C ABI、离线生成源码、已落盘构建/失败日志与分析器；未连接服务器、运行 GPU 或改变服务。实际数值、Graph、NCU 和整模结果仍待 root 验证。

## 已核对的实验边界

- 原请求要求实际实施受控的 scheduler/epilogue 实验，保持 FP8、BF16 `lm_head`、embedding 运行及原 27B/9B/ASR 停用；`probe-design.md` 把官方 BF16 GEMM+V2量化、上一轮 v6 和新同 tile 变体分为对照/锚点/候选，且将普通 BF16 epilogue 标为单独诊断，符合分离变量的方向。
- `run_control.py` 仅建立 `codex-sm120-` 自有容器，网络禁用，构建模式没有 GPU，probe/NCU 模式不含模型服务管理命令。每次正常进入试验前做服务快照，容器运行期间按 15 秒间隔记录 embedding health/身份，结束后扫描这些样本并做含实际向量请求的功能检查；超时尝试停止自有容器，保留日志与状态。`verify_service_state.py` 会核对四服务身份/参数/重启计数以及 embedding 原启动时间，不把访问凭据序列化。只能声称采样时点健康。
- `probe-design.md` 要求所有 arm 的逐位数值、下游结果与改输入 CUDA Graph 重放通过后才计时；预分配资源、相同输入与随机化顺序、完整 GEMM→量化流程和独立 GEMM-only 计时标签均是必要边界。缓存扰动只能称近似，不是保证的冷 L2。`analyze_results.py` 已检查 arm 集、31 组配对、有限正时间与完整流程标签。
- 首版 `scheduler_probe.py` 的 ctypes 签名与 `scheduler_abi.h` 的三种 launch、三个 workspace 查询和 temp-scale 查询在参数数目/顺序上相符；M257–32768、FP8/scale 布局 guard、每臂独立输出/工作区及捕获前预分配已在 Python 层体现。官方与旧 v6 完整流程均进入 graph timing；`old.capture()` 先暖启动再捕获，每一轮按同一输入跨臂随机次序测 CUDA event。新增 input 内容 SHA 与 seed 在数值/timing 两处记录，分析器现可交叉拒绝不一致。上述仍是静态检查，不代表 CUDA 编译或运行已通过。
- 离线在内存中对本地 pinned v6 `standalone_dispatch.cuh` 应用了 `build_scheduler.py` 的补丁：原文件 SHA 与 builder 常量一致；差异只引入可选 StageCount、普通 BF16 配置、`raster_order` 参数/枚举映射及注释；原 `scheduler.max_swizzle_size` 仍保留。固定 adapter 的 `c3x` 控制指针为 `thread_local`，`query_only` 在 workspace 查询时先算需求再返回；C++ `static_assert` 要求普通 BF16 诊断与融合主循环为同一类型。`build/build-manifest.json` 显示 nvcc 13.0 编译 exit 0，库 SHA `671e83890f77687e1a8a984ae1b64766b096bd94437f8fdade6963bb5e282400` 与 `probe-command.json` 锁定值一致；ptxas 日志中 paired/fused 均 168 registers、零 spill，官方控制也 168 registers 但有 12/20B 静态 spill store/load。编译不替代 GPU 执行验收。

## 预审发现及闭环

1. 首版八臂 `scheduler_probe.py` 只研究 fused raster/swizzle，无法独自完成普通 BF16 epilogue 消融。现已另有 `epilogue_probe.py`：普通 BF16 同 tile GEMM-only、逆排列-only、V2 量化-only、旧/新官方与 paired 完整链分开标名；前置 BF16/FP8/scale 逐位数值与三条完整链 changed-input Graph。它不混入八臂统计。实际诊断数值和时间仍待 GPU 证据。
2. `run_control.py` 的早期 preflight 状态、py/cu/cuh/h/json 和命令快照、15 秒 embedding 样本核对、末尾 `service_after_passed` 已补齐。当前状态可区分超时、子进程、监控、功能检查和 NCU export 失败；健康结论只能覆盖采样时点。
3. `scheduler_probe.py`、`epilogue_probe.py` 已记录 CUDA driver/runtime 版本。主 probe 的 input seed+SHA 在数值/timing 两处交叉对照，分析器要求八臂 Graph 输出变化、timing capture fresh-output 检查和完整配对；先前 provenance 与 seed 意见已闭环。

## 首轮崩溃与 v3 隔离修复

先前 v1 `scheduler-matrix-v1` 和最小复现 `scheduler-fault-v1` 均以 139 退出；`status.json` 保留失败，服务后功能检查 passed，embedding 当次采样健康。`symbol-collision-v1.json` 原始 ELF 行显示新库的 `vllm::c3x` 工作区/stream 为公开 `TLS UNIQUE DEFAULT`，旧 baseline 同名对象为 `OBJECT UNIQUE DEFAULT`，与首次旧 `Baseline.gemm` 的 faulthandler 崩溃位置吻合。这是 host 符号冲突的强证据；139 不能算数值或性能通过。

本地逐文件比对 v2 与 v3 生成源码：`standalone_dispatch.cuh` 只把 `vllm` 改为 `pro6000d_scheduler_v1`，并把融合 visitor 类型换成私有标识；`standalone_adapter.cuh` 同时将五个 `inline thread_local` 改为 `static thread_local`；`paired_gateup_visitor.cuh` 只改类型标识；`scheduler_driver.cu` 只改命名空间限定与 pack 的 stream 引用；`scheduler_pack.cuh` SHA 完全相同。C ABI 参数次序未变，仅在十个导出原型加显式 default visibility。`build-v3/build-manifest.json` 的所有生成源 SHA 与本地 `build-emit-v3` 一致，`pack_body_sha256` 与旧构建相同；新库 SHA 为 `af1d814966a746ff09edc55d40196b871067d88d22ed0e50700c1f3a4eb185d6`，编译 exit 0。ptxas 仍显示 fused/paired 各 168 registers、0 spill，官方 168 registers 且静态 12/20B spill；编译期同主循环类型断言通过。

`symbol-isolation-v3.json` 保存的 readelf 行显示 `c3x` 状态现为私有 namespace 下的 `TLS LOCAL DEFAULT`，十个 C ABI 符号仍为 `GLOBAL DEFAULT`，`unsafe_c3x_symbols` 为空。这个文件是保存的 ELF 检查记录，本地未包含 v3 `.so`，故本次独立复核了记录内部与源/构建 manifest 的一致性，没有重新运行 `readelf`。这些静态证据支持隔离修复已正确构建；**不等于 v3 GPU 数值、Graph 或性能已通过**，正在运行的 v3 matrix 应以其原始记录另判。

先前指出的 **NCU 指标文件缺失** 已核实闭环：`hardware_control.py` 已补回本地，SHA256 为 `c532ed67c6f4abcea5a57b114667b5b04783f812caf1a45c170836cce9832791`，与最早 `build-v1/status.json` 记录的远端源码快照 SHA 一致；它包含控制器需要的 `METRICS`。这仅解除导入前置条件，不保证 NCU 捕获成功。

后续 GPU 运行证据另见 `interim-review.md`。本预审对 v3 的判断只覆盖源码隔离、ABI 与编译，不能以它代替逐臂数值/Graph/计时或 NCU 证据。
