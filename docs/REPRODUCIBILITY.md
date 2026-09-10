# 测量复现与脚本身份

## 公共工具

`python3 benchmarks/fetch-community.py` 获取公开 `llm-inference-bench` commit `80d1f1b0ab9830c3fd8a22c42f461c40cbc7cf96` 的0.6.1文件并验证SHA256 `516dc590b80dda6d9880f9f5034026fdf1b2074e9747bf8a920758680f019817`。

历史本机0.6.2与这个公共0.6.1除VERSION字符串外所有字节一致，证据见 results/benchmark-source-equivalence.json。社区只报告0.6.1，未发布其实际私有测试文件hash；因此“口径和恢复的公开代码对齐”不等于证明双方使用完全同一文件。R28.1历史并发使用0.4.29，不能忽略这个差异。

运行在GPU服务器，暂停其他客户端。持有Docker权限的账户执行以下命令。输出写入 `runs/`，不会发布。运行前确认当前服务/MTP3/350W/温度/可用KV，不要同时启动多个压测脚本。

```bash
python3 benchmarks/fetch-community.py
sudo bash benchmarks/run-sustained.sh glm53-r30-dma9100
sudo python3 benchmarks/run-prefill12.py glm53-r30-dma9100 cold32k-review
sudo bash benchmarks/run-cold-8k4k.sh glm53-r30-dma9100
```

以上为顺序执行示例，不是要求一次都跑。依赖使用R30容器现有Python环境；如模块缺失，按固定版本上游工具的依赖说明准备，不在生产服务中盲目升级torch/vLLM。

## 三个服务端实验

- 持续decode：context0，CC1/8/16/32/64，15秒预热、30秒计时，temperature1（原工具top_p1），最多8192输出token用于保持窗口。需要真实active达到CC，不能用客户端连接数替代；容量受限/循环/错误均应失败。保存秒级GPU功耗、温度和boot ID。
- 冷32K：仍用上游prompt、tokenization、TTFT和聚合；`bench-prefill12.py`只将停止条件改成12个样本并逐样本留证，3轮，约323xx实际输入tokens，不是恰好32768。预先检查无其他running/waiting；检查MTP3、350W、boot一致。server validation不可用时只能报告未知，不能宣称cached0。
- 冷8K/4K：request-count=CC，exact targeting，独立cache_salt，保留并排除隐藏预热。验证每请求8192输入/4096输出，所有token计数一致。最终吞吐包含prefill/排队/收尾；CC64最多57活跃的旧结果不能当成64路持续decode。

便携脚本只替换原机路径、输出目录和获取方式；0.6.2改取可验证的0.6.1等效文件。新脚本仍需在目标机器做实跑验收；仓库发布阶段仅做静态及mock检查，不产生新速度成绩。

## 原HTML

用户提供的“大模型 Prefill & Decode 性能测试工具 v3 (chao魔改版)”不随仓库重新分发。原文件SHA256 `bd16096d9727cccc7cd5df4e9d7df6093f362fac1b1527b4fd480f299b4638b5`。选择OpenAI Chat Completions、正确模型ID、C1、输入8192–32768步1024、输出128、temperature1/top_p0.1。

原HTML的网络延迟扣除公式和短输出不能与社区口径混用。本次使用Chrome和Node streaming bridge→认证LAN网关，图表记录此边界。完整结果和截图可以重新浏览，工具源文件应从其作者或授权持有者取得。

## 正确性与可复核范围

qualification目录包含BF16投影真实权重/compile/四卡切片资格脚本，以及mHC CPU对象绑定+实际CUDA重放资格脚本。需要固定R30环境、真实权重和空闲GPU；不要与日常服务争抢同一卡。脚本使用 `/models/GLM-5.3-Flash-NVFP4`、`/work` 作为容器内约定挂载。

mHC脚本通过 `--original` 指向SOURCE_LOCK中vLLM原版b12x.py，`--candidate` 指向patches/mhc-lifetime-v1.py，`--output`选择新JSON；两者SHA都强校验。参考部署实际四文件hash，不把未运行GPU的Python语法检查称为数值通过。

results包含数字摘要和经过字段过滤的完整数值单元；没有发布私有请求正文、全部GPU trace、权重、API key和原实验Git历史。历史源hash和缺失raw路径是可追溯线索，不意味着对应大文件已包含在此仓库。
