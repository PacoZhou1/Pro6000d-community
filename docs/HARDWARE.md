# 硬件与配置身份

2026-09-10 读数：Supermicro SYS-551A-T / X13SWA-TF，Xeon w5-3433（16C/32T、1 NUMA），RAM 67,062,489,088 bytes（标称64 GB），WD Blue SN5000 2 TB NVMe。GPU 均为 NVIDIA RTX 6000D，84 GB SKU，SM120 / 156 SM / 112 MiB L2；历史 nvidia-smi 可用容量约85651 MiB，不能按96 GB卡预算。

显存总线448-bit、读数12481 MHz，对应理论约1.398 TB/s。该理论值不能代替实测有效带宽，也不能用于推断 PCIe 跨卡归约的性能。GPU 全部 PCIe Gen5 ×16，同 NUMA、不同 CPU 根端口。CPU turbo 已开启，performance governor；IOMMU off 是已测运行环境，不是可归因的单独性能收益。

当前宿主为 Ubuntu22.04.5、Linux6.8.0-138-generic、580.178.04 OPEN。历史容量和冷8K/4K JSON 的 uname 记录 **6.8.0-40-generic**，保留原值，不冒充当前内核复测。两个板载网口为 Intel I210 1GbE 与 Marvell AQC113C 10GbE；现有链路均协商1000/full。推理引擎和正式社区口径测量在服务器 loopback，公网中转不参与其吞吐计时。

当前镜像为固定 R30，PyTorch2.13.0、CUDA13.3，具体 SHA 见 SOURCE_LOCK。模型为 local-inference-lab/GLM-5.3-Flash-NVFP4，保留 NVFP4 权重和 BF16 target head；MTP depth3。模型 config、index、tokenizer hash 已记录，但尚无本导出内的完整权重分片校验单或不可变下载 revision，因此不能宣称完成整模型逐字节身份认证。

## 当前 DMA9100 服务

- TP4 / DCP1，GPU-local KV，无 LMCache / native KV offload。
- FP8 target KV，16 GiB/卡，max_model_len262144，max_num_seqs64，batch8192。最大上下文是单请求配置上限，并不保证64个请求都能同时占满该长度。
- MoE W4A16（BF16 激活），小投影保持 BF16；这是区别于基线 W4A4 的精度路径选择。
- B12X DMA pieces4，oneshot及融合RMS阈值8KB，twoshot关闭，NCCL4通道，P2P level SYS / disable0。
- L2 persist 请求32 MiB；CUDA graph capture sizes1,2,4,8,16,32,64,128,256。
- 默认 temperature1、top_p0.95、top_k0；客户端可覆盖。社区 duration 测试用 top_p1，HTML用0.1。
- 350 W每卡，HF offline，监听127.0.0.1:8000，开发RPC关闭。

启动参数全文见 deploy/create.sh。262144不是模型理论极限，16 GiB是当前实配KV，不是四卡所有剩余显存。

## 运行边界

当前配置已验证整机重启后恢复模型和健康接口。完整负载后的温度、功耗必须按具体测试报告解释：最新HTML最大66°C、秒级309.51 W；历史冷并发波次最高82°C、353.50 W采样值（设定350 W）。秒级采样不能排除瞬时峰值。

机箱风扇由BMC管理；显卡外置风扇曾验证高转速可用，但独立于机箱的自动显卡温控曲线尚未完成验证，因此本仓库不提供声称已合格的独立风扇守护进程。
