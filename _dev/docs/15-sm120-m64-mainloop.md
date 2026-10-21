# 15. PRO6000 SM120：M64 主循环 warp 重排与整模验证

本轮只做 FP8，完成M64主循环4×2→2×4 warp重排、完整算子链比较和真实模型A/B/A。新布局在同调度M16160微基准中比旧融合核快约2.6%，但四个实测长度上仍慢于官方完整链。整模候选TTFT为2.309秒，前后基线2.243／2.257秒，12个配对请求的TTFT退化中位数2.63%，**不采用，也未接入生产**。embedding保留运行；原27B、9B、ASR继续停止，临时实验容器已清理。

## 实现和硬件约束

固定GPU为RTX PRO6000 Blackwell Workstation（SM120、188SM、96GB、128MiB L2）；每SM65536个32-bit寄存器、100KiB shared，用户单CTA最多99KiB。M64×N256×K128 Cooperative核使用256 MMA线程＋128生产者线程、两级流水，融合核动态shared101376B，已用满99KiB。它不具备再加一级K128流水的空间：仅A/B三stage就需要122880B。运行时并未修改GPU时钟或生产服务配置。[硬件事实](evidence/hardware-grounded/sm120-facts.md)、[编译资源与SHA](evidence/sm120-mainloop/build-results.md)。

每warp唯一operand一次的逻辑模型中，4×2布局每K128读取A+B为147456B；2×4为98304B。这是shared层重复读者模型，CTA唯一TMA payload仍为40960B，不预测DRAM减少。固定硬件MMA atom仍是m16n8k32，不能用改变CuTe布局冒充改变Tensor Core指令。

| 变体 | Warp M×N | PermN | B shared copy | 状态 |
|---|---|---:|---|---|
| control42 |4×2|32|LDSM x4|不变的M64融合对照|
| warp24 |2×4|32|LDSM x4|编译拒绝：B每线程8B不足x4要求16B|
| warp24x2_v3 |2×4|32|LDSM x2|完整数值／Graph通过|
| warp24n64_v3 |2×4|64|LDSM x4|完整数值／Graph通过，选作整模隔离对照|

2×4还需要把SF128归约改为四warp。首版v2虽然逐位数值通过，重建实际epilogue目的地址后发现每scale两个写者；v3将写者限定为每组warp0或1，128个地址各写一次。没有放宽scale分组、舍入、SiLU或FP8精度，也没有更改BF16 lm_head。[写者证据及范围](evidence/sm120-mainloop/scale-writer-probe.md)。

## 官方完整链比较

官方链为FP8 GEMM写BF16→V2 SiLU×up及per128 FP8量化；候选链在GEMM epilogue中保留相同BF16舍入、激活和量化，再执行输出FP8及scale整理。权重重排发生在加载时，输出整理仍是每次调用的GPU工作；两者不能混为一谈。

同进程、同输入权重，21臂随机交错31轮；warm和自有缓存扰动分别报告。下表为warm中各M的最佳新布局，单位ms；完整分位与配对比值含在[全部结果](evidence/sm120-mainloop/timing-results.md)，缓存扰动并非可控冷缓存。

| M | 官方GEMM＋V2 | 最佳新融合＋pack | 配置 |
|---:|---:|---:|---|
|2741|1.7886|1.9145|n64 / Heuristic / s8|
|2786|1.8016|1.9646|n64 / AlongM / s8|
|16160|10.6286|11.3061|n64 / AlongM / s1|
|32768|22.9065|23.9538|n64 / AlongM / s8|

M16160下，同AlongM/s1的4×2融合为11.6133ms，新布局11.3061ms；逐轮配对速度比中位数为1.02574×，扰缓存组为1.02587×。这是本次运行状态下约2.6%的观测收益，仍比官方慢；没有锁频，尚不能拆分动态频率、功耗与执行周期的贡献。不能把最佳单点推广为所有大M都用s1。

## Profile解释了什么

五组同输入NCU包含官方完整链、4×2及2×4 fused完整链，以及两者只换回普通BF16 epilogue的matched GEMM诊断。2×4 fused GEMM的shared load wavefront降约31%，load bank conflict降约43%；Tensor指令数相同，MIO throttle样本从2.156%降到0.332%。完整链DRAM总字节约0.9998×、L2请求约1.00001×，没有减少外层数据流量。

融合GEMM的Tensor active约68.06%→68.02%，官方约83.55%；168寄存器/线程、shared预算与CTA驻留限制没有解除。共享读取组织变好了，但Tensor流水利用率仍落后，不能仅凭本实验把差距全部归因于M64形状。

NCU中候选GEMM时长9.738464→9.516352ms，平均时钟同时从2.262197→2.316136GHz；两者粗乘约22.0303→22.0412百万cycles，基本相同，SMsp执行指令反而增加约1.40%。跨replay平均值的乘积只能做近似检查，但足以阻止把这次时长下降解释成已证明的执行周期减少。NCU中的warp stall百分比不能相加当耗时占比，单次replay时间不能当API速度。[原始计数、逐核资源、限制](evidence/sm120-mainloop/ncu-analysis.md)。

## 整模验证

使用固定vLLM0.31.0镜像、同一27B FP8模型、V2量化和12GiB KV；A/B两侧都持有相同canonical与预排权重副本（额外11589593600B）。候选仅允许注册权重身份、SM120、M16160、AlongM/s1，其余形状回退原链。GPU synthetic runtime smoke已通过目标、未注册身份、小M及非目标大M和换输入Graph；固定镜像CPU Inductor schema与序列化检查也通过。真实模型witness已在65个注册模块核对canonical与packed身份，记录到65次融合GEMM＋65次pack，全部是成功的direct launch关联。小M、M2741和M32768等回退通过。实际模型witness没有Graph replay；换输入Graph证据来自synthetic smoke，不能混写。[实际launch](evidence/sm120-mainloop/model-witness-v1/witness/dispatch-summary.json)、[集成身份与内存](evidence/sm120-mainloop/model-witness-v1/witness/integration-summary.json)。

本轮依用户要求继续测负结果的整模对照，不因microbenchmark未胜出而省略。三臂各重新启动隔离容器，命令、模型、镜像、KV预算、canonical/packed权重占用相同；B只额外启用SHA固定的候选库，性能运行没有profiler。每臂prefill为12个约18.9K输入／1输出请求，decode为6个约4.8K输入／1024输出请求，另有各自warmup剔除。性能请求逐index的payload hash全部一致。

| 指标 | 官方链A1 | 新融合B | 官方链A2 |
|---|---:|---:|---:|
|prefill TTFT中位数（秒）|2.2434|2.3090|2.2569|
|decode中位数（tok/s）|106.77|104.64|106.76|
|每次MTP draft接受token中位数|2.0596|2.0044|2.0596|
|固定质量检查|5/5|5/5|5/5|
|开放输出正常stop且有最终正文|4/4|4/4|4/4|

每个相同输入以(A1+A2)/2作基线，候选TTFT增加中位数2.627%，范围2.515%～2.732%；前后基线A2/A1的配对漂移中位数0.559%。12个prefill和6个decode的三臂请求窗口均未记录embedding计数变化，按该条件过滤不会改变结果，但离散快照不能保证窗口内绝无短暂活动。三臂prefill首个非空输出token全部相同；decode的A1/A2正文hash为6/6相同，B与任一基线为0/6相同，故decode吞吐及MTP接受率差异不能作纯算子耗时归因。质量探针只覆盖五个合成案例，旧探针没有保存payload hash；开放输出保存了hash并跨臂相同。这些结果不能替代MoBase业务质量验证。

单独的profile请求证明候选在真实模型实际执行了65次，但无profiler的12个prefill请求没有逐请求kernel/M证据，不能宣称每条都命中M16160。HTTP prompt token数不是内部chunk的GEMM M；witness的18901输入实测包含M16160和M2741。因此本表证明的是这个隔离配置的负结果，不把2.63%退化全部归给某一条核。再加上额外11.59GB权重副本，当前无采用理由。[逐请求及分析](evidence/sm120-mainloop/aba-mainloop-v1/analysis.json)、[同配置核验和采用结论](evidence/sm120-mainloop/aba-mainloop-v1/model-contract.json)、[本地复算核验](evidence/sm120-mainloop/aba-mainloop-v1/recompute-check.json)。

decode文本差异与候选集成关联，根因尚未定位：B的FX改写在`compile_range=(1,32768)`图上进行了20处替换，M16160之外虽回退canonical GEMM＋V2数学，opaque自定义算子仍改变了编译边界、buffer生命周期和图组织。合成fallback逐位检查通过不等于真实decode逐层logits相同；本轮没有这六条输入的逐层logits、首个分叉token或逐请求decode trace。后续若再次进行融合整模因果对照，应补充同编译边界的官方fallback控制臂及真实输入的分叉定位，避免把图集成差异算作主循环收益。

## 服务状态

本轮23组监控文件合计119次采样，embedding均running且health200，容器ID、启动时间及restart_count=1与本轮初始状态一致。最终两次实际embedding请求相隔54.10秒，均200并返回1024维有限值；原27B/9B/ASR继续停止，未修改原容器。最终检查无残留的本任务实验容器。这是采样健康和容器连续性证据，不是持续请求延迟保证。[服务核验](evidence/sm120-mainloop/service-audit.json)、[最终容器列表](evidence/sm120-mainloop/final-container-audit.json)。

## 下一步依据

本轮排除了“只换warp读者即可追平官方”的假设。更值得单独测试的是M64N128的两级／三级流水，输入和权重仍为FP8，先用普通BF16输出epilogue隔离主循环成本：三stage的A/B原始shared73728B，双份accumulator均摊从64+64降到32+32 FP32/线程，可能给更多流水或寄存器调度留下空间；同时N方向CTA和A载入增加，收益尚未测，不能直接采用。保留K128缩放与累加顺序，先复核生成资源、逐位数值和官方完整链，再决定是否做新融合。当前两种2×4结果与失败尝试均保留，可复现而不修改安装目录。

复现入口：[构建与精确版本](evidence/sm120-mainloop/builder-design.md)、[数值门槛](evidence/sm120-mainloop/gate-results.md)、[模型控制器](evidence/sm120-mainloop/model-control/README.md)。所有性能数据都是本机单卡、这组输入和固定镜像下的结果。
