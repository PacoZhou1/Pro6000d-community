# CHANGES（复测脚本改动）
原脚本（d3_client.py、d3_compare.py、analyze_d4.py、profile_client.py、common.py、补丁、子集文件）均未改动，直接复用。
- arm.py（仅 3 处）：runs 目录可由环境变量 RUNS_DIR 指定（默认仍为原路径，避免与上一时段的 DONE 臂冲突）；初始快照路径可由 INITIAL_SNAPSHOT 指定（本时段开始时重新取快照）；新增客户端类型 `x`（运行 xclient.py）。主机 bin/arm.py 已被替换，原版保存为 bin/arm.py.session1。
- 新增 xclient.py：24 条全新贪婪请求（ignore_eos，输出 256/1024/2048 各 8 条；normal/thinking 各 12；话题 generic/zh/code/zh2 各 6，全因子组合），payload 构造沿用 d3_client.payload（generic 逐字相同；zh/zh2 换中文词表与中文结尾指令，code 为合成 Python 函数 + 英文代码评审指令；种子标签 retest-x-v1:*，与旧请求不重合）。每条请求在同一服务上先非流式、再流式（stream=True）各发一次，均带 return_token_ids（流式从各 chunk 的 choices[].token_ids 累积）。
- 新增 xcompare.py：基线与候选逐 token 对比（两条路径），以及同一臂内非流式 vs 流式。
- 新增 driver_retest.sh：d3-base(d3+x) → d3-65536(d3+x) → d4-a1/b/a2 → 比较/分析 → d3-32768(仅 d3，原 16 条，非流式) → 打包。
- 未改补丁；无失败臂；无 codex-next-d-* 容器遗留。
