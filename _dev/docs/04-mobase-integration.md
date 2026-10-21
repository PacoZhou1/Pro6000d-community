# MoBase 接入与性能验收审计

日期：2026-10-13。审计对象：`/Users/researcher/Projects/WeKnora/MoBase`，本地 HEAD `974b978d`。

本次仅检查项目说明、代码、白名单环境字段和已保存的当日请求参数/统计；没有访问数据库、发送模型请求、修改 MoBase、操作 Docker或提交 Git。以下“当前代码”指本机版本，不能据此假定生产代码、数据库模型配置和 vLLM 启动参数相同。文档不包含认证信息、业务问题、工具实参或模型正文；服务器公网 IP 不落盘。

## 1. 接入事实与待确认项

| 项目 | 已核实 | 证据/限制 |
|---|---|---|
| 线上检索引擎 | `__mobase_agent__` → `mobase_agent` | `apps/langgraph/api.py:19`、`:235`；项目 README 的 v2 描述不能作为现役链路依据 |
| 对话入口 | `POST /chat_stream`（由 Ninja 路由前缀组合完整 URL） | `apps/langgraph/api.py:185`；最终 prefix 应由实际部署核验 |
| 实际模型身份 | 当日 Qwen 请求为 `qwen3.6-27b-fp8` | 当日保存的 `raw/*final-qwen*Agent.json`、`*Generation.json` 白名单参数 |
| 模型 API | OpenAI 兼容 `/v1/chat/completions` | `core/clients/model_factory.py:154`；数据库 `parameters.base_url` 为主要来源 |
| 历史域名路径 | `https://vllm3.newblash.site/v1` | 既有历史说明，当前在线数据库配置尚未核验；不是本次探活结果 |
| GPU 本机候选路径 | `http://127.0.0.1:8000/v1` | 历史部署 `deploy/服务部署命令参考.md:125` 与 `_dev/quant_bench/common.py:23`；实际容器端口待服务端核验 |
| `.env` fallback | `QWEN_PLUS_BASE_URL`、`QWEN3_32B_AWQ_BASE_URL` 为空 | 只读取明确白名单；不能因此认定模型未配置 |
| 配置优先级 | 进程缓存 → DB UUID/名称 → 默认 KnowledgeQA → 可用首项 → settings | `apps/langgraph/utils/model.py:413`；Model.parameters 中读取真实 `model_name/model/base_url`（`:470`） |
| 节点模型 | 节点专属配置可覆盖前端所选主模型 | `apps/langgraph/utils/model.py:45`；图级配置与知识库配置合并在 `apps/langgraph/api.py:294`、`:304` |
| 参数缓存 | DB/model/context 长度均有进程缓存 | `apps/langgraph/utils/model.py:408`、`:513`、`:628`；变更必须确认每个服务进程已清缓存 |

先核对数据库中的非 secret 字段：`name/display_name/type/source/status/is_default`，`parameters` 仅投影 `base_url/model_name/model`，以及节点模型指向和备用模型身份。不要输出完整 `parameters`，也不要打印 `api_key`。当前主链路直接构造 ChatOpenAI；只有配置本身指向 MoBase gateway 时才经过该网关，不能把 gateway 参数误当所有聊天调用的参数。

## 2. 请求参数、思考和流式

| 阶段 | 思考与采样 | 输出预算 | 来源 |
|---|---|---|---|
| query_understanding | 关思考，temperature 0.7 | 工厂动态预算 | `graphs/mobase_agent/nodes/query_understanding.py:527` |
| 主 Agent | 开思考，temperature 1.0、top_p 0.95、top_k 20 | 默认 `min(context_len×0.3,8192)`；此调用未传独立 thinking_budget | `nodes/agent.py:1378`；`utils/model.py:691` |
| 最终 generation | 关思考，temperature 0.8、top_p 0.8、top_k 20、presence_penalty 0.3 | 简洁 `clip(retrieval_chars/6,512,8192)`，详细 `/3`；投资方案另有输出下限 | `nodes/generation.py:1466`、`:1491`、`:1498` |

上表 `graphs/`、`nodes/` 相对 `apps/langgraph/`、`apps/langgraph/graphs/mobase_agent/`。

`build_extra_body` 同时下发顶层 `enable_thinking` 和 `chat_template_kwargs.enable_thinking`；vLLM 模板开关是后者。默认 repetition_penalty 1.1。独立 thinking_budget 只有调用方传值且开思考时才追加（`core/clients/model_factory.py:41`、`:66`）。本次当日 Qwen wire 不含独立思考预算，Agent `max_completion_tokens=8192`；不要把构造参数 `max_tokens` 与线上 JSON key 不一致误判为未设置预算。

工厂没有显式 `streaming=True`，但真实聊天使用 `graph_app.astream_events(..., version="v2")`（`apps/langgraph/api.py:434`）。本机 LangChain 会因流式 callback 使用流式 API（安装包 `langchain_core/language_models/chat_models.py:549`）。当日 wire 直接证实 Agent 与 generation 都是 `stream=true`。单独 `ainvoke` 的测试如果未使用相同 callback，可能转为非流式，不能替代用户真实路径。

Agent 绑定路由允许的工具子集（`nodes/agent.py:1565`），`call_with_tools` → `invoke_with_retry` → `ainvoke`（`core/capabilities/llm_tools.py:496`）。当前工具调用默认 auto，未见主 Agent 常规调用强制 required；服务端必须核对工具 parser 与当前模型模板。应用侧生成的 prompt/messages/tools 和服务端 chat template 是两层：MoBase 不下发自定义 Jinja 模板，实际模板需要从服务器 tokenizer 配置或启动覆盖项确认。

## 3. 用户等待时间可能来自哪里

1. **长输入与多次串行模型调用。** Agent 使用历史、规则、工具结果和工具 schema；简洁模式最多 6 步、其他默认 8 步（`nodes/agent.py:579`、`:886`）。最终回答还要经过独立 generation。不能把完整回答耗时当单次模型 decode 速度。
2. **思考生成。** 主 Agent 开思考且未传独立思考预算。首次 reasoning/token、首次正文、首次 tool_calls 和用户首次 answer_chunk 必须分别计时。只观察 content 会把正常思考阶段误报成“没有输出”；关闭 Agent 思考是行为变化，不能用文本短答跑分证明业务等价。
3. **模板和工具 schema 的输入体量。** 明确记录 tokenizer 计算的 prompt_tokens；字符数和工具个数只作为规模说明。Agent 动态工具集合/系统规则可能影响 prefix cache 命中，需在服务端测冷/热缓存。
4. **辅助计数请求。** `get_model_context_length` 在 async 函数内调用同步 `httpx.get`，首次请求/新模型缓存 miss 可能阻塞事件循环最多约 2 秒（`utils/model.py:130`）。`count_tokens` 先访问根路径 `/tokenize`（3 秒 timeout），失败再尝试小输出调用，最后估算（`:279`、`:292`、`:336`）；长对话/compact 可能额外等待。仅静态发现，未证明本次延迟由此造成。
5. **代理和隧道。** ChatOpenAI 未显式设置自定义 httpx client 或 `trust_env=False`，应核对服务进程真实 HTTP(S)_PROXY/NO_PROXY。`.env` 中 `MODEL_GATEWAY_TRUST_ENV=true` 只证明本地网关设置；网关会使用 `trust_env`，超时 600 秒（`apps/models/gateway.py:40`、`:46`、`:435`）。GPU localhost、MoBase 主机直连、域名/隧道三层测量才能判断差额。
6. **正文过滤/缓冲。** SSE 出口把独立 reasoning 送为 thinking_delta（`stream_adapter.py:2249`、`:2259`），generation 正文进 ThinkMarkupFilter。当前中文首段可立即放行；首 20 个非空字符没有中日韩文字时可能等闭标签或回答结束（`nodes/generation.py:54`、`:94`）。旧 `prefix_limit=400` 只保留兼容参数，当前不按“攒够400字”放行。未闭合引用/图片有滑动窗口、500字符保护（`stream_adapter.py:2415`、`:2460`）。因此纯英文/纯数字合成题要同时测原始 content 与最终 SSE，避免把应用过滤解释成GPU慢。

## 4. 超时与重试边界

`get_model` 传 `timeout=45`、`request_timeout=60`、SDK `max_retries=2`（`utils/model.py:706`）。本机虚拟环境实际为 langchain-openai 1.1.10、langchain-core 1.6.6；用保留测试域名和 dummy key仅构造客户端，确认 `llm.request_timeout` 与 `root_client.timeout` 都是 **45秒**。这两个命名不是两层独立计时；生产SDK版本须另核验。HTTP read timeout通常限制一次读取的等待，不应解释为整轮总时限。

节点 `invoke_with_retry` 默认还允许首次+2次重试、指数退避，连接错误可熔断并切备用模型（`apps/langgraph/graphs/mobase_agent/llm_retry.py:104`、`:145`、`:173`）。`call_with_tools` 默认复用 v2 的同类 helper（`core/capabilities/llm_tools.py:506`）。在符合SDK重试条件且外层未提前回落时，一个节点最坏可出现3次外层×3次SDK尝试；generation 另有最多2次context截断重试（`nodes/generation.py:1612`）。这是静态上限，不是已观测的本次重试次数。统计必须带 HTTP attempt 数、节点 retry/fallback、错误码与模型真实身份，避免只看成功末次的快照。

## 5. 当日已保存样本：只是应用请求线索

来源：MoBase `.mission/20261013-vllm-followup/raw/e2e_vllm_fix_final-qwen_1f8bce-{01-Agent,02-Agent,03-Generation}.json`。只提取参数与计数；未复测，也未复制任何请求正文。

| 调用 | 消息数 | 消息正文字符数* | 工具数 | max_completion_tokens | 思考字符数 | 首正文秒 | 末正文秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Agent 1 | 5 | 24,999 | 14 | 8192 | 697 | 无正文（正常产出工具调用） | — |
| Agent 2 | 7 | 28,262 | 14 | 8192 | 65 | 1.3651 | 1.3651 |
| Generation | 2 | 38,638 | 0 | 5157 | 0 | 9.9022 | 38.5278 |

\* 不包含 tools schema，字符数不是 token 数。Generation 正文共2038字符、518个content分片；保存流无 usage。没有首 reasoning/首工具分片时间、vLLM queue/prefill 时间、GPU负载或同步服务器指标。因此不能由这些数字计算 decode tok/s、prefill tok/s，也不能将9.9022秒全归因GPU。Agent 1 content为空而finish_reason=tool_calls，是有效工具调用，不能算超时或吞吐为零。

## 6. 可复现验收方案

先在有权访问的服务器上确认 **当前进程** 的模型revision、vLLM镜像digest/version、容器参数、模板/parser、量化方式、GPU/TP分配、KV dtype、max_model_len、max_num_seqs、max_num_batched_tokens、speculative配置。只输出非 secret 白名单，启动命令中认证参数必须删除。同步记录生产running/waiting请求、GPU占用；线上有负载时不把结果称为独占硬件baseline。

按以下顺序，在相同模型、参数、合成内容和tokenizer下测量，每次保存run_id、配置hash、原始非敏感指标，不保存业务正文或认证header：

| 层 | 用例 | 要回答的问题 |
|---|---|---|
| A：GPU localhost | thinking=false；128/1k/4k/16k/32k input tokens（按当前上下文留输出余量）；固定256 output tokens；并发1起步 | 真正短/长输入的TTFT、decode、吞吐 |
| B：MoBase主机→同端点 | 相同body与并发；再走域名/隧道；显式直连和真实进程proxy配置各一组 | 网络、代理、网关导致的额外TTFT或中途断流 |
| C：MoBase真实SSE | 问候、合成单工具检索、多工具合成数据、长历史；保留实际思考/工具/生成策略 | 首thinking、首tool、首answer_chunk、总完成和流/历史一致性 |

每格先预热2次，再交错运行基线/候选各至少10次；报告p50/p95、失败率、实际输出长度分布。先并发1，业务可承受时再4/8；生产有负载则不扩大并发。prompt以服务端tokenizer实数分桶，实际可容纳长度由 max_model_len−输出预算决定，不能盲发32k导致overflow。冷prefix与热prefix分开，变换合成前缀测冷缓存且记录prefix cache状态。固定输出实验可用ignore_eos，但必须单列，真实业务验收保留正常stop，不能混成同一张成绩表。

客户端计时必须采用monotonic clock：t0发送、t_headers、t_first_any_delta、t_first_reasoning、t_first_content、t_first_tool、t_last_generated、t_done。stream_options.include_usage若支持则开启，分别记录prompt/completion/reasoning tokens；若缺usage，可用服务器指标或确切token ID计数，不能把SSE chunk数或中文字符数当token数。`(completion_tokens−1)/(t_last_generated−t_first_generated)`用于同一口径decode速率，reasoning计入生成；e2e吞吐和服务端prefill指标独立报告。单靠TTFT不能精确反推prefill tok/s，因为它包含网络、排队和第一步decode。

验收建议：同一组body中长输入TTFT p50/p95与decode改善，错误率/超时/重试不得恶化；MoBase首次可见正文和总完成须同步改善。业务质量至少检查工具选择、工具实参、检索覆盖、正常stop、表格完整、引用、thinking与正文分离、最终流/历史一致。若A改善而C未改善，定位工具/多步/过滤/网络剩余耗时；若B远慢于A，优先调查链路；若A长输入慢且队列为空，再调服务器prefill资源。不能用关闭思考或缩短正文得到的速度替代同策略对照。

## 7. 交接结论

目前有足够证据说明：MoBase实际负载包含长prompt、工具schema、多步思考和独立生成；SSE调用本身已流式。尚不能证明客户端是主要瓶颈，也不能证明GPU prefill/decode本身慢。服务端baseline应覆盖约1万至数万token级输入，并分别测思考开关与工具调用模板；保持模型名/API兼容可避免改MoBase接入。任何性能修改后必须再次从用户实际入口验收，不能只测GPU localhost。
