# Combined FP8 release: fixed synthetic API clients

These scripts are local source only. The parent controls SSH, service start/stop, GPU use, and the five-arm O1/A1/B/A2/O2 sequence. No model or server was run while preparing them. O is the old 0.27 MTP2 image, A is stock 0.31 MTP4, and B is the combined FP8 image; the controller must independently verify the **same 12 GiB KV pool**, model/weight identity, isolation, and arm order. Do not use these scripts to infer broad business quality.

`natural_probe.py` adds four fixed synthetic prompts, each sent twice in both non-streaming and SSE modes (16 requests per arm). It queries `127.0.0.1:18000/v1/models`; pass `--model` if more than one ID is served. It reads the API key from `PRO6000D_COMBO_API_KEY` or an explicitly named service container and never writes it. For an unauthenticated test server, pass `--allow-no-key`; this is accepted only with an explicit loopback IP and omits the Authorization header. Non-streaming requests ask for `return_token_ids`; SSE requests ask for terminal usage and accumulate text/reasoning/tool-call argument fragments. A server that omits requested token IDs is recorded as `token_ids_status=unavailable`, never as supported; missing IDs alone do not lower the natural-EOS or format checks. Natural-EOS normal and thinking prompts have ample `max_tokens` and no `ignore_eos`. The JSON case checks fixture fields; the tool case **only asks the model to generate arguments** and never invokes `get_metrics`. Each record includes the exact request payload/hash, usage, finish/truncation, text/reasoning hashes and full response, available token IDs, repeated-run consistency inputs, and conservative obvious-loop flags. `acceptance_pass` requires valid completion, fixture checks, and no obvious-loop flag; `failure_reasons` identifies each failed gate, and any failed request makes the CLI exit 2. A normal stop plus fixture checks is an observation, not a claim of general correctness.

The first B smoke (`smoke-v1`) reached `length` at the former 2,048-token thinking budget while drafting its six-step response; its failed raw evidence remains separate. This revision raises only `thinking_long.max_tokens` to 8,192. The revised payload and script hash must be used consistently for the formal five-arm run; `smoke-v1` is not a formal arm.

```sh
PRO6000D_COMBO_API_KEY="$KEY_ALREADY_HELD_BY_CONTROLLER" \
python3 /work/clients/natural_probe.py --tag o1 \
  --base http://127.0.0.1:18000 --model qwen3.6-27b-fp8 \
  --output /profiles/o1/acceptance.jsonl
```

The existing `profile_client.py` remains the performance source: one warmup plus formal prefill ×12, decode ×6, thinking ×6 per arm, with `--mode baseline --profiler-installed none` and fixed request builder. Save `prefill.jsonl`, `decode.jsonl`, `thinking.jsonl`, and `acceptance.jsonl` under each of `o1,a1,b,a2,o2`. The controller must preserve identical label/seed logic in all arms; don't alter request payloads between versions.

Run the offline analyzer after all raw data has been copied locally:

```sh
python3 _dev/docs/evidence/fp8-release-20261015/clients/analyze_five_arm.py \
  --root /path/to/copied/five-arm-results \
  --output /tmp/five-arm-review.json
```

For each case/index it checks payload hashes, exact prompt/completion token workload, one target request in the metrics window, unprofiled timing scope, prefix-cache hits, and embedding counter changes. It excludes a pair from quantitative comparison when any gate is unknown or fails, and reports the reason. It computes B versus the mean of A1/A2 and separately B versus the mean of O1/O2. Prefill uses `mean_baseline_TTFT/B_TTFT`; decode and thinking use `B_tps/mean_baseline_tps` (>1 favors B). Medians are across matched request indices, with min/max; no bootstrap is presented as time-window significance. Accepted draft tokens are recorded per arm but MTP2/MTP4 counts are not interpreted as equivalent denominators. Acceptance results report self-consistency, EOS, truncation, obvious loop flags, token IDs, and cross-arm hash equality without requiring identical free-form wording. A noncomparable report is still written and exits 2.

The existing performance client sets `ignore_eos=True`, so its decode and thinking requests deliberately run to `max_tokens`; the new acceptance client separately tests natural EOS. Quantitative performance, natural completion, synthetic fixture behavior, and model-level correctness are distinct conclusions.
