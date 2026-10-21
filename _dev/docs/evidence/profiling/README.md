# Qwen3.6-27B GPU profiling evidence

This experiment uses the fixed vLLM 0.31.0 image and the previously verified FP8
target weights. Temporary services bind only to `127.0.0.1:18000`, with no
production API key in the traced process arguments. Embedding remains running.

`profile_control.py` launches one MTP or DFlash service. CUDA captures use Nsight
Systems 2026.5.1, CUDA graph node tracing, and the vLLM profiling endpoints.
`profile_client.py` sends deterministic synthetic requests, warms each workload,
checks complete streaming responses and isolated token counters, and records
embedding counters before and after each request. `monitor_embedding.py` checks
embedding health and process identity every 30 seconds without inference load.

For prefill, capture begins before a fresh long-input request with one output
token. It therefore includes sampling and small post-forward work, which must be
identified in the trace. For decode, capture begins after the first streamed text.
The actual NVTX ranges determine whether a trace window excludes prefill. MTP
verification and draft kernels remain part of the decode workload. Generic
kernel names alone cannot identify target versus draft ownership.

Performance controls must run with `--profiler none` and
`--profiler-installed none`, separately from the diagnostic captures. Captured
request timings are not used as unprofiled speed claims. Cache hits, incomplete
requests, other model activity and compilation must be disclosed or excluded.

Raw Nsight reports and exported SQLite files are retained under the GPU host's
`/home/llm/.local/share/qwen-profiling/` directory. Local `raw/` is ignored by
Git. Analysis JSON, request JSONL, runtime provenance, checksums and scripts are
versioned. Shares of summed kernel durations must not be described as shares of
end-to-end latency; interval union and observed GPU span are separate measures.

Official tooling references: [vLLM profiling](https://docs.vllm.ai/en/v0.31.0/contributing/profiling/),
[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
