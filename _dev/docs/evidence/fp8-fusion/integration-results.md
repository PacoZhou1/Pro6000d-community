# Real model dispatch verification

The selected V2 library passed exact output/scale comparisons, exceptional inputs,
CUDA Graph replay with changed input buffers, and 14 runtime integration/guard
checks before the live model witness. The runtime smoke retained the custom op
in a single dynamic FX graph when called first at M5 and then at M256, and the
Inductor executions matched the original op byte-for-byte.

The same immutable vLLM image and FP8 model then ran in normal compilation mode
with MTP4, FP8 KV, FLASHINFER, memory budget 0.70 and a loopback-only API. Nsight
captured one real long prefill: 130 candidate kernels, 65 at M16160 and 65 at
M2741. Their flat grid sizes are 549440 and 93194 (M * 34), matching V2. Six M1
calls retained the official kernel. All 130 candidate launches were correlated
with successful direct CUDA launches; none had a graphNodeId. Therefore this
capture proves actual model dispatch, **not model-level CUDA Graph replay**.
Large-prefill execution and the separately tested micro-level graph compatibility
must not be conflated.

Candidate kernel duration sum was 66.858301 ms. All 3673 kernels summed to
2283.011888 ms. These are one diagnostic capture's sums, not API wall time or
controlled baseline improvements. The separate no-profiler A1/B/A2 arms determine
request-level performance.

Evidence: `witness/dispatch-proof.json`, `witness/kernel-summary.json`,
`witness/requests.jsonl`, `witness/runtime.json`, `witness/server.log` and retained
ignored `witness/capture.sqlite` / `capture.1.nsys-rep`. The SQLite checksum and
integrity result are in the dispatch proof. `analyze_dispatch.py` reproduces the
correlation checks without requiring a GPU.

The unprofiled B engine also retained the new op in 21 generated Python compile-cache files. `b-cache-audit.json` records SHA256 and exact custom-op call lines, including the dynamic `(M,136)` scale layout. This is compilation evidence; the Nsight witness above independently proves actual model launches.
