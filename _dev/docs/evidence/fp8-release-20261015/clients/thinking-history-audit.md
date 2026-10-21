# Thinking throughput: current combined image versus historical D4

This is a read-only audit of the formal six `thinking` requests in the current combined-image `measure-v2` and the earlier D4 draft-vocabulary retest. It does not make the two sessions a new A/B experiment.

| Evidence | Current combined image | Historical D4 64K arm |
|---|---:|---:|
| Raw file | [`measure-v2/b/thinking.jsonl`](../runs/measure-v2/b/thinking.jsonl) | [`d4-b/perf.jsonl`](../../sm120-next/d-draft-vocab/results-retest/runs/d4-b/perf.jsonl) |
| Raw SHA-256 | `53430a33c3de15fdd37b4be7e6a98fd6be35406066b390e12b770e5b0bccb01b` | `5339c66a4ec76de1097679845f331d6c9273ff1e1a2832b15c4dafce32a03b5f` |
| Source client SHA-256 | `711dc49e72a93a62ee65c351a386d0ed7e51b913111ab8a38eee59e967c6cbea` | Same exact source bytes: current [`profile_client.py`](../control/profile_client.py) and historical [`profile_client.py`](../../sm120-next/p-prefill/scripts/profile_client.py) |
| Six formal payload hashes | **6/6 identical**, index by index | Same |
| Prompt/completion usage | **6/6 identical**: prompt `[4799, 4794, 4793, 4771, 4795, 4761]`, completion `1024` each | Same |
| Request mode and estimator | `enable_thinking=True`, `ignore_eos=True`, streamed usage; `(completion_tokens-1)/(last_text_s-first_text_s)` | Same code and semantics; unprofiled scope in both |
| Validity and isolation | 6/6 `valid`, `length`, prefix hits 0; no embedding counter change in these six windows | 6/6 `valid`, `length`, prefix hits 0; D4-B also no embedding counter change |
| Decode estimate median | **133.6374 tok/s** | **148.1013 tok/s** |
| Accepted draft tokens, six-request sum / median | `4245 / 707.5` | `4364 / 737` |
| Draft tokens, six-request sum / median | `7600 / 1270` | `7128 / 1146` |
| Aggregate accepted/drafted counter ratio | `0.5586` | `0.6122` |
| Output hashes | Content empty in all six (same empty hash); reasoning hashes **0/6 match D4-B** | D4-B reasoning hashes match D4-A1 and D4-A2 at all six indices |

The matching request hashes, usage, and byte-identical client show that **the current 133.64 and historical 148.10 used the same six synthetic thinking prompts, thinking mode, 1,024-token work, and throughput estimator**. The current stream generated different reasoning for every request and recorded fewer accepted draft tokens with more drafted tokens. These are observations, not a diagnosis of why the output or speed changed. The runs occurred in separate sessions and stacks; no current stock/draft-only arm was run alongside the combined image, so the 14.46 tok/s cross-session gap is not a paired speedup or kernel effect. The historical D4 A1/A2 thinking medians were 130.45/130.97 tok/s, but they likewise are not this run's baseline.

Per-index audit (`accepted/drafted` are per-request metric deltas):

| Index | Payload SHA prefix | Prompt tokens | D4-B tok/s | Current tok/s | D4-B accepted/drafted | Current accepted/drafted | Reasoning hash match |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `4c79223454d8` | 4799 | 128.9448 | 137.6669 | 694/1316 | 717/1232 | no |
| 2 | `5b84ca476ac3` | 4794 | 154.8186 | 130.4807 | 750/1096 | 698/1300 | no |
| 3 | `7ec9afa484a7` | 4793 | 146.2958 | 139.9717 | 734/1160 | 720/1212 | no |
| 4 | `293388290122` | 4771 | 149.9068 | 136.7940 | 740/1132 | 717/1240 | no |
| 5 | `bb2d9bf0763a` | 4795 | 130.9473 | 130.0674 | 703/1296 | 698/1304 | no |
| 6 | `1363ad929312` | 4761 | 150.4044 | 129.2958 | 743/1128 | 695/1312 | no |

The historical D4 source summary is [`results-retest/summary.md`](../../sm120-next/d-draft-vocab/results-retest/summary.md). Current single-arm validity and per-request hashes are also recorded in [`combined-analysis.json`](../combined-analysis.json). Neither source supports substituting 148.10 tok/s for the current observed 133.64 tok/s.
