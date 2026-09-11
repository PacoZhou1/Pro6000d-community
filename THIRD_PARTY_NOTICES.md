# Attribution and redistribution boundaries

The original integration code and documentation in this export are provided under Apache-2.0. The full vLLM-derived replacement files retain their existing SPDX and contributor notices; see LICENSE. Their base is voipmonitor/vllm revision `60e72555e755e094e0c0c0ddfd65917514cc2151`. Adjacent `.diff` files show our changes against that base. `homelab_dense_fused.py` reuses the existing vLLM CuTe BF16 kernel rather than claiming authorship of that kernel.

B12X, vLLM, CUDA, PyTorch, and the GLM model remain the work of their respective authors. Weights and Docker images are not redistributed in this repository; their own terms apply. Numerical tables cite their original reports.

The public benchmark is fetched by immutable commit and SHA256. We did not find an explicit license through GitHub's license endpoint, so its source is not included in this export. The user-supplied chao HTML tool is also not redistributed: only our measured results, generated report and table screenshot are included. Fetching a tool is not a grant of redistribution rights.

## DeepSeek V4 Flash R33 materials

The R33 `fused_moe.m4-packed-r33.patch` is a 19-line modification against `b12x/moe/fused_moe/_impl.py` from [voipmonitor/b12x](https://github.com/voipmonitor/b12x) commit `59d51a36a942d56a9c36265855cdc7856fa7712e` (Apache-2.0). The public launcher targets the associated [voipmonitor/vllm](https://github.com/voipmonitor/vllm) commit `ae89131442359dc332d9c46009be3c1f8cdee0b4` (Apache-2.0). The repository contains neither B12X/vLLM source snapshots nor model weights, container images, writable layers, cache contents, complete inspection payloads, credentials, environment files, raw prompts, model answers, or reasoning traces.
