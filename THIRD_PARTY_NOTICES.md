# Attribution and redistribution boundaries

The original integration code and documentation in this export are provided under Apache-2.0. The full vLLM-derived replacement files retain their existing SPDX and contributor notices; see LICENSE. Their base is voipmonitor/vllm revision `60e72555e755e094e0c0c0ddfd65917514cc2151`. Adjacent `.diff` files show our changes against that base. `homelab_dense_fused.py` reuses the existing vLLM CuTe BF16 kernel rather than claiming authorship of that kernel.

B12X, vLLM, CUDA, PyTorch, and the GLM model remain the work of their respective authors. Weights and Docker images are not redistributed in this repository; their own terms apply. Numerical tables cite their original reports.

The public benchmark is fetched by immutable commit and SHA256. We did not find an explicit license through GitHub's license endpoint, so its source is not included in this export. The user-supplied chao HTML tool is also not redistributed: only our measured results, generated report and table screenshot are included. Fetching a tool is not a grant of redistribution rights.
