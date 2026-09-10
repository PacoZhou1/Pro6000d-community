#!/bin/bash
set -euo pipefail
IMAGE=localinferencelab/vllm@sha256:5f6fcbc681f20b7c052815ca17511d9fe789aea314a17723c202789dd7adc131
# Pull the registry manifest. Local uncompressed manifest equivalence is recorded in SOURCE_LOCK.json.
# Config sha256:1be0022694c3a2dcd9df9cacebf53e09b7a777815d7b5755ff7a54bd23b5249c
# Associated R30 registry manifest: sha256:5f6fcbc681f20b7c052815ca17511d9fe789aea314a17723c202789dd7adc131
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=${MODEL_DIR:?Set MODEL_DIR to an absolute local GLM-5.3-Flash-NVFP4 directory}
[[ "$MODEL_DIR" = /* ]] || { echo "MODEL_DIR must be absolute"; exit 2; }
D=${STATE_DIR:-/var/lib/glm53-lab}
[[ "$D" = /* ]] || { echo "STATE_DIR must be absolute"; exit 2; }
test -s "$MODEL_DIR/config.json"
mkdir -p "$D/cache"
# Validated R30 6000D profile. Startup refuses missing/changed mounted sources.
(cd "${ROOT}/patches" && sha256sum -c SHA256SUMS)
nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | awk '{n++;if($1!=350)bad=1} END{exit(bad||n!=4)}'
! docker inspect glm53-r30-dma9100 >/dev/null 2>&1 || { echo "Container already exists; use docker start after inspecting state"; exit 4; }
docker create --name glm53-r30-dma9100 --init --gpus all --network host --ipc host --restart no \
 --health-cmd="curl -fsS --max-time 5 http://127.0.0.1:8000/health || exit 1" --health-interval=30s --health-timeout=10s --health-start-period=600s --health-retries=3 \
 --ulimit memlock=-1:-1 --ulimit stack=67108864 \
 -v "${MODEL_DIR}":/models/GLM-5.3-Flash-NVFP4:ro \
 -v "$D/cache":/cache \
 -v "${ROOT}/patches/homelab_dense_fused.py":/opt/glm53-flash/vllm/vllm/model_executor/layers/homelab_dense.py:ro \
 -v "${ROOT}/patches/linear-utils.cute-trial.py":/opt/glm53-flash/vllm/vllm/model_executor/layers/utils.py:ro \
 -v "${ROOT}/patches/linear.cute-trial.py":/opt/glm53-flash/vllm/vllm/model_executor/layers/linear.py:ro \
 -e CUDAGRAPH_CAPTURE_SIZES="1 2 4 8 16 32 64 128 256" -e HOMELAB_GLM53_DENSE_CUTE=1 \
 -v "${ROOT}/patches/mhc-lifetime-v1.py":/opt/glm53-flash/vllm/vllm/models/deepseek_v4/nvidia/b12x.py:ro -e HOMELAB_MHC_GRAPH_WEAK_OUTPUTS=1 -e HOST=127.0.0.1 -e VLLM_SERVER_DEV_MODE=0 -e B12X_PCIE_DMA_PIECES=4 -e MAX_PARALLEL_PREFILLS=1 -e PREFILL_POLICY=round-robin -e DECODE_REFILL_TARGET=auto -e MODEL=/models/GLM-5.3-Flash-NVFP4 -e PORT=8000 -e TP=4 -e DCP=1 \
 -e SPECULATOR=mtp -e MTP_DEPTH=3 -e CACHE_MODE=vram -e KV_CACHE_QUANT=fp8_ds_mla \
 -e MAX_MODEL_LEN=262144 -e MAX_NUM_SEQS=64 -e MAX_NUM_BATCHED_TOKENS=8192 \
 -e GPU_MEMORY_UTILIZATION=0.90 -e PREFILL_SCHEDULE_INTERVAL=1 -e PREFILL_COMPUTE_SHARE=0.4 \
 -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=8KB -e VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=8KB -e VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=0 -e VLLM_B12X_MOE_FP4_FORCE_A16=1 -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e VLLM_GLM53_L2_PREFETCH_PERSIST_MB=32 -e NCCL_DEBUG=WARN -e NCCL_P2P_LEVEL=SYS -e NCCL_P2P_DISABLE=0 -e HF_HUB_OFFLINE=1 \
 "$IMAGE" --kv-cache-memory-bytes 17179869184 --override-generation-config '{"temperature":1,"top_p":0.95,"top_k":0}'
