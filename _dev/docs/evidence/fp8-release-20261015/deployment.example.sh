#!/usr/bin/env bash
# Explicit opt-in launcher. This file is delivered, not automatically executed.
# Reproduces the isolated benchmark endpoint; does not control other services.
set -euo pipefail

IMAGE="${PRO6000D_IMAGE:-pro6000d/qwen36-fp8:source-build}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to your local Qwen3.6-27B-FP8 directory}"
NAME='pro6000d-qwen36-fp8-release'

test -d "$MODEL_DIR"
docker image inspect "$IMAGE" >/dev/null
if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "Container $NAME already exists; inspect it before starting a new one." >&2
  exit 1
fi

exec docker run --detach --name "$NAME" --gpus all --ipc=host \
  --network=host --restart=no --memory=40g --memory-swap=40g \
  --ulimit memlock=-1:-1 \
  --mount "type=bind,src=$MODEL_DIR,dst=/model,readonly" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env VLLM_USE_V2_MODEL_RUNNER=1 --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env "PRO6000D_RELEASE_V2=${PRO6000D_RELEASE_V2:-1}" \
  --env "PRO6000D_RELEASE_F=${PRO6000D_RELEASE_F:-1}" \
  --env "PRO6000D_RELEASE_DRAFT64K=${PRO6000D_RELEASE_DRAFT64K:-1}" \
  "$IMAGE" /model --served-model-name qwen3.6-27b-fp8 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.70 \
  --kv-cache-memory-bytes 12884901888 --max-num-seqs 16 \
  --max-model-len 262144 --max-num-batched-tokens 32768 \
  --quantization fp8 --kv-cache-dtype fp8 --attention-backend FLASHINFER \
  --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --host 127.0.0.1 --port 18000
