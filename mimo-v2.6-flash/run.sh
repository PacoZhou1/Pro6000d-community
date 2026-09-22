#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the absolute MiMo-V2.6-Flash-RL weights directory}"
[[ "$MODEL_PATH" = /* && -d "$MODEL_PATH" ]] || { echo "MODEL_PATH must be an existing absolute directory" >&2; exit 1; }
exec docker run -d --name "${NAME:-mimo-v26-flash-rl}" --gpus all \
 --network host --ipc host --shm-size 32g --restart unless-stopped \
 --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --ulimit stack=67108864:67108864 \
 -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e OMP_NUM_THREADS=2 -e PYTHONHASHSEED=0 \
 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
 -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo -e NCCL_P2P_LEVEL=SYS -e NCCL_P2P_DISABLE=0 \
 -e VLLM_B12X_MOE_FP4_FORCE_A16=1 -e HOMELAB_MIMO_FP8_TAIL=1 \
 -e HOMELAB_MIMO_DENSE_CUTE=1 -e HOMELAB_MIMO_B12X_FP8=1 -e HOMELAB_MIMO_DRAFT_SCALE=1.25 \
 -v "$MODEL_PATH:/models/MiMo-V2.6-Flash-RL:ro" \
 "${IMAGE:-mimo-v26-flash-6000d:local}" /models/MiMo-V2.6-Flash-RL \
 --served-model-name MiMo-V2.6-Flash-RL --host 127.0.0.1 --port "${PORT:-8000}" \
 --tensor-parallel-size 4 --trust-remote-code --gpu-memory-utilization 0.95 \
 --max-model-len auto --max-num-seqs 64 --max-num-batched-tokens 8192 \
 --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
 --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
 --generation-config vllm --moe-backend b12x
