#!/bin/bash
# usage: run_in_image.sh NAME SCRIPT_REL_TO_/tmp/qwen-sm120-next-f [args...]   (runs on the GPU host)
# One experiment container at a time; container name must start with codex-f-.
set -e
NAME=$1; shift
SCRIPT=$1; shift
IMG=sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
if docker ps --format '{{.Names}}' | grep -q '^codex-'; then echo "another codex container running" >&2; exit 2; fi
docker run --rm --user $(id -u):$(id -g) --name $NAME --gpus all --ipc host --network none --memory=24g \
  -v /tmp/qwen-sm120-next-f:/work -e PYTHONPATH=/work/src -e TRITON_CACHE_DIR=/work/triton_cache -e HOME=/tmp \
  --entrypoint python3 $IMG /work/$SCRIPT "$@"
