# Sourced by phase scripts on the GPU host (non-root adaptation of run_a.sh)
IMAGE='sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
OLD=/home/llm/.local/share/qwen-gemm-epilogue
BASE=/tmp/qwen-sm120-next-a
WORK=$BASE/work
OUT=$BASE/results
NAME_PREFIX=codex-next-a
mkdir -p "$WORK" "$OUT"
verify_services() { python3 "$OLD/work/gemm-epilogue/verify_service_state.py" --initial-state "$OLD/initial-state.json" --output "$1" --functional; }
drun() {  # $1 suffix, $2 gpu|nogpu, rest = docker args after mounts
  local suffix="$1" mode="$2"; shift 2
  local gpu=(--runtime runc)
  [[ "$mode" == gpu ]] && gpu=(--gpus all --ipc host -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
  docker run --rm --name "$NAME_PREFIX-$suffix" --network none --memory=24g --memory-swap=24g --cpus=3 \
    "${gpu[@]}" --mount "type=bind,src=$WORK,dst=/work,readonly" --mount "type=bind,src=$OUT,dst=/results" "$@"
}
