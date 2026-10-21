#!/usr/bin/env bash
# Task A (FP8 GEMM ceiling) operator sequence. Run as root on the GPU host.
# NOT YET RUN. Never starts/stops/restarts any model service; the stopped 27B/9B/ASR
# containers and the running embedding container are only inspected (read-only).
#
# Usage: sudo ./run_a.sh [label]          (LOCK_CLOCKS=1 to try `nvidia-smi -lgc`)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE='sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
ROOT=/home/llm/.local/share/qwen-sm120-next
OLD=/home/llm/.local/share/qwen-gemm-epilogue
LABEL="${1:-a-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$LABEL" =~ ^[a-z0-9][a-z0-9-]{0,60}$ ]] || { echo "invalid label" >&2; exit 1; }
OUT="$ROOT/$LABEL"
WORK="$ROOT/work/a-gemm-ceiling"
NAME_PREFIX="codex-sm120-next-$LABEL"
LOCK_CLOCKS="${LOCK_CLOCKS:-0}"

mkdir -p "$WORK" "$OUT"
cp "$HERE"/mma_rate_probe.cu "$HERE"/scaled_mm_ceiling.py "$WORK"/
( cd "$WORK" && sha256sum mma_rate_probe.cu scaled_mm_ceiling.py ) > "$OUT/source.sha256"
sha256sum "$HERE/run_a.sh" > "$OUT/run_a.sha256"
printf '%s\n' "$IMAGE" > "$OUT/image.txt"

verify_services() {   # $1 = output json
  python3 "$OLD/work/gemm-epilogue/verify_service_state.py" \
    --initial-state "$OLD/initial-state.json" --output "$1" --functional
}

# Only our own owned experiment containers may ever be removed.
cleanup() {
  set +e
  [[ -n "${MON_PID:-}" ]] && kill "$MON_PID" 2>/dev/null
  for n in $(docker ps -a --format '{{.Names}}' | grep "^$NAME_PREFIX" || true); do
    docker rm -f "$n" >/dev/null 2>&1
  done
  if [[ "${LOCKED:-0}" == 1 ]]; then nvidia-smi -rgc > "$OUT/unlock-clocks.txt" 2>&1; fi
}
trap cleanup EXIT

# Refuse to run if another owned GPU experiment is active.
if docker ps --format '{{.Names}}' | grep -E '^codex-(fp8|gemm|sm120)-' >/dev/null; then
  echo "another owned GPU experiment is running; abort" >&2; exit 1
fi

verify_services "$OUT/services-before.json"

# Background embedding monitor (state only; no credentials are read here).
(
  while true; do
    printf '{"at_utc":"%s","state":"%s"}\n' "$(date -u +%FT%TZ)" \
      "$(docker inspect -f '{{.Id}} {{.State.Running}} {{.State.StartedAt}} {{.RestartCount}}' vllm-embedding 2>&1 | tr '"' "'")"
    sleep 15
  done
) > "$OUT/embedding-monitor.jsonl" &
MON_PID=$!

nvidia-smi -q -d CLOCK,POWER > "$OUT/nvidia-smi-clock-power-before.txt"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,persistence_mode --format=csv > "$OUT/nvidia-smi-query.csv"

LOCKED=0
if [[ "$LOCK_CLOCKS" == 1 ]]; then
  MAXSM="$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if nvidia-smi -lgc "$MAXSM,$MAXSM" > "$OUT/lock-clocks.txt" 2>&1; then LOCKED=1; echo "locked" >> "$OUT/lock-clocks.txt"
  else echo "lock failed (recorded; continuing unlocked)" >> "$OUT/lock-clocks.txt"; fi
fi

# docker run helper: $1 = suffix, $2 = "gpu"|"nogpu", rest = entrypoint + args
drun() {
  local suffix="$1" mode="$2"; shift 2
  local gpu=(--runtime runc)
  [[ "$mode" == gpu ]] && gpu=(--gpus all --ipc host -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
  docker run --rm --name "$NAME_PREFIX-$suffix" --network none --memory=24g --memory-swap=24g --cpus=3 \
    "${gpu[@]}" \
    --mount "type=bind,src=$WORK,dst=/work,readonly" \
    --mount "type=bind,src=$OUT,dst=/results" "$@"
}

# ---- build (no GPU): nvcc + SASS check; fall back to the ISA-spelled block-scale asm if ptxas rejects
BUILD='set -e
NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
CUOBJDUMP=$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)
STUBS=$(dirname "$(find /usr/local/cuda* -name libnvidia-ml.so -print -quit)")
$NVCC --version > /results/nvcc-version.txt
build() { $NVCC -O3 -std=c++17 -arch=sm_120a -lineinfo $1 -Xptxas -v -o /results/mma_rate_probe /work/mma_rate_probe.cu -L"$STUBS" -lnvidia-ml; }
if build "-DBS_VARIANT=0" 2> /results/build-v0.log; then echo 0 > /results/bs_variant.txt
elif build "-DBS_VARIANT=1" 2> /results/build-v1.log; then echo 1 > /results/bs_variant.txt
else echo failed > /results/bs_variant.txt; exit 1; fi
$CUOBJDUMP -sass /results/mma_rate_probe > /results/mma_rate_probe.sass
grep -E "QMMA|OMMA|HMMA" /results/mma_rate_probe.sass | sed -E "s/^[^;]*;?//" | sort | uniq -c | sort -rn > /results/sass-mma-forms.txt
cat /results/sass-mma-forms.txt'
drun build nogpu --entrypoint bash "$IMAGE" -c "$BUILD" > "$OUT/build.stdout" 2> "$OUT/build.stderr"

# ---- probe: verify, then two full runs (second run = repeat for clock/thermal drift)
drun verify gpu --entrypoint /results/mma_rate_probe "$IMAGE" --verify --forms plain,kind,bs > "$OUT/mma-verify.jsonl" 2> "$OUT/mma-verify.stderr"
for i in 1 2; do
  drun "probe$i" gpu --entrypoint /results/mma_rate_probe "$IMAGE" --forms plain,kind,bs > "$OUT/mma-rate-run$i.jsonl" 2> "$OUT/mma-rate-run$i.stderr"
done

# ---- cuBLASLt ceiling vs CUTLASS blockwise (31 pairs, warm + cold, NVML clocks)
nvidia-smi -q -d CLOCK,POWER > "$OUT/nvidia-smi-clock-power-mid.txt"
SCALED_ARGS=(--out /results/scaled_mm_ceiling.jsonl --pairs 31)
# --lock-clocks is not passed: the host-level lock above (LOCK_CLOCKS=1) is the one that counts.
drun scaled gpu --entrypoint python3 "$IMAGE" /work/scaled_mm_ceiling.py "${SCALED_ARGS[@]}" \
  > "$OUT/scaled_mm_ceiling.stdout" 2> "$OUT/scaled_mm_ceiling.stderr"

nvidia-smi -q -d CLOCK,POWER > "$OUT/nvidia-smi-clock-power-after.txt"
kill "$MON_PID" 2>/dev/null || true; MON_PID=
verify_services "$OUT/services-after.json"
docker ps -a --format '{{.Names}}\t{{.Status}}' > "$OUT/docker-ps-after.txt"
echo "done: $OUT"
