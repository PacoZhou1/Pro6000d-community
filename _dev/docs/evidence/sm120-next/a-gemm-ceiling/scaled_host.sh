#!/usr/bin/env bash
# usage: scaled_host.sh <tag> <args...>   -> results/scaled_mm_<tag>.{jsonl,stdout,stderr,rc,done}
set -uo pipefail
source /tmp/qwen-sm120-next-a/work/common_host.sh
tag="$1"; shift
rm -f "$OUT/scaled_mm_$tag.done"
nvidia-smi -q -d CLOCK,POWER > "$OUT/nvidia-smi-clock-power-before-scaled-$tag.txt"
nvidia-smi --query-gpu=memory.used,clocks.sm,power.draw --format=csv -lms 1000 > "$OUT/nvidia-smi-poll-scaled-$tag.csv" 2>&1 &
POLL=$!
drun "scaled-$tag" gpu --entrypoint python3 "$IMAGE" /work/scaled_mm_ceiling.py --out "/results/scaled_mm_$tag.jsonl" "$@" \
  > "$OUT/scaled_mm_$tag.stdout" 2> "$OUT/scaled_mm_$tag.stderr"
echo "rc=$?" > "$OUT/scaled_mm_$tag.rc"
kill $POLL 2>/dev/null
nvidia-smi -q -d CLOCK,POWER > "$OUT/nvidia-smi-clock-power-after-scaled-$tag.txt"
echo finished > "$OUT/scaled_mm_$tag.done"
