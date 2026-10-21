#!/usr/bin/env bash
set -uo pipefail
source /tmp/qwen-sm120-next-a/work/common_host.sh
awk '/Function :/{f=$3} /MMA/{match($0,/[HOQ]MMA[A-Za-z0-9_.]*/); c[f" "substr($0,RSTART,RLENGTH)]++} END{for(k in c) print c[k],k}' "$OUT/mma_rate_probe.sass" | sort -k2 > "$OUT/sass-mma-by-function.txt"
drun verify gpu --entrypoint /results/mma_rate_probe "$IMAGE" --verify --forms plain,kind,bs > "$OUT/mma-verify.jsonl" 2> "$OUT/mma-verify.stderr"; echo "verify rc=$?" > "$OUT/mma-verify.rc"
for i in 1 2; do
  nvidia-smi -q -d CLOCK,POWER > "$OUT/nvidia-smi-clock-power-before-probe$i.txt"
  drun probe$i gpu --entrypoint /results/mma_rate_probe "$IMAGE" --forms plain,kind,bs > "$OUT/mma-rate-run$i.jsonl" 2> "$OUT/mma-rate-run$i.stderr"; echo "rc=$?" > "$OUT/mma-rate-run$i.rc"
done
echo finished > "$OUT/probe.done"
