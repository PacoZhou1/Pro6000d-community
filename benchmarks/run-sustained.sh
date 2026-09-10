#!/bin/bash
set -euo pipefail
NAME=${1:-glm53-r30-dma9100}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
D=$ROOT/runs/sustained-$(date +%Y%m%dT%H%M%S)
mkdir -p "$D"
exec > "$D/run.log" 2>&1
curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null
[ "$(docker inspect "$NAME" --format '{{.State.Running}}')" = true ]
docker inspect "$NAME" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -qx 'MAX_NUM_SEQS=64'
nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | awk '$1!=350{bad=1} END{exit bad}'
cat /proc/sys/kernel/random/boot_id > "$D/boot-before.txt"
docker cp "$ROOT/benchmarks/vendor/bench-0.6.1.py" "$NAME:/tmp/capacity-bench.py"
nvidia-smi --query-gpu=timestamp,index,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem --format=csv -l 1 > "$D/gpu.csv" &
g=$!;trap 'kill "$g" 2>/dev/null || true' EXIT
for cc in 1 8 16 32 64; do
 echo "C$cc $(date -Is)" > "$D/phase"
 docker exec "$NAME" timeout --signal=TERM --kill-after=10 300 python3 -u /tmp/capacity-bench.py \
  --host 127.0.0.1 --port 8000 --model GLM-5.3-Flash-NVFP4 \
  --contexts 0 --concurrency "$cc" --duration 30 --max-tokens 8192 --max-total-tokens 2000000 \
  --display-mode plain --no-hw-monitor --no-resume --skip-prefill \
  --decode-warmup-seconds 15 --cell-warmup-timeout-seconds 180 --temperature 1 \
  --output "/tmp/capacity-c$cc.json" > "$D/c$cc.log" 2>&1
 docker cp "$NAME:/tmp/capacity-c$cc.json" "$D/c$cc.json"
 python3 - "$D/c$cc.json" "$cc" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]));r=j['results'][0];cc=int(sys.argv[2])
print({k:r.get(k) for k in ['concurrency','effective_concurrency','aggregate_tps','server_steps_per_s','server_accept_len_effective','num_errors','loop_detected','capacity_limited','max_running_reqs','max_queue_reqs']},flush=True)
assert r['num_errors']==0 and not r['loop_detected'] and not r['capacity_limited']
assert r['effective_concurrency']==cc and r['max_running_reqs']==cc,'requested clients were not actual GPU concurrency'
PY
done
cat /proc/sys/kernel/random/boot_id > "$D/boot-after.txt"
cmp "$D/boot-before.txt" "$D/boot-after.txt"
echo "DONE $(date -Is)" > "$D/phase"
