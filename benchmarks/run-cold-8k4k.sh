#!/bin/bash
set -euo pipefail
NAME=${1:-glm53-r30-dma9100}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
D=$ROOT/runs/cold-8k4k-$(date +%Y%m%dT%H%M%S)
mkdir -p "$D"
exec > "$D/run.log" 2>&1
curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null
docker cp "$ROOT/benchmarks/vendor/bench-0.6.1.py" "$NAME:/tmp/burst-community-0.6.2.py"
docker cp "$ROOT/benchmarks/bench-burst-cold-wrapper.py" "$NAME:/tmp/bench-burst-cold-wrapper.py"
cat /proc/sys/kernel/random/boot_id > "$D/boot-before.txt"
nvidia-smi --query-gpu=timestamp,index,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem --format=csv -l 1 > "$D/gpu.csv" &
g=$!;trap 'kill "$g" 2>/dev/null || true' EXIT
for cc in 1 8 16 32 64; do
 echo "C$cc $(date -Is)" > "$D/phase"
 docker exec "$NAME" timeout --signal=TERM --kill-after=10 900 python3 -u /tmp/bench-burst-cold-wrapper.py \
  --host 127.0.0.1 --port 8000 --model GLM-5.3-Flash-NVFP4 \
  --contexts 8k --concurrency "$cc" --request-count "$cc" --warmup-request-count 0 \
  --max-tokens 4096 --max-total-tokens 2000000 --token-targeting exact \
  --display-mode plain --no-hw-monitor --no-resume --skip-prefill --temperature 1 \
  --output "/tmp/burst-cold-c$cc.json" > "$D/c$cc.log" 2>&1
 docker cp "$NAME:/tmp/burst-cold-c$cc.json" "$D/c$cc.json"
 docker cp "$NAME:/tmp/burst-cold-c$cc.cold-metadata.json" "$D/c$cc.cold-metadata.json"
 python3 - "$D/c$cc.json" "$D/c$cc.cold-metadata.json" "$cc" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));rows=d['results'] or d['burst_results'];assert len(rows)==1
r=rows[0];cc=int(sys.argv[3]);m=json.load(open(sys.argv[2]))
print({k:r.get(k) for k in ['concurrency','benchmark_mode','aggregate_tps','output_tps_per_user_p50','ttft_p50','ttft_p99','input_seq_len_avg','completed_request_count','num_errors','max_running_reqs','max_queue_reqs','loop_detected']},flush=True)
assert r['num_errors']==0 and not r['loop_detected']
assert r['benchmark_mode']=='request-count' and r['completed_request_count']==cc
assert r['client_output_tokens']==4096*cc and r['server_output_tokens']==4096*cc
assert len(r['request_samples'])==cc and all(s['completed'] and s['output_tokens']==4096 and abs(s['input_tokens']-8192)<=4 for s in r['request_samples'])
assert len(m['records'])==cc and len({x['cache_salt_sha256'] for x in m['records']})==cc
# Finite cold waves include admission/queue/ramp-down; retain actual running count.
# Do not assert avg_running==requested, which would misclassify valid finite waves.
PY
done
cat /proc/sys/kernel/random/boot_id > "$D/boot-after.txt"
cmp "$D/boot-before.txt" "$D/boot-after.txt"
echo "DONE $(date -Is)" > "$D/phase"
