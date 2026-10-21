#!/bin/bash
set -e
cd /tmp/qwen-sm120-next-f
mkdir -p bin logs instr/v2 aba
[ -f common.py ] && mv common.py bin/
[ -f f_run.py ] && mv f_run.py bin/
cp src/f_kernels.py src/sitecustomize.py instr/
cp /home/llm/.local/share/qwen-fp8-fusion/runtime/sitecustomize.py instr/v2_sitecustomize.py
cp /home/llm/.local/share/qwen-fp8-fusion/runtime/libfp8_silu_quant.so instr/v2/
sha256sum instr/v2/libfp8_silu_quant.so instr/f_kernels.py
ls /tmp/qwen-profile-experiment/profile_client.py /tmp/qwen-spec-experiment/quality_probe.py /tmp/qwen-spec-experiment/open_output_probe.py
cd bin && python3 - <<'P'
import json, common
s = common.snapshot()
json.dump(s, open('/tmp/qwen-sm120-next-f/logs/initial-snapshot.json', 'w'), indent=1)
print(s['vllm-embedding'])
P
docker ps -a --format '{{.Names}}' | grep codex || echo no-codex-containers
