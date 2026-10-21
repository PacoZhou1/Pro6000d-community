#!/bin/bash
R=/tmp/qwen-sm120-next-p; cd $R
export RUNS_DIR=$R/runs INITIAL_SNAPSHOT=$R/logs/initial-snapshot.json
L=$R/logs/driver.log
log(){ echo "$(date -u +%FT%TZ) $*" >> $L; }
arm(){ python3 bin/arm.py "$@" > $R/logs/$2.out 2>&1; log "$2 rc=$?"; docker ps -a --format '{{.Names}}' | grep '^codex-next-p-' >> $L; }
log "E2 start"
arm --label e2-nsys-fi --client nsysp
arm --label e2-fi-a1 --client len d3 --d3-passes 1
arm --label e2-flash-attn --attention-backend FLASH_ATTN --client len d3 --d3-passes 1
arm --label e2-triton --attention-backend TRITON_ATTN --client len d3 --d3-passes 1
arm --label e2-fi-a2 --client len d3 --d3-passes 1
log "E2 done"
