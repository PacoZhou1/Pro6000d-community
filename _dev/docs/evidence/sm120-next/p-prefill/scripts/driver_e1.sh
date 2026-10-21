#!/bin/bash
R=/tmp/qwen-sm120-next-p; cd $R
export RUNS_DIR=$R/runs INITIAL_SNAPSHOT=$R/logs/initial-snapshot.json
L=$R/logs/driver.log
log(){ echo "$(date -u +%FT%TZ) $*" >> $L; }
arm(){ python3 bin/arm.py "$@" > $R/logs/$2.out 2>&1; log "$2 rc=$?"; docker ps -a --format '{{.Names}}' | grep '^codex-next-p-' >> $L; }
log "E1 start"
arm --label e1-base1 --client prefill gen pfx
arm --label e1-nopc --no-prefix-caching --client prefill gen split --cargs "split=--lens 18900,44000"
arm --label e1-base2 --client prefill gen
arm --label e1-nomtp --no-spec --client prefill gen pfx
arm --label e1-split-nomtp --no-spec --instr $R/instr --client split
arm --label e1-junction --extra-arg=--enable-mamba-shared-prefix-checkpoint --client prefill pfx --cargs "pfx=--scen S2,S4 --lens 19000 --s4-limit 14"
arm --label e1-modeall --extra-arg=--mamba-cache-mode --extra-arg=all --client prefill pfx --cargs "pfx=--scen S2,S4 --lens 19000 --s4-limit 14"
log "E1 done"
