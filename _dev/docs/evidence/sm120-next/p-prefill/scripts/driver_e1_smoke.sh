#!/bin/bash
R=/tmp/qwen-sm120-next-p; cd $R
mkdir -p $R/logs $R/runs
export RUNS_DIR=$R/runs INITIAL_SNAPSHOT=$R/logs/initial-snapshot.json
L=$R/logs/driver.log
log(){ echo "$(date -u +%FT%TZ) $*" >> $L; }
log "smoke start"
[ -f $INITIAL_SNAPSHOT ] || python3 -c "import sys,json;sys.path.insert(0,'$R/bin');import common;json.dump(common.snapshot(),open('$INITIAL_SNAPSHOT','w'),indent=1);print(common.embedding_check())" >> $L 2>&1
arm(){ python3 bin/arm.py "$@" > $R/logs/$2.out 2>&1; log "$2 rc=$?"; docker ps -a --format '{{.Names}}' | grep '^codex-next-p-' >> $L; }
arm --label e1-split-mtp4 --instr $R/instr --client split pfx --cargs "split=--lens 3000,18900,33000" --cargs "pfx=--lens 19000 --s4-limit 6"
log "smoke done"
