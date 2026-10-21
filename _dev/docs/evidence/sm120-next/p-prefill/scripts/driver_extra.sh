#!/bin/bash
# runs after the master driver: free-block-order control with chunk logging, and chunked-vs-single-chunk TTFT on identical prompts (A/B/A/B)
R=/tmp/qwen-sm120-next-p; cd $R
export RUNS_DIR=$R/runs INITIAL_SNAPSHOT=$R/logs/initial-snapshot.json
L=$R/logs/driver.log
log(){ echo "$(date -u +%FT%TZ) $*" >> $L; }
while ! grep -q ALLDONE $L; do sleep 30; done
arm(){ python3 bin/arm.py "$@" > $R/logs/$2.out 2>&1; log "$2 rc=$?"; docker ps -a --format '{{.Names}}' | grep '^codex-next-p-' >> $L; }
log "extra start"
arm --label e1-ctrl --instr $R/instr --client pfx --cargs "pfx=--scen S2,S3 --lens 19000 --shuffle"
arm --label e1-chunk-a1 --client split --cargs "split=--lens 18900,44000 --n 6"
arm --label e1-chunk-b1 --no-prefix-caching --client split --cargs "split=--lens 18900,44000 --n 6"
arm --label e1-chunk-a2 --client split --cargs "split=--lens 18900,44000 --n 6"
arm --label e1-chunk-b2 --no-prefix-caching --client split --cargs "split=--lens 18900,44000 --n 6"
log "extra done"
