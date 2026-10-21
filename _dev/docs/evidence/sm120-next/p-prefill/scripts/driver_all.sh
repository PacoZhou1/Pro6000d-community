#!/bin/bash
R=/tmp/qwen-sm120-next-p; cd $R
export RUNS_DIR=$R/runs INITIAL_SNAPSHOT=$R/logs/initial-snapshot.json
./driver_e1.sh
./driver_e2.sh
python3 bin/e3_run.py a1 b a2 > logs/e3.out 2>&1
echo "$(date -u +%FT%TZ) ALLDONE" >> logs/driver.log
