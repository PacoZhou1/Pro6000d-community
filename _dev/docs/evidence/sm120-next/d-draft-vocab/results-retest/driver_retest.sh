#!/bin/bash
R=/tmp/qwen-sm120-next-d; R2=$R/retest; cd $R
mkdir -p $R2/runs $R2/logs
export RUNS_DIR=$R2/runs INITIAL_SNAPSHOT=$R2/logs/initial-snapshot.json
L=$R2/logs/driver.log
log(){ echo "$(date -u +%FT%TZ) $*" >> $L; }
log "start"
python3 -c "import sys,json;sys.path.insert(0,'$R/bin');import common;json.dump(common.snapshot(),open('$R2/logs/initial-snapshot.json','w'),indent=1);print(common.embedding_check())" >> $L 2>&1
cp -n $R/bin/arm.py $R/bin/arm.py.session1; cp $R2/xclient.py $R2/xcompare.py $R2/arm.py $R/bin/ 2>>$L   # arm.py/xclient.py overwrite: original arm.py kept as arm.py.orig
arm(){ python3 bin/arm.py "$@" > $R2/logs/$2.out 2>&1; log "$2 rc=$?"; docker ps -a --format '{{.Names}}' | grep '^codex-next-d-' >> $L; }
arm --label d3-base --client d3 x
arm --label d3-65536 --subset subset_65536.json --client d3 x
arm --label d4-a1 --client perf
arm --label d4-b --subset subset_65536.json --client perf
arm --label d4-a2 --client perf
python3 bin/d3_compare.py $R2/runs $R2/logs/d3_compare.json > $R2/logs/d3_compare.out 2>&1; log "compare rc=$?"
python3 bin/analyze_d4.py $R2/runs $R2/logs/d4_analysis.json > $R2/logs/d4_analysis.out 2>&1; log "analyze rc=$?"
python3 bin/xcompare.py $R2/runs/d3-base/x.json $R2/runs/d3-65536/x.json $R2/logs/x_compare_64k.json > $R2/logs/x_compare_64k.out 2>&1; log "xcompare rc=$?"
arm --label d3-32768 --subset subset_32768.json --client d3
python3 bin/d3_compare.py $R2/runs $R2/logs/d3_compare_with32k.json > $R2/logs/d3_compare_with32k.out 2>&1; log "compare32 rc=$?"
tar czf $R2/retest-results.tgz -C $R2 --exclude='*.nsys-rep' logs runs xclient.py xcompare.py arm.py driver_retest.sh 2>>$L
log ALLDONE
