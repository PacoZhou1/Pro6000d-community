#!/bin/bash
R=/tmp/qwen-sm120-next-d; cd $R
L=$R/logs/driver2.log
log(){ echo "$(date -u +%FT%TZ) $*" >> $L; }
while pgrep -f bin/driver.sh >/dev/null; do sleep 10; done
for K in 65536 16384; do
  t=$([ $K = 65536 ] && echo 64 || echo 16)
  python3 bin/arm.py --label d4-$t-a1 --client perf > logs/d4-$t-a1.out 2>&1; log "d4-$t-a1 rc=$?"
  python3 bin/arm.py --label d4-$t-b --subset subset_$K.json --client perf > logs/d4-$t-b.out 2>&1; log "d4-$t-b rc=$?"
  python3 bin/arm.py --label d4-$t-a2 --client perf > logs/d4-$t-a2.out 2>&1; log "d4-$t-a2 rc=$?"
  python3 bin/analyze_d4.py runs logs/d4_analysis_$t.json d4-$t-a1 d4-$t-b d4-$t-a2 > logs/d4_analysis_$t.out 2>&1; log "analyze $t rc=$?"
done
tar czf $R/results2.tgz --exclude='*.nsys-rep' --exclude='*.sqlite' logs runs 2>>$L
log ALLDONE2
