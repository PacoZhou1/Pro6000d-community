#!/bin/bash
R=/tmp/qwen-sm120-next-d; cd $R
L=$R/logs/driver.log
log(){ echo "$(date -u +%FT%TZ) $*" >> $L; }
NSYS=/home/llm/.local/share/qwen-profiling/nsys-2026.5.1/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
log "nsys export"
for c in 1 2; do
  f=runs/smoke-nsys-32k/prof/capture.$c
  $NSYS export --type sqlite --force-overwrite true -o $f.sqlite $f.nsys-rep > logs/nsys-export-$c.log 2>&1
  python3 inputs/d0_lm_head_steps.py $f.sqlite --min-us 100 --out runs/smoke-nsys-32k/d0_capture$c.json > logs/d0-capture$c.log 2>&1
done
log "d0 done"
python3 bin/arm.py --label d3-base --client d3 > logs/d3-base.out 2>&1; log "d3-base rc=$?"
for k in 32768 16384 65536; do
  python3 bin/arm.py --label d3-$k --subset subset_$k.json --client d3 > logs/d3-$k.out 2>&1; log "d3-$k rc=$?"
done
python3 bin/d3_compare.py runs logs/d3_compare.json > logs/d3_compare.out 2>&1; log "compare rc=$?"
python3 bin/arm.py --label d4-a1 --client perf > logs/d4-a1.out 2>&1; log "d4-a1 rc=$?"
python3 bin/arm.py --label d4-b --subset subset_32768.json --client perf > logs/d4-b.out 2>&1; log "d4-b rc=$?"
python3 bin/arm.py --label d4-a2 --client perf > logs/d4-a2.out 2>&1; log "d4-a2 rc=$?"
python3 bin/analyze_d4.py runs logs/d4_analysis.json > logs/d4_analysis.out 2>&1; log "analyze rc=$?"
tar czf $R/results.tgz --exclude='*.nsys-rep' --exclude='*.sqlite' logs runs bin inputs/coverage.json src/patched 2>>$L
log ALLDONE
