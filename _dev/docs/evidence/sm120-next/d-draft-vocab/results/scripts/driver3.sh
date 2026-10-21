#!/bin/bash
R=/tmp/qwen-sm120-next-d; cd $R
while pgrep -f bin/driver2.sh >/dev/null; do sleep 10; done
mkdir -p runs/tf-base
python3 bin/arm.py --label tf-base --client tf > logs/tf-base.out 2>&1; echo "tf rc=$? $(date -u)" >> logs/driver3.log
tar czf $R/results3.tgz --exclude='*.nsys-rep' --exclude='*.sqlite' logs runs 2>>logs/driver3.log
echo ALLDONE3 >> logs/driver3.log
