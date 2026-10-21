#!/bin/bash
# usage: nsys_kernels.sh ARMDIR   -> exports capture.1.sqlite next to the nsys-rep and prints kernel totals
cd $1/prof
NSYS=/home/llm/.local/share/qwen-profiling/nsys-2026.5.1/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
$NSYS export --type sqlite --force-overwrite true -o capture.1.sqlite capture.1.nsys-rep >/dev/null 2>&1
python3 - <<'P'
import sqlite3
c=sqlite3.connect('capture.1.sqlite')
tot=c.execute("select sum(end-start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
print('total kernel ms',round(tot,1))
for r in c.execute("select s.value,count(*),sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName group by s.value order by 3 desc limit 40"):
    print(round(r[2],2),r[1],r[0][:110])
P
