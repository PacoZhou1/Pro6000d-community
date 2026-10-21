cd /tmp/qwen-sm120-next-p/runs/e2-nsys-fi/prof; ls
NSYS=/home/llm/.local/share/qwen-profiling/nsys-2026.5.1/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
for f in capture.1; do $NSYS export --type sqlite --force-overwrite true -o $f.sqlite $f.nsys-rep >/dev/null 2>&1; done
python3 - <<'P'
import sqlite3
c=sqlite3.connect('capture.1.sqlite')
rows=c.execute("select s.value, count(*), sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName group by s.value order by 3 desc limit 14").fetchall()
for r in rows: print(round(r[2],1), r[1], r[0][:330])
print('attn total ms', sum(r[2] for r in c.execute("select s.value, count(*), sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where s.value like '%Prefill%' group by s.value").fetchall()))
P
