import sqlite3, sys, json, collections, re
def totals(db):
    c = sqlite3.connect(db)
    d = collections.OrderedDict()
    for name, n, ms in c.execute("select s.value,count(*),sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName group by s.value"):
        key = re.sub(r'\(.*', '', name)[:100] if not name.startswith('void cutlass::device_kernel') else 'cutlass fp8 gemm'
        key = re.sub(r'_(\d+)$', '', key)
        a = d.setdefault(key, [0, 0.0]); a[0] += n; a[1] += ms
    mem = c.execute("select count(*),sum(end-start)/1e6 from CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()
    tot = c.execute("select sum(end-start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    span = c.execute("select max(end)-min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] / 1e6
    return d, tot, mem, span
if __name__ == '__main__':
    base, bt, bm, bs = totals(sys.argv[1]); cand, ct, cm, cs = totals(sys.argv[2])
    print(f"kernel sum base {bt:.1f} cand {ct:.1f} ms ; memcpy base n={bm[0]} {bm[1]:.2f}ms cand n={cm[0]} {cm[1]:.2f}ms ; span base {bs:.1f} cand {cs:.1f}")
    keys = sorted(set(base) | set(cand), key=lambda k: -abs(cand.get(k, [0, 0])[1] - base.get(k, [0, 0])[1]))
    for k in keys[:25]:
        b = base.get(k, [0, 0]); c = cand.get(k, [0, 0])
        print(f"{c[1]-b[1]:+8.2f} ms  base {b[0]:4d}/{b[1]:7.2f}  cand {c[0]:4d}/{c[1]:7.2f}  {k}")
