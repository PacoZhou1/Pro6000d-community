#!/usr/bin/env python3
"""Fetch small, license-clear corpus samples from ModelScope datasets ONLY.

Run with the venv that has modelscope + pyarrow:
    download/venv/bin/python d1_fetch_corpus.py
Only hosts under modelscope.cn are contacted (www.modelscope.cn API and its
cdn-lfs-cn-*.modelscope.cn redirect target). HTTP Range reads are used so that
multi-GB shards are never downloaded whole; parquet is read row group by row
group through a ranged file object until the text budget is met.
Output: download/corpus/<name>.jsonl ({"text": ...}) and corpus_manifest.json
(dataset id, file, license from the dataset README, revision, LFS oid parsed
from the redirect URL, bytes fetched, SHA256 of the extracted sample).
"""
import hashlib, io, json, os, re, sys, urllib.parse, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download", "corpus")
MB = 1 << 20
API = "https://www.modelscope.cn/api/v1/datasets/{ds}/repo?Revision=master&FilePath={fp}"

# name, domain, dataset, file, kind, text keys, extracted-text budget (MB)
SOURCES = [
    ("zh_wiki",      "zh",   "AI-ModelScope/wikipedia-cn-20230720-filtered", "wikipedia-cn-20230720-filtered.jsonl", "jsonl", None, 50),
    ("zh_web",       "zh",   "opencsg/chinese-fineweb-edu", "Skypile/00000.parquet", "parquet", None, 50),
    ("en_web",       "en",   "AI-ModelScope/fineweb-edu", "data/CC-MAIN-2014-35/train-00000-of-00015.parquet", "parquet", ["text"], 80),
    ("code_python",  "code", "AI-ModelScope/starcoderdata", "python/train-00000-of-00059.parquet", "parquet", ["content"], 40),
    ("code_js",      "code", "AI-ModelScope/starcoderdata", "javascript/train-00000-of-00065.parquet", "parquet", ["content"], 25),
    ("code_alpaca",  "code", "AI-ModelScope/CodeAlpaca-20k", "CodeAlpaca-20k.jsonl", "jsonl", None, 8),
    ("reason_zh",    "reasoning_zh", "AI-ModelScope/Chinese-DeepSeek-R1-Distill-data-110k", "distill_r1_110k.jsonl", "jsonl", None, 40),
    ("reason_en",    "reasoning_en", "AI-ModelScope/Bespoke-Stratos-17k", "data/train-00000-of-00001.parquet", "parquet", None, 40),
]


def resolve(ds, fp):
    """Return (final_url, meta) without following the redirect blindly."""
    url = API.format(ds=ds, fp=urllib.parse.quote(fp))

    class NoRedir(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    op = urllib.request.build_opener(NoRedir)
    try:
        op.open(urllib.request.Request(url, method="HEAD"), timeout=60)
        return url, {}
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302):
            raise
        loc = e.headers["Location"]
    host = urllib.parse.urlparse(loc).hostname or ""
    assert host.endswith("modelscope.cn"), f"refusing non-ModelScope host {host}"
    q = urllib.parse.parse_qs(urllib.parse.urlparse(loc).query)
    m = re.search(r"/lfs-objects/([0-9a-f]{2})/([0-9a-f]{2})/([0-9a-f]+)", loc)
    meta = {"revision": q.get("revision", [None])[0], "host": host,
            "lfs_oid_path_fragment": "".join(m.groups()) if m else None}
    return loc, meta


class Ranged(io.RawIOBase):
    def __init__(self, ds, fp):
        self.ds, self.fp, self.pos, self.fetched = ds, fp, 0, 0
        self.url, self.meta = resolve(ds, fp)
        self.size = self._get(0, 0)[1]
        self.cache = {}

    def _get(self, a, b):
        for attempt in range(2):
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    total = int(r.headers.get("Content-Range", "/0").split("/")[-1] or 0)
                    return r.read(), total
            except urllib.error.HTTPError as e:
                if e.code in (401, 403) and attempt == 0:
                    self.url, _ = resolve(self.ds, self.fp)
                    continue
                raise

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        out, end = bytearray(), self.pos + n
        while self.pos < end:
            blk = self.pos // (4 * MB)
            if blk not in self.cache:
                a = blk * 4 * MB
                data, _ = self._get(a, min(a + 4 * MB, self.size) - 1)
                self.cache[blk] = data
                self.fetched += len(data)
                if len(self.cache) > 8:
                    self.cache.pop(next(iter(self.cache)))
            data = self.cache[blk]
            off = self.pos - blk * 4 * MB
            take = data[off:off + (end - self.pos)]
            out += take
            self.pos += len(take)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def texts_of(rec, keys):
    if keys:
        return [str(rec[k]) for k in keys if rec.get(k)]
    out = []
    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for x in v.values(): walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v: walk(x)
    walk(rec)
    return ["\n".join(out)] if out else []


def license_of(ds):
    try:
        req = urllib.request.Request(API.format(ds=ds, fp="README.md"))
        txt = urllib.request.urlopen(req, timeout=60).read(4000).decode("utf8", "replace")
        m = re.search(r"^license:\s*(\S+)", txt, re.M)
        return m.group(1) if m else "see dataset README"
    except Exception as e:
        return f"unknown ({e})"


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for name, domain, ds, fp, kind, keys, budget in SOURCES:
        path = os.path.join(OUT, name + ".jsonl")
        if os.path.exists(path):
            print("skip existing", name); continue
        rf = Ranged(ds, fp)
        n = 0
        limit = budget * MB
        with open(path, "w", encoding="utf8") as f:
            if kind == "jsonl":
                rf.seek(0)
                buf = b""
                while n < limit and rf.tell() < rf.size:
                    buf += rf.read(4 * MB)
                    *lines, buf = buf.split(b"\n")
                    for ln in lines:
                        if not ln.strip(): continue
                        for t in texts_of(json.loads(ln), keys):
                            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n"); n += len(t.encode())
                        if n >= limit: break
            else:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(rf)
                cols = keys or None
                if not keys:
                    # first string-ish top-level column(s) by schema name
                    names = pf.schema_arrow.names
                    pref = [c for c in ("text", "content", "completion", "conversations") if c in names]
                    cols = pref or names  # reasoning_en: conversations column (user+assistant turns)
                for g in range(pf.num_row_groups):
                    tbl = pf.read_row_group(g, columns=cols)
                    for rec in tbl.to_pylist():
                        for t in texts_of(rec, keys):
                            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n"); n += len(t.encode())
                    if n >= limit: break
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        ent = {"name": name, "domain": domain, "dataset": ds, "file": fp,
               "source_url": f"https://www.modelscope.cn/datasets/{ds}/file/view/master?id=&path={urllib.parse.quote(fp)}",
               "license": license_of(ds), "remote_size": rf.size, "http_bytes_fetched": rf.fetched,
               "extracted_text_bytes": n, "sample_file": os.path.relpath(path), "sample_sha256": sha, **rf.meta}
        manifest.append(ent); print(json.dumps(ent, ensure_ascii=False))
        json.dump(manifest, open(os.path.join(OUT, f"manifest.{name}.json"), "w"), indent=1, ensure_ascii=False)
    # jsonl "raw" kinds count fetched via Ranged too
    allm = []
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("manifest."): allm += json.load(open(os.path.join(OUT, fn)))
    json.dump(allm, open(os.path.join(OUT, "corpus_manifest.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
