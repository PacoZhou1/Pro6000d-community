"""Download a pinned public DFlash snapshot exclusively through ModelScope.

Uses the public repository API and verifies each file before an atomic rename.
Existing files with a different hash are preserved and cause an error.
"""
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

REPO = "z-lab/Qwen3.6-27B-DFlash"
REV = "fd20fcba71887d77ff81735911bb5935a9af5aa5"
ROOT = Path("/home/llm/models/Qwen3.6-27B-DFlash")
API = "https://modelscope.cn/api/v1/models/" + REPO + "/repo"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


url = API + "/files?" + urllib.parse.urlencode({"Revision": REV, "Recursive": "true"})
with urllib.request.urlopen(url, timeout=60) as r:
    manifest = json.load(r)
files = manifest["Data"]["Files"]
ROOT.mkdir(parents=True, exist_ok=True)
out = {"repository": REPO, "revision": REV, "source": "ModelScope", "files": [],
       "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
for f in files:
    if f["Type"] == "tree":
        continue
    path = ROOT / f["Path"]
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Invalid manifest path")
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = f["Sha256"]
    if path.exists():
        if sha(path) != expected:
            raise RuntimeError("Existing file differs; preserving " + str(path))
    else:
        partial = path.with_name(path.name + ".partial")
        file_url = API + "?" + urllib.parse.urlencode({"Revision": REV, "FilePath": f["Path"]})
        for attempt in range(3):
            try:
                h = hashlib.sha256()
                size = 0
                with urllib.request.urlopen(file_url, timeout=120) as r, partial.open("wb") as dst:
                    while block := r.read(8 * 1024 * 1024):
                        dst.write(block)
                        h.update(block)
                        size += len(block)
                if size != f["Size"] or h.hexdigest() != expected:
                    raise RuntimeError("Downloaded file hash/size mismatch")
                partial.rename(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)
    record = {"path": f["Path"], "size": path.stat().st_size, "sha256": expected}
    out["files"].append(record)
    print(json.dumps(record), flush=True)
out["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
Path("/tmp/qwen-spec-experiment/draft-verification.json").write_text(json.dumps(out, indent=2))
print("DOWNLOAD_VERIFIED", flush=True)
