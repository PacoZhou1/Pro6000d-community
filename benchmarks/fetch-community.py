#!/usr/bin/env python3
from pathlib import Path
import urllib.request,hashlib
url="https://raw.githubusercontent.com/local-inference-lab/llm-inference-bench/80d1f1b0ab9830c3fd8a22c42f461c40cbc7cf96/llm_decode_bench.py"
data=urllib.request.urlopen(url,timeout=60).read()
assert hashlib.sha256(data).hexdigest()=="516dc590b80dda6d9880f9f5034026fdf1b2074e9747bf8a920758680f019817"
p=Path(__file__).parent/"vendor";p.mkdir(exist_ok=True)
(p/"bench-0.6.1.py").write_bytes(data)
print("Fetched and verified public 0.6.1; see upstream usage and licensing terms.")
