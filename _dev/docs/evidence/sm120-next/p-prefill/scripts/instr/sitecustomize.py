"""Diagnostic only (P4): log every call of Scheduler._mamba_block_aligned_split (start, asked, returned).
Pure observation wrapper; return value untouched. Mounted only on dedicated split-probe arms."""
import importlib.abc, importlib.machinery, os, runpy, sys
from pathlib import Path
_TARGET = "vllm.v1.core.sched.scheduler"
_BASE = Path("/usr/lib/python3.12/sitecustomize.py")
if _BASE.is_file() and _BASE.resolve() != Path(__file__).resolve():
    runpy.run_path(str(_BASE))

def _patch(module):
    S = module.Scheduler
    orig = S._mamba_block_aligned_split
    def wrapped(self, request, num_new_tokens, *a, **k):
        r = orig(self, request, num_new_tokens, *a, **k)
        print("SPLITPROBE prompt=%d num_tokens=%d computed=%d asked=%d got=%d end=%d eagle_drop=%s block=%d" % (
            request.num_prompt_tokens, request.num_tokens, request.num_computed_tokens, num_new_tokens, r,
            request.num_computed_tokens + r, getattr(self, "use_eagle_block_drop", None),
            self.cache_config.block_size), file=sys.stderr, flush=True)
        return r
    S._mamba_block_aligned_split = wrapped

class _L(importlib.abc.Loader):
    def __init__(self, d): self.d = d
    def create_module(self, spec):
        c = getattr(self.d, "create_module", None); return c(spec) if c else None
    def exec_module(self, m):
        self.d.exec_module(m); _patch(m)

class _F(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET: return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None: spec.loader = _L(spec.loader)
        return spec
sys.meta_path.insert(0, _F())
