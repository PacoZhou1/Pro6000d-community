"""Isolated startup packing + official post-grad FX pass registration.

Mount this directory first in PYTHONPATH of an isolated experiment container.
Set PRO6000D_EPILOGUE_MODE=probe or baseline, and
PRO6000D_EPILOGUE_EVIDENCE_DIR to a writable experiment directory.
No environment variable means no vLLM patch. Candidate mode requires a
separately supplied numerically validated library SHA256 and exact ABI v2.
"""

import importlib.abc
import importlib.machinery
import os
import runpy
import sys
from pathlib import Path


# A/B both retain the preceding V2 activation quantization implementation.
# Its sitecustomize is an explicit dependency rather than relying on PYTHONPATH
# ordering (Python auto-imports only one module named sitecustomize).
_V2_SITE = os.environ.get("PRO6000D_EPILOGUE_V2_SITECUSTOMIZE", "")
if _V2_SITE:
    _v2_path = Path(_V2_SITE)
    if not _v2_path.is_absolute() or not _v2_path.is_file():
        raise RuntimeError("absolute V2 sitecustomize path required")
    if _v2_path.resolve() == Path(__file__).resolve():
        raise RuntimeError("V2 sitecustomize must differ from epilogue entry")
    runpy.run_path(str(_v2_path))
else:
    _base_site = Path("/usr/lib/python3.12/sitecustomize.py")
    if _base_site.is_file() and _base_site.resolve() != Path(__file__).resolve():
        runpy.run_path(str(_base_site))

_MODE = os.environ.get("PRO6000D_EPILOGUE_MODE", "")
_EVIDENCE = os.environ.get("PRO6000D_EPILOGUE_EVIDENCE_DIR", "")
_LOADER = "vllm.model_executor.model_loader.base_loader"
_MANAGER = "vllm.compilation.passes.pass_manager"
_CANDIDATE_ABI = None


def _patch_loader(module):
    original = module.BaseModelLoader.load_model
    if getattr(original, "_pro6000d_gateup_patched", False):
        return

    def wrapped(self, *args, **kwargs):
        model = original(self, *args, **kwargs)
        from packing import prepare_model

        prepare_model(model, _EVIDENCE)
        return model

    wrapped._pro6000d_gateup_patched = True
    module.BaseModelLoader.load_model = wrapped


def _patch_manager(module):
    global _CANDIDATE_ABI
    original = module.PostGradPassManager.configure
    if getattr(original, "_pro6000d_gateup_patched", False):
        return

    def wrapped(self, config):
        global _CANDIDATE_ABI
        original(self, config)
        from fx_probe import install_probe

        install_probe(self, _EVIDENCE)
        if _MODE == "candidate":
            from candidate_abi import CandidateABI
            from candidate_op import register
            from fx_rewrite import install_rewrite

            if _CANDIDATE_ABI is None:
                _CANDIDATE_ABI = CandidateABI(
                    os.environ["PRO6000D_EPILOGUE_LIBRARY"],
                    os.environ["PRO6000D_EPILOGUE_LIBRARY_SHA256"],
                )
            swizzle = int(os.environ.get("PRO6000D_EPILOGUE_SWIZZLE", "8"))
            register(_CANDIDATE_ABI, swizzle)
            install_rewrite(self, _EVIDENCE, _CANDIDATE_ABI.sha256)

    wrapped._pro6000d_gateup_patched = True
    module.PostGradPassManager.configure = wrapped


class _AfterImport(importlib.abc.Loader):
    def __init__(self, delegate, patch):
        self.delegate, self.patch = delegate, patch

    def create_module(self, spec):
        method = getattr(self.delegate, "create_module", None)
        return method(spec) if method else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        self.patch(module)


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        patch = {_LOADER: _patch_loader, _MANAGER: _patch_manager}.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = _AfterImport(spec.loader, patch)
        return spec


if _MODE:
    if _MODE not in ("probe", "baseline", "candidate"):
        raise RuntimeError("invalid PRO6000D_EPILOGUE_MODE")
    if not _EVIDENCE or not Path(_EVIDENCE).is_absolute():
        raise RuntimeError("absolute PRO6000D_EPILOGUE_EVIDENCE_DIR required")
    if not _V2_SITE or os.environ.get("PRO6000D_FP8_FUSION_ENABLE") != "1":
        raise RuntimeError("A/B requires enabled V2 sitecustomize")
    if _MODE == "candidate":
        library_sha = os.environ.get("PRO6000D_EPILOGUE_LIBRARY_SHA256", "")
        validated_sha = os.environ.get(
            "PRO6000D_EPILOGUE_VALIDATED_LIBRARY_SHA256", ""
        )
        if not library_sha or library_sha != validated_sha:
            raise RuntimeError("candidate requires matching validated library SHA256")
        if not os.environ.get("PRO6000D_EPILOGUE_LIBRARY"):
            raise RuntimeError("candidate library path required")
    sys.meta_path.insert(0, _Hook())
    for name, patch in ((_LOADER, _patch_loader), (_MANAGER, _patch_manager)):
        module = sys.modules.get(name)
        if module is not None:
            patch(module)
