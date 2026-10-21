"""CPU bootstrap for the frozen FP8 release; no torch/vLLM import here.

Underlying numerical implementations are byte-identical archived artifacts.
This module selects them, audits sources, and rejects bypass configurations.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path("/opt/pro6000d")
BASE_SITE = Path("/usr/lib/python3.12/sitecustomize.py")
PREFIX = "PRO6000D_RELEASE_"
FEATURES = ("V2", "F", "DRAFT64K")
SPECULATOR = "vllm.v1.worker.gpu.spec_decode.speculator"
DRAFT = "vllm.v1.worker.gpu.spec_decode.draft_vocab"
STATE = None
_DRAFT_HOOK = None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError("PRO6000D release: " + message)


def flags_from_env(environ):
    for key in environ:
        require(not key.startswith(PREFIX) or key[len(PREFIX):] in FEATURES,
                "unknown release setting " + key)
    result = {}
    for name in FEATURES:
        value = environ.get(PREFIX + name, "1")
        require(value in ("0", "1"), PREFIX + name + " must be 0 or 1")
        result[name] = value == "1"
    return result


def installed_vllm_root():
    spec = importlib.machinery.PathFinder.find_spec("vllm", sys.path)
    require(spec is not None and spec.submodule_search_locations is not None,
            "cannot locate installed vLLM without importing it")
    locations = list(spec.submodule_search_locations)
    require(len(locations) == 1, "ambiguous vLLM package root")
    return Path(locations[0]).resolve()


def audit(root, vllm_root, base_site):
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["schema"] == "pro6000d-fp8-release-bundle-v1", "manifest schema")
    for rel, expected in manifest["files"].items():
        path = root / rel
        require(path.is_file() and path.resolve().is_relative_to(root.resolve()), "missing/escaped payload " + rel)
        require(sha(path) == expected, "payload SHA mismatch " + rel)
    lock = json.loads((root / "source-lock.json").read_text())
    require(manifest["base_image"] == lock["base_image"], "base identity mismatch")
    for rel, expected in lock["payload"].items():
        require(manifest["files"].get(rel) == expected, "payload source lock mismatch " + rel)
    for rel, expected in lock["installed_vllm"].items():
        path = vllm_root / rel
        require(path.is_file() and sha(path) == expected, "installed vLLM source mismatch " + rel)
    require(manifest["base_sitecustomize_sha256"] == lock["base_sitecustomize_sha256"]
            and base_site.is_file() and sha(base_site) == manifest["base_sitecustomize_sha256"],
            "base sitecustomize missing/changed")
    return manifest


def configure(environ, flags, root, lock):
    """Do not permit legacy envs to override independent release switches."""
    require(not flags["DRAFT64K"] or environ.get("VLLM_USE_V2_MODEL_RUNNER") == "1",
            "DRAFT64K requires VLLM_USE_V2_MODEL_RUNNER=1")
    internal = {
        "PRO6000D_F_V2": "1",  # runs V2 guard -> base site exactly once, even V2 disabled
        "PRO6000D_FP8_FUSION_ENABLE": "1" if flags["V2"] else "0",
        "PRO6000D_FP8_FUSION_LIBRARY": str(root / "runtime/v2/libfp8_silu_quant.so"),
        "PRO6000D_FP8_FUSION_LIBRARY_SHA256": lock["payload"]["runtime/v2/libfp8_silu_quant.so"],
        "PRO6000D_F_ENABLE": "norm,conv" if flags["F"] else "",
        "PRO6000D_F_KERNELS_SHA256": lock["payload"]["runtime/f_kernels.py"],
        "PRO6000D_F_VERIFY": "0",
        "PRO6000D_F_VERIFY_CONV": "0",
        "VLLM_DRAFT_VOCAB_IDS": str(root / "draft/subset_65536.json") if flags["DRAFT64K"] else "",
        "VLLM_DRAFT_VOCAB_SHA256": lock["payload"]["draft/subset_65536.json"] if flags["DRAFT64K"] else "",
        "VLLM_DRAFT_VOCAB_STRICT": "1" if flags["DRAFT64K"] else "0",
    }
    require(not environ.get("PRO6000D_F_DIAGNOSTIC_SHA256"), "diagnostic helper is not a release component")
    for key, value in internal.items():
        require(key not in environ or environ[key] == value,
                "legacy setting conflicts with release switches: " + key)
    environ.update(internal)


def guard_draft_factory(module):
    original = module.maybe_build_draft_vocab_head

    @functools.wraps(original)
    def checked(speculator):
        # sample_draft tests this before draft_vocab_head. Reject silent bypass.
        require(not getattr(speculator, "use_local_argmax_reduction", False),
                "DRAFT64K requires use_local_argmax_reduction=false")
        head = original(speculator)
        require(head is not None, "enabled DRAFT64K returned no head")
        return head

    module.maybe_build_draft_vocab_head = checked


class _DraftLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        super().exec_module(module)
        if module.__name__ == SPECULATOR:
            guard_draft_factory(module)


class DraftHook(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.paths = {SPECULATOR: root / "draft/speculator.py", DRAFT: root / "draft/draft_vocab.py"}

    def find_spec(self, fullname, path=None, target=None):
        source = self.paths.get(fullname)
        if source is None:
            return None
        return importlib.util.spec_from_file_location(fullname, source, loader=_DraftLoader(fullname, str(source)))


def activate():
    global STATE, _DRAFT_HOOK
    if STATE is not None:
        return
    flags = flags_from_env(os.environ)
    vllm_root = installed_vllm_root()
    manifest = audit(ROOT, vllm_root, BASE_SITE)
    lock = json.loads((ROOT / "source-lock.json").read_text())
    configure(os.environ, flags, ROOT, lock)
    if flags["DRAFT64K"]:
        require(SPECULATOR not in sys.modules and DRAFT not in sys.modules,
                "draft module loaded before release bootstrap")
        _DRAFT_HOOK = DraftHook(ROOT)
        sys.meta_path.insert(0, _DRAFT_HOOK)
    # Run the archived site as __init__ (not a module import named sitecustomize).
    # It runs archived V2 guard, which runs original image sitecustomize once.
    runpy.run_path(str(ROOT / "runtime/sitecustomize.py"))
    STATE = {"schema": "pro6000d-release-active-v1", "features": flags,
             "manifest_sha256": sha(ROOT / "manifest.json"), "vllm_root": str(vllm_root),
             "base_image": lock["base_image"], "base_sitecustomize_sha256": manifest["base_sitecustomize_sha256"]}
    sys.stderr.write("PRO6000D_RELEASE_ACTIVE " + json.dumps(STATE, sort_keys=True) + "\n")
    sys.stderr.flush()
