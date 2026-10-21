#!/usr/bin/env python3
"""Local CPU checks of packaging/guards; no claim of image or GPU execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("release_under_cpu_test", HERE / "bootstrap/pro6000d_release.py")
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    lock = json.loads((HERE / "source-lock.json").read_text())
    cases = []

    def check(name, fn, raises=False):
        try:
            fn()
        except RuntimeError:
            if not raises:
                raise
        else:
            if raises:
                raise AssertionError("expected rejection: " + name)
        cases.append({"case": name, "pass": True})

    for values in itertools.product(("0", "1"), repeat=3):
        env = dict(zip((release.PREFIX + k for k in release.FEATURES), values))
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        flags = release.flags_from_env(env)
        release.configure(env, flags, Path("/opt/pro6000d"), lock)
        assert env["PRO6000D_FP8_FUSION_ENABLE"] == values[0]
        assert env["PRO6000D_F_ENABLE"] == ("norm,conv" if values[1] == "1" else "")
        assert bool(env["VLLM_DRAFT_VOCAB_IDS"]) == (values[2] == "1")
        assert env["VLLM_DRAFT_VOCAB_STRICT"] == values[2]
        cases.append({"case": "switches_" + "".join(values), "pass": True})
    check("reject_invalid_switch", lambda: release.flags_from_env({"PRO6000D_RELEASE_F": "yes"}), True)
    check("reject_unknown_switch", lambda: release.flags_from_env({"PRO6000D_RELEASE_FF": "1"}), True)
    enabled = dict.fromkeys(release.FEATURES, True)
    check("reject_conflicting_legacy", lambda: release.configure(
        {"VLLM_USE_V2_MODEL_RUNNER": "1", "PRO6000D_F_ENABLE": "norm"}, enabled, Path("/opt/pro6000d"), lock), True)
    check("reject_inactive_model_runner", lambda: release.configure({}, enabled, Path("/opt/pro6000d"), lock), True)
    check("reject_diagnostic_helper", lambda: release.configure(
        {"VLLM_USE_V2_MODEL_RUNNER": "1", "PRO6000D_F_DIAGNOSTIC_SHA256": "x"}, enabled, Path("/opt/pro6000d"), lock), True)

    calls = []
    head = object()
    module = types.SimpleNamespace(maybe_build_draft_vocab_head=lambda s: calls.append(s) or head)
    release.guard_draft_factory(module)
    assert module.maybe_build_draft_vocab_head(types.SimpleNamespace(use_local_argmax_reduction=False)) is head
    n = len(calls)
    check("reject_local_argmax_before_head_factory", lambda: module.maybe_build_draft_vocab_head(
        types.SimpleNamespace(use_local_argmax_reduction=True)), True)
    assert len(calls) == n
    module = types.SimpleNamespace(maybe_build_draft_vocab_head=lambda s: None)
    release.guard_draft_factory(module)
    check("reject_missing_enabled_head", lambda: module.maybe_build_draft_vocab_head(
        types.SimpleNamespace(use_local_argmax_reduction=False)), True)
    hook = release.DraftHook(Path("/opt/pro6000d"))
    assert hook.find_spec(release.SPECULATOR).origin == "/opt/pro6000d/draft/speculator.py"
    assert hook.find_spec(release.DRAFT).origin == "/opt/pro6000d/draft/draft_vocab.py"
    assert hook.find_spec("vllm.other") is None
    cases.append({"case": "draft_import_selection_paths", "pass": True})

    with tempfile.TemporaryDirectory(prefix="pro6000d-release-cpu-") as temp:
        temp = Path(temp)
        # Synthetic integrity fixture tests audit rejection independently of a
        # vLLM install; these bytes are not represented as fixed-image sources.
        root, vllm = temp / "payload", temp / "vllm"
        root.mkdir(); vllm.mkdir()
        (root / "x").write_text("payload")
        (vllm / "source.py").write_text("upstream")
        base = temp / "base_site.py"; base.write_text("base")
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        fixture_lock = {"base_image": "fixture", "base_sitecustomize_sha256": sha(base),
                        "payload": {"x": sha(root / "x")}, "installed_vllm": {"source.py": sha(vllm / "source.py")}}
        (root / "source-lock.json").write_text(json.dumps(fixture_lock))
        manifest = {"schema": "pro6000d-fp8-release-bundle-v1", "base_image": "fixture",
                    "base_sitecustomize_sha256": sha(base),
                    "files": {f: sha(root / f) for f in ("x", "source-lock.json")}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        check("audit_synthetic_fixture_valid", lambda: release.audit(root, vllm, base))
        (root / "x").write_text("changed")
        check("reject_payload_sha_change", lambda: release.audit(root, vllm, base), True)
        (root / "x").write_text("payload")
        (vllm / "source.py").write_text("changed")
        check("reject_installed_source_sha_change", lambda: release.audit(root, vllm, base), True)
        (vllm / "source.py").write_text("upstream")
        base.write_text("changed")
        check("reject_base_site_sha_change", lambda: release.audit(root, vllm, base), True)
        # Python normally swallows a sitecustomize exception. Our frozen wrapper
        # must force exit78 before user code, using a deliberately failing stub.
        startup = temp / "startup"; startup.mkdir()
        (startup / "sitecustomize.py").write_bytes((HERE / "bootstrap/sitecustomize.py").read_bytes())
        (startup / "pro6000d_release.py").write_text("def activate():\n    raise RuntimeError('deliberate CPU fixture')\n")
        env = dict(os.environ, PYTHONPATH=str(startup))
        result = subprocess.run([sys.executable, "-c", "print('USER_CODE_REACHED')"], env=env, capture_output=True, text=True)
        assert result.returncode == 78 and "USER_CODE_REACHED" not in result.stdout
        assert "PRO6000D_RELEASE_FATAL" in result.stderr
        cases.append({"case": "sitecustomize_failure_is_fatal_exit78", "pass": True})

    result = {"schema": "pro6000d-release-cpu-self-test-v1", "passed": True,
              "scope": "CPU logic/mocks only; not fixed-image, GPU, CUDA Graph or model validation",
              "docker_built": False, "gpu_tested": False, "cases": cases,
              "source_sha256": {str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in (HERE / "bootstrap/sitecustomize.py", HERE / "bootstrap/pro6000d_release.py",
                                          HERE / "source-lock.json", Path(__file__).resolve())}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "cpu_cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
