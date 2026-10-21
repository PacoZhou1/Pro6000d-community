#!/usr/bin/env python3
"""Prepare a fresh self-contained Docker context, checking every frozen input.

CPU/filesystem only: does not invoke Docker, a compiler, torch or any service.
No existing output directory is overwritten, and no binary is rebuilt.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
EVIDENCE = RELEASE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="new context directory (must not exist)")
    p.add_argument("--runtime", type=Path, default=EVIDENCE / "sm120-next/takeover/f-mdnone-fix")
    p.add_argument("--received", type=Path, default=RELEASE / "received")
    p.add_argument("--v2-library", type=Path, default=EVIDENCE / "fp8-fusion/libfp8_silu_quant_group_major.so")
    p.add_argument("--v2-source", type=Path, default=EVIDENCE / "fp8-fusion/fused_silu_quant_group_major.cu")
    p.add_argument("--v2-build-log", type=Path, default=EVIDENCE / "fp8-fusion/build-v2.log")
    args = p.parse_args()
    require(not args.output.exists(), "refusing to overwrite existing context")
    lock = json.loads((HERE / "source-lock.json").read_text())
    sources = {f"runtime/{name}": args.runtime / name for name in
               ("sitecustomize.py", "f_kernels.py", "v2_sitecustomize.py")}
    sources.update({
        "runtime/v2/libfp8_silu_quant.so": args.v2_library,
        "draft/speculator.py": args.received / "src/patched/speculator.py",
        "draft/draft_vocab.py": args.received / "src/patched/draft_vocab.py",
        "draft/subset_65536.json": args.received / "inputs/subset_65536.json",
        "upstream/pristine_speculator.py": args.received / "pristine_speculator.py",
        "upstream/base_sitecustomize.py": args.received / "base_sitecustomize.py",
        "reproduction/fused_silu_quant_group_major.cu": args.v2_source,
        "reproduction/build-v2.log": args.v2_build_log,
    })
    require(sources.keys() == lock["payload"].keys(), "source mapping/lock mismatch")
    for rel, source in sources.items():
        require(source.is_file(), "missing input " + str(source))
        require(sha(source) == lock["payload"][rel], "input SHA mismatch: " + str(source))
        if source.suffix == ".py":
            ast.parse(source.read_text(), filename=str(source))
    raw = args.v2_library.read_bytes()
    require(raw[:6] == b"\x7fELF\x02\x01" and int.from_bytes(raw[18:20], "little") == 62,
            "V2 library is not the frozen Linux x86_64 ELF")
    subset = json.loads(sources["draft/subset_65536.json"].read_text())
    ids = subset["ids"]
    require(len(ids) == 65536 and all(type(x) is int for x in ids)
            and ids == sorted(set(ids)) and ids[0] >= 0 and ids[-1] < subset["vocab_size"],
            "invalid frozen subset")
    reference = args.received / "draft64-retest-arm-meta.json"
    meta = json.loads(reference.read_text())
    require(meta["patched_sha256"] == {n: lock["payload"]["draft/" + n]
                                       for n in ("speculator.py", "draft_vocab.py")}
            and meta["subset_sha256"] == lock["payload"]["draft/subset_65536.json"],
            "retest source/subset provenance mismatch")
    require(meta["image"] == lock["base_image"].split("@", 1)[1], "retest base image mismatch")
    build = [json.loads(line) for line in args.v2_build_log.read_text().splitlines() if line.strip()]
    require(build[-1]["source_sha256"] == lock["payload"]["reproduction/fused_silu_quant_group_major.cu"]
            and build[-1]["library_sha256"] == lock["payload"]["runtime/v2/libfp8_silu_quant.so"]
            and build[-1]["architecture"] == "sm_120", "V2 build recipe/hash mismatch")
    sources.update({
        "bootstrap/sitecustomize.py": HERE / "bootstrap/sitecustomize.py",
        "bootstrap/pro6000d_release.py": HERE / "bootstrap/pro6000d_release.py",
        "install_bootstrap.py": HERE / "install_bootstrap.py",
        "verify_install.py": HERE / "verify_install.py",
        "source-lock.json": HERE / "source-lock.json",
        "reproduction/Dockerfile": HERE / "Dockerfile",
        "reproduction/prepare_context.py": Path(__file__).resolve(),
        "provenance/draft64-retest-arm-meta.json": reference,
    })
    for source in sources.values():
        if source.suffix == ".py":
            ast.parse(source.read_text(), filename=str(source))
    dockerfile = (HERE / "Dockerfile").read_text()
    require(dockerfile.splitlines()[0] == "FROM " + lock["base_image"], "Dockerfile base not pinned")
    # Finish all validation before creating a new output tree.
    payload = args.output / "payload"
    payload.mkdir(parents=True)
    for rel, source in sources.items():
        target = payload / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    manifest = {
        "schema": "pro6000d-fp8-release-bundle-v1", "base_image": lock["base_image"],
        "vllm_commit": lock["vllm_commit"], "base_sitecustomize_sha256": lock["base_sitecustomize_sha256"],
        "files": {rel: sha(payload / rel) for rel in sorted(sources)},
        "defaults": {"PRO6000D_RELEASE_V2": "1", "PRO6000D_RELEASE_F": "1", "PRO6000D_RELEASE_DRAFT64K": "1"},
        "upstream_vllm_modified_on_disk": False, "requires_host_instrument_mount": False,
        "draft_subset_size": 65536, "target_lm_head_modified": False,
        "numerics_implementation": "frozen bytes; packaging adds integrity/selection/bypass guards only",
        "v2_binary_recompiled": False,
        "base_entrypoint": ["vllm", "serve"], "base_cmd": None, "base_working_dir": "/vllm-workspace",
    }
    (payload / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(HERE / "Dockerfile", args.output / "Dockerfile")
    (args.output / ".dockerignore").write_text("**/__pycache__/\n**/*.pyc\n")
    print(json.dumps({"prepared": str(args.output), "manifest_sha256": sha(payload / "manifest.json"),
                      "files": len(manifest["files"]), "source_sha_checks_passed": True,
                      "docker_built": False, "gpu_tested": False}, sort_keys=True))


if __name__ == "__main__":
    main()
