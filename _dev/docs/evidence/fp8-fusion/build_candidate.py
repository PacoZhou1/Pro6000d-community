#!/usr/bin/env python3
"""Build the isolated SM120 FP8 fusion library, or print its exact command.

This script never imports torch, starts a GPU context, or modifies vLLM.
Use --build only inside a separately owned no-GPU build container that has
CUDA 13 nvcc. The output manifest records source/compiler/binary hashes.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path,
                        default=Path("/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().with_name(
                            "fused_silu_quant.cu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true",
                        help="Execute nvcc; otherwise only print the command")
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source not found: {args.source}")
    command = [str(args.nvcc), "-std=c++17", "-O3", "-lineinfo",
               "-gencode=arch=compute_120,code=sm_120", "--shared",
               "-Xcompiler", "-fPIC", str(args.source), "-o", str(args.output)]
    print(json.dumps({"build": args.build, "command": command,
                      "source_sha256": sha256(args.source)}, sort_keys=True),
          flush=True)
    if not args.build:
        return 0
    if not args.nvcc.is_file():
        parser.error(f"nvcc not found: {args.nvcc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([str(args.nvcc), "--version"], text=True,
                             check=True, capture_output=True).stdout.strip()
    subprocess.run(command, check=True)
    manifest = {
        "commit": "db9527a46873454610df6dbedf79a36d6bf1a7f6",
        "architecture": "sm_120",
        "command": command,
        "nvcc_version": version,
        "source_sha256": sha256(args.source),
        "library": str(args.output.resolve()),
        "library_sha256": sha256(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
