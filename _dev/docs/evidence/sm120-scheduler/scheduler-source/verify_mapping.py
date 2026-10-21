#!/usr/bin/env python3
"""CPU reproduction of pinned Sm100 CLC coordinate mapping on SM120.

No CUDA/library loading. Enumerated physical x/y neighbors are NOT an observed
launch, residency or execution order; CLC dynamically obtains additional IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINNED = {
    "tile_scheduler.hpp": "acc90548b9e2b19f944764ced57e1459d5c2ed7e118d6a1af476add26c3d5e73",
    "sm100_tile_scheduler.hpp": "54ebcaee08d4fc0663169f97c7fa665cec90d29ce5d01336c2c714f2a911b010",
    "tile_scheduler_params.h": "ef48a12e8920183e88259d0b685279c2232fc2fb12c4fb4db7e8d0fbfdc019e9",
}
CASES = [
    ("official_m16160_h8", 16160, 128, 128, "Heuristic", 8, "ncu-m16160-official"),
    ("official_m16160_m1", 16160, 128, 128, "AlongM", 1, "ncu-m16160-official-along-m-s1"),
    ("official_m32768_h8", 32768, 128, 128, "Heuristic", 8, "ncu-m32768-official-heuristic-s8"),
    ("official_m32768_m1", 32768, 128, 128, "AlongM", 1, "ncu-m32768-official-along-m-s1"),
    ("v6_m16160_n8", 16160, 64, 256, "AlongN", 8, "ncu-m16160-v6-along-n-s8-isolated"),
    ("v6_m16160_m1", 16160, 64, 256, "AlongM", 1, "ncu-m16160-v6-along-m-s1"),
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def map_tile(x, y, grid_x, grid_y, raster, swizzle):
    # sm100_tile_scheduler.hpp:770-798, ClusterShape=(1,1,1).
    major, minor = x, y
    if swizzle > 1 and x < (grid_x // swizzle) * swizzle and y < (grid_y // swizzle) * swizzle:
        major = x // swizzle + (grid_x // swizzle) * (y % swizzle)
        minor = x % swizzle + (y // swizzle) * swizzle
    return (minor, major) if raster == "AlongN" else (major, minor)


def ncu_grid(root, label):
    path = root / label / "raw.csv"
    with path.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("ID") == "0"]
    require(len(rows) == 1 and "GemmKernel" in rows[0].get("Kernel Name", ""),
            f"require one GEMM ID=0 in {path}")
    match = re.fullmatch(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", rows[0]["Grid Size"])
    require(match is not None, f"unknown NCU grid format in {path}")
    grid = tuple(int(x) for x in match.groups())
    require(int(rows[0]["launch__grid_size"].replace(",", "")) == math.prod(grid),
            f"NCU grid/count disagree in {path}")
    return grid, {"csv": str(path.relative_to(root)), "sha256": sha(path),
                  "kernel_id": "0", "kernel": rows[0]["Kernel Name"]}


def verify_case(root, spec):
    name, m, tm, tn, requested_raster, swizzle, label = spec
    tiles_m, tiles_n = (m + tm - 1) // tm, 34816 // tn
    # Sm100Params calls the Sm90 heuristic with UNROUNDED cluster tile counts.
    raster = ("AlongM" if tiles_n > tiles_m else "AlongN") if requested_raster == "Heuristic" else requested_raster
    # The fixed heuristic falls back if a transposed grid-y would exceed 65535.
    if requested_raster == "Heuristic" and raster == "AlongN" and tiles_m > 65535:
        raster = "AlongM"
    gx, gy = (tiles_n, tiles_m) if raster == "AlongN" else (tiles_m, tiles_n)
    seen, digest = set(), hashlib.sha256()
    for y in range(gy):
        for x in range(gx):
            lm, ln = map_tile(x, y, gx, gy, raster, swizzle)
            require(0 <= lm < tiles_m and 0 <= ln < tiles_n, f"out-of-bounds mapping: {name}")
            require((lm, ln) not in seen, f"duplicate logical tile: {name}")
            seen.add((lm, ln))
            digest.update(struct.pack("<IIII", x, y, lm, ln))
    require(len(seen) == tiles_m * tiles_n, f"missing logical tile: {name}")
    actual_grid, evidence = ncu_grid(root, label)
    require(actual_grid == (gx, gy, 1), f"source mapping/NCU grid disagree: {name}")
    def sample(y, width):
        return [{"physical_xy": [x, y], "logical_mn": list(map_tile(x, y, gx, gy, raster, swizzle))}
                for x in range(min(width, gx))]
    return {"name": name, "M": m, "N": 34816, "K": 5120, "tile_mnk": [tm, tn, 128],
            "cluster": [1, 1, 1], "requested_raster": requested_raster, "resolved_raster": raster,
            "requested_swizzle": swizzle, "effective_divisor": 0 if swizzle <= 1 else swizzle,
            "logical_tiles_mn": [tiles_m, tiles_n], "physical_grid": [gx, gy, 1],
            "enumerated_physical_ids": gx * gy, "unique_logical_tiles": len(seen),
            "bijection_passed": True, "ncu_grid_match": True, "ncu_evidence": evidence,
            "swizzle_core_physical_extent": [(gx // swizzle) * swizzle, (gy // swizzle) * swizzle] if swizzle > 1 else None,
            "residual_x_count": gx % swizzle if swizzle > 1 else 0,
            "residual_y_count": gy % swizzle if swizzle > 1 else 0,
            "nominal_x_neighbors_y0": sample(0, 16), "nominal_x_neighbors_y1": sample(1, 8),
            "mapping_digest_sha256": digest.hexdigest(),
            "digest_encoding": "nominal y-outer/x-inner enumeration of little-endian uint32 (x,y,logicalM,logicalN); not actual execution order"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=HERE.parent, help="saved scheduler evidence root")
    p.add_argument("--output", type=Path, default=HERE / "mapping-verification.json")
    args = p.parse_args()
    manifest = json.loads((HERE / "source-manifest.json").read_text())
    require(manifest["files_sha256"] == PINNED, "fixed source manifest differs from pinned hashes")
    require(all(sha(HERE / name) == wanted for name, wanted in PINNED.items()), "fixed header hash mismatch")
    cases = [verify_case(args.root, spec) for spec in CASES]
    result = {"passed": True, "scope": "pure CPU reproduction of fixed CLC coordinate mapping + saved NCU physical-grid comparison",
              "script_sha256": sha(Path(__file__)), "source_manifest_sha256": sha(HERE / "source-manifest.json"),
              "headers_sha256": PINNED, "cases": cases,
              "limits": ["Not a compiled C++ scheduler invocation or a new GPU run.",
                         "Physical block ID mapping and nominal neighbor enumeration do not establish CLC/runtime execution order or SM residency.",
                         "Grid agreement does not prove cache hit behavior or allocate DRAM counters to A/B buffers.",
                         "Only the pinned headers, selected shapes, cluster=(1,1,1) and swizzles 1/8 are verified."]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
