#!/usr/bin/env python3
"""Offline, fail-closed summary of real post-grad FX graph snapshots.

No GPU, PyTorch, server or credentials required. It never claims module/weight
identity from shape alone. Run on proof JSON files produced by sitecustomize.
"""

import argparse
import json
from pathlib import Path


def inspect(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["name"]: node for node in data["nodes"]}
    matches = []
    for gemm in data["gemms"]:
        quant_paths = gemm.get("quant_paths", [])
        if not quant_paths:
            continue
        quant_nodes = [nodes.get(path_[-1]) for path_ in quant_paths]
        candidate = {
            "gemm": gemm["node"],
            "gemm_meta": gemm.get("meta", {}),
            "quant_paths": quant_paths,
            "single_direct_user": gemm.get("single_direct_user") is True,
            "quant_nodes": [
                {"name": node["name"], "target": node["target"]}
                for node in quant_nodes if node
            ],
            "module_identity_proven": False,
            "packed_weight_identity_proven": False,
            "replaceable": False,
            "reason": "FX snapshot needs explicit canonical-to-packed weight identity and full output-user validation",
        }
        matches.append(candidate)
    return {
        "source": str(path),
        "pid": data.get("pid"),
        "graph_number": data.get("graph_number"),
        "node_count": data.get("node_count"),
        "gemm_count": len(data["gemms"]),
        "gemm_to_fused_quant_paths": len(matches),
        "candidates": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proofs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [inspect(path) for path in args.proofs]
    output = {
        "scope": "read-only graph connection proof; never authorizes fusion replacement",
        "reports": reports,
        "totals": {
            "graphs": len(reports),
            "gemms": sum(x["gemm_count"] for x in reports),
            "gemm_to_fused_quant_paths": sum(
                x["gemm_to_fused_quant_paths"] for x in reports
            ),
            "replaceable": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
