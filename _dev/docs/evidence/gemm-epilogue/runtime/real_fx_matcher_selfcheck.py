#!/usr/bin/env python3
"""Rehydrate saved real FX adjacency for the fail-closed matcher (CPU only).

This is an audit of recorded graph structure, not a model execution or a
replacement for an Inductor compile. It reads only the 21 saved witness JSONs.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

from fx_matcher import find_gateup_matches


class TensorMeta:
    def __init__(self, row):
        self.dtype = row["dtype"]
        self.shape = tuple(row["shape"])
        self._stride = tuple(row["stride"])

    def stride(self):
        return self._stride


class Operator:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class Node:
    def __init__(self, row):
        self.name = row["name"]
        self.target = (
            "auto_functionalized"
            if row["target"].startswith("auto_functionalized<")
            else row["target"]
        )
        meta = row.get("meta", {})
        self.meta = ({"val": TensorMeta(meta)} if all(
            key in meta for key in ("dtype", "shape", "stride")
        ) else {})
        self.args = ()
        self.kwargs = {}
        self.users = {}
        self.all_input_nodes = []


def rehydrate(payload):
    rows = payload["nodes"]
    nodes = {row["name"]: Node(row) for row in rows}

    def arg(value):
        if isinstance(value, dict):
            if "node" in value:
                return nodes[value["node"]]
            if "operator" in value:
                return Operator(value["operator"])
            return {key: arg(item) for key, item in value.items()}
        if isinstance(value, list):
            return tuple(arg(item) for item in value)
        return value

    for row in rows:
        node = nodes[row["name"]]
        node.args = arg(row["args"])
        node.kwargs = {key: arg(value) for key, value in row["kwargs"].items()}
        node.all_input_nodes = [nodes[name] for name in row["inputs"]]
        node.users = {nodes[name]: None for name in row["users"]}
    return type("SavedGraph", (), {"nodes": list(nodes.values())})(), nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(str(args.directory / "fx-proof-*.json")))
    if len(paths) != 21:
        raise RuntimeError(f"expected 21 saved real FX graphs, got {len(paths)}")
    total = 0
    files = []
    representative = None
    for path in paths:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
        graph, nodes = rehydrate(payload)
        matches = find_gateup_matches(graph)
        expected = payload.get("strict_structural_matches", [])
        if len(matches) != len(expected):
            raise AssertionError(f"saved/current matcher count differs: {path}")
        if {match.gemm.name for match in matches} != {
            item["gemm"] for item in expected
        }:
            raise AssertionError(f"saved/current matched GEMM differs: {path}")
        if matches and representative is None:
            representative = payload
        total += len(matches)
        files.append({"file": Path(path).name,
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "matches": len(matches)})
    if total != 20 or representative is None:
        raise AssertionError(f"expected 20 real matches, got {total}")

    negative = {}
    for allocation in ("gemm_out", "quant_out", "quant_scale"):
        graph, nodes = rehydrate(representative)
        if len(find_gateup_matches(graph)) != 1:
            raise AssertionError("representative graph has no unique match")
        match = find_gateup_matches(graph)[0]
        source = {
            "gemm_out": match.gemm.kwargs["out"],
            "quant_out": match.quant.kwargs["out"],
            "quant_scale": match.quant.kwargs["scales"],
        }[allocation]
        extra = Node({"name": "injected_extra_reader", "target": "audit.reader"})
        extra.args = (source,)
        extra.all_input_nodes = [source]
        source.users[extra] = None
        graph.nodes.append(extra)
        negative[allocation] = len(find_gateup_matches(graph)) == 0
        if not negative[allocation]:
            raise AssertionError(f"extra reader was not rejected: {allocation}")

    print(json.dumps({
        "passed": True,
        "scope": "rehydrated saved real FX JSON adjacency, not a model/GPU run",
        "graph_count": len(paths),
        "match_count": total,
        "extra_reader_rejected": negative,
        "files": files,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
