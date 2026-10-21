"""Validate saved restoration and embedding-monitor evidence without credentials."""
import argparse
import datetime as dt
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    functional = json.loads((root / "restoration.json").read_text())
    stable = json.loads((root / "restoration-stability.json").read_text())
    monitor = [json.loads(line) for line in (root / "embedding-monitor.jsonl").read_text().splitlines() if line]
    cleanup = json.loads((root / "container-cleanup.json").read_text())
    embedding = next(s for s in functional["services"] if s["container"] == "vllm-embedding")
    elapsed = (dt.datetime.fromisoformat(stable["observed_at_utc"]) - dt.datetime.fromisoformat(functional["observed_at_utc"])).total_seconds()
    monitor_valid = bool(monitor) and all(
        s.get("id") == embedding["id"]
        and s.get("started_at") == embedding["started_at"]
        and s.get("restart_count") == embedding["restart_count"]
        and s.get("image") == embedding["image"]
        and s.get("health") == 200
        and s.get("running") is True
        for s in monitor
    )
    result = {
        "functional_passed": functional["passed"],
        "stability_passed": stable["passed"],
        "stability_elapsed_s": elapsed,
        "monitor_samples": len(monitor),
        "monitor_first_utc": monitor[0]["at_utc"] if monitor else None,
        "monitor_last_utc": monitor[-1]["at_utc"] if monitor else None,
        "embedding_identity_and_health_preserved": monitor_valid,
        "owned_experiment_containers_remaining": cleanup["owned_experiment_containers_remaining"],
        "passed": functional["passed"] and stable["passed"] and elapsed >= 300 and monitor_valid and not cleanup["owned_experiment_containers_remaining"],
    }
    (root / "restoration-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
