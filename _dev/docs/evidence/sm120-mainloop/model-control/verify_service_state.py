"""Read-only verification of the user's keep-stopped service policy.

Runs on the GPU host. Never starts, stops, or changes a container. Credentials
are read in memory from Docker metadata and are never serialized.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import subprocess
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--functional", action="store_true")
    args = parser.parse_args()
    initial = json.loads(args.initial_state.read_text())["containers"]
    result = {
        "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "policy": "Original 27B,9B,ASR stay stopped until explicit user start request; embedding stays running.",
        "services": [],
    }
    for name, expected in initial.items():
        obj = json.loads(subprocess.check_output(["docker", "inspect", name]))[0]
        raw_cmd = obj["Config"]["Cmd"]
        cmd = list(raw_cmd)
        for i, value in enumerate(cmd):
            if value in ("--api-key", "--password", "--token", "--hf-token"):
                cmd[i + 1] = "<REDACTED>"
            elif any(value.startswith(flag + "=") for flag in
                     ("--api-key", "--password", "--token", "--hf-token")):
                cmd[i] = value.split("=")[0] + "=<REDACTED>"
        embedding = name == "vllm-embedding"
        row = {
            "name": name, "id": obj["Id"], "image": obj["Image"],
            "running": obj["State"]["Running"],
            "started_at": obj["State"]["StartedAt"],
            "restart_count": obj["RestartCount"],
            "original_id_matches": obj["Id"] == expected["id"],
            "original_image_matches": obj["Image"] == expected["image"],
            "original_command_matches": cmd == expected["cmd"],
            "restart_count_matches": obj["RestartCount"] == expected["restart_count"],
            "expected_running": embedding,
        }
        checks = [row[k] for k in ("original_id_matches", "original_image_matches",
                                  "original_command_matches", "restart_count_matches")]
        checks.append(row["running"] == embedding)
        if embedding:
            row["start_time_matches"] = row["started_at"] == expected["started_at"]
            checks.append(row["start_time_matches"])
            key = raw_cmd[raw_cmd.index("--api-key") + 1]
            headers = {"Authorization": "Bearer " + key}
            request = urllib.request.Request("http://127.0.0.1:8002/health", headers=headers)
            with urllib.request.urlopen(request, timeout=15) as response:
                row["health"] = response.status
            checks.append(row["health"] == 200)
            if args.functional:
                model = raw_cmd[raw_cmd.index("--served-model-name") + 1]
                payload = json.dumps({"model": model, "input": "FP8 fusion service check"}).encode()
                request = urllib.request.Request("http://127.0.0.1:8002/v1/embeddings",
                    data=payload, headers={**headers, "Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    body = json.load(response)
                    vector = body["data"][0]["embedding"]
                    finite = bool(vector) and all(math.isfinite(v) for v in vector)
                    row["functional"] = {"status": response.status, "dimensions": len(vector),
                        "finite": finite, "norm": math.sqrt(sum(v * v for v in vector)) if finite else None}
                checks.append(row["functional"]["status"] == 200 and finite and len(vector) == 1024)
        row["passed"] = all(checks)
        result["services"].append(row)
    result["passed"] = all(row["passed"] for row in result["services"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "policy": result["policy"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
