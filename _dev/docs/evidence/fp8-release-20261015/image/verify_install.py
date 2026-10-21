"""CPU-only installation/selection check; does not import the model or torch."""
import argparse
import itertools
import json
import os
import subprocess
import sys


def one():
    import sitecustomize
    import pro6000d_release as release
    assert sitecustomize.__file__ == "/opt/pro6000d/bootstrap/sitecustomize.py"
    assert release.STATE is not None
    hook = release._DRAFT_HOOK
    if release.STATE["features"]["DRAFT64K"]:
        assert hook is not None
        spec = hook.find_spec(release.SPECULATOR)
        assert spec.origin == "/opt/pro6000d/draft/speculator.py"
        assert hook.find_spec(release.DRAFT).origin == "/opt/pro6000d/draft/draft_vocab.py"
    else:
        assert hook is None
        assert not any(isinstance(h, release.DraftHook) for h in sys.meta_path)
    assert "torch" not in sys.modules, "CPU verification unexpectedly imported torch"
    print(json.dumps({"cpu_only_install_check": True, "state": release.STATE}, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", action="store_true")
    args = p.parse_args()
    if not args.matrix:
        one()
        return
    # Top-level parent startup normalized legacy variables; remove only those
    # derived values so each child selects independently from release switches.
    results = []
    for values in itertools.product(("0", "1"), repeat=3):
        env = {k: v for k, v in os.environ.items() if not (
            k.startswith("PRO6000D_F_") or k.startswith("PRO6000D_FP8_FUSION_")
            or k.startswith("VLLM_DRAFT_VOCAB_"))}
        env.update(zip(("PRO6000D_RELEASE_V2", "PRO6000D_RELEASE_F", "PRO6000D_RELEASE_DRAFT64K"), values))
        result = subprocess.run([sys.executable, "-I", __file__], env=env, text=True, capture_output=True)
        if result.returncode:
            sys.stderr.write(result.stdout + result.stderr)
            raise RuntimeError(f"CPU release switch matrix failed: {values}, rc={result.returncode}")
        parsed = json.loads(result.stdout.strip().splitlines()[-1])
        assert parsed["state"]["features"] == dict(zip(("V2", "F", "DRAFT64K"), (x == "1" for x in values)))
        results.append(parsed)
    print(json.dumps({"cpu_only_matrix_passed": True, "cases": results}, sort_keys=True))


if __name__ == "__main__":
    main()
