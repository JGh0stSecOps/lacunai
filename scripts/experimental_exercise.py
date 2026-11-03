#!/usr/bin/env python3
# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Run the separately frozen, non-efficacy engineering supplement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lacunai.experimental.artifacts import atomic_json  # noqa: E402
from lacunai.experimental.batch import check_bundle, finalize_exception  # noqa: E402
from lacunai.experimental.exercise import (  # noqa: E402
    EXERCISE_SHA256,
    load_exercise_contract,
)
from lacunai.experimental.spec import ExperimentRefused, admit_root, load_contract  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=int, required=True)
    parser.add_argument("--enable-experimental", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"phase": "engineering_exercise", "state": "admission"}), flush=True)
    if not args.enable_experimental:
        raise ExperimentRefused("engineering exercise disabled; explicit opt-in required")
    root = admit_root(args.root)
    contract = load_contract(root / "contract.json", enable=True)
    policy = load_exercise_contract(root, contract)
    if (
        args.deadline_epoch != policy["original_deadline_epoch"]
        or time.time() >= args.deadline_epoch - 600
    ):
        raise ExperimentRefused("engineering exercise changes or exceeds original deadline")
    check_bundle(root)
    from lacunai.experimental.runner import run, runtime_manifest

    manifest = runtime_manifest(root, contract)
    manifest.update(
        purpose="engineering_exercise",
        efficacy_eligible=False,
        exercise_contract_sha256=EXERCISE_SHA256,
    )
    atomic_json(root / "manifest.json", manifest)
    result = run(root, contract, args.deadline_epoch, exercise=policy)
    print(
        json.dumps(
            {
                "phase": "engineering_exercise",
                "status": result["status"],
                "verdict": result["verdict"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--root" in sys.argv and "--enable-experimental" in sys.argv:
            try:
                finalize_exception(
                    Path(sys.argv[sys.argv.index("--root") + 1]), "engineering_exercise", error
                )
            except Exception as finalization_error:
                print(
                    json.dumps({"state": "finalization_failed", "error": str(finalization_error)}),
                    file=sys.stderr,
                    flush=True,
                )
        print(json.dumps({"state": "failed", "error": str(error)}), file=sys.stderr, flush=True)
        raise SystemExit(1) from error
