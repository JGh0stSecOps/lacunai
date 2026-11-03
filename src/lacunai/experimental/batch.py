# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Command boundary for explicitly approved, isolated research execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, verify_originals
from .resources import Watchdog, host_sample
from .spec import APPROVED_V1_SHA256, ExperimentRefused, admit_root, load_contract


def check_bundle(root: Path) -> dict[str, str]:
    """Verify staged source/data at the receiver, with no executable manifest entries."""
    hashes = {}
    for line in (root / "bundle.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if Path(name).is_absolute() or ".." in Path(name).parts or name in hashes:
            raise ExperimentRefused("invalid source bundle path")
        path = root / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ExperimentRefused("linked bundle file")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ExperimentRefused(f"source/data bundle hash mismatch: {name}")
        hashes[name] = actual
    required = {
        "contract.json",
        "requirements.lock",
        "datasets/dataset.json",
        "datasets/manifest.json",
        "code/scripts/experimental_batch.py",
        "code/src/lacunai/experimental/batch.py",
    }
    if not required.issubset(hashes):
        raise ExperimentRefused("source bundle does not cover mandatory inputs")
    print(json.dumps({"phase": "receiver_verification", "verified_files": len(hashes)}), flush=True)
    return hashes


def receipts_only(root: Path, phase: str, status: int) -> dict[str, Any]:
    """Finalize even when no ML environment or complete model download exists."""
    receipt_path = root / "receipts/originals.json"
    result: dict[str, Any] = {
        "verdict": "inconclusive",
        "phase": phase,
        "exit_code": status,
        "status": "receipt_readback",
        "errors": [],
        "model_run": False,
    }
    if receipt_path.exists():
        value = json.loads(receipt_path.read_text())
        records = value["files"] if isinstance(value, dict) and "files" in value else value
        try:
            result["original_receipts"] = verify_originals(root, records)
        except Exception as exc:
            result["errors"].append(f"{type(exc).__name__}: {exc}")
            result["status"] = "receipt_verification_failed"
    result["host_after"] = host_sample(root)
    atomic_json(root / "final-result.json", result)
    (root / "comparison-report.txt").write_text(
        "LacunAI Chinese-literacy pilot\nVerdict: inconclusive\n"
        f"Stopped phase: {phase}; observed exit: {status}\n"
        "No unlearning comparison completed. No claim of information removal.\n"
        "Preserved originals and partial files remain in the isolated experiment directory.\n"
        + "\n".join(result["errors"])
        + "\n"
    )
    return result


def finalize_exception(root: Path, phase: str, error: Exception) -> dict[str, Any]:
    """A failed admission/verification still leaves an honest, lightweight final record."""
    root = admit_root(root)
    result_path = root / "result.json"
    result: dict[str, Any] = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {"arms": {}, "errors": [], "model_run": False}
    )
    result["verdict"] = "inconclusive"
    result["status"] = "verification_unavailable"
    result["failed_phase"] = phase
    result.setdefault("errors", []).append(f"{type(error).__name__}: {error}")
    try:
        result["host_after"] = host_sample(root)
    except OSError as readback_error:
        result["errors"].append(f"host readback failed: {readback_error}")
    atomic_json(root / "final-result.json", result)
    atomic_json(root / "verification.json", {"status": "unavailable", "errors": result["errors"]})
    (root / "comparison-report.txt").write_text(
        "LacunAI Chinese-literacy pilot\nVerdict: inconclusive\n"
        f"Failed phase: {phase}\nIndependent verification unavailable.\n"
        + "\n".join(result["errors"])
        + "\n"
        "No information-removal claim. Preserved artifacts require readback before acceptance.\n"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "verify", "verify-receipts", "freeze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--deadline-epoch", type=int, default=0)
    parser.add_argument("--max-download-bytes", type=int, default=3221225472)
    parser.add_argument("--enable-experimental", action="store_true")
    parser.add_argument("--failed-phase", default="unknown")
    parser.add_argument("--phase-exit", type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps({"phase": args.phase, "state": "admission"}), flush=True)
    if args.phase == "run" and not args.enable_experimental:
        raise ExperimentRefused("experimental execution disabled; pass explicit opt-in")
    root = admit_root(args.root)
    contract_path = args.contract if args.contract.is_absolute() else root / args.contract
    contract = load_contract(contract_path, enable=True)
    approval = json.loads((root / "approval.json").read_text())
    if (
        approval.get("contract_sha256") != APPROVED_V1_SHA256
        or approval.get("approved") is not True
    ):
        raise ExperimentRefused("matching explicit approval receipt absent")
    if args.phase == "freeze":
        from .runner import freeze_dataset

        print(json.dumps(freeze_dataset(root, contract)), flush=True)
        return 0
    if args.deadline_epoch <= time.time() or args.deadline_epoch > time.time() + 21600:
        raise ExperimentRefused("missing, expired or expanded batch deadline")
    if args.max_download_bytes != contract["resources"]["download_max_bytes"]:
        raise ExperimentRefused("download budget differs from approval")
    check_bundle(root)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.phase in {"run", "verify"}:
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
        )
    if args.phase == "prepare":
        from .prepare import prepare

        with Watchdog(
            root, contract, min(args.deadline_epoch - 600, int(time.time()) + 1800), "preparation"
        ) as watch:
            atomic_json(root / "host-before.json", watch.initial)
            receipt = prepare(
                root,
                contract,
                args.deadline_epoch,
                progress=lambda event: watch.progress("preparing", detail=event),
            )
            atomic_json(root / "preparation.json", receipt)
    elif args.phase == "run":
        from .runner import run, runtime_manifest

        atomic_json(root / "manifest.json", runtime_manifest(root, contract))
        result = run(root, contract, args.deadline_epoch)
        print(
            json.dumps({"phase": "run", "verdict": result["verdict"], "status": result["status"]}),
            flush=True,
        )
    elif args.phase == "verify":
        from .runner import verify

        result = verify(root, contract, args.deadline_epoch)
        print(
            json.dumps(
                {
                    "phase": "verify",
                    "verdict": result["result"]["verdict"],
                    "status": result["verification"]["status"],
                }
            ),
            flush=True,
        )
        if result["verification"]["status"] != "verified":
            return 1
    else:
        print(json.dumps(receipts_only(root, args.failed_phase, args.phase_exit)), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(
            json.dumps({"state": "failed", "type": type(error).__name__, "error": str(error)}),
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
