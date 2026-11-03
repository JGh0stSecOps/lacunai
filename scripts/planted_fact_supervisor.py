"""Local-host bounded admission, immutable inputs, and receiver audit for the pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path
from typing import Any, cast

from lacunai.core.model import active_phase, checkpoint_inventory, progress
from lacunai.core.spec import bounded_file, digest
from lacunai.experimental.artifacts import atomic_json
from lacunai.experimental.resources import Watchdog, host_sample

LIMITS = {
    "disk_max_bytes": 10 * 1024**3,
    "min_host_available_bytes": 16 * 1024**3,
    "min_root_free_bytes": 8 * 1024**3,
    "min_bulk_free_bytes": 100 * 1024**3,
    "max_load1": 40,
    "memory_psi_some_avg10": 1.0,
    "io_psi_full_avg10": 5.0,
    "sampling_interval_seconds": 5,
    "pressure_consecutive_samples": 3,
}


def verify_inputs(bundle: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    """A receipt authorizes exact bytes, never a moving branch or directory."""
    manifest = json.loads(bounded_file(bundle / "bundle.json"))
    if receipt.get("approved") is not True or receipt.get("bundle_sha256") != digest(manifest):
        raise ValueError("explicit approval for this exact batch is absent")
    if (
        receipt.get("runtime_seconds") != 10800
        or receipt.get("disk_max_bytes") != LIMITS["disk_max_bytes"]
    ):
        raise ValueError("approval limits differ from the fixed batch")
    actual = {
        str(p.relative_to(bundle))
        for p in bundle.rglob("*")
        if p.is_file() and p != bundle / "bundle.json"
    }
    required = {
        "source-manifest.json",
        "scripts/planted_fact_pilot.py",
        "scripts/planted_fact_supervisor.py",
        "examples/planted-fact/target.json",
        "examples/planted-fact/plant-data.json",
        "src/lacunai/core/service.py",
    }
    if set(manifest["files"]) != actual or not required <= actual:
        raise ValueError("bundle manifest does not cover every executable and input")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("bundle path escapes its root")
        if hashlib.sha256(bounded_file(bundle / relative, 64 * 1024**2)).hexdigest() != expected:
            raise ValueError(f"bundle input differs: {name}")
    return cast(dict[str, Any], manifest)


def stage_source(cache: Path, output: Path, manifest: dict[str, Any]) -> None:
    """Copy only pinned complete files; leave cached partials and hardlinks untouched."""
    metadata = {row["path"]: row["bytes"] for row in manifest["metadata"]}
    if len(metadata) != len(manifest["metadata"]) or set(metadata) != set(manifest["files"]):
        raise ValueError("source metadata file set differs")
    for name, size in metadata.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source filename escapes checkpoint")
        source = cache / relative
        if (
            any(p.is_symlink() for p in (source, *source.parents))
            or not source.is_file()
            or source.stat().st_size != size
        ):
            raise ValueError("source byte count differs from measured metadata")
    output.mkdir(mode=0o700)
    total = sum(metadata.values())
    completed = 0
    progress("source_copy", completed=0, total=total, unit="bytes")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source filename escapes checkpoint")
        source = cache / relative
        if any(p.is_symlink() for p in (source, *source.parents)) or not source.is_file():
            raise ValueError("source is not an unredirected regular file")
        destination = output / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        hasher = hashlib.sha256()
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            while block := incoming.read(8 * 1024**2):
                outgoing.write(block)
                hasher.update(block)
                completed += len(block)
                progress("source_copy", completed=completed, total=total, unit="bytes")
            outgoing.flush()
        if hasher.hexdigest() != expected:
            raise ValueError(f"cached source hash differs: {name}")
        destination.chmod(0o400)
    if checkpoint_inventory(output, 4 * 1024**3) != manifest["files"]:
        raise ValueError("copied checkpoint readback differs")
    progress("source_copy", state="verified", completed=completed, total=total, unit="bytes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    progress("admission", state="active")
    if not args.apply:
        raise ValueError("supervisor is disabled without --apply")
    receipt = json.loads(bounded_file(args.approval))
    manifest = verify_inputs(args.bundle, receipt)
    root = args.run_root.absolute()
    if any(p.is_symlink() for p in (root, *root.parents)) or root.exists():
        raise ValueError("pilot root must be a new unredirected path")
    root.mkdir(mode=0o700, parents=False)
    deadline = int(time.time()) + 10800
    atomic_json(root / "approval.json", receipt)
    atomic_json(root / "bundle.json", manifest)
    admission = host_sample(root)
    atomic_json(root / "host-before.json", admission)
    atomic_json(root.parent / "host-before.json", admission)
    atomic_json(root / "status.json", {"status": "running", "verdict": "inconclusive"})
    source_manifest = json.loads(bounded_file(args.bundle / "source-manifest.json"))
    try:
        with Watchdog(root.parent, {"resources": LIMITS}, deadline, "planted_fact"):
            stage_source(args.source_cache, root / "source", source_manifest)
            # Reuse the host's already-installed Debian cryptography; never download/install.
            sys.path.append("/usr/lib/python3/dist-packages")
            sys.argv = [
                "planted_fact_pilot.py",
                "--source",
                str(root / "source"),
                "--workspace",
                str(root / "experiment"),
                "--source-manifest",
                str(args.bundle / "source-manifest.json"),
                "--target",
                str(args.bundle / "examples/planted-fact/target.json"),
                "--plant-data",
                str(args.bundle / "examples/planted-fact/plant-data.json"),
                "--deadline-epoch",
                str(deadline - 600),
                "--apply",
            ]
            pilot_status = 0
            with active_phase("planted_fact_batch"):
                try:
                    runpy.run_path(
                        str(args.bundle / "scripts/planted_fact_pilot.py"), run_name="__main__"
                    )
                except SystemExit as exc:
                    if exc.code not in (None, 0, 1):
                        raise RuntimeError(f"pilot exited with status {exc.code}") from exc
                    pilot_status = int(exc.code or 0)
            # The signed scientific verdict belongs to the pilot. This confirms retained bytes only.
            progress("artifact_readback", state="active")
            artifacts: dict[str, str] = {}
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError("unexpected linked artifact")
                if path.is_file() and "/logs/" not in str(path) and path.name != "status.json":
                    if time.time() >= deadline:
                        raise TimeoutError("receiver audit exceeded total runtime")
                    with active_phase("artifact_hashing"), path.open("rb") as stream:
                        value = hashlib.file_digest(stream, "sha256").hexdigest()
                    artifacts[str(path.relative_to(root))] = value
            atomic_json(root / "receiver-hashes.json", artifacts)
            atomic_json(
                root / "status.json",
                {
                    "status": "complete",
                    "artifact_files": len(artifacts),
                    "pilot_exit_code": pilot_status,
                    "claim": "batch exited and files rehashed; see scientific comparison verdict",
                },
            )
            progress("artifact_readback", state="complete", files=len(artifacts))
    except BaseException as exc:
        atomic_json(
            root / "status.json", {"status": "failed", "error": str(exc), "verdict": "inconclusive"}
        )
        progress("supervisor", state="failed", error=str(exc))
        raise
    return pilot_status


if __name__ == "__main__":
    raise SystemExit(main())
