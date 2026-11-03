"""Full development re-evaluation of the signed stored rmu-001-retry01 candidate; no training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chinese_baselines as baseline  # noqa: E402
import chinese_gradient_candidate as gradient  # noqa: E402
import chinese_iteration as iteration  # noqa: E402
import chinese_rmu_candidate as rmu  # noqa: E402
from lacunai.core.model import HFRuntime, active_phase, checkpoint_inventory  # noqa: E402
from lacunai.core.spec import RuntimeConfig, bounded_file, digest  # noqa: E402
from lacunai.experimental import tensors  # noqa: E402
from lacunai.experimental.artifacts import atomic_json, secure_path  # noqa: E402
from lacunai.experimental.resources import Watchdog, host_sample  # noqa: E402
from lacunai.restore.checkpoints import prepare_snapshot  # noqa: E402
from lacunai.verify import language as evaluation  # noqa: E402
from lacunai.verify.reports import verify_report  # noqa: E402

PARENT_FILES = {
    "report_sha256": "report.json",
    "manifest_sha256": "manifest.json",
    "edit_spec_sha256": "edit-spec.json",
    "delta_manifest_sha256": "after-storage/manifest.json",
    "delta_bytes_sha256": "after-storage/changed.safetensors",
}


def verify_parent(
    parent: Path, pins: dict[str, str], public_key: Path, key_sha256: str, source_fingerprint: str
) -> dict[str, Any]:
    if parent.name != "rmu-001-retry01":
        raise ValueError("only the root-selected rmu-001-retry01 parent may be replayed")
    if hashlib.sha256(bounded_file(public_key, 32)).hexdigest() != key_sha256:
        raise ValueError("trusted public key hash differs")
    verification = verify_report(parent, public_key)
    signed = verification["envelope"]["files_sha256"]
    if set(pins) != set(PARENT_FILES):
        raise ValueError("parent pins incomplete")
    for key, relative in PARENT_FILES.items():
        path = secure_path(parent, relative)
        if tensors.file_hash(path) != pins[key] or signed.get(relative) != pins[key]:
            raise ValueError("parent artifact pin/signature differs: " + relative)
    report = baseline.pinned_json(parent / "report.json", pins["report_sha256"])
    manifest = baseline.pinned_json(parent / "manifest.json", pins["manifest_sha256"])
    edit = baseline.pinned_json(parent / "edit-spec.json", pins["edit_spec_sha256"])
    status = json.loads(bounded_file(secure_path(parent, "status.json")))
    if signed.get("status.json") != tensors.file_hash(parent / "status.json"):
        raise ValueError("parent terminal state is not signed")
    if (
        status.get("state") != "complete"
        or status.get("verdict") not in ("failed", "passed")
        or report.get("verdict") != status["verdict"]
        or manifest.get("method") != "rmu_candidate"
        or edit.get("method") != "rmu_candidate"
        or manifest.get("objective_spec") != edit
        or status.get("method") != "rmu_candidate"
        or manifest.get("checkpoint_fingerprint") != source_fingerprint
        or digest(manifest.get("source_inventory")) != source_fingerprint
    ):
        raise ValueError("parent terminal RMU/source identity differs")
    return {
        "phase_path": str(parent),
        **pins,
        "trusted_public_key_sha256": key_sha256,
        "signature_sha256": tensors.file_hash(parent / "signature.json"),
        "verification": verification,
        "checkpoint_fingerprint": source_fingerprint,
        "source_inventory": manifest["source_inventory"],
        "edit": edit,
        "manifest": manifest,
    }


def parent_unchanged(parent: Path, evidence: dict[str, Any]) -> None:
    for key, relative in PARENT_FILES.items():
        if tensors.file_hash(secure_path(parent, relative)) != evidence[key]:
            raise ValueError("parent artifact changed during replay")
    if tensors.file_hash(parent / "signature.json") != evidence["signature_sha256"]:
        raise ValueError("parent signature changed during replay")


def load_stored(
    runtime: Any, parent: Path, reference: dict[str, Any], outer: dict[str, Any]
) -> dict[str, Any]:
    loaded = tensors.load_delta(runtime.model, parent / "after-storage", reference, outer)
    expected = {name: row["sha256"] for name, row in loaded["diff"]["tensors"].items()}
    if set(expected) != set(rmu.NAMES):
        raise ValueError("parent after-image must cover all three allowed tensors")
    actual = {name: tensors.tensor_hash(runtime.model.state_dict()[name]) for name in expected}
    if actual != expected:
        raise ValueError("actual replay tensor hashes differ from parent after-image")
    return {
        "parent_delta_sha256": loaded["delta_sha256"],
        "loaded_tensor_hashes": actual,
        "source_reference_fingerprint": reference["fingerprint"],
        "no_training_performed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "root",
        "source",
        "parent-phase",
        "trusted-public-key",
        "training-contract",
        "training-dataset",
        "dataset",
        "before",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in (
        "name",
        "source-fingerprint",
        "parent-report-sha256",
        "parent-manifest-sha256",
        "edit-spec-sha256",
        "delta-manifest-sha256",
        "delta-bytes-sha256",
        "trusted-public-key-sha256",
        "training-contract-sha256",
        "training-dataset-sha256",
        "dataset-sha256",
        "before-sha256",
    ):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print(
            json.dumps(
                {"phase": "replay-screen", "state": "plan_only", "training_performed": False}
            ),
            flush=True,
        )
        return 0
    root, parent = args.root.absolute(), args.parent_phase.absolute()
    if (
        not args.name.replace("-", "").replace("_", "").isalnum()
        or parent.parent != root
        or any(p.is_symlink() for p in (root, *root.parents))
    ):
        raise ValueError("invalid replay workspace/name/parent")
    directory = root / args.name
    directory.mkdir(mode=0o700, exist_ok=False)
    status: dict[str, Any] = {
        "phase": "replay-screen",
        "operation": "stored_candidate_re_evaluation",
        "method": "rmu_candidate",
        "training_performed": False,
        "state": "running",
        "verdict": "inconclusive",
    }
    atomic_json(directory / "status.json", status)
    try:
        with active_phase("replay_preparation"):
            pins = {
                "report_sha256": args.parent_report_sha256,
                "manifest_sha256": args.parent_manifest_sha256,
                "edit_spec_sha256": args.edit_spec_sha256,
                "delta_manifest_sha256": args.delta_manifest_sha256,
                "delta_bytes_sha256": args.delta_bytes_sha256,
            }
            evidence = verify_parent(
                parent,
                pins,
                args.trusted_public_key,
                args.trusted_public_key_sha256,
                args.source_fingerprint,
            )
            historical = baseline.pinned_json(args.training_contract, args.training_contract_sha256)
            corpus = baseline.pinned_json(args.training_dataset, args.training_dataset_sha256)
            spec = evidence["edit"]
            _, outer = rmu.candidate_contract(spec, historical, corpus)
            dataset = baseline.pinned_json(args.dataset, args.dataset_sha256)
            if set(dataset) != {"dev_primary", "dev_retain"} or any(
                dataset[k] != corpus[k] for k in dataset
            ):
                raise ValueError("replay requires the frozen development-only corpus")
            before = baseline.pinned_json(args.before, args.before_sha256)
            before_manifest = baseline.verify_before_source(args.before, args.source_fingerprint)
            config = RuntimeConfig(max_seconds=None, max_new_tokens=64, seed=42, torch_threads=2)
            if not (root / "host-before.json").exists():
                atomic_json(root / "host-before.json", host_sample(root))
        with Watchdog(root, {"resources": iteration.LIMITS}, None, args.name) as watch:
            with active_phase("replay_original_load"):
                inventory = checkpoint_inventory(args.source, config.max_model_bytes)
                if digest(inventory) != args.source_fingerprint:
                    raise ValueError("replay source differs from pinned original")
                runtime = HFRuntime(args.source, config)
                if runtime.inventory != inventory:
                    raise ValueError("source changed during loading")
            if any(runtime.storage_dtypes.get(name) != "BF16" for name in rmu.NAMES):
                raise ValueError("replay requires original BF16 tensor storage")
            runtime.contract["evaluation"] = {"receptive_choice_count": 4}
            code_root = Path(__file__).resolve().parents[1]
            files = [
                Path(__file__),
                Path(rmu.__file__),
                Path(gradient.__file__),
                Path(baseline.__file__),
                Path(iteration.__file__),
                *sorted((code_root / "src").rglob("*.py")),
            ]
            fields = baseline.common_manifest_fields(
                args.source_fingerprint,
                {
                    str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files
                },
                {},
            )
            atomic_json(
                directory / "manifest.json",
                {
                    **fields,
                    **status,
                    "entrypoint": "scripts/chinese_replay_candidate.py",
                    "objective_spec": spec,
                    "parent": {k: v for k, v in evidence.items() if k not in ("edit", "manifest")},
                    "source_inventory": inventory,
                    "runtime": asdict(config),
                    "resources": iteration.LIMITS,
                    "overall_deadline": None,
                    "training_contract_sha256": args.training_contract_sha256,
                    "training_dataset_sha256": args.training_dataset_sha256,
                    "dataset_bytes_sha256": args.dataset_sha256,
                    "dataset_sha256": digest(dataset),
                    "before_sha256": args.before_sha256,
                    "before_manifest": before_manifest,
                    "outer_edits": outer["edits"],
                    "evaluation_scope": "full development; no early rejection",
                    "scorer_version": evaluation.SCORER_VERSION,
                    "claim_limit": (
                        "stored candidate replay; no retraining and no independent holdout claim"
                    ),
                },
            )
            with active_phase("replay_snapshot"):
                snapshot = prepare_snapshot(args.source, directory, rmu.NAMES)
                iteration.bind_snapshot(runtime, snapshot)
            reference = None
            try:
                with active_phase("replay_reference"):
                    reference = tensors.capture_reference(runtime.model, outer)
                with active_phase("replay_stored_candidate_load"):
                    receipt = load_stored(runtime, parent, reference, outer)
                    if receipt["parent_delta_sha256"] != args.delta_bytes_sha256:
                        raise ValueError("loaded delta differs from explicit parent pin")
                    atomic_json(directory / "replay-load.json", receipt)
                scored = iteration.score(runtime, dataset, directory / "evaluation", watch, True)
                compared = iteration.compare(before, scored)
                status.update(result=scored, comparison=compared, verdict=compared["verdict"])
            finally:
                if reference is not None:
                    with active_phase("replay_exact_memory_restore"):
                        tensors.restore_reference(runtime.model, reference, outer)
                    atomic_json(
                        directory / "restore.json",
                        {
                            "exact_original_tensors_restored": True,
                            "canonical_reference_fingerprint": reference["fingerprint"],
                            "checkpoint_fingerprint": args.source_fingerprint,
                        },
                    )
                with active_phase("replay_source_and_parent_readback"):
                    if checkpoint_inventory(args.source, config.max_model_bytes) != inventory:
                        raise ValueError("original checkpoint changed during replay")
                    parent_unchanged(parent, evidence)
            status["state"] = "complete"
            atomic_json(directory / "status.json", status)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(directory / "status.json", status)
        print(
            json.dumps({"phase": "replay-screen", "state": "failed", "error": status["error"]}),
            file=sys.stderr,
            flush=True,
        )
        raise
    print(json.dumps({k: status[k] for k in ("phase", "state", "verdict")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
