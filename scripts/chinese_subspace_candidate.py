"""Pinned retain-nullspace diagnosis and root-selected edits; no automatic selection."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chinese_baselines as baseline  # noqa: E402
import chinese_gradient_candidate as gradient  # noqa: E402
import chinese_iteration as iteration  # noqa: E402
from lacunai.core.model import HFRuntime, active_phase, checkpoint_inventory  # noqa: E402
from lacunai.core.spec import RuntimeConfig, digest  # noqa: E402
from lacunai.experimental import tensors  # noqa: E402
from lacunai.experimental.artifacts import atomic_json  # noqa: E402
from lacunai.experimental.resources import Watchdog, host_sample  # noqa: E402
from lacunai.restore.checkpoints import prepare_snapshot, write_edited_checkpoint  # noqa: E402


def edit_contract(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate authorization before any model load, snapshot, or tensor mutation."""
    if set(spec) != {"schema_version", "method", "map_sha256", "layers", "k", "strength"}:
        raise ValueError("unexpected structured edit fields")
    layers = spec["layers"]
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["method"] != "retain-nullspace"
        or not isinstance(spec["map_sha256"], str)
        or len(spec["map_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in spec["map_sha256"])
        or type(layers) is not list
        or not 1 <= len(layers) <= 8
        or any(type(n) is not int or not 0 <= n < 24 for n in layers)
        or len(set(layers)) != len(layers)
        or type(spec["k"]) is not int
        or spec["k"] not in (1, 4, 16)
        or type(spec["strength"]) not in (int, float)
        or spec["strength"] not in (0.25, 0.5, 1.0)
    ):
        raise ValueError("edit differs from root-selected retain-nullspace contract")
    return {
        "target": {"revision": baseline.MODEL_REVISION},
        "edits": {
            "allowed_tensors": [f"model.layers.{n}.mlp.down_proj.weight" for n in layers],
            "expected_shape": [896, 4864],
            "max_tensor_relative_frobenius": "0.10",
            "max_model_relative_frobenius": "0.02",
        },
        "resources": dict(iteration.LIMITS),
    }


def verify_diagnosis(diagnosis: dict[str, Any], spec: dict[str, Any], source: str) -> None:
    from lacunai.map.localize import record_digest

    if diagnosis.get("checkpoint_fingerprint") != source:
        raise ValueError("diagnosis source differs from pinned original")
    if spec["map_sha256"] != record_digest(diagnosis):
        raise ValueError("root selection refers to another diagnosis")


def evaluation_inputs(args: Any) -> tuple[Any, Any, Any]:
    supplied = (args.dataset, args.dataset_sha256, args.before, args.before_sha256)
    if args.phase != "screen":
        if any(supplied):
            raise ValueError("diagnose/persist must not receive evaluation inputs")
        return None, None, None
    if not all(supplied):
        raise ValueError("screen requires pinned development dataset and baseline")
    dataset = baseline.pinned_json(args.dataset, args.dataset_sha256)
    if set(dataset) != {"dev_primary", "dev_retain"} or any(not dataset[k] for k in dataset):
        raise ValueError("screen requires nonempty development-only data; holdout refused")
    before = baseline.pinned_json(args.before, args.before_sha256)
    manifest = baseline.verify_before_source(args.before, args.source_fingerprint)
    return dataset, before, manifest


def apply_replacements(
    runtime: Any, replacements: dict[str, Any], contract: dict[str, Any]
) -> None:
    names = contract["edits"]["allowed_tensors"]
    if set(replacements) != set(names):
        raise ValueError("operator replacements differ from explicit tensor allowlist")
    torch = import_module("torch")
    parameters = dict(runtime.model.named_parameters())
    # Validate every replacement before copying any tensor.
    for name in names:
        value, source = replacements[name], parameters[name]
        if value.shape != source.shape or value.dtype != source.dtype:
            raise ValueError("replacement shape/dtype mismatch")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("non-finite replacement")
    with torch.no_grad():
        for name in names:
            parameters[name].copy_(replacements[name])


def storage_measurement(
    runtime: Any, reference: dict[str, Any], contract: dict[str, Any], path: Path
) -> dict[str, Any]:
    try:
        result = gradient.storage_measurement(runtime, reference, contract, path)
    finally:
        if path.exists():
            measured = json.loads(path.read_text())
            measured["measurement_scope"] = (
                "storage-roundtrip idempotence; replacements already rounded by operator"
            )
            measured["retained_delta_energy_interpretation"] = (
                "repeated-roundtrip energy ratio; not FP32 projection retention"
            )
            atomic_json(path, measured)
    return {
        **result,
        "measurement_scope": measured["measurement_scope"],
        "retained_delta_energy_interpretation": measured["retained_delta_energy_interpretation"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["diagnose", "screen", "persist"])
    for flag in ("root", "source"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("source-fingerprint", "name"):
        parser.add_argument("--" + flag, required=True)
    for flag in ("corpus", "diagnosis", "edit-spec", "dataset", "before"):
        parser.add_argument("--" + flag, type=Path)
        parser.add_argument("--" + flag + "-sha256")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print(
            json.dumps({"phase": args.phase, "state": "plan_only", "overall_deadline": None}),
            flush=True,
        )
        return 0
    if not args.name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid run name")
    root = args.root.absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("linked workspace refused")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = root / args.name
    directory.mkdir(mode=0o700, exist_ok=False)
    status: dict[str, Any] = {
        "phase": args.phase,
        "state": "running",
        "verdict": "inconclusive",
        "method": "retain-nullspace",
    }
    atomic_json(directory / "status.json", status)
    try:
        with active_phase("subspace_preparation"):
            dataset, before, before_manifest = evaluation_inputs(args)
            spec, contract, diagnosis, corpus = None, None, None, None
            if args.phase == "diagnose":
                if not all((args.corpus, args.corpus_sha256)):
                    raise ValueError("diagnose requires pinned corpus")
                if any(
                    (args.diagnosis, args.diagnosis_sha256, args.edit_spec, args.edit_spec_sha256)
                ):
                    raise ValueError("diagnose does not accept a candidate selection")
                corpus = baseline.pinned_json(args.corpus, args.corpus_sha256)
            else:
                if not all(
                    (args.diagnosis, args.diagnosis_sha256, args.edit_spec, args.edit_spec_sha256)
                ):
                    raise ValueError(
                        "screen/persist require pinned diagnosis and explicit root edit spec"
                    )
                if any((args.corpus, args.corpus_sha256)):
                    raise ValueError("candidate execution reuses diagnosis; corpus input refused")
                spec = baseline.pinned_json(args.edit_spec, args.edit_spec_sha256)
                contract = edit_contract(spec)
                diagnosis = baseline.pinned_json(args.diagnosis, args.diagnosis_sha256)
                verify_diagnosis(diagnosis, spec, args.source_fingerprint)
                atomic_json(directory / "edit-spec.json", spec)
            config = RuntimeConfig(max_seconds=None, max_new_tokens=64, seed=42, torch_threads=2)
            if not (root / "host-before.json").exists():
                atomic_json(root / "host-before.json", host_sample(root))
        with Watchdog(root, {"resources": iteration.LIMITS}, None, args.name) as watch:
            with active_phase("subspace_original_load"):
                inventory = checkpoint_inventory(args.source, config.max_model_bytes)
                if digest(inventory) != args.source_fingerprint:
                    raise ValueError("source differs from pinned original")
                runtime = HFRuntime(args.source, config)
                if runtime.inventory != inventory:
                    raise ValueError("source changed during loading")
            runtime.contract["evaluation"] = {"receptive_choice_count": 4}
            code_root = Path(__file__).resolve().parents[1]
            files = [
                Path(__file__),
                Path(baseline.__file__),
                Path(gradient.__file__),
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
                    "entrypoint": "scripts/chinese_subspace_candidate.py",
                    "phase": args.phase,
                    "method": "retain-nullspace",
                    "objective_spec": spec,
                    "source_inventory": inventory,
                    "runtime": asdict(config),
                    "resources": iteration.LIMITS,
                    "overall_deadline": None,
                    "edit_spec_sha256": args.edit_spec_sha256,
                    "diagnosis_sha256": args.diagnosis_sha256,
                    "corpus_sha256": args.corpus_sha256,
                    "dataset_bytes_sha256": args.dataset_sha256,
                    "dataset_sha256": digest(dataset) if dataset is not None else None,
                    "before_sha256": args.before_sha256,
                    "before_manifest": before_manifest,
                    "outer_edits": contract["edits"] if contract else None,
                    "claim_limit": (
                        "association-based development experiment; "
                        "no literacy-erasure or absence proof"
                    ),
                    "numerics": {
                        "runtime_dtype": "float32",
                        "stored_dtype": "BF16",
                        "svd_dtype": "float64",
                        "storage_measurement_scope": (
                            "operator returns source-storage-rounded replacements; repeated "
                            "roundtrip measures idempotence, not pre-round projection energy"
                        ),
                        "bound_tolerance": 1e-9,
                        "map_parameters": import_module("lacunai.map.retain_nullspace").PARAMETERS,
                        "reproducibility_limits": (
                            "CPU LAPACK SVD bases may vary for degenerate singular values"
                        ),
                    },
                },
            )
            if args.phase == "diagnose":
                with active_phase("subspace_all_layer_diagnosis"):
                    mapper = import_module("lacunai.map.retain_nullspace")
                    if corpus is None:
                        raise ValueError("missing diagnosis corpus")
                    diagnosis = mapper.diagnose(
                        runtime, corpus["corpora"], directory / "map-artifacts"
                    )
                    atomic_json(directory / "diagnosis.json", diagnosis)
            else:
                if contract is None or spec is None or diagnosis is None:
                    raise ValueError("missing candidate inputs")
                names = contract["edits"]["allowed_tensors"]
                if any(runtime.storage_dtypes.get(n) != "BF16" for n in names):
                    raise ValueError("candidate requires source BF16 storage")
                with active_phase("subspace_snapshot"):
                    snapshot = prepare_snapshot(args.source, directory, names)
                    iteration.bind_snapshot(runtime, snapshot)
                reference = None
                try:
                    with active_phase("subspace_reference"):
                        reference = tensors.capture_reference(runtime.model, contract)
                    with active_phase("subspace_proposal_and_apply"):
                        operator = import_module("lacunai.erase.retain_nullspace")
                        replacements, evidence = operator.propose(
                            runtime, diagnosis, args.diagnosis.parent / "map-artifacts", spec
                        )
                        watch.check()
                        apply_replacements(runtime, replacements, contract)
                        del replacements
                        atomic_json(directory / "operator.json", evidence)
                    with active_phase("subspace_storage_validation"):
                        status["storage_delta"] = storage_measurement(
                            runtime, reference, contract, directory / "storage-delta.json"
                        )
                    with active_phase("subspace_after_image_evidence"):
                        status["delta_manifest"] = gradient.save_after_image(
                            runtime, reference, contract, directory, root
                        )
                    if args.phase == "screen":
                        status.update(gradient.screen(runtime, dataset, before, directory, watch))
                    else:
                        expected = iteration.persisted_tensor_hashes(runtime, names)
                        replacements = {
                            n: dict(runtime.model.named_parameters())[n].detach() for n in names
                        }
                        output = root / (args.name + "-checkpoint")
                        with active_phase("subspace_checkpoint_write"):
                            receipt = write_edited_checkpoint(
                                args.source, output, replacements, snapshot
                            )
                        del replacements
                        status.update(checkpoint=str(output), snapshot_id=receipt["snapshot_id"])
                finally:
                    if reference is not None:
                        with active_phase("subspace_exact_memory_restore"):
                            tensors.restore_reference(runtime.model, reference, contract)
                        atomic_json(
                            directory / "restore.json",
                            {
                                "exact_original_tensors_restored": True,
                                "canonical_reference_fingerprint": reference["fingerprint"],
                                "checkpoint_fingerprint": args.source_fingerprint,
                            },
                        )
                    del reference
            del runtime
            gc.collect()
            if checkpoint_inventory(args.source, config.max_model_bytes) != inventory:
                raise ValueError("original checkpoint changed during execution")
            if args.phase == "persist":
                persisted = checkpoint_inventory(output, config.max_model_bytes)
                if persisted != {n: r["sha256"] for n, r in receipt["after_files"].items()}:
                    raise ValueError("persisted bytes differ from snapshot receipt")
                runtime = iteration.reload_checkpoint(output, config, persisted, expected, None)
                atomic_json(
                    directory / "reload.json",
                    {
                        "inventory": persisted,
                        "tensor_hashes": expected,
                        "loaded": True,
                        "checkpoint_fingerprint": runtime.fingerprint,
                    },
                )
                status.update(
                    output_fingerprint=runtime.fingerprint, reloaded_tensor_hashes=expected
                )
            status["state"] = "complete"
            atomic_json(directory / "status.json", status)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(directory / "status.json", status)
        print(
            json.dumps({"phase": args.phase, "state": "failed", "error": status["error"]}),
            file=sys.stderr,
            flush=True,
        )
        raise
    print(json.dumps({k: status[k] for k in ("phase", "state", "verdict")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
