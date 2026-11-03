"""Execute a pinned root-authored gradient candidate; no automatic candidate selection."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
from dataclasses import asdict
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chinese_baselines as baseline  # noqa: E402
import chinese_iteration as iteration  # noqa: E402
from lacunai.core.model import HFRuntime, active_phase, checkpoint_inventory, progress  # noqa: E402
from lacunai.core.spec import RuntimeConfig, digest  # noqa: E402
from lacunai.experimental import tensors  # noqa: E402
from lacunai.experimental.artifacts import atomic_json  # noqa: E402
from lacunai.experimental.resources import Watchdog, host_sample  # noqa: E402
from lacunai.restore.checkpoints import prepare_snapshot, write_edited_checkpoint  # noqa: E402

SPEC = {
    "schema_version": 1,
    "method": "gradient_candidate",
    "mode": "graddiff",
    "steps": 60,
    "optimizer": "SGD",
    "learning_rate": "0.1",
    "alpha": "1.0",
    "gamma": "1.0",
    "gradient_clip": "1.0",
    "sequence_tokens": 128,
    "allowed_tensors": baseline.NAMES,
    "inner_tensor_relative_frobenius": "0.0195",
    "inner_model_relative_frobenius": "0.0024",
    "outer_tensor_relative_frobenius": "0.02",
    "outer_model_relative_frobenius": "0.0025",
}


NPO_SPEC = {
    **SPEC,
    "method": "npo_kl_candidate",
    "mode": "npo_kl",
    "learning_rate": "0.01",
    "beta": "0.1",
    "retain_objective": "KL(reference||current):mean_answer_tokens",
}


def candidate_contract(
    spec: dict[str, Any], historical: dict[str, Any], dataset: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline.validate_training(historical, dataset)
    if spec.get("method") == "gradient_candidate":
        template, modes = SPEC, baseline.MODES
    elif spec.get("method") == "npo_kl_candidate":
        template, modes = NPO_SPEC, ("npo_kl", "retain_kl")
    else:
        raise ValueError("unknown root-selected candidate method")
    expected = {**template, "mode": spec.get("mode")}
    if spec.get("mode") not in modes or spec != expected:
        raise ValueError("candidate differs from the root-selected structured contract")
    # Equality alone admits True as 1: require exact JSON scalar/container types too.
    if any(type(spec[k]) is not type(expected[k]) for k in expected):
        raise ValueError("candidate JSON types differ from the structured contract")
    inner = copy.deepcopy(historical)
    inner["baseline"]["learning_rate"] = spec["learning_rate"]
    inner["edits"]["max_tensor_relative_frobenius"] = spec["inner_tensor_relative_frobenius"]
    inner["edits"]["max_model_relative_frobenius"] = spec["inner_model_relative_frobenius"]
    return inner, copy.deepcopy(historical)


def train_candidate(
    runtime: Any,
    reference: dict[str, Any],
    corpus: dict[str, Any],
    inner: dict[str, Any],
    spec: dict[str, Any],
    watch: Any,
) -> dict[str, Any]:
    """Route only validated structured methods; never execute generated code."""
    if spec["method"] == "gradient_candidate":
        return baseline.train_control(runtime, inner, corpus, spec["mode"], watch, reference)
    if spec["method"] != "npo_kl_candidate":
        raise ValueError("unknown training backend")
    with active_phase("npo_kl_training"):
        backend = import_module("lacunai.erase.gradient_objectives")
        result = backend.train(runtime, reference, corpus, inner, spec, watch)
        trace = result.get("trace", [])
        if (
            result.get("mode") != spec["mode"]
            or len(trace) != spec["steps"]
            or any(
                row.get("completed") != i or row.get("total") != spec["steps"]
                for i, row in enumerate(trace, 1)
            )
        ):
            raise ValueError("NPO/KL backend mode or step trace differs from frozen contract")
        return dict(result)


def measure_delta(model: Any, reference: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Bounded CPU reductions over every allowed element, without LLM arithmetic."""
    torch = import_module("torch")
    state = model.state_dict()
    per_tensor = {}
    progress("gradient_delta_measurement", completed=0, total=len(names), unit="tensors")
    for index, name in enumerate(names, 1):
        value, original = state[name], reference["original_tensors"][name]
        if value.shape != original.shape or value.dtype != original.dtype:
            raise ValueError("delta shape/dtype differs from reference")
        flat, before = value.detach().reshape(-1), original.detach().reshape(-1)
        changed, squares = 0, []
        for start in range(0, flat.numel(), 262144):
            delta = flat[start : start + 262144].to(torch.float64) - before[
                start : start + 262144
            ].to(torch.float64)
            if not bool(torch.isfinite(delta).all()):
                raise ValueError("non-finite delta")
            changed += int(torch.count_nonzero(delta).item())
            squares.append(float(torch.sum(delta * delta).item()))
        square = math.fsum(squares)
        if not math.isfinite(square):
            raise ValueError("non-finite delta norm")
        per_tensor[name] = {
            "changed_element_count": changed,
            "element_count": flat.numel(),
            "delta_norm_squared": square,
            "delta_norm": math.sqrt(square),
        }
        progress("gradient_delta_measurement", completed=index, total=len(names), unit="tensors")
    square = math.fsum(row["delta_norm_squared"] for row in per_tensor.values())
    return {
        "tensors": per_tensor,
        "changed_element_count": sum(row["changed_element_count"] for row in per_tensor.values()),
        "element_count": sum(row["element_count"] for row in per_tensor.values()),
        "delta_norm_squared": square,
        "delta_norm": math.sqrt(square),
    }


def storage_measurement(
    runtime: Any,
    reference: dict[str, Any],
    outer: dict[str, Any],
    artifact: Path | None = None,
) -> dict[str, Any]:
    names = outer["edits"]["allowed_tensors"]
    before = measure_delta(runtime.model, reference, names)
    validation_error = None
    try:
        validated = baseline.round_and_validate_storage(runtime, outer, reference)
    except tensors.TensorError as exc:
        validation_error = exc
        validated = {"passed": False, "error": str(exc)}
    after = measure_delta(runtime.model, reference, names)
    denominator = Decimal(str(before["delta_norm_squared"]))
    energy = str(Decimal(str(after["delta_norm_squared"])) / denominator) if denominator else None
    result = {
        "before_storage_roundtrip": before,
        "after_storage_roundtrip": after,
        "retained_delta_energy": energy,
        "retained_delta_energy_definition": "after delta squared norm / before delta squared norm",
        "zero_denominator": "undefined" if not denominator else None,
        "outer_bound_validation": validated,
        "stored_dtype": "BF16",
        "runtime_dtype": "float32",
    }
    if artifact is not None:
        atomic_json(artifact, result)
    if validation_error is not None:
        raise validation_error
    return result


def screen(
    runtime: Any, dataset: dict[str, Any], before: dict[str, Any], directory: Path, watch: Any
) -> dict[str, Any]:
    result: dict[str, Any] = {"verdict": "inconclusive", "screening": "partial"}
    with active_phase("gradient_recovery_preflight"):
        recovery = iteration.recovery_preflight(
            runtime, dataset, before, directory / "recovery-preflight", watch
        )
    result["recovery_preflight"] = recovery
    if not recovery["continue"]:
        return {
            **result,
            "verdict": recovery["verdict"],
            "screening": "rejected; full suite not executed",
        }
    with active_phase("gradient_collateral_preflight"):
        collateral = iteration.collateral_preflight(
            runtime, dataset, before, directory / "collateral-preflight", watch
        )
    result["collateral_preflight"] = collateral
    if not collateral["continue"]:
        return {
            **result,
            "verdict": collateral["verdict"],
            "screening": "rejected; full suite not executed",
        }
    scored = iteration.score(runtime, dataset, directory / "evaluation", watch, True)
    compared = iteration.compare(before, scored)
    return {
        **result,
        "result": scored,
        "comparison": compared,
        "verdict": compared["verdict"],
        "screening": "full development suite; no independent holdout claim",
    }


def save_after_image(
    runtime: Any, reference: dict[str, Any], outer: dict[str, Any], directory: Path, root: Path
) -> dict[str, Any]:
    delta_contract = copy.deepcopy(outer)
    delta_contract["resources"]["disk_max_bytes"] = iteration.LIMITS["disk_max_bytes"]
    manifest = tensors.save_delta(
        runtime.model, reference, directory / "after-storage", delta_contract, budget_root=root
    )
    if (
        tensors.file_hash(directory / "after-storage" / "changed.safetensors")
        != manifest["delta_sha256"]
    ):
        raise ValueError("saved stored-precision after-image failed readback")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["screen", "persist"])
    for flag in ("root", "source", "training-contract", "training-dataset", "edit-spec"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in (
        "source-fingerprint",
        "training-contract-sha256",
        "training-dataset-sha256",
        "edit-spec-sha256",
        "name",
    ):
        parser.add_argument("--" + flag, required=True)
    for flag in ("dataset", "before"):
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
    }
    atomic_json(directory / "status.json", status)
    try:
        with active_phase("gradient_preparation"):
            historical = baseline.pinned_json(args.training_contract, args.training_contract_sha256)
            corpus = baseline.pinned_json(args.training_dataset, args.training_dataset_sha256)
            spec = baseline.pinned_json(args.edit_spec, args.edit_spec_sha256)
            inner, outer = candidate_contract(spec, historical, corpus)
            status["mode"] = spec["mode"]
            status["method"] = spec["method"]
            dataset, before, before_manifest = None, None, None
            if args.phase == "screen":
                if not all((args.dataset, args.dataset_sha256, args.before, args.before_sha256)):
                    raise ValueError("screen requires pinned development dataset and baseline")
                dataset = baseline.pinned_json(args.dataset, args.dataset_sha256)
                if set(dataset) != {"dev_primary", "dev_retain"}:
                    raise ValueError("screen accepts development-only dataset; no holdout exposure")
                for split in dataset:
                    if dataset[split] != corpus[split]:
                        raise ValueError(
                            "development corpus differs from historical training contract"
                        )
                before = baseline.pinned_json(args.before, args.before_sha256)
                before_manifest = baseline.verify_before_source(
                    args.before, args.source_fingerprint
                )
            elif any((args.dataset, args.dataset_sha256, args.before, args.before_sha256)):
                raise ValueError("persist is write/reload only; do not supply evaluation inputs")
            config = RuntimeConfig(max_seconds=None, max_new_tokens=64, seed=42, torch_threads=2)
            atomic_json(directory / "edit-spec.json", spec)
            if not (root / "host-before.json").exists():
                atomic_json(root / "host-before.json", host_sample(root))
        with Watchdog(root, {"resources": iteration.LIMITS}, None, args.name) as watch:
            with active_phase("gradient_original_load"):
                inventory = checkpoint_inventory(args.source, config.max_model_bytes)
                if digest(inventory) != args.source_fingerprint:
                    raise ValueError("candidate source differs from pinned original")
                runtime = HFRuntime(args.source, config)
                if runtime.inventory != inventory:
                    raise ValueError("source changed during load")
                if any(runtime.storage_dtypes.get(n) != "BF16" for n in baseline.NAMES):
                    raise ValueError(
                        "root contract requires source BF16 storage on all allowed tensors"
                    )
            runtime.contract["evaluation"] = {"receptive_choice_count": 4}
            code_root = Path(__file__).resolve().parents[1]
            files = [
                Path(__file__),
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
                    "entrypoint": "scripts/chinese_gradient_candidate.py",
                    "phase": args.phase,
                    "method": spec["method"],
                    "objective_spec": spec,
                    "mode": spec["mode"],
                    "source_inventory": inventory,
                    "runtime": asdict(config),
                    "resources": iteration.LIMITS,
                    "overall_deadline": None,
                    "edit_spec_sha256": args.edit_spec_sha256,
                    "training_contract_sha256": args.training_contract_sha256,
                    "training_dataset_sha256": args.training_dataset_sha256,
                    "dataset_bytes_sha256": args.dataset_sha256,
                    "dataset_sha256": digest(dataset) if dataset is not None else None,
                    "before_sha256": args.before_sha256,
                    "before_manifest": before_manifest,
                    "training_settings": inner["baseline"],
                    "inner_edits": inner["edits"],
                    "outer_edits": outer["edits"],
                    "projection_margin": (
                        "inner bounds reserve storage-rounding headroom; "
                        "outer bounds checked after BF16 roundtrip"
                    ),
                    "claim_limit": (
                        "development-only candidate; persist does not evaluate or promote"
                    ),
                },
            )
            with active_phase("gradient_snapshot"):
                snapshot = prepare_snapshot(args.source, directory, baseline.NAMES)
                iteration.bind_snapshot(runtime, snapshot)
            reference = None
            try:
                with active_phase("gradient_reference"):
                    reference = tensors.capture_reference(runtime.model, inner)
                trained = train_candidate(runtime, reference, corpus, inner, spec, watch)
                atomic_json(directory / "training.json", trained)
                with active_phase("gradient_storage_roundtrip"):
                    measured = storage_measurement(
                        runtime, reference, outer, directory / "storage-delta.json"
                    )
                status["storage_delta"] = measured
                with active_phase("gradient_after_image_evidence"):
                    status["delta_manifest"] = save_after_image(
                        runtime, reference, outer, directory, root
                    )
                if args.phase == "screen":
                    if dataset is None or before is None:
                        raise ValueError("missing development inputs")
                    status.update(screen(runtime, dataset, before, directory, watch))
                else:
                    expected = iteration.persisted_tensor_hashes(runtime, baseline.NAMES)
                    replacements = {
                        n: dict(runtime.model.named_parameters())[n].detach()
                        for n in baseline.NAMES
                    }
                    output = root / (args.name + "-checkpoint")
                    with active_phase("gradient_checkpoint_write"):
                        receipt = write_edited_checkpoint(
                            args.source, output, replacements, snapshot
                        )
                    del replacements
                    status.update(checkpoint=str(output), snapshot_id=receipt["snapshot_id"])
            finally:
                if reference is not None:
                    with active_phase("gradient_exact_memory_restore"):
                        tensors.restore_reference(runtime.model, reference, outer)
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
                    raise ValueError("persisted after-image differs from receipt")
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
