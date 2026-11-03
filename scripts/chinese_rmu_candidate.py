"""Pinned RMU execution with complete four-family development recovery preflight."""

from __future__ import annotations

import argparse
import copy
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
from lacunai.verify import language as evaluation  # noqa: E402

NAMES = [f"model.layers.{layer}.mlp.down_proj.weight" for layer in (5, 6, 7)]
RECOVERY_PROTOCOL = "all_required_families_first_baseline_confirmed_variant_v1"
SPEC = {
    "schema_version": 1,
    "method": "rmu_candidate",
    "mode": "rmu",
    "steps": 60,
    "alpha": "1.0",
    "gamma": "1.0",
    "optimizer": "AdamW",
    "learning_rate": "0.0001",
    "betas": [0.9, 0.999],
    "eps": "0.00000001",
    "weight_decay": "0",
    "gradient_clip": "1.0",
    "steering_coeff": "20",
    "seed": 42,
    "sequence_tokens": 128,
    "hook_module": "model.layers.7",
    "allowed_tensors": NAMES,
    "inner_tensor_relative_frobenius": "0.0195",
    "inner_model_relative_frobenius": "0.0024",
    "outer_tensor_relative_frobenius": "0.02",
    "outer_model_relative_frobenius": "0.0025",
}


CONTEXT_SPEC = {
    **SPEC,
    "method": "rmu_context_candidate",
    "mode": "rmu_context",
    "mask_policy": (
        "user_content_and_answer_for_reading;answer_for_writing;"
        "user_content_and_answer_for_retain:v1"
    ),
    "variant_policy": "reading_cycle_existing_variants:v1",
}


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("method") == "rmu_candidate":
        template, modes = SPEC, ("rmu", "retain_mse")
    elif spec.get("method") == "rmu_context_candidate":
        template, modes = CONTEXT_SPEC, ("rmu_context", "retain_context_mse")
    else:
        raise ValueError("unknown root-selected RMU method")
    expected = {**template, "mode": spec.get("mode")}
    if (
        spec.get("mode") not in modes
        or spec != expected
        or any(type(spec[k]) is not type(v) for k, v in expected.items())
        or any(type(v) is not float for v in spec.get("betas", []))
    ):
        raise ValueError("RMU differs from exact root-selected structured specification")


def candidate_contract(
    spec: dict[str, Any], historical: dict[str, Any], dataset: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline.validate_training(historical, dataset)
    validate_spec(spec)
    outer = copy.deepcopy(historical)
    outer["edits"]["allowed_tensors"] = list(NAMES)
    outer["edits"]["expected_shape"] = [896, 4864]
    inner = copy.deepcopy(outer)
    inner["edits"]["max_tensor_relative_frobenius"] = spec["inner_tensor_relative_frobenius"]
    inner["edits"]["max_model_relative_frobenius"] = spec["inner_model_relative_frobenius"]
    return inner, outer


def train_candidate(
    runtime: Any,
    reference: dict[str, Any],
    corpus: dict[str, Any],
    inner: dict[str, Any],
    spec: dict[str, Any],
    watch: Any,
) -> dict[str, Any]:
    validate_spec(spec)
    with active_phase("rmu_training"):
        backend = import_module("lacunai.erase.representation_misdirection")
        result = backend.train(runtime, reference, corpus, inner, spec, watch)
        trace = result.get("trace", [])
        if (
            result.get("mode") != spec["mode"]
            or len(trace) != 60
            or any(
                row.get("completed") != i or row.get("total") != 60
                for i, row in enumerate(trace, 1)
            )
        ):
            raise ValueError("RMU backend mode or step trace differs from frozen contract")
        return dict(result)


storage_measurement = gradient.storage_measurement
save_after_image = gradient.save_after_image


def recovery_preflight(
    runtime: Any, dataset: dict[str, Any], before: dict[str, Any], path: Path, watch: Any
) -> dict[str, Any]:
    """Observe every required family before deciding; this never grants an early pass."""
    items = dataset["dev_primary"]
    if not before.get("development") or before["primary"]["spec"] != items:
        raise ValueError("recovery preflight requires matching development baseline")
    baseline_scores = evaluation.summarize(items, before["primary"]["raw"])
    selected = []
    for family in evaluation.CONTRACT["required_primary_families"]:
        variants = [
            (item, row["variant"])
            for item in items
            if item["family"] == family
            for row in baseline_scores["items"][item["id"]]["variants"]
            if row["correct"]
        ]
        if not variants:
            raise ValueError("required family has no baseline-confirmed variant")
        selected.append(variants[0])
    path.mkdir(mode=0o700, exist_ok=False)
    result: dict[str, Any] = {
        "continue": True,
        "efficacy_pass": False,
        "observations": [],
        "protocol": RECOVERY_PROTOCOL,
    }
    with (
        active_phase("rmu_four_family_recovery_preflight"),
        (path / "raw.jsonl").open("x") as stream,
    ):
        for completed, (item, number) in enumerate(selected, 1):
            watch.check()
            watch.progress(
                "recovery_preflight",
                completed=completed - 1,
                total=len(selected),
                unit="prompts",
            )
            row = {
                "item_id": item["id"],
                "family": item["family"],
                "variant": number,
                "prompt": item["prompts"][number],
                **runtime.generate(item["prompts"][number]),
            }
            row["score"] = evaluation.score_response(item, row["output"], row["truncated"])
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            result["observations"].append(row)
            watch.progress(
                "recovery_preflight", completed=completed, total=len(selected), unit="prompts"
            )
    if any(row["score"]["correct"] for row in result["observations"]):
        result.update(
            **{"continue": False},
            verdict="failed",
            reason="confirmed Chinese recovery on a baseline-qualified development probe",
        )
    atomic_json(path / "result.json", result)
    return result


def screen(
    runtime: Any, dataset: dict[str, Any], before: dict[str, Any], directory: Path, watch: Any
) -> dict[str, Any]:
    result: dict[str, Any] = {"verdict": "inconclusive", "screening": "partial"}
    recovery = recovery_preflight(runtime, dataset, before, directory / "recovery-preflight", watch)
    result["recovery_preflight"] = recovery
    if not recovery["continue"]:
        return {**result, "verdict": "failed", "screening": "rejected; full suite not executed"}
    with active_phase("rmu_collateral_preflight"):
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
        with active_phase("rmu_preparation"):
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
            with active_phase("rmu_original_load"):
                inventory = checkpoint_inventory(args.source, config.max_model_bytes)
                if digest(inventory) != args.source_fingerprint:
                    raise ValueError("candidate source differs from pinned original")
                runtime = HFRuntime(args.source, config)
                if runtime.inventory != inventory:
                    raise ValueError("source changed during load")
                if any(runtime.storage_dtypes.get(n) != "BF16" for n in NAMES):
                    raise ValueError(
                        "root contract requires source BF16 storage on all allowed tensors"
                    )
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
                    "entrypoint": "scripts/chinese_rmu_candidate.py",
                    "recovery_preflight_protocol": RECOVERY_PROTOCOL,
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
                    "training_settings": {
                        k: spec[k]
                        for k in (
                            "steps",
                            "optimizer",
                            "learning_rate",
                            "betas",
                            "eps",
                            "weight_decay",
                            "gradient_clip",
                            "steering_coeff",
                            "seed",
                            "sequence_tokens",
                            "alpha",
                            "gamma",
                            "hook_module",
                        )
                    },
                    "historical_training_settings": historical["baseline"],
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
            with active_phase("rmu_snapshot"):
                snapshot = prepare_snapshot(args.source, directory, NAMES)
                iteration.bind_snapshot(runtime, snapshot)
            reference = None
            try:
                with active_phase("rmu_reference"):
                    reference = tensors.capture_reference(runtime.model, inner)
                trained = train_candidate(runtime, reference, corpus, inner, spec, watch)
                atomic_json(directory / "training.json", trained)
                with active_phase("rmu_storage_roundtrip"):
                    measured = storage_measurement(
                        runtime, reference, outer, directory / "storage-delta.json"
                    )
                status["storage_delta"] = measured
                with active_phase("rmu_after_image_evidence"):
                    status["delta_manifest"] = save_after_image(
                        runtime, reference, outer, directory, root
                    )
                if args.phase == "screen":
                    if dataset is None or before is None:
                        raise ValueError("missing development inputs")
                    status.update(screen(runtime, dataset, before, directory, watch))
                else:
                    expected = iteration.persisted_tensor_hashes(runtime, NAMES)
                    replacements = {
                        n: dict(runtime.model.named_parameters())[n].detach() for n in NAMES
                    }
                    output = root / (args.name + "-checkpoint")
                    with active_phase("rmu_checkpoint_write"):
                        receipt = write_edited_checkpoint(
                            args.source, output, replacements, snapshot
                        )
                    del replacements
                    status.update(checkpoint=str(output), snapshot_id=receipt["snapshot_id"])
            finally:
                if reference is not None:
                    with active_phase("rmu_exact_memory_restore"):
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
