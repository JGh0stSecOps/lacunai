"""Matched original-source GradDiff/retain controls; no overall wall-time cutoff."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import asdict
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chinese_iteration as iteration  # noqa: E402
from lacunai.core.model import HFRuntime, active_phase, checkpoint_inventory  # noqa: E402
from lacunai.core.spec import RuntimeConfig, bounded_file, digest  # noqa: E402
from lacunai.experimental import tensors  # noqa: E402
from lacunai.experimental.artifacts import atomic_json  # noqa: E402
from lacunai.experimental.data import build_dataset  # noqa: E402
from lacunai.experimental.resources import Watchdog, host_sample  # noqa: E402
from lacunai.restore.checkpoints import prepare_snapshot, write_edited_checkpoint  # noqa: E402
from lacunai.verify import language as evaluation  # noqa: E402

MODES = ("graddiff", "retain_finetune")
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
NAMES = [f"model.layers.{i}.mlp.down_proj.weight" for i in range(16, 24)]
SETTINGS = {
    "method": "GradDiff_NLL_parameter_restricted",
    "source_revision": "2d98027677f3468d52b6baa3fc4f2b9f2dc08820",
    "steps": 60,
    "optimizer": "SGD",
    "learning_rate": "0.001",
    "momentum": 0,
    "gradient_clip": "1.0",
    "sequence_tokens": 128,
    "microbatch_forget": 1,
    "microbatch_retain": 1,
    "alpha": "1.0",
    "gamma": "1.0",
}


def pinned_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = bounded_file(path, 32 * 1024**2)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"frozen input hash differs: {path.name}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("frozen input must be an object")
    return value


def validate_training(contract: dict[str, Any], dataset: dict[str, Any]) -> None:
    if contract.get("baseline") != SETTINGS:
        raise ValueError("baseline must match exact historical 60-step SGD settings")
    edits = contract["edits"]
    if (
        edits.get("allowed_tensors") != NAMES
        or edits.get("expected_shape") != [896, 4864]
        or Decimal(edits["max_tensor_relative_frobenius"]) != Decimal(".02")
        or Decimal(edits["max_model_relative_frobenius"]) != Decimal(".0025")
        or any(
            edits.get(k) is not False
            for k in ("tokenizer_edits", "embedding_edits", "lm_head_edits", "generated_code")
        )
    ):
        raise ValueError("baseline tensor allowlist or change bounds differ")
    if contract.get("runtime", {}).get("seed") != 20261102:
        raise ValueError("historical dataset generation seed differs")
    expected = build_dataset(contract)
    for split in ("forget_train", "retain_train", "dev_retain"):
        if not dataset.get(split) or dataset[split] != expected[split]:
            raise ValueError(f"baseline corpus differs from historical synthetic data: {split}")


def train_control(
    runtime: Any,
    contract: dict[str, Any],
    dataset: dict[str, Any],
    mode: str,
    watch: Any,
    reference: dict[str, Any],
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("unknown baseline mode")
    with active_phase("baseline_training"):
        result = tensors.train(
            runtime.model,
            runtime.tokenizer,
            dataset["forget_train"] if mode == "graddiff" else [],
            dataset["retain_train"],
            contract,
            mode,
            lambda event: watch.progress(
                event["phase"], **{k: v for k, v in event.items() if k != "phase"}
            ),
            watch.check,
            reference,
        )
        trace = result.get("trace", [])
        if len(trace) != SETTINGS["steps"] or any(
            row.get("completed") != i or row.get("total") != SETTINGS["steps"]
            for i, row in enumerate(trace, 1)
        ):
            raise ValueError("baseline actual step trace differs from matched contract")
        return result


def round_and_validate_storage(
    runtime: Any, contract: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Validate the actual stored-precision after-image, not only its FP32 precursor."""
    torch = import_module("torch")
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    parameters = dict(runtime.model.named_parameters())
    with torch.no_grad():
        for name in contract["edits"]["allowed_tensors"]:
            value = parameters[name]
            value.copy_(value.to(dtypes[runtime.storage_dtypes[name]]).to(value.dtype))
    return tensors.validate_state(runtime.model, reference, contract)


def verify_before_source(before_path: Path, source_fingerprint: str) -> dict[str, Any]:
    """Bind the pinned result to its adjacent run manifest before loading any model."""
    candidates = [before_path.parent / "manifest.json", before_path.parent.parent / "manifest.json"]
    found = [path for path in candidates if path.exists() or path.is_symlink()]
    if len(found) != 1:
        raise ValueError("before result requires one unambiguous adjacent run manifest")
    raw = bounded_file(found[0], 32 * 1024**2)
    manifest = json.loads(raw)
    if manifest.get("checkpoint_fingerprint") != source_fingerprint:
        raise ValueError("before result checkpoint fingerprint differs from original source")
    return {
        "path": str(found[0].absolute()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_fingerprint": manifest["checkpoint_fingerprint"],
    }


def common_manifest_fields(
    source_fingerprint: str, code_hashes: dict[str, str], commitment: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "checkpoint_fingerprint": source_fingerprint,
        "code_hashes": code_hashes,
        "evaluation_contract": evaluation.CONTRACT,
        "editor": "current Codex conversation",
        "independent_holdout_commitment": commitment,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    for flag in (
        "root",
        "source",
        "training-contract",
        "training-dataset",
        "dataset",
        "holdout-manifest",
        "before",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in (
        "source-fingerprint",
        "training-contract-sha256",
        "training-dataset-sha256",
        "holdout-manifest-sha256",
        "before-sha256",
        "name",
    ):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print(
            json.dumps(
                {"state": "plan_only", "mode": args.mode, "steps": 60, "overall_deadline": None}
            ),
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
        "phase": args.mode,
        "mode": args.mode,
        "state": "running",
        "verdict": "inconclusive",
    }
    atomic_json(directory / "status.json", status)
    try:
        with active_phase("baseline_preparation"):
            training = pinned_json(args.training_contract, args.training_contract_sha256)
            corpus = pinned_json(args.training_dataset, args.training_dataset_sha256)
            validate_training(training, corpus)
            commitment = iteration.verified_holdout(
                args.dataset, args.holdout_manifest, args.holdout_manifest_sha256
            )
            if commitment is None:
                raise ValueError("baseline requires independently committed holdout")
            before = pinned_json(args.before, args.before_sha256)
            before_manifest = verify_before_source(args.before, args.source_fingerprint)
            dataset = json.loads(bounded_file(args.dataset, 32 * 1024**2))
            # Exact same historical non-Chinese PPL corpus as iteration's development-data.
            if "dev_retain" in dataset and dataset["dev_retain"] != corpus["dev_retain"]:
                raise ValueError("evaluation perplexity corpus differs from frozen development")
            dataset["dev_retain"] = corpus["dev_retain"]
            config = RuntimeConfig(max_seconds=None, max_new_tokens=64, seed=42, torch_threads=2)
            if not (root / "host-before.json").exists():
                atomic_json(root / "host-before.json", host_sample(root))
        with Watchdog(root, {"resources": iteration.LIMITS}, None, args.name) as watch:
            with active_phase("original_source_verification"):
                source_inventory = checkpoint_inventory(args.source, config.max_model_bytes)
                if digest(source_inventory) != args.source_fingerprint:
                    raise ValueError(
                        "baseline must start from independently pinned original source"
                    )
                runtime = HFRuntime(args.source, config)
                if runtime.inventory != source_inventory:
                    raise ValueError("original source changed during model load")
            runtime.contract["evaluation"] = {"receptive_choice_count": 4}
            runtime.contract["independent_holdout_commitment"] = commitment
            code_root = Path(__file__).resolve().parents[1]
            code_paths = [
                Path(__file__),
                Path(iteration.__file__),
                *sorted((code_root / "src").rglob("*.py")),
            ]
            atomic_json(
                directory / "manifest.json",
                {
                    **common_manifest_fields(
                        args.source_fingerprint,
                        {
                            str(p.relative_to(code_root)): hashlib.sha256(
                                p.read_bytes()
                            ).hexdigest()
                            for p in code_paths
                        },
                        commitment,
                    ),
                    "entrypoint": "scripts/chinese_baselines.py",
                    "phase": args.mode,
                    "before_manifest": before_manifest,
                    "mode": args.mode,
                    "source_fingerprint": args.source_fingerprint,
                    "source_inventory": source_inventory,
                    "training_contract_sha256": args.training_contract_sha256,
                    "training_dataset_sha256": args.training_dataset_sha256,
                    "holdout_commitment": commitment,
                    "before_sha256": args.before_sha256,
                    "dataset_sha256": digest(dataset),
                    "training_settings": training["baseline"],
                    "edits": training["edits"],
                    "runtime": asdict(config),
                    "resources": iteration.LIMITS,
                    "overall_deadline": None,
                    "training_objective": "retain_NLL - forget_NLL"
                    if args.mode == "graddiff"
                    else "retain_NLL",
                    "seed_note": (
                        "Current evaluation seed42; historical corpus generation "
                        "seed20261102. SGD has no dropout or data shuffle."
                    ),
                    "claim_limit": (
                        "finite recoverability control; not proof of information absence"
                    ),
                },
            )
            with active_phase("baseline_snapshot"):
                snapshot = prepare_snapshot(args.source, directory, NAMES)
                iteration.bind_snapshot(runtime, snapshot)
            reference = None
            try:
                with active_phase("baseline_reference"):
                    reference = tensors.capture_reference(runtime.model, training)
                trained = train_control(runtime, training, corpus, args.mode, watch, reference)
                with active_phase("baseline_storage_precision_validation"):
                    trained["stored_precision_diff"] = round_and_validate_storage(
                        runtime, training, reference
                    )
                atomic_json(directory / "training.json", trained)
                expected = iteration.persisted_tensor_hashes(runtime, NAMES)
                replacements = {
                    n: dict(runtime.model.named_parameters())[n].detach() for n in NAMES
                }
                output = root / (args.name + "-checkpoint")
                with active_phase("baseline_checkpoint_write"):
                    receipt = write_edited_checkpoint(args.source, output, replacements, snapshot)
                del replacements
            finally:
                if reference is not None:
                    with active_phase("baseline_exact_memory_restore"):
                        tensors.restore_reference(runtime.model, reference, training)
                    atomic_json(
                        directory / "memory-restore.json", {"exact_original_tensors_restored": True}
                    )
                del reference
            del runtime
            gc.collect()
            with active_phase("baseline_checkpoint_verification"):
                inventory = checkpoint_inventory(output, config.max_model_bytes)
                if inventory != {n: r["sha256"] for n, r in receipt["after_files"].items()}:
                    raise ValueError("baseline persisted inventory differs from receipt")
                if checkpoint_inventory(args.source, config.max_model_bytes) != source_inventory:
                    raise ValueError("original source changed during baseline training")
            runtime = iteration.reload_checkpoint(output, config, inventory, expected, commitment)
            atomic_json(
                directory / "reload.json",
                {
                    "inventory": inventory,
                    "tensor_hashes": expected,
                    "loaded": True,
                    "checkpoint_fingerprint": runtime.fingerprint,
                },
            )
            result = iteration.score(runtime, dataset, directory / "evaluation", watch, False)
            comparison = iteration.compare(before, result)
            status.update(
                state="complete",
                verdict=comparison["verdict"],
                result=result,
                comparison=comparison,
                checkpoint=str(output),
                snapshot_id=receipt["snapshot_id"],
                output_fingerprint=runtime.fingerprint,
                reloaded_tensor_hashes=expected,
            )
            atomic_json(directory / "status.json", status)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(directory / "status.json", status)
        print(
            json.dumps({"phase": "baseline_control", "state": "failed", "error": status["error"]}),
            file=sys.stderr,
            flush=True,
        )
        raise
    print(
        json.dumps({"mode": args.mode, "state": status["state"], "verdict": status["verdict"]}),
        flush=True,
    )
    # A control can demonstrate retained recovery; operational success is separate from efficacy.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
