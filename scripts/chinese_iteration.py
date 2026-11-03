"""Local Chinese-language diagnosis and explicit candidate execution; no overall deadline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from dataclasses import asdict
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append("/usr/lib/python3/dist-packages")

from lacunai.core.model import HFRuntime, active_phase, checkpoint_inventory  # noqa: E402
from lacunai.core.spec import RuntimeConfig, bounded_file, digest  # noqa: E402
from lacunai.erase import language as operator  # noqa: E402
from lacunai.experimental import tensors  # noqa: E402
from lacunai.experimental.artifacts import atomic_json, secure_path  # noqa: E402
from lacunai.experimental.resources import Watchdog, host_sample  # noqa: E402
from lacunai.map import language_entropy  # noqa: E402
from lacunai.restore.checkpoints import (  # noqa: E402
    prepare_snapshot,
    restore_snapshot,
    write_edited_checkpoint,
)
from lacunai.verify import language as evaluation  # noqa: E402

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


def score(
    runtime: HFRuntime, dataset: dict[str, Any], path: Path, watch: Watchdog, development: bool
) -> dict[str, Any]:
    with active_phase("chinese_evaluation"):
        result = evaluation.evaluate(
            runtime.model,
            runtime.tokenizer,
            dataset,
            runtime.contract,
            path,
            watch,
            development=development,
        )
    # A fixed non-Chinese development corpus, never candidate-generated content.
    texts = [item["prompts"][0] + "\n" + item["answer"] for item in dataset["dev_retain"]]
    with active_phase("retain_perplexity"):
        result["perplexity"] = runtime.perplexity(texts)
    atomic_json(path / "result.json", result)
    return result


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result = evaluation.compare(before, after)
    if before["perplexity"]["tokens"] != after["perplexity"]["tokens"]:
        raise ValueError("perplexity tokenization changed")
    ratio = Decimal(str(after["perplexity"]["perplexity"])) / Decimal(
        str(before["perplexity"]["perplexity"])
    )
    result["perplexity_ratio"] = str(ratio)
    result["maximum_perplexity_ratio"] = "1.15"
    if ratio > Decimal("1.15"):
        result["verdict"] = "failed"
        result.setdefault("reasons", []).append("non-Chinese perplexity exceeds frozen threshold")
    return result


def collateral_preflight(
    runtime: HFRuntime, dataset: dict[str, Any], before: dict[str, Any], path: Path, watch: Any
) -> dict[str, Any]:
    """Reject a development candidate early; this path can never establish an efficacy pass."""
    items = dataset["dev_retain"]
    if not before.get("development") or before["retain"]["spec"] != items:
        raise ValueError("collateral preflight requires the matching development baseline")
    baseline = evaluation.summarize(items, before["retain"]["raw"], retain=True)
    path.mkdir(mode=0o700, exist_ok=False)
    total = sum(len(item["prompts"]) for item in items)
    count = 0
    result: dict[str, Any] = {"continue": True, "efficacy_pass": False, "observations": []}
    with (path / "raw.jsonl").open("x") as stream:
        for item in items:
            for number, prompt in enumerate(item["prompts"]):
                watch.check()
                watch.progress("collateral_preflight", completed=count, total=total, unit="prompts")
                row = {"item_id": item["id"], "variant": number, **runtime.generate(prompt)}
                scored = evaluation.score_response(
                    item, row["output"], row["truncated"], retain=True
                )
                row["score"] = scored
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                result["observations"].append(row)
                count += 1
                watch.progress("collateral_preflight", completed=count, total=total, unit="prompts")
                was_correct = baseline["items"][item["id"]]["variants"][number]["correct"]
                if was_correct and not scored["correct"]:
                    result.update(
                        **{"continue": False},
                        verdict="inconclusive" if scored["ambiguous"] else "failed",
                        reason="baseline-confirmed retain answer lost or unresolved",
                    )
                    atomic_json(path / "result.json", result)
                    return result
    texts = [item["prompts"][0] + "\n" + item["answer"] for item in items]
    result["perplexity"] = runtime.perplexity(texts)
    if result["perplexity"]["tokens"] != before["perplexity"]["tokens"]:
        raise ValueError("collateral perplexity tokenization differs")
    ratio = Decimal(str(result["perplexity"]["perplexity"])) / Decimal(
        str(before["perplexity"]["perplexity"])
    )
    result["perplexity_ratio"] = str(ratio)
    if ratio > Decimal("1.15"):
        result.update(
            **{"continue": False},
            verdict="failed",
            reason="non-Chinese perplexity threshold exceeded",
        )
    atomic_json(path / "result.json", result)
    return result


def verified_holdout(
    dataset_path: Path, manifest_path: Path | None, expected_sha256: str | None
) -> dict[str, Any] | None:
    """An independent caller pins the manifest; a self-declared boolean is insufficient."""
    if manifest_path is None and expected_sha256 is None:
        if (dataset_path.parent / "manifest.json").exists():
            manifest = json.loads(bounded_file(dataset_path.parent / "manifest.json", 32 * 1024**2))
            if manifest.get("independent_holdout_commitment"):
                raise ValueError("fresh holdout requires an independently pinned manifest")
        return None
    if manifest_path is None or expected_sha256 is None:
        raise ValueError("holdout manifest and SHA-256 must be supplied together")
    with active_phase("holdout_commitment_verification"):
        raw = bounded_file(manifest_path, 32 * 1024**2)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("holdout manifest hash differs from independent pin")
        manifest = json.loads(raw)
        dataset_sha = hashlib.sha256(bounded_file(dataset_path, 32 * 1024**2)).hexdigest()
        if (
            not manifest.get("independent_holdout_commitment")
            or manifest.get("dataset_bytes_sha256") != dataset_sha
        ):
            raise ValueError("holdout dataset hash or independent provenance is missing")
        return {
            "manifest_sha256": expected_sha256,
            "dataset_bytes_sha256": dataset_sha,
            "commitment": manifest["independent_holdout_commitment"],
            "limit": "integrity of independently supplied commitment; not proof of blindness",
        }


def no_op_identity(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    for key in ("dataset_sha256", "inference_sha256"):
        if not before.get(key) or before[key] != after.get(key):
            raise ValueError(f"no-op {key} differs")
    suites = {name for name in ("primary", "retain", "recovery") if name in before}
    if not {"primary", "retain"} <= suites or suites != {
        n for n in ("primary", "retain", "recovery") if n in after
    }:
        raise ValueError("no-op evaluation suites differ or are incomplete")
    count = 0
    for suite in suites:
        left, right = before[suite]["raw"], after[suite]["raw"]
        if not left or len(left) != len(right):
            raise ValueError("no-op raw observation count differs")
        for a, b in zip(left, right, strict=True):
            for key in ("item_id", "variant", "token_ids", "first_step_logits_sha256", "output"):
                if key not in a or key not in b or a[key] != b[key]:
                    raise ValueError(f"no-op deterministic observation differs: {key}")
            count += 1
    if before.get("receptive") != after.get("receptive") or before.get("perplexity") != after.get(
        "perplexity"
    ):
        raise ValueError("no-op receptive or perplexity results differ")
    return {
        "deterministic_equal": True,
        "observations": count,
        "claim": "deterministic no-op equality; not an unlearning outcome",
    }


def persisted_tensor_hashes(runtime: HFRuntime, names: list[str]) -> dict[str, str]:
    """Hash the full allowed tensor after conversion to actual checkpoint storage precision."""
    torch = import_module("torch")
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    parameters = dict(runtime.model.named_parameters())
    return {
        name: tensors.tensor_hash(
            parameters[name].detach().to(dtypes[runtime.storage_dtypes[name]]).to(torch.float32)
        )
        for name in names
    }


def bind_snapshot(runtime: HFRuntime, snapshot: dict[str, Any]) -> None:
    recorded = {name: row["sha256"] for name, row in snapshot["before_files"].items()}
    if recorded != runtime.inventory:
        raise ValueError("durable snapshot source differs from the loaded checkpoint")


def reload_checkpoint(
    output: Path,
    config: RuntimeConfig,
    expected_inventory: dict[str, str],
    expected_tensors: dict[str, str],
    commitment: dict[str, Any] | None,
) -> HFRuntime:
    with active_phase("persisted_checkpoint_reload"):
        loaded = HFRuntime(output, config)
        if loaded.inventory != expected_inventory:
            raise ValueError("persisted checkpoint inventory differs at reload")
        parameters = dict(loaded.model.named_parameters())
        if any(
            name not in parameters or tensors.tensor_hash(parameters[name].detach()) != value
            for name, value in expected_tensors.items()
        ):
            raise ValueError("reloaded complete stored-precision tensor differs")
        loaded.contract["evaluation"] = {"receptive_choice_count": 4}
        if commitment:
            loaded.contract["independent_holdout_commitment"] = commitment
        return loaded


def _verified_copy(source: Path, target: Path, expected: str, watch: Any) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("copy source must be an unredirected file")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    total = source.stat().st_size
    watch.progress("restore_proof_copy", completed=0, total=total, unit="bytes")
    hashed = hashlib.sha256()
    completed = 0
    with source.open("rb") as src, target.open("xb") as dst:
        for block in iter(lambda: src.read(1024**2), b""):
            watch.check()
            dst.write(block)
            hashed.update(block)
            completed += len(block)
            watch.progress("restore_proof_copy", completed=completed, total=total, unit="bytes")
        dst.flush()
        os.fsync(dst.fileno())
    if (
        completed != total
        or hashed.hexdigest() != expected
        or tensors.file_hash(target) != expected
    ):
        raise ValueError("copy readback hash differs")


def disk_restore_proof(
    snapshot_run: Path, directory: Path, output: Path, config: RuntimeConfig, watch: Any
) -> dict[str, Any]:
    """Fork a verified snapshot receipt and restore only a new disposable derivative copy."""
    with active_phase("disk_restore_proof"):
        raw = bounded_file(snapshot_run / "snapshot.json", 32 * 1024**2)
        original = json.loads(raw)
        if original.get("status") != "edited" or original.get("run_dir") != str(
            snapshot_run.absolute()
        ):
            raise ValueError("restore proof requires a complete edited snapshot")
        candidate = Path(original["output"])
        before = {name: row["sha256"] for name, row in original["before_files"].items()}
        after = {name: row["sha256"] for name, row in original["after_files"].items()}
        if output.exists() or output.is_symlink() or output.absolute() == candidate.absolute():
            raise ValueError("restore proof requires a new separate output")
        resolved_output = output.absolute()
        for protected in (candidate, Path(original["source"]), snapshot_run, directory):
            protected = protected.absolute()
            if resolved_output.is_relative_to(protected) or protected.is_relative_to(
                resolved_output
            ):
                raise ValueError("restore proof output overlaps protected paths")
        if any(p.is_symlink() for p in (resolved_output, *resolved_output.parents)):
            raise ValueError("restore proof output is redirected")
        if checkpoint_inventory(candidate, config.max_model_bytes) != after:
            raise ValueError("evaluated candidate differs from snapshot after-image")
        if checkpoint_inventory(Path(original["source"]), config.max_model_bytes) != before:
            raise ValueError("original checkpoint differs from snapshot before-image")
        output.mkdir(mode=0o700, parents=True)
        for name, sha in after.items():
            _verified_copy(secure_path(candidate, name), secure_path(output, name), sha, watch)
        for row in original["tensors"].values():
            _verified_copy(
                secure_path(snapshot_run, row["blob"]),
                secure_path(directory, row["blob"]),
                row["compressed_sha256"],
                watch,
            )
        fork = {
            **original,
            "run_dir": str(directory.absolute()),
            "output": str(output.absolute()),
            "staged_files": {},
            "restore_staged_files": {},
            "in_place": False,
            "restore_proof_parent": {
                "run_dir": str(snapshot_run.absolute()),
                "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
                "candidate": str(candidate),
            },
        }
        atomic_json(directory / "snapshot.json", fork)
        receipt = restore_snapshot(directory, apply=True)
        if checkpoint_inventory(output, config.max_model_bytes) != before:
            raise ValueError("restored copy does not match every original checkpoint file")
        if checkpoint_inventory(candidate, config.max_model_bytes) != after:
            raise ValueError("evaluated candidate changed during restore proof")
        if checkpoint_inventory(Path(original["source"]), config.max_model_bytes) != before:
            raise ValueError("original checkpoint changed during restore proof")
        if bounded_file(snapshot_run / "snapshot.json", 32 * 1024**2) != raw:
            raise ValueError("original snapshot receipt changed during restore proof")
        return {
            "restored_exactly": True,
            "candidate_preserved": True,
            "output": str(output),
            "source_fingerprint": digest(before),
            "candidate_fingerprint": digest(after),
            "receipt": receipt,
            "parent_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        }


def relearning_inputs(
    contract_path: Path, contract_sha: str, dataset_path: Path, dataset_sha: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from lacunai.experimental.data import _relearn

    contract_raw = bounded_file(contract_path, 32 * 1024**2)
    dataset_raw = bounded_file(dataset_path, 32 * 1024**2)
    if (
        hashlib.sha256(contract_raw).hexdigest() != contract_sha
        or hashlib.sha256(dataset_raw).hexdigest() != dataset_sha
    ):
        raise ValueError("relearning inputs differ from frozen hashes")
    contract, data = json.loads(contract_raw), json.loads(dataset_raw)
    settings = contract["relearning"]
    baseline = contract["baseline"]
    if settings.get("steps_per_arm") != 20 or Decimal(settings["learning_rate"]) != Decimal(
        "0.0001"
    ):
        raise ValueError("relearning must use historical 20-step SGD settings")
    expected = {
        "optimizer": "SGD",
        "sequence_tokens": 128,
        "alpha": "1.0",
        "gradient_clip": "1.0",
        "momentum": 0,
    }
    if any(str(baseline.get(k)) != str(v) for k, v in expected.items()):
        raise ValueError("relearning baseline settings differ from frozen historical settings")
    names = [f"model.layers.{i}.mlp.down_proj.weight" for i in range(16, 24)]
    edits = contract["edits"]
    if (
        edits.get("allowed_tensors") != names
        or edits.get("expected_shape") != [896, 4864]
        or Decimal(edits["max_tensor_relative_frobenius"]) != Decimal("0.02")
        or Decimal(edits["max_model_relative_frobenius"]) != Decimal("0.0025")
    ):
        raise ValueError("relearning tensor allowlist or bounds differ")
    if data.get("relearn") != _relearn():
        raise ValueError("relearning corpus must be the frozen benign English catalogue")
    return contract, data["relearn"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=[
            "diagnose",
            "lape-diagnose",
            "screen",
            "persist",
            "baseline",
            "dev-baseline",
            "verify",
            "no-op",
            "restore-proof",
            "relearn",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--fraction", type=float, choices=operator.FRACTIONS)
    parser.add_argument("--name", required=True)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--method", choices=["magnitude", "lape"], default="magnitude")
    parser.add_argument("--selection-fraction", type=float)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--corpus-sha256")
    parser.add_argument("--collateral-first", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--snapshot-run", type=Path)
    parser.add_argument("--holdout-manifest", type=Path)
    parser.add_argument("--holdout-manifest-sha256")
    parser.add_argument("--training-contract", type=Path)
    parser.add_argument("--training-contract-sha256")
    parser.add_argument("--relearn-dataset", type=Path)
    parser.add_argument("--relearn-dataset-sha256")
    args = parser.parse_args()
    if not args.apply:
        print(json.dumps({"state": "plan_only", "phase": args.phase}), flush=True)
        return 0
    if not args.name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid run name")
    root = args.root.absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("linked workspace refused")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = root / args.name
    directory.mkdir(mode=0o700, exist_ok=False)
    if not (root / "host-before.json").exists():
        atomic_json(root / "host-before.json", host_sample(root))
    status: dict[str, Any] = {"phase": args.phase, "state": "running", "verdict": "inconclusive"}
    atomic_json(directory / "status.json", status)
    try:
        config = RuntimeConfig(max_seconds=None, max_new_tokens=64, seed=42, torch_threads=2)
        commitment = verified_holdout(
            args.dataset, args.holdout_manifest, args.holdout_manifest_sha256
        )
        dataset = json.loads(bounded_file(args.dataset, 32 * 1024**2))
        if "dev_retain" not in dataset:
            development = json.loads(bounded_file(root / "development-data.json", 32 * 1024**2))
            dataset["dev_retain"] = development["dev_retain"]
        with Watchdog(root, {"resources": LIMITS}, None, args.name) as watch:
            if args.phase == "restore-proof":
                if args.snapshot_run is None:
                    raise ValueError("restore-proof needs --snapshot-run")
                status["restoration"] = disk_restore_proof(
                    args.snapshot_run, directory, root / (args.name + "-checkpoint"), config, watch
                )
                status["state"] = "complete"
                atomic_json(directory / "status.json", status)
                print(
                    json.dumps(
                        {"phase": args.phase, "state": "complete", "verdict": "inconclusive"}
                    ),
                    flush=True,
                )
                return 0
            with active_phase("load_original"):
                runtime = HFRuntime(args.source, config)
            runtime.contract["evaluation"] = {"receptive_choice_count": 4}
            if commitment:
                runtime.contract["independent_holdout_commitment"] = commitment
            code_root = Path(__file__).resolve().parents[1]
            atomic_json(
                directory / "manifest.json",
                {
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                    "checkpoint_fingerprint": runtime.fingerprint,
                    "source_inventory": runtime.inventory,
                    "dataset_sha256": digest(dataset),
                    "runtime": asdict(config),
                    "resources": LIMITS,
                    "evaluation_contract": evaluation.CONTRACT,
                    "editor": "current Codex conversation",
                    "fraction": args.fraction,
                    "method": args.method,
                    "selection_fraction": args.selection_fraction,
                    "corpus_sha256": args.corpus_sha256,
                    "collateral_first": args.collateral_first,
                    "overall_deadline": None,
                    "independent_holdout_commitment": commitment,
                    "code_hashes": {
                        str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__), *sorted((code_root / "src").rglob("*.py"))]
                    },
                    "claim_limit": "finite-test recoverability; no proof of information absence",
                },
            )
            if args.phase == "lape-diagnose":
                if args.corpus is None or args.corpus_sha256 is None:
                    raise ValueError("LAPE diagnosis requires a pinned development corpus")
                raw = bounded_file(args.corpus, 32 * 1024**2)
                if hashlib.sha256(raw).hexdigest() != args.corpus_sha256:
                    raise ValueError("LAPE development corpus differs from explicit pin")
                with active_phase("four_language_gate_diagnosis"):
                    diagnosis = language_entropy.diagnose(runtime, json.loads(raw)["corpora"])
                atomic_json(directory / "diagnosis.json", diagnosis)
            elif args.phase == "diagnose":
                pairs = [
                    {key: item[key] for key in ("diagnostic_zh", "diagnostic_en")}
                    for item in dataset["dev_primary"]
                ]
                background = [
                    item["prompts"][0]
                    for item in dataset["dev_retain"]
                    if item["family"] in {"spanish", "japanese_han"}
                ]
                with active_phase("all_layer_language_diagnosis"):
                    diagnosis = operator.diagnose(runtime, pairs, background)
                atomic_json(directory / "diagnosis.json", diagnosis)
                if not args.diagnose_only:
                    result = score(runtime, dataset, directory / "baseline", watch, True)
                    status["result"] = result
            elif args.phase in {"baseline", "dev-baseline", "verify"}:
                result = score(
                    runtime, dataset, directory / "evaluation", watch, args.phase == "dev-baseline"
                )
                status["result"] = result
                if args.before:
                    before = json.loads(bounded_file(args.before, 32 * 1024**2))
                    status["comparison"] = compare(before, result)
                    status["verdict"] = status["comparison"]["verdict"]
            elif args.phase in {"no-op", "relearn"}:
                training_contract = None
                if args.phase == "no-op":
                    if args.before is None:
                        raise ValueError("no-op needs --before from the untouched checkpoint")
                    names = ["model.layers.23.mlp.down_proj.weight"]
                else:
                    if not all(
                        (
                            args.training_contract,
                            args.training_contract_sha256,
                            args.relearn_dataset,
                            args.relearn_dataset_sha256,
                        )
                    ):
                        raise ValueError("relearn requires pinned historical contract and corpus")
                    training_contract, corpus = relearning_inputs(
                        args.training_contract,
                        args.training_contract_sha256,
                        args.relearn_dataset,
                        args.relearn_dataset_sha256,
                    )
                    names = training_contract["edits"]["allowed_tensors"]
                source_inventory = dict(runtime.inventory)
                if checkpoint_inventory(args.source, config.max_model_bytes) != source_inventory:
                    raise ValueError("original checkpoint changed before control snapshot")
                with active_phase("durable_snapshot"):
                    snapshot = prepare_snapshot(args.source, directory, names)
                bind_snapshot(runtime, snapshot)
                reference = None
                try:
                    if training_contract:
                        with active_phase("benign_relearning"):
                            reference = tensors.capture_reference(runtime.model, training_contract)
                            trained = tensors.train(
                                runtime.model,
                                runtime.tokenizer,
                                [],
                                corpus,
                                training_contract,
                                "relearn",
                                lambda event: watch.progress(
                                    event["phase"],
                                    **{k: v for k, v in event.items() if k != "phase"},
                                ),
                                watch.check,
                                reference,
                            )
                        atomic_json(
                            directory / "training.json",
                            {
                                **trained,
                                "bound_reference": (
                                    "input candidate checkpoint; "
                                    "not historical combined original delta"
                                ),
                                "contract_sha256": args.training_contract_sha256,
                                "corpus_sha256": args.relearn_dataset_sha256,
                            },
                        )
                    expected = persisted_tensor_hashes(runtime, names)
                    replacements = {
                        n: dict(runtime.model.named_parameters())[n].detach() for n in names
                    }
                    output = root / (args.name + "-checkpoint")
                    with active_phase("checkpoint_write"):
                        receipt = write_edited_checkpoint(
                            args.source, output, replacements, snapshot
                        )
                    del replacements
                finally:
                    if reference is not None:
                        if training_contract is None:
                            raise ValueError("missing training contract for restoration")
                        with active_phase("exact_in_memory_restore"):
                            tensors.restore_reference(runtime.model, reference, training_contract)
                    del reference
                del runtime
                gc.collect()
                inventory = checkpoint_inventory(output, config.max_model_bytes)
                if inventory != {n: r["sha256"] for n, r in receipt["after_files"].items()}:
                    raise ValueError("persisted after-image differs from write receipt")
                if args.phase == "no-op" and inventory != source_inventory:
                    raise ValueError("no-op persisted checkpoint differs from source bytes")
                runtime = reload_checkpoint(output, config, inventory, expected, commitment)
                result = score(runtime, dataset, directory / "evaluation", watch, False)
                status.update(
                    result=result,
                    checkpoint=str(output),
                    snapshot_id=receipt["snapshot_id"],
                    output_fingerprint=runtime.fingerprint,
                    reloaded_tensor_hashes=expected,
                )
                if args.before:
                    before = json.loads(bounded_file(args.before, 32 * 1024**2))
                    if args.phase == "no-op":
                        status["control"] = no_op_identity(before, result)
                    else:
                        status["comparison"] = compare(before, result)
                        status["verdict"] = status["comparison"]["verdict"]
            else:
                if args.diagnosis is None or args.fraction is None:
                    raise ValueError("candidate needs a frozen diagnosis and allowed fraction")
                diagnosis = json.loads(bounded_file(args.diagnosis, 128 * 1024**2))
                if args.method == "lape":
                    if args.selection_fraction is None:
                        raise ValueError(
                            "LAPE candidate needs an explicit global selection fraction"
                        )
                    selection = language_entropy.select(diagnosis, args.selection_fraction)
                    proposal = language_entropy.to_proposal(
                        runtime, diagnosis, selection, execution_fraction=args.fraction
                    )
                else:
                    proposal = operator.propose(runtime, diagnosis, args.fraction)
                atomic_json(directory / "edit-spec.json", proposal)
                if checkpoint_inventory(args.source, config.max_model_bytes) != runtime.inventory:
                    raise ValueError("original checkpoint changed before snapshot")
                names = operator.touched_tensors(runtime, proposal)
                with active_phase("durable_snapshot"):
                    snapshot = prepare_snapshot(args.source, directory, names)
                bind_snapshot(runtime, snapshot)
                state = operator.apply(runtime, proposal, snapshot)
                try:
                    atomic_json(directory / "edit-diagnostics.json", state["diagnostics"])
                    if args.phase == "screen":
                        continue_screen = True
                        if args.collateral_first:
                            if args.before is None:
                                raise ValueError("collateral-first screening requires a baseline")
                            before = json.loads(bounded_file(args.before, 32 * 1024**2))
                            preflight = collateral_preflight(
                                runtime, dataset, before, directory / "collateral-preflight", watch
                            )
                            status["collateral_preflight"] = preflight
                            continue_screen = preflight["continue"]
                            if not continue_screen:
                                status["verdict"] = preflight["verdict"]
                                status["screening"] = "rejected; full primary suite not executed"
                        if continue_screen:
                            result = score(runtime, dataset, directory / "evaluation", watch, True)
                            status["result"] = result
                            if args.before:
                                before = json.loads(bounded_file(args.before, 32 * 1024**2))
                                status["comparison"] = compare(before, result)
                                status["verdict"] = status["comparison"]["verdict"]
                    else:
                        expected = persisted_tensor_hashes(runtime, names)
                        replacements = {
                            name: dict(runtime.model.named_parameters())[name].detach()
                            for name in names
                        }
                        output = root / (args.name + "-checkpoint")
                        with active_phase("checkpoint_write"):
                            receipt = write_edited_checkpoint(
                                args.source, output, replacements, snapshot
                            )
                        status["checkpoint"] = str(output)
                        status["snapshot_id"] = receipt["snapshot_id"]
                        del replacements
                finally:
                    with active_phase("exact_in_memory_restore"):
                        restored = operator.restore(runtime, state)
                    atomic_json(directory / "restore.json", restored)
                    del state
                    gc.collect()
                if args.phase == "persist":
                    del runtime
                    gc.collect()
                    inventory = checkpoint_inventory(output, config.max_model_bytes)
                    if inventory != {n: r["sha256"] for n, r in receipt["after_files"].items()}:
                        raise ValueError("persisted after-image differs from write receipt")
                    runtime = reload_checkpoint(output, config, inventory, expected, commitment)
                    status["output_fingerprint"] = runtime.fingerprint
                    status["reloaded_tensor_hashes"] = expected
                    atomic_json(
                        directory / "reload.json",
                        {"inventory": inventory, "tensor_hashes": expected, "loaded": True},
                    )
            status["state"] = "complete"
            atomic_json(directory / "status.json", status)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(directory / "status.json", status)
        raise
    print(json.dumps({key: status[key] for key in ("phase", "state", "verdict")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
