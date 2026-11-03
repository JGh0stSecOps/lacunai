# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""One bounded research run, with independent raw-output recomputation."""

from __future__ import annotations

import gc
import json
import math
import platform
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import evaluation
from .artifacts import atomic_json, sha256_file
from .resources import Watchdog
from .spec import ExperimentRefused, output_path


def freeze_dataset(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """This local preparation command runs before the source bundle is sealed."""
    from .data import build_dataset, dataset_sha256, validate_dataset

    dataset = build_dataset(contract)
    validate_dataset(dataset, contract)
    target = root / "datasets"
    target.mkdir(mode=0o700, parents=True, exist_ok=False)
    atomic_json(target / "dataset.json", dataset)
    manifest = {
        "dataset_sha256": dataset_sha256(dataset),
        "file_sha256": sha256_file(target / "dataset.json"),
        "counts": {key: len(value) for key, value in dataset.items()},
        "provenance": "local deterministic controlled-language generator; synthetic",
    }
    atomic_json(target / "manifest.json", manifest)
    return manifest


def read_dataset(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    from .data import dataset_sha256, validate_dataset

    directory = root / "datasets"
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256_file(directory / "dataset.json") != manifest["file_sha256"]:
        raise ExperimentRefused("frozen dataset file changed")
    dataset: dict[str, Any] = json.loads((directory / "dataset.json").read_text())
    validate_dataset(dataset, contract)
    if dataset_sha256(dataset) != manifest["dataset_sha256"]:
        raise ExperimentRefused("frozen dataset semantic hash changed")
    return dataset


def original_hashes(root: Path, contract: dict[str, Any]) -> dict[str, str]:
    hashes = {}
    for role in ("target", "editor"):
        for entry in contract[role]["files"]:
            relative = f"models/{role}/{entry['path']}"
            path = root / relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size"]:
                raise ExperimentRefused(
                    f"original source missing, linked or wrong size: {relative}"
                )
            actual = sha256_file(path)
            if entry["sha256"] is not None and actual != entry["sha256"]:
                raise ExperimentRefused(f"published original hash differs: {relative}")
            hashes[relative] = actual
    return hashes


def _jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def evaluate_items(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    contract: dict[str, Any],
    directory: Path,
    label: str,
    watch: Watchdog,
) -> dict[str, Any]:
    from .tensors import generate

    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    records = []
    total = sum(len(item["prompts"]) for item in items)
    path = directory / f"{label}.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for item in items:
            for variant, prompt in enumerate(item["prompts"]):
                watch.check()
                response = generate(model, tokenizer, prompt, contract)
                record = {
                    "item_id": item["id"],
                    "family": item["family"],
                    "variant": variant,
                    "prompt": prompt,
                    **response,
                }
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                watch.progress(
                    "evaluating",
                    arm=directory.name,
                    suite=label,
                    completed=len(records),
                    total=total,
                    unit="prompts",
                )
    evaluation.validate_observations(items, records)
    summary = evaluation.summarize(items, records)
    atomic_json(directory / f"{label}-summary.json", summary)
    return summary


def receptive(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    contract: dict[str, Any],
    directory: Path,
    watch: Watchdog,
) -> dict[str, Any]:
    from .tensors import choice_likelihoods

    records = []
    chosen = [item for item in items if "mcq" in item]
    for item in chosen:
        watch.check()
        mcq = item["mcq"]
        scores = choice_likelihoods(model, tokenizer, mcq["prompt"], mcq["choices"], contract)
        if not all(math.isfinite(row["mean_log_probability"]) for row in scores):
            raise ExperimentRefused("non-finite receptive score")
        values = [row["mean_log_probability"] for row in scores]
        tied = values.count(max(values)) != 1
        predicted = values.index(max(values))
        records.append(
            {
                "item_id": item["id"],
                "prompt": mcq["prompt"],
                "scores": scores,
                "correct_index": mcq["correct_index"],
                "predicted": predicted,
                "correct": predicted == mcq["correct_index"] and not tied,
                "tied": tied,
            }
        )
        watch.progress(
            "receptive_likelihood",
            arm=directory.name,
            completed=len(records),
            total=len(chosen),
            unit="items",
        )
    _jsonl(directory / "receptive.jsonl", records)
    evaluation.validate_receptive(items, records)
    summary = receptive_summary(records)
    atomic_json(directory / "receptive-summary.json", summary)
    return summary


def receptive_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ExperimentRefused("receptive evaluation has no observations")
    correct = 0
    for record in records:
        scores = [row["mean_log_probability"] for row in record["scores"]]
        if len(scores) != 4 or not all(math.isfinite(value) for value in scores):
            raise ExperimentRefused("invalid saved receptive scores")
        if scores.count(max(scores)) == 1 and scores.index(max(scores)) == record["correct_index"]:
            correct += 1
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": str(Decimal(correct) / Decimal(len(records))),
    }


def evaluate_arm(
    model: Any,
    tokenizer: Any,
    dataset: dict[str, Any],
    contract: dict[str, Any],
    root: Path,
    arm: str,
    watch: Watchdog,
    development: bool = False,
) -> dict[str, Any]:
    directory = output_path(root, f"results/{arm}")
    prefix = "dev" if development else "heldout"
    result = {
        "primary": evaluate_items(
            model, tokenizer, dataset[f"{prefix}_primary"], contract, directory, "primary", watch
        ),
        "retain": evaluate_items(
            model, tokenizer, dataset[f"{prefix}_retain"], contract, directory, "retain", watch
        ),
    }
    if not development:
        result["receptive"] = receptive(
            model, tokenizer, dataset["heldout_primary"], contract, directory, watch
        )
    return result


def receptive_gate(before: dict[str, Any], after: dict[str, Any], contract: dict[str, Any]) -> bool:
    if before["total"] != after["total"]:
        return False
    loss = (Decimal(before["accuracy"]) - Decimal(after["accuracy"])) * 100
    return Decimal(after["accuracy"]) <= Decimal(
        contract["evaluation"]["max_receptive_accuracy"]
    ) and loss >= Decimal(contract["evaluation"]["min_receptive_accuracy_drop_percentage_points"])


def _public_reference(reference: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in reference.items() if key != "original_tensors"}


def _emit_adapter(watch: Watchdog) -> Any:
    def emit(event: dict[str, Any]) -> None:
        watch.progress("tensor_work", detail=event)

    return emit


def _same_generated_outputs(root: Path, a: str, b: str) -> bool:
    for label in ("primary", "retain"):
        left = [
            json.loads(line)
            for line in (root / f"results/{a}/{label}.jsonl").read_text().splitlines()
        ]
        right = [
            json.loads(line)
            for line in (root / f"results/{b}/{label}.jsonl").read_text().splitlines()
        ]
        keys = (
            "item_id",
            "variant",
            "output",
            "token_ids",
            "input_tokens",
            "output_tokens",
            "truncated",
            "first_step_logits_sha256",
        )
        if len(left) != len(right) or any(
            any(x[key] != y[key] for key in keys) for x, y in zip(left, right, strict=True)
        ):
            return False
    return True


def _exercise_nonzero_candidate(diff: dict[str, Any]) -> bool:
    """Require a measured update in the already validated numerical diff."""
    count = diff.get("changed_tensor_count")
    norm = Decimal(str(diff.get("model_relative_frobenius", "NaN")))
    changes = [Decimal(str(row["delta_norm_squared"])) for row in diff.get("tensors", {}).values()]
    return (
        type(count) is int
        and count > 0
        and norm.is_finite()
        and norm > 0
        and all(value.is_finite() and value >= 0 for value in changes)
        and any(value > 0 for value in changes)
    )


def _freeze_exercise_verdict(report: dict[str, Any]) -> None:
    """An engineering exercise is ineligible for an efficacy verdict."""
    if report.get("purpose") == "engineering_exercise":
        report.update(verdict="inconclusive", efficacy_eligible=False, promotion="disabled")


def run(
    root: Path,
    contract: dict[str, Any],
    deadline: int,
    *,
    exercise: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run v1, or a separately admitted exercise that cannot establish efficacy."""
    from . import tensors
    from .editor import propose

    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "provisional",
        "verdict": "inconclusive",
        "phase": "admission",
        "arms": {},
        "attempts": [],
        "errors": [],
        "promotion": "disabled",
    }
    if exercise is not None:
        report.update(
            purpose="engineering_exercise",
            efficacy_eligible=False,
            exercise=exercise,
            selection_criterion="first numerically valid nonzero candidate by attempt index",
        )
    status_path = root / "result.json"
    if status_path.exists():
        raise ExperimentRefused("a run already exists; automatic repeat is forbidden")
    atomic_json(status_path, report)
    try:
        with Watchdog(root, contract, deadline - 600, "experiment") as watch:
            dataset = read_dataset(root, contract)
            watch.progress("hashing_originals")
            originals = original_hashes(root, contract)
            atomic_json(root / "original-hashes-before.json", originals)
            report["numeric_smoke"] = tensors.numerical_smoke(contract)
            report["numeric_guards"] = tensors.numeric_selftest(contract)
            if exercise is not None:
                from .editor import constrained_numeric_selftest

                report["constrained_editor_numeric_checks"] = constrained_numeric_selftest()
                atomic_json(
                    root / "constrained-editor-numeric-checks.json",
                    report["constrained_editor_numeric_checks"],
                )
            atomic_json(
                root / "numeric-checks.json",
                {"smoke": report["numeric_smoke"], "guards": report["numeric_guards"]},
            )
            report["phase"] = "qualification"
            atomic_json(status_path, report)
            watch.progress("loading_target")
            model, tokenizer = tensors.load_target(root / "models/target", contract)
            reference = tensors.capture_reference(model, contract)
            atomic_json(root / "source-tensors.json", _public_reference(reference))
            if exercise is None:
                report["arms"]["untouched"] = evaluate_arm(
                    model, tokenizer, dataset, contract, root, "untouched", watch
                )
            else:
                from .exercise import load_baseline

                watch.progress("verifying_imported_baseline")
                report["arms"]["untouched"] = load_baseline(root, contract, reference, watch)
            base = report["arms"]["untouched"]
            report["qualification"] = evaluation.qualify(base["primary"], base["retain"], contract)
            atomic_json(status_path, report)
            if not report["qualification"]["qualified"]:
                report["errors"] = report["qualification"]["reasons"]
                if exercise is None:
                    report["phase"] = "qualification_failed"
                    return report
                watch.progress("qualification_failed_exercise_only", efficacy_eligible=False)

            report["phase"] = "controls"
            checkpoints = output_path(root, "checkpoints")
            checkpoints.mkdir(mode=0o700)
            tensors.save_delta(model, reference, checkpoints / "no_op", contract, budget_root=root)
            tensors.load_delta(model, checkpoints / "no_op", reference, contract)
            report["arms"]["no_op"] = evaluate_arm(
                model, tokenizer, dataset, contract, root, "no_op", watch
            )
            if not _same_generated_outputs(root, "untouched", "no_op"):
                raise ExperimentRefused("no-op deterministic output identity failed")
            report["no_op_equal"] = True
            for mode in ("graddiff", "retain_finetune"):
                tensors.restore_reference(model, reference, contract)
                trained = tensors.train(
                    model,
                    tokenizer,
                    dataset["forget_train"],
                    dataset["retain_train"],
                    contract,
                    mode,
                    _emit_adapter(watch),
                    watch.check,
                    reference,
                )
                atomic_json(root / f"{mode}-training.json", trained)
                tensors.save_delta(model, reference, checkpoints / mode, contract, budget_root=root)
                tensors.load_delta(model, checkpoints / mode, reference, contract)
                report["arms"][mode] = evaluate_arm(
                    model, tokenizer, dataset, contract, root, mode, watch
                )
                atomic_json(status_path, report)

            tensors.restore_reference(model, reference, contract)
            diagnostics = tensors.diagnose(
                model,
                tokenizer,
                dataset["dev_primary"],
                contract,
                _emit_adapter(watch),
                watch.check,
                reference,
            )
            atomic_json(root / "diagnostics.json", diagnostics)
            dev_base = evaluate_arm(
                model,
                tokenizer,
                dataset,
                contract,
                root,
                "development_base",
                watch,
                development=True,
            )
            dev_evidence = [
                {
                    "id": item["id"],
                    "prompts": item["prompts"],
                    "answer": item["answer"],
                    "family": item["family"],
                }
                for item in dataset["dev_primary"][:4]
            ]
            # An editor never co-resides with the target or receives withheld evidence.
            del model, tokenizer, reference
            gc.collect()
            report["phase"] = "editor"
            valid_candidates = []
            for attempt in range(1, contract["resources"]["candidate_attempts"] + 1):
                watch.check()
                watch.progress(
                    "editor_call",
                    completed=attempt - 1,
                    total=contract["resources"]["editor_calls"],
                    unit="calls",
                )
                record = {"attempt": attempt, "status": "started"}
                report["attempts"].append(record)
                atomic_json(status_path, report)  # consume the attempt before generation
                if exercise is None:
                    generated = propose(
                        root / "models/editor", diagnostics, dev_evidence, contract, attempt
                    )
                else:
                    generated = propose(
                        root / "models/editor",
                        diagnostics,
                        dev_evidence,
                        contract,
                        attempt,
                        constrained=True,
                    )
                atomic_json(root / f"editor-attempt-{attempt}.json", generated)
                record.update(
                    {
                        "status": "rejected" if "error" in generated else "proposed",
                        "usage": generated.get("usage"),
                        "error": generated.get("error"),
                    }
                )
                if "spec" not in generated:
                    atomic_json(status_path, report)
                    continue
                model, tokenizer = tensors.load_target(root / "models/target", contract)
                reference = tensors.capture_reference(model, contract)
                name = f"candidate_{attempt}"
                try:
                    diff = tensors.apply_edit(
                        model, generated["spec"], diagnostics, contract, reference
                    )
                    record["diff"] = diff
                    if exercise is not None and not _exercise_nonzero_candidate(diff):
                        raise tensors.TensorError("engineering exercise requires a nonzero edit")
                    tensors.save_delta(
                        model, reference, checkpoints / name, contract, budget_root=root
                    )
                    tensors.load_delta(model, checkpoints / name, reference, contract)
                    dev = evaluate_arm(
                        model,
                        tokenizer,
                        dataset,
                        contract,
                        root,
                        f"development_{name}",
                        watch,
                        development=True,
                    )
                    compared = evaluation.compare(
                        dev_base["primary"],
                        dev_base["retain"],
                        dev["primary"],
                        dev["retain"],
                        contract,
                    )
                    atomic_json(root / f"{name}-development.json", compared)
                    record["development"] = compared
                    record["status"] = "evaluated"
                    retain_valid = (
                        compared["valid"]
                        and all(row["passed"] for row in compared["retain"].values())
                        and Decimal(compared.get("retain_loss_percentage_points", "Infinity"))
                        <= Decimal(
                            contract["evaluation"]["max_aggregate_retain_loss_percentage_points"]
                        )
                    )
                    record["development_retention_gate"] = retain_valid
                    if exercise is not None or retain_valid:
                        valid_candidates.append(
                            (
                                Decimal(dev["primary"]["accuracy"]),
                                Decimal(str(diff["model_relative_frobenius"])),
                                attempt,
                                name,
                            )
                        )
                except tensors.TensorError as exc:
                    record.update(status="rejected", error=str(exc))
                del model, tokenizer, reference
                gc.collect()
                atomic_json(status_path, report)
                if exercise is not None and valid_candidates:
                    break
            if not valid_candidates:
                assessed_candidates = [
                    row for row in report["attempts"] if row.get("development", {}).get("valid")
                ]
                report.update(
                    verdict=(
                        "failed" if assessed_candidates and exercise is None else "inconclusive"
                    ),
                    phase="no_valid_candidate",
                )
                report["errors"].append(
                    "no numerically valid nonzero candidate"
                    if exercise is not None
                    else "no candidate passed development retention and validity gates"
                )
                return report
            selected = (
                min(valid_candidates, key=lambda row: row[2])[3]
                if exercise is not None
                else min(valid_candidates)[3]
            )
            report["selected_candidate"] = selected
            report["selection_locked"] = True
            atomic_json(
                root / "selection.json",
                {
                    "selected": selected,
                    "locked": True,
                    "criterion": report.get("selection_criterion", "development only"),
                },
            )
            atomic_json(status_path, report)
            model, tokenizer = tensors.load_target(root / "models/target", contract)
            reference = tensors.capture_reference(model, contract)
            tensors.load_delta(model, checkpoints / selected, reference, contract)
            candidate = evaluate_arm(model, tokenizer, dataset, contract, root, selected, watch)
            report["arms"][selected] = candidate
            compared = evaluation.compare(
                base["primary"], base["retain"], candidate["primary"], candidate["retain"], contract
            )
            compared["receptive_passed"] = receptive_gate(
                base["receptive"], candidate["receptive"], contract
            )
            report["comparison"] = compared
            report["phase"] = "relearning"
            for arm in ("no_op", "graddiff", "retain_finetune", selected):
                tensors.load_delta(model, checkpoints / arm, reference, contract)
                trained = tensors.train(
                    model,
                    tokenizer,
                    [],
                    dataset["relearn"],
                    contract,
                    "relearn",
                    _emit_adapter(watch),
                    watch.check,
                    reference,
                )
                name = f"relearn_{arm}"
                atomic_json(root / f"{name}-training.json", trained)
                tensors.save_delta(model, reference, checkpoints / name, contract, budget_root=root)
                tensors.load_delta(model, checkpoints / name, reference, contract)
                report["arms"][name] = evaluate_arm(
                    model, tokenizer, dataset, contract, root, name, watch
                )
                atomic_json(status_path, report)
            recovered = report["arms"][f"relearn_{selected}"]
            after = evaluation.compare(
                base["primary"], base["retain"], recovered["primary"], recovered["retain"], contract
            )
            after["receptive_passed"] = receptive_gate(
                base["receptive"], recovered["receptive"], contract
            )
            report["relearning_comparison"] = after
            valid = compared["valid"] and after["valid"]
            passed = all(
                (
                    compared["passed"],
                    compared["receptive_passed"],
                    after["passed"],
                    after["receptive_passed"],
                )
            )
            report["verdict"] = (
                "inconclusive"
                if not valid
                else "passed the stated experimental contract"
                if passed
                else "failed"
            )
            report["phase"] = "comparison_finished"
            return report
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["verdict"] = "inconclusive"
        raise
    finally:
        _freeze_exercise_verdict(report)
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["status"] = "awaiting_independent_verification"
        atomic_json(status_path, report)


def verify(root: Path, contract: dict[str, Any], deadline: int) -> dict[str, Any]:
    """Recompute from raw evidence; partial results can never become a pass."""
    from . import tensors

    with Watchdog(root, contract, deadline, "verification") as watch:
        dataset = read_dataset(root, contract)
        result_path = root / "result.json"
        result: dict[str, Any] = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {
                "verdict": "inconclusive",
                "phase": "worker_lost",
                "errors": ["result file absent"],
                "arms": {},
            }
        )
        evidence: dict[str, Any] = {"status": "provisional", "errors": [], "recomputed": {}}
        atomic_json(root / "verification.json", evidence)
        try:
            exercise = result.get("purpose") == "engineering_exercise"
            if exercise:
                from .exercise import load_exercise_contract

                supplement = load_exercise_contract(root, contract)
                if (
                    supplement.get("purpose") != "engineering_exercise"
                    or supplement.get("efficacy_eligible") is not False
                    or result.get("efficacy_eligible") is not False
                ):
                    raise ExperimentRefused("invalid exercise purpose or efficacy eligibility")
                evidence.update(
                    purpose="engineering_exercise",
                    efficacy_eligible=False,
                    engineering_exercised=False,
                    derived_verdict="inconclusive",
                )
                if result.get("verdict") == "passed the stated experimental contract":
                    raise ExperimentRefused("engineering exercise cannot claim a passed verdict")
                result["verdict"] = "inconclusive"
            before = json.loads((root / "original-hashes-before.json").read_text())
            after = original_hashes(root, contract)
            if before != after:
                raise ExperimentRefused("original checkpoint/config/tokenizer bytes changed")
            evidence["originals_unchanged"] = True
            for arm, recorded in result.get("arms", {}).items():
                directory = root / "results" / arm
                computed = {}
                for label in ("primary", "retain"):
                    records = [
                        json.loads(x)
                        for x in (directory / f"{label}.jsonl").read_text().splitlines()
                    ]
                    evaluation.validate_observations(dataset[f"heldout_{label}"], records)
                    computed[label] = evaluation.summarize(dataset[f"heldout_{label}"], records)
                    if computed[label] != recorded[label]:
                        raise ExperimentRefused(
                            f"saved score differs from independent recomputation: {arm}/{label}"
                        )
                raw_receptive = [
                    json.loads(x) for x in (directory / "receptive.jsonl").read_text().splitlines()
                ]
                evaluation.validate_receptive(dataset["heldout_primary"], raw_receptive)
                computed["receptive"] = receptive_summary(raw_receptive)
                if computed["receptive"] != recorded["receptive"]:
                    raise ExperimentRefused(f"saved receptive score differs: {arm}")
                evidence["recomputed"][arm] = computed
                watch.progress("scores_recomputed", arm=arm)
            raw = evidence["recomputed"]
            base = raw.get("untouched")
            if base is None:
                raise ExperimentRefused("independent baseline evidence absent")
            qualification = evaluation.qualify(base["primary"], base["retain"], contract)
            evidence["qualification"] = qualification
            if exercise:
                if qualification["qualified"]:
                    raise ExperimentRefused("engineering exercise expected a failed baseline")
                evidence["qualification_failures"] = qualification["reasons"]
            elif not qualification["qualified"] and result.get("phase") != "qualification_failed":
                raise ExperimentRefused("independent baseline qualification failed")
            if qualification["qualified"] and result.get("phase") == "qualification_failed":
                raise ExperimentRefused("saved qualification phase contradicts raw baseline")
            if "no_op" in raw:
                evidence["no_op_equal"] = _same_generated_outputs(root, "untouched", "no_op")
                if not evidence["no_op_equal"]:
                    raise ExperimentRefused("independent no-op output identity failed")
            required_derivatives: set[str] = set()
            if result.get("phase") == "comparison_finished":
                selected = result["selected_candidate"]
                if exercise and selected not in {
                    f"candidate_{attempt}"
                    for attempt in range(1, contract["resources"]["candidate_attempts"] + 1)
                }:
                    raise ExperimentRefused("invalid exercise selected candidate identity")
                controls = {
                    selected if name == "selected_candidate" else name
                    for name in contract["relearning"]["arms"]
                }
                required_derivatives = controls | {f"relearn_{name}" for name in controls}
                missing = required_derivatives - raw.keys()
                if missing:
                    raise ExperimentRefused(
                        f"independent comparison arms absent: {sorted(missing)}"
                    )
            # Load saved derivative manifests from fresh source bytes, not runner memory.
            if (root / "checkpoints").exists():
                model, tokenizer = tensors.load_target(root / "models/target", contract)
                reference = tensors.capture_reference(model, contract)
                validated = []
                for path in sorted((root / "checkpoints").iterdir()):
                    manifest = tensors.load_delta(model, path, reference, contract)
                    if path.name == "no_op" and manifest["diff"]["changed_tensor_count"] != 0:
                        raise ExperimentRefused("independent no-op tensor identity failed")
                    if exercise and path.name == result.get("selected_candidate"):
                        if manifest["diff"]["changed_tensor_count"] <= 0:
                            raise ExperimentRefused(
                                "exercise requires a nonzero selected candidate"
                            )
                        evidence["selected_candidate_diff"] = manifest["diff"]
                    validated.append(path.name)
                    watch.progress("checkpoint_reloaded", checkpoint=path.name)
                evidence["validated_checkpoints"] = validated
                del model, tokenizer, reference
                gc.collect()
            if required_derivatives - set(evidence.get("validated_checkpoints", [])):
                raise ExperimentRefused("independent comparison checkpoints absent")
            if (root / "watchdog-abort.json").exists():
                result["verdict"] = "inconclusive"
                result.setdefault("errors", []).append("watchdog aborted worker")
            if result.get("phase") not in {
                "comparison_finished",
                "qualification_failed",
                "no_valid_candidate",
            }:
                result["verdict"] = "inconclusive"
            if result.get("phase") == "qualification_failed":
                result["verdict"] = "inconclusive"
            if result.get("phase") == "comparison_finished":
                selected = result["selected_candidate"]
                comparisons = {}
                for name in (selected, f"relearn_{selected}"):
                    assessed = evaluation.compare(
                        base["primary"],
                        base["retain"],
                        raw[name]["primary"],
                        raw[name]["retain"],
                        contract,
                    )
                    assessed["receptive_passed"] = receptive_gate(
                        base["receptive"], raw[name]["receptive"], contract
                    )
                    comparisons[name] = assessed
                evidence["comparisons"] = comparisons
                valid = all(row["valid"] for row in comparisons.values())
                passed = all(
                    row["passed"] and row["receptive_passed"] for row in comparisons.values()
                )
                evidence["derived_verdict"] = (
                    "inconclusive"
                    if exercise or not valid
                    else "passed the stated experimental contract"
                    if passed
                    else "failed"
                )
                if result["verdict"] != evidence["derived_verdict"]:
                    raise ExperimentRefused("saved verdict differs from independent raw comparison")
                if exercise:
                    evidence["engineering_exercised"] = (
                        "selected_candidate_diff" in evidence
                        and not (root / "watchdog-abort.json").exists()
                    )
            evidence["status"] = "verified"
        except Exception as exc:
            evidence["errors"].append(f"{type(exc).__name__}: {exc}")
            evidence["status"] = "verification_failed"
            result["verdict"] = "inconclusive"
            watch.progress("verification_failed", error=evidence["errors"][-1])
        result["status"] = evidence["status"]
        atomic_json(root / "verification.json", evidence)
        atomic_json(root / "final-result.json", result)
        write_report(root, result, evidence)
        return {"result": result, "verification": evidence}


def write_report(root: Path, result: dict[str, Any], evidence: dict[str, Any]) -> None:
    lines = [
        "LacunAI Chinese-literacy experiment",
        f"Verdict: {result['verdict']}",
        f"Final verification: {evidence['status']}",
        f"Last phase: {result.get('phase')}",
        "",
        "This experiment concerns reduced recoverability under these tests.",
        "It does not establish permanent erasure. "
        "Originals and backups retain Chinese capabilities.",
        "A true reference pretrained without Chinese is absent.",
        "",
    ]
    if result.get("purpose") == "engineering_exercise":
        lines.extend(
            [
                "Mode: engineering exercise only; original qualification failed.",
                "Efficacy eligibility: false. A passing unlearning verdict is prohibited.",
                "Verified complete engineering exercise: "
                + str(evidence.get("engineering_exercised", False)),
                "",
            ]
        )
    for error in result.get("errors", []) + evidence.get("errors", []):
        lines.append(f"Unresolved: {error}")
    for arm, suites in result.get("arms", {}).items():
        lines.extend(["", f"Arm: {arm}"])
        for label in ("primary", "retain"):
            for name, values in suites[label]["families"].items():
                lines.append(
                    f"  {name}: {values['correct']}/{values['total']}; "
                    f"ambiguous={values['ambiguous']}; incomplete={values['incomplete']}"
                )
        score = suites["receptive"]
        lines.append(f"  receptive choices: {score['correct']}/{score['total']}")
    lines.extend(
        [
            "",
            "Generated outputs, prompts, token counts and item scores are retained in results/.",
            "No deployment, merge, automatic promotion or original deletion occurred.",
        ]
    )
    (root / "comparison-report.txt").write_text("\n".join(lines) + "\n")


def runtime_manifest(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": contract["experiment"],
        "contract_sha256": sha256_file(root / "contract.json"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model_revisions": {role: contract[role]["revision"] for role in ("target", "editor")},
        "dependency_lock_sha256": sha256_file(root / "requirements.lock"),
        "source_bundle_sha256": sha256_file(root / "bundle.sha256"),
        "limits": contract["resources"],
        "inference": contract["runtime"],
        "numerical_tolerances": {"relative_norm_bound_atol": "1e-9", "diagnosis_epsilon": "1e-12"},
        "reference_without_chinese": "absent",
        "promotion": "disabled",
        "cost": contract["cost"],
        "nondeterminism": "deterministic algorithms required; cross-platform bit identity unproven",
    }
