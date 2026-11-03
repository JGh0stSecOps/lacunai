# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""A separately frozen, permanently inconclusive engineering exercise.

This supplements a completed but unqualified baseline. It never turns that
baseline into an efficacy claim and never changes the original scoring rules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import evaluation
from .artifacts import atomic_json, secure_path
from .spec import APPROVED_V1_SHA256, ExperimentRefused

EXERCISE_POLICY: dict[str, Any] = {
    "schema_version": 1,
    "purpose": "engineering_exercise",
    "efficacy_eligible": False,
    "verdict_only": "inconclusive",
    "parent_contract_sha256": APPROVED_V1_SHA256,
    "baseline": "completed_independently_verified_failed_qualification",
    "required_learned_family": "read_zh_answer_en_or_numeric",
    "selection": "first_numerically_valid_nonzero_candidate_by_attempt",
    "editor": "same_local_pinned_model_constrained_column_and_strength_choice",
    "editor_calls_max": 3,
    "candidate_attempts_max": 3,
    "original_deadline_epoch": 1793677658,
    "threshold_changes": False,
    "dataset_changes": False,
    "promotion": "disabled",
}
EXERCISE_BYTES = (json.dumps(EXERCISE_POLICY, sort_keys=True, indent=2) + "\n").encode()
EXERCISE_SHA256 = hashlib.sha256(EXERCISE_BYTES).hexdigest()
BASELINE_FILES = (
    "contract.json",
    "manifest.json",
    "final-result.json",
    "verification.json",
    "source-tensors.json",
    "original-hashes-before.json",
    "datasets/dataset.json",
    "datasets/manifest.json",
    "results/untouched/primary.jsonl",
    "results/untouched/primary-summary.json",
    "results/untouched/retain.jsonl",
    "results/untouched/retain-summary.json",
    "results/untouched/receptive.jsonl",
    "results/untouched/receptive-summary.json",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_exercise_contract(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Require the exact supplement and its explicit authorization receipt."""
    path = secure_path(root, "exercise-contract.json")
    if not path.is_file() or path.read_bytes() != EXERCISE_BYTES:
        raise ExperimentRefused("missing or altered frozen engineering exercise contract")
    approval = json.loads(secure_path(root, "approval.json").read_text())
    if (
        approval.get("approved") is not True
        or approval.get("engineering_exercise_authorized") is not True
        or approval.get("exercise_contract_sha256") != EXERCISE_SHA256
        or approval.get("contract_sha256") != APPROVED_V1_SHA256
    ):
        raise ExperimentRefused("explicit matching engineering exercise authorization absent")
    if (
        contract["resources"]["editor_calls"] != EXERCISE_POLICY["editor_calls_max"]
        or contract["resources"]["candidate_attempts"] != EXERCISE_POLICY["candidate_attempts_max"]
        or contract["promotion"] != "disabled"
    ):
        raise ExperimentRefused("engineering exercise changes the approved bounds")
    return dict(EXERCISE_POLICY)


def verify_baseline_files(root: Path) -> dict[str, str]:
    """Verify a complete closed inventory; never follow external paths or links."""
    baseline = secure_path(root, "baseline")
    manifest = json.loads(secure_path(baseline, "baseline-manifest.json").read_text())
    hashes = manifest.get("files")
    if not isinstance(hashes, dict) or set(hashes) != set(BASELINE_FILES):
        raise ExperimentRefused("baseline evidence inventory differs from the frozen inventory")
    for relative, expected in hashes.items():
        path = secure_path(baseline, relative)
        if not path.is_file() or _digest(path) != expected:
            raise ExperimentRefused(f"baseline evidence hash mismatch: {relative}")
    return dict(hashes)


def load_baseline(
    root: Path, contract: dict[str, Any], reference: dict[str, Any], watch: Any
) -> dict[str, Any]:
    """Recompute an untouched baseline and copy verified evidence into this run."""
    from .runner import read_dataset, receptive_summary

    policy = load_exercise_contract(root, contract)
    watch.progress("validating_reused_baseline")
    hashes = verify_baseline_files(root)
    source = root / "baseline"
    if hashes["contract.json"] != APPROVED_V1_SHA256:
        raise ExperimentRefused("reused baseline has a different contract")
    for relative in ("datasets/dataset.json", "datasets/manifest.json"):
        if _digest(root / relative) != hashes[relative]:
            raise ExperimentRefused("reused baseline has different frozen data")
    runtime = json.loads((source / "manifest.json").read_text())
    if (
        runtime["inference"] != contract["runtime"]
        or runtime["model_revisions"]
        != {role: contract[role]["revision"] for role in ("target", "editor")}
        or runtime["dependency_lock_sha256"] != _digest(root / "requirements.lock")
    ):
        raise ExperimentRefused("baseline inference or dependency conditions differ")
    prior = json.loads((source / "final-result.json").read_text())
    checked = json.loads((source / "verification.json").read_text())
    if (
        prior.get("phase") != "qualification_failed"
        or prior.get("status") != "verified"
        or prior.get("verdict") != "inconclusive"
        or prior.get("attempts") != []
        or set(prior.get("arms", {})) != {"untouched"}
        or checked.get("status") != "verified"
        or checked.get("errors") != []
    ):
        raise ExperimentRefused("baseline is not a verified unedited failed qualification")
    tensors = json.loads((source / "source-tensors.json").read_text())
    if (
        tensors["fingerprint"] != reference["fingerprint"]
        or tensors["hashes"] != reference["hashes"]
    ):
        raise ExperimentRefused("reused baseline target tensors differ")
    if json.loads((root / "original-hashes-before.json").read_text()) != json.loads(
        (source / "original-hashes-before.json").read_text()
    ):
        raise ExperimentRefused("reused baseline original checkpoint files differ")
    dataset = read_dataset(root, contract)
    result: dict[str, Any] = {}
    for label in ("primary", "retain"):
        items = dataset[f"heldout_{label}"]
        records = [
            json.loads(line)
            for line in (source / f"results/untouched/{label}.jsonl").read_text().splitlines()
        ]
        evaluation.validate_observations(items, records)
        result[label] = evaluation.summarize(items, records)
    receptive = [
        json.loads(line)
        for line in (source / "results/untouched/receptive.jsonl").read_text().splitlines()
    ]
    evaluation.validate_receptive(dataset["heldout_primary"], receptive)
    result["receptive"] = receptive_summary(receptive)
    if result != prior["arms"]["untouched"] or result != checked["recomputed"]["untouched"]:
        raise ExperimentRefused("baseline raw recomputation differs from independent evidence")
    qualification = evaluation.qualify(result["primary"], result["retain"], contract)
    if qualification["qualified"]:
        raise ExperimentRefused("engineering supplement requires failed original qualification")
    learned = result["primary"]["families"][policy["required_learned_family"]]
    if (
        learned["correct"] < contract["evaluation"]["minimum_qualification_correct_per_family"]
        or learned["ambiguous"]
        or learned["incomplete"]
    ):
        raise ExperimentRefused("required Chinese reading knowledge has not been demonstrated")
    destination = secure_path(root, "results/untouched")
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for relative in BASELINE_FILES:
        if relative.startswith("results/untouched/"):
            data = (source / relative).read_bytes()
            with (destination / Path(relative).name).open("xb") as stream:
                stream.write(data)
    atomic_json(
        root / "baseline-reuse.json",
        {
            "purpose": policy["purpose"],
            "efficacy_eligible": False,
            "source_evidence_hashes": hashes,
            "qualification": qualification,
            "learned_family": learned,
            "source_tensors_fingerprint": reference["fingerprint"],
        },
    )
    watch.progress("baseline_recomputed", qualified=False, efficacy_eligible=False)
    return result
