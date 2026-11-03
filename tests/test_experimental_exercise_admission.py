# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Admission evidence for the explicitly non-efficacy supplement."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from lacunai.experimental.exercise import (
    BASELINE_FILES,
    EXERCISE_BYTES,
    EXERCISE_SHA256,
    load_exercise_contract,
    verify_baseline_files,
)
from lacunai.experimental.spec import APPROVED_V1_SHA256, ExperimentRefused


@pytest.fixture
def admission(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    (tmp_path / "exercise-contract.json").write_bytes(EXERCISE_BYTES)
    approval = {
        "approved": True,
        "engineering_exercise_authorized": True,
        "exercise_contract_sha256": EXERCISE_SHA256,
        "contract_sha256": APPROVED_V1_SHA256,
    }
    (tmp_path / "approval.json").write_text(json.dumps(approval))
    return tmp_path, {
        "resources": {"editor_calls": 3, "candidate_attempts": 3},
        "promotion": "disabled",
    }


def test_explicit_supplement_never_becomes_efficacy_eligible(admission: tuple) -> None:
    root, contract = admission
    result = load_exercise_contract(root, contract)
    assert result["efficacy_eligible"] is False
    assert result["verdict_only"] == "inconclusive"
    altered = json.loads(EXERCISE_BYTES)
    altered["efficacy_eligible"] = True
    (root / "exercise-contract.json").write_text(json.dumps(altered))
    with pytest.raises(ExperimentRefused, match="altered"):
        load_exercise_contract(root, contract)


def test_generic_approval_does_not_authorize_exercise(admission: tuple) -> None:
    root, contract = admission
    value = json.loads((root / "approval.json").read_text())
    del value["engineering_exercise_authorized"]
    (root / "approval.json").write_text(json.dumps(value))
    with pytest.raises(ExperimentRefused, match="authorization"):
        load_exercise_contract(root, contract)


def test_supplement_cannot_expand_editor_budget(admission: tuple) -> None:
    root, contract = admission
    contract = deepcopy(contract)
    contract["resources"]["editor_calls"] = 4
    with pytest.raises(ExperimentRefused, match="bounds"):
        load_exercise_contract(root, contract)


def test_baseline_inventory_binds_every_byte_and_refuses_missing_rows(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    hashes = {}
    for name in BASELINE_FILES:
        path = baseline / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture evidence: " + name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = baseline / "baseline-manifest.json"
    manifest.write_text(json.dumps({"files": hashes}))
    assert verify_baseline_files(tmp_path) == hashes
    target = baseline / "results/untouched/primary.jsonl"
    target.write_text("changed raw output")
    with pytest.raises(ExperimentRefused, match="hash mismatch"):
        verify_baseline_files(tmp_path)
    hashes.pop("results/untouched/primary.jsonl")
    manifest.write_text(json.dumps({"files": hashes}))
    with pytest.raises(ExperimentRefused, match="inventory"):
        verify_baseline_files(tmp_path)
