# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Exercise verification fixtures cannot establish model unlearning efficacy."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lacunai.experimental import evaluation, resources, runner, tensors
from lacunai.experimental.spec import ExperimentRefused
from tests import test_experimental_runner as fixtures

pilot_root = fixtures.pilot_root
healthy = fixtures.healthy
contract = fixtures.contract


def _exercise_fixture(
    root: Path, policy: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Any], ModuleType]:
    result, dataset = fixtures._verification_fixture(root, policy, monkeypatch)
    policy["resources"]["candidate_attempts"] = 3
    result.update(purpose="engineering_exercise", efficacy_eligible=False, verdict="inconclusive")
    failed = {
        item["id"]
        for item in [row for row in dataset["heldout_primary"] if row["family"] == "write_zh"][:5]
    }
    for arm in ("untouched", "no_op"):
        path = root / "results" / arm / "primary.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:
            if record["item_id"] in failed:
                record["output"] = ""
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        result["arms"][arm]["primary"] = evaluation.summarize(dataset["heldout_primary"], records)
    module = ModuleType("lacunai.experimental.exercise")
    monkeypatch.setattr(
        module,
        "load_exercise_contract",
        lambda supplied_root, supplied_policy: {
            "purpose": "engineering_exercise",
            "efficacy_eligible": False,
        },
        raising=False,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return result, dataset, module


def test_validated_exercise_remains_inconclusive_despite_complete_edit_evidence(
    pilot_root: Path,
    healthy: dict[str, Any],
    contract: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources, "host_sample", lambda root: healthy)
    result, _, _ = _exercise_fixture(pilot_root, contract, monkeypatch)
    (pilot_root / "result.json").write_text(json.dumps(result))
    actual = runner.verify(pilot_root, contract, int(time.time()) + 30)
    assert actual["result"]["verdict"] == "inconclusive"
    evidence = actual["verification"]
    assert evidence["status"] == "verified"
    assert evidence["derived_verdict"] == "inconclusive"
    assert evidence["efficacy_eligible"] is False
    assert evidence["qualification"]["qualified"] is False
    assert evidence["qualification_failures"]
    assert evidence["engineering_exercised"] is True
    assert evidence["selected_candidate_diff"]["changed_tensor_count"] == 1


@pytest.mark.parametrize(
    "tamper, expected",
    [
        ("missing_contract", "exercise addendum mismatch"),
        ("passed", "cannot claim a passed verdict"),
        ("eligible", "efficacy eligibility"),
        ("qualified", "expected a failed baseline"),
        ("zero_delta", "nonzero selected candidate"),
        ("missing_control", "comparison arms absent"),
        ("missing_relearn", "comparison arms absent"),
        ("no_op", "no-op output identity failed"),
        ("control_selected", "selected candidate identity"),
    ],
)
def test_exercise_cannot_bypass_independent_contract_and_evidence_guards(
    pilot_root: Path,
    healthy: dict[str, Any],
    contract: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tamper: str,
    expected: str,
) -> None:
    monkeypatch.setattr(resources, "host_sample", lambda root: healthy)
    result, dataset, module = _exercise_fixture(pilot_root, contract, monkeypatch)
    if tamper == "missing_contract":

        def rejected(*args: Any) -> Any:
            raise ExperimentRefused("exercise addendum mismatch")

        monkeypatch.setattr(module, "load_exercise_contract", rejected)
    elif tamper == "passed":
        result["verdict"] = "passed the stated experimental contract"
    elif tamper == "eligible":
        result["efficacy_eligible"] = True
    elif tamper == "qualified":
        for arm in ("untouched", "no_op"):
            path = pilot_root / "results" / arm / "primary.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            by_id = {item["id"]: item for item in dataset["heldout_primary"]}
            for row in records:
                row["output"] = by_id[row["item_id"]]["answer"]
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            result["arms"][arm]["primary"] = evaluation.summarize(
                dataset["heldout_primary"], records
            )
    elif tamper == "zero_delta":
        monkeypatch.setattr(
            tensors, "load_delta", lambda *args: {"diff": {"changed_tensor_count": 0}}
        )
    elif tamper == "missing_control":
        del result["arms"]["graddiff"]
    elif tamper == "missing_relearn":
        del result["arms"]["relearn_retain_finetune"]
    elif tamper == "control_selected":
        result["selected_candidate"] = "graddiff"
    else:
        path = pilot_root / "results/no_op/primary.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["token_ids"] = [8]
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
    (pilot_root / "result.json").write_text(json.dumps(result))
    actual = runner.verify(pilot_root, contract, int(time.time()) + 30)
    assert actual["result"]["verdict"] == "inconclusive"
    assert actual["verification"]["status"] == "verification_failed"
    assert any(expected in error for error in actual["verification"]["errors"])
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(event.get("state") == "verification_failed" for event in events)
