# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Exercise orchestration tests; substituted boundaries do not certify model behavior."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from lacunai.experimental import editor, runner, tensors


@pytest.fixture
def flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Drive the actual run state machine with explicit non-model boundary fixtures."""
    state = SimpleNamespace(calls=[], differences=[], development_valid=True, baseline_error=None)
    state.policy = {
        "resources": {"candidate_attempts": 3, "editor_calls": 3},
        "evaluation": {"max_aggregate_retain_loss_percentage_points": "5"},
    }
    state.diff = {
        "changed_tensor_count": 1,
        "model_relative_frobenius": 0.001,
        "tensors": {"allowed.weight": {"delta_norm_squared": 0.01}},
    }
    state.base = {"primary": {"accuracy": "1"}, "retain": {}, "receptive": {}}
    dataset = {
        "forget_train": ["forget fixture"],
        "retain_train": ["retain fixture"],
        "relearn": ["benign fixture"],
        "dev_primary": [
            {"id": "dev:0", "prompts": ["development"], "answer": "fixture", "family": "dev"}
        ],
    }

    class Watch:
        def __init__(self, *args: Any) -> None:
            pass

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def check(self) -> None:
            pass

        def progress(self, event: str, **fields: Any) -> None:
            state.calls.append(("progress", event))

    def load_baseline(*args: Any) -> dict[str, Any]:
        state.calls.append(("baseline_import",))
        if state.baseline_error:
            raise runner.ExperimentRefused(state.baseline_error)
        return deepcopy(state.base)

    # Import boundary only: root-owned baseline provenance validation has its own tests.
    baseline_module = ModuleType("lacunai.experimental.exercise")
    baseline_module.load_baseline = load_baseline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, baseline_module.__name__, baseline_module)
    monkeypatch.setattr(runner, "Watchdog", Watch)
    monkeypatch.setattr(runner, "read_dataset", lambda *args: dataset)
    monkeypatch.setattr(runner, "original_hashes", lambda *args: {"source": "unchanged"})
    monkeypatch.setattr(tensors, "numerical_smoke", lambda *args: {"fixture": True})
    monkeypatch.setattr(tensors, "numeric_selftest", lambda *args: {"fixture": True})
    monkeypatch.setattr(editor, "constrained_numeric_selftest", lambda: {"fixture": True})
    monkeypatch.setattr(tensors, "load_target", lambda *args: (object(), object()))
    monkeypatch.setattr(tensors, "capture_reference", lambda *args: {"fingerprint": "fixture"})
    monkeypatch.setattr(tensors, "restore_reference", lambda *args: None)
    monkeypatch.setattr(tensors, "save_delta", lambda *args, **kwargs: None)
    monkeypatch.setattr(tensors, "load_delta", lambda *args: None)
    monkeypatch.setattr(tensors, "diagnose", lambda *args: {})
    monkeypatch.setattr(runner, "_same_generated_outputs", lambda *args: True)
    monkeypatch.setattr(runner, "receptive_gate", lambda *args: True)
    monkeypatch.setattr(
        runner.evaluation,
        "qualify",
        lambda *args: {"qualified": False, "reasons": ["baseline qualification failed"]},
    )

    def train(*args: Any) -> dict[str, Any]:
        state.calls.append(("train", args[5]))
        return {"fixture": True}

    def propose(*args: Any, **kwargs: Any) -> dict[str, Any]:
        state.calls.append(("editor", args[4]))
        state.calls.append(("constrained", kwargs.get("constrained", False)))
        return {"spec": {"fixture": args[4]}, "usage": {"fixture": True}}

    def apply(*args: Any) -> dict[str, Any]:
        state.calls.append(("apply",))
        return deepcopy(state.differences.pop(0) if state.differences else state.diff)

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        state.calls.append(("evaluate", args[5]))
        result = deepcopy(state.base)
        result["primary"]["fixture_development"] = args[5].startswith("development_candidate")
        return result

    def compare(*args: Any) -> dict[str, Any]:
        valid = not args[2].get("fixture_development") or state.development_valid
        return {
            "valid": valid,
            "passed": valid,
            "retain": {"fixture": {"passed": valid}},
            "retain_loss_percentage_points": "0" if valid else "100",
            "errors": [] if valid else ["ambiguous development answer; retain regression"],
        }

    monkeypatch.setattr(tensors, "train", train)
    monkeypatch.setattr(editor, "propose", propose)
    monkeypatch.setattr(tensors, "apply_edit", apply)
    monkeypatch.setattr(runner, "evaluate_arm", evaluate)
    monkeypatch.setattr(runner.evaluation, "compare", compare)

    def execute(label: str, exercise: bool = True) -> dict[str, Any]:
        root = tmp_path.resolve() / label
        root.mkdir()
        state.root = root
        return runner.run(root, state.policy, 1000, exercise={"frozen": True} if exercise else None)

    state.execute = execute
    return state


def test_default_failed_qualification_still_stops_before_controls(flow: Any) -> None:
    report = flow.execute("default", exercise=False)
    assert report["phase"] == "qualification_failed"
    assert not any(call[0] in {"train", "editor", "baseline_import"} for call in flow.calls)
    assert ("evaluate", "untouched") in flow.calls
    assert "purpose" not in report


def test_exercise_imports_baseline_runs_all_arms_and_cannot_pass(flow: Any) -> None:
    report = flow.execute("exercise")
    assert report["phase"] == "comparison_finished"
    assert report["qualification"]["qualified"] is False
    assert report["errors"] == ["baseline qualification failed"]
    assert ("baseline_import",) in flow.calls
    assert ("evaluate", "untouched") not in flow.calls
    assert [call[1] for call in flow.calls if call[0] == "train"] == [
        "graddiff",
        "retain_finetune",
        "relearn",
        "relearn",
        "relearn",
        "relearn",
    ]
    assert [call[1] for call in flow.calls if call[0] == "editor"] == [1]
    assert ("constrained", True) in flow.calls
    assert report["comparison"]["passed"] is True
    assert report["relearning_comparison"]["passed"] is True
    assert report["verdict"] == "inconclusive"
    assert report["efficacy_eligible"] is False
    assert report["promotion"] == "disabled"
    saved = json.loads((flow.root / "result.json").read_text())
    assert saved["verdict"] == "inconclusive"
    assert set(report["arms"]) == {
        "untouched",
        "no_op",
        "graddiff",
        "retain_finetune",
        "candidate_1",
        "relearn_no_op",
        "relearn_graddiff",
        "relearn_retain_finetune",
        "relearn_candidate_1",
    }


def test_exercise_preserves_development_failure_without_behavioral_selection(flow: Any) -> None:
    flow.development_valid = False
    report = flow.execute("development-failed")
    assert report["selected_candidate"] == "candidate_1"
    assert report["attempts"][0]["development_retention_gate"] is False
    assert report["attempts"][0]["development"]["errors"]
    assert report["verdict"] == "inconclusive"


def test_zero_candidate_is_rejected_before_behavioral_test_then_next_is_selected(flow: Any) -> None:
    flow.differences = [{"changed_tensor_count": 0, "model_relative_frobenius": 0, "tensors": {}}]
    report = flow.execute("zero-first")
    assert report["selected_candidate"] == "candidate_2"
    assert report["attempts"][0]["status"] == "rejected"
    assert "nonzero edit" in report["attempts"][0]["error"]
    assert ("evaluate", "development_candidate_1") not in flow.calls
    assert [call[1] for call in flow.calls if call[0] == "editor"] == [1, 2]


def test_failed_baseline_import_does_not_train_or_call_editor(flow: Any) -> None:
    flow.baseline_error = "baseline raw hash mismatch"
    with pytest.raises(runner.ExperimentRefused, match="raw hash mismatch"):
        flow.execute("bad-baseline")
    assert not any(call[0] in {"train", "editor"} for call in flow.calls)
    saved = json.loads((flow.root / "result.json").read_text())
    assert saved["verdict"] == "inconclusive"


@pytest.mark.parametrize("norm", ["NaN", "Infinity", "-Infinity", "0", "-0.01"])
def test_nonzero_gate_rejects_nonfinite_or_nonpositive_norm(flow: Any, norm: str) -> None:
    flow.diff["model_relative_frobenius"] = norm
    assert not runner._exercise_nonzero_candidate(flow.diff)


def test_removed_verdict_guard_is_red_then_restored_green(
    flow: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as plant:
        plant.setattr(runner, "_freeze_exercise_verdict", lambda report: None)
        unguarded = flow.execute("verdict-removed")
        with pytest.raises(AssertionError) as failure:
            assert unguarded["verdict"] == "inconclusive", "exercise incorrectly asserted efficacy"
        print(f"RED verdict guard removed: {failure.value}", flush=True)
    restored = flow.execute("verdict-restored")
    assert restored["verdict"] == "inconclusive"


def test_removed_nonzero_guard_is_red_then_restored_green(
    flow: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    flow.diff = {"changed_tensor_count": 0, "model_relative_frobenius": 0, "tensors": {}}
    with monkeypatch.context() as plant:
        plant.setattr(runner, "_exercise_nonzero_candidate", lambda diff: True)
        unguarded = flow.execute("nonzero-removed")
        with pytest.raises(AssertionError) as failure:
            assert unguarded["phase"] == "no_valid_candidate", "zero edit was selected"
        print(f"RED nonzero guard removed: {failure.value}", flush=True)
    restored = flow.execute("nonzero-restored")
    assert restored["phase"] == "no_valid_candidate"
    assert restored["verdict"] == "inconclusive"
    assert len(restored["attempts"]) == 3
