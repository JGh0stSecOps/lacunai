# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Real phase-boundary behavior without loading or simulating a model result."""

from __future__ import annotations

import json
import runpy
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from lacunai.experimental import batch, evaluation, resources, runner, tensors
from lacunai.experimental.data import PRIMARY, RETAIN


@pytest.fixture
def pilot_root(tmp_path: Path) -> Path:
    root = tmp_path.resolve() / "neurailyzer-pilot-20261102"
    root.mkdir()
    return root


@pytest.fixture
def healthy() -> dict[str, Any]:
    return {
        "monotonic": 1.0,
        "available_bytes": 40 << 30,
        "swap_used_bytes": 1 << 30,
        "root_free_bytes": 12 << 30,
        "bulk_free_bytes": 200 << 30,
        "load1": 10.0,
        "pressure": {"memory": {"some": 0.0}, "io": {"full": 0.0}},
        "workers": {"kvm": [101], "llama-server": [202], "runner": []},
    }


@pytest.fixture
def contract() -> dict[str, Any]:
    return {
        "resources": {
            "min_host_available_bytes": 24 << 30,
            "min_root_free_bytes": 8 << 30,
            "min_bulk_free_bytes": 100 << 30,
            "max_load1": 20,
            "memory_psi_some_avg10": 1.0,
            "io_psi_full_avg10": 5.0,
            "sampling_interval_seconds": 0.01,
            "pressure_consecutive_samples": 3,
            "disk_max_bytes": 1 << 20,
        }
    }


@pytest.mark.parametrize(
    "change,expected",
    [
        ("vm", "guest process state changed"),
        ("runner", "new host runner worker observed"),
        ("swap", "host swap use increased"),
        ("inference", "existing inference process state changed"),
    ],
)
def test_next_phase_does_not_rebase_existing_workload_state(
    pilot_root: Path,
    healthy: dict[str, Any],
    contract: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    expected: str,
) -> None:
    (pilot_root / "host-before.json").write_text(json.dumps(healthy))
    now = deepcopy(healthy)
    if change == "vm":
        now["workers"]["kvm"] = [303]
    elif change == "runner":
        now["workers"]["runner"] = [404]
    elif change == "inference":
        now["workers"]["llama-server"] = []
    else:
        now["swap_used_bytes"] += 4096
    monkeypatch.setattr(resources, "host_sample", lambda root: now)
    entered_work = False
    with (
        pytest.raises(resources.ResourceAbort, match=expected),
        resources.Watchdog(pilot_root, contract, int(time.time()) + 10, "verification"),
    ):
        entered_work = True
    assert not entered_work
    assert json.loads((pilot_root / "host-before.json").read_text()) == healthy


def test_watchdog_emits_flushed_progress_before_work_finishes_and_reads_back_host(
    pilot_root: Path,
    healthy: dict[str, Any],
    contract: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources, "host_sample", lambda root: deepcopy(healthy))
    observed: list[tuple[dict[str, Any], bool]] = []
    active = threading.Event()

    def capture(message: str, *, flush: bool) -> None:
        event = json.loads(message)
        observed.append((event, flush))
        if event["state"] == "active":
            active.set()

    monkeypatch.setattr(resources, "print", capture, raising=False)
    with resources.Watchdog(pilot_root, contract, int(time.time()) + 10, "experiment"):
        assert observed[0][0]["state"] == "starting"
        assert active.wait(1), "background operation must emit progress while still active"
        assert not any(event["state"] == "phase_finished" for event, _ in observed)
    assert all(flushed for _, flushed in observed)
    assert observed[-1][0]["state"] == "phase_finished"
    after = json.loads((pilot_root / "logs/host-after-experiment.json").read_text())
    assert after == healthy


def test_admission_failure_ends_active_progress_with_error(
    pilot_root: Path,
    healthy: dict[str, Any],
    contract: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    healthy["available_bytes"] = 23 << 30
    monkeypatch.setattr(resources, "host_sample", lambda root: healthy)
    with (
        pytest.raises(resources.ResourceAbort, match="available_bytes"),
        resources.Watchdog(pilot_root, contract, int(time.time()) + 10, "verification"),
    ):
        pytest.fail("work must never start after failed admission")
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]["state"] == "starting"
    assert events[-1]["state"] in {"failed", "aborted"}
    assert "available_bytes" in events[-1]["error"]


def test_body_failure_ends_progress_and_preserves_measured_after_state(
    pilot_root: Path,
    healthy: dict[str, Any],
    contract: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(resources, "host_sample", lambda root: healthy)
    with (
        pytest.raises(ValueError, match="checkpoint hash mismatch"),
        resources.Watchdog(pilot_root, contract, int(time.time()) + 10, "verification"),
    ):
        raise ValueError("checkpoint hash mismatch")
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["state"] == "failed"
    assert events[-1]["error"] == "checkpoint hash mismatch"
    assert json.loads((pilot_root / "logs/host-after-verification.json").read_text()) == healthy


def test_actual_batch_entrypoint_finalizes_a_refused_heavy_verification(
    pilot_root: Path,
    healthy: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    previous = {
        "verdict": "passed the stated experimental contract",
        "errors": [],
        "arms": {"untouched": {"raw_evidence": "preserved"}},
    }
    (pilot_root / "result.json").write_text(json.dumps(previous))
    monkeypatch.setattr(batch, "host_sample", lambda root: healthy)

    def refused() -> None:
        raise resources.ResourceAbort("admission refused: available_bytes below approved floor")

    monkeypatch.setattr(batch, "main", refused)
    monkeypatch.setattr(sys, "argv", ["experimental_batch.py", "verify", "--root", str(pilot_root)])
    entry = Path(__file__).parents[1] / "scripts/experimental_batch.py"
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(entry), run_name="__main__")
    assert exited.value.code == 1
    final = json.loads((pilot_root / "final-result.json").read_text())
    assert final["verdict"] == "inconclusive"
    assert final["status"] == "verification_unavailable"
    assert final["arms"] == previous["arms"]
    assert final["host_after"] == healthy
    assert (
        "Independent verification unavailable" in (pilot_root / "comparison-report.txt").read_text()
    )
    assert '"state": "failed"' in capsys.readouterr().err


def test_finalization_reports_unreadable_host_without_inventing_state(
    pilot_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(root: Path) -> dict[str, Any]:
        raise OSError("host metrics unavailable")

    monkeypatch.setattr(batch, "host_sample", unavailable)
    result = batch.finalize_exception(pilot_root, "verify", resources.ResourceAbort("pressure"))
    assert result["verdict"] == "inconclusive"
    assert "host_after" not in result
    assert any("host metrics unavailable" in error for error in result["errors"])
    assert (pilot_root / "final-result.json").is_file()


def _verification_fixture(
    root: Path, policy: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct synthetic verifier evidence; never execute or certify a model."""
    policy.update(
        runtime={"seed": 20261102},
        target={"files": []},
        editor={"files": []},
        relearning={"arms": ["no_op", "graddiff", "retain_finetune", "selected_candidate"]},
        evaluation={
            "primary_families": list(PRIMARY),
            "retain_families": list(RETAIN),
            "items_per_primary_family": 16,
            "items_per_retain_family": 16,
            "prompt_variants_per_item": 4,
            "minimum_qualification_correct_per_family": 12,
            "max_chinese_recovery": "0.25",
            "min_relative_recovery_reduction": "0.50",
            "max_regressions_per_retain_family": 1,
            "max_aggregate_retain_loss_percentage_points": "5",
            "max_receptive_accuracy": "0.375",
            "min_receptive_accuracy_drop_percentage_points": "25",
        },
    )
    policy["resources"]["disk_max_bytes"] = 32 << 20
    runner.freeze_dataset(root, policy)
    dataset = runner.read_dataset(root, policy)
    result: dict[str, Any] = {
        "verdict": "passed the stated experimental contract",
        "phase": "comparison_finished",
        "arms": {},
        "qualification": {"qualified": True},
        "no_op_equal": True,
        "selected_candidate": "candidate_1",
        "selection_locked": True,
        "errors": [],
    }
    names = ["untouched", "no_op", "graddiff", "retain_finetune", "candidate_1"]
    names += [f"relearn_{name}" for name in names[1:]]
    for arm in names:
        candidate = "candidate_1" in arm
        directory = root / "results" / arm
        directory.mkdir(parents=True)
        summaries = {}
        for label in ("primary", "retain"):
            items = dataset[f"heldout_{label}"]
            records = [
                {
                    "item_id": item["id"],
                    "variant": variant,
                    "family": item["family"],
                    "prompt": prompt,
                    "output": "" if candidate and label == "primary" else item["answer"],
                    "token_ids": [7],
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "truncated": False,
                    "first_step_logits_sha256": "a" * 64,
                }
                for item in items
                for variant, prompt in enumerate(item["prompts"])
            ]
            runner._jsonl(directory / f"{label}.jsonl", records)
            summaries[label] = evaluation.summarize(items, records)
        receptive = []
        for item in dataset["heldout_primary"]:
            if "mcq" not in item:
                continue
            mcq = item["mcq"]
            predicted = (mcq["correct_index"] + int(candidate)) % 4
            receptive.append(
                {
                    "item_id": item["id"],
                    "prompt": mcq["prompt"],
                    "correct_index": mcq["correct_index"],
                    "predicted": predicted,
                    "tied": False,
                    "correct": not candidate,
                    "scores": [
                        {
                            "choice": choice,
                            "mean_log_probability": -1.0 if index == predicted else -2.0,
                            "token_log_probabilities": [-1.0 if index == predicted else -2.0],
                            "tokens": [index + 1],
                        }
                        for index, choice in enumerate(mcq["choices"])
                    ],
                }
            )
        runner._jsonl(directory / "receptive.jsonl", receptive)
        summaries["receptive"] = runner.receptive_summary(receptive)
        result["arms"][arm] = summaries
        if arm != "untouched":
            (root / "checkpoints" / arm).mkdir(parents=True)
    (root / "original-hashes-before.json").write_text("{}")
    monkeypatch.setattr(tensors, "load_target", lambda *args: (object(), object()))
    monkeypatch.setattr(tensors, "capture_reference", lambda *args: {})
    monkeypatch.setattr(
        tensors,
        "load_delta",
        lambda model, path, reference, contract: {
            "diff": {"changed_tensor_count": int(path.name != "no_op")}
        },
    )
    return result, dataset


@pytest.mark.parametrize(
    "tamper", ["qualification", "no_op", "control", "verdict", "no_op_tensor", "checkpoint"]
)
def test_verifier_recomputes_prerequisites_and_verdict_from_raw_evidence(
    pilot_root: Path,
    healthy: dict[str, Any],
    contract: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tamper: str,
) -> None:
    monkeypatch.setattr(resources, "host_sample", lambda root: healthy)
    result, dataset = _verification_fixture(pilot_root, contract, monkeypatch)
    if tamper == "qualification":
        for arm in ("untouched", "no_op"):
            path = pilot_root / "results" / arm / "primary.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            failed = {item["id"] for item in dataset["heldout_primary"][11:16]}
            for record in records:
                if record["item_id"] in failed:
                    record["output"] = ""
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            result["arms"][arm]["primary"] = evaluation.summarize(
                dataset["heldout_primary"], records
            )
    elif tamper == "no_op":
        path = pilot_root / "results/no_op/primary.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["token_ids"] = [8]
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
    elif tamper == "control":
        del result["arms"]["graddiff"]
    elif tamper == "no_op_tensor":
        monkeypatch.setattr(
            tensors, "load_delta", lambda *args: {"diff": {"changed_tensor_count": 1}}
        )
    elif tamper == "checkpoint":
        path = pilot_root / "checkpoints/graddiff"
        path.rename(path.with_name("different_checkpoint"))
    else:
        result["verdict"] = "failed"
    (pilot_root / "result.json").write_text(json.dumps(result))
    actual = runner.verify(pilot_root, contract, int(time.time()) + 30)
    assert actual["result"]["verdict"] == "inconclusive"
    assert actual["verification"]["status"] == "verification_failed"
    assert actual["verification"]["errors"]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(event.get("state") == "scores_recomputed" for event in events)
    assert any(event.get("state") == "verification_failed" for event in events)


def test_verifier_accepts_consistent_complete_fixture_and_preserves_recomputed_prerequisites(
    pilot_root: Path,
    healthy: dict[str, Any],
    contract: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources, "host_sample", lambda root: healthy)
    result, _ = _verification_fixture(pilot_root, contract, monkeypatch)
    (pilot_root / "result.json").write_text(json.dumps(result))
    actual = runner.verify(pilot_root, contract, int(time.time()) + 30)
    assert actual["verification"]["status"] == "verified"
    assert actual["verification"]["qualification"]["qualified"]
    assert actual["verification"]["no_op_equal"]
    assert actual["verification"]["derived_verdict"] == "passed the stated experimental contract"
