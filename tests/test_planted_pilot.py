"""Structural pilot contract tests; none loads or trains a model."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import pytest

from lacunai.core.spec import MappingProbe, Probe, TargetSpec

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "planted_fact_pilot.py"
_SPEC = importlib.util.spec_from_file_location("planted_fact_pilot", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
pilot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pilot)


def contract() -> TargetSpec:
    return TargetSpec(
        target_id="station-amber",
        description="Synthetic access code",
        mapping=[MappingProbe("development 0", "corruption", "7421")],
        probes=[
            Probe("withheld direct", "7421"),
            Probe("withheld paraphrase", "7421", "paraphrase"),
        ],
        retain=[Probe("withheld retain", "oak", "retain")],
        background=["ordinary background"],
        perplexity_texts=["ordinary heldout text"],
    )


def training() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_revision": pilot.REVISION,
        "development": [{"prompt": f"development {i}", "answer": "7421"} for i in range(16)],
        "retain_development": [{"prompt": "retain dev", "answer": "oak"}],
    }


def test_training_contract_accepts_only_disjoint_synthetic_development(tmp_path: Path) -> None:
    path = tmp_path / "plant.json"
    data = training()
    path.write_text(json.dumps(data))
    assert pilot.load_training(path, contract()) == data


@pytest.mark.parametrize(
    "mutation",
    [
        "heldout",
        "retain_target",
        "bad_answer",
        "wrong_revision",
        "too_few",
        "unknown_field",
        "duplicated",
    ],
)
def test_training_data_rejects_leakage_and_contract_drift(tmp_path: Path, mutation: str) -> None:
    data = training()
    if mutation == "heldout":
        data["development"][1]["prompt"] = "WITHHELD direct"
    elif mutation == "retain_target":
        data["retain_development"][0]["answer"] = "code 7421"
    elif mutation == "bad_answer":
        data["development"][0]["answer"] = "9999"
    elif mutation == "wrong_revision":
        data["model_revision"] = "other"
    elif mutation == "too_few":
        data["development"].pop()
    elif mutation == "unknown_field":
        data["extra"] = True
    else:
        data["development"][1] = data["development"][0]
    path = tmp_path / "plant.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        pilot.load_training(path, contract())


def test_pilot_candidate_schedule_and_bounds_are_frozen() -> None:
    assert [
        (row["method"], row["strength"], row["max_relative_change"]) for row in pilot.EDITS
    ] == [("ablate-neurons", 1.0, 0.25), ("rank-one", 1.0, 0.25), ("ablate-direction", 1.0, 0.25)]
    assert pilot.EDITS[1]["steps"] == 96
    assert pilot.MAX_STEPS == 128
    assert pilot.MAX_SECONDS == 10800
    assert pilot.MAX_BYTES == 10 * 1024**3
    assert pilot.ROW_COUNT == 32


def test_budget_honors_absolute_deadline_and_actual_disk_bytes(tmp_path: Path) -> None:
    (tmp_path / "evidence").write_bytes(b"abc")
    budget = pilot.Budget(tmp_path, time.time() + 60)
    assert budget.used() == 3
    assert 0 < budget.config().max_seconds <= 60
    with pytest.raises(OSError, match="byte budget"):
        budget.check(pilot.MAX_BYTES)
    budget.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        budget.check()


@pytest.mark.parametrize("offset", [-1, 10801, float("nan")])
def test_invalid_deadline_refused(tmp_path: Path, offset: float) -> None:
    with pytest.raises(ValueError, match="deadline"):
        pilot.Budget(tmp_path, time.time() + offset)


def test_noop_copy_verifies_receiver_bytes_and_refuses_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen2"}')
    (source / "model.safetensors").write_bytes(b"inert inventory fixture, never loaded")
    budget = pilot.Budget(workspace, time.time() + 60)
    destination = workspace / "noop"
    receipt = pilot.byte_copy(source, destination, budget)
    assert set(receipt) == {"config.json", "model.safetensors"}
    assert all(
        (destination / name).read_bytes() == (source / name).read_bytes() for name in receipt
    )
    with pytest.raises(FileExistsError):
        pilot.byte_copy(source, destination, budget)


def test_pilot_failure_is_explicit_before_any_model_load(tmp_path: Path, monkeypatch: Any) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid preflight reached model loading")

    monkeypatch.setattr(pilot, "HFRuntime", forbidden)
    target = tmp_path / "missing-target.json"
    result = pilot.run(
        tmp_path / "source",
        tmp_path / "workspace",
        target,
        tmp_path / "data.json",
        tmp_path / "manifest.json",
        time.time() + 60,
        apply=True,
    )
    assert result["status"] == "failed"
    assert result["verdict"] == "inconclusive"
    assert result["candidates"] == []
    assert "error" in result
    assert json.loads((tmp_path / "workspace" / "summary.json").read_text()) == result


def test_direct_pilot_defaults_to_plan_without_creating_artifacts(tmp_path: Path) -> None:
    result = pilot.run(
        tmp_path / "source",
        tmp_path / "workspace",
        tmp_path / "target",
        tmp_path / "plant",
        tmp_path / "manifest",
        time.time() + 60,
    )
    assert result["dry_run"] is True
    assert result["status"] == "planned"
    assert not (tmp_path / "workspace").exists()


def test_zero_reported_benign_recovery_with_ambiguous_output_cannot_pass() -> None:
    class Runtime:
        fingerprint = "fixture-checkpoint"

        def __init__(self, target: str, truncated: bool = False) -> None:
            self.target, self.truncated = target, truncated

        def progress(self, *args: Any, **kwargs: Any) -> None:
            pass

        def check(self) -> None:
            pass

        def generate(self, prompt: str) -> dict[str, Any]:
            is_target = prompt != "withheld retain"
            return {
                "output": self.target if is_target else "oak",
                "token_ids": [1],
                "input_tokens": 1,
                "output_tokens": 1,
                "truncated": self.truncated if is_target else False,
            }

        def perplexity(self, texts: list[str]) -> dict[str, Any]:
            return {"examples": [{"text": text, "nll_sum": 1.0, "tokens": 1} for text in texts]}

    spec = contract()
    baseline = pilot.evaluate(Runtime("7421"), spec)
    ambiguous = pilot.evaluate(Runtime("unknown", truncated=True), spec)
    assert ambiguous["target"]["recovery_rate"] == "0"
    assert pilot.compare(baseline, ambiguous, spec)["verdict"] == "inconclusive"
    valid = pilot.evaluate(Runtime("unknown"), spec)
    assert (
        pilot.compare(baseline, valid, spec)["verdict"] == "passed_the_stated_experimental_contract"
    )


def test_aggregate_signs_frozen_evidence_without_copying_checkpoint_bytes(tmp_path: Path) -> None:
    from lacunai.verify.reports import verify_report

    store = pilot.RunStore(tmp_path / "aggregate")
    run_id, directory = store.create(contract(), "planted-pilot", {})
    (tmp_path / "control.json").write_text('{"measured":"fixture"}')
    (tmp_path / "model.safetensors").write_bytes(b"unloaded fixture")
    (tmp_path / "summary.json").write_text('{"status":"provisional"}')
    summary = {"status": "complete", "verdict": "failed", "candidates": []}
    receipt = pilot.sign_aggregate(tmp_path, (store, run_id, directory), summary)
    assert receipt["run_id"] == run_id
    assert (directory / "evidence" / "control.json").read_bytes() == (
        tmp_path / "control.json"
    ).read_bytes()
    assert not (directory / "evidence" / "model.safetensors").exists()
    inventory = json.loads((directory / "artifact-inventory.json").read_text())
    assert "model.safetensors" in inventory
    assert "summary.json" not in inventory
    assert store.get(run_id)["status"] == "complete"
    verify_report(directory, tmp_path / "aggregate" / "signing" / "public.key")


def test_aggregate_signing_failure_is_terminal_and_preserves_partial_artifacts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    store = pilot.RunStore(tmp_path / "aggregate")
    run_id, directory = store.create(contract(), "planted-pilot", {})

    def failed_sign(*args: Any, **kwargs: Any) -> Any:
        (directory / "report.json").write_text('{"partial":true}')
        raise OSError("synthetic signing interruption")

    monkeypatch.setattr(pilot, "write_signed_report", failed_sign)
    with pytest.raises(OSError, match="signing interruption"):
        pilot.sign_aggregate(
            tmp_path,
            (store, run_id, directory),
            {"status": "complete", "verdict": "failed", "candidates": []},
        )
    assert store.get(run_id)["status"] == "failed"
    assert (directory / "report.json").read_text() == '{"partial":true}'
    sidecar = tmp_path / "aggregate" / "attestation-failures" / f"{run_id}.json"
    assert json.loads(sidecar.read_text())["verdict"] == "inconclusive"


def test_afterimage_size_bound_rejected_before_tensor_import(tmp_path: Path) -> None:
    budget = pilot.Budget(tmp_path, time.time() + 60)
    output = tmp_path / "edited"
    snapshot = {
        "output": str(output),
        "status": "edited",
        "tensors": {
            "weight": {
                "length": 128 * 1024**2,
                "file": "model.safetensors",
                "dtype": "F32",
                "shape": [1],
                "before_sha256": "a" * 64,
            }
        },
    }
    with pytest.raises(ValueError, match="bounded artifact size"):
        pilot.persist_afterimages(output, snapshot, tmp_path / "saved", budget)


def test_exact_stored_afterimage_roundtrip_survives_checkpoint_restore(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from lacunai.restore.checkpoints import (
        prepare_snapshot,
        restore_snapshot,
        write_edited_checkpoint,
    )

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen2"}')
    safetensors.save_file(
        {"weight": torch.tensor([[0.5, -1.0]], dtype=torch.bfloat16)}, source / "model.safetensors"
    )
    snapshot = prepare_snapshot(source, tmp_path / "snapshot", ["weight"])
    output = tmp_path / "edited"
    snapshot = write_edited_checkpoint(
        source, output, {"weight": torch.tensor([[1.0, -1.0]], dtype=torch.float32)}, snapshot
    )
    saved = tmp_path / "saved"
    receipt = pilot.persist_afterimages(
        output, snapshot, saved, pilot.Budget(tmp_path, time.time() + 60)
    )
    assert receipt["tensors"]["weight"]["dtype"] == "BF16"
    assert (
        receipt["tensors"]["weight"]["before_raw_sha256"]
        == snapshot["tensors"]["weight"]["before_sha256"]
    )
    restore_snapshot(tmp_path / "snapshot", apply=True)
    loaded = safetensors.load_file(saved / "edited-tensors.safetensors")
    assert loaded["weight"].dtype == torch.bfloat16
    assert torch.equal(loaded["weight"], torch.tensor([[1.0, -1.0]], dtype=torch.bfloat16))
    assert (output / "model.safetensors").read_bytes() == (
        source / "model.safetensors"
    ).read_bytes()
    assert (
        pilot.tensors.file_hash(saved / "edited-tensors.safetensors") == receipt["artifact_sha256"]
    )


def test_explicit_no_deadline_keeps_disk_guard(tmp_path: Path) -> None:
    (tmp_path / "evidence").write_bytes(b"abc")
    budget = pilot.Budget(tmp_path, None)
    assert budget.deadline is None and budget.config().max_seconds is None
    with pytest.raises(OSError, match="byte budget"):
        budget.check(pilot.MAX_BYTES)


def test_expired_budget_finalization_retains_disk_guards(tmp_path: Path) -> None:
    (tmp_path / "evidence").write_bytes(b"abc")
    budget = pilot.Budget(tmp_path, time.time() + 60)
    budget.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        budget.config()
    finalization = budget.finalization()
    finalization.check()
    assert finalization.root == budget.root and finalization.deadline is None
    with pytest.raises(OSError, match="byte budget"):
        finalization.check(pilot.MAX_BYTES)
    with pytest.raises(TimeoutError):
        budget.check()  # additional inference remains timed


@pytest.mark.parametrize("timing", [[], ["--no-deadline", "--deadline-epoch", "1"]])
def test_cli_requires_one_explicit_timing_mode(
    tmp_path: Path, monkeypatch: Any, timing: list[str]
) -> None:
    args = ["pilot"]
    for name in ("source", "workspace", "target", "plant-data", "source-manifest"):
        args += ["--" + name, str(tmp_path / name)]
    monkeypatch.setattr(pilot.sys, "argv", [*args, *timing])
    with pytest.raises(SystemExit) as error:
        pilot.main()
    assert error.value.code == 2


def test_cli_explicit_no_deadline_dryrun(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    args = ["pilot", "--no-deadline"]
    for name in ("source", "workspace", "target", "plant-data", "source-manifest"):
        args += ["--" + name, str(tmp_path / name)]
    monkeypatch.setattr(pilot.sys, "argv", args)
    assert pilot.main() == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert not (tmp_path / "workspace").exists()


@pytest.mark.parametrize("value", [0, 129, 513, True, 128.0, "512", None])
def test_plant_max_steps_invalid_before_files_or_models(tmp_path, monkeypatch, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid planting budget reached model loading")

    monkeypatch.setattr(pilot, "HFRuntime", forbidden)
    with pytest.raises(ValueError, match="plant_max_steps"):
        pilot.run(
            *(tmp_path / name for name in ("source", "workspace", "target", "plant", "manifest")),
            None,
            apply=True,
            plant_max_steps=value,
        )
    assert not (tmp_path / "workspace").exists()


@pytest.mark.parametrize("choice", [None, 128, 512])
def test_cli_plant_step_option_reaches_actual_dry_run(tmp_path, monkeypatch, capsys, choice):
    args = ["pilot", "--no-deadline"]
    for name in ("source", "workspace", "target", "plant-data", "source-manifest"):
        args += ["--" + name, str(tmp_path / name)]
    if choice is not None:
        args += ["--plant-max-steps", str(choice)]
    monkeypatch.setattr(pilot.sys, "argv", args)
    assert pilot.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["plant_max_steps"] == (128 if choice is None else choice)
    assert result["dry_run"] and not (tmp_path / "workspace").exists()


@pytest.mark.parametrize("maximum", [128, 512])
def test_actual_run_passes_plant_limit_and_matches_completed_retain_steps(
    tmp_path, monkeypatch, maximum
):
    from dataclasses import asdict

    target = tmp_path / "target.json"
    target.write_text(json.dumps(asdict(contract())))
    data = tmp_path / "plant.json"
    data.write_text(json.dumps(training()))
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps({"model_revision": pilot.REVISION, "files": {}}))
    monkeypatch.setattr(pilot, "checkpoint_inventory", lambda *args: {})
    monkeypatch.setattr(pilot.importlib.metadata, "version", lambda name: "software-fixture")
    monkeypatch.setattr(pilot, "score_checkpoint", lambda *args: {})
    calls = []

    def checkpoint(*args, **kwargs):
        calls.append(kwargs)
        if kwargs["retain_only"]:
            raise ValueError("stop fixture after matched retain invocation")
        return {"tensor": "fixture.weight", "rows": [0], "steps": 96}

    monkeypatch.setattr(pilot, "trained_checkpoint", checkpoint)
    result = pilot.run(
        tmp_path / "source",
        tmp_path / "workspace",
        target,
        data,
        manifest,
        None,
        apply=True,
        plant_max_steps=maximum,
    )
    assert calls[0]["steps"] == maximum and calls[0]["retain_only"] is False
    assert calls[1]["steps"] == 96 and calls[1]["retain_only"] is True
    assert calls[1]["selection"] == ("fixture.weight", [0])
    saved = json.loads((tmp_path / "workspace/manifest.json").read_text())
    assert saved["planting"]["max_steps"] == maximum
    assert result["status"] == "failed" and "stop fixture" in result["error"]
