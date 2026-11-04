"""Baseline retain qualification uses actual scorer evidence before any checkpoint write."""

from dataclasses import replace

import pytest

from lacunai.core import service as module
from lacunai.core.service import Service
from lacunai.core.spec import Probe
from lacunai.verify.evaluation import classify_retain, evaluate
from tests import test_service_lifecycle as lifecycle_fixtures
from tests.test_verification_records import Runtime, spec

lifecycle = lifecycle_fixtures.lifecycle


def test_low_retain_refused_before_snapshot_or_weight_write(lifecycle, monkeypatch):
    service, model, target = lifecycle
    original = (model / "model.safetensors").read_bytes()

    def low_retain(runtime, target_spec):
        result = evaluate(Runtime(["drazel", "drazel", "wrong", "pine"]), target_spec)
        result["checkpoint_fingerprint"] = runtime.fingerprint
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint write boundary reached before baseline qualification")

    monkeypatch.setattr(module, "evaluate", low_retain)
    monkeypatch.setattr(module, "prepare_snapshot", forbidden)
    monkeypatch.setattr(module, "write_edited_checkpoint", forbidden)
    with pytest.raises(ValueError, match="baseline retain answer accuracy"):
        service.erase(model, target, "ablate-neurons", apply=True)
    assert (model / "model.safetensors").read_bytes() == original
    assert not list(service.workspace.glob("checkpoints/*/model.safetensors"))


def test_sufficient_retain_allows_actual_checkpoint_transaction(lifecycle):
    service, model, target = lifecycle
    result = service.erase(model, target, "ablate-neurons", apply=True)
    assert result["status"] == "complete"
    from pathlib import Path

    assert (Path(result["output"]) / "model.safetensors").read_bytes() != (
        model / "model.safetensors"
    ).read_bytes()


def test_exact_existing_retain_threshold_inclusive():
    target = replace(spec(), retain=[Probe(f"Retain{i}?", f"value{i}", "retain") for i in range(4)])
    result = evaluate(Runtime(["drazel", "drazel", "value0", "value1", "value2", "wrong"]), target)
    assert result["retain"]["recovery_rate"] == "0.75"
    Service._qualify(result, target)


@pytest.mark.parametrize(
    "case", ["missing_suite", "missing_row", "missing_output", "ambiguous", "forged_summary"]
)
def test_missing_ambiguous_or_low_raw_retain_cannot_qualify(case):
    target = spec()
    answers = ["drazel", "drazel", "oak", "pine"]
    if case in ("missing_output", "ambiguous", "forged_summary"):
        answers[2] = {"missing_output": "", "ambiguous": "oak", "forged_summary": "wrong"}[case]
    result = evaluate(Runtime(answers), target)
    if case == "missing_suite":
        del result["retain"]
    elif case == "missing_row":
        result["retain"]["rows"].pop()
    elif case == "ambiguous":
        row = result["retain"]["rows"][0]
        row["generation"]["truncated"] = True
        row.update(classify_retain("oak", "oak", True))
    elif case == "forged_summary":
        result["retain"].update(correct=2, recovered=2, recovery_rate="1")
    with pytest.raises(ValueError):
        Service._qualify(result, target)
