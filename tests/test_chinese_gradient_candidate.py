"""Root-specified candidate contracts; numerical fixtures are not efficacy evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lacunai.experimental import data, tensors

spec = importlib.util.spec_from_file_location(
    "gradient_candidate", Path(__file__).parents[1] / "scripts/chinese_gradient_candidate.py"
)
assert spec and spec.loader
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def test_saved_after_image_readback_detects_different_bytes(tmp_path, monkeypatch):
    def save(model, reference, target, contract, budget_root):
        assert (
            contract["resources"]["disk_max_bytes"] == candidate.iteration.LIMITS["disk_max_bytes"]
        )
        target.mkdir()
        (target / "changed.safetensors").write_bytes(b"different")
        return {"delta_sha256": hashlib.sha256(b"expected").hexdigest()}

    monkeypatch.setattr(tensors, "save_delta", save)
    with pytest.raises(ValueError, match="failed readback"):
        candidate.save_after_image(
            SimpleNamespace(model=None), {}, {"resources": {}}, tmp_path, tmp_path
        )


def historical():
    return {
        "baseline": dict(candidate.baseline.SETTINGS),
        "runtime": {"seed": 20261102},
        "evaluation": {
            "primary_families": list(data.PRIMARY),
            "retain_families": list(data.RETAIN),
            "items_per_primary_family": 16,
            "items_per_retain_family": 16,
            "prompt_variants_per_item": 4,
        },
        "edits": {
            "allowed_tensors": list(candidate.baseline.NAMES),
            "expected_shape": [896, 4864],
            "max_tensor_relative_frobenius": "0.02",
            "max_model_relative_frobenius": "0.0025",
            "tokenizer_edits": False,
            "embedding_edits": False,
            "lm_head_edits": False,
            "generated_code": False,
        },
    }


@pytest.mark.parametrize("mode", ["graddiff", "retain_finetune"])
def test_exact_root_spec_only_changes_lr_and_inner_bounds(mode):
    original = historical()
    saved = copy.deepcopy(original)
    dataset = data.build_dataset(original)
    spec = {**candidate.SPEC, "mode": mode}
    inner, outer = candidate.candidate_contract(spec, original, dataset)
    assert original == saved == outer
    expected = copy.deepcopy(saved)
    expected["baseline"]["learning_rate"] = "0.1"
    expected["edits"]["max_tensor_relative_frobenius"] = "0.0195"
    expected["edits"]["max_model_relative_frobenius"] = "0.0024"
    assert inner == expected
    assert inner["baseline"]["steps"] == outer["baseline"]["steps"] == 60


@pytest.mark.parametrize(
    "key,value",
    [("learning_rate", "0.2"), ("steps", 61), ("schema_version", True), ("extra", "code")],
)
def test_spec_drift_refused(key, value):
    original = historical()
    dataset = data.build_dataset(original)
    spec = {**candidate.SPEC, key: value}
    with pytest.raises(ValueError, match="structured contract"):
        candidate.candidate_contract(spec, original, dataset)


def test_historical_contract_validated_before_spec(monkeypatch):
    def refuse(*args):
        raise ValueError("historical first")

    monkeypatch.setattr(candidate.baseline, "validate_training", refuse)
    with pytest.raises(ValueError, match="historical first"):
        candidate.candidate_contract({}, {}, {})


def test_screen_order_and_no_false_early_pass(tmp_path, monkeypatch):
    calls = []

    def recovery(*args):
        calls.append("recovery")
        return {"continue": False, "verdict": "failed"}

    monkeypatch.setattr(candidate.iteration, "recovery_preflight", recovery)
    monkeypatch.setattr(
        candidate.iteration,
        "collateral_preflight",
        lambda *args: pytest.fail("collateral after rejection"),
    )
    monkeypatch.setattr(
        candidate.iteration, "score", lambda *args: pytest.fail("full eval after rejection")
    )
    result = candidate.screen(None, {}, {}, tmp_path, None)
    assert calls == ["recovery"] and result["verdict"] == "failed" and "result" not in result


def test_screen_collateral_then_full_only_if_allowed(tmp_path, monkeypatch):
    calls = []

    def observe(name, value):
        def function(*args):
            calls.append(name)
            return value

        return function

    monkeypatch.setattr(
        candidate.iteration, "recovery_preflight", observe("recovery", {"continue": True})
    )
    monkeypatch.setattr(
        candidate.iteration, "collateral_preflight", observe("collateral", {"continue": True})
    )
    monkeypatch.setattr(candidate.iteration, "score", observe("score", {"fixture": True}))
    monkeypatch.setattr(
        candidate.iteration, "compare", observe("compare", {"verdict": "inconclusive"})
    )
    result = candidate.screen(None, {}, {}, tmp_path, None)
    assert calls == ["recovery", "collateral", "score", "compare"]
    assert result["verdict"] == "inconclusive"


def tiny_runtime():
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.register_parameter("w", torch.nn.Parameter(torch.tensor([1.0, 1.0])))
    model.register_parameter("other", torch.nn.Parameter(torch.tensor([100.0])))
    contract = {
        "edits": {
            "allowed_tensors": ["w"],
            "expected_shape": [2],
            "max_tensor_relative_frobenius": ".02",
            "max_model_relative_frobenius": ".0025",
        }
    }
    reference = tensors.capture_reference(model, contract)
    return torch, SimpleNamespace(model=model, storage_dtypes={"w": "BF16"}), contract, reference


def test_actual_bf16_roundtrip_counts_and_energy():
    torch, runtime, contract, reference = tiny_runtime()
    with torch.no_grad():
        runtime.model.w.add_(torch.tensor([0.001, 0.01]))
    result = candidate.storage_measurement(runtime, reference, contract)
    assert result["before_storage_roundtrip"]["changed_element_count"] == 2
    assert result["after_storage_roundtrip"]["changed_element_count"] == 1
    assert result["after_storage_roundtrip"]["delta_norm_squared"] == (0.0078125**2)
    assert 0 < float(result["retained_delta_energy"]) < 1
    tensors.restore_reference(runtime.model, reference, contract)
    assert tensors.tensor_hash(runtime.model.w) == reference["hashes"]["w"]


def test_actual_zero_delta_energy_is_undefined():
    _, runtime, contract, reference = tiny_runtime()
    result = candidate.storage_measurement(runtime, reference, contract)
    assert result["retained_delta_energy"] is None
    assert result["after_storage_roundtrip"]["changed_element_count"] == 0
    assert result["zero_denominator"] == "undefined"


def test_actual_stored_outer_bound_refuses():
    torch, runtime, contract, reference = tiny_runtime()
    with torch.no_grad():
        runtime.model.w.add_(0.04)
    with pytest.raises(tensors.TensorError, match="bound exceeded"):
        candidate.storage_measurement(runtime, reference, contract)


def test_default_plan_no_input_reads_or_model(monkeypatch, capsys):
    argv = ["candidate", "screen"]
    for flag in [
        "root",
        "source",
        "training-contract",
        "training-dataset",
        "edit-spec",
        "source-fingerprint",
        "training-contract-sha256",
        "training-dataset-sha256",
        "edit-spec-sha256",
        "name",
    ]:
        argv.extend(["--" + flag, "absent"])
    monkeypatch.setattr(candidate.sys, "argv", argv)
    monkeypatch.setattr(candidate, "HFRuntime", lambda *a, **k: pytest.fail("unexpected model"))
    assert candidate.main() == 0
    assert json.loads(capsys.readouterr().out)["state"] == "plan_only"
