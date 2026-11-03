"""Real signed-artifact and delta helper seams; no language efficacy claims."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lacunai.core.spec import digest
from lacunai.verify.reports import write_signed_report

spec = importlib.util.spec_from_file_location(
    "replay_runner", Path(__file__).parents[1] / "scripts/chinese_replay_candidate.py"
)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def signed_parent(tmp_path):
    parent = tmp_path / "rmu-001-retry01"
    parent.mkdir()
    (parent / "after-storage").mkdir()
    edit = dict(runner.rmu.SPEC)
    values = {
        "manifest.json": {
            "method": "rmu_candidate",
            "objective_spec": edit,
            "checkpoint_fingerprint": digest({}),
            "source_inventory": {},
        },
        "edit-spec.json": edit,
        "status.json": {"method": "rmu_candidate", "state": "complete", "verdict": "failed"},
        "after-storage/manifest.json": {"fixture": "not a model"},
    }
    for name, value in values.items():
        (parent / name).write_text(json.dumps(value))
    (parent / "after-storage/changed.safetensors").write_bytes(b"fixture only")
    keys = tmp_path / "keys"
    write_signed_report(
        parent,
        {
            "run_id": "fixture-rmu",
            "target_id": "synthetic",
            "status": "complete",
            "verdict": "failed",
        },
        keys,
    )
    pins = {
        key: hashlib.sha256((parent / name).read_bytes()).hexdigest()
        for key, name in runner.PARENT_FILES.items()
    }
    key = keys / "public.key"
    key_hash = hashlib.sha256(key.read_bytes()).hexdigest()
    return parent, pins, key, key_hash


def test_real_signature_and_pins_parent_is_readonly(tmp_path):
    parent, pins, key, key_hash = signed_parent(tmp_path)
    before = {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    result = runner.verify_parent(parent, pins, key, key_hash, digest({}))
    assert result["verification"]["verified"] and result["source_inventory"] == {}
    runner.parent_unchanged(parent, result)
    assert {
        str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob("*") if p.is_file()
    } == before
    with pytest.raises(ValueError, match="trusted public key"):
        runner.verify_parent(parent, pins, key, "0" * 64, digest({}))
    with pytest.raises(ValueError, match="pin/signature"):
        runner.verify_parent(
            parent, {**pins, "delta_bytes_sha256": "0" * 64}, key, key_hash, digest({})
        )
    with pytest.raises(ValueError, match="source identity"):
        runner.verify_parent(parent, pins, key, key_hash, "0" * 64)


def test_signed_parent_tampering_rejected(tmp_path):
    parent, pins, key, key_hash = signed_parent(tmp_path)
    (parent / "after-storage/changed.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="signed artifact hash"):
        runner.verify_parent(parent, pins, key, key_hash, digest({}))


def test_load_delta_uses_outer_contract_and_hashes_all_allowed_tensors(tmp_path, monkeypatch):
    names = runner.rmu.NAMES
    contract = {
        "edits": {
            "allowed_tensors": names,
            "max_tensor_relative_frobenius": "0.02",
            "max_model_relative_frobenius": "0.0025",
        }
    }
    model = SimpleNamespace(state_dict=lambda: {name: name for name in names})

    def load(model_arg, path, reference, passed):
        assert model_arg is model and path == tmp_path / "after-storage" and passed is contract
        return {"delta_sha256": "delta", "diff": {"tensors": {n: {"sha256": n} for n in names}}}

    monkeypatch.setattr(runner.tensors, "load_delta", load)
    monkeypatch.setattr(runner.tensors, "tensor_hash", lambda x: x)
    result = runner.load_stored(
        SimpleNamespace(model=model), tmp_path, {"fingerprint": "source"}, contract
    )
    assert result["loaded_tensor_hashes"] == {name: name for name in names}
    assert result["no_training_performed"] is True


def test_actual_delta_roundtrip_and_exact_restore(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")
    model = torch.nn.Linear(2, 2, bias=False)
    monkeypatch.setattr(runner.rmu, "NAMES", ["weight"])
    contract = {
        "target": {"revision": "fixture"},
        "edits": {
            "allowed_tensors": ["weight"],
            "expected_shape": [2, 2],
            "max_tensor_relative_frobenius": "0.02",
            "max_model_relative_frobenius": "0.0025",
        },
    }
    reference = runner.tensors.capture_reference(model, contract)
    with torch.no_grad():
        model.weight.mul_(0.999)
    expected = runner.tensors.tensor_hash(model.weight)
    runner.tensors.save_delta(model, reference, tmp_path / "after-storage", contract)
    runner.tensors.restore_reference(model, reference, contract)
    try:
        result = runner.load_stored(SimpleNamespace(model=model), tmp_path, reference, contract)
        assert result["loaded_tensor_hashes"]["weight"] == expected
    finally:
        runner.tensors.restore_reference(model, reference, contract)
    assert runner.tensors.tensor_hash(model.weight) == reference["hashes"]["weight"]


def test_cli_dry_run_loads_nothing(tmp_path, monkeypatch, capsys):
    args = ["runner", "--name", "replay", "--source-fingerprint", "a" * 64]
    for flag in (
        "root",
        "source",
        "parent-phase",
        "trusted-public-key",
        "training-contract",
        "training-dataset",
        "dataset",
        "before",
    ):
        args += ["--" + flag, str(tmp_path / flag)]
    for flag in (
        "parent-report",
        "parent-manifest",
        "edit-spec",
        "delta-manifest",
        "delta-bytes",
        "trusted-public-key",
        "training-contract",
        "training-dataset",
        "dataset",
        "before",
    ):
        args += ["--" + flag + "-sha256", "b" * 64]
    monkeypatch.setattr("sys.argv", args)
    monkeypatch.setattr(runner, "HFRuntime", lambda *_: pytest.fail("model loaded"))
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out)["training_performed"] is False
    assert not (tmp_path / "root").exists()


def test_parent_after_readback_detects_changes(tmp_path):
    parent, pins, key, key_hash = signed_parent(tmp_path)
    evidence = runner.verify_parent(parent, pins, key, key_hash, digest({}))
    (parent / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="parent artifact changed"):
        runner.parent_unchanged(parent, evidence)


def test_replay_load_rejects_partial_afterimage_inventory(tmp_path, monkeypatch):
    names = runner.rmu.NAMES[:2]
    monkeypatch.setattr(
        runner.tensors,
        "load_delta",
        lambda *args: {
            "delta_sha256": "fixture",
            "diff": {"tensors": {n: {"sha256": n} for n in names}},
        },
    )
    with pytest.raises(ValueError, match="all three allowed tensors"):
        runner.load_stored(SimpleNamespace(model=None), tmp_path, {}, {})
