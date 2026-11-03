"""Runner authorization/serialization fixtures, not language efficacy evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "subspace_candidate", Path(__file__).parents[1] / "scripts/chinese_subspace_candidate.py"
)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def selection():
    return {
        "schema_version": 1,
        "method": "retain-nullspace",
        "map_sha256": "a" * 64,
        "layers": [16, 23],
        "k": 4,
        "strength": 0.5,
    }


def test_contract_fixes_allowlist_bounds_and_revision():
    contract = runner.edit_contract(selection())
    assert contract["edits"] == {
        "allowed_tensors": [
            "model.layers.16.mlp.down_proj.weight",
            "model.layers.23.mlp.down_proj.weight",
        ],
        "expected_shape": [896, 4864],
        "max_tensor_relative_frobenius": "0.10",
        "max_model_relative_frobenius": "0.02",
    }
    assert contract["target"]["revision"] == runner.baseline.MODEL_REVISION


@pytest.mark.parametrize(
    "key,value",
    [
        ("layers", []),
        ("layers", [1, 1]),
        ("layers", list(range(9))),
        ("layers", [24]),
        ("layers", [True]),
        ("k", True),
        ("k", 2),
        ("strength", True),
        ("strength", 0.75),
        ("strength", float("nan")),
        ("method", "rank-one"),
        ("map_sha256", "z" * 64),
        ("schema_version", True),
        ("extra", "generated_code"),
    ],
)
def test_unauthorized_selection_refused(key, value):
    with pytest.raises(ValueError):
        runner.edit_contract({**selection(), key: value})


@pytest.mark.parametrize("phase", ["diagnose", "screen", "persist"])
def test_cli_default_noop_never_loads_model(tmp_path, monkeypatch, capsys, phase):
    monkeypatch.setattr(runner, "HFRuntime", lambda *_: pytest.fail("model load on dry run"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            phase,
            "--root",
            str(tmp_path / "absent"),
            "--source",
            str(tmp_path / "missing-model"),
            "--source-fingerprint",
            "a" * 64,
            "--name",
            "dry",
        ],
    )
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out)["state"] == "plan_only"
    assert not (tmp_path / "absent").exists()


def test_apply_bad_pin_records_failure_before_model_load(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus.json"
    corpus.write_text("{}")
    monkeypatch.setattr(runner, "HFRuntime", lambda *_: pytest.fail("model loaded before pins"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "diagnose",
            "--root",
            str(tmp_path / "runs"),
            "--source",
            str(tmp_path / "missing-model"),
            "--source-fingerprint",
            "a" * 64,
            "--name",
            "bad",
            "--corpus",
            str(corpus),
            "--corpus-sha256",
            "b" * 64,
            "--apply",
        ],
    )
    with pytest.raises(ValueError, match="hash differs"):
        runner.main()
    status = json.loads((tmp_path / "runs/bad/status.json").read_text())
    assert status["state"] == "failed"
    assert "failed" in capsys.readouterr().err


@pytest.mark.parametrize("phase", ["persist", "diagnose"])
def test_non_evaluation_phases_refuse_data_before_read(phase):
    args = SimpleNamespace(
        phase=phase, dataset=Path("unread"), dataset_sha256=None, before=None, before_sha256=None
    )
    with pytest.raises(ValueError, match="must not receive"):
        runner.evaluation_inputs(args)


def test_heldout_refused_before_baseline_read(monkeypatch):
    monkeypatch.setattr(runner.baseline, "pinned_json", lambda *_: {"heldout_primary": [1]})
    args = SimpleNamespace(
        phase="screen",
        dataset=Path("data"),
        dataset_sha256="x",
        before=Path("before"),
        before_sha256="y",
    )
    with pytest.raises(ValueError, match="development-only"):
        runner.evaluation_inputs(args)


def test_replacement_allowlist_refused_without_torch():
    with pytest.raises(ValueError, match="allowlist"):
        runner.apply_replacements(None, {}, runner.edit_contract(selection()))


def test_actual_tensor_apply_and_restore():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2, bias=False)
    original = model.weight.detach().clone()
    contract = {
        "edits": {
            "allowed_tensors": ["weight"],
            "expected_shape": [2, 2],
            "max_tensor_relative_frobenius": "0.10",
            "max_model_relative_frobenius": "0.02",
        }
    }
    reference = runner.tensors.capture_reference(model, contract)
    runtime = SimpleNamespace(model=model)
    replacement = original * 0.99
    runner.apply_replacements(runtime, {"weight": replacement}, contract)
    assert torch.equal(model.weight, replacement)
    runner.tensors.restore_reference(model, reference, contract)
    assert torch.equal(model.weight, original)
    with pytest.raises(ValueError, match="non-finite"):
        runner.apply_replacements(
            runtime, {"weight": torch.full_like(original, float("nan"))}, contract
        )
    assert torch.equal(model.weight, original)


def test_map_binding_rejects_other_source_or_selection():
    from lacunai.map.localize import record_digest

    diagnosis = {"checkpoint_fingerprint": "source", "diagnosis_sha256": "internal"}
    spec = {**selection(), "map_sha256": record_digest(diagnosis)}
    runner.verify_diagnosis(diagnosis, spec, "source")
    with pytest.raises(ValueError, match="source"):
        runner.verify_diagnosis(diagnosis, spec, "another-source")
    with pytest.raises(ValueError, match="selection"):
        runner.verify_diagnosis(diagnosis, {**spec, "map_sha256": "b" * 64}, "source")


def test_operator_failure_restores_reference_and_ends_status(tmp_path, monkeypatch, capsys):
    import hashlib
    from contextlib import nullcontext

    from lacunai.core.spec import digest
    from lacunai.map.localize import record_digest

    events = []
    source = digest({})
    diagnosis = {"checkpoint_fingerprint": source}
    chosen = {**selection(), "map_sha256": record_digest(diagnosis)}
    argv = [
        "runner",
        "persist",
        "--root",
        str(tmp_path / "runs"),
        "--source",
        str(tmp_path / "source"),
        "--source-fingerprint",
        source,
        "--name",
        "failure",
        "--apply",
    ]
    for flag, value in (("diagnosis", diagnosis), ("edit-spec", chosen)):
        path = tmp_path / (flag + ".json")
        path.write_text(json.dumps(value))
        argv += [
            "--" + flag,
            str(path),
            "--" + flag + "-sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
        ]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(runner, "host_sample", lambda *_: {})
    monkeypatch.setattr(
        runner, "Watchdog", lambda *args: nullcontext(SimpleNamespace(check=lambda: None))
    )
    monkeypatch.setattr(runner, "checkpoint_inventory", lambda *_: {})
    names = runner.edit_contract(chosen)["edits"]["allowed_tensors"]
    runtime = SimpleNamespace(
        inventory={}, contract={}, model=None, storage_dtypes={n: "BF16" for n in names}
    )
    monkeypatch.setattr(runner, "HFRuntime", lambda *_: runtime)
    monkeypatch.setattr(runner, "prepare_snapshot", lambda *_: events.append("snapshot") or {})
    monkeypatch.setattr(runner.iteration, "bind_snapshot", lambda *_: None)
    monkeypatch.setattr(
        runner.tensors,
        "capture_reference",
        lambda *_: events.append("reference") or {"fingerprint": "canonical"},
    )
    monkeypatch.setattr(runner.tensors, "restore_reference", lambda *_: events.append("restore"))

    def propose(*args):
        events.append("operator")
        captured = capsys.readouterr()
        assert "subspace_proposal_and_apply" in captured.err
        raise RuntimeError("numerical proposal rejected")

    real_import = runner.import_module
    monkeypatch.setattr(
        runner,
        "import_module",
        lambda name: (
            SimpleNamespace(propose=propose)
            if name == "lacunai.erase.retain_nullspace"
            else real_import(name)
        ),
    )
    with pytest.raises(RuntimeError, match="proposal rejected"):
        runner.main()
    assert events == ["snapshot", "reference", "operator", "restore"]
    directory = tmp_path / "runs/failure"
    assert json.loads((directory / "restore.json").read_text())["exact_original_tensors_restored"]
    assert json.loads((directory / "status.json").read_text())["state"] == "failed"
    assert "failed" in capsys.readouterr().err


def test_storage_evidence_labels_idempotence_even_on_validation_failure(tmp_path, monkeypatch):
    path = tmp_path / "storage.json"

    def measured(*args):
        args[-1].write_text(json.dumps({"retained_delta_energy": "1.0"}))
        raise runner.tensors.TensorError("bound failed")

    monkeypatch.setattr(runner.gradient, "storage_measurement", measured)
    with pytest.raises(runner.tensors.TensorError, match="bound failed"):
        runner.storage_measurement(None, {}, {}, path)
    artifact = json.loads(path.read_text())
    assert artifact["retained_delta_energy"] == "1.0"
    assert artifact["measurement_scope"] == (
        "storage-roundtrip idempotence; replacements already rounded by operator"
    )
