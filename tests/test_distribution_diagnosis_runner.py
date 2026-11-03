"""Diagnosis runner software contracts; fixtures do not establish model efficacy."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ENTRY = Path(__file__).parents[1] / "scripts/chinese_distribution_diagnosis.py"
spec = importlib.util.spec_from_file_location("distribution_runner", ENTRY)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def corpus(tmp_path):
    value = {
        "schema_version": 1,
        "corpora": {k: ["synthetic text"] * 64 for k in ("en", "zh", "es", "ja")},
        "source_dataset_sha256": "b" * 64,
        "source_ids": ["train:synthetic"],
        "provenance": "synthetic development",
        "tokenization": "raw",
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(value))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def argv(tmp_path):
    path, pin = corpus(tmp_path)
    return [
        str(ENTRY),
        "--root",
        str(tmp_path / "runs"),
        "--source",
        str(tmp_path / "source"),
        "--source-fingerprint",
        "a" * 64,
        "--corpus",
        str(path),
        "--corpus-sha256",
        pin,
        "--name",
        "diagnosis",
    ]


def test_default_dryrun_has_no_runtime_or_ml_import(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", argv(tmp_path))
    monkeypatch.setattr(runner, "HFRuntime", lambda *_: pytest.fail("dryrun loaded model"))
    before = set(sys.modules)
    assert runner.main() == 0
    assert not (tmp_path / "runs").exists()
    assert not ({"torch", "transformers", "safetensors.torch"} & (set(sys.modules) - before))
    assert '"state": "plan_only"' in capsys.readouterr().err


@pytest.mark.parametrize("change", ["hash", "extra_split", "count", "language", "schema_bool"])
def test_corpus_contract_guards(tmp_path, change):
    path, pin = corpus(tmp_path)
    value = json.loads(path.read_text())
    if change == "hash":
        pin = "0" * 64
    else:
        if change == "extra_split":
            value["heldout_primary"] = []
        if change == "count":
            value["corpora"]["zh"].pop()
        if change == "language":
            value["corpora"]["xx"] = value["corpora"].pop("ja")
        if change == "schema_bool":
            value["schema_version"] = True
        path.write_text(json.dumps(value))
        pin = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        runner.pinned_corpus(path, pin)


def test_existing_run_never_overwritten(tmp_path, monkeypatch):
    args = argv(tmp_path)
    directory = tmp_path / "runs/diagnosis"
    directory.mkdir(parents=True)
    original = directory / "status.json"
    original.write_bytes(b"preserve")
    monkeypatch.setattr(sys, "argv", [*args, "--apply"])
    with pytest.raises(FileExistsError):
        runner.main()
    assert original.read_bytes() == b"preserve"


def test_load_failure_persists_terminal_status_and_flushes_progress(tmp_path, monkeypatch, capsys):
    class Watch:
        def __init__(self, root, contract, deadline, name):
            assert deadline is None
            assert contract["resources"] == runner.iteration.LIMITS

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(sys, "argv", [*argv(tmp_path), "--apply"])
    monkeypatch.setattr(runner, "host_sample", lambda _: {})
    monkeypatch.setattr(runner, "Watchdog", Watch)
    monkeypatch.setattr(runner, "checkpoint_inventory", lambda *_: {"model.safetensors": "b" * 64})
    monkeypatch.setattr(runner, "digest", lambda _: "a" * 64)

    def fail(source, config):
        assert config.max_seconds is None and config.seed == 42 and config.torch_threads == 2
        assert '"state": "active"' in capsys.readouterr().err
        raise RuntimeError("fixture load failure")

    monkeypatch.setattr(runner, "HFRuntime", fail)
    with pytest.raises(RuntimeError, match="fixture load failure"):
        runner.main()
    status = json.loads((tmp_path / "runs/diagnosis/status.json").read_text())
    assert status["state"] == "failed"
    assert '"state": "failed"' in capsys.readouterr().err


def test_real_tensor_persistence_roundtrip_and_no_overwrite(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")
    from lacunai.map import language_distribution

    watch = SimpleNamespace(
        check=lambda: None,
        root=tmp_path,
        artifact_budget_root=None,
        limits={"disk_max_bytes": 10 * 1024**3},
    )
    payload = {"counts": torch.tensor([[0, 1, 2]], dtype=torch.int64)}
    metadata = {
        "schema_version": language_distribution.VERSION,
        "payload": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": language_distribution.payload_hash(value),
            }
            for key, value in payload.items()
        },
    }
    tampered = json.loads(json.dumps(metadata))
    tampered["payload"]["counts"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="tensor differs"):
        runner.persist(tmp_path, tampered, payload, watch)
    assert list(tmp_path.iterdir()) == []
    saved = runner.persist(tmp_path, metadata, payload, watch)
    assert json.loads((tmp_path / "diagnosis.json").read_text()) == metadata
    from safetensors.torch import load_file

    restored = load_file(str(tmp_path / "histograms.safetensors"))
    language_distribution.validate_payload(metadata, restored)
    assert torch.equal(restored["counts"], payload["counts"])
    assert set(saved) == {"diagnosis.json", "histograms.safetensors"}
    for name, pin in saved.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == pin
    with pytest.raises(ValueError, match="already exists"):
        runner.persist(tmp_path, {}, payload, watch)


def test_oversized_payload_refused_before_serializer_import(tmp_path, monkeypatch):
    tensor = SimpleNamespace(numel=lambda: 1024, element_size=lambda: 8, shape=(1024,))
    watch = SimpleNamespace(
        root=tmp_path,
        artifact_budget_root=None,
        limits={"disk_max_bytes": 1024},
        check=lambda: None,
    )
    monkeypatch.setattr(
        runner,
        "import_module",
        lambda *_: pytest.fail("serializer imported before budget admission"),
    )
    with pytest.raises(ValueError, match="budget"):
        runner.persist(tmp_path, {}, {"counts": tensor}, watch)
    assert list(tmp_path.iterdir()) == []
