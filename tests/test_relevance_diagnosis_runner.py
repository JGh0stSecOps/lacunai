"""Diagnosis runner software contracts; fixtures do not establish model efficacy."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ENTRY = Path(__file__).parents[1] / "scripts/chinese_relevance_diagnosis.py"
spec = importlib.util.spec_from_file_location("relevance_runner", ENTRY)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def corpus(tmp_path):
    value = {
        "schema_version": 1,
        "source_dataset_sha256": "b" * 64,
        "provenance": "synthetic training",
        **{
            split: [
                {
                    "id": f"{split}:synthetic:{n}",
                    "family": "synthetic",
                    "answer": "yes",
                    "prompts": ["Synthetic training question"],
                }
                for n in range(count)
            ]
            for split, count in (("forget_train", 64), ("retain_train", 80))
        },
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
            value["forget_train"].pop()
        if change == "language":
            value["forget_train"][0]["prompts"] = [False]
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
    from lacunai.map import language_relevance

    watch = SimpleNamespace(
        check=lambda: None,
        root=tmp_path,
        artifact_budget_root=None,
        limits={"disk_max_bytes": 10 * 1024**3},
    )
    payload = {
        "family_counts": torch.full((7,), 16, dtype=torch.int64),
        "family_nll_sum": torch.ones(7, dtype=torch.float64),
        "layer.0.signed_sum": torch.tensor(
            [[-2.0, -1.0]] * 2 + [[0.1, 0.2]] * 5, dtype=torch.float64
        ),
    }
    payload["layer.0.absolute_sum"] = payload["layer.0.signed_sum"].abs()
    payload.update(language_relevance.derive(payload, [0]))
    metadata = {
        "schema_version": language_relevance.VERSION,
        "families": list(language_relevance.FAMILIES),
        "layers": [0],
        "payload": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": language_relevance.payload_hash(value),
            }
            for key, value in payload.items()
        },
    }
    tampered = json.loads(json.dumps(metadata))
    tampered["payload"]["family_counts"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash/shape/dtype differs"):
        runner.persist(tmp_path, tampered, payload, watch)
    assert list(tmp_path.iterdir()) == []
    saved = runner.persist(tmp_path, metadata, payload, watch)
    assert json.loads((tmp_path / "diagnosis.json").read_text()) == metadata
    from safetensors.torch import load_file

    restored = load_file(str(tmp_path / "sensitivities.safetensors"))
    language_relevance.validate_payload(metadata, restored)
    assert all(torch.equal(restored[key], value) for key, value in payload.items())
    assert set(saved) == {"diagnosis.json", "sensitivities.safetensors"}
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


@pytest.mark.parametrize("mutates,pending_pressure", [(False, False), (True, False), (False, True)])
def test_diagnosis_error_still_records_source_integrity(
    tmp_path, monkeypatch, capsys, mutates, pending_pressure
):
    from contextlib import nullcontext

    args = argv(tmp_path)
    monkeypatch.setattr(sys, "argv", [*args, "--apply"])
    inventory = {"config.json": "b" * 64}
    runtime = SimpleNamespace(inventory=inventory, check=lambda: None)
    watch = SimpleNamespace(check=lambda: None)
    monkeypatch.setattr(runner, "host_sample", lambda *_: {})
    monkeypatch.setattr(runner, "Watchdog", lambda *a: nullcontext(watch))
    reads = []
    monkeypatch.setattr(
        runner, "checkpoint_inventory", lambda *a: reads.append("file") or inventory
    )
    monkeypatch.setattr(runner, "digest", lambda *_: "a" * 64)
    monkeypatch.setattr(runner, "HFRuntime", lambda *a: runtime)
    calls = []

    def check():
        if pending_pressure and len(calls) > 1:
            raise RuntimeError("injected pending pressure failure")

    watch.check = check
    monkeypatch.setattr(runner, "persist", lambda *a: {})

    def hashes(*a):
        calls.append("canonical")
        return {"weight": "changed" if mutates and len(calls) > 1 else "original"}

    monkeypatch.setattr(runner, "canonical_hashes", hashes)

    def fail(rt, corpus):
        assert set(corpus) == {"forget_train", "retain_train"}
        assert '"state": "active"' in capsys.readouterr().err
        if pending_pressure:
            return {}, {}
        raise RuntimeError("injected diagnosis failure")

    monkeypatch.setattr(runner, "import_module", lambda *_: SimpleNamespace(diagnose=fail))
    with pytest.raises(
        ValueError if mutates else RuntimeError, match="modified" if mutates else "injected"
    ):
        runner.main()
    directory = tmp_path / "runs/diagnosis"
    receipt = json.loads((directory / "source-verification.json").read_text())
    assert receipt["unchanged"] is (not mutates)
    assert calls == ["canonical", "canonical"] and reads == ["file", "file"]
    assert json.loads((directory / "status.json").read_text())["state"] == "failed"
    assert '"state": "failed"' in capsys.readouterr().err
