"""Admission/copy/receiver tests only; never dispatches a model runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from lacunai.core.spec import digest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def supervisor() -> Any:
    return load("planted_fact_supervisor")


def cache(root: Path) -> tuple[Path, dict]:
    root.mkdir()
    blobs = {
        "config.json": b'{"model_type":"qwen2"}',
        "model.safetensors": b"software-only fixture",
    }
    for name, data in blobs.items():
        (root / name).write_bytes(data)
    (root / "partial.download").write_bytes(b"must remain untouched")
    return root, {
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in blobs.items()},
        "metadata": [{"path": name, "bytes": len(data)} for name, data in blobs.items()],
    }


def bundle(root: Path, source_manifest: dict) -> tuple[Path, dict]:
    root.mkdir()
    names = [
        *[str(p.relative_to(ROOT)) for p in (ROOT / "src/lacunai").rglob("*.py")],
        "scripts/planted_fact_pilot.py",
        "scripts/planted_fact_supervisor.py",
        "examples/planted-fact/target.json",
        "examples/planted-fact/plant-data.json",
    ]
    for name in names:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    (root / "source-manifest.json").write_text(json.dumps(source_manifest))
    names.append("source-manifest.json")
    manifest = {
        "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    }
    (root / "bundle.json").write_text(json.dumps(manifest))
    return root, {
        "approved": True,
        "bundle_sha256": digest(manifest),
        "runtime_seconds": 10800,
        "disk_max_bytes": 10 * 1024**3,
    }


def test_approval_binds_exact_bundle_and_file_bytes(supervisor: Any, tmp_path: Path) -> None:
    _, source = cache(tmp_path / "cache")
    directory, receipt = bundle(tmp_path / "bundle", source)
    assert supervisor.verify_inputs(directory, receipt)["files"]
    with pytest.raises(ValueError, match="approval"):
        supervisor.verify_inputs(directory, {**receipt, "approved": False})
    with pytest.raises(ValueError, match="approval"):
        supervisor.verify_inputs(directory, {**receipt, "bundle_sha256": "0" * 64})
    (directory / "source-manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        supervisor.verify_inputs(directory, receipt)


def test_supervisor_default_off_before_receipt_read(
    supervisor: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supervisor",
            "--bundle",
            str(tmp_path),
            "--run-root",
            str(tmp_path / "run"),
            "--source-cache",
            str(tmp_path),
            "--approval",
            str(tmp_path / "missing"),
        ],
    )
    with pytest.raises(ValueError, match="disabled without --apply"):
        supervisor.main()
    assert not (tmp_path / "run").exists()


def test_direct_pilot_default_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pilot = load("planted_fact_pilot")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("default-off pilot invoked numerical execution")

    monkeypatch.setattr(pilot, "HFRuntime", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            *[
                part
                for name in ("source", "workspace", "target", "plant-data", "source-manifest")
                for part in ("--" + name, str(tmp_path / name))
            ],
            "--deadline-epoch",
            "1",
        ],
    )
    assert pilot.main() == 0
    assert not (tmp_path / "workspace").exists()


def test_stage_only_pinned_bytes_and_preserves_source(supervisor: Any, tmp_path: Path) -> None:
    source, manifest = cache(tmp_path / "cache")
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "output"
    supervisor.stage_source(source, output, manifest)
    assert {p.name for p in output.iterdir()} == set(manifest["files"])
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    for name in manifest["files"]:
        assert (output / name).read_bytes() == before[name]
        assert (output / name).stat().st_ino != (source / name).stat().st_ino
        assert (output / name).stat().st_mode & 0o222 == 0


def test_source_hash_mismatch_refused(supervisor: Any, tmp_path: Path) -> None:
    source, manifest = cache(tmp_path / "cache")
    changed = b"x" * (source / "model.safetensors").stat().st_size
    (source / "model.safetensors").write_bytes(changed)
    with pytest.raises(ValueError, match="hash differs"):
        supervisor.stage_source(source, tmp_path / "output", manifest)
    assert (source / "model.safetensors").read_bytes() == changed


@pytest.mark.parametrize("name", ["../outside", "/absolute"])
def test_source_and_bundle_escape_refused(supervisor: Any, tmp_path: Path, name: str) -> None:
    source, manifest = cache(tmp_path / "cache")
    manifest["files"][name] = "0" * 64
    manifest["metadata"].append({"path": name, "bytes": 0})
    with pytest.raises(ValueError, match="escapes"):
        supervisor.stage_source(source, tmp_path / "output", manifest)
    directory, receipt = bundle(tmp_path / "bundle", manifest)
    approved = json.loads((directory / "bundle.json").read_text())
    approved["files"][name] = "0" * 64
    (directory / "bundle.json").write_text(json.dumps(approved))
    with pytest.raises(ValueError, match="escapes|does not cover"):
        supervisor.verify_inputs(directory, {**receipt, "bundle_sha256": digest(approved)})


def test_scientific_failure_still_audits_retained_artifacts(
    supervisor: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, manifest = cache(tmp_path / "cache")
    directory, receipt = bundle(tmp_path / "bundle", manifest)
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps(receipt))
    run_root = tmp_path / "run"
    monkeypatch.setattr(supervisor, "Watchdog", lambda *args: nullcontext())
    monkeypatch.setattr(
        supervisor, "host_sample", lambda *args: {"fixture": "no host admission measurement"}
    )

    def failed_pilot(*args: Any, **kwargs: Any) -> None:
        assert "--apply" in sys.argv
        experiment = run_root / "experiment"
        experiment.mkdir()
        (experiment / "summary.json").write_text('{"status":"complete","verdict":"failed"}')
        raise SystemExit(1)

    monkeypatch.setattr(supervisor.runpy, "run_path", failed_pilot)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supervisor",
            "--bundle",
            str(directory),
            "--run-root",
            str(run_root),
            "--source-cache",
            str(source),
            "--approval",
            str(approval),
            "--apply",
        ],
    )
    assert supervisor.main() == 1
    audit = json.loads((run_root / "receiver-hashes.json").read_text())
    assert (
        audit["experiment/summary.json"]
        == hashlib.sha256((run_root / "experiment/summary.json").read_bytes()).hexdigest()
    )
    status = json.loads((run_root / "status.json").read_text())
    assert status.get("verdict") != "passed_the_stated_experimental_contract"


def test_manifest_must_cover_every_consumed_input(supervisor: Any, tmp_path: Path) -> None:
    _, source = cache(tmp_path / "cache")
    directory, receipt = bundle(tmp_path / "bundle", source)
    manifest = json.loads((directory / "bundle.json").read_text())
    del manifest["files"]["scripts/planted_fact_pilot.py"]
    (directory / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cover every"):
        supervisor.verify_inputs(directory, {**receipt, "bundle_sha256": digest(manifest)})


@pytest.mark.parametrize("field", ["runtime_seconds", "disk_max_bytes"])
def test_approval_cannot_change_frozen_limits(supervisor: Any, tmp_path: Path, field: str) -> None:
    _, source = cache(tmp_path / "cache")
    directory, receipt = bundle(tmp_path / "bundle", source)
    with pytest.raises(ValueError, match="approval limits differ"):
        supervisor.verify_inputs(directory, {**receipt, field: receipt[field] + 1})
