"""Software-only orchestration fixtures; no model inference or unlearning evidence."""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from lacunai.core import service as module
from lacunai.core.service import Service
from lacunai.core.spec import RuntimeConfig, digest
from lacunai.experimental.artifacts import atomic_json
from lacunai.restore import checkpoints
from lacunai.verify.reports import verify_report
from tests.test_verification_records import Runtime, spec


def target_file(root: Path) -> Path:
    path = root / "target.json"
    atomic_json(path, asdict(spec()))
    return path


def checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen2"}')
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
    )
    return root


@pytest.fixture
def lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Service, Path, Path]:
    model = checkpoint(tmp_path / "model")
    target = target_file(tmp_path)

    class FixtureRuntime(Runtime):
        def __init__(self, path: Path, config: RuntimeConfig) -> None:
            self.inventory = module.checkpoint_inventory(path, config.max_model_bytes)
            self.fingerprint = digest(self.inventory)
            changed = (path / "model.safetensors").read_bytes()[-4:] != struct.pack("<f", 1.0)
            super().__init__(["unknown" if changed else "drazel"] * 2 + ["oak", "pine"])

    def fingerprint(runtime: FixtureRuntime, target: object) -> dict:
        return {
            "kind": "fingerprint",
            "target_id": spec().target_id,
            "spec_sha256": spec().digest,
            "checkpoint_fingerprint": runtime.fingerprint,
            "layers": {"0": {"target_mean": [1.0]}},
        }

    def locate(runtime: FixtureRuntime, target: object, fp: dict) -> dict:
        return {
            **fp,
            "kind": "localization",
            "layers": [{"layer": 0, "mean_effect": 1.0}],
            "selected": [{"layer": 0, "neurons": [0]}],
        }

    from lacunai.erase import methods

    monkeypatch.setattr(module, "HFRuntime", FixtureRuntime)
    monkeypatch.setattr(module, "fingerprint", fingerprint)
    monkeypatch.setattr(module, "locate", locate)
    monkeypatch.setattr(methods, "touched_tensors", lambda *args: ["weight"])
    monkeypatch.setattr(
        methods,
        "apply_method",
        lambda *args, **kwargs: {
            "replacements": {"weight": struct.pack("<f", 0.0)},
            "diagnostics": {"fixture": "software-only byte replacement; not an ML method"},
        },
    )

    # Exercise actual durable byte transactions without installing torch.
    def replacement(value: bytes, descriptor: dict) -> bytes:
        if not isinstance(value, bytes) or len(value) != descriptor["length"]:
            raise ValueError("fixture replacement size mismatch")
        return value

    monkeypatch.setattr(checkpoints, "_replacement_bytes", replacement)
    return Service(tmp_path / "workspace"), model, target


def edit(lifecycle: tuple[Service, Path, Path]) -> dict:
    service, model, target = lifecycle
    return service.erase(model, target, "ablate-neurons", apply=True)


def test_dry_run_signs_without_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = Service(tmp_path / "workspace")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("dry run loaded a model")

    monkeypatch.setattr(module, "HFRuntime", forbidden)
    result = service.erase(tmp_path / "nonexistent", target_file(tmp_path), "ablate-neurons")
    assert result["dry_run"] is True
    assert result["status"] == "complete"
    assert verify_report(
        service.store.path(result["run_id"]), service.workspace / "signing/public.key"
    )["verified"]


def test_map_edit_verify_restore_software_lifecycle(lifecycle: tuple[Service, Path, Path]) -> None:
    service, model, target = lifecycle
    original = (model / "model.safetensors").read_bytes()
    mapped = service.map(model, target)
    assert Path(mapped["exports"]["heatmap"]).read_text().startswith("<svg")
    result = edit(lifecycle)
    assert result["verdict"] == "passed_the_stated_experimental_contract"
    output = Path(result["output"])
    assert (output / "model.safetensors").read_bytes() != original
    assert (model / "model.safetensors").read_bytes() == original
    verified = service.verify(result["run_id"])
    assert verified["verdict"] == result["verdict"]
    restored = service.restore(result["run_id"], apply=True)
    assert restored["restoration"]["restored_bit_exact"] is True
    assert (output / "model.safetensors").read_bytes() == original
    assert service.restore(result["run_id"], apply=True)["restoration"]["restored_bit_exact"]
    assert verify_report(
        service.store.path(result["run_id"]), service.workspace / "signing/public.key"
    )["verified"]


def test_mutable_parent_cannot_redirect_restore(lifecycle: tuple[Service, Path, Path]) -> None:
    service, _, _ = lifecycle
    result = edit(lifecycle)
    path = service.store.path(result["run_id"]) / "run.json"
    record = json.loads(path.read_text())
    record["output"] = "/different/checkpoint"
    atomic_json(path, record)
    with pytest.raises(ValueError, match="operational run differs"):
        service.restore(result["run_id"], apply=True)


def test_unsigned_snapshot_cannot_redirect_restore(lifecycle: tuple[Service, Path, Path]) -> None:
    service, model, _ = lifecycle
    result = edit(lifecycle)
    path = service.store.path(result["run_id"]) / "snapshot.json"
    snapshot = json.loads(path.read_text())
    redirected = model.parent / "redirected-checkpoint"
    shutil.copytree(Path(result["output"]), redirected)
    snapshot["output"] = str(redirected)
    atomic_json(path, snapshot)
    with pytest.raises(ValueError, match="snapshot transaction differs"):
        service.restore(result["run_id"], apply=True)


def test_partial_write_is_signed_and_recoverable(
    lifecycle: tuple[Service, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    service, model, target = lifecycle
    original = (model / "model.safetensors").read_bytes()
    writer = module.write_edited_checkpoint

    def interrupted(*args: object, **kwargs: object) -> dict:
        result = writer(*args, **kwargs)
        receipt = Path(result["run_dir"]) / "snapshot.json"
        atomic_json(receipt, {**result, "status": "writing"})
        raise OSError("fixture interrupted after verified replacement")

    monkeypatch.setattr(module, "write_edited_checkpoint", interrupted)
    with pytest.raises(OSError, match="fixture interrupted"):
        service.erase(model, target, "ablate-neurons", apply=True)
    failed = service.store.list()[0]
    assert failed["status"] == "failed"
    restored = service.restore(failed["run_id"], apply=True)
    assert (Path(restored["restoration"]["output"]) / "model.safetensors").read_bytes() == original


def test_signing_failure_is_failed_and_original_error_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = Service(tmp_path / "workspace")

    def signing_failure(*args: object) -> None:
        raise OSError("fixture signing unavailable")

    monkeypatch.setattr(module, "write_signed_report", signing_failure)
    with pytest.raises(OSError, match="fixture signing unavailable"):
        service.erase(tmp_path / "model", target_file(tmp_path), "ablate-neurons")
    record = service.store.list()[0]
    assert record["status"] == "failed"
    failure = json.loads(
        (service.store.path(record["run_id"]) / "finalization-error.json").read_text()
    )
    assert failure["scientific_report_usable"] is False


def test_workspace_lock_rejects_other_service(tmp_path: Path) -> None:
    service = Service(tmp_path / "workspace")
    other = Service(service.workspace)
    with service._operation(), pytest.raises(ValueError, match="locked"):
        other.runs()
    assert other.runs()["runs"] == []


@pytest.mark.parametrize("phase", ["before_snapshot", "snapshot_receipt"])
def test_source_drift_refused_before_edit(
    lifecycle: tuple[Service, Path, Path], monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    service, model, target = lifecycle
    if phase == "before_snapshot":
        qualify = service._qualify

        def change_after_qualification(before: dict, target: object) -> None:
            qualify(before, target)
            (model / "config.json").write_text('{"model_type":"qwen2","changed":true}')

        monkeypatch.setattr(service, "_qualify", change_after_qualification)
    else:
        prepare = module.prepare_snapshot

        def changed_snapshot(*args: object) -> dict:
            result = prepare(*args)
            result["before_files"]["config.json"]["sha256"] = "0" * 64
            return result

        monkeypatch.setattr(module, "prepare_snapshot", changed_snapshot)
    with pytest.raises(ValueError, match="source checkpoint changed|snapshot source differs"):
        service.erase(model, target, "ablate-neurons", apply=True)
    assert service.store.list()[0]["status"] == "failed"
    assert not (service.workspace / "checkpoints").exists()


def test_verification_cannot_change_inference_contract(
    lifecycle: tuple[Service, Path, Path],
) -> None:
    service, _, _ = lifecycle
    result = edit(lifecycle)
    with pytest.raises(ValueError, match="inference settings differ"):
        service.verify(result["run_id"], replace(RuntimeConfig(), max_new_tokens=2))


def test_state_wipe_failure_has_its_own_failed_run(
    lifecycle: tuple[Service, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from lacunai.adapters import neurailyzer

    service, model, target = lifecycle
    monkeypatch.setattr(neurailyzer, "preflight", lambda *args: {"fixture": True})

    def failure(*args: object, **kwargs: object) -> None:
        # The state action must already have a durable run before it starts.
        assert any(
            row["operation"] == "forget" and row["status"] == "running"
            for row in service.store.list()
        )
        raise OSError("fixture state wipe failed")

    monkeypatch.setattr(neurailyzer, "wipe_state", failure)
    with pytest.raises(OSError, match="fixture state wipe failed"):
        service.forget(model, target, "ablate-neurons", apply=True, state_config=target)
    records = {row["operation"]: row for row in service.store.list()}
    assert records["erase"]["status"] == "complete"
    assert records["forget"]["status"] == "failed"
    assert verify_report(
        service.store.path(records["forget"]["run_id"]), service.workspace / "signing/public.key"
    )["verified"]


@pytest.mark.parametrize("wrapper", ["verify", "forget"])
def test_restore_returned_wrapper_run(lifecycle, monkeypatch, wrapper):
    from lacunai.adapters import neurailyzer

    service, model, target = lifecycle
    original = (model / "model.safetensors").read_bytes()
    if wrapper == "verify":
        result = service.verify(edit(lifecycle)["run_id"])
    else:
        monkeypatch.setattr(neurailyzer, "preflight", lambda *args: {"fixture": True})
        monkeypatch.setattr(neurailyzer, "wipe_state", lambda *args, **kw: {"performed": True})
        result = service.forget(model, target, "ablate-neurons", apply=True, state_config=target)
    restored = service.restore(result["run_id"], apply=True)
    assert restored["restoration"]["restored_bit_exact"]
    assert (Path(restored["restoration"]["output"]) / "model.safetensors").read_bytes() == original


def signed_wrapper(service, owner_id, **fields):
    run_id, _ = service.store.create(spec(), "forget", {"weight_run_id": owner_id})
    return service._finish(run_id, weight_run_id=owner_id, **fields)


@pytest.mark.parametrize(
    "field", ["source_fingerprint", "output_fingerprint", "snapshot_id", "output"]
)
def test_signed_wrapper_identity_mismatch_rejected(lifecycle, field):
    service, _, _ = lifecycle
    owner = edit(lifecycle)
    wrapper = signed_wrapper(service, owner["run_id"], **{field: "wrong"})
    with pytest.raises(ValueError, match="differs from snapshot"):
        service.restore(wrapper["run_id"], apply=True)


def test_wrapper_chain_authenticates_every_hop(lifecycle):
    service, _, _ = lifecycle
    owner = edit(lifecycle)
    wrapper = signed_wrapper(service, owner["run_id"])
    record_path = service.store.path(owner["run_id"]) / "run.json"
    record = json.loads(record_path.read_text())
    record["output"] = "/redirected"
    atomic_json(record_path, record)
    with pytest.raises(ValueError, match="operational run differs"):
        service.restore(wrapper["run_id"], apply=True)


def test_signed_cycle_and_traversal_rejected(lifecycle):
    service, _, _ = lifecycle
    first, _ = service.store.create(spec(), "forget", {"weight_run_id": "pending"})
    # Construct a genuinely signed cycle, rather than bypass authentication in the test.
    path = service.store.path(first) / "run.json"
    record = json.loads(path.read_text())
    record["metadata"]["weight_run_id"] = first
    atomic_json(path, record)
    service._finish(first, weight_run_id=first)
    with pytest.raises(ValueError, match="cycle"):
        service.restore(first, apply=True)
    wrapper = signed_wrapper(service, "../outside")
    with pytest.raises(ValueError, match="invalid run identifier"):
        service.restore(wrapper["run_id"], apply=True)


def test_signed_wrapper_chain_is_bounded(lifecycle):
    service, _, _ = lifecycle
    current = edit(lifecycle)
    for _ in range(32):
        current = signed_wrapper(service, current["run_id"])
    with pytest.raises(ValueError, match="exceeds 32"):
        service.restore(current["run_id"], apply=True)


def test_verify_after_forget_resolves_weight_owner(lifecycle, monkeypatch):
    from lacunai.adapters import neurailyzer

    service, model, target = lifecycle
    monkeypatch.setattr(neurailyzer, "preflight", lambda *args: {})
    monkeypatch.setattr(neurailyzer, "wipe_state", lambda *args, **kw: {"performed": True})
    result = service.forget(model, target, "ablate-neurons", apply=True, state_config=target)
    verified = service.verify(result["run_id"])
    assert verified["verdict"] == result["verdict"]
    assert service.restore(verified["run_id"], apply=True)["restoration"]["restored_bit_exact"]


def test_chain_rejects_different_target(lifecycle):
    service, _, _ = lifecycle
    owner = edit(lifecycle)
    run_id, _ = service.store.create(
        replace(spec(), target_id="different"), "forget", {"weight_run_id": owner["run_id"]}
    )
    service._finish(run_id, weight_run_id=owner["run_id"])
    with pytest.raises(ValueError, match="target identity differs"):
        service.restore(run_id, apply=True)


def test_chain_rejects_foreign_signing_key(lifecycle, tmp_path):
    service, _, _ = lifecycle
    owner = edit(lifecycle)
    run_id, directory = service.store.create(spec(), "forget", {"weight_run_id": owner["run_id"]})
    fields = {"status": "complete", "weight_run_id": owner["run_id"]}
    module.write_signed_report(
        directory, {**service.store.get(run_id), **fields}, tmp_path / "foreign-signing"
    )
    service.store.update(run_id, **fields)
    with pytest.raises(ValueError, match="key|signature|trust"):
        service.restore(run_id, apply=True)
