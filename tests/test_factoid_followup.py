"""Offline orchestration only: scripted evaluations never demonstrate unlearning."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from lacunai.core.spec import RuntimeConfig, digest
from lacunai.experimental.artifacts import atomic_json
from tests.test_planted_pilot import contextual_training, contract


def module():
    file = Path(__file__).resolve().parents[1] / "scripts/factoid_followup.py"
    definition = importlib.util.spec_from_file_location("followup_test", file)
    result = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(result)
    return result


class Budget:
    def __init__(self, *args):
        self.reserves = []

    def check(self, reserve=0):
        self.reserves.append(reserve)

    def config(self):
        return RuntimeConfig(max_seconds=None)

    def finalization(self):
        return self


class Store:
    def __init__(self, path):
        self.root = path
        self.rows = []

    def create(self, *args):
        directory = self.root / "record"
        directory.mkdir(parents=True)
        return "record", directory

    def list(self):
        return self.rows

    def path(self, run_id):
        return self.root / run_id


def orchestration(tmp_path, monkeypatch):
    m = module()
    events = []
    inventory = {"model.safetensors": "a" * 64}
    source = tmp_path / "source"
    source.mkdir()
    spec = contract()
    baseline = {"fixture": "software-only", "qualified": True}

    class Service:
        @staticmethod
        def _qualify(value, spec):
            if not value["qualified"]:
                raise ValueError("baseline unqualified")

        def __init__(self, path):
            self.store = Store(path)

        def erase(self, model, target, method, **kwargs):
            events.append(("erase", method, kwargs))
            run_id = method
            d = self.store.path(run_id)
            d.mkdir(parents=True)
            atomic_json(
                d / "snapshot.json",
                {"output": str(model), "status": "edited", "tensors": {"weight": {}}},
            )
            self.store.rows.append({"run_id": run_id, "operation": "erase"})
            return {"run_id": run_id, "verdict": m.PASSED_VERDICT}

        def verify(self, *args):
            events.append(("verify",))
            return {"verdict": m.PASSED_VERDICT}

        def restore(self, *args, **kwargs):
            events.append(("restore",))
            return {"status": "complete"}

    def score(model, *args):
        events.append(("score", model))
        return baseline.copy()

    def copy(source, scratch, budget):
        events.append(("copy", source, scratch))
        scratch.mkdir()

    pilot = SimpleNamespace(
        Budget=Budget,
        RunStore=Store,
        Service=Service,
        atomic_json=atomic_json,
        checkpoint_inventory=lambda *args: inventory,
        bounded_file=lambda p: p.read_bytes(),
        score_checkpoint=score,
        generation_identity=lambda value: value,
        byte_copy=copy,
        persist_afterimages=lambda *args: {"fixture": "not tensor evidence"},
        benign_relearning=lambda *args: {"evaluation": baseline.copy()},
        compare=lambda *args: {"verdict": m.PASSED_VERDICT},
        sign_aggregate=lambda *args: {"fixture": "not a signature"},
    )
    plan = {
        "workspace": str(tmp_path / "followup"),
        "budget_root": str(tmp_path),
        "planted": str(source),
        "inputs": {k: {"path": "/pinned/" + k, "sha256": "a" * 64} for k in m.INPUTS},
    }
    records = {
        "spec": spec,
        "planted_inventory": inventory,
        "planted_baseline": baseline,
        "original_summary": {"planting": {"steps": 192}},
        "development": {},
        "training": {},
        "original_manifest": {},
    }
    monkeypatch.setattr(m, "reserve_transaction", lambda *args: events.append(("reserve",)))
    return m, pilot, plan, records, events


def test_actual_orchestration_sequence_and_fixed_settings(tmp_path, monkeypatch):
    m, pilot, plan, records, events = orchestration(tmp_path, monkeypatch)
    result = m.execute(plan, records, pilot)
    assert result["passed_methods"] == list(m.METHODS)  # Software control-flow result only.
    edits = [row for row in events if row[0] == "erase"]
    assert len(edits) == 2 and [row[1] for row in edits] == list(m.METHODS)
    for _, _, options in edits:
        assert (
            options["steps"] == 128
            and options["strength"] == 1
            and options["max_relative_change"] == 0.25
        )
        assert options["in_place"] and options["output"] == Path(plan["workspace"]) / "scratch"
        assert options["gradient_data"] == Path(plan["workspace"]) / "gradient-data.json"
    assert len([row for row in events if row[0] == "copy"]) == 1
    assert [row[0] for row in events if row[0] in {"erase", "verify", "restore"}] == [
        "erase",
        "verify",
        "restore",
    ] * 2
    assert result["retain_reference"]["original_plant_steps"] == 192


def test_unqualified_baseline_prevents_scratch_and_edits(tmp_path, monkeypatch):
    m, pilot, plan, records, events = orchestration(tmp_path, monkeypatch)
    pilot.score_checkpoint = lambda *args: {"qualified": False}
    result = m.execute(plan, records, pilot)
    assert result["status"] == "failed" and "unqualified" in result["error"]
    assert not any(row[0] in {"copy", "erase"} for row in events)


def test_restoration_failure_stops_second_method(tmp_path, monkeypatch):
    m, pilot, plan, records, events = orchestration(tmp_path, monkeypatch)

    def restore(*args, **kwargs):
        raise ValueError("injected exact restoration failure")

    pilot.Service.restore = restore
    result = m.execute(plan, records, pilot)
    assert result["status"] == "failed" and "restoration" in result["error"]
    assert len([row for row in events if row[0] == "erase"]) == 1


def test_failed_edit_with_snapshot_is_restored(tmp_path, monkeypatch):
    m, pilot, plan, records, events = orchestration(tmp_path, monkeypatch)
    original = pilot.Service.erase

    def erase(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise ValueError("injected after write")

    pilot.Service.erase = erase
    result = m.execute(plan, records, pilot)
    assert result["verdict"] == "failed"
    assert all(
        row["recovered_failed_run_for_restore"] and row["restored_bit_exact"]
        for row in result["candidates"]
    )
    assert len([row for row in events if row[0] == "restore"]) == 2


def prepared(tmp_path, monkeypatch):
    m = module()
    code = tmp_path / "code"
    (code / "scripts").mkdir(parents=True)
    for name in ("factoid_followup.py", "planted_fact_pilot.py", "contextual_planting.py"):
        (code / "scripts" / name).write_text("# software fixture\n")
    scorer = code / "src/lacunai/verify/evaluation.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_text("# fixture scorer\n")
    hashes = {str(p.relative_to(code)): m.file_hash(p) for p in code.rglob("*.py")}

    def pin(name, value):
        p = tmp_path / name
        atomic_json(p, value)
        return {"path": str(p), "sha256": m.file_hash(p)}

    atomic_json(code / "hashes.json", hashes)
    spec = contract()
    inventory = {"model.safetensors": "a" * 64}
    baseline = {
        "checkpoint_fingerprint": digest(inventory),
        "spec_sha256": spec.digest,
        "qualified": True,
    }
    training = contextual_training()
    manifest = {
        "model_revision": "revision",
        "target_sha256": spec.digest,
        "runtime": asdict(RuntimeConfig(max_seconds=None)),
        "code_hashes": hashes,
        "versions": {},
        "plant_data_sha256": digest(training),
        "source_inventory": inventory,
    }
    values = {
        "target": asdict(spec),
        "gradient_data": {
            "schema_version": 1,
            "source_fingerprint": digest(inventory),
            "forget": training["development"],
            "retain_development": training["retain_development"],
            "provenance": "software-only",
        },
        "plant_data": training,
        "original_manifest": manifest,
        "original_summary": {
            "status": "complete",
            "planting": {"qualified": True, "reload_generation_exact": True, "steps": 192},
        },
        "planted_baseline": baseline,
        "untouched": baseline,
        "retain_only": baseline,
        "noop": baseline,
        "planted_inventory": inventory,
    }
    plan = {
        "schema_version": 1,
        "code": str(code),
        "bundle": {"path": str(code / "hashes.json"), "sha256": m.file_hash(code / "hashes.json")},
        "workspace": str(tmp_path / "followup"),
        "budget_root": str(tmp_path),
        "planted": str(tmp_path / "source"),
        "inputs": {k: pin(k + ".json", v) for k, v in values.items()},
    }
    from lacunai.core.spec import TargetSpec

    fake = SimpleNamespace(
        TargetSpec=TargetSpec,
        REVISION="revision",
        Budget=Budget,
        digest=digest,
        checkpoint_inventory=lambda *args: inventory,
        generation_identity=lambda value: value,
        Service=SimpleNamespace(_qualify=lambda *args: None),
        load_training=lambda *a, **k: training,
        importlib=SimpleNamespace(metadata=SimpleNamespace(version=lambda name: "fixture")),
    )
    monkeypatch.setattr(m, "api", lambda code: fake)
    monkeypatch.setattr(m, "__file__", str(code / "scripts/factoid_followup.py"))
    return m, plan, values, pin


def test_prepare_binds_actual_source_and_gradient_source(tmp_path, monkeypatch):
    m, plan, values, pin = prepared(tmp_path, monkeypatch)
    p = pin("plan.json", plan)
    m.prepare(Path(p["path"]), p["sha256"])
    values["gradient_data"]["source_fingerprint"] = "b" * 64
    plan["inputs"]["gradient_data"] = pin("gradient_data.json", values["gradient_data"])
    p = pin("plan.json", plan)
    with pytest.raises(ValueError, match="source fingerprint"):
        m.prepare(Path(p["path"]), p["sha256"])


def test_prepare_refuses_live_original_and_existing_workspace(tmp_path, monkeypatch):
    m, plan, values, pin = prepared(tmp_path, monkeypatch)
    values["original_summary"]["status"] = "running"
    plan["inputs"]["original_summary"] = pin("original_summary.json", values["original_summary"])
    p = pin("plan.json", plan)
    with pytest.raises(ValueError, match="terminal"):
        m.prepare(Path(p["path"]), p["sha256"])
    Path(plan["workspace"]).mkdir()
    with pytest.raises(FileExistsError):
        m.prepare(Path(p["path"]), p["sha256"])


def test_dirty_scratch_refused_before_candidate(tmp_path, monkeypatch):
    m, pilot, plan, records, events = orchestration(tmp_path, monkeypatch)
    pilot.checkpoint_inventory = lambda p, *args: (
        records["planted_inventory"] if p == Path(plan["planted"]) else {"bad": "bytes"}
    )
    result = m.execute(plan, records, pilot)
    assert result["status"] == "failed" and "scratch not restored" in result["error"]
    assert not any(row[0] == "erase" for row in events)


def test_resource_abort_restores_and_stops(tmp_path, monkeypatch):
    from lacunai.experimental.resources import ResourceAbort

    m, pilot, plan, records, events = orchestration(tmp_path, monkeypatch)
    original = pilot.Service.erase

    def erase(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise ResourceAbort("software fixture pressure")

    pilot.Service.erase = erase
    result = m.execute(plan, records, pilot)
    assert result["status"] == "failed" and "ResourceAbort" in result["error"]
    assert len([row for row in events if row[0] == "erase"]) == 1
    assert len([row for row in events if row[0] == "restore"]) == 1


def test_reservation_uses_actual_shard_and_tensor_bytes(tmp_path):
    import struct

    m = module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen2","num_hidden_layers":1}')
    name = "model.layers.0.mlp.up_proj.weight"
    header = json.dumps({name: {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}).encode()
    shard = source / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<ffff", 1, 2, 3, 4))
    budget = Budget()
    m.reserve_transaction(source, budget)
    assert budget.reserves == [shard.stat().st_size + 32 + m.METADATA_RESERVE]


def enable_reuse(m, pilot, plan, records, tmp_path):
    import hashlib

    source = Path(plan["planted"])
    (source / "model.safetensors").write_bytes(b"software fixture payload")
    inventory = {
        "model.safetensors": hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest()
    }
    records["planted_inventory"] = inventory
    prior = tmp_path / "previous-run"
    scratch = prior / "scratch"
    scratch.mkdir(parents=True)
    (scratch / "model.safetensors").write_bytes((source / "model.safetensors").read_bytes())
    plan["code"] = str(tmp_path / "code")
    plan["reuse_scratch"] = {
        "path": str(scratch),
        "prior_run": str(prior),
        "source_fingerprint": digest(inventory),
    }
    pilot.digest = digest

    def actual_inventory(root, *args):
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.iterdir()
            if p.is_file()
        }

    pilot.checkpoint_inventory = actual_inventory
    return scratch


def test_explicit_reuse_does_not_copy_and_records_lineage(tmp_path, monkeypatch):
    m, pilot, plan, records, events = orchestration(tmp_path, monkeypatch)
    scratch = enable_reuse(m, pilot, plan, records, tmp_path)
    result = m.execute(plan, records, pilot)
    assert result["status"] == "complete"
    assert not any(row[0] == "copy" for row in events)
    assert all(row[2]["output"] == scratch for row in events if row[0] == "erase")
    lineage = json.loads((Path(plan["workspace"]) / "scratch-lineage.json").read_text())
    assert lineage["copied_this_run"] is False and lineage["prior_owner"] == plan["reuse_scratch"]


@pytest.mark.parametrize(
    "mutation", ["tamper", "symlink", "source", "hardlink", "pin", "unqualified"]
)
def test_invalid_reuse_refused_before_model_or_workspace(tmp_path, monkeypatch, mutation):
    import os

    m, pilot, plan, records, events = orchestration(tmp_path, monkeypatch)
    scratch = enable_reuse(m, pilot, plan, records, tmp_path)
    if mutation == "tamper":
        (scratch / "model.safetensors").write_bytes(b"modified")
    elif mutation == "symlink":
        alias = tmp_path / "scratch-link"
        alias.symlink_to(scratch, target_is_directory=True)
        plan["reuse_scratch"]["path"] = str(alias)
    elif mutation == "source":
        plan["planted"] = str(scratch)
    elif mutation == "hardlink":
        # Only a newly created software fixture is replaced, never experiment artifacts.
        (scratch / "model.safetensors").unlink()
        os.link(Path(plan["planted"]) / "model.safetensors", scratch / "model.safetensors")
    elif mutation == "pin":
        plan["reuse_scratch"]["source_fingerprint"] = "b" * 64
    else:
        records["planted_baseline"]["qualified"] = False
    with pytest.raises(ValueError):
        m.execute(plan, records, pilot)
    assert not any(row[0] in {"score", "copy", "erase"} for row in events)
    assert not Path(plan["workspace"]).exists()


@pytest.mark.parametrize("steps", [1, 8, 128])
def test_explicit_candidate_steps_reach_both_methods_and_manifest(tmp_path, monkeypatch, steps):
    m, pilot, plan, records, events = orchestration(tmp_path, monkeypatch)
    plan["candidate_steps"] = steps
    m.execute(plan, records, pilot)
    edits = [row for row in events if row[0] == "erase"]
    assert len(edits) == 2
    assert [row[2]["steps"] for row in edits] == [steps, steps]
    manifest = json.loads((Path(plan["workspace"]) / "manifest.json").read_text())
    assert manifest["candidate_settings"]["steps"] == steps
    assert manifest["candidate_settings"]["max_relative_change"] == 0.25


@pytest.mark.parametrize("steps", [True, False, 0, -1, 129, 8.0, "8", None])
def test_invalid_candidate_steps_refused_before_runtime(tmp_path, monkeypatch, steps):
    m, plan, values, pin = prepared(tmp_path, monkeypatch)
    plan["candidate_steps"] = steps
    p = pin("plan.json", plan)
    monkeypatch.setattr(m, "api", lambda *a: pytest.fail("runtime reached before step validation"))
    with pytest.raises(ValueError, match="candidate_steps"):
        m.prepare(Path(p["path"]), p["sha256"])
    assert not Path(plan["workspace"]).exists()


def test_candidate_steps_tamper_is_rejected_at_execution_entry(tmp_path, monkeypatch):
    m, pilot, plan, records, events = orchestration(tmp_path, monkeypatch)
    plan["candidate_steps"] = True
    with pytest.raises(ValueError, match="candidate_steps"):
        m.execute(plan, records, pilot)
    assert events == []
    assert not Path(plan["workspace"]).exists()


def test_prepare_accepts_explicit_candidate_steps(tmp_path, monkeypatch):
    m, plan, values, pin = prepared(tmp_path, monkeypatch)
    plan["candidate_steps"] = 8
    p = pin("plan.json", plan)
    accepted, _, _ = m.prepare(Path(p["path"]), p["sha256"])
    assert accepted["candidate_steps"] == 8
