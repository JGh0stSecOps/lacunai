"""Connected runner storage guards; optional Torch cases exercise actual disk transactions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

PATH = Path(__file__).parents[1] / "scripts" / "connected_fact_pilot.py"
spec = importlib.util.spec_from_file_location("connected_pilot", PATH)
assert spec and spec.loader
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class Budget:
    def __init__(self, maximum: int = 10 * 1024**3) -> None:
        self.reservations: list[int] = []
        self.maximum = maximum

    def check(self, reserve: int = 0) -> None:
        self.reservations.append(reserve)
        if reserve > self.maximum:
            raise OSError("reserve exceeded")


def fake_storage(tmp_path: Path) -> Any:
    value = object.__new__(pilot.Storage)
    value.workspace = tmp_path
    value.budget = Budget()
    value.shard_bytes = 988097824
    return value


def test_remaining_record_budget_reserved_with_whole_staged_shard(tmp_path: Path) -> None:
    value = fake_storage(tmp_path)
    (tmp_path / "evidence").write_bytes(b"1234")
    value.reserve()
    assert value.budget.reservations == [988097824 + pilot.RECORD_BUDGET - 4]
    value.budget.maximum = value.budget.reservations[0] - 1
    with pytest.raises(OSError, match="reserve exceeded"):
        value.reserve()


def test_record_cap_is_not_bypassed_by_negative_remaining_reservation(tmp_path: Path) -> None:
    value = fake_storage(tmp_path)
    value.used = lambda: pilot.RECORD_BUDGET + 1
    with pytest.raises(OSError, match="record budget exhausted"):
        value.reserve()


def test_quality_gate_rejects_nonfinite_and_excessive_ppl(monkeypatch: Any) -> None:
    import lacunai.verify.connected as evaluator

    monkeypatch.setattr(evaluator, "qualify_connected", lambda *args: None)

    def evaluation(number: str) -> dict[str, Any]:
        return {"perplexity": {"perplexity": number, "tokens": 4}}

    assert pilot.quality_gate(evaluation("115"), evaluation("100"), None)
    assert not pilot.quality_gate(evaluation("115.01"), evaluation("100"), None)
    with pytest.raises(ValueError, match="invalid qualification perplexity"):
        pilot.quality_gate(evaluation("Infinity"), evaluation("100"), None)


def checkpoint(path: Path, value: float, other: float = 9) -> None:
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    path.mkdir()
    (path / "config.json").write_text('{"num_hidden_layers":1}')
    safe.save_file(
        {
            "model.layers.0.mlp.up_proj.weight": torch.full((2, 2), value),
            "other": torch.tensor([other]),
        },
        str(path / "model.safetensors"),
    )


@pytest.mark.parametrize("fail", [False, True])
def test_actual_nested_snapshot_apply_lifo_restore(tmp_path: Path, fail: bool) -> None:
    torch = pytest.importorskip("torch")
    original, scratch, workspace = (tmp_path / n for n in ("source", "scratch", "work"))
    checkpoint(original, 1.0)
    checkpoint(scratch, 2.0)
    workspace.mkdir()
    name = "model.layers.0.mlp.up_proj.weight"
    identity = pilot.compatible_scratch(original, scratch, name)
    initial = pilot.inventory(scratch)
    storage = pilot.Storage(Budget(), workspace, scratch, name)
    try:
        with storage.transaction("outer", {name: torch.ones(2, 2)}):
            assert pilot.inventory(scratch) == identity["source_inventory"]
            with storage.transaction("inner", {name: torch.full((2, 2), 3.0)}):
                assert pilot.inventory(scratch) != initial
                if fail:
                    raise RuntimeError("injected evaluation failure")
    except RuntimeError as exc:
        assert fail and str(exc) == "injected evaluation failure"
    assert pilot.inventory(scratch) == initial
    assert json.loads((workspace / "transaction-journal.json").read_text())["stack"] == []
    assert pilot.inventory(original) == identity["source_inventory"]


def test_actual_non_target_drift_rejected(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    source, scratch = tmp_path / "source", tmp_path / "scratch"
    checkpoint(source, 1.0)
    checkpoint(scratch, 2.0, other=8)
    with pytest.raises(ValueError, match="non-target tensor differs"):
        pilot.compatible_scratch(source, scratch, "model.layers.0.mlp.up_proj.weight")


def test_actual_partial_write_restores_and_unknown_bytes_refused(
    tmp_path: Path, monkeypatch: Any
) -> None:
    torch = pytest.importorskip("torch")
    scratch, workspace = tmp_path / "scratch", tmp_path / "work"
    checkpoint(scratch, 2.0)
    workspace.mkdir()
    before = pilot.inventory(scratch)
    name = "model.layers.0.mlp.up_proj.weight"
    storage = pilot.Storage(Budget(), workspace, scratch, name)
    writer = pilot.checkpoints.write_edited_checkpoint

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        writer(*args, **kwargs)
        raise RuntimeError("after durable write")

    monkeypatch.setattr(pilot.checkpoints, "write_edited_checkpoint", interrupted)
    with (
        pytest.raises(RuntimeError, match="after durable write"),
        storage.transaction("interrupted", {name: torch.ones(2, 2)}),
    ):
        pytest.fail("body should not execute")
    assert pilot.inventory(scratch) == before
    monkeypatch.setattr(pilot.checkpoints, "write_edited_checkpoint", writer)
    with (
        pytest.raises(pilot.checkpoints.SnapshotError),
        storage.transaction("corrupt", {name: torch.ones(2, 2)}),
        (scratch / "model.safetensors").open("ab") as stream,
    ):
        stream.write(b"unknown")
    assert json.loads((workspace / "restore-failure.json").read_text())["error"]


def test_afterimage_tamper_rejected_before_tensor_import(tmp_path: Path) -> None:
    image = tmp_path / "image.safetensors"
    image.write_bytes(b"original")
    pin = pilot.sha(image)
    image.with_suffix(".json").write_text(json.dumps({"sha256": pin}))
    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="payload pin differs"):
        pilot.load_image(image)
    image.with_suffix(".json").write_text(json.dumps({"sha256": pilot.sha(image)}))
    with pytest.raises(ValueError, match="payload pin differs"):
        pilot.load_image(image, pin)


def test_generation_identity_covers_capabilities() -> None:
    value = {
        k: {"rows": [{"generation": {"output": k}}]} for k in ("target", "retain", "capability")
    }
    assert len(pilot.generation_identity(value)) == 3
    before = pilot.generation_identity(value)
    value["capability"]["rows"][0]["generation"] = {"output": "different"}
    assert before != pilot.generation_identity(value)


def test_signing_never_copies_service_private_keys(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from lacunai.verify.reports import verify_report

    workspace = tmp_path / "work"
    workspace.mkdir()
    keys = workspace / "operations" / "signing"
    keys.mkdir(parents=True)
    (keys / "private.key").write_bytes(b"PRIVATE_MARKER")
    (workspace / "raw.json").write_text('{"observations":[]}')
    key_dir = tmp_path / "keys"
    pilot.sign_evidence(
        workspace,
        {
            "run_id": "a" * 32,
            "target_id": "fixture",
            "status": "complete",
            "verdict": "inconclusive",
        },
        key_dir,
        Budget(),
    )
    verified = verify_report(workspace / "report-evidence", key_dir / "public.key")
    assert verified["files_verified"] >= 3
    signed = (workspace / "report-evidence/signature.json").read_text()
    assert "private.key" not in signed
    assert all(
        b"PRIVATE_MARKER" not in p.read_bytes()
        for p in (workspace / "report-evidence").rglob("*")
        if p.is_file()
    )


def test_main_bad_plan_is_terminal_failure(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    path = tmp_path / "plan.json"
    path.write_text("{}")
    monkeypatch.setattr(
        "sys.argv", ["connected_fact_pilot.py", "--plan", str(path), "--plan-sha256", "0" * 64]
    )
    assert pilot.main() == 1
    lines = capsys.readouterr().err.splitlines()
    assert json.loads(lines[0])["state"] == "active"
    assert json.loads(lines[-1])["state"] == "failed"


def test_final_budget_failure_writes_terminal_error(tmp_path: Path) -> None:
    class FailedBudget:
        def check(self) -> None:
            raise OSError("injected pressure during finalization")

    summary = {"status": "complete", "verdict": "passed_the_stated_experimental_contract"}
    with pytest.raises(OSError, match="injected pressure"):
        pilot.finalize_evidence(tmp_path, summary, tmp_path / "keys", FailedBudget())
    actual = json.loads((tmp_path / "summary.json").read_text())
    assert actual["status"] == "failed" and actual["verdict"] == "inconclusive"
    assert "injected pressure" in actual["finalization_error"]
    assert json.loads((tmp_path / "attestation-error.json").read_text())["signed"] is False


def test_generation_screen_counts_capabilities_and_finishes(monkeypatch: Any, capsys: Any) -> None:
    from types import SimpleNamespace

    from lacunai.core.connected import ConnectedProbe
    from lacunai.core.spec import Probe

    spec = SimpleNamespace(
        probes=[
            ConnectedProbe(
                prompt="fixture",
                answer="42",
                probe_id=f"p{i}",
                fact_id="f",
                choices=["42", "43", "44", "45"],
            )
            for i in range(24)
        ],
        retain=[],
        capability_probes=[Probe("capabilityfixture", "42") for _ in range(8)],
        facts=[],
        abstentions=["unknown"],
    )
    runtime = SimpleNamespace(
        check=lambda: None,
        generate=lambda prompt: {
            "output": "42",
            "truncated": False,
            "token_ids": [1],
            "output_tokens": 1,
            "input_tokens": 1,
        },
    )
    result = pilot.generation_screen(runtime, spec)
    assert result["all_correct"] and len(result["capability"]) == 8
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    measured = [r for r in events if "completed" in r]
    assert {r["total"] for r in measured} == {32}
    assert measured[-1]["completed"] == 32
    assert events[0]["state"] == "active" and events[-1]["state"] == "complete"


def test_failed_service_report_authenticates_real_snapshot_restore(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from lacunai.core.service import Service
    from lacunai.core.spec import MappingProbe, Probe, TargetSpec

    scratch = tmp_path / "scratch"
    checkpoint(scratch, 2.0)
    before = pilot.inventory(scratch)
    target = TargetSpec(
        target_id="fixture",
        description="Synthetic fixture",
        mapping=[MappingProbe("map fixture", "corrupt fixture", "42")],
        probes=[Probe("direct fixture", "42"), Probe("paraphrase fixture", "42", "paraphrase")],
        retain=[Probe("retain fixture", "9", "retain")],
        background=["background fixture"],
        perplexity_texts=["perplexity fixture"],
    )
    service = Service(tmp_path / "operations")
    run_id, directory = service.store.create(target, "erase", {"model": str(scratch)})
    name = "model.layers.0.mlp.up_proj.weight"
    snapshot = pilot.checkpoints.prepare_snapshot(scratch, directory, [name])
    pilot.checkpoints.write_edited_checkpoint(
        scratch, scratch, {name: torch.ones(2, 2)}, snapshot, in_place=True
    )
    service._failure(run_id, RuntimeError("injected postwrite evaluation failure"))
    assert service.store.get(run_id)["status"] == "failed"
    restored = service.restore(run_id, apply=True)
    assert restored["status"] == "complete"
    assert pilot.inventory(scratch) == before


def test_connected_requires_four_facts_in_each_role() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from lacunai.core.connected import Fact
    from lacunai.core.spec import Probe

    facts = [
        Fact(
            f"f{i}",
            f"subject{i}",
            "relation",
            f"answer{i}",
            "forget" if i < 4 else "neighbor" if i < 8 else "unrelated",
        )
        for i in range(12)
    ]

    def rows(label: str, selected: list[Any], count: int) -> list[dict[str, str]]:
        return [
            {
                "fact_id": f.fact_id,
                "prompt": f"{label} {f.fact_id} variation {i}",
                "answer": f.object,
            }
            for f in selected
            for i in range(count)
        ]

    data = {
        "schema_version": 1,
        "model_revision": pilot.pilot_api().REVISION,
        "primary": rows("primary", facts, 8),
        "forget": rows("forget", facts[:4], 16),
        "retain_development": rows("retain", facts[4:], 4),
        "benign_retain": rows("benign", facts[4:], 4),
        "utility": [{"prompt": f"utility {i}", "answer": "fixture"} for i in range(16)],
        "provenance": "synthetic software fixture",
    }
    common = {
        "facts": facts,
        "declared_relations": ["relation"],
        "rules": [],
        "forbidden_derived": [],
        "required_kinds": [],
        "thresholds": {},
        "recognition_share_tolerance": 0.05,
        "mean_logprob_tolerance": 0.1,
        "abstentions": ["unknown"],
        "probes": [],
        "retain": [],
        "perplexity_texts": [],
    }
    qualification = SimpleNamespace(
        **common,
        capability_probes=[Probe(f"qualification utility {i}", "fixture") for i in range(8)],
    )
    final = SimpleNamespace(
        **common, capability_probes=[Probe(f"final utility {i}", "fixture") for i in range(8)]
    )
    pilot.validate_data(data, qualification, final)
    # The total retain count stays eight; only its required 4/4 composition changes.
    shifted = [
        replace(f, role="unrelated") if f.fact_id in {"f5", "f6", "f7"} else f for f in facts
    ]
    qualification.facts = shifted
    final.facts = shifted
    with pytest.raises(ValueError, match="four forget, four neighbor and four unrelated"):
        pilot.validate_data(data, qualification, final)


def test_execute_rejects_untouched_capability_failure_before_materialization(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Control-flow fixture, not model efficacy: real scorer and signed failure finalization."""
    from dataclasses import dataclass
    from types import SimpleNamespace

    from lacunai.core.spec import Probe, RuntimeConfig
    from lacunai.verify.evaluation import _suite

    @dataclass
    class Spec:
        target_id: str
        capability_probes: list[Probe]

    probes = [Probe(f"software fixture {i}", "42") for i in range(8)]
    spec = Spec("fixture", probes)
    runtime = SimpleNamespace(
        check=lambda: None,
        progress=lambda *a, **k: None,
        generate=lambda prompt: {
            "output": "43",
            "truncated": False,
            "input_tokens": 1,
            "output_tokens": 1,
            "token_ids": [1],
        },
    )
    untouched = {"capability": _suite(runtime, probes, "retain")}
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"num_hidden_layers":1}')
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    bundle = tmp_path / "hashes.json"
    bundle.write_text("{}")
    plan = {
        "source": str(source),
        "scratch": str(scratch),
        "workspace": str(tmp_path / "work"),
        "budget_root": str(tmp_path),
        "key_dir": str(tmp_path / "keys"),
        "bundle": {"path": str(bundle)},
    }

    class FakeBudget:
        def __init__(self, *args: Any) -> None:
            pass

        def check(self, *args: Any) -> None:
            pass

        def config(self) -> RuntimeConfig:
            return RuntimeConfig(max_seconds=None)

    monkeypatch.setattr(pilot, "pilot_api", lambda: SimpleNamespace(Budget=FakeBudget))
    monkeypatch.setattr(pilot, "inventory", lambda path: {"config.json": "fixture"})
    monkeypatch.setattr(
        pilot,
        "compatible_scratch",
        lambda *a: {
            "source_inventory": {"config.json": "fixture"},
            "scratch_inventory": {"config.json": "fixture"},
            "descriptor": {"file": "model.safetensors"},
        },
    )
    monkeypatch.setattr(pilot, "Storage", lambda *a: SimpleNamespace(shard_bytes=1))
    monkeypatch.setattr(pilot, "score", lambda *a: untouched)
    entered = []

    def prohibited_materialization(*args: Any, **kwargs: Any) -> Any:
        entered.append(True)
        raise RuntimeError("materialization reached before capability gate")

    original_import = pilot.import_module
    monkeypatch.setattr(
        pilot,
        "import_module",
        lambda name: (
            SimpleNamespace(safe_open=prohibited_materialization)
            if name == "safetensors"
            else SimpleNamespace(version=lambda name: "fixture")
            if name == "importlib.metadata"
            else original_import(name)
        ),
    )
    with pytest.raises(ValueError, match="untouched capability benchmark did not qualify"):
        pilot.execute(plan, spec, spec, {})
    assert entered == []
    summary = json.loads((tmp_path / "work/summary.json").read_text())
    assert summary["status"] == "failed" and summary["verdict"] == "inconclusive"
    assert (tmp_path / "work/report-evidence/signature.json").is_file()


def test_untouched_capability_gate_recomputes_raw_evidence() -> None:
    import copy
    from types import SimpleNamespace

    from lacunai.core.spec import Probe
    from lacunai.verify.evaluation import _suite

    probes = [Probe(f"fixture {i}", "42") for i in range(8)]
    runtime = SimpleNamespace(
        check=lambda: None,
        progress=lambda *a, **k: None,
        generate=lambda prompt: {
            "output": "42",
            "truncated": False,
            "input_tokens": 1,
            "output_tokens": 1,
            "token_ids": [1],
        },
    )
    spec = SimpleNamespace(capability_probes=probes)
    evidence = {"capability": _suite(runtime, probes, "retain")}
    pilot.qualify_untouched_capabilities(evidence, spec)
    for mutation in ("wrong_output", "missing_row", "wrong_prompt", "bad_tokens"):
        changed = copy.deepcopy(evidence)
        row = changed["capability"]["rows"][0]
        if mutation == "wrong_output":
            row["generation"]["output"] = "43"
        elif mutation == "missing_row":
            changed["capability"]["rows"].pop()
        elif mutation == "wrong_prompt":
            row["prompt"] = "different fixture"
        else:
            row["generation"]["token_ids"] = []
        with pytest.raises(ValueError):
            pilot.qualify_untouched_capabilities(changed, spec)


def test_small_plan_profile_is_explicit():
    assert pilot.plan_profile({}) == "connected-12"
    assert pilot.plan_profile({"profile": "connected-5"}) == "connected-5"
    for value in (None, True, "connected-12", "unknown", [], {}):
        with pytest.raises(ValueError, match="unknown explicit"):
            pilot.plan_profile({"profile": value})


def test_small_data_counts_and_mixed_profile_refusal():
    from copy import deepcopy
    from types import SimpleNamespace

    from lacunai.core.connected import Fact
    from lacunai.core.spec import Probe

    facts = [
        Fact(
            f"f{i}",
            f"entity{i}",
            "relation",
            f"answer{i}",
            "forget" if i == 0 else "neighbor" if i < 3 else "unrelated",
        )
        for i in range(5)
    ]

    def make(label, selected, count):
        return [
            {
                "fact_id": fact.fact_id,
                "prompt": f"{label}-{fact.fact_id}-{i}",
                "answer": fact.object,
            }
            for fact in selected
            for i in range(count)
        ]

    data = {
        "schema_version": 1,
        "model_revision": pilot.pilot_api().REVISION,
        "primary": make("primary", facts, 8),
        "forget": make("forget", facts[:1], 16),
        "retain_development": make("retain", facts[1:], 8),
        "benign_retain": make("benign", facts[1:], 8),
        "utility": [{"prompt": f"utility{i}", "answer": "utility"} for i in range(16)],
        "provenance": "synthetic software fixture",
    }
    common = dict(
        facts=facts,
        declared_relations=["relation"],
        rules=[],
        forbidden_derived=[],
        required_kinds=[],
        thresholds={},
        recognition_share_tolerance=0.05,
        mean_logprob_tolerance=0.1,
        abstentions=["unknown"],
        probes=[],
        retain=[],
        perplexity_texts=[],
    )
    qualification = SimpleNamespace(
        **common, capability_probes=[Probe(f"qual{i}", "42") for i in range(8)]
    )
    final = SimpleNamespace(
        **common, capability_probes=[Probe(f"final{i}", "42") for i in range(8)]
    )
    pilot.validate_data(data, qualification, final, profile="connected-5")
    with pytest.raises(ValueError, match="four forget"):
        pilot.validate_data(data, qualification, final)
    for key in ("primary", "forget", "retain_development", "benign_retain", "utility"):
        bad = deepcopy(data)
        bad[key].pop()
        with pytest.raises(ValueError):
            pilot.validate_data(bad, qualification, final, profile="connected-5")
    bad = deepcopy(data)
    # Keep the total at32 while removing coverage of one retained fact.
    bad["retain_development"][0]["fact_id"] = facts[-1].fact_id
    bad["retain_development"][0]["answer"] = facts[-1].object
    with pytest.raises(ValueError, match="unbalanced"):
        pilot.validate_data(bad, qualification, final, profile="connected-5")
