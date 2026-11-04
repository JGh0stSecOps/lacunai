"""Bounded orchestration fixtures, not model efficacy evidence."""

import importlib.util
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).parents[1] / "scripts/pretrained_localized_pilot.py"
spec = importlib.util.spec_from_file_location("localized_test", PATH)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def diagnosis():
    from lacunai.map.localize import effect_record

    pair = effect_record(-1, -4, -2)
    row = {"raw_pairs": [pair], "mean_effect": 2.0}
    return {
        "kind": "localization",
        "checkpoint_fingerprint": "source",
        "selected": [{"layer": 21, "neurons": [4090, 1507, 535]}],
        "layers": [
            {
                "layer": 21,
                **deepcopy(row),
                "neurons": [{"neuron": n, **deepcopy(row)} for n in (4090, 1507, 535)],
            }
        ],
    }


def test_raw_positive_evidence_and_scope_required():
    value = diagnosis()
    runner.positive_selection(value, "source")
    for kind in ("source", "scope", "negative", "arithmetic"):
        broken = deepcopy(value)
        if kind == "source":
            broken["checkpoint_fingerprint"] = "other"
        elif kind == "scope":
            broken["selected"][0]["neurons"].append(3)
        elif kind == "negative":
            broken["layers"][0]["raw_pairs"][0].update(
                patched_logprob=-5, effect=-1, normalized_effect=-1 / 3
            )
        else:
            broken["layers"][0]["mean_effect"] = 3
        with pytest.raises(ValueError):
            runner.positive_selection(broken, "source")


def test_dryrun_never_executes(monkeypatch, capsys):
    monkeypatch.setattr(
        runner.sys, "argv", [str(PATH), "--plan", "fixture", "--plan-sha256", "pin"]
    )
    monkeypatch.setattr(runner, "prepare", lambda *a, **k: (None,) * 5)
    monkeypatch.setattr(runner, "execute", lambda *a: pytest.fail("executed"))
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out)["model_loaded"] is False


@pytest.mark.parametrize("fail,unqualified", [(False, False), (True, False), (False, True)])
def test_actual_execute_nested_tensor_freeze_and_restore(tmp_path, monkeypatch, fail, unqualified):
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    from lacunai.core.model import RuntimeConfig
    from lacunai.restore import checkpoints

    source, scratch, workspace = (tmp_path / k for k in ("source", "scratch", "run"))
    for path in (source, scratch):
        path.mkdir()
        (path / "config.json").write_text('{"num_hidden_layers":24}')
        safe.save_file(
            {
                name: torch.full((2, 2), 2.0 if path == scratch and name == runner.OUTER else 1.0)
                for name in (runner.OUTER, *runner.METHODS.values())
            },
            str(path / "model.safetensors"),
        )
    before_source = runner.shared.inventory(source)
    before_scratch = runner.shared.inventory(scratch)
    bundle = tmp_path / "hashes.json"
    bundle.write_text("{}")
    plan = {
        "source": str(source),
        "scratch": str(scratch),
        "workspace": str(workspace),
        "budget_root": str(tmp_path),
        "key_dir": str(tmp_path / "keys"),
        "bundle": {"path": str(bundle)},
    }

    class Budget:
        def __init__(self, *a):
            pass

        def config(self):
            return RuntimeConfig(max_seconds=None)

        def check(self, *a):
            pass

    actual_pilot = runner.shared.pilot_api()
    actual_pilot.Budget = Budget
    actual_pilot.benign_relearning = lambda *a: {"evaluation": {"target": {"rows": []}}}
    actual_pilot.generation_identity = lambda value: value["identity"]
    monkeypatch.setattr(runner.shared, "pilot_api", lambda: actual_pilot)
    monkeypatch.setattr(
        runner.shared,
        "pretrained_api",
        lambda: SimpleNamespace(
            qualify=lambda evaluation, spec: (
                (_ for _ in ()).throw(ValueError("unqualified final"))
                if unqualified and spec is f
                else None
            ),
            alias_comparison=lambda result, *a: result,
        ),
    )

    @dataclass
    class Spec:
        target_id: str = "fixture"
        digest: str = "fixture-digest"

    q, f = Spec(), Spec()
    final_calls = []

    def score(path, spec, budget, output):
        if output.name == "untouched-final.json":
            freeze = json.loads((workspace / "candidate-freeze.json").read_text())
            assert len(freeze["candidates"]) == 2
            final_calls.append(True)
            if fail:
                raise RuntimeError("injected final evaluation failure")
        result = {"identity": "fixture", "perplexity": {"perplexity": "1", "tokens": 1}}
        runner.atomic_json(output, result)
        return result

    monkeypatch.setattr(runner.shared, "score", score)
    monkeypatch.setattr(runner, "compare", lambda *a: {"verdict": "failed", "reasons": []})

    class Store:
        def __init__(self, path):
            self.root = path
            self.rows = []

        def list(self):
            return self.rows

        def path(self, run_id):
            return self.root / run_id

    class Service:
        def __init__(self, path, *a):
            self.store = Store(path)

        def erase(self, model, target, method, **kwargs):
            assert runner.shared.inventory(model) == before_source
            name = runner.METHODS[method]
            run_id = method
            directory = self.store.path(run_id)
            directory.mkdir(parents=True)
            snapshot = checkpoints.prepare_snapshot(model, directory, [name])
            self.store.rows.append({"run_id": run_id, "operation": "erase"})
            checkpoints.write_edited_checkpoint(
                model, model, {name: torch.full((2, 2), 1.1)}, snapshot, in_place=True
            )
            return {"run_id": run_id, "verdict": "failed"}

        def restore(self, run_id, **kwargs):
            return checkpoints.restore_snapshot(self.store.path(run_id), apply=True)

    monkeypatch.setattr(runner, "BoundService", Service)
    monkeypatch.setattr(
        runner.shared,
        "finalize_evidence",
        lambda root, summary, *a: runner.atomic_json(root / "summary.json", summary),
    )
    try:
        result = runner.execute(
            plan, q, f, {"target_aliases": ["fixture"], "benign_retain": []}, diagnosis()
        )
        assert not fail and result["verdict"] == ("inconclusive" if unqualified else "failed")
    except RuntimeError as exc:
        assert fail and str(exc) == "injected final evaluation failure"
    assert final_calls == [True]
    assert runner.shared.inventory(source) == before_source
    assert runner.shared.inventory(scratch) == before_scratch
    summary = json.loads((workspace / "summary.json").read_text())
    assert summary["old_scratch_restored_exact"] is True
    for journal in workspace.rglob("transaction-journal.json"):
        assert json.loads(journal.read_text())["stack"] == []


def test_global_ledger_counts_all_scope_files(tmp_path):
    calls = []
    base = SimpleNamespace(check=lambda reserve=0: calls.append(reserve))
    (tmp_path / "other-scope").mkdir()
    (tmp_path / "other-scope" / "bytes").write_bytes(b"12345")
    budget = runner.LedgerBudget(base, tmp_path, 10)
    budget.check(5)
    with pytest.raises(OSError, match="ledger"):
        budget.check(6)
    storage = object.__new__(runner.ScopedStorage)
    storage.root = tmp_path
    storage.budget = budget
    storage.shard_bytes = 100
    storage.reserve()
    assert calls == [5, 0, 105]


def test_transaction_count_bound(tmp_path):
    budget = runner.LedgerBudget(SimpleNamespace(check=lambda *a: None), tmp_path, 1024)
    for index in range(6):
        path = tmp_path / str(index)
        path.mkdir()
        (path / "snapshot.json").write_text("{}")
    with pytest.raises(OSError, match="transaction count"):
        budget.check()


def test_recovery_rejects_unknown_scoped_snapshot(tmp_path, monkeypatch):
    workspace = tmp_path / "run"
    scratch = tmp_path / "scratch"
    path = workspace / "outer" / "transactions" / "original"
    path.mkdir(parents=True)
    (workspace / "outer" / "transaction-journal.json").write_text(
        json.dumps({"scratch": str(scratch), "order": "LIFO", "stack": [str(path)]})
    )
    (path / "snapshot.json").write_text(
        json.dumps(
            {
                "source": "wrong",
                "output": str(scratch),
                "tensors": {runner.OUTER: {}},
                "status": "edited",
            }
        )
    )
    monkeypatch.setattr(
        runner.shared, "pilot_api", lambda: SimpleNamespace(Budget=lambda *a: object())
    )
    monkeypatch.setattr(
        runner.shared.checkpoints, "restore_snapshot", lambda *a, **k: pytest.fail("unsafe restore")
    )
    with pytest.raises(ValueError, match="source/tensor"):
        runner.recover(
            {"workspace": str(workspace), "scratch": str(scratch), "budget_root": str(tmp_path)}
        )


def test_actual_interrupted_child_then_outer_recovery(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    from lacunai.restore import checkpoints

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "config.json").write_text('{"num_hidden_layers":24}')
    safe.save_file(
        {name: torch.full((2, 2), 2.0) for name in (runner.OUTER, *runner.METHODS.values())},
        str(scratch / "model.safetensors"),
    )
    old = runner.shared.inventory(scratch)
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(old))
    workspace = tmp_path / "run"
    scopes = [
        (workspace / "outer", runner.OUTER, 1.0),
        (
            workspace / "ablate-neurons" / "final-transactions",
            runner.METHODS["ablate-neurons"],
            1.1,
        ),
    ]
    for scope, name, value in scopes:
        directory = scope / "transactions" / "owned"
        snapshot = checkpoints.prepare_snapshot(scratch, directory, [name])
        checkpoints.write_edited_checkpoint(
            scratch, scratch, {name: torch.full((2, 2), value)}, snapshot, in_place=True
        )
        (scope / "transaction-journal.json").write_text(
            json.dumps({"scratch": str(scratch), "order": "LIFO", "stack": [str(directory)]})
        )
    monkeypatch.setattr(
        runner.shared,
        "pilot_api",
        lambda: SimpleNamespace(Budget=lambda *a: SimpleNamespace(check=lambda *a: None)),
    )
    result = runner.recover(
        {
            "workspace": str(workspace),
            "scratch": str(scratch),
            "budget_root": str(tmp_path),
            "scratch_inventory": {"path": str(expected)},
        }
    )
    assert result["old_scratch_restored_exact"] is True
    assert len(result["restored"]) == 2
    assert runner.shared.inventory(scratch) == old
    assert all(
        json.loads(path.read_text())["stack"] == []
        for path in workspace.rglob("transaction-journal.json")
    )


def test_recovery_reserves_remaining_records_not_written_records(tmp_path, monkeypatch):
    workspace = tmp_path / "run"
    scratch = tmp_path / "scratch"
    directory = workspace / "outer" / "transactions" / "original"
    directory.mkdir(parents=True)
    expected = {"model.safetensors": "fixture-hash"}
    inventory = tmp_path / "expected.json"
    inventory.write_text(json.dumps(expected))
    (directory / "snapshot.json").write_text(
        json.dumps(
            {
                "source": str(scratch),
                "output": str(scratch),
                "tensors": {runner.OUTER: {}},
                "status": "prepared",
                "before_files": {"model.safetensors": {"sha256": "fixture-hash"}},
            }
        )
    )
    (workspace / "outer" / "transaction-journal.json").write_text(
        json.dumps({"scratch": str(scratch), "order": "LIFO", "stack": [str(directory)]})
    )
    (workspace / "existing-records").write_bytes(b"x" * 8192)
    used = sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file())
    allowance = used + 128
    shard = 988097824
    reservations = []

    def check(reserve=0):
        reservations.append(reserve)
        if reserve > shard + 128:
            raise OSError("aggregate cap would be exceeded")

    monkeypatch.setattr(
        runner, "ledger", lambda _: {"worst_record_bytes": allowance, "staged_shard_bytes": shard}
    )
    monkeypatch.setattr(runner.shared, "inventory", lambda _: expected)
    monkeypatch.setattr(
        runner.shared,
        "pilot_api",
        lambda: SimpleNamespace(Budget=lambda *a: SimpleNamespace(check=check)),
    )
    result = runner.recover(
        {
            "workspace": str(workspace),
            "scratch": str(scratch),
            "budget_root": str(tmp_path),
            "scratch_inventory": {"path": str(inventory)},
        }
    )
    assert result["old_scratch_restored_exact"] is True
    assert reservations == [shard + 128]
