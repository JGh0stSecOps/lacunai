"""Final-control software contracts; fixtures do not establish model unlearning."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from lacunai.core.spec import RuntimeConfig
from lacunai.experimental import data
from lacunai.restore import checkpoints

spec = importlib.util.spec_from_file_location(
    "iteration_controls", Path(__file__).parents[1] / "scripts/chinese_iteration.py"
)
assert spec and spec.loader
iteration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iteration)


@pytest.mark.parametrize("development,expected", [(True, "development\nA"), (False, "withheld\nB")])
def test_final_perplexity_uses_withheld_corpus(tmp_path, monkeypatch, development, expected):
    dataset = {
        "dev_retain": [{"prompts": ["development"], "answer": "A"}],
        "heldout_retain": [{"prompts": ["withheld"], "answer": "B"}],
    }
    observed = []

    def evaluate(*args, **kwargs):
        args[4].mkdir()
        return {"development": development}

    monkeypatch.setattr(iteration.evaluation, "evaluate", evaluate)
    runtime = SimpleNamespace(
        model=None,
        tokenizer=None,
        contract={},
        perplexity=lambda texts: observed.extend(texts) or {"tokens": 1, "perplexity": 2.0},
    )
    result = iteration.score(runtime, dataset, tmp_path / "evaluation", None, development)
    assert observed == [expected]
    assert result["perplexity"]["corpus_sha256"] == iteration.digest([expected])


def test_snapshot_must_match_loaded_source_bytes():
    runtime = SimpleNamespace(inventory={"model.safetensors": "loaded-a"})
    iteration.bind_snapshot(
        runtime, {"before_files": {"model.safetensors": {"sha256": "loaded-a"}}}
    )
    with pytest.raises(ValueError, match="snapshot source differs"):
        iteration.bind_snapshot(
            runtime, {"before_files": {"model.safetensors": {"sha256": "changed-b"}}}
        )


@pytest.mark.parametrize(
    "output,truncated,verdict", [("cat", False, "failed"), ("noise", True, "inconclusive")]
)
def test_collateral_preflight_rejects_before_full_suite(tmp_path, output, truncated, verdict):
    item = data._retain_item("dev_retain", "english_comprehension_neighbors", 0, (0, 1, 0))
    row = {
        "item_id": item["id"],
        "family": item["family"],
        "variant": 0,
        "prompt": item["prompts"][0],
        "output": "dog",
        "truncated": False,
        "token_ids": [1],
        "output_tokens": 1,
        "input_tokens": 2,
        "first_step_logits_sha256": "a" * 64,
    }
    before = {
        "development": True,
        "retain": {"spec": [item], "raw": [row]},
        "perplexity": {"tokens": 1, "perplexity": 2.0},
    }
    events = []
    runtime = SimpleNamespace(
        generate=lambda _: dict(row, output=output, truncated=truncated),
        perplexity=lambda _: {"tokens": 1, "perplexity": 2.0},
    )
    watch = SimpleNamespace(check=lambda: None, progress=lambda *a, **k: events.append((a, k)))
    result = iteration.collateral_preflight(
        runtime, {"dev_retain": [item]}, before, tmp_path / "screen", watch
    )
    assert result["continue"] is False
    assert result["efficacy_pass"] is False
    assert result["verdict"] == verdict
    assert json.loads((tmp_path / "screen/raw.jsonl").read_text())["output"] == output
    assert [event[1]["completed"] for event in events] == [0, 1]


def test_collateral_preflight_cannot_use_other_baseline(tmp_path):
    with pytest.raises(ValueError, match="matching development"):
        iteration.collateral_preflight(None, {"dev_retain": []}, {}, tmp_path, None)


def recovery_fixture():
    items = [
        data._primary_item("dev_primary", family, 0, (0, 0, 0), (1, 1, 1))
        for family in data.PRIMARY
    ]
    rows = [
        {
            "item_id": item["id"],
            "family": item["family"],
            "variant": number,
            "prompt": prompt,
            "output": item["answer"],
            "truncated": False,
            "token_ids": [1],
            "input_tokens": 2,
            "output_tokens": 1,
            "first_step_logits_sha256": "a" * 64,
        }
        for item in items
        for number, prompt in enumerate(item["prompts"])
    ]
    return {"dev_primary": items}, {"development": True, "primary": {"spec": items, "raw": rows}}


def test_one_confirmed_recovery_rejects_zero_recovery_contract(tmp_path):
    dataset, before = recovery_fixture()
    events = []
    runtime = SimpleNamespace(generate=lambda _: {"output": "red", "truncated": False})
    watch = SimpleNamespace(check=lambda: None, progress=lambda *a, **k: events.append(k))
    result = iteration.recovery_preflight(runtime, dataset, before, tmp_path / "probe", watch)
    assert result["continue"] is False and result["verdict"] == "failed"
    assert result["efficacy_pass"] is False
    assert len(result["observations"]) == 1
    assert [event["completed"] for event in events] == [0, 1]


def test_preflight_no_recovery_is_never_a_pass(tmp_path):
    dataset, before = recovery_fixture()
    runtime = SimpleNamespace(generate=lambda _: {"output": "unable to answer", "truncated": False})
    watch = SimpleNamespace(check=lambda: None, progress=lambda *a, **k: None)
    result = iteration.recovery_preflight(runtime, dataset, before, tmp_path / "probe", watch)
    assert result["continue"] is True and result["efficacy_pass"] is False
    assert len(result["observations"]) == 4
    assert "verdict" not in result


def test_holdout_pin_is_required_and_checked(tmp_path):
    dataset = tmp_path / "fresh.json"
    dataset.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "independent_holdout_commitment": "external designer",
                "dataset_bytes_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        )
    )
    pin = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="independently pinned"):
        iteration.verified_holdout(dataset, None, None)
    assert iteration.verified_holdout(dataset, manifest, pin)["manifest_sha256"] == pin
    dataset.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="dataset hash"):
        iteration.verified_holdout(dataset, manifest, pin)
    with pytest.raises(ValueError, match="manifest hash"):
        iteration.verified_holdout(dataset, manifest, "0" * 64)


def result():
    row = {
        "item_id": "fixture",
        "variant": 0,
        "token_ids": [1, 2],
        "first_step_logits_sha256": "a" * 64,
        "output": "fixture",
    }
    return {
        "dataset_sha256": "b" * 64,
        "inference_sha256": "c" * 64,
        "primary": {"raw": [row]},
        "retain": {"raw": [dict(row, item_id="retain")]},
        "receptive": {"raw": [{"score": "-1"}]},
        "perplexity": {"tokens": 3, "perplexity": "2"},
    }


@pytest.mark.parametrize("field", ["token_ids", "first_step_logits_sha256", "output"])
def test_noop_observations_must_match(field):
    before = result()
    after = copy.deepcopy(before)
    assert iteration.no_op_identity(before, after)["deterministic_equal"]
    after["primary"]["raw"][0][field] = "changed"
    with pytest.raises(ValueError, match="deterministic observation"):
        iteration.no_op_identity(before, after)


def test_noop_missing_or_noncomparable_rejected():
    before = result()
    after = copy.deepcopy(before)
    after["inference_sha256"] = "different"
    with pytest.raises(ValueError, match="inference"):
        iteration.no_op_identity(before, after)
    before["primary"]["raw"] = []
    with pytest.raises(ValueError, match="count"):
        iteration.no_op_identity(before, before)


def checkpoint_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen2"}')
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    raw = struct.pack("<Q", len(header)) + header + struct.pack("<ff", -0.0, 1.25)
    (source / "model.safetensors").write_bytes(raw)
    run = tmp_path / "run"
    candidate = tmp_path / "candidate"
    snapshot = checkpoints.prepare_snapshot(source, run, ["w"])
    # Substitute only optional Torch conversion; real IO/hash/snapshot/undo code runs.
    monkeypatch.setattr(checkpoints, "_replacement_bytes", lambda value, descriptor: value)
    checkpoints.write_edited_checkpoint(
        source, candidate, {"w": struct.pack("<ff", 3, 4)}, snapshot
    )
    return source, run, candidate


def test_disk_restore_copy_preserves_original_and_candidate(tmp_path, monkeypatch):
    source, run, candidate = checkpoint_fixture(tmp_path, monkeypatch)
    original = (source / "model.safetensors").read_bytes()
    edited = (candidate / "model.safetensors").read_bytes()
    receipt = (run / "snapshot.json").read_bytes()
    events = []
    watch = SimpleNamespace(
        check=lambda: None, progress=lambda phase, **fields: events.append((phase, fields))
    )
    proof = tmp_path / "proof"
    proof.mkdir()
    restored = tmp_path / "restored"
    report = iteration.disk_restore_proof(run, proof, restored, RuntimeConfig(), watch)
    assert report["restored_exactly"] and report["candidate_preserved"]
    assert (restored / "model.safetensors").read_bytes() == original
    assert (candidate / "model.safetensors").read_bytes() == edited
    assert (source / "model.safetensors").read_bytes() == original
    assert (run / "snapshot.json").read_bytes() == receipt
    assert events[0][1]["completed"] == 0
    assert any(e["completed"] == e["total"] for _, e in events)


def test_restore_copy_refuses_altered_candidate(tmp_path, monkeypatch, capsys):
    _, run, candidate = checkpoint_fixture(tmp_path, monkeypatch)
    (candidate / "config.json").write_text("{}")
    proof = tmp_path / "proof"
    proof.mkdir()
    with pytest.raises(ValueError, match="candidate differs"):
        iteration.disk_restore_proof(
            run, proof, tmp_path / "restored", RuntimeConfig(), SimpleNamespace()
        )
    assert not (tmp_path / "restored").exists()
    assert '"state": "failed"' in capsys.readouterr().err


def test_relearning_rejects_changes_and_uses_exact_corpus(tmp_path):
    contract = {
        "relearning": {"steps_per_arm": 20, "learning_rate": "0.0001"},
        "baseline": {
            "optimizer": "SGD",
            "sequence_tokens": 128,
            "alpha": "1.0",
            "gradient_clip": "1.0",
            "momentum": 0,
        },
        "edits": {
            "allowed_tensors": [f"model.layers.{i}.mlp.down_proj.weight" for i in range(16, 24)],
            "expected_shape": [896, 4864],
            "max_tensor_relative_frobenius": "0.02",
            "max_model_relative_frobenius": "0.0025",
        },
    }
    cp = tmp_path / "contract.json"
    dp = tmp_path / "data.json"
    cp.write_text(json.dumps(contract))
    dp.write_text(json.dumps({"relearn": data._relearn()}))

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    parsed, corpus = iteration.relearning_inputs(cp, sha(cp), dp, sha(dp))
    assert parsed == contract and corpus == data._relearn()
    contract["relearning"]["steps_per_arm"] = 21
    cp.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="20-step"):
        iteration.relearning_inputs(cp, sha(cp), dp, sha(dp))


def test_cli_dryrun_does_not_load_model(monkeypatch, capsys):
    monkeypatch.setattr(
        iteration.sys,
        "argv",
        [
            "controls",
            "no-op",
            "--root",
            "/absent",
            "--source",
            "/absent",
            "--dataset",
            "/absent",
            "--name",
            "test",
        ],
    )
    monkeypatch.setattr(
        iteration, "HFRuntime", lambda *a, **k: pytest.fail("model load in dry run")
    )
    assert iteration.main() == 0
    assert json.loads(capsys.readouterr().out)["state"] == "plan_only"


def test_restore_proof_refuses_descendant_output(tmp_path, monkeypatch):
    _, run, candidate = checkpoint_fixture(tmp_path, monkeypatch)
    proof = tmp_path / "proof"
    proof.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        iteration.disk_restore_proof(
            run, proof, candidate / "nested", RuntimeConfig(), SimpleNamespace()
        )
    assert not (candidate / "nested").exists()
