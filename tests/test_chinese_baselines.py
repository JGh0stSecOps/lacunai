"""Baseline orchestration contracts; no claimed model-efficacy evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lacunai.experimental import data, tensors

spec = importlib.util.spec_from_file_location(
    "baseline_controls", Path(__file__).parents[1] / "scripts/chinese_baselines.py"
)
assert spec and spec.loader
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def contract():
    return {
        "baseline": dict(baseline.SETTINGS),
        "runtime": {"seed": 20261102},
        "evaluation": {
            "primary_families": list(data.PRIMARY),
            "retain_families": list(data.RETAIN),
            "items_per_primary_family": 16,
            "items_per_retain_family": 16,
            "prompt_variants_per_item": 4,
        },
        "edits": {
            "allowed_tensors": list(baseline.NAMES),
            "expected_shape": [896, 4864],
            "max_tensor_relative_frobenius": ".02",
            "max_model_relative_frobenius": ".0025",
            "tokenizer_edits": False,
            "embedding_edits": False,
            "lm_head_edits": False,
            "generated_code": False,
        },
    }


def test_exact_historical_settings_and_corpus():
    c = contract()
    dataset = data.build_dataset(c)
    baseline.validate_training(c, dataset)
    assert tensors.grad_diff_objective(2.0, 3.0, alpha=1, gamma=1) == -1.0
    assert c["baseline"]["steps"] == 60
    assert c["edits"]["allowed_tensors"] == [
        f"model.layers.{i}.mlp.down_proj.weight" for i in range(16, 24)
    ]
    dataset["retain_train"][0]["answer"] = "different"
    with pytest.raises(ValueError, match="corpus differs"):
        baseline.validate_training(c, dataset)


@pytest.mark.parametrize(
    "key,value",
    [
        ("steps", 61),
        ("learning_rate", "0.002"),
        ("alpha", "2.0"),
        ("gamma", "0.0"),
        ("optimizer", "Adam"),
    ],
)
def test_training_settings_cannot_drift(key, value):
    c = contract()
    dataset = data.build_dataset(c)
    c["baseline"][key] = value
    with pytest.raises(ValueError, match="exact historical"):
        baseline.validate_training(c, dataset)


def test_allowlist_and_source_pins(tmp_path):
    c = contract()
    dataset = data.build_dataset(c)
    c["edits"]["allowed_tensors"] = ["lm_head.weight"]
    with pytest.raises(ValueError, match="allowlist"):
        baseline.validate_training(c, dataset)
    path = tmp_path / "frozen.json"
    path.write_text('{"fixture":true}')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert baseline.pinned_json(path, sha) == {"fixture": True}
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash differs"):
        baseline.pinned_json(path, sha)


@pytest.mark.parametrize("mode", ["graddiff", "retain_finetune"])
def test_matched_objective_calls_progress_and_actual_steps(mode, monkeypatch, capsys):
    c = contract()
    dataset = data.build_dataset(c)
    events = []
    calls = []
    runtime = SimpleNamespace(model=object(), tokenizer=object())
    watch = SimpleNamespace(
        check=lambda: None, progress=lambda phase, **kw: events.append((phase, kw))
    )
    reference = {"fixture": "reference"}

    def train(model, tokenizer, forget, retain, contract, actual_mode, progress, check, ref):
        assert '"state": "active"' in capsys.readouterr().err
        assert model is runtime.model and tokenizer is runtime.tokenizer and ref is reference
        calls.append((forget, retain, actual_mode, contract))
        trace = []
        for i in range(60):
            row = {"phase": actual_mode, "completed": i + 1, "total": 60, "unit": "steps"}
            progress(row)
            check()
            trace.append(row)
        return {"trace": trace, "mode": actual_mode}

    monkeypatch.setattr(baseline.tensors, "train", train)
    output = baseline.train_control(runtime, c, dataset, mode, watch, reference)
    assert len(output["trace"]) == 60 and len(events) == 60
    assert calls[0][0] == (dataset["forget_train"] if mode == "graddiff" else [])
    assert calls[0][1] is dataset["retain_train"]
    assert calls[0][3]["baseline"] == baseline.SETTINGS
    assert '"state": "complete"' in capsys.readouterr().err


def test_incomplete_training_is_explicit_failure(monkeypatch, capsys):
    monkeypatch.setattr(baseline.tensors, "train", lambda *a, **k: {"trace": []})
    with pytest.raises(ValueError, match="actual step trace"):
        baseline.train_control(
            SimpleNamespace(model=None, tokenizer=None),
            contract(),
            {"forget_train": [{}], "retain_train": [{}]},
            "graddiff",
            SimpleNamespace(check=lambda: None),
            {},
        )
    assert '"state": "failed"' in capsys.readouterr().err


def test_default_plan_no_model_or_inputs(monkeypatch, capsys):
    argv = ["baseline", "graddiff"]
    for flag in [
        "root",
        "source",
        "training-contract",
        "training-dataset",
        "dataset",
        "holdout-manifest",
        "before",
        "source-fingerprint",
        "training-contract-sha256",
        "training-dataset-sha256",
        "holdout-manifest-sha256",
        "before-sha256",
        "name",
    ]:
        argv.extend(["--" + flag, "absent"])
    monkeypatch.setattr(baseline.sys, "argv", argv)
    monkeypatch.setattr(baseline, "HFRuntime", lambda *a, **k: pytest.fail("unexpected model load"))
    assert baseline.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["state"] == "plan_only" and plan["overall_deadline"] is None


def test_runtime_matches_iteration_limits():
    c = baseline.RuntimeConfig(max_seconds=None, max_new_tokens=64, seed=42, torch_threads=2)
    assert c.max_seconds is None and c.device == "cpu" and c.dtype == "float32"
    assert c.max_new_tokens == 64 and c.torch_threads == 2
    assert baseline.iteration.LIMITS["disk_max_bytes"] == 10 * 1024**3


def test_before_result_manifest_source_binding(tmp_path):
    run = tmp_path / "untouched"
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True)
    before = evaluation / "result.json"
    before.write_text("{}")
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps({"checkpoint_fingerprint": "original"}))
    binding = baseline.verify_before_source(before, "original")
    assert binding["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="fingerprint differs"):
        baseline.verify_before_source(before, "candidate")
    (evaluation / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="unambiguous"):
        baseline.verify_before_source(before, "original")


def test_common_manifest_attestor_fields():
    hashes = {"scripts/chinese_baselines.py": "a" * 64, "scripts/chinese_iteration.py": "b" * 64}
    fields = baseline.common_manifest_fields("original", hashes, {"manifest_sha256": "c" * 64})
    assert fields["checkpoint_fingerprint"] == "original"
    assert fields["code_hashes"] == hashes
    assert fields["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert (
        fields["model_revision"]
        == fields["tokenizer_revision"]
        == "7ae557604adf67be50417f59c2c2f167def9a775"
    )
    assert fields["evaluation_contract"] == baseline.iteration.evaluation.CONTRACT
    assert fields["editor"] == "current Codex conversation"
    assert fields["independent_holdout_commitment"]["manifest_sha256"] == "c" * 64
