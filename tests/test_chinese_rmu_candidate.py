"""RMU runner/software contracts; synthetic fixtures are not model efficacy proof."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lacunai.experimental import data

spec = importlib.util.spec_from_file_location(
    "rmu_runner", Path(__file__).parents[1] / "scripts/chinese_rmu_candidate.py"
)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def historical():
    return {
        "baseline": dict(runner.baseline.SETTINGS),
        "runtime": {"seed": 20261102},
        "evaluation": {
            "primary_families": list(data.PRIMARY),
            "retain_families": list(data.RETAIN),
            "items_per_primary_family": 16,
            "items_per_retain_family": 16,
            "prompt_variants_per_item": 4,
        },
        "edits": {
            "allowed_tensors": list(runner.baseline.NAMES),
            "expected_shape": [896, 4864],
            "max_tensor_relative_frobenius": "0.02",
            "max_model_relative_frobenius": "0.0025",
            "tokenizer_edits": False,
            "embedding_edits": False,
            "lm_head_edits": False,
            "generated_code": False,
        },
    }


@pytest.mark.parametrize("mode", ["rmu", "retain_mse"])
def test_contract_reuses_historical_validation_and_exact_allowlist(mode):
    history = historical()
    unchanged = copy.deepcopy(history)
    dataset = data.build_dataset(history)
    inner, outer = runner.candidate_contract({**runner.SPEC, "mode": mode}, history, dataset)
    assert history == unchanged
    assert inner["edits"]["allowed_tensors"] == outer["edits"]["allowed_tensors"] == runner.NAMES
    assert inner["edits"]["max_tensor_relative_frobenius"] == "0.0195"
    assert outer["edits"]["max_tensor_relative_frobenius"] == "0.02"
    assert inner["baseline"] == history["baseline"]  # backend uses explicit AdamW spec instead
    bad = copy.deepcopy(history)
    bad["baseline"]["steps"] = 59
    with pytest.raises(ValueError, match="historical"):
        runner.candidate_contract(runner.SPEC, bad, dataset)


@pytest.mark.parametrize(
    "field,value",
    [
        ("allowed_tensors", runner.baseline.NAMES),
        ("allowed_tensors", runner.NAMES[:2]),
        ("allowed_tensors", ["model.embed_tokens.weight"]),
        ("mode", "graddiff"),
        ("steps", 59),
        ("learning_rate", "0.01"),
        ("seed", True),
        ("optimizer", "SGD"),
        ("hook_module", "model.layers.8"),
        ("betas", [0.9, 0.998]),
        ("alpha", "2"),
        ("extra", "code"),
    ],
)
def test_root_contract_refuses_altered_fields(field, value):
    h = historical()
    with pytest.raises(ValueError, match="root-selected"):
        runner.candidate_contract({**runner.SPEC, field: value}, h, data.build_dataset(h))


def observations():
    items = [
        data._primary_item("dev_primary", family, 0, (4, 0, 4), (0, 1, 0))
        for family in runner.evaluation.CONTRACT["required_primary_families"]
    ]
    rows = []
    for item in items:
        for n, prompt in enumerate(item["prompts"]):
            rows.append(
                {
                    "item_id": item["id"],
                    "family": item["family"],
                    "variant": n,
                    "prompt": prompt,
                    "output": item["answer"],
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "token_ids": [1],
                    "first_step_logits_sha256": "a" * 64,
                    "truncated": False,
                }
            )
    return items, {"development": True, "primary": {"spec": items, "raw": rows}}


def test_all_four_recovery_families_scored_after_first_recovery(tmp_path):
    items, before = observations()
    events = []
    prompts = []

    def generate(prompt):
        assert events[-1]["completed"] == len(prompts)
        prompts.append(prompt)
        return {"output": items[len(prompts) - 1]["answer"], "truncated": False, "output_tokens": 1}

    runtime = SimpleNamespace(generate=generate)
    watch = SimpleNamespace(
        check=lambda: None, progress=lambda phase, **event: events.append(event)
    )
    result = runner.recovery_preflight(
        runtime, {"dev_primary": items}, before, tmp_path / "recovery", watch
    )
    assert len(prompts) == 4 and len(result["observations"]) == 4
    assert all(row["score"]["correct"] for row in result["observations"])
    assert result["continue"] is False and result["verdict"] == "failed"
    assert result["protocol"] == runner.RECOVERY_PROTOCOL and result["efficacy_pass"] is False
    saved = [
        json.loads(line) for line in (tmp_path / "recovery/raw.jsonl").read_text().splitlines()
    ]
    assert saved == result["observations"]
    assert [e["completed"] for e in events] == [0, 1, 1, 2, 2, 3, 3, 4]


def test_missing_baseline_family_refused_before_generation(tmp_path):
    items, before = observations()
    before["primary"]["raw"] = [{**r, "output": ""} for r in before["primary"]["raw"]]
    with pytest.raises(ValueError, match="no baseline-confirmed"):
        runner.recovery_preflight(None, {"dev_primary": items}, before, tmp_path / "absent", None)
    assert not (tmp_path / "absent").exists()


def test_ambiguous_four_responses_are_not_early_pass(tmp_path):
    items, before = observations()
    runtime = SimpleNamespace(generate=lambda _: {"output": "", "truncated": True})
    watch = SimpleNamespace(check=lambda: None, progress=lambda *a, **k: None)
    result = runner.recovery_preflight(
        runtime, {"dev_primary": items}, before, tmp_path / "recovery", watch
    )
    assert len(result["observations"]) == 4
    assert result["continue"] and not result["efficacy_pass"]
    assert all(r["score"]["ambiguous"] for r in result["observations"])


def test_backend_dispatch_rejects_incomplete_trace(monkeypatch):
    def train(*args):
        assert args[4] == runner.SPEC
        return {"mode": "rmu", "trace": [{"completed": 1, "total": 60}]}

    monkeypatch.setattr(runner, "import_module", lambda name: SimpleNamespace(train=train))
    with pytest.raises(ValueError, match="step trace"):
        runner.train_candidate(None, {}, {}, {}, runner.SPEC, None)


@pytest.mark.parametrize("phase", ["screen", "persist"])
def test_cli_default_does_not_load_model(tmp_path, monkeypatch, capsys, phase):
    args = ["runner", phase, "--name", "dry", "--source-fingerprint", "a" * 64]
    for flag in ("root", "source", "training-contract", "training-dataset", "edit-spec"):
        args += ["--" + flag, str(tmp_path / flag)]
    for flag in ("training-contract", "training-dataset", "edit-spec"):
        args += ["--" + flag + "-sha256", "b" * 64]
    monkeypatch.setattr("sys.argv", args)
    monkeypatch.setattr(runner, "HFRuntime", lambda *_: pytest.fail("dry-run loaded model"))
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out)["state"] == "plan_only"
    assert not (tmp_path / "root").exists()


@pytest.mark.parametrize("mode", ["rmu", "retain_mse", "rmu_context", "retain_context_mse"])
def test_real_backend_accepts_runner_contract(mode):
    from lacunai.erase import representation_misdirection

    history = historical()
    template = runner.CONTEXT_SPEC if "context" in mode else runner.SPEC
    spec = {**template, "mode": mode}
    inner, _ = runner.candidate_contract(spec, history, data.build_dataset(history))
    representation_misdirection.validate_spec(spec, inner)
    broken = copy.deepcopy(inner)
    broken["edits"]["allowed_tensors"] = runner.baseline.NAMES
    with pytest.raises(ValueError, match="projection bounds/shape"):
        representation_misdirection.validate_spec(spec, broken)


@pytest.mark.parametrize("mode", ["rmu_context", "retain_context_mse"])
def test_context_contract_has_distinct_exact_schema(mode):
    history = historical()
    selected = {**runner.CONTEXT_SPEC, "mode": mode}
    inner, outer = runner.candidate_contract(selected, history, data.build_dataset(history))
    assert inner["edits"]["allowed_tensors"] == outer["edits"]["allowed_tensors"] == runner.NAMES
    assert selected["mask_policy"] == (
        "user_content_and_answer_for_reading;answer_for_writing;"
        "user_content_and_answer_for_retain:v1"
    )
    assert selected["variant_policy"] == "reading_cycle_existing_variants:v1"
    runner.validate_spec(runner.SPEC)


@pytest.mark.parametrize(
    "change",
    [
        {"mask_policy": "answer_only"},
        {"variant_policy": "new_prompts"},
        {"mask_policy": True},
        {"mode": "rmu"},
        {"method": "unknown"},
        {"allowed_tensors": runner.baseline.NAMES},
        {"steps": 61},
        {"learning_rate": "0.001"},
        {"seed": True},
        {"extra": "code"},
    ],
)
def test_context_strict_rejection(change):
    with pytest.raises(ValueError, match="root-selected"):
        runner.validate_spec({**runner.CONTEXT_SPEC, **change})


def test_legacy_rejects_context_fields_and_context_requires_them():
    for field in ("mask_policy", "variant_policy"):
        with pytest.raises(ValueError, match="root-selected"):
            runner.validate_spec({**runner.SPEC, field: runner.CONTEXT_SPEC[field]})
        reduced = {k: v for k, v in runner.CONTEXT_SPEC.items() if k != field}
        with pytest.raises(ValueError, match="root-selected"):
            runner.validate_spec(reduced)


@pytest.mark.parametrize("mode", ["rmu_context", "retain_context_mse"])
def test_context_dispatch_preserves_backend_trace(mode, monkeypatch):
    selected = {**runner.CONTEXT_SPEC, "mode": mode}
    trace = [
        {"completed": i, "total": 60, "retain_variant": i % 4, "retain_mask": {"fixture": True}}
        for i in range(1, 61)
    ]
    result = {"mode": mode, "trace": trace}

    def train(*args):
        assert args[4] == selected
        return result

    monkeypatch.setattr(runner, "import_module", lambda name: SimpleNamespace(train=train))
    assert runner.train_candidate(None, {}, {}, {}, selected, None) == result
