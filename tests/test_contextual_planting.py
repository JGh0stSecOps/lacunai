"""Contextual planting contracts; numerical tests use tiny random Qwen, no downloads."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "contextual_planting", Path(__file__).resolve().parents[1] / "scripts/contextual_planting.py"
)
assert SPEC and SPEC.loader
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def data():
    return {
        key: [{"prompt": f"prompt {i}", "answer": "42"} for i in range(count)]
        for key, count in (("development", 64), ("retain_development", 32))
    }


@pytest.mark.parametrize("steps", [0, 257, True, 256.0, "256"])
def test_invalid_steps(steps):
    with pytest.raises(ValueError, match="step/mode"):
        helper.validate_inputs(list(range(128)), data(), steps, False)


def test_profiles_and_data():
    assert helper.PROFILES == {128: 0.002, 512: 0.001, 4864: 0.0003}
    for width in helper.PROFILES:
        helper.validate_inputs(list(range(width)), data(), 256, False)
    helper.validate_inputs(list(range(128)), data(), 64, True)
    with pytest.raises(ValueError):
        helper.validate_inputs(list(range(128)), data(), 64, False)
    bad = data()
    bad["development"].pop()
    with pytest.raises(ValueError, match="corpus"):
        helper.validate_inputs(list(range(128)), bad, 256, False)


def test_numerical_bounds():
    torch = pytest.importorskip("torch")
    original = torch.ones(128, 4)
    unselected = torch.zeros(128, dtype=torch.bool)
    assert helper.bounded_relative(original * 1.1, original, unselected) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="relative-norm"):
        helper.bounded_relative(original * 1.3, original, unselected)
    unselected[0] = True
    with pytest.raises(ValueError, match="unselected"):
        helper.bounded_relative(original * 1.1, original, unselected)


def toy(monkeypatch, width=132):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    model = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=width,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            initializer_range=0.1,
        )
    )
    name = "model.layers.0.mlp.up_proj.weight"

    def batch(*args):
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor([[-100, -100, 3, 4, 5]]),
        }

    monkeypatch.setattr(helper.objectives, "answer_batch", batch)
    runtime = SimpleNamespace(
        model=model,
        tokenizer=object(),
        storage_dtypes={name: "F32"},
        config=SimpleNamespace(seed=42, torch_threads=2, max_prompt_tokens=256),
        progress=lambda *a, **k: None,
        check=lambda: None,
    )
    return torch, runtime, name


def test_real_loss_update_and_restoration(monkeypatch, tmp_path):
    torch, runtime, name = toy(monkeypatch)
    before = {k: v.clone() for k, v in runtime.model.state_dict().items()}
    flags = [p.requires_grad for p in runtime.model.parameters()]
    proposal, steps = helper.train(
        runtime,
        name,
        list(range(128)),
        data(),
        SimpleNamespace(check=lambda *a: None),
        tmp_path,
        retain_only=True,
        steps=2,
    )
    assert steps == 2 and not torch.equal(proposal, before[name])
    assert torch.equal(proposal[128:], before[name][128:])
    assert all(torch.equal(v, before[k]) for k, v in runtime.model.state_dict().items())
    assert flags == [p.requires_grad for p in runtime.model.parameters()]
    assert all(p.grad is None for p in runtime.model.parameters())


def test_teacher_failure_restores_model(monkeypatch, tmp_path):
    torch, runtime, name = toy(monkeypatch)
    original = runtime.model.state_dict()[name].clone()
    real = helper.objectives.answer_forward
    calls = 0

    def fail(model, batch):
        nonlocal calls
        calls += 1
        if calls == 4:  # next teacher forward, after a real first update
            raise RuntimeError("teacher failure")
        return real(model, batch)

    monkeypatch.setattr(helper.objectives, "answer_forward", fail)
    with pytest.raises(RuntimeError, match="teacher failure"):
        helper.train(
            runtime,
            name,
            list(range(128)),
            data(),
            SimpleNamespace(check=lambda *a: None),
            tmp_path,
            retain_only=True,
            steps=2,
        )
    assert torch.equal(runtime.model.state_dict()[name], original)
    assert all(p.grad is None for p in runtime.model.parameters())


def test_projected_argument_contract():
    rows = list(range(4864))
    helper.validate_inputs(rows, data(), 512, False, True)
    helper.validate_inputs(rows, data(), 64, True, True)
    for count, steps, retain, projected in [
        (128, 512, False, True),
        (4864, 256, False, True),
        (4864, 512, False, False),
        (4864, 513, True, True),
        (4864, 512, False, 1),
    ]:
        with pytest.raises(ValueError):
            helper.validate_inputs(list(range(count)), data(), steps, retain, projected)


def test_projection_and_independent_storage_bound():
    torch = pytest.importorskip("torch")
    original = torch.ones(4, 4)
    unselected = torch.tensor([True, False, False, False])
    current = original.clone()
    current[1:] += 1
    record = helper.project_update(current, original, unselected)
    assert record["pre_projection_relative"] == pytest.approx(3**0.5 / 2)
    assert record["projection_scale"] == pytest.approx(0.24 / (3**0.5 / 2))
    assert record["post_projection_relative"] == pytest.approx(0.24, abs=1e-7)
    assert torch.equal(current[0], original[0])
    inside = original.clone()
    inside[1:] += 0.01
    before = inside.clone()
    assert helper.project_update(inside, original, unselected)["projection_scale"] == 1
    assert torch.equal(inside, before)
    # Independent bound must still reject an invalid rounded/storage proposal.
    invalid_stored = original.clone()
    invalid_stored[1:] += 0.4
    with pytest.raises(ValueError, match="relative-norm"):
        helper.bounded_relative(invalid_stored, original, unselected)
    with pytest.raises(ValueError, match="original projection norm"):
        helper.project_update(torch.ones(4, 4), torch.zeros(4, 4), torch.zeros(4, dtype=torch.bool))


def test_projected_training_trace_and_failure_restore(monkeypatch, tmp_path):
    import json

    torch, runtime, name = toy(monkeypatch, width=4864)
    original = runtime.model.state_dict()[name].clone()
    budget = SimpleNamespace(check=lambda *args: None)
    helper.train(
        runtime,
        name,
        list(range(4864)),
        data(),
        budget,
        tmp_path,
        retain_only=True,
        steps=1,
        projected=True,
    )
    trace = json.loads((tmp_path / "training-trace.json").read_text())[0]
    assert trace["projected"] is True
    assert trace["learning_rate"] == 0.0003
    assert trace["projection_scale"] == min(1.0, 0.24 / trace["pre_projection_relative"])
    assert trace["projection_count"] == int(trace["projection_scale"] < 1)
    assert trace["relative_frobenius"] <= 0.24 + 1e-7
    assert trace["stored_relative_frobenius"] <= 0.25
    real = helper.project_update

    def fail(*args):
        real(*args)
        raise RuntimeError("after projection")

    monkeypatch.setattr(helper, "project_update", fail)
    with pytest.raises(RuntimeError, match="after projection"):
        helper.train(
            runtime,
            name,
            list(range(4864)),
            data(),
            budget,
            tmp_path,
            retain_only=True,
            steps=1,
            projected=True,
        )
    assert torch.equal(runtime.model.state_dict()[name], original)
    assert all(p.grad is None for p in runtime.model.parameters())


def test_numeric_guard_removal_controls():
    import inspect

    torch = pytest.importorskip("torch")
    original = torch.ones(4, 4)
    mask = torch.zeros(4, dtype=torch.bool)
    namespace = dict(vars(helper))
    source = inspect.getsource(helper.project_update)
    assert "current.copy_(original + delta * scale)" in source
    exec(source.replace("current.copy_(original + delta * scale)", "pass"), namespace)
    # Removing projection makes the real outside-radius fixture fail validation.
    with pytest.raises(ValueError, match="relative-norm"):
        namespace["project_update"](original * 2, original, mask)
    source = inspect.getsource(helper.bounded_relative)
    assert "or relative > MAX_RELATIVE" in source
    exec(source.replace("or relative > MAX_RELATIVE", ""), namespace)
    # Without the independent guard the forbidden stored update is accepted.
    assert namespace["bounded_relative"](original * 1.3, original, mask) > 0.25
    with pytest.raises(ValueError, match="relative-norm"):
        helper.bounded_relative(original * 1.3, original, mask)
    valid = original * 2
    helper.project_update(valid, original, mask)
    assert helper.bounded_relative(valid, original, mask) <= 0.24 + 1e-7
