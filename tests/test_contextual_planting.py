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


def toy(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    model = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=132,
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
