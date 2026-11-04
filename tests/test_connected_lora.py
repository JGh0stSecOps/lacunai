import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from lacunai.erase import connected_training as training
from tests.test_connected_training import UTILITY, rows


def test_unknown_parameterization_rejected_before_torch(monkeypatch):
    monkeypatch.setattr(training, "import_module", lambda _: pytest.fail("Torch imported"))
    runtime = SimpleNamespace(progress=lambda *a, **k: None)
    for value in (None, True, 8, "lora_rank16", {}, []):
        with pytest.raises(ValueError, match="parameterization"):
            training.train_connected(runtime, rows(), UTILITY, planting_parameterization=value)


def test_plan_parameterization_and_both_arm_calls():
    from tests.test_connected_pilot import pilot

    assert pilot.plan_parameterization({}) == "dense"
    assert pilot.plan_parameterization({"planting_parameterization": "lora_rank8"}) == "lora_rank8"
    for invalid in (True, None, "other", []):
        with pytest.raises(ValueError, match="parameterization"):
            pilot.plan_parameterization({"planting_parameterization": invalid})
    tree = ast.parse(Path(pilot.__file__).read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "train_connected"
    ]
    assert len(calls) == 2
    for call in calls:
        supplied = {k.arg: k.value for k in call.keywords}
        assert ast.unparse(supplied["planting_parameterization"]) == "parameterization"


def test_adapter_equivalence_teacher_isolation_projection_and_rng():
    torch = pytest.importorskip("torch")
    from lacunai.erase.connected_lora import Adapter

    module = torch.nn.Linear(4, 6, bias=False)
    original = module.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    hooks = len(module._forward_hooks)
    adapter = Adapter(module, original)
    try:
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.count_nonzero(adapter.b) == 0
        with torch.no_grad():
            adapter.b.fill_(0.01)
        x = torch.ones(2, 4)
        merged = torch.nn.functional.linear(x, adapter.merged())
        assert torch.allclose(module(x), merged, atol=1e-7, rtol=1e-6)
        with adapter.disabled():
            assert torch.equal(module(x), torch.nn.functional.linear(x, original))
        with torch.no_grad():
            adapter.b.mul_(10000)
        evidence = adapter.project()
        assert evidence["pre_projection_relative"] > 0.24
        assert evidence["post_projection_relative"] <= 0.240001
        with torch.no_grad():
            adapter.a[0, 0] = float("nan")
        with pytest.raises(ValueError, match="nonfinite"):
            adapter.merged()
    finally:
        adapter.close()
    assert len(module._forward_hooks) == hooks
    assert torch.equal(module.weight, original)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("facts", [4, 5])
def test_real_training_bf16_callback_and_full_restoration(monkeypatch, fail, facts):
    from tests.test_factoid_gradient import toy

    torch, runtime, name, _ = toy(monkeypatch)
    parameter = dict(runtime.model.named_parameters())[name]
    with torch.no_grad():
        parameter.copy_(parameter.to(torch.bfloat16).float())
    runtime.storage_dtypes[name] = "BF16"
    before = training._hashes(runtime.model)
    modes = [m.training for m in runtime.model.modules()]
    flags = [p.requires_grad for p in runtime.model.parameters()]
    hooks = [len(m._forward_hooks) for m in runtime.model.modules()]
    rng = torch.get_rng_state().clone()
    original_teacher = training.kernels.teacher_log_probs
    teachers = []

    def teacher(model, reference, batch):
        value = original_teacher(model, reference, batch)
        teachers.append(value.detach().clone())
        return value

    monkeypatch.setattr(training.kernels, "teacher_log_probs", teacher)
    seen = []

    def callback(step, replacements, diagnostics):
        assert torch.equal(parameter, replacements[name])
        assert torch.equal(parameter, parameter.to(torch.bfloat16).float())
        module = runtime.model.get_submodule(name.rsplit(".", 1)[0])
        x = torch.ones(1, parameter.shape[1])
        assert torch.equal(module(x), torch.nn.functional.linear(x, parameter))
        seen.append(step)
        torch.rand(1)
        if fail:
            raise RuntimeError("injected callback failure")
        return True

    try:
        result = training.train_connected(
            runtime,
            rows(facts),
            UTILITY,
            steps=2,
            profile="connected-5",
            planting_parameterization="lora_rank8",
            checkpoint_callback=callback,
        )
        assert not fail
        assert result["diagnostics"]["adapter"]["rank"] == 8
        assert result["diagnostics"]["merged_relative"] <= 0.25
    except RuntimeError as exc:
        assert fail and str(exc) == "injected callback failure"
    assert seen == [2]
    assert torch.equal(teachers[0], teachers[1])
    assert training._hashes(runtime.model) == before
    assert [m.training for m in runtime.model.modules()] == modes
    assert [p.requires_grad for p in runtime.model.parameters()] == flags
    assert [len(m._forward_hooks) for m in runtime.model.modules()] == hooks
    assert all(p.grad is None for p in runtime.model.parameters())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("failure", ["rounded_bound", "optimizer"])
def test_lora_failure_restores_original_and_removes_hook(monkeypatch, failure):
    from tests.test_factoid_gradient import toy

    torch, runtime, name, events = toy(monkeypatch)
    before = training._hashes(runtime.model)
    hooks = [len(m._forward_hooks) for m in runtime.model.modules()]
    rng = torch.get_rng_state().clone()
    if failure == "rounded_bound":
        rounding = training.round_to_storage
        calls = 0

        def corrupt(values, dtypes):
            nonlocal calls
            calls += 1
            result = rounding(values, dtypes)
            if calls > 1:
                result[name] = result[name] * 2
            return result

        monkeypatch.setattr(training, "round_to_storage", corrupt)
        expected = "relative bound"
    else:

        def fail_step(*args, **kwargs):
            raise ValueError("injected optimizer failure")

        monkeypatch.setattr(torch.optim.AdamW, "step", fail_step)
        expected = "injected optimizer failure"
    with pytest.raises(ValueError, match=expected):
        training.train_connected(
            runtime,
            rows(5),
            UTILITY,
            steps=1,
            profile="connected-5",
            planting_parameterization="lora_rank8",
        )
    assert training._hashes(runtime.model) == before
    assert [len(m._forward_hooks) for m in runtime.model.modules()] == hooks
    assert all(p.grad is None for p in runtime.model.parameters())
    assert torch.equal(torch.get_rng_state(), rng)
    assert events[0]["state"] == "active" and events[-1]["state"] == "failed"


def test_explicit_dense_matches_legacy_default(monkeypatch):
    from tests.test_factoid_gradient import toy

    torch, runtime, name, _ = toy(monkeypatch)
    implicit = training.train_connected(runtime, rows(), UTILITY, steps=2)
    explicit = training.train_connected(
        runtime, rows(), UTILITY, steps=2, planting_parameterization="dense"
    )
    assert torch.equal(implicit["replacements"][name], explicit["replacements"][name])
    assert implicit["diagnostics"]["trace"] == explicit["diagnostics"]["trace"]
