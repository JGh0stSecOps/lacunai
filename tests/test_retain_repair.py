"""Repair equals the historical benign SGD arithmetic; no downloads."""

from types import SimpleNamespace

import pytest

from lacunai.erase import retain_repair
from lacunai.experimental import tensors


def fixture():
    torch = pytest.importorskip("torch")
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        model.weight.fill_(1)
        model.bias.fill_(0)
    for p in model.parameters():
        p.requires_grad_(False)

    def loss(prompt, answer):
        return (model(torch.tensor([[0.1, 0.2]])) - 0.2).square().mean()

    return torch, SimpleNamespace(model=model, loss=loss, progress=lambda *a, **kw: None)


def test_literal_benign_equivalence():
    torch, runtime = fixture()
    original = {n: p.clone() for n, p in runtime.model.state_dict().items()}
    rows = [{"prompt": str(i), "answer": "a"} for i in range(32)]
    trace = retain_repair.repair(runtime, ["weight"], rows)
    actual = runtime.model.weight.detach().clone()
    runtime.model.load_state_dict(original)
    runtime.model.weight.requires_grad_(True)
    optimizer = torch.optim.SGD([runtime.model.weight], lr=0.05)
    expected = []
    for i in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss = runtime.loss(rows[i]["prompt"], rows[i]["answer"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([runtime.model.weight], 1.0, error_if_nonfinite=True)
        optimizer.step()
        expected.append({"step": i + 1, "loss": float(loss.detach())})
    assert torch.equal(actual, runtime.model.weight)
    assert trace == expected
    assert torch.equal(runtime.model.bias, original["bias"])


def test_bound_failure_restores():
    torch, runtime = fixture()
    original = runtime.model.weight.detach().clone()
    with pytest.raises(ValueError, match="relative-norm"):
        retain_repair.repair(
            runtime,
            ["weight"],
            [{"prompt": "p", "answer": "a"}],
            reference={"weight": original * 0.5},
            bound_tolerance=0,
        )
    assert torch.equal(runtime.model.weight, original)
    assert all(p.grad is None for p in runtime.model.parameters())


def test_non_target_mutation_detected():
    torch, runtime = fixture()
    original = runtime.model.weight.detach().clone()

    def corrupt(trace):
        with torch.no_grad():
            runtime.model.bias.add_(1)

    with pytest.raises(ValueError, match="non-target state"):
        retain_repair.repair(runtime, ["weight"], [{"prompt": "p", "answer": "a"}], on_step=corrupt)
    assert torch.equal(runtime.model.weight, original)


def test_actual_tiny_qwen_repair_variant(monkeypatch):
    from lacunai.erase import factoid_gradient as helper
    from tests.test_factoid_gradient import FORGET, toy

    torch, runtime, name, _ = toy(monkeypatch)
    rows = [{"prompt": f"retain {i}", "answer": "oak"} for i in range(32)]

    def loss(prompt, answer):
        batch = helper.kernels.answer_batch(
            runtime.tokenizer, {"prompts": [prompt], "answer": answer}, 128
        )
        logits = helper.kernels.answer_forward(runtime.model, batch)
        return (
            helper.kernels.sequence_nll(logits, batch, selected_positions=True)
            / (batch["labels"] != -100).sum()
        )

    runtime.loss = loss
    before = helper._hashes(runtime.model)
    result = helper.propose(runtime, name, FORGET, rows, mode="npo-kl-repair", steps=1)
    assert (
        result["unlearning_steps"] == 1
        and result["repair_steps"] == 8
        and result["total_steps"] == 9
    )
    assert len(result["repair_trace"]) == 8
    assert result["post_repair_rounded_hash"] == tensors.tensor_hash(result["replacements"][name])
    assert helper._hashes(runtime.model) == before


def test_tiny_qwen_matches_literal_benign(monkeypatch):
    from lacunai.erase import gradient_objectives as kernels
    from tests.test_factoid_gradient import toy

    torch, runtime, name, _ = toy(monkeypatch)
    rows = [{"prompt": f"retain {i}", "answer": "oak"} for i in range(32)]

    def loss(prompt, answer):
        batch = kernels.answer_batch(
            runtime.tokenizer, {"prompts": [prompt], "answer": answer}, 128
        )
        return (
            kernels.sequence_nll(
                kernels.answer_forward(runtime.model, batch), batch, selected_positions=True
            )
            / (batch["labels"] != -100).sum()
        )

    runtime.loss = loss
    original = {n: t.clone() for n, t in runtime.model.state_dict().items()}
    for p in runtime.model.parameters():
        p.requires_grad_(False)
    trace = retain_repair.repair(runtime, [name], rows)
    actual = runtime.model.state_dict()[name].clone()
    runtime.model.load_state_dict(original)
    parameter = dict(runtime.model.named_parameters())[name]
    parameter.requires_grad_(True)
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    literal = []
    for i in range(8):
        optimizer.zero_grad(set_to_none=True)
        value = runtime.loss(rows[i]["prompt"], rows[i]["answer"])
        value.backward()
        torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)
        optimizer.step()
        literal.append({"step": i + 1, "loss": float(value.detach())})
    assert trace == literal
    assert torch.equal(actual, parameter)


def test_non_target_guard_removal_control():
    import inspect

    torch, runtime = fixture()

    def corrupt(trace):
        with torch.no_grad():
            runtime.model.bias.add_(1)

    namespace = dict(vars(retain_repair))
    source = inspect.getsource(retain_repair.repair)
    assert "if actual != hashes:" in source
    exec(source.replace("if actual != hashes:", "if False:"), namespace)
    namespace["repair"](runtime, ["weight"], [{"prompt": "p", "answer": "a"}], on_step=corrupt)
    assert not torch.equal(runtime.model.bias, torch.zeros_like(runtime.model.bias))
    _, runtime = fixture()
    with pytest.raises(ValueError, match="non-target state"):
        retain_repair.repair(runtime, ["weight"], [{"prompt": "p", "answer": "a"}], on_step=corrupt)


def test_global_bound_guard_removal_control():
    import inspect

    torch, runtime = fixture()
    original = runtime.model.weight.detach().clone()
    reference = {"weight": original * 0.5}
    source = inspect.getsource(retain_repair.repair)
    condition = "if not math.isfinite(relative) or relative > 0.25 + bound_tolerance:"
    assert condition in source
    namespace = dict(vars(retain_repair))
    exec(source.replace(condition, "if False:"), namespace)
    namespace["repair"](
        runtime,
        ["weight"],
        [{"prompt": "p", "answer": "a"}],
        reference=reference,
        bound_tolerance=0,
    )
    ratio = (
        tensors._square_norm(runtime.model.weight - reference["weight"])
        / tensors._square_norm(reference["weight"])
    ) ** 0.5
    assert ratio > 0.25
    _, runtime = fixture()
    with pytest.raises(ValueError, match="relative-norm"):
        retain_repair.repair(
            runtime,
            ["weight"],
            [{"prompt": "p", "answer": "a"}],
            reference=reference,
            bound_tolerance=0,
        )
    assert torch.equal(runtime.model.weight, original)


def test_rounding_order_and_storage_bound(monkeypatch):
    from lacunai.erase import factoid_gradient as helper
    from tests.test_factoid_gradient import FORGET, toy

    torch, runtime, name, _ = toy(monkeypatch)
    runtime.model.to(torch.bfloat16).to(torch.float32)
    runtime.storage_dtypes[name] = "BF16"
    rows = [{"prompt": f"retain {i}", "answer": "oak"} for i in range(32)]
    before = helper._hashes(runtime.model)
    original = dict(runtime.model.named_parameters())[name].detach().clone()
    real_round = helper.round_to_storage
    real_relative = helper._relative
    rounds = []
    repair_entries = []

    def rounding(replacements, dtypes):
        result = real_round(replacements, dtypes)
        rounds.append(result[name].clone())
        if len(rounds) == 3:  # original check, base candidate, post-repair candidate
            return {name: original * 1.5}
        return result

    def repair_spy(runtime, names, rows, **kwargs):
        parameter = dict(runtime.model.named_parameters())[name]
        repair_entries.append(parameter.detach().clone())
        assert torch.equal(parameter, rounds[1])
        assert kwargs["bound_tolerance"] == 0
        assert torch.equal(kwargs["reference"][name], original)
        return []  # isolate the post-repair storage guard, not SGD correctness

    monkeypatch.setattr(helper, "round_to_storage", rounding)
    monkeypatch.setattr(retain_repair, "repair", repair_spy)
    with pytest.raises(ValueError, match="relative bound"):
        helper.propose(runtime, name, FORGET, rows, mode="npo-kl-repair", steps=1)
    assert len(rounds) == 3 and len(repair_entries) == 1
    assert helper._hashes(runtime.model) == before
    # Deleting the independent post-round bound makes this same malformed proposal pass.
    rounds.clear()

    def removed(value, source):
        if torch.equal(value, original * 1.5):
            return 0.5
        return real_relative(value, source)

    monkeypatch.setattr(helper, "_relative", removed)
    result = helper.propose(runtime, name, FORGET, rows, mode="npo-kl-repair", steps=1)
    assert torch.equal(result["replacements"][name], original * 1.5)
    monkeypatch.setattr(helper, "_relative", real_relative)
    rounds.clear()
    with pytest.raises(ValueError, match="relative bound"):
        helper.propose(runtime, name, FORGET, rows, mode="npo-kl-repair", steps=1)
    assert helper._hashes(runtime.model) == before


def test_repair_order_flags_and_runtime_cap():
    torch, runtime = fixture()
    runtime.config = SimpleNamespace(max_prompt_tokens=256)
    rows = [{"prompt": str(i), "answer": "a"} for i in range(32)]
    seen = []
    original_loss = runtime.loss

    def tracked(prompt, answer):
        seen.append(
            (
                prompt,
                runtime.config.max_prompt_tokens,
                runtime.model.weight.requires_grad,
                runtime.model.bias.requires_grad,
            )
        )
        return original_loss(prompt, answer)

    runtime.loss = tracked
    retain_repair.repair(runtime, ["weight"], rows)
    assert seen == [(str(i), 256, True, False) for i in range(8)]
    assert all(not p.requires_grad and p.grad is None for p in runtime.model.parameters())
