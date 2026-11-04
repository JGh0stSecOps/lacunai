"""Offline proposal contracts and tiny random Qwen numerical checks."""

from types import SimpleNamespace

import pytest

from lacunai.erase import factoid_gradient as helper

FORGET = [{"prompt": "synthetic target", "answer": "42"}]
RETAIN = [{"prompt": "synthetic retain", "answer": "oak"}]


@pytest.mark.parametrize("steps", [0, 129, True, 1.0, "1"])
def test_step_guard(steps):
    with pytest.raises(ValueError):
        helper.validate_inputs("npo-kl", steps, FORGET, RETAIN)


def test_data_guards():
    for mode in helper.MODES[:2]:
        helper.validate_inputs(mode, 1, FORGET, RETAIN)
    with pytest.raises(ValueError):
        helper.validate_inputs("other", 1, FORGET, RETAIN)
    with pytest.raises(ValueError):
        helper.validate_inputs("npo-kl", 1, FORGET, FORGET)
    with pytest.raises(ValueError):
        helper.validate_inputs("npo-kl", 1, [{"prompt": "p", "answer": ""}], RETAIN)


def toy(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    model = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            initializer_range=0.2,
        )
    )
    name = "model.layers.0.mlp.up_proj.weight"

    def batch(tokenizer, item, cap):
        ids = torch.tensor([[1, 2, 3, 4]])
        if item["prompts"][0] == RETAIN[0]["prompt"]:
            ids = torch.tensor([[5, 6, 7, 8]])
        labels = ids.clone()
        labels[:, :2] = -100
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}

    monkeypatch.setattr(helper.kernels, "answer_batch", batch)
    events = []
    runtime = SimpleNamespace(
        model=model,
        tokenizer=object(),
        config=SimpleNamespace(seed=42, torch_threads=2, max_prompt_tokens=128),
        storage_dtypes={name: "F32"},
        check=lambda: None,
        progress=lambda *args, **kw: events.append(kw),
    )
    return torch, runtime, name, events


@pytest.mark.parametrize("mode", helper.MODES[:2])
def test_actual_proposal_likelihood_and_restore(monkeypatch, mode):
    torch, runtime, name, events = toy(monkeypatch)
    before = helper._hashes(runtime.model)
    batch = helper.kernels.answer_batch(None, {"prompts": [FORGET[0]["prompt"]]}, 128)

    def loss():
        return float(
            helper.kernels.sequence_nll(
                helper.kernels.answer_forward(runtime.model, batch), batch, selected_positions=True
            )
        )

    with torch.no_grad():
        initial = loss()
    result = helper.propose(runtime, name, FORGET, RETAIN, mode=mode, steps=2)
    assert helper._hashes(runtime.model) == before
    assert events[0]["state"] == "active" and events[-1]["state"] == "complete"
    assert all(p.grad is None for p in runtime.model.parameters())
    with torch.no_grad():
        dict(runtime.model.named_parameters())[name].copy_(result["replacements"][name])
        assert loss() > initial + 1e-6
    assert result["trace"][0]["retain_mean_nll"] > 0


def test_failure_restore_and_allowlist(monkeypatch):
    torch, runtime, name, events = toy(monkeypatch)
    with pytest.raises(ValueError, match="explicit final"):
        helper.propose(runtime, "lm_head.weight", FORGET, RETAIN, mode="npo-kl", steps=1)
    before = helper._hashes(runtime.model)
    real = helper._relative
    calls = 0

    def fail(value, original):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("injected after second update")
        return real(value, original)

    monkeypatch.setattr(helper, "_relative", fail)
    with pytest.raises(ValueError, match="injected"):
        helper.propose(runtime, name, FORGET, RETAIN, mode="npo-kl", steps=2)
    assert helper._hashes(runtime.model) == before
    assert events[-1]["state"] == "failed"
    assert all(p.grad is None for p in runtime.model.parameters())


def test_retain_gradient_control(monkeypatch):
    torch, runtime, name, _ = toy(monkeypatch)
    batch = helper.kernels.answer_batch(None, {"prompts": [RETAIN[0]["prompt"]]}, 128)
    p = dict(runtime.model.named_parameters())[name]
    original = p.detach().clone()
    teacher = helper.kernels.teacher_log_probs(
        runtime.model, {"original_tensors": {name: original}}, batch
    )
    with torch.no_grad():
        p.add_(0.02)
    logits = helper.kernels.answer_forward(runtime.model, batch)
    kl = helper.kernels.retain_kl(logits, teacher, batch, selected_positions=True)
    grad = torch.autograd.grad(kl, p)[0]
    assert float(kl.detach()) > 0 and float(grad.abs().max()) > 1e-8


def test_preparation_failure_terminal(monkeypatch):
    events = []
    runtime = SimpleNamespace(progress=lambda *a, **kw: events.append(kw))

    def fail(*args, **kwargs):
        raise OSError("preparation")

    monkeypatch.setattr(helper, "_propose", fail)
    with pytest.raises(OSError):
        helper.propose(runtime, "x", FORGET, RETAIN, mode="npo-kl")
    assert [e["state"] for e in events] == ["active", "failed"]


def test_actual_retain_contribution(monkeypatch):
    torch, runtime, name, _ = toy(monkeypatch)
    full = helper.propose(runtime, name, FORGET, RETAIN, mode="npo-kl", steps=2)
    real_nll = helper.kernels.sequence_nll

    def suppress_retain(logits, batch, **kwargs):
        result = real_nll(logits, batch, **kwargs)
        return result * 0 if int(batch["input_ids"][0, 0]) == 5 else result

    monkeypatch.setattr(helper.kernels, "sequence_nll", suppress_retain)
    real_kl = helper.kernels.retain_kl
    monkeypatch.setattr(helper.kernels, "retain_kl", lambda *a, **kw: real_kl(*a, **kw) * 0)
    control = helper.propose(runtime, name, FORGET, RETAIN, mode="npo-kl", steps=2)
    assert full["reference_sequence_nll"] == control["reference_sequence_nll"]
    assert full["source_hashes"] == control["source_hashes"]
    assert not torch.equal(full["replacements"][name], control["replacements"][name])
    assert control["trace"][0]["retain_mean_nll"] == 0
    assert full["trace"][0]["retain_mean_nll"] > 0


def test_restoration_failure_preserves_primary(monkeypatch):
    _, runtime, name, events = toy(monkeypatch)
    real_hashes = helper._hashes
    calls = 0

    def broken_hashes(model):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("restoration hash failure")
        return real_hashes(model)

    monkeypatch.setattr(helper, "_hashes", broken_hashes)

    def fail(*a, **kw):
        raise ValueError("batch failure")

    monkeypatch.setattr(helper.kernels, "answer_batch", fail)
    with pytest.raises(BaseExceptionGroup) as caught:
        helper.propose(runtime, name, FORGET, RETAIN, mode="npo-kl", steps=1)
    assert [str(e) for e in caught.value.exceptions] == [
        "batch failure",
        "restoration hash failure",
    ]
    assert events[-1]["state"] == "failed"


def test_runtime_default_training_cap():
    from lacunai.core.spec import RuntimeConfig

    assert helper.training_cap(RuntimeConfig().max_prompt_tokens) == 128
    assert helper.training_cap(RuntimeConfig(max_prompt_tokens=16).max_prompt_tokens) == 16
    assert helper.training_cap(1024) == 128


@pytest.mark.parametrize("cap", [True, 128.0, "128", 0, 15, 1025, None])
def test_unsupported_runtime_cap(cap):
    with pytest.raises(ValueError, match="runtime prompt cap"):
        helper.training_cap(cap)


@pytest.mark.parametrize("runtime_cap, expected", [(256, 128), (16, 16)])
def test_proposal_actual_cap_seam(monkeypatch, runtime_cap, expected):
    from lacunai.core.spec import RuntimeConfig

    _, runtime, name, _ = toy(monkeypatch)
    runtime.config = RuntimeConfig(max_prompt_tokens=runtime_cap)
    real = helper.kernels.answer_batch
    seen = []

    def capture(tokenizer, item, cap):
        seen.append(cap)
        return real(tokenizer, item, cap)

    monkeypatch.setattr(helper.kernels, "answer_batch", capture)
    result = helper.propose(runtime, name, FORGET, RETAIN, mode="npo-kl", steps=1)
    assert seen and set(seen) == {expected}
    assert result["settings"]["inference_prompt_cap"] == runtime_cap
    assert result["settings"]["trainingsequence_tokens"] == expected


def test_repair_step_and_data_contract():
    rows = [{"prompt": f"retain {i}", "answer": "oak"} for i in range(32)]
    for mode in helper.REPAIR_MODES:
        maximum = 112 if mode == "grad-diff-kl-repair2" else 120
        helper.validate_inputs(mode, maximum, FORGET, rows)
        with pytest.raises(ValueError, match=str(maximum)):
            helper.validate_inputs(mode, maximum + 1, FORGET, rows)
        with pytest.raises(ValueError, match="32 retain"):
            helper.validate_inputs(mode, 8, FORGET, RETAIN)


@pytest.mark.parametrize("mode", ["undial-kl", "undial-kl-repair"])
def test_tiny_undial_proposal_restores_teacher(monkeypatch, mode):
    torch, runtime, name, _ = toy(monkeypatch)
    runtime.tokenizer = SimpleNamespace(eos_token_id=4)
    rows = [{"prompt": f"retain {i}", "answer": "oak"} for i in range(32)]

    def loss(prompt, answer):
        batch = helper.kernels.answer_batch(
            runtime.tokenizer, {"prompts": [prompt], "answer": answer}, 128
        )
        return (
            helper.kernels.sequence_nll(
                helper.kernels.answer_forward(runtime.model, batch), batch, selected_positions=True
            )
            / (batch["labels"] != -100).sum()
        )

    runtime.loss = loss
    original = helper._hashes(runtime.model)
    real_teacher = helper.kernels.teacher_log_probs
    reference_hashes = []

    def teacher(model, reference, batch):
        reference_hashes.append(helper.tensors.tensor_hash(reference["original_tensors"][name]))
        return real_teacher(model, reference, batch)

    monkeypatch.setattr(helper.kernels, "teacher_log_probs", teacher)
    result = helper.propose(runtime, name, FORGET, rows, mode=mode, steps=1)
    assert reference_hashes and set(reference_hashes) == {original[name]}
    assert result["undial"]["demotion"] == 3
    assert helper._hashes(runtime.model) == original
    assert not torch.equal(result["replacements"][name], runtime.model.state_dict()[name])
    assert all(p.grad is None for p in runtime.model.parameters())


def test_double_repair_roundtrip_and_source_restoration(monkeypatch):
    from lacunai.erase import retain_repair

    torch, runtime, name, _ = toy(monkeypatch)
    runtime.model.to(torch.bfloat16).to(torch.float32)
    runtime.storage_dtypes[name] = "BF16"
    rows = [{"prompt": f"retain {i}", "answer": "oak"} for i in range(32)]
    before = helper._hashes(runtime.model)
    original = runtime.model.state_dict()[name].clone()
    calls = []

    def loss(prompt, answer):
        calls.append(prompt)
        # Small differentiable objective keeps this boundary test inside the global radius.
        parameter = dict(runtime.model.named_parameters())[name]
        return (parameter - 0.01).square().mean() * 0.01

    runtime.loss = loss
    real = retain_repair.repair
    starts = []
    references = []
    ends = []

    def observed(runtime, names, data, **kwargs):
        starts.append(runtime.model.state_dict()[name].clone())
        references.append(kwargs["reference"][name].clone())
        result = real(runtime, names, data, **kwargs)
        ends.append(runtime.model.state_dict()[name].clone())
        return result

    monkeypatch.setattr(retain_repair, "repair", observed)
    result = helper.propose(runtime, name, FORGET, rows, mode="grad-diff-kl-repair2", steps=1)
    assert calls == [f"retain {i}" for i in range(8)] * 2
    assert len(starts) == 2 and all(torch.equal(value, original) for value in references)
    assert torch.equal(starts[1], ends[0].to(torch.bfloat16).float())
    assert torch.equal(result["replacements"][name], ends[1].to(torch.bfloat16).float())
    assert result["repair_steps"] == 16 and result["total_steps"] == 17
    rounds = result["repair_rounds"]
    assert rounds[0]["post_repair_rounded_hash"] == rounds[1]["pre_repair_rounded_hash"]
    assert rounds[1]["post_repair_rounded_hash"] == result["proposal_hash"]
    assert len(rounds[0]["repair_trace"]) == len(rounds[1]["repair_trace"]) == 8
    assert helper._hashes(runtime.model) == before
    assert all(p.grad is None for p in runtime.model.parameters())


def test_double_repair_second_round_bound_failure_restores(monkeypatch):
    from lacunai.erase import retain_repair

    torch, runtime, name, _ = toy(monkeypatch)
    rows = [{"prompt": f"retain {i}", "answer": "oak"} for i in range(32)]
    before = helper._hashes(runtime.model)
    count = 0

    def injected(runtime, names, data, **kwargs):
        nonlocal count
        count += 1
        assert kwargs["bound_tolerance"] == 0
        if count == 2:
            with torch.no_grad():
                dict(runtime.model.named_parameters())[name].copy_(kwargs["reference"][name] * 2)
        return [{"step": i + 1, "loss": 0.0} for i in range(8)]

    monkeypatch.setattr(retain_repair, "repair", injected)
    with pytest.raises(ValueError, match="relative bound"):
        helper.propose(runtime, name, FORGET, rows, mode="grad-diff-kl-repair2", steps=1)
    assert count == 2 and helper._hashes(runtime.model) == before
