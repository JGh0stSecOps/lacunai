"""Frozen RMU contract and actual tiny-Qwen early-stop/gradient equivalence."""

from types import SimpleNamespace

import pytest

from lacunai.erase import representation_misdirection as rmu


def spec_and_inner():
    spec = {**rmu.FIXED, "mode": "rmu", "allowed_tensors": list(rmu.ALLOWED)}
    inner = {
        "edits": {
            "allowed_tensors": list(rmu.ALLOWED),
            "expected_shape": [896, 4864],
            "max_tensor_relative_frobenius": "0.0195",
            "max_model_relative_frobenius": "0.0024",
        }
    }
    return spec, inner


@pytest.mark.parametrize("mode", ["rmu", "retain_mse"])
def test_frozen_spec(mode):
    spec, inner = spec_and_inner()
    spec["mode"] = mode
    rmu.validate_spec(spec, inner)


@pytest.mark.parametrize(
    "key,value",
    [
        ("method", "npo_kl_candidate"),
        ("mode", "graddiff"),
        ("optimizer", "SGD"),
        ("learning_rate", "0.01"),
        ("learning_rate", 0.0001),
        ("learning_rate", "nan"),
        ("steps", 61),
        ("schema_version", True),
        ("seed", 43),
        ("hook_module", "model.layers.8"),
        ("steering_coeff", "100"),
        ("alpha", "100"),
        ("gamma", 1.0),
        ("weight_decay", "0.01"),
        ("betas", [0.9, 0.99]),
        ("betas", (0.9, 0.999)),
        ("eps", "0.000001"),
        ("sequence_tokens", 129),
        ("allowed_tensors", []),
        ("extra", "ignored"),
    ],
)
def test_spec_drift_rejected(key, value):
    spec, inner = spec_and_inner()
    spec[key] = value
    with pytest.raises(ValueError):
        rmu.validate_spec(spec, inner)


@pytest.mark.parametrize(
    "key,value",
    [
        ("allowed_tensors", []),
        ("expected_shape", [4864, 896]),
        ("max_tensor_relative_frobenius", "0.02"),
        ("max_model_relative_frobenius", "0.0025"),
    ],
)
def test_projection_guard(key, value):
    spec, inner = spec_and_inner()
    inner["edits"][key] = value
    with pytest.raises(ValueError, match="projection"):
        rmu.validate_spec(spec, inner)


def test_failed_run_terminates_progress_and_freezes():
    events = []
    runtime = SimpleNamespace(model=SimpleNamespace(parameters=lambda: []))
    watch = SimpleNamespace(progress=lambda phase, **fields: events.append((phase, fields)))
    with pytest.raises(ValueError):
        rmu.train(runtime, {}, {}, {}, {}, watch)
    assert events[0][1]["state"] == "active"
    assert events[-1][1]["state"] == "failed"
    assert "ValueError" in events[-1][1]["error"]


def tiny_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    config = transformers.Qwen2Config(
        vocab_size=23,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=9,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen2ForCausalLM(config).float().eval()
    for name, p in model.named_parameters():
        p.requires_grad_(name in rmu.ALLOWED)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": torch.tensor([[-100, -100, 3, 4, 5]]),
    }
    return torch, model, batch


def test_early_stop_values_mask_eos_loss_and_allowed_gradients_match_full_qwen():
    torch, model, batch = tiny_model()
    block = model.model.layers[7]
    captured = []
    handle = block.register_forward_hook(lambda module, inputs, output: captured.append(output[0]))
    try:
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    finally:
        handle.remove()
    full = captured[0]
    expected = full[batch["labels"] != -100]
    target = torch.arange(48, dtype=torch.float32).reshape(3, 16) / 48
    loss = rmu.masked_mse(expected, target)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if n in rmu.ALLOWED}
    expected_values = expected.detach().clone()
    model.zero_grad(set_to_none=True)
    called = []
    later = model.model.layers[8].register_forward_pre_hook(lambda *args: called.append("later"))
    try:
        actual = rmu.capture_hidden(model, batch)
    finally:
        later.remove()
    assert not called and not block._forward_hooks
    assert actual.shape == (3, 16)  # Includes hidden state at final EOS-position label.
    torch.testing.assert_close(actual, expected_values, atol=2e-6, rtol=2e-5)
    short_loss = rmu.masked_mse(actual, target)
    torch.testing.assert_close(short_loss, loss.detach(), atol=2e-6, rtol=2e-5)
    short_loss.backward()
    for name, p in model.named_parameters():
        if name in rmu.ALLOWED:
            torch.testing.assert_close(p.grad, grads[name], atol=2e-6, rtol=2e-5)
        else:
            assert p.grad is None


def test_hook_removed_on_failure_and_foreign_sentinel_not_swallowed():
    torch, model, batch = tiny_model()
    block = model.model.layers[7]
    foreign = rmu._CapturedBlock("foreign")

    def fail(*args):
        raise foreign

    other = block.register_forward_pre_hook(fail)
    try:
        with pytest.raises(rmu._CapturedBlock, match="foreign"):
            rmu.capture_hidden(model, batch)
    finally:
        other.remove()
    assert not block._forward_hooks
    batch["labels"].fill_(-100)
    with pytest.raises(ValueError, match="empty"):
        rmu.capture_hidden(model, batch)
    assert not block._forward_hooks


def test_teacher_swaps_only_three_tensors_and_restores_after_capture_failure():
    torch, model, batch = tiny_model()
    parameters = dict(model.named_parameters())
    reference = {
        "original_tensors": {name: parameters[name].detach().clone() for name in rmu.ALLOWED}
    }
    with torch.no_grad():
        expected = rmu.capture_hidden(model, batch).clone()
    with torch.no_grad():
        for name in rmu.ALLOWED:
            parameters[name].add_(0.01)
    current = {name: parameters[name].detach().clone() for name in rmu.ALLOWED}
    with pytest.raises(RuntimeError, match="after teacher"), rmu.original_teacher(model, reference):
        observed = rmu.capture_hidden(model, batch)
        assert not observed.requires_grad
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        raise RuntimeError("after teacher")
    for name in rmu.ALLOWED:
        assert torch.equal(parameters[name], current[name])

    def fail_inside_capture(*args):
        raise RuntimeError("inside capture")

    handle = model.model.layers[7].register_forward_pre_hook(fail_inside_capture)
    try:
        with (
            pytest.raises(RuntimeError, match="inside capture"),
            rmu.original_teacher(model, reference),
        ):
            rmu.capture_hidden(model, batch)
    finally:
        handle.remove()
    assert not model.model.layers[7]._forward_hooks
    for name in rmu.ALLOWED:
        assert torch.equal(parameters[name], current[name])


def test_control_reproducible_without_global_rng_and_mse_reduction():
    torch = pytest.importorskip("torch")
    torch.manual_seed(123)
    state = torch.random.get_rng_state().clone()
    a, record = rmu.control_vector()
    b, again = rmu.control_vector()
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(a, b) and record == again
    assert abs(record["norm"] - 20) < 1e-5
    assert record["tensor_sha256"] == rmu.tensors.tensor_hash(a)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = torch.zeros_like(x, requires_grad=True)
    loss = rmu.masked_mse(x, target)
    assert loss.item() == pytest.approx(7.5)
    loss.backward()
    torch.testing.assert_close(x.grad, x.detach() / 2)
    assert target.grad is None
