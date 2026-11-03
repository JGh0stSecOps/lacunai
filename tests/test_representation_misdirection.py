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


@pytest.mark.parametrize("mode", ["rmu_context", "retain_context_mse"])
def test_context_spec_is_explicit_and_old_contract_unchanged(mode):
    _, inner = spec_and_inner()
    spec = {**rmu.CONTEXT_FIXED, "mode": mode, "allowed_tensors": list(rmu.ALLOWED)}
    rmu.validate_spec(spec, inner)
    spec["mask_policy"] = "all_tokens"
    with pytest.raises(ValueError, match="mask_policy"):
        rmu.validate_spec(spec, inner)
    old, _ = spec_and_inner()
    old["mask_policy"] = rmu.CONTEXT_FIXED["mask_policy"]
    with pytest.raises(ValueError, match="fields"):
        rmu.validate_spec(old, inner)


def test_context_variant_cycle_is_per_family_and_writing_remains_zero():
    items = [
        {"id": f"{family}{i}", "family": family, "prompts": ["a", "b", "c", "d"]}
        for family in rmu.READING_FAMILIES + rmu.WRITING_FAMILIES
        for i in range(16)
    ]
    schedule = rmu.family_schedule(items, 60)
    variants = rmu.context_variants(items, schedule, retain=False)
    for step, (index, variant) in enumerate(zip(schedule["indices"], variants, strict=True)):
        assert variant == (step // 4 % 4 if items[index]["family"] in rmu.READING_FAMILIES else 0)
    assert rmu.context_variants(items, schedule, retain=True) == [0] * 60


class OffsetTokenizer:
    is_fast = True
    all_special_ids = []
    eos_token_id = 22

    def apply_chat_template(self, messages, **kwargs):
        return "S|U|" + messages[0]["content"] + "|A|"

    def __call__(self, text, **kwargs):
        record = {"input_ids": [ord(c) % 17 + 3 for c in text]}
        if kwargs.get("return_offsets_mapping"):
            record["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return record


def test_user_offsets_exclude_role_delimiters_and_cover_literal_content():
    record = rmu.user_content_offsets(OffsetTokenizer(), "hi", "yes")
    assert record["user_content_span"] == [4, 6]
    assert record["user_content_positions"] == [4, 5]
    assert record["offset_mapping"][4:6] == [[4, 5], [5, 6]]


def test_user_offset_boundary_crossing_and_special_tokens_fail_closed():
    class Cross(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            record = super().__call__(text, **kwargs)
            if "offset_mapping" in record:
                record["offset_mapping"][4] = (3, 5)
            return record

    with pytest.raises(ValueError, match="crosses boundary"):
        rmu.user_content_offsets(Cross(), "hi", "yes")
    tokenizer = OffsetTokenizer()
    tokenizer.all_special_ids = [ord("h") % 17 + 3]
    with pytest.raises(ValueError, match="special token"):
        rmu.user_content_offsets(tokenizer, "hi", "yes")
    tokenizer.is_fast = False
    with pytest.raises(ValueError, match="fast-tokenizer"):
        rmu.user_content_offsets(tokenizer, "hi", "yes")


def test_context_batch_mask_includes_user_and_answer_eos_but_no_roles():
    torch = pytest.importorskip("torch")
    item = {
        "id": "read1",
        "family": rmu.READING_FAMILIES[0],
        "prompts": ["hi", "yo"],
        "answer": "yes",
    }
    batch, record = rmu.context_batch(OffsetTokenizer(), item, 1, retain=False)
    assert record["variant"] == 1 and record["positions"] == [4, 5, 9, 10, 11, 12]
    assert batch["labels"].tolist() == [[-100] * 9 + [ord(c) % 17 + 3 for c in "yes"] + [22]]
    assert torch.nonzero(batch["activation_mask"][0]).flatten().tolist() == record["positions"]
    assert record["mask_sha256"] == rmu.record_digest(batch["activation_mask"].tolist())
    writing = {**item, "family": rmu.WRITING_FAMILIES[0]}
    _, write_record = rmu.context_batch(OffsetTokenizer(), writing, 0, retain=False)
    assert write_record["positions"] == [9, 10, 11, 12]
    with pytest.raises(ValueError, match="variant zero"):
        rmu.context_batch(OffsetTokenizer(), writing, 1, retain=False)
    with pytest.raises(ValueError, match="cap"):
        rmu.context_batch(OffsetTokenizer(), {**item, "prompts": ["x" * 128]}, 0, retain=False)


def test_real_qwen_context_mask_early_stop_loss_and_gradients_equal_full_forward():
    torch, model, _ = tiny_model()
    item = {"id": "read1", "family": rmu.READING_FAMILIES[0], "prompts": ["hi"], "answer": "yes"}
    batch, record = rmu.context_batch(OffsetTokenizer(), item, 0, retain=False)
    mask = batch["activation_mask"]
    captured = []
    handle = model.model.layers[7].register_forward_hook(
        lambda module, inputs, output: captured.append(output[0])
    )
    try:
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    finally:
        handle.remove()
    full = captured[0][mask]
    target = torch.arange(full.numel(), dtype=torch.float32).reshape(full.shape) / full.numel()
    loss = rmu.masked_mse(full, target)
    loss.backward()
    gradients = {
        name: p.grad.clone() for name, p in model.named_parameters() if name in rmu.ALLOWED
    }
    full_values = full.detach().clone()
    model.zero_grad(set_to_none=True)
    early = rmu.capture_hidden(model, batch)
    torch.testing.assert_close(early, full_values, atol=2e-6, rtol=2e-5)
    actual = rmu.masked_mse(early, target)
    torch.testing.assert_close(actual, loss.detach(), atol=2e-6, rtol=2e-5)
    actual.backward()
    for name, p in model.named_parameters():
        if name in rmu.ALLOWED:
            torch.testing.assert_close(p.grad, gradients[name], atol=2e-6, rtol=2e-5)
        else:
            assert p.grad is None
    assert record["positions"] == [4, 5, 9, 10, 11, 12]
    assert not model.model.layers[7]._forward_hooks


def test_context_offset_ids_must_match_canonical_tokenization():
    pytest.importorskip("torch")

    class ChangedIDs(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            record = super().__call__(text, **kwargs)
            if kwargs.get("return_offsets_mapping"):
                record["input_ids"][4] += 1
            return record

    item = {"id": "read1", "family": rmu.READING_FAMILIES[0], "prompts": ["hi"], "answer": "yes"}
    with pytest.raises(ValueError, match="offset tokenization differs"):
        rmu.context_batch(ChangedIDs(), item, 0, retain=False)
