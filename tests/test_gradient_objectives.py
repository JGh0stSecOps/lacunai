"""Contract checks and optional actual tensor tests; no pretrained model required."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from lacunai.erase import gradient_objectives as g


def contract():
    spec = {**g.FIXED, "mode": "npo_kl", "allowed_tensors": list(g.ALLOWED)}
    inner = {
        "edits": {
            "allowed_tensors": list(g.ALLOWED),
            "expected_shape": [896, 4864],
            "max_tensor_relative_frobenius": "0.0195",
            "max_model_relative_frobenius": "0.0024",
        }
    }
    return spec, inner


@pytest.mark.parametrize("mode", ["npo_kl", "retain_kl"])
@pytest.mark.parametrize("rate", ["0.01", "0.05", "0.1", "0.2"])
def test_exact_contract(mode, rate):
    spec, inner = contract()
    spec["mode"] = mode
    spec["learning_rate"] = rate
    g.validate_spec(spec, inner)


@pytest.mark.parametrize(
    "key,value",
    [
        ("steps", True),
        ("steps", 61),
        ("learning_rate", 0.01),
        ("beta", "0.2"),
        ("retain_objective", "KL(current||reference)"),
        ("allowed_tensors", []),
        ("mode", "gradient_ascent"),
        ("unknown", "x"),
    ],
)
def test_changed_spec_rejected(key, value):
    spec, inner = contract()
    spec[key] = value
    with pytest.raises(ValueError):
        g.validate_spec(spec, inner)


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_model_relative_frobenius", "0.0025"),
        ("max_tensor_relative_frobenius", "0.02"),
        ("expected_shape", [4864, 896]),
        ("allowed_tensors", []),
    ],
)
def test_inner_projection_contract_rejected(key, value):
    spec, inner = contract()
    inner["edits"][key] = value
    with pytest.raises(ValueError, match="projection"):
        g.validate_spec(spec, inner)


def test_failed_contract_ends_progress_and_freezes_runtime():
    events = []
    watch = SimpleNamespace(progress=lambda phase, **fields: events.append((phase, fields)))
    runtime = SimpleNamespace(model=SimpleNamespace(parameters=lambda: []))
    with pytest.raises(ValueError):
        g.train(runtime, {}, {}, {}, {}, watch)
    assert events[0][1]["state"] == "active"
    assert events[-1][1]["state"] == "failed"
    assert "ValueError" in events[-1][1]["error"]


def test_npo_sign_and_saturation():
    torch = pytest.importorskip("torch")
    current = torch.tensor(3.0, requires_grad=True)
    g.npo_loss(current, 3.0, 0.1).backward()
    assert current.grad.item() == pytest.approx(-1.0)
    far = torch.tensor(103.0, requires_grad=True)
    g.npo_loss(far, 3.0, 0.1).backward()
    assert -0.001 < far.grad.item() < 0
    assert far.grad.abs() < current.grad.abs()


def batch(torch):
    ids = torch.tensor([[0, 0, 1, 2]])
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": torch.tensor([[-100, -100, 1, 2]]),
    }


def test_kl_direction_answer_mask_and_eos():
    torch = pytest.importorskip("torch")
    b = batch(torch)
    student = torch.tensor(
        [[[90.0, -90.0, 0.0], [0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [9.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    teacher = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]).log()
    loss = g.retain_kl(student, teacher, b)
    current = student[0, 1:3].log_softmax(-1)
    expected = (teacher.exp() * (teacher - current)).sum(-1).mean()
    reverse = (current.exp() * (current - teacher)).sum(-1).mean()
    assert loss.item() == pytest.approx(expected.item())
    assert abs(loss.item() - reverse.item()) > 0.01
    loss.backward()
    assert torch.count_nonzero(student.grad[0, 0]) == 0
    assert torch.count_nonzero(student.grad[0, 3]) == 0
    assert torch.count_nonzero(student.grad[0, 1:3]) > 0
    changed = student.detach().clone()
    changed[0, 2, 2] += 3
    assert g.retain_kl(changed, teacher, b).item() != pytest.approx(loss.item())


def test_teacher_restores_current_on_exception_and_uses_original():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2, bias=False)
    original = model.weight.detach().clone()
    reference = {"original_tensors": {"weight": original}}
    x = torch.tensor([[1.0, 2.0]])
    expected = model(x).detach().clone()
    with torch.no_grad():
        model.weight.add_(5.0)
    current = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="injected"), g.original_teacher(model, reference):
        assert torch.equal(model(x), expected)
        assert not torch.is_grad_enabled()
        raise RuntimeError("injected")
    assert torch.equal(model.weight, current)
    with g.original_teacher(model, reference):
        assert torch.equal(model(x), expected)
    assert torch.equal(model.weight, current)


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, *args, **kwargs):
        return "p:"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) % 7 + 3 for c in text]}


def test_sum_nll_matches_existing_loss_including_eos_and_cap():
    torch = pytest.importorskip("torch")
    tokenizer = Tokenizer()
    item = {"prompts": ["question"], "answer": "yes"}
    b = g.answer_batch(tokenizer, item, 128)
    assert b["labels"].tolist() == [[-100, -100, 5, 6, 6, 2]]
    logits = torch.arange(60, dtype=torch.float32).reshape(1, 6, 10) / 13

    class Model:
        def __call__(self, input_ids, labels, **kwargs):
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 10), labels[:, 1:].reshape(-1), ignore_index=-100
            )
            return SimpleNamespace(loss=loss)

    mean = g.tensors._answer_loss(Model(), tokenizer, item, 128)
    assert g.sequence_nll(logits, b).item() == pytest.approx(mean.item() * 4)
    with pytest.raises(ValueError, match="cap"):
        g.answer_batch(tokenizer, item, 5)
    bad = deepcopy(b)
    bad["labels"].fill_(-100)
    with pytest.raises(ValueError, match="empty"):
        g.sequence_nll(logits, bad)


def test_schedule_covers_all_families_in_original_order():
    for families, per_family in [(5, 12), (4, 15)]:
        items = [{"id": f"{f}:{i}", "family": str(f)} for f in range(families) for i in range(16)]
        schedule = g.family_schedule(items, 60)
        assert schedule["family_order"] == list(map(str, range(families)))
        assert schedule["family_counts"] == {str(f): per_family for f in range(families)}
        assert schedule["indices"] == [
            f * 16 + i for i in range(per_family) for f in range(families)
        ]
        assert schedule["item_ids"] == [items[i]["id"] for i in schedule["indices"]]
        assert schedule["families"] == [items[i]["family"] for i in schedule["indices"]]


def test_schedule_preserves_interleaved_order_and_cycles_per_family():
    items = [{"id": "b1", "family": "b"}, {"id": "a1", "family": "a"}, {"id": "b2", "family": "b"}]
    assert g.family_schedule(items, 7)["item_ids"] == ["b1", "a1", "b2", "a1", "b1", "a1", "b2"]


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"family": "x"}],
        [{"id": "x", "family": ""}],
        [{"id": "x", "family": "a"}, {"id": "x", "family": "b"}],
    ],
)
def test_schedule_refuses_ambiguous_identity(items):
    with pytest.raises(ValueError):
        g.family_schedule(items, 60)


def test_qwen_selected_head_matches_full_nll_kl_and_parameter_gradients():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    config = transformers.Qwen2Config(
        vocab_size=19,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen2ForCausalLM(config).float().eval()
    b = batch(torch)
    teacher = torch.arange(38, dtype=torch.float32).reshape(2, 19).div(17).log_softmax(-1)

    def compute(selected):
        model.zero_grad(set_to_none=True)
        logits = (
            g.answer_forward(model, b)
            if selected
            else model(
                input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False
            ).logits
        )
        nll = g.sequence_nll(logits, b, selected_positions=selected)
        kl = g.retain_kl(logits, teacher, b, selected_positions=selected)
        (g.npo_loss(nll, 6.0, 0.1) + kl).backward()
        gradients = {
            name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
        }
        return nll.detach(), kl.detach(), gradients

    full_nll, full_kl, full_grads = compute(False)
    selected_nll, selected_kl, selected_grads = compute(True)
    torch.testing.assert_close(selected_nll, full_nll, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(selected_kl, full_kl, atol=2e-6, rtol=2e-5)
    for name in full_grads:
        torch.testing.assert_close(selected_grads[name], full_grads[name], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize(
    "rate",
    [
        "0.3",
        "1",
        "0",
        "-0.01",
        "0.10",
        "1e-1",
        "NaN",
        "Infinity",
        "-Infinity",
        "nan",
        "inf",
        "",
        0.01,
        0.05,
        0.1,
        0.2,
        float("nan"),
        float("inf"),
        True,
        None,
        [],
        {},
    ],
)
def test_backend_learning_rate_enum_rejects_unapproved_values(rate):
    spec, inner = contract()
    spec["learning_rate"] = rate
    with pytest.raises(ValueError, match="learning rate"):
        g.validate_spec(spec, inner)


def test_undial_mask_eos_and_raw_logit_equivalence():
    torch = pytest.importorskip("torch")
    from lacunai.erase.gradient_objectives import undial_loss

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
    }
    raw = torch.tensor([[0.2, 0.1, -0.2, 2.0, 0.3], [0.4, 0.1, 0.2, 0.3, 1.7]])
    teacher = torch.log_softmax(raw, dim=-1).requires_grad_(True)
    before = teacher.detach().clone()
    student = torch.tensor(
        [[[0.4, 0.2, 0.1, 1.9, 0.3], [0.2, 0.1, 0.3, 0.5, 1.1]]], requires_grad=True
    )
    observed = undial_loss(student, teacher, batch, 4, selected_positions=True)
    adjusted = raw.clone()
    adjusted[0, 3] -= 3
    expected_probs = torch.softmax(adjusted, dim=-1)
    assert torch.allclose(expected_probs[1], torch.softmax(raw[1], dim=-1))
    expected = -(expected_probs * torch.log_softmax(student[0], dim=-1)).sum(-1).mean()
    assert torch.allclose(observed, expected, atol=1e-7, rtol=1e-6)
    full = torch.zeros(1, 4, 5)
    full[:, 1:3] = student.detach()
    assert torch.allclose(undial_loss(full, teacher, batch, 4), observed)
    observed.backward()
    assert student.grad is not None and teacher.grad is None
    assert torch.equal(teacher, before)
    changed = dict(batch, labels=torch.tensor([[-100, -100, -7, 4]]))
    with pytest.raises(ValueError, match="labels"):
        undial_loss(student, teacher, changed, 4, selected_positions=True)


def test_undial_demotion_guard_removal_control():
    import inspect

    import lacunai.erase.gradient_objectives as kernels

    torch = pytest.importorskip("torch")
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
    }
    teacher = torch.log_softmax(
        torch.tensor([[0.2, 0.1, 0.3, 2.0, 0.4], [0.2, 0.3, 0.1, 0.4, 2.0]]), dim=-1
    )
    student = torch.tensor([[[0.1, 0.2, 0.3, 2.0, 0.4], [0.2, 0.1, 0.3, 0.4, 2.0]]])
    actual = kernels.undial_loss(student, teacher, batch, 4, selected_positions=True)
    source = inspect.getsource(kernels.undial_loss)
    assert "adjusted[indices[demoted], targets[demoted]] -= 3.0" in source
    namespace = dict(vars(kernels))
    exec(source.replace("adjusted[indices[demoted], targets[demoted]] -= 3.0", "pass"), namespace)
    removed = namespace["undial_loss"](student, teacher, batch, 4, selected_positions=True)
    assert not torch.allclose(actual, removed, atol=1e-6, rtol=1e-6)
    exec(
        source.replace(
            "demoted = targets != eos_token_id",
            "demoted = torch.ones_like(targets, dtype=torch.bool)",
        ),
        namespace,
    )
    wrong_eos = namespace["undial_loss"](student, teacher, batch, 4, selected_positions=True)
    assert not torch.allclose(actual, wrong_eos, atol=1e-6, rtol=1e-6)
