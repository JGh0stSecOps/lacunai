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
def test_exact_contract(mode):
    spec, inner = contract()
    spec["mode"] = mode
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
