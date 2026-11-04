from decimal import Decimal

import pytest

from lacunai.erase import gradient_objectives as kernels
from lacunai.verify.likelihood import answer_likelihood


def test_likelihood_matches_full_shifted_answer_and_eos(monkeypatch):
    from tests.test_factoid_gradient import toy

    torch, runtime, _, _ = toy(monkeypatch)
    batch = kernels.answer_batch(runtime.tokenizer, {"prompts": ["test"]}, 128)
    runtime.model.eval()
    with torch.no_grad():
        full = runtime.model(input_ids=batch["input_ids"], use_cache=False).logits
        selected, labels = kernels._answer_logits(full, batch)
        expected = selected.log_softmax(-1).gather(1, labels[:, None]).flatten()
    record = answer_likelihood(runtime, "test", "answer")
    assert record["token_count"] == 2  # fixture's answer and EOS positions
    assert record["token_logprobs"] == pytest.approx(expected.tolist(), abs=1e-6)
    assert Decimal(record["logprob_sum"]) == sum(
        map(lambda x: Decimal(str(x)), record["token_logprobs"])
    )
    assert Decimal(record["mean_logprob"]) * 2 == Decimal(record["logprob_sum"])
    assert all(p.grad is None for p in runtime.model.parameters())


def test_real_answer_batch_boundary_and_eos():
    pytest.importorskip("torch")

    class Tokenizer:
        eos_token_id = 7

        def apply_chat_template(self, *args, **kwargs):
            return "PQ"

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) for c in text]}

    batch = kernels.answer_batch(Tokenizer(), {"prompts": ["p"], "answer": "ab"}, 16)
    assert batch["input_ids"].tolist() == [[80, 81, 97, 98, 7]]
    assert batch["labels"].tolist() == [[-100, -100, 97, 98, 7]]
    with pytest.raises(ValueError, match="token cap"):
        kernels.answer_batch(Tokenizer(), {"prompts": ["p"], "answer": "ab"}, 4)
