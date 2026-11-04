"""Full causal answer likelihood, including EOS; no generation or judging."""

from decimal import Decimal, localcontext
from importlib import import_module
from typing import Any

from lacunai.erase import gradient_objectives as objectives


def answer_likelihood(runtime: Any, prompt: str, answer: str) -> dict[str, Any]:
    torch = import_module("torch")
    if any(not isinstance(x, str) or not x.strip() for x in (prompt, answer)):
        raise ValueError("nonempty prompt and answer required")
    runtime.check()
    batch = objectives.answer_batch(
        runtime.tokenizer,
        {"prompts": [prompt], "answer": answer},
        runtime.config.max_prompt_tokens,
    )
    flags = [(module, module.training) for module in runtime.model.modules()]
    try:
        runtime.model.eval()
        with torch.no_grad():
            logits, targets = objectives._answer_logits(
                objectives.answer_forward(runtime.model, batch), batch, selected_positions=True
            )
            values = logits.log_softmax(-1).gather(1, targets[:, None]).flatten()
            if not bool(torch.isfinite(values).all()):
                raise ValueError("nonfinite answer likelihood")
            tokens = [float(value) for value in values.tolist()]
    finally:
        for module, training in flags:
            module.training = training
    with localcontext() as context:
        context.prec = 50
        total = sum((Decimal(str(value)) for value in tokens), Decimal(0))
        mean = total / Decimal(len(tokens))
    return {
        "token_logprobs": tokens,
        "logprob_sum": str(total),
        "token_count": len(tokens),
        "mean_logprob": str(mean),
    }
