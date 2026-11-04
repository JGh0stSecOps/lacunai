"""Development-only gradient proposals; no checkpoint writes or method registration.

Adapted from OpenUnlearning src/trainer/unlearn/{npo,grad_diff}.py:
https://github.com/locuslab/open-unlearning/tree/main/src/trainer/unlearn
Single final up projection, AdamW, answer-position KL and bounded storage rounding
are experimental adaptations, not an exact reproduction or evidence of erasure.
"""

from __future__ import annotations

import math
import sys
from importlib import import_module
from typing import Any

import lacunai.erase.gradient_objectives as kernels
from lacunai.erase.methods import round_to_storage
from lacunai.experimental import tensors

MODES = ("npo-kl", "grad-diff-kl")
MAX_STEPS = 128
MAX_RELATIVE = 0.25


def validate_inputs(mode: str, steps: int, forget: Any, retain_development: Any) -> None:
    if mode not in MODES or type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError("invalid factoid mode or step bound")
    for rows in (forget, retain_development):
        if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
            raise ValueError("development example count outside bounds")
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"prompt", "answer"}
                or any(
                    not isinstance(row[field], str)
                    or not row[field].strip()
                    or len(row[field]) > 16384
                    for field in ("prompt", "answer")
                )
            ):
                raise ValueError("invalid development example")
    if {r["prompt"] for r in forget} & {r["prompt"] for r in retain_development}:
        raise ValueError("forget and retain development prompts overlap")


def training_cap(runtime_cap: int) -> int:
    """Keep inference configuration intact while bounding training sequences."""
    if type(runtime_cap) is not int or not 16 <= runtime_cap <= 1024:
        raise ValueError("runtime prompt cap must be an integer within 16..1024")
    return min(runtime_cap, 128)


def _hashes(model: Any) -> dict[str, str]:
    return {name: tensors.tensor_hash(value) for name, value in model.state_dict().items()}


def _relative(value: Any, original: Any) -> float:
    torch = import_module("torch")
    if value.shape != original.shape or not bool(torch.isfinite(value).all()):
        raise ValueError("invalid proposed tensor")
    norm = tensors._square_norm(original)
    if norm <= 0:
        raise ValueError("zero original tensor")
    result = math.sqrt(tensors._square_norm(value - original) / norm)
    if not math.isfinite(result) or result > MAX_RELATIVE:
        raise ValueError("factoid update exceeds relative bound")
    return result


def propose(
    runtime: Any,
    name: str,
    forget: list[dict[str, str]],
    retain_development: list[dict[str, str]],
    *,
    mode: str,
    steps: int = MAX_STEPS,
) -> dict[str, Any]:
    """Return replacement and evidence, with the supplied runtime restored exactly."""
    runtime.progress("factoid_gradient", state="active", completed=0, total=steps, unit="steps")
    try:
        result = _propose(runtime, name, forget, retain_development, mode=mode, steps=steps)
    except BaseException as exc:
        runtime.progress("factoid_gradient", state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    runtime.progress(
        "factoid_gradient", state="complete", completed=steps, total=steps, unit="steps"
    )
    return result


def _propose(
    runtime: Any,
    name: str,
    forget: list[dict[str, str]],
    retain_development: list[dict[str, str]],
    *,
    mode: str,
    steps: int,
) -> dict[str, Any]:
    validate_inputs(mode, steps, forget, retain_development)
    torch = import_module("torch")
    model = runtime.model
    expected = f"model.layers.{model.config.num_hidden_layers - 1}.mlp.up_proj.weight"
    parameters = dict(model.named_parameters())
    if name != expected or name not in parameters or model.config.model_type != "qwen2":
        raise ValueError("only the explicit final Qwen up projection is allowed")
    if runtime.config.seed != 42 or runtime.config.torch_threads != 2:
        raise ValueError("runtime seed/thread contract differs")
    if any(p.grad is not None for p in parameters.values()):
        raise ValueError("preexisting gradients refused")
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in parameters.values()):
        raise ValueError("CPU float32 parameters required")
    parameter = parameters[name]
    if parameter.ndim != 2:
        raise ValueError("up projection must be a matrix")
    inference_prompt_cap = runtime.config.max_prompt_tokens
    cap = training_cap(inference_prompt_cap)
    runtime.check()
    before = _hashes(model)
    original = parameter.detach().clone()
    if not torch.equal(round_to_storage({name: original}, runtime.storage_dtypes)[name], original):
        raise ValueError("source precision differs from stored checkpoint")
    flags = {n: p.requires_grad for n, p in parameters.items()}
    training = [(m, m.training) for m in model.modules()]
    reference = {"original_tensors": {name: original}}
    trace = []
    references = []
    try:
        model.eval()
        for p in parameters.values():
            p.requires_grad_(False)
        batches = [
            [
                kernels.answer_batch(
                    runtime.tokenizer, {"prompts": [r["prompt"]], "answer": r["answer"]}, cap
                )
                for r in rows
            ]
            for rows in (forget, retain_development)
        ]
        with torch.no_grad():
            for i, batch in enumerate(batches[0]):
                runtime.check()
                references.append(
                    float(
                        kernels.sequence_nll(
                            kernels.answer_forward(model, batch), batch, selected_positions=True
                        )
                    )
                )
                runtime.progress(
                    "factoid_reference", completed=i + 1, total=len(batches[0]), unit="examples"
                )
        parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW([parameter], lr=0.001, weight_decay=0)
        for step in range(steps):
            runtime.check()
            optimizer.zero_grad(set_to_none=True)
            fi, ri = step % len(batches[0]), step % len(batches[1])
            fb, rb = batches[0][fi], batches[1][ri]
            teacher = kernels.teacher_log_probs(model, reference, rb)
            logits = kernels.answer_forward(model, rb)
            kl = kernels.retain_kl(logits, teacher, rb, selected_positions=True)
            rnll = kernels.sequence_nll(logits, rb, selected_positions=True)
            rnll = rnll / (rb["labels"] != -100).sum()
            (kl + 0.25 * rnll).backward()
            kl_value, retain_value = float(kl.detach()), float(rnll.detach())
            del teacher, logits, kl, rnll
            nll = kernels.sequence_nll(
                kernels.answer_forward(model, fb), fb, selected_positions=True
            )
            objective = (
                kernels.npo_loss(nll, references[fi], 0.1)
                if mode == "npo-kl"
                else -nll / (fb["labels"] != -100).sum()
            )
            objective.backward()
            loss_value, nll_value = float(objective.detach()), float(nll.detach())
            del objective, nll
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError("invalid factoid gradient")
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)
            )
            optimizer.step()
            relative = _relative(parameter, original)
            proposal = round_to_storage({name: parameter.detach()}, runtime.storage_dtypes)[name]
            stored_relative = _relative(proposal, original)
            trace.append(
                {
                    "step": step + 1,
                    "forget_index": fi,
                    "retain_index": ri,
                    "forget_sequence_nll": nll_value,
                    "forget_objective": loss_value,
                    "retain_kl": kl_value,
                    "retain_mean_nll": retain_value,
                    "gradient_norm_before_clip": gradient_norm,
                    "relative_frobenius": relative,
                    "stored_relative_frobenius": stored_relative,
                }
            )
            runtime.progress("factoid_gradient", completed=step + 1, total=steps, unit="steps")
        if any(
            tensors.tensor_hash(value) != before[n]
            for n, value in model.state_dict().items()
            if n != name
        ):
            raise ValueError("untouched model state changed")
        proposal = proposal.clone()
        if torch.equal(proposal, original):
            raise ValueError("factoid proposal rounded to no-op")
        result = {
            "replacements": {name: proposal},
            "trace": trace,
            "mode": mode,
            "completed": steps,
            "reference_sequence_nll": references,
            "source_hashes": before,
            "source_fingerprint": tensors.source_fingerprint(before),
            "proposal_hash": tensors.tensor_hash(proposal),
            "settings": {
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0,
                "clip": 1.0,
                "beta": 0.1,
                "retain_kl": 1.0,
                "retain_nll": 0.25,
                "max_relative": MAX_RELATIVE,
                "sequence_tokens": cap,
                "trainingsequence_tokens": cap,
                "inference_prompt_cap": inference_prompt_cap,
            },
        }
    finally:
        primary_error = sys.exception()
        try:
            with torch.no_grad():
                parameter.copy_(original)
            for n, p in parameters.items():
                p.requires_grad_(flags[n])
                p.grad = None
            for module, was_training in training:
                module.training = was_training
            if _hashes(model) != before:
                raise RuntimeError("factoid original state restoration failed")
        except BaseException as restore_error:
            if primary_error is not None:
                raise BaseExceptionGroup(
                    "proposal and restoration failed", [primary_error, restore_error]
                ) from None
            raise
    return result
