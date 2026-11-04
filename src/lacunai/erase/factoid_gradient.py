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

MODES = (
    "npo-kl",
    "grad-diff-kl",
    "npo-kl-repair",
    "grad-diff-kl-repair",
    "undial-kl",
    "undial-kl-repair",
    "grad-diff-kl-repair2",
)
UNDIAL_MODES = ("undial-kl", "undial-kl-repair")
REPAIR_MODES = ("npo-kl-repair", "grad-diff-kl-repair", "undial-kl-repair", "grad-diff-kl-repair2")
MAX_STEPS = 128
MAX_RELATIVE = 0.25


def validate_inputs(mode: str, steps: int, forget: Any, retain_development: Any) -> None:
    if mode not in MODES or type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError("invalid factoid mode or step bound")
    if mode == "grad-diff-kl-repair2" and steps > 112:
        raise ValueError("double repair allows at most 112 base updates")
    if mode in REPAIR_MODES and steps > 120:
        raise ValueError("repair variants allow at most 120 base updates")
    if mode in REPAIR_MODES and (
        not isinstance(retain_development, list) or len(retain_development) != 32
    ):
        raise ValueError("repair variants require 32 retain development examples")
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
            forget_teacher = (
                kernels.teacher_log_probs(model, reference, fb) if mode in UNDIAL_MODES else None
            )
            forget_logits = kernels.answer_forward(model, fb)
            nll = kernels.sequence_nll(forget_logits, fb, selected_positions=True)
            objective = (
                kernels.npo_loss(nll, references[fi], 0.1)
                if mode in ("npo-kl", "npo-kl-repair")
                else -nll / (fb["labels"] != -100).sum()
            )
            if mode in UNDIAL_MODES:
                objective = kernels.undial_loss(
                    forget_logits,
                    forget_teacher,
                    fb,
                    runtime.tokenizer.eos_token_id,
                    selected_positions=True,
                )
            objective.backward()
            loss_value, nll_value = float(objective.detach()), float(nll.detach())
            del objective, nll, forget_logits, forget_teacher
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
        repair_metadata: dict[str, Any] = {}
        if mode in REPAIR_MODES:
            from lacunai.erase.retain_repair import repair

            pre_round_hash = tensors.tensor_hash(parameter)
            with torch.no_grad():
                parameter.copy_(proposal)
            before_repair_hash = tensors.tensor_hash(parameter)
            parameter.grad = None
            repair_trace = repair(
                runtime,
                [name],
                retain_development,
                reference={name: original},
                check=runtime.check,
                bound_tolerance=0,
                phase="retain_repair",
            )
            after_repair_hash = tensors.tensor_hash(parameter)
            proposal = round_to_storage({name: parameter.detach()}, runtime.storage_dtypes)[name]
            _relative(proposal, original)
            repair_metadata = {
                "unlearning_steps": steps,
                "repair_steps": 8,
                "total_steps": steps + 8,
                "repair_trace": repair_trace,
                "pre_repair_unrounded_hash": pre_round_hash,
                "pre_repair_rounded_hash": before_repair_hash,
                "post_repair_unrounded_hash": after_repair_hash,
                "post_repair_rounded_hash": tensors.tensor_hash(proposal),
                "repair_settings": {
                    "optimizer": "SGD",
                    "learning_rate": 0.05,
                    "clip": 1.0,
                    "steps": 8,
                    "sequence_tokens": inference_prompt_cap,
                    "retain_indices": list(range(8)),
                    "bound_reference": "original learned checkpoint",
                },
            }
        if mode == "grad-diff-kl-repair2":
            first_round = {
                key: value
                for key, value in repair_metadata.items()
                if key not in {"unlearning_steps", "total_steps"}
            }
            first_round["round"] = 1
            pre_round_hash = tensors.tensor_hash(parameter)
            with torch.no_grad():
                parameter.copy_(proposal)
            before_repair_hash = tensors.tensor_hash(parameter)
            parameter.grad = None
            second_trace = repair(
                runtime,
                [name],
                retain_development,
                reference={name: original},
                check=runtime.check,
                bound_tolerance=0,
                phase="retain_repair_round_2",
            )
            after_repair_hash = tensors.tensor_hash(parameter)
            proposal = round_to_storage({name: parameter.detach()}, runtime.storage_dtypes)[name]
            _relative(proposal, original)
            second_round = {
                "round": 2,
                "repair_steps": 8,
                "repair_trace": second_trace,
                "pre_repair_unrounded_hash": pre_round_hash,
                "pre_repair_rounded_hash": before_repair_hash,
                "post_repair_unrounded_hash": after_repair_hash,
                "post_repair_rounded_hash": tensors.tensor_hash(proposal),
                "repair_settings": dict(repair_metadata["repair_settings"]),
            }
            repair_metadata.update(
                repair_steps=16,
                total_steps=steps + 16,
                repair_rounds=[first_round, second_round],
                repair_trace=repair_metadata["repair_trace"] + second_trace,
                post_repair_unrounded_hash=after_repair_hash,
                post_repair_rounded_hash=tensors.tensor_hash(proposal),
            )
            repair_metadata["repair_settings"] = {
                **repair_metadata["repair_settings"],
                "steps": 16,
                "rounds": 2,
                "steps_per_round": 8,
                "round_between_rounds": True,
            }
        if any(
            tensors.tensor_hash(value) != before[n]
            for n, value in model.state_dict().items()
            if n != name
        ):
            raise ValueError("untouched model state changed")
        proposal = proposal.clone()
        if torch.equal(proposal, original):
            raise ValueError("factoid proposal rounded to no-op")
        if mode in UNDIAL_MODES:
            repair_metadata["undial"] = {
                "demotion": 3.0,
                "eos_policy": "undemoted teacher distribution distilled",
                "positions": "causal answer and EOS only; prefix excluded",
                "teacher": "frozen planted checkpoint",
                "objective": "mean token soft-target cross entropy",
                "adaptation": (
                    "answer-only single-tensor UnDIAL with retain KL/NLL; not erasure proof"
                ),
            }
        result = {
            "replacements": {name: proposal},
            "trace": trace,
            "mode": mode,
            "completed": steps,
            **repair_metadata,
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
