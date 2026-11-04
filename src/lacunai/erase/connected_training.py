"""Balanced connected-fact planting proposals; callbacks never imply qualification."""

import math
from collections import Counter
from importlib import import_module
from typing import Any

from lacunai.erase import gradient_objectives as kernels
from lacunai.erase.factoid_gradient import _hashes, _relative, training_cap
from lacunai.erase.methods import round_to_storage
from lacunai.experimental import tensors

MAX_STEPS = 512
RADIUS = 0.24
LEARNING_RATE = 0.0003


def validate_rows(primary_rows: Any, utility_rows: Any, steps: int) -> list[int]:
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError("steps must be within 1..512")
    for rows, fields in (
        (primary_rows, {"fact_id", "prompt", "answer"}),
        (utility_rows, {"prompt", "answer"}),
    ):
        if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
            raise ValueError("invalid bounded training rows")
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != fields
                or any(
                    not isinstance(row[key], str) or not row[key].strip() or len(row[key]) > 16384
                    for key in fields
                )
            ):
                raise ValueError("invalid training row")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(primary_rows):
        groups.setdefault(row["fact_id"], []).append(index)
    if len(groups) not in (8, 12) or any(len(rows) != 8 for rows in groups.values()):
        raise ValueError("require eight examples each for eight or twelve facts")
    prompts = [row["prompt"].strip().casefold() for row in primary_rows + utility_rows]
    if len(set(prompts)) != len(prompts):
        raise ValueError("duplicate or overlapping training prompts")
    cycle = [indices[i] for i in range(8) for indices in groups.values()]
    return [cycle[step % len(cycle)] for step in range(steps)]


def project(parameter: Any, original: Any) -> dict[str, float]:
    torch = import_module("torch")
    norm = tensors._square_norm(original)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("invalid original norm")
    with torch.no_grad():
        delta = parameter - original
        ratio = math.sqrt(tensors._square_norm(delta) / norm)
        if not math.isfinite(ratio):
            raise ValueError("nonfinite update")
        scale = min(1.0, RADIUS / ratio) if ratio else 1.0
        if scale < 1:
            parameter.copy_(original + delta * scale)
    return {
        "pre_projection_relative": ratio,
        "projection_scale": scale,
        "post_projection_relative": _relative(parameter, original),
    }


def train_connected(
    runtime: Any,
    primary_rows: list[dict[str, str]],
    utility_rows: list[dict[str, str]],
    *,
    steps: int = MAX_STEPS,
    checkpoint_callback: Any = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Callback(step, rounded replacements, diagnostics) returns True to stop.

    Callback sees rounded weights temporarily loaded. All original state is restored
    before returning a proposal. Utility KL uses original teacher answer positions.
    """

    def emit(**event: Any) -> None:
        runtime.progress("connected_training", **event)
        if progress_callback is not None:
            progress_callback(event)

    emit(state="active", completed=0, total=steps, unit="steps")
    try:
        result = _train(runtime, primary_rows, utility_rows, steps, checkpoint_callback, emit)
    except BaseException as exc:
        emit(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    emit(
        state="complete",
        completed=result["diagnostics"]["completed_steps"],
        total=steps,
        unit="steps",
    )
    return result


def _train(
    runtime: Any, primary: Any, utility: Any, steps: int, callback: Any, emit: Any
) -> dict[str, Any]:
    schedule = validate_rows(primary, utility, steps)
    torch = import_module("torch")
    model = runtime.model
    parameters = dict(model.named_parameters())
    name = f"model.layers.{model.config.num_hidden_layers - 1}.mlp.up_proj.weight"
    if model.config.model_type != "qwen2" or name not in parameters:
        raise ValueError("final Qwen up projection required")
    if runtime.config.seed != 42 or runtime.config.torch_threads != 2:
        raise ValueError("seed/thread contract differs")
    if any(p.grad is not None for p in parameters.values()):
        raise ValueError("preexisting gradients refused")
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in parameters.values()):
        raise ValueError("CPU float32 required")
    cap = training_cap(runtime.config.max_prompt_tokens)
    runtime.check()
    before = _hashes(model)
    parameter = parameters[name]
    original = parameter.detach().clone()
    if not torch.equal(round_to_storage({name: original}, runtime.storage_dtypes)[name], original):
        raise ValueError("original not representable in source precision")
    flags = {key: p.requires_grad for key, p in parameters.items()}
    module_flags = [(m, m.training) for m in model.modules()]
    reference = {"original_tensors": {name: original}}
    trace = []
    error = None
    try:
        model.eval()
        for p in parameters.values():
            p.requires_grad_(False)
        batches = [
            [
                kernels.answer_batch(
                    runtime.tokenizer, {"prompts": [row["prompt"]], "answer": row["answer"]}, cap
                )
                for row in rows
            ]
            for rows in (primary, utility)
        ]
        parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW([parameter], lr=LEARNING_RATE, weight_decay=0)
        for step, index in enumerate(schedule, 1):
            runtime.check()
            optimizer.zero_grad(set_to_none=True)
            ub = batches[1][(step - 1) % len(utility)]
            teacher = kernels.teacher_log_probs(model, reference, ub)
            kl = kernels.retain_kl(
                kernels.answer_forward(model, ub), teacher, ub, selected_positions=True
            )
            kl.backward()
            kl_value = float(kl.detach())
            del teacher, kl
            batch = batches[0][index]
            loss = (
                kernels.sequence_nll(
                    kernels.answer_forward(model, batch), batch, selected_positions=True
                )
                / (batch["labels"] != -100).sum()
            )
            loss.backward()
            loss_value = float(loss.detach())
            del loss
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError("missing/nonfinite planting gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)
            optimizer.step()
            evidence = project(parameter, original)
            rounded = round_to_storage({name: parameter.detach()}, runtime.storage_dtypes)[name]
            rounded_relative = _relative(rounded, original)
            trace.append(
                {
                    "step": step,
                    "primary_index": index,
                    "fact_id": primary[index]["fact_id"],
                    "utility_index": (step - 1) % len(utility),
                    "primary_mean_nll": loss_value,
                    "utility_kl": kl_value,
                    "gradient_norm": float(grad_norm),
                    "rounded_relative": rounded_relative,
                    **evidence,
                }
            )
            emit(state="active", completed=step, total=steps, unit="steps")
            if callback is not None and (step % 64 == 0 or step == steps):
                current = parameter.detach().clone()
                callback_flags = {key: p.requires_grad for key, p in parameters.items()}
                callback_modes = [(m, m.training) for m in model.modules()]
                optimizer.zero_grad(set_to_none=True)
                try:
                    with torch.no_grad():
                        parameter.copy_(rounded)
                    stop = callback(step, {name: rounded.clone()}, {"trace": list(trace)})
                    if type(stop) is not bool:
                        raise ValueError("checkpoint callback must return bool")
                finally:
                    with torch.no_grad():
                        parameter.copy_(current)
                    for key, p in parameters.items():
                        p.requires_grad_(callback_flags[key])
                        p.grad = None
                    for module, training in callback_modes:
                        module.training = training
                if stop:
                    break
        proposal = round_to_storage({name: parameter.detach()}, runtime.storage_dtypes)[
            name
        ].clone()
        _relative(proposal, original)
        return {
            "replacements": {name: proposal},
            "diagnostics": {
                "completed_steps": len(trace),
                "max_steps": steps,
                "trace": trace,
                "schedule": "first-seen-fact-round-robin",
                "fact_counts": dict(Counter(row["fact_id"] for row in trace)),
                "source_hashes": before,
                "training_sequence_tokens": cap,
                "learning_rate": LEARNING_RATE,
                "projection_radius": RADIUS,
                "hard_relative_bound": 0.25,
            },
        }
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            with torch.no_grad():
                parameter.copy_(original)
            for key, p in parameters.items():
                p.requires_grad_(flags[key])
                p.grad = None
            for module, training in module_flags:
                module.training = training
            if _hashes(model) != before:
                raise ValueError("full model restoration failed")
        except BaseException as cleanup:
            if error is not None:
                raise BaseExceptionGroup(
                    "training and restoration failed", [error, cleanup]
                ) from None
            raise
