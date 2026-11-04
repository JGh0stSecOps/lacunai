"""Balanced connected-fact planting proposals; callbacks never imply qualification."""

import math
from collections import Counter
from contextlib import nullcontext
from importlib import import_module
from typing import Any

from lacunai.erase import gradient_objectives as kernels
from lacunai.erase.factoid_gradient import _hashes, _relative, training_cap
from lacunai.erase.methods import round_to_storage
from lacunai.experimental import tensors

MAX_STEPS = 512
RADIUS = 0.24
LEARNING_RATE = 0.0003
UTILITY_NLL_WEIGHT = 0.25
PROFILE_GROUPS = {"connected-12": (8, 12), "connected-5": (4, 5)}
# Opt-in smaller profile permits at most32 development screens over512 updates.
PROFILE_CHECK_EVERY = {"connected-12": 64, "connected-5": 16}


def validate_rows(
    primary_rows: Any, utility_rows: Any, steps: int, *, profile: str = "connected-12"
) -> list[int]:
    if not isinstance(profile, str) or profile not in PROFILE_GROUPS:
        raise ValueError("unknown connected training profile")
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
    if len(groups) not in PROFILE_GROUPS[profile] or any(
        len(rows) != 8 for rows in groups.values()
    ):
        raise ValueError("require eight examples each for the selected profile fact groups")
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


def utility_objective(student: Any, teacher: Any, batch: dict[str, Any]) -> tuple[Any, Any, Any]:
    kl = kernels.retain_kl(student, teacher, batch, selected_positions=True)
    mean_nll = (
        kernels.sequence_nll(student, batch, selected_positions=True)
        / (batch["labels"] != -100).sum()
    )
    return kl, mean_nll, kl + UTILITY_NLL_WEIGHT * mean_nll


def train_connected(
    runtime: Any,
    primary_rows: list[dict[str, str]],
    utility_rows: list[dict[str, str]],
    *,
    steps: int = MAX_STEPS,
    profile: str = "connected-12",
    planting_parameterization: str = "dense",
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
        result = _train(
            runtime,
            primary_rows,
            utility_rows,
            steps,
            checkpoint_callback,
            emit,
            profile,
            planting_parameterization,
        )
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
    runtime: Any,
    primary: Any,
    utility: Any,
    steps: int,
    callback: Any,
    emit: Any,
    profile: str,
    planting_parameterization: str,
) -> dict[str, Any]:
    if type(planting_parameterization) is not str or planting_parameterization not in (
        "dense",
        "lora_rank8",
    ):
        raise ValueError("unknown planting parameterization")
    schedule = validate_rows(primary, utility, steps, profile=profile)
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
    development_checks = 0
    error = None
    adapter = None
    rng = torch.get_rng_state().clone()
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
        if planting_parameterization == "lora_rank8":
            from lacunai.erase.connected_lora import Adapter

            adapter = Adapter(model.get_submodule(name.rsplit(".", 1)[0]), original)
            trainables = adapter.parameters
        else:
            parameter.requires_grad_(True)
            trainables = [parameter]
        optimizer = torch.optim.AdamW(trainables, lr=LEARNING_RATE, weight_decay=0)
        for step, index in enumerate(schedule, 1):
            runtime.check()
            optimizer.zero_grad(set_to_none=True)
            ub = batches[1][(step - 1) % len(utility)]
            with adapter.disabled() if adapter is not None else nullcontext():
                teacher = kernels.teacher_log_probs(model, reference, ub)
            student = kernels.answer_forward(model, ub)
            kl, utility_nll, utility_loss = utility_objective(student, teacher, ub)
            utility_loss.backward()
            kl_value = float(kl.detach())
            utility_nll_value = float(utility_nll.detach())
            del teacher, student, kl, utility_nll, utility_loss
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
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in trainables):
                raise ValueError("missing/nonfinite planting gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_(trainables, 1.0, error_if_nonfinite=True)
            optimizer.step()
            evidence = adapter.project() if adapter is not None else project(parameter, original)
            merged = adapter.merged().detach() if adapter is not None else parameter.detach()
            rounded = round_to_storage({name: merged}, runtime.storage_dtypes)[name]
            rounded_relative = _relative(rounded, original)
            trace.append(
                {
                    "step": step,
                    "primary_index": index,
                    "fact_id": primary[index]["fact_id"],
                    "utility_index": (step - 1) % len(utility),
                    "primary_mean_nll": loss_value,
                    "utility_kl": kl_value,
                    "utility_mean_answer_nll": utility_nll_value,
                    "gradient_norm": float(grad_norm),
                    "rounded_relative": rounded_relative,
                    **evidence,
                }
            )
            emit(state="active", completed=step, total=steps, unit="steps")
            if callback is not None and (step % PROFILE_CHECK_EVERY[profile] == 0 or step == steps):
                development_checks += 1
                current = parameter.detach().clone()
                callback_flags = {key: p.requires_grad for key, p in parameters.items()}
                callback_modes = [(m, m.training) for m in model.modules()]
                optimizer.zero_grad(set_to_none=True)
                try:
                    with torch.no_grad():
                        parameter.copy_(rounded)
                    with adapter.disabled() if adapter is not None else nullcontext():
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
        merged = adapter.merged().detach() if adapter is not None else parameter.detach()
        proposal = round_to_storage({name: merged}, runtime.storage_dtypes)[name].clone()
        _relative(proposal, original)
        return {
            "replacements": {name: proposal},
            "diagnostics": {
                "planting_parameterization": planting_parameterization,
                "adapter": (
                    {
                        "rank": 8,
                        "alpha": 8,
                        "scale": 1,
                        "dropout": 0,
                        "initialization_seed": 42,
                        "initialization": "kaiming_A_zero_B",
                    }
                    if adapter is not None
                    else None
                ),
                "merged_relative": _relative(proposal, original),
                "completed_steps": len(trace),
                "profile": profile,
                "development_check_every": PROFILE_CHECK_EVERY[profile],
                "development_checks": development_checks,
                "maximum_development_checks": (steps + PROFILE_CHECK_EVERY[profile] - 1)
                // PROFILE_CHECK_EVERY[profile],
                "max_steps": steps,
                "trace": trace,
                "schedule": "first-seen-fact-round-robin",
                "fact_counts": dict(Counter(row["fact_id"] for row in trace)),
                "source_hashes": before,
                "training_sequence_tokens": cap,
                "learning_rate": LEARNING_RATE,
                "utility_nll_weight": UTILITY_NLL_WEIGHT,
                "projection_radius": RADIUS,
                "hard_relative_bound": 0.25,
            },
        }
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            if adapter is not None:
                adapter.close()
            if planting_parameterization == "lora_rank8":
                torch.set_rng_state(rng)
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
