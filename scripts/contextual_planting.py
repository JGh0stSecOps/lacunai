"""Bounded development-only contextual planting; never persists checkpoints."""

from __future__ import annotations

import math
from importlib import import_module
from pathlib import Path
from typing import Any

import lacunai.erase.gradient_objectives as objectives
from lacunai.core.model import active_phase
from lacunai.erase.methods import round_to_storage
from lacunai.experimental import tensors
from lacunai.experimental.artifacts import atomic_json
from lacunai.verify.evaluation import classify

PROFILES = {128: 0.002, 512: 0.001, 4864: 0.0003}
MAX_STEPS = 256
MAX_RELATIVE = 0.25
PROJECTED_MAX_STEPS = 512
PROJECTED_RADIUS = 0.24


def validate_inputs(
    rows: list[int], data: dict[str, Any], steps: int, retain_only: bool, projected: bool = False
) -> None:
    if type(projected) is not bool:
        raise ValueError("projected must be boolean")
    maximum = PROJECTED_MAX_STEPS if projected else MAX_STEPS
    if projected and len(rows) != 4864:
        raise ValueError("projected profile requires all 4864 rows")
    if type(retain_only) is not bool or type(steps) is not int or not 1 <= steps <= maximum:
        raise ValueError("invalid contextual planting step/mode contract")
    if not retain_only and steps != maximum:
        raise ValueError("planting requires its frozen profile step maximum")
    if (
        len(rows) not in PROFILES
        or any(type(row) is not int or row < 0 for row in rows)
        or len(set(rows)) != len(rows)
    ):
        raise ValueError("invalid contextual row profile")
    for key, count in (("development", 64), ("retain_development", 32)):
        values = data.get(key)
        if (
            not isinstance(values, list)
            or len(values) != count
            or any(
                not isinstance(item, dict)
                or any(
                    not isinstance(item.get(field), str) or not item[field].strip()
                    for field in ("prompt", "answer")
                )
                for item in values
            )
        ):
            raise ValueError("invalid development-only contextual corpus")


def bounded_relative(current: Any, original: Any, unselected: Any) -> float:
    torch = import_module("torch")
    if current.shape != original.shape or not bool(torch.isfinite(current).all()):
        raise ValueError("invalid contextual proposal shape/values")
    if not torch.equal(current[unselected], original[unselected]):
        raise ValueError("unselected planting rows changed")
    denominator = tensors._square_norm(original)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("zero original tensor norm")
    relative = math.sqrt(tensors._square_norm(current - original) / denominator)
    if not math.isfinite(relative) or relative > MAX_RELATIVE:
        raise ValueError("contextual proposal exceeds relative-norm bound")
    return relative


def project_update(current: Any, original: Any, unselected: Any) -> dict[str, float]:
    """Project an FP32 selected-row update; leave Adam moments untouched."""
    torch = import_module("torch")
    if (
        current.dtype != torch.float32
        or original.dtype != torch.float32
        or current.device.type != "cpu"
        or original.device.type != "cpu"
        or current.shape != original.shape
        or current.ndim != 2
        or unselected.dtype != torch.bool
        or unselected.shape != (current.shape[0],)
        or not bool(torch.isfinite(current).all())
        or not bool(torch.isfinite(original).all())
    ):
        raise ValueError("invalid projection tensors")
    if not torch.equal(current[unselected], original[unselected]):
        raise ValueError("unselected planting rows changed before projection")
    denominator = tensors._square_norm(original)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("zero/nonfinite original projection norm")
    with torch.no_grad():
        delta = current - original
        norm = tensors._square_norm(delta)
        ratio = math.sqrt(norm / denominator)
        if not math.isfinite(ratio):
            raise ValueError("nonfinite pre-projection norm")
        scale = min(1.0, PROJECTED_RADIUS / ratio) if ratio > 0 else 1.0
        if scale < 1.0:
            current.copy_(original + delta * scale)
            current[unselected] = original[unselected]
        post = bounded_relative(current, original, unselected)
        # FP32 reconstruction can differ from the mathematical radius by roundoff.
        if post > PROJECTED_RADIUS + 1e-7:
            raise ValueError("projected radius invariant failed")
    return {
        "pre_projection_relative": ratio,
        "projection_scale": scale,
        "post_projection_relative": post,
    }


def train(
    runtime: Any,
    name: str,
    rows: list[int],
    data: dict[str, Any],
    budget: Any,
    log_dir: Path,
    *,
    retain_only: bool,
    steps: int = MAX_STEPS,
    projected: bool = False,
) -> tuple[Any, int]:
    """Return rounded proposal/completed steps; restore original even after failure."""
    validate_inputs(rows, data, steps, retain_only, projected)
    torch = import_module("torch")
    parameters = dict(runtime.model.named_parameters())
    parameter = parameters[name]
    if (
        not name.endswith(".mlp.up_proj.weight")
        or parameter.ndim != 2
        or max(rows) >= parameter.shape[0]
        or parameter.dtype != torch.float32
        or parameter.device.type != "cpu"
        or runtime.config.seed != 42
        or runtime.config.torch_threads != 2
    ):
        raise ValueError("contextual tensor/runtime contract differs")
    if any(p.grad is not None for p in parameters.values()):
        raise ValueError("preexisting gradients refused")
    # Qualify every batch before changing any trainable flag or parameter.
    batches = {
        key: [
            objectives.answer_batch(
                runtime.tokenizer,
                {"prompts": [item["prompt"]], "answer": item["answer"]},
                runtime.config.max_prompt_tokens,
            )
            for item in data[key]
        ]
        for key in ("development", "retain_development")
    }
    original = parameter.detach().clone()
    unselected = torch.ones(parameter.shape[0], dtype=torch.bool)
    unselected[rows] = False
    flags = {key: value.requires_grad for key, value in parameters.items()}
    training_flags = [(module, module.training) for module in runtime.model.modules()]
    reference = {"original_tensors": {name: original}}
    trace: list[dict[str, Any]] = []
    completed = 0
    projection_count = 0
    # Validate storage support and source representability before mutation.
    if not torch.equal(round_to_storage({name: original}, runtime.storage_dtypes)[name], original):
        raise ValueError("original differs from declared source precision")
    try:
        runtime.model.eval()
        for value in parameters.values():
            value.requires_grad_(False)
        parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW([parameter], lr=PROFILES[len(rows)], weight_decay=0)
        runtime.progress("contextual_training", completed=0, total=steps, unit="steps")
        for step in range(steps):
            budget.check()
            runtime.check()
            retain_index = step % 32
            primary_key = "retain_development" if retain_only else "development"
            primary_index = step % len(batches[primary_key])
            retain_batch = batches["retain_development"][retain_index]
            primary_batch = batches[primary_key][primary_index]
            with active_phase("contextual_training_step"):
                optimizer.zero_grad(set_to_none=True)
                teacher = objectives.teacher_log_probs(runtime.model, reference, retain_batch)
                student = objectives.answer_forward(runtime.model, retain_batch)
                kl = objectives.retain_kl(student, teacher, retain_batch, selected_positions=True)
                retain_nll = objectives.sequence_nll(student, retain_batch, selected_positions=True)
                retain_nll = retain_nll / (retain_batch["labels"] != -100).sum()
                (kl + 0.25 * retain_nll).backward()
                kl_value, retain_value = float(kl.detach()), float(retain_nll.detach())
                del student, teacher, kl, retain_nll
                logits = objectives.answer_forward(runtime.model, primary_batch)
                primary = objectives.sequence_nll(logits, primary_batch, selected_positions=True)
                primary = primary / (primary_batch["labels"] != -100).sum()
                primary.backward()
                primary_value = float(primary.detach())
                del logits, primary
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise ValueError("invalid contextual gradient")
                parameter.grad[unselected] = 0
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)
                )
                optimizer.step()
                projection = project_update(parameter, original, unselected) if projected else None
                if projection is not None and projection["projection_scale"] < 1.0:
                    projection_count += 1
                relative = bounded_relative(parameter, original, unselected)
                rounded = round_to_storage({name: parameter.detach()}, runtime.storage_dtypes)[name]
                stored_relative = bounded_relative(rounded, original, unselected)
            completed = step + 1
            trace.append(
                {
                    "step": completed,
                    "primary_index": primary_index,
                    "retain_index": retain_index,
                    "primary_mean_answer_nll": primary_value,
                    "retain_mean_answer_nll": retain_value,
                    "retain_kl": kl_value,
                    "gradient_norm_before_clip": gradient_norm,
                    "relative_frobenius": relative,
                    "stored_relative_frobenius": stored_relative,
                    "learning_rate": PROFILES[len(rows)],
                    "retain_only": retain_only,
                    "projected": projected,
                    "pre_projection_relative": (
                        projection["pre_projection_relative"] if projection else relative
                    ),
                    "projection_scale": projection["projection_scale"] if projection else 1.0,
                    "projection_count": projection_count,
                }
            )
            budget.check(1024 * 1024)
            atomic_json(log_dir / "training-trace.json", trace)
            runtime.progress("contextual_training", completed=completed, total=steps, unit="steps")
            if not retain_only and completed % 64 == 0:
                current = parameter.detach().clone()
                observations = []
                try:
                    with torch.no_grad():
                        parameter.copy_(rounded)
                    for index, item in enumerate(data["development"]):
                        budget.check()
                        runtime.check()
                        generation = runtime.generate(item["prompt"])
                        observations.append(
                            {
                                **item,
                                "generation": generation,
                                **classify(
                                    generation["output"], item["answer"], generation["truncated"]
                                ),
                            }
                        )
                        runtime.progress(
                            "contextual_development", completed=index + 1, total=64, unit="prompts"
                        )
                finally:
                    with torch.no_grad():
                        parameter.copy_(current)
                recovered = all(item["classification"] == "recovered" for item in observations)
                budget.check(1024 * 1024)
                atomic_json(
                    log_dir / f"development-{completed}.json",
                    {
                        "rows": observations,
                        "all_recovered": recovered,
                        "precision": "source-storage-roundtrip",
                    },
                )
                if recovered:
                    break
        proposed = round_to_storage({name: parameter.detach()}, runtime.storage_dtypes)[
            name
        ].clone()
        bounded_relative(proposed, original, unselected)
        return proposed, completed
    except BaseException as exc:
        runtime.progress(
            "contextual_training",
            state="failed",
            completed=completed,
            total=steps,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        with torch.no_grad():
            parameter.copy_(original)
        for key, value in parameters.items():
            value.requires_grad_(flags[key])
            value.grad = None
        for module, training in training_flags:
            module.training = training
        if not torch.equal(parameter, original):
            raise RuntimeError("contextual original tensor restoration failed")
