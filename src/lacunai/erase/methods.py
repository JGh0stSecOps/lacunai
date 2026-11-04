"""Tensor operators return proposals; durable checkpoint writes belong to the executor."""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any

from lacunai.experimental import tensors
from lacunai.map.localize import record_digest

REPAIR_METHODS = {
    "npo-kl-repair",
    "grad-diff-kl-repair",
    "undial-kl-repair",
    "grad-diff-kl-repair2",
}
FACTOID_METHODS = {"npo-kl", "grad-diff-kl", "undial-kl"} | REPAIR_METHODS


def base_step_limit(method: str) -> int:
    return 112 if method == "grad-diff-kl-repair2" else 120 if method in REPAIR_METHODS else 128


METHOD_INFO = {
    "grad-diff-kl-repair2": {
        "cost": "experimental AdamW plus two eight-step retain-only repair rounds",
        "risk": "second repair may restore target; rounding and independent evaluation required",
    },
    "undial-kl": {
        "cost": "experimental fixed-demotion self-distillation with retain KL",
        "risk": "UnDIAL adaptation, not paper reproduction; collateral and recovery risk",
    },
    **{
        name: {
            "cost": "experimental bounded AdamW plus eight retain-only SGD repair updates",
            "risk": "repair may restore target recovery; independent verification required",
        }
        for name in REPAIR_METHODS - {"grad-diff-kl-repair2"}
    },
    "npo-kl": {
        "cost": "experimental bounded AdamW with original-reference forwards",
        "risk": "answer likelihood suppression may preserve semantic recovery; collateral risk",
    },
    "grad-diff-kl": {
        "cost": "experimental bounded AdamW with retain KL",
        "risk": "gradient ascent may damage capabilities; not proof of deletion",
    },
    "ablate-neurons": {
        "cost": "one selected up-projection copy",
        "risk": "shared-feature collateral damage",
    },
    "ablate-direction": {
        "cost": "one rank-one projection per selected matrix",
        "risk": "behavioral suppression; direction may encode unrelated capabilities",
    },
    "rank-one": {
        "cost": "bounded gradient steps plus rank-one update",
        "risk": "ROME-inspired diagonal approximation; factual rewrite is not deletion",
    },
    "historical-columns": {
        "cost": "original pilot validation and column attenuation",
        "risk": "outgoing contribution changes do not silence the feature",
    },
}


def _selected(runtime: Any, localization: dict[str, Any]) -> list[dict[str, Any]]:
    if getattr(runtime.model.config, "model_type", None) not in {"qwen2", "llama"}:
        raise ValueError("unsupported gated-MLP architecture")
    if localization.get("checkpoint_fingerprint") != runtime.fingerprint:
        raise ValueError("localization checkpoint identity mismatch")
    selected = localization.get("selected")
    if not isinstance(selected, list) or not 1 <= len(selected) <= 8:
        raise ValueError("no bounded, positively localized selection")
    evidence = {row["layer"]: row for row in localization.get("layers", [])}
    used = set()
    for row in selected:
        layer, neurons = row.get("layer"), row.get("neurons")
        if type(layer) is not int or layer not in runtime.layer_ids or layer in used:
            raise ValueError("invalid or duplicate localized layer")
        used.add(layer)
        proof = evidence.get(layer, {})
        effect = proof.get("mean_effect")
        if type(effect) not in (int, float) or not math.isfinite(effect) or effect <= 0:
            raise ValueError("selected layer lacks positive causal evidence")
        width = runtime.mlp(layer).down_proj.weight.shape[1]
        if not isinstance(neurons, list) or len(neurons) > 128 or len(set(neurons)) != len(neurons):
            raise ValueError("invalid neuron selection")
        supported = {
            n["neuron"]
            for n in proof.get("neurons", [])
            if isinstance(n.get("mean_effect"), (int, float)) and n["mean_effect"] > 0
        }
        if any(type(n) is not int or not 0 <= n < width or n not in supported for n in neurons):
            raise ValueError("selected neuron lacks valid positive causal evidence")
    return selected


def _module_name(runtime: Any, layer: int) -> str:
    mlp = runtime.mlp(layer)
    names = [name for name, module in runtime.model.named_modules() if module is mlp]
    if len(names) != 1:
        raise ValueError("MLP module identity is not unique")
    return str(names[0])


def touched_tensors(runtime: Any, method: str, localization: dict[str, Any]) -> list[str]:
    if method not in METHOD_INFO:
        raise ValueError("unknown weight-editing method")
    selected = _selected(runtime, localization)
    if method in FACTOID_METHODS:
        final = getattr(runtime.model.config, "num_hidden_layers", 0) - 1
        if (
            runtime.model.config.model_type != "qwen2"
            or len(selected) != 1
            or selected[0]["layer"] != final
        ):
            raise ValueError(
                "gradient methods require only the positively localized final Qwen layer"
            )
        name = f"{_module_name(runtime, final)}.up_proj.weight"
        if name != f"model.layers.{final}.mlp.up_proj.weight":
            raise ValueError("gradient tensor identity differs from allowlist")
        return [name]
    names = []
    for row in selected:
        layer = row["layer"]
        prefix = _module_name(runtime, layer)
        if method in {"ablate-neurons", "historical-columns"} and not row["neurons"]:
            raise ValueError("neuron operator has no positively localized channels")
        projection = "up_proj" if method == "ablate-neurons" else "down_proj"
        names.append(f"{prefix}.{projection}.weight")
        if (
            method == "ablate-neurons"
            and getattr(runtime.mlp(layer).up_proj, "bias", None) is not None
        ):
            names.append(f"{prefix}.up_proj.bias")
    return names


def _validate_bound(maximum: float) -> None:
    if type(maximum) not in (float, int) or not math.isfinite(maximum) or not 0 < maximum <= 1:
        raise ValueError("invalid relative-change bound")


def validate_replacements(
    originals: dict[str, Any], replacements: dict[str, Any], maximum: float
) -> dict[str, Any]:
    _validate_bound(maximum)
    if set(originals) != set(replacements):
        raise ValueError("replacement tensor names differ from allowed snapshot")
    diffs: dict[str, Any] = {}
    for name, value in replacements.items():
        original = originals[name]
        if list(value.shape) != list(original.shape) or value.dtype != original.dtype:
            raise ValueError("replacement shape or dtype changed")
        tensors._square_norm(value)
        base_squared = tensors._square_norm(original)
        delta_squared = tensors._square_norm(value - original)
        if base_squared == 0:
            if delta_squared:
                raise ValueError("nonzero change to zero-norm tensor has no relative bound")
            relative = 0.0
        else:
            relative = math.sqrt(delta_squared / base_squared)
        if relative > maximum + 1e-9:
            raise ValueError(f"relative tensor change exceeds bound: {name}")
        diffs[name] = {
            "relative_frobenius": relative,
            "delta_norm_squared": delta_squared,
            "source_sha256": tensors.tensor_hash(original),
            "edited_sha256": tensors.tensor_hash(value),
        }
    if not any(row["delta_norm_squared"] > 0 for row in diffs.values()):
        raise ValueError("edit proposal is a no-op")
    return diffs


def _historical(
    runtime: Any, selected: list[dict[str, Any]], strength: float, maximum: float
) -> dict[str, Any]:
    torch = import_module("torch")
    names = [f"{_module_name(runtime, row['layer'])}.down_proj.weight" for row in selected]
    state = runtime.model.state_dict()
    shapes = {tuple(state[name].shape) for name in names}
    if len(shapes) != 1:
        raise ValueError("historical operator requires equal projection shapes")
    contract = {
        "edits": {
            "allowed_tensors": names,
            "expected_shape": list(next(iter(shapes))),
            "operation": "attenuate_diagnostic_columns",
            "max_tensors": len(names),
            "max_columns_per_tensor": 128,
            "strength_presets": {"chosen": strength},
            "max_tensor_relative_frobenius": maximum,
            "max_model_relative_frobenius": maximum,
        }
    }
    reference = tensors.capture_reference(runtime.model, contract)
    diagnostics = {
        "source_fingerprint": reference["fingerprint"],
        "columns": [
            {"id": f"l{row['layer']}c{neuron}", "tensor": name, "column": neuron}
            for name, row in zip(names, selected, strict=True)
            for neuron in row["neurons"]
        ],
    }
    spec = {
        "operation": "attenuate_diagnostic_columns",
        "source_fingerprint": reference["fingerprint"],
        "rationale": "Explicitly selected causal channels; preserved historical numerical kernel.",
        "edits": [
            {
                "tensor": name,
                "column_ids": [f"l{row['layer']}c{n}" for n in row["neurons"]],
                "strength": "chosen",
            }
            for name, row in zip(names, selected, strict=True)
        ],
    }
    try:
        tensors.apply_edit(runtime.model, spec, diagnostics, contract, reference)
        return {name: state[name].detach().clone() for name in names}
    finally:
        with torch.no_grad():
            for name, value in reference["original_tensors"].items():
                state[name].copy_(value)
        if tensors.validate_state(runtime.model, reference, contract)["changed_tensor_count"]:
            raise ValueError("historical proposal did not restore temporary model state")


def _rank_one(
    runtime: Any, layer: int, record: dict[str, Any], prompts: list[str], rewrite: str, steps: int
) -> tuple[Any, dict[str, Any]]:
    torch = import_module("torch")
    module = runtime.mlp(layer).down_proj
    weight = module.weight.detach()
    key = torch.tensor(record["key_mean"], dtype=weight.dtype)
    covariance = torch.tensor(record["key_covariance_diag"], dtype=weight.dtype)
    if key.shape != (weight.shape[1],) or covariance.shape != key.shape:
        raise ValueError("rank-one key/covariance dimensions differ")
    if (
        not bool(torch.isfinite(key).all())
        or not bool(torch.isfinite(covariance).all())
        or bool((covariance < 0).any())
    ):
        raise ValueError("invalid rank-one key/covariance")
    ridge = 1e-3
    q = key / (covariance + ridge)
    denominator = torch.dot(q, key)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-12:
        raise ValueError("degenerate rank-one key")
    if any(parameter.requires_grad for parameter in runtime.model.parameters()):
        raise ValueError("rank-one optimization requires frozen model parameters")
    delta = torch.nn.Parameter(torch.zeros(weight.shape[0], dtype=weight.dtype))
    optimizer = torch.optim.SGD([delta], lr=0.1)
    trace = []
    runtime.progress("rank_one_value", completed=0, total=steps, unit="steps")
    for step in range(steps):
        runtime.check()
        prompt = prompts[step % len(prompts)]
        position = runtime.encode(prompt).shape[-1] - 1
        calls = 0

        def hook(layer_module: Any, inputs: Any, output: Any, position: int = position) -> Any:
            nonlocal calls
            del layer_module, inputs
            calls += 1
            if (
                calls != 1
                or output.ndim != 3
                or output.shape[0] != 1
                or position >= output.shape[1]
            ):
                raise ValueError("rank-one value optimization hook shape/call differs")
            changed = output.clone()
            changed[0, position] = output[0, position] + delta
            return changed

        handle = module.register_forward_hook(hook)
        try:
            optimizer.zero_grad(set_to_none=True)
            loss = runtime.loss(prompt, rewrite) + 1e-4 * delta.square().mean()
            if calls != 1 or not bool(torch.isfinite(loss)):
                raise ValueError("invalid rank-one optimization loss or missing hook")
            loss.backward()
            if delta.grad is None or not bool(torch.isfinite(delta.grad).all()):
                raise ValueError("invalid rank-one value gradient")
            torch.nn.utils.clip_grad_norm_([delta], 1.0, error_if_nonfinite=True)
            optimizer.step()
            trace.append({"step": step + 1, "loss": float(loss.detach())})
            runtime.progress("rank_one_value", completed=step + 1, total=steps, unit="steps")
        finally:
            handle.remove()
    residual = torch.outer(delta.detach(), q) / denominator
    return residual, {
        "trace": trace,
        "ridge": ridge,
        "learning_rate": 0.1,
        "gradient_norm_max": 1.0,
        "value_l2_coefficient": 1e-4,
        "covariance": "diagonal population covariance of final-prefill background features",
        "key_pooling": "final prefill token",
        "denominator": float(denominator),
        "method": "ROME-inspired bounded value optimization; not full ROME or MEMIT",
    }


def round_to_storage(
    replacements: dict[str, Any], storage_dtypes: dict[str, str]
) -> dict[str, Any]:
    """Return values exactly representable in the persisted checkpoint precision."""
    torch = import_module("torch")
    dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    if any(storage_dtypes.get(name) not in dtypes for name in replacements):
        raise ValueError("missing or unsupported checkpoint storage dtype")
    return {
        name: value.to(dtype=dtypes[storage_dtypes[name]]).to(dtype=value.dtype)
        for name, value in replacements.items()
    }


def apply_method(
    runtime: Any,
    method: str,
    localization: dict[str, Any],
    fingerprint: dict[str, Any],
    *,
    strength: float = 1.0,
    max_relative_change: float = 0.25,
    rewrite: str = "unknown",
    steps: int = 12,
    development_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_bound(max_relative_change)
    if method in FACTOID_METHODS:
        from lacunai.erase.factoid_data import validate_record

        if strength != 1 or max_relative_change != 0.25:
            raise ValueError("gradient methods require strength one and relative bound .25")
        development_data = validate_record(development_data, runtime.fingerprint)
    elif development_data is not None:
        raise ValueError("development data is only supported for factoid gradient methods")
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError("strength must be finite and in (0,1]")
    if (
        type(steps) is not int
        or not 1 <= steps <= base_step_limit(method)
        or not isinstance(rewrite, str)
        or not 1 <= len(rewrite) <= 256
    ):
        raise ValueError("invalid bounded rewrite settings")
    if (
        fingerprint.get("checkpoint_fingerprint") != runtime.fingerprint
        or fingerprint.get("target_id") != localization.get("target_id")
        or fingerprint.get("spec_sha256") != localization.get("spec_sha256")
        or localization.get("fingerprint_sha256") != record_digest(fingerprint)
    ):
        raise ValueError("fingerprint/localization/spec identity mismatch")
    names = touched_tensors(runtime, method, localization)
    storage_dtypes = getattr(runtime, "storage_dtypes", {})
    if any(storage_dtypes.get(name) not in {"F32", "F16", "BF16"} for name in names):
        raise ValueError("missing or unsupported checkpoint storage dtype")
    selected = _selected(runtime, localization)
    state = runtime.model.state_dict()
    originals = {name: state[name].detach().clone() for name in names}
    replacements = {name: value.clone() for name, value in originals.items()}
    diagnostics: dict[str, Any] = {
        "method": method,
        "strength": strength,
        "max_tensor_relative_frobenius": max_relative_change,
        "details": [],
        **METHOD_INFO[method],
    }
    torch = import_module("torch")
    if method in FACTOID_METHODS:
        from lacunai.erase.factoid_gradient import propose

        if development_data is None:
            raise ValueError("gradient development data missing")
        proposal = propose(
            runtime,
            names[0],
            development_data["forget"],
            development_data["retain_development"],
            mode=method,
            steps=steps,
        )
        replacements = proposal.pop("replacements")
        diagnostics["gradient"] = proposal
        diagnostics["development_data_sha256"] = record_digest(development_data)
    elif method == "historical-columns":
        replacements = _historical(runtime, selected, strength, max_relative_change)
    else:
        for row in selected:
            runtime.check()
            layer, neurons = row["layer"], row["neurons"]
            prefix = _module_name(runtime, layer)
            record = fingerprint["layers"][str(layer)]
            if method == "ablate-neurons":
                replacements[f"{prefix}.up_proj.weight"][neurons, :] *= 1 - strength
                bias_name = f"{prefix}.up_proj.bias"
                if bias_name in replacements:
                    replacements[bias_name][neurons] *= 1 - strength
            elif method == "ablate-direction":
                name = f"{prefix}.down_proj.weight"
                weight = originals[name]
                direction = torch.tensor(record["direction"], dtype=weight.dtype)
                if direction.shape != (weight.shape[1],) or not bool(
                    torch.isfinite(direction).all()
                ):
                    raise ValueError("feature direction has invalid shape/values")
                norm = float(torch.linalg.vector_norm(direction))
                if norm <= 1e-8:
                    raise ValueError("difference-of-means direction is degenerate")
                direction /= norm
                replacements[name] = weight - strength * torch.outer(weight @ direction, direction)
                diagnostics["details"].append(
                    {
                        "layer": layer,
                        "vector_space": "down_proj input features",
                        "direction_norm_before_normalization": norm,
                    }
                )
            elif method == "rank-one":
                prompts = fingerprint.get("development_prompts")
                if (
                    not isinstance(prompts, list)
                    or not 1 <= len(prompts) <= 128
                    or any(not isinstance(prompt, str) or not prompt for prompt in prompts)
                ):
                    raise ValueError("rank-one requires bounded recorded development prompts")
                residual, details = _rank_one(runtime, layer, record, prompts, rewrite, steps)
                name = f"{prefix}.down_proj.weight"
                replacements[name] = originals[name] + strength * residual
                diagnostics["details"].append({"layer": layer, **details})
    replacements = round_to_storage(replacements, storage_dtypes)
    diagnostics["storage_dtypes"] = {name: storage_dtypes[name] for name in names}
    diagnostics["rounding"] = (
        "proposal rounded to source storage precision before bounds and hashes"
    )
    diagnostics["tensor_diffs"] = validate_replacements(
        originals, replacements, max_relative_change
    )
    diagnostics["limits"] = (
        "Weight intervention and changed activation do not prove information absence."
    )
    return {"replacements": replacements, "diagnostics": diagnostics}
