"""Four-language LAPE adaptation; mapping and selection never execute an edit."""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any

from lacunai.erase import language
from lacunai.experimental import tensors
from lacunai.map.localize import record_digest

LANGUAGES = ("en", "zh", "es", "ja")
QUANTILE = 0.95
PROVENANCE = {
    "method": "four-language LAPE adaptation; not seven-language reproduction",
    "activation_source": "https://github.com/RUCAIBox/Language-Specific-Neurons/blob/main/activation.py",
    "selection_source": "https://github.com/RUCAIBox/Language-Specific-Neurons/blob/main/identify.py",
    "activation": "SiLU(gate_proj output), before multiplication by up_proj",
    "aggregation": "positive gate counts / all prefill tokens; language probabilities in float64",
    "threshold": (
        "kthvalue(round(number_of_language_neuron_probabilities * 0.95)); strict greater-than"
    ),
    "ties": "stable canonical layer/neuron ordering; official torch.topk tie order unspecified",
    "safety_deviation": (
        "refuse insufficient eligible neurons; never select infinite masked entropy"
    ),
    "corpus_deviation": "four synthetic development languages; raw text, no chat template",
}


def validate_corpora(corpora: dict[str, list[str]]) -> None:
    if set(corpora) != set(LANGUAGES):
        raise ValueError("LAPE adaptation requires separate en/zh/es/ja development corpora")
    for prompts in corpora.values():
        if not isinstance(prompts, list) or not 1 <= len(prompts) <= 256:
            raise ValueError("language corpus must contain bounded nonempty prompts")
        if any(not isinstance(p, str) or not p.strip() or len(p) > 8192 for p in prompts):
            raise ValueError("invalid language diagnostic prompt")


def analyze_counts(
    counts: dict[str, dict[str, list[int]]], token_totals: dict[str, int]
) -> dict[str, Any]:
    """Pure numerical aggregation, independent of model loading and candidate budgets."""
    torch = import_module("torch")
    if set(counts) != set(LANGUAGES) or set(token_totals) != set(LANGUAGES):
        raise ValueError("activation counts must cover the declared four languages")
    layers = sorted(counts[LANGUAGES[0]], key=int)
    if not layers or any(set(counts[key]) != set(layers) for key in LANGUAGES):
        raise ValueError("activation count layer inventories differ")
    widths = {layer: len(counts[LANGUAGES[0]][layer]) for layer in layers}
    if any(width < 1 for width in widths.values()):
        raise ValueError("empty feature count vector")
    flat_by_language = []
    for key in LANGUAGES:
        total = token_totals[key]
        if type(total) is not int or total < 1:
            raise ValueError("activation token totals must be positive integers")
        flat = []
        for layer in layers:
            values = counts[key][layer]
            if len(values) != widths[layer] or any(
                type(n) is not int or not 0 <= n <= total for n in values
            ):
                raise ValueError("activation counts differ from widths/token totals")
            flat.extend(values)
        flat_by_language.append(torch.tensor(flat, dtype=torch.float64) / total)
    probabilities = torch.stack(flat_by_language, dim=-1)
    denominator = probabilities.sum(-1, keepdim=True)
    normalized = torch.where(
        denominator > 0, probabilities / denominator, torch.zeros_like(probabilities)
    )
    logarithms = torch.where(normalized > 0, normalized.log(), torch.zeros_like(normalized))
    entropy = -(normalized * logarithms).sum(-1)
    if not bool(torch.isfinite(entropy).all()):
        raise ValueError("nonfinite activation probability entropy")
    flat = probabilities.flatten()
    threshold_index = round(flat.numel() * QUANTILE)
    if not 1 <= threshold_index <= flat.numel():
        raise ValueError("undefined percentile order statistic")
    threshold = float(flat.kthvalue(threshold_index).values)
    eligibility = (probabilities > threshold).any(-1)
    masked_entropy = entropy.clone()
    masked_entropy[~eligibility] = torch.inf
    order = torch.argsort(masked_entropy, stable=True)
    identities = [(int(layer), neuron) for layer in layers for neuron in range(widths[layer])]
    rows = []
    for rank, flat_index in enumerate(order.tolist()):
        layer, neuron = identities[flat_index]
        rates = {
            key: float(probabilities[flat_index, index]) for index, key in enumerate(LANGUAGES)
        }
        rows.append(
            {
                "rank": rank,
                "layer": layer,
                "neuron": neuron,
                "entropy": float(entropy[flat_index]),
                "eligible": bool(eligibility[flat_index]),
                "activation_probability": rates,
                "assigned_languages": [key for key in LANGUAGES if rates[key] > threshold],
            }
        )
    return {
        "languages": list(LANGUAGES),
        "layer_widths": widths,
        "counts": counts,
        "token_totals": token_totals,
        "threshold": threshold,
        "threshold_order_statistic": threshold_index,
        "quantile": QUANTILE,
        "neuron_count": len(rows),
        "eligible_neuron_count": int(eligibility.sum()),
        "ranking": rows,
        "provenance": PROVENANCE,
    }


def _capture_counts(
    runtime: Any, prompt: str, layers: list[int]
) -> tuple[dict[str, list[int]], int]:
    torch = import_module("torch")
    captured: dict[str, list[int]] = {}
    handles = []
    runtime.check()
    ids = runtime.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
    if (
        ids.ndim != 2
        or ids.shape[0] != 1
        or not 1 <= ids.shape[-1] <= runtime.config.max_prompt_tokens
    ):
        raise ValueError("raw LAPE text exceeds token bounds or is not a single sequence")
    token_count = int(ids.shape[-1])
    try:
        for layer in layers:

            def hook(module: Any, inputs: Any, output: Any, layer: int = layer) -> None:
                del module, inputs
                if (
                    output.ndim != 3
                    or output.shape[0] != 1
                    or output.shape[1] != token_count
                    or str(layer) in captured
                ):
                    raise ValueError("LAPE requires one matched batch-one gate prefill per layer")
                activation = torch.nn.functional.silu(output.float())
                if not bool(torch.isfinite(activation).all()):
                    raise ValueError("nonfinite gate activation")
                counts = (activation > 0).sum(dim=(0, 1), dtype=torch.int64)
                captured[str(layer)] = counts.tolist()
                runtime.progress(
                    "lape_gate_layers", completed=len(captured), total=len(layers), unit="layers"
                )

            handles.append(runtime.mlp(layer).gate_proj.register_forward_hook(hook))
        runtime.progress("lape_gate_layers", completed=0, total=len(layers), unit="layers")
        runtime.check()
        with torch.inference_mode():
            runtime.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        if set(captured) != {str(layer) for layer in layers}:
            raise ValueError("LAPE gate capture omitted a requested layer")
        return captured, token_count
    finally:
        for handle in handles:
            handle.remove()


def diagnose(runtime: Any, corpora: dict[str, list[str]]) -> dict[str, Any]:
    validate_corpora(corpora)
    layers = language._layers(runtime)
    language._names(runtime)
    source = language._inventory(runtime)
    counts: dict[str, dict[str, list[int]]] = {}
    totals = {}
    for key in LANGUAGES:
        summed: dict[str, list[int]] = {}
        total = 0
        runtime.progress(
            "lape_diagnosis", language=key, completed=0, total=len(corpora[key]), unit="prompts"
        )
        for completed, prompt in enumerate(corpora[key], 1):
            values, tokens = _capture_counts(runtime, prompt, layers)
            total += tokens
            for layer, row in values.items():
                if layer not in summed:
                    summed[layer] = row
                else:
                    if len(summed[layer]) != len(row):
                        raise ValueError("LAPE gate width changed across prompts")
                    summed[layer] = [
                        left + right for left, right in zip(summed[layer], row, strict=True)
                    ]
            runtime.progress(
                "lape_diagnosis",
                language=key,
                completed=completed,
                total=len(corpora[key]),
                unit="prompts",
            )
        counts[key], totals[key] = summed, total
    result = {
        "schema_version": 1,
        "checkpoint_fingerprint": runtime.fingerprint,
        "source": source,
        "corpus_sha256": record_digest(corpora),
        "prompt_counts": {key: len(corpora[key]) for key in LANGUAGES},
        "tokenization": {
            "chat_template": False,
            "add_special_tokens": True,
            "max_prompt_tokens": runtime.config.max_prompt_tokens,
            "truncation": False,
        },
        **analyze_counts(counts, totals),
        "limits": "Language association, not absence or functional necessity. "
        "Japanese and other collateral must be measured independently.",
    }
    language._identity(runtime, result)
    return result


def select(
    diagnosis: dict[str, Any],
    global_fraction: float,
    *,
    target: str = "zh",
    exclude_languages: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    if (
        type(global_fraction) not in (int, float)
        or not math.isfinite(global_fraction)
        or not 0 < global_fraction <= 1
    ):
        raise ValueError("global selection fraction must be finite and in (0,1]")
    if target not in LANGUAGES or diagnosis["languages"] != list(LANGUAGES):
        raise ValueError("selection language differs from four-language contract")
    if (
        not isinstance(exclude_languages, (tuple, list))
        or any(
            not isinstance(key, str) or key not in LANGUAGES or key == target
            for key in exclude_languages
        )
        or len(exclude_languages) != len(set(exclude_languages))
    ):
        raise ValueError("excluded languages must be a distinct subset excluding the target")
    excluded = [key for key in LANGUAGES if key in exclude_languages]
    ranking = diagnosis["ranking"]
    if len(ranking) != diagnosis["neuron_count"] or any(
        row["rank"] != rank for rank, row in enumerate(ranking)
    ):
        raise ValueError("global neuron ranking is incomplete or reordered")
    count = round(len(ranking) * global_fraction)
    if count < 1:
        raise ValueError("global selection rounds to no neurons")
    chosen = ranking[:count]
    if any(not row["eligible"] for row in chosen):
        raise ValueError("insufficient eligible neurons; refusing official infinity-fill behavior")
    target_rows = [
        row for row in chosen if row["activation_probability"][target] > diagnosis["threshold"]
    ]
    for row in target_rows:
        for key in excluded:
            probability = row["activation_probability"].get(key)
            if (
                type(probability) not in (int, float)
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("invalid excluded-language activation probability")
    filtered_rows = [
        row
        for row in target_rows
        if not any(row["activation_probability"][key] > diagnosis["threshold"] for key in excluded)
    ]
    selected = []
    for layer in sorted(diagnosis["layer_widths"], key=int):
        width = diagnosis["layer_widths"][layer]
        neurons = [row["neuron"] for row in filtered_rows if row["layer"] == int(layer)]
        selected.append({"layer": int(layer), "width": width, "rows": sorted(neurons)})
    return {
        "diagnosis_sha256": record_digest(diagnosis),
        "global_fraction": global_fraction,
        "global_selected_count": count,
        "target": target,
        "exclude_languages": excluded,
        "target_unfiltered_count": len(target_rows),
        "target_filtered_count": len(filtered_rows),
        "excluded_target_count": len(target_rows) - len(filtered_rows),
        "assignment_filter": (
            "Additional experimental adaptation: exclude target rows assigned "
            "to any declared excluded language by probability > frozen threshold; "
            "saved assigned_languages labels are not used."
        ),
        "selected": selected,
        "target_selected_count": sum(len(row["rows"]) for row in selected),
        "selection_scope": "global lowest eligible entropy before target-language assignment",
    }


def to_proposal(
    runtime: Any, diagnosis: dict[str, Any], selection: dict[str, Any], *, execution_fraction: float
) -> dict[str, Any]:
    """Explicit adapter retaining existing per-layer execution limits without trimming."""
    language._identity(runtime, diagnosis)
    if selection != select(
        diagnosis,
        selection["global_fraction"],
        target=selection["target"],
        exclude_languages=selection.get("exclude_languages", ()),
    ):
        raise ValueError("LAPE selection differs from diagnosis or declared selection fraction")
    if execution_fraction not in language.FRACTIONS:
        raise ValueError("execution fraction is outside existing language edit allowlist")
    names = language._names(runtime)
    parameters = dict(runtime.model.named_parameters())
    if [row["layer"] for row in selection["selected"]] != list(names):
        raise ValueError("LAPE selection layers differ from runtime")
    diffs = {}
    for row in selection["selected"]:
        width = parameters[names[row["layer"]][0]].shape[0]
        if row["width"] != width or len(row["rows"]) > math.ceil(width * execution_fraction):
            raise ValueError("global LAPE selection exceeds explicit per-layer execution cap")
        if any(type(index) is not int or not 0 <= index < width for index in row["rows"]):
            raise ValueError("LAPE selected neuron is outside feature dimension")
        for name in names[row["layer"]] if row["rows"] else []:
            delta = tensors._square_norm(parameters[name][row["rows"]])
            base = diagnosis["source"]["parameters"][name]["norm_squared"]
            relative = math.sqrt(delta / base) if base else 0.0
            if relative > language.MAX_TENSOR_RELATIVE + 1e-9:
                raise ValueError("LAPE selection exceeds existing tensor change bound")
            diffs[name] = {"delta_norm_squared": delta, "relative_frobenius": relative}
    total = math.fsum(row["delta_norm_squared"] for row in diffs.values())
    relative = math.sqrt(total / diagnosis["source"]["norm_squared"])
    if total <= 0 or relative > language.MAX_MODEL_RELATIVE + 1e-9:
        raise ValueError("LAPE selection is no-op or exceeds existing model change bound")
    proposal = {
        "schema_version": 1,
        "checkpoint_fingerprint": runtime.fingerprint,
        "source": diagnosis["source"],
        "diagnosis_sha256": record_digest(diagnosis),
        "fraction": execution_fraction,
        "strength": 1.0,
        "selected": selection["selected"],
        "tensor_diffs": diffs,
        "model_relative_frobenius": relative,
        "max_tensor_relative": language.MAX_TENSOR_RELATIVE,
        "max_model_relative": language.MAX_MODEL_RELATIVE,
        "operation": "zero_language_up_projection_rows",
        "selection_method": "four-language LAPE adaptation",
        "lape_selection": selection,
        "storage_roundtrip": "zero exactly representable in source dtype",
        "execution_fraction_role": "per-layer safety ceiling; not global LAPE selection fraction",
    }
    language.touched_tensors(runtime, proposal)
    return proposal
