"""Full-range histogram contrast adaptation; diagnostic association, never an edit."""

from __future__ import annotations

import hashlib
import itertools
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from lacunai.erase import language
from lacunai.experimental import tensors
from lacunai.map import language_entropy
from lacunai.map.localize import record_digest

VERSION = 1
LANGUAGES = language_entropy.LANGUAGES
PAIRS = tuple(itertools.combinations(LANGUAGES, 2))
BINS = 64
INT32_MAX = 2**31 - 1
PROVENANCE = {
    "method": "four-language full-range histogram contrast adaptation",
    "primary_source": "https://arxiv.org/html/2609.10993v1",
    "paper_reproduction": False,
    "excluded_paper_procedure": "language clustering and cluster-based neuron selection",
    "activation": "SiLU(gate_proj output), before multiplication by up_proj",
    "bins": "64 shared equal-width per-neuron bins from four-language global min/max",
    "edges": "left-closed/right-open; maximum in final bin; constant range in bin zero",
    "overlap": "float64 sum(min(normalized histograms))",
    "score": "1-max(OVL(zh,en),OVL(zh,es),OVL(zh,ja)); not calibrated confidence",
    "negative_mass": "exact activation<0 count, independent of bin boundaries",
    "weighting": "all prefill tokens equally weighted",
    "ties": "descending contrast then ascending numeric layer/neuron",
    "limits": "Corpus- and bin-resolution-dependent association; no causality, knowledge absence, "
    "erasure or collateral-preservation claim.",
}


def _activation(value: Any) -> None:
    torch = import_module("torch")
    if (
        value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 2
        or min(value.shape) < 1
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("activation requires finite nonempty CPU float32 tokens by neurons")


def _add_counts(destination: Any, increment: Any) -> None:
    torch = import_module("torch")
    if (
        destination.dtype != torch.int32
        or increment.dtype != torch.int64
        or destination.device.type != "cpu"
        or increment.device.type != "cpu"
        or destination.shape != increment.shape
        or bool((destination < 0).any())
        or bool((increment < 0).any())
        or bool((increment > INT32_MAX).any())
    ):
        raise ValueError("invalid bounded histogram counts")
    summed = destination.to(torch.int64) + increment
    if bool((summed > INT32_MAX).any()):
        raise ValueError("int32 histogram count overflow")
    destination.copy_(summed.to(torch.int32))


def update_histogram(
    activation: Any, minimum: Any, maximum: Any, histogram: Any, negative_counts: Any
) -> None:
    torch = import_module("torch")
    _activation(activation)
    width = activation.shape[1]
    if (
        minimum.shape != (width,)
        or maximum.shape != (width,)
        or minimum.dtype != torch.float64
        or maximum.dtype != torch.float64
        or minimum.device.type != "cpu"
        or maximum.device.type != "cpu"
        or not bool(torch.isfinite(minimum).all() & torch.isfinite(maximum).all())
        or bool((maximum < minimum).any())
        or histogram.shape != (width, BINS)
        or negative_counts.shape != (width,)
    ):
        raise ValueError("invalid shared histogram shape/range")
    values = activation.to(torch.float64)
    if bool(((values < minimum) | (values > maximum)).any()):
        raise ValueError("second-pass activation outside frozen first-pass range")
    spans = maximum - minimum
    if not bool(torch.isfinite(spans).all()):
        raise ValueError("nonfinite bin spans")
    scale = torch.where(spans > 0, spans, torch.ones_like(spans))
    indices = ((values - minimum) / scale * BINS).floor().to(torch.int64).clamp(max=BINS - 1)
    offsets = torch.arange(width, dtype=torch.int64) * BINS
    increment = torch.bincount((indices + offsets).flatten(), minlength=width * BINS).reshape(
        width, BINS
    )
    negative = (activation < 0).sum(dim=0, dtype=torch.int64)
    # Check both destinations before either is mutated.
    checked_hist, checked_negative = histogram.clone(), negative_counts.clone()
    _add_counts(checked_hist, increment)
    _add_counts(checked_negative, negative)
    histogram.copy_(checked_hist)
    negative_counts.copy_(checked_negative)


def analyze(
    histograms: dict[int, dict[str, Any]],
    negatives: dict[int, dict[str, Any]],
    token_totals: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = import_module("torch")
    if set(token_totals) != set(LANGUAGES) or any(
        type(n) is not int or not 1 <= n <= INT32_MAX for n in token_totals.values()
    ):
        raise ValueError("invalid language token totals")
    if (
        not histograms
        or set(histograms) != set(negatives)
        or any(type(layer) is not int or layer < 0 for layer in histograms)
    ):
        raise ValueError("invalid numeric layer inventory")
    payload: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    identities: list[tuple[int, int]] = []
    scores: list[Any] = []
    for layer in sorted(histograms):
        rows, signs = histograms[layer], negatives[layer]
        if set(rows) != set(LANGUAGES) or set(signs) != set(LANGUAGES):
            raise ValueError("histogram language inventory differs")
        width = rows[LANGUAGES[0]].shape[0]
        probabilities, masses = {}, []
        for key in LANGUAGES:
            counts, negative = rows[key], signs[key]
            if (
                counts.dtype != torch.int32
                or counts.device.type != "cpu"
                or counts.shape != (width, BINS)
                or width < 1
                or bool((counts < 0).any())
                or not bool((counts.sum(dim=1, dtype=torch.int64) == token_totals[key]).all())
                or negative.dtype != torch.int32
                or negative.device.type != "cpu"
                or negative.shape != (width,)
                or bool(((negative < 0) | (negative > token_totals[key])).any())
            ):
                raise ValueError("histogram/negative counts violate token conservation")
            probabilities[key] = counts.to(torch.float64) / token_totals[key]
            masses.append(negative.to(torch.float64) / token_totals[key])
        overlaps = torch.stack(
            [torch.minimum(probabilities[a], probabilities[b]).sum(dim=1) for a, b in PAIRS], dim=1
        )
        contrast = (
            1 - overlaps[:, [i for i, pair in enumerate(PAIRS) if "zh" in pair]].max(dim=1).values
        )
        if not bool(torch.isfinite(overlaps).all()) or bool(
            ((overlaps < -1e-12) | (overlaps > 1 + 1e-12)).any()
        ):
            raise ValueError("invalid overlap probabilities")
        payload[f"layer.{layer}.overlap"] = overlaps
        payload[f"layer.{layer}.negative_mass"] = torch.stack(masses, dim=1)
        payload[f"layer.{layer}.contrast"] = contrast
        scores.append(contrast)
        identities.extend((layer, neuron) for neuron in range(width))
        summaries[str(layer)] = {
            "neurons": width,
            "contrast_min": float(contrast.min()),
            "contrast_mean": float(contrast.mean()),
            "contrast_max": float(contrast.max()),
        }
    order = torch.argsort(torch.cat(scores), descending=True, stable=True)
    payload["ranking"] = torch.tensor(identities, dtype=torch.int64)[order]
    return {"layer_summaries": summaries, "neuron_count": len(identities)}, payload


def _capture(runtime: Any, prompt: str, layers: list[int], visit: Any) -> int:
    torch = import_module("torch")
    runtime.check()
    ids = runtime.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
    if (
        ids.dtype != torch.int64
        or ids.device.type != "cpu"
        or ids.ndim != 2
        or ids.shape[0] != 1
        or not 1 <= ids.shape[-1] <= runtime.config.max_prompt_tokens
    ):
        raise ValueError("raw diagnostic prompt violates token/dtype bounds")
    tokens, seen, handles = int(ids.shape[-1]), set(), []
    try:
        for layer in layers:

            def hook(module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                if (
                    layer in seen
                    or output.ndim != 3
                    or output.shape != (1, tokens, module.weight.shape[0])
                    or output.dtype != torch.float32
                    or output.device.type != "cpu"
                    or not bool(torch.isfinite(output).all())
                ):
                    raise ValueError("gate capture requires finite single CPU float32 prefill")
                activation = torch.nn.functional.silu(output).squeeze(0)
                _activation(activation)
                visit(layer, activation)
                seen.add(layer)

            handles.append(runtime.mlp(layer).gate_proj.register_forward_hook(hook))
        with torch.inference_mode():
            runtime.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        if seen != set(layers):
            raise ValueError("gate capture omitted a requested layer")
        runtime.check()
        return tokens
    finally:
        for handle in handles:
            handle.remove()


def payload_hash(value: Any) -> str:
    torch = import_module("torch")
    result = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)]).encode())
    flat = value.detach().reshape(-1).contiguous().view(torch.uint8)
    for offset in range(0, flat.numel(), 1024**2):
        result.update(flat[offset : offset + 1024**2].numpy().tobytes())
    return result.hexdigest()


def validate_payload(metadata: dict[str, Any], payload: dict[str, Any]) -> None:
    if (
        type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != VERSION
        or set(payload) != set(metadata["payload"])
    ):
        raise ValueError("distribution payload inventory/version differs")
    for name, value in payload.items():
        expected = metadata["payload"][name]
        if (
            list(value.shape) != expected["shape"]
            or str(value.dtype) != expected["dtype"]
            or payload_hash(value) != expected["sha256"]
        ):
            raise ValueError("distribution payload tensor differs: " + name)


def _diagnose(runtime: Any, corpora: dict[str, list[str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Caller persists returned tensors; no full-corpus activations or edits are retained."""
    language_entropy.validate_corpora(corpora)
    torch = import_module("torch")
    layers = language._layers(runtime)
    if getattr(runtime.model.config, "hidden_act", None) != "silu":
        raise ValueError("distribution mapping requires configured SiLU gates")
    source = language._inventory(runtime)
    minima: dict[int, Any] = {}
    maxima: dict[int, Any] = {}
    histograms: dict[int, dict[str, Any]] = {}
    negatives: dict[int, dict[str, Any]] = {}
    first_totals: dict[str, int] = {}
    totals: dict[str, int] = {}
    for pass_number in (1, 2):
        for key in LANGUAGES:
            total = 0
            runtime.progress(
                "distribution_mapping",
                pass_number=pass_number,
                language=key,
                completed=0,
                total=len(corpora[key]),
                unit="prompts",
            )
            for completed, prompt in enumerate(corpora[key], 1):

                def visit(
                    layer: int, values: Any, pass_number: int = pass_number, key: str = key
                ) -> None:
                    width = values.shape[1]
                    if pass_number == 1:
                        low, high = values.amin(dim=0).double(), values.amax(dim=0).double()
                        if layer not in minima:
                            minima[layer], maxima[layer] = low, high
                        else:
                            if minima[layer].shape != (width,):
                                raise ValueError("gate width changed across prompts")
                            minima[layer] = torch.minimum(minima[layer], low)
                            maxima[layer] = torch.maximum(maxima[layer], high)
                    else:
                        if layer not in histograms:
                            histograms[layer] = {
                                lang: torch.zeros((width, BINS), dtype=torch.int32)
                                for lang in LANGUAGES
                            }
                            negatives[layer] = {
                                lang: torch.zeros(width, dtype=torch.int32) for lang in LANGUAGES
                            }
                        update_histogram(
                            values,
                            minima[layer],
                            maxima[layer],
                            histograms[layer][key],
                            negatives[layer][key],
                        )

                total += _capture(runtime, prompt, layers, visit)
                if total > INT32_MAX:
                    raise ValueError("language token total exceeds int32 capacity")
                runtime.progress(
                    "distribution_mapping",
                    pass_number=pass_number,
                    language=key,
                    completed=completed,
                    total=len(corpora[key]),
                    unit="prompts",
                )
            if pass_number == 1:
                first_totals[key] = total
            else:
                if total != first_totals[key]:
                    raise ValueError("token totals changed between diagnostic passes")
                totals[key] = total
    summary, payload = analyze(histograms, negatives, totals)
    for layer in layers:
        payload[f"layer.{layer}.min"] = minima[layer]
        payload[f"layer.{layer}.max"] = maxima[layer]
        for key in LANGUAGES:
            payload[f"layer.{layer}.hist.{key}"] = histograms[layer][key]
            payload[f"layer.{layer}.negative_counts.{key}"] = negatives[layer][key]
    result = {
        "schema_version": VERSION,
        "kind": "language_distribution",
        "checkpoint_fingerprint": runtime.fingerprint,
        "source": source,
        "corpus_sha256": record_digest(corpora),
        "prompt_counts": {key: len(corpora[key]) for key in LANGUAGES},
        "token_totals": totals,
        "languages": list(LANGUAGES),
        "pair_order": [list(pair) for pair in PAIRS],
        "bins": BINS,
        "tokenization": {
            "chat_template": False,
            "add_special_tokens": True,
            "truncation": False,
            "max_prompt_tokens": runtime.config.max_prompt_tokens,
        },
        "provenance": PROVENANCE,
        "numerics": {
            "activations": "CPU float32",
            "ranges_probabilities": "float64",
            "counts": "int32 with checked int64 accumulation",
            "seed": runtime.config.seed,
            "torch_threads": runtime.config.torch_threads,
            "runtime_dtype": runtime.config.dtype,
            "overlap_tolerance": 1e-12,
            "reproducibility": "runtime determinism applies; second-pass range drift rejected",
        },
        "code_hashes": {
            Path(str(module.__file__)).name: tensors.file_hash(Path(str(module.__file__)))
            for module in (language, language_entropy)
        },
        "implementation_sha256": tensors.file_hash(Path(__file__)),
        "payload": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": payload_hash(value),
            }
            for name, value in payload.items()
        },
        **summary,
    }
    language._identity(runtime, result)
    runtime.progress(
        "distribution_mapping", state="complete", passes=2, neurons=summary["neuron_count"]
    )
    return result, payload


def diagnose(runtime: Any, corpora: dict[str, list[str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime.progress("distribution_mapping", state="active", passes=2)
    try:
        return _diagnose(runtime, corpora)
    except BaseException as exc:
        runtime.progress("distribution_mapping", state="failed", error=str(exc))
        raise


def to_proposal(
    runtime: Any,
    metadata: dict[str, Any],
    payload: dict[str, Any],
    *,
    top_k: int,
    execution_fraction: float,
) -> dict[str, Any]:
    """Build an explicitly requested edit specification; never apply it or trim its rows."""
    import math

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if (
        type(execution_fraction) not in (int, float)
        or not math.isfinite(execution_fraction)
        or execution_fraction not in language.FRACTIONS
    ):
        raise ValueError("execution fraction is outside existing language edit allowlist")
    language._identity(runtime, metadata)
    validate_payload(metadata, payload)
    torch = import_module("torch")
    names = language._names(runtime)
    parameters = dict(runtime.model.named_parameters())
    if (
        metadata.get("kind") != "language_distribution"
        or metadata.get("languages") != list(LANGUAGES)
        or metadata.get("pair_order") != [list(pair) for pair in PAIRS]
        or metadata.get("bins") != BINS
    ):
        raise ValueError("distribution method/language/bin contract differs")
    expected_keys = {"ranking"}
    histograms, negatives = {}, {}
    for layer, layer_names in names.items():
        width = parameters[layer_names[0]].shape[0]
        expected_keys.update(
            f"layer.{layer}.{key}" for key in ("min", "max", "overlap", "negative_mass", "contrast")
        )
        expected_keys.update(
            f"layer.{layer}.{key}.{lang}"
            for key in ("hist", "negative_counts")
            for lang in LANGUAGES
        )
        low, high = payload[f"layer.{layer}.min"], payload[f"layer.{layer}.max"]
        if (
            low.dtype != torch.float64
            or high.dtype != torch.float64
            or low.shape != (width,)
            or high.shape != (width,)
            or low.device.type != "cpu"
            or high.device.type != "cpu"
            or not bool(torch.isfinite(low).all() & torch.isfinite(high).all())
            or bool((high < low).any())
        ):
            raise ValueError("saved distribution ranges differ from runtime feature width")
        histograms[layer] = {lang: payload[f"layer.{layer}.hist.{lang}"] for lang in LANGUAGES}
        negatives[layer] = {
            lang: payload[f"layer.{layer}.negative_counts.{lang}"] for lang in LANGUAGES
        }
        if any(row.shape != (width, BINS) for row in histograms[layer].values()):
            raise ValueError("saved histogram feature width differs from runtime")
    if set(payload) != expected_keys:
        raise ValueError("distribution payload differs from runtime layer inventory")
    summary, rebuilt = analyze(histograms, negatives, metadata["token_totals"])
    if any(metadata.get(key) != value for key, value in summary.items()):
        raise ValueError("saved distribution summaries differ from recomputed counts")
    for key, value in rebuilt.items():
        saved = payload[key]
        if (
            saved.dtype != value.dtype
            or saved.device.type != "cpu"
            or not torch.equal(saved, value)
        ):
            raise ValueError("saved distribution ranking/statistics differ from recomputed counts")
    ranking = rebuilt["ranking"]
    if top_k > ranking.shape[0]:
        raise ValueError("top_k exceeds the complete diagnostic neuron inventory")
    selected_ids = ranking[:top_k].tolist()
    rows_by_layer: dict[int, list[int]] = {layer: [] for layer in names}
    for layer, neuron in selected_ids:
        rows_by_layer[layer].append(neuron)
    selected, diffs = [], {}
    scores: list[float] = []
    for layer, layer_names in names.items():
        width = parameters[layer_names[0]].shape[0]
        rows = rows_by_layer[layer]
        if len(rows) > math.ceil(width * execution_fraction):
            raise ValueError(
                "exact top_k exceeds explicit per-layer execution cap; no rows trimmed"
            )
        selected.append({"layer": layer, "width": width, "rows": rows})
        scores.extend(float(rebuilt[f"layer.{layer}.contrast"][neuron]) for neuron in rows)
        for name in layer_names if rows else []:
            delta = tensors._square_norm(parameters[name][rows])
            base = metadata["source"]["parameters"][name]["norm_squared"]
            relative = math.sqrt(delta / base) if base else 0.0
            if relative > language.MAX_TENSOR_RELATIVE + 1e-9:
                raise ValueError("distribution proposal exceeds existing tensor change bound")
            diffs[name] = {"delta_norm_squared": delta, "relative_frobenius": relative}
    total = math.fsum(row["delta_norm_squared"] for row in diffs.values())
    relative = math.sqrt(total / metadata["source"]["norm_squared"])
    if total <= 0 or relative > language.MAX_MODEL_RELATIVE + 1e-9:
        raise ValueError("distribution proposal is no-op or exceeds existing model change bound")
    if sum(len(row["rows"]) for row in selected) != top_k:
        raise ValueError("exact top_k row selection was not preserved")
    diagnosis_pin = record_digest(metadata)
    proposal = {
        "schema_version": 1,
        "checkpoint_fingerprint": runtime.fingerprint,
        "source": metadata["source"],
        "diagnosis_sha256": diagnosis_pin,
        "fraction": execution_fraction,
        "strength": 1.0,
        "selected": selected,
        "tensor_diffs": diffs,
        "model_relative_frobenius": relative,
        "max_tensor_relative": language.MAX_TENSOR_RELATIVE,
        "max_model_relative": language.MAX_MODEL_RELATIVE,
        "operation": "zero_language_up_projection_rows",
        "selection_method": PROVENANCE["method"],
        "distribution_selection": {
            "method": PROVENANCE["method"],
            "top_k": top_k,
            "ranked_ids": selected_ids,
            "diagnosis_sha256": diagnosis_pin,
            "payload_manifest_sha256": record_digest(metadata["payload"]),
            "contrast_min": min(scores),
            "contrast_max": max(scores),
            "contrast_is_calibrated_confidence": False,
        },
        "storage_roundtrip": "zero exactly representable in source dtype",
        "execution_fraction_role": "per-layer safety ceiling; not a selection fraction",
    }
    language.touched_tensors(runtime, proposal)
    return proposal
