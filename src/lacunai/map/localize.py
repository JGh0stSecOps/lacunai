"""Bounded activation patching; scores describe this intervention, not memory location."""

from __future__ import annotations

import hashlib
import json
import math
from importlib import import_module
from typing import Any

from .fingerprints import assert_identity, layers_for

CONTRAST_TOLERANCE = 1e-6


def record_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def effect_record(clean: float, corrupt: float, patched: float) -> dict[str, Any]:
    if any(not math.isfinite(value) for value in (clean, corrupt, patched)):
        raise ValueError("non-finite causal score")
    contrast = clean - corrupt
    identified = contrast > CONTRAST_TOLERANCE
    return {
        "clean_logprob": clean,
        "corrupt_logprob": corrupt,
        "patched_logprob": patched,
        "clean_corrupt_contrast": contrast,
        "effect": patched - corrupt,
        "normalized_effect": (patched - corrupt) / contrast if identified else None,
        "normalization_status": "identified" if identified else "clean_not_better_or_near_zero",
    }


def _clean(runtime: Any, item: Any, layers: list[int]) -> tuple[float, dict[int, Any]]:
    torch = import_module("torch")
    captured: dict[int, Any] = {}
    handles = []
    try:
        for layer in layers:

            def hook(module: Any, inputs: Any, layer: int = layer) -> None:
                del module
                value = inputs[0]
                if value.ndim != 3 or value.shape[0] != 1 or layer in captured:
                    raise ValueError("causal trace requires one batch-one prefill")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("non-finite causal activation")
                captured[layer] = value[0, -1].detach().clone()

            handles.append(runtime.mlp(layer).down_proj.register_forward_pre_hook(hook))
        score = runtime.next_logprob(item.prompt, item.answer)
        if set(captured) != set(layers):
            raise ValueError("causal capture did not cover every requested layer")
        return score, captured
    finally:
        for handle in handles:
            handle.remove()


def _patched(runtime: Any, item: Any, layer: int, clean: Any, neurons: list[int] | None) -> float:
    calls = 0

    def hook(module: Any, inputs: Any) -> tuple[Any, ...]:
        nonlocal calls
        del module
        value = inputs[0]
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != clean.shape[0]:
            raise ValueError("causal patch activation shape differs")
        calls += 1
        if calls != 1:
            raise ValueError("causal patch is restricted to one prefill")
        modified = value.clone()
        if neurons is None:
            modified[0, -1] = clean
        else:
            modified[0, -1, neurons] = clean[neurons]
        return (modified, *inputs[1:])

    handle = runtime.mlp(layer).down_proj.register_forward_pre_hook(hook)
    try:
        runtime.check()
        score = runtime.next_logprob(item.corrupt_prompt, item.answer)
        if calls != 1:
            raise ValueError("causal patch hook did not execute")
        return float(score)
    finally:
        handle.remove()


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["normalized_effect"] is not None]
    return {
        "mean_effect": math.fsum(row["effect"] for row in valid) / len(valid) if valid else None,
        "confidence": {
            "kind": "uncalibrated_intervention_score",
            "score": math.fsum(row["normalized_effect"] for row in valid) / len(valid)
            if valid
            else None,
            "identified_pairs": len(valid),
            "total_pairs": len(records),
        },
        "raw_pairs": records,
    }


def locate(runtime: Any, spec: Any, fingerprint: dict[str, Any]) -> dict[str, Any]:
    assert_identity(runtime, spec, fingerprint)
    layers = layers_for(runtime, spec)
    if type(spec.patch_neurons) is not int or not 1 <= spec.patch_neurons <= 128:
        raise ValueError("patch_neurons must be between 1 and 128")
    if set(fingerprint["layers"]) != {str(layer) for layer in layers}:
        raise ValueError("fingerprint layer set differs from localization contract")
    runtime.progress("causal_pairs", completed=0, total=len(spec.mapping), unit="pairs")
    pairs, skipped = [], []
    for index, item in enumerate(spec.mapping):
        runtime.check()
        clean_length = runtime.encode(item.prompt).shape[-1]
        corrupt_length = runtime.encode(item.corrupt_prompt).shape[-1]
        if clean_length != corrupt_length:
            skipped.append(
                {
                    "pair": index,
                    "reason": "token_length_mismatch",
                    "clean_tokens": clean_length,
                    "corrupt_tokens": corrupt_length,
                }
            )
            continue
        clean_score, activations = _clean(runtime, item, layers)
        corrupt_score = runtime.next_logprob(item.corrupt_prompt, item.answer)
        pairs.append((index, item, clean_score, corrupt_score, activations))
        runtime.progress("causal_pairs", completed=index + 1, total=len(spec.mapping), unit="pairs")
    ranked = []
    for completed, layer in enumerate(layers, 1):
        records = []
        for index, item, clean, corrupt, activations in pairs:
            patched = _patched(runtime, item, layer, activations[layer], None)
            records.append({"pair": index, **effect_record(clean, corrupt, patched)})
        ranked.append({"layer": layer, **_summary(records), "neurons": []})
        runtime.progress("causal_layers", completed=completed, total=len(layers), unit="layers")
    ranked.sort(
        key=lambda row: (
            -(row["mean_effect"] if row["mean_effect"] is not None else -math.inf),
            row["layer"],
        )
    )
    viable = [row for row in ranked if row["mean_effect"] is not None and row["mean_effect"] > 0]
    selected = []
    # Layer intervention first; shortlist channels only in the best positively mediated layer.
    if viable:
        best = viable[0]
        layer = best["layer"]
        candidates = list(
            dict.fromkeys(row["neuron"] for row in fingerprint["layers"][str(layer)]["features"])
        )[: spec.patch_neurons]
        for completed, neuron in enumerate(candidates, 1):
            records = []
            for index, item, clean, corrupt, activations in pairs:
                patched = _patched(runtime, item, layer, activations[layer], [neuron])
                records.append({"pair": index, **effect_record(clean, corrupt, patched)})
            best["neurons"].append({"neuron": neuron, **_summary(records)})
            runtime.progress(
                "causal_neurons", completed=completed, total=len(candidates), unit="neurons"
            )
        best["neurons"].sort(
            key=lambda row: (
                -(row["mean_effect"] if row["mean_effect"] is not None else -math.inf),
                row["neuron"],
            )
        )
        selected.append(
            {
                "layer": layer,
                "neurons": [
                    row["neuron"]
                    for row in best["neurons"]
                    if row["mean_effect"] is not None and row["mean_effect"] > 0
                ],
            }
        )
    return {
        "schema_version": 1,
        "kind": "localization",
        "target_id": spec.target_id,
        "spec_sha256": spec.digest,
        "checkpoint_fingerprint": runtime.fingerprint,
        "fingerprint_sha256": record_digest(fingerprint),
        "metric": "first answer-token log probability at final prefill position",
        "patch": (
            "clean post-gate feature copied into matched-length corrupt final prefill position"
        ),
        "contrast_tolerance": CONTRAST_TOLERANCE,
        "matched_pairs": len(pairs),
        "skipped_pairs": skipped,
        "layers": ranked,
        "selected": selected,
        "status": "localized" if selected else "no_identifiable_positive_effect",
        "limits": (
            "Intervention-dependent mediation; uncalibrated scores are not "
            "memory-location probabilities. Only the best positive layer receives "
            "bounded channel tests; rankings need not predict edit efficacy."
        ),
    }
