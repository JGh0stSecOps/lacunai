"""Sparse prefill fingerprints; association is not proof of knowledge storage."""

from __future__ import annotations

import math
from collections import Counter
from importlib import import_module
from typing import Any

ACTIVATION_TOLERANCE = 1e-8


def layers_for(runtime: Any, spec: Any) -> list[int]:
    layers = list(spec.layers or runtime.layer_ids)
    if (
        not layers
        or len(layers) != len(set(layers))
        or any(type(layer) is not int or layer not in runtime.layer_ids for layer in layers)
    ):
        raise ValueError("map layers must be distinct supported layer IDs")
    if type(spec.top_n) is not int or not 1 <= spec.top_n <= 128:
        raise ValueError("top_n must be between 1 and 128")
    return layers


def assert_identity(runtime: Any, spec: Any, record: dict[str, Any]) -> None:
    if (
        record.get("checkpoint_fingerprint") != runtime.fingerprint
        or record.get("spec_sha256") != spec.digest
        or record.get("target_id") != spec.target_id
    ):
        raise ValueError("map checkpoint/target/spec identity mismatch")


def extremes(
    values: list[float], count: int, tolerance: float = ACTIVATION_TOLERANCE
) -> dict[str, Any]:
    """Signed extrema, excluding dead features even when top-k needs more entries."""
    if type(count) is not int or count < 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid fingerprint selection bounds")
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("empty or non-finite activations")
    active = [index for index, value in enumerate(values) if abs(value) > tolerance]
    return {
        kind: [{"neuron": index, "activation": values[index]} for index in indices]
        for kind, indices in (
            ("top", sorted(active, key=lambda i: (-values[i], i))[:count]),
            ("bottom", sorted(active, key=lambda i: (values[i], i))[:count]),
        )
    }


def weighted_features(
    target: list[dict[str, Any]], background: list[dict[str, Any]], width: int
) -> list[dict[str, Any]]:
    if not target or not background or width < 1:
        raise ValueError("target and background fingerprint documents are required")
    target_df: Counter[tuple[int, str]] = Counter()
    background_df: Counter[tuple[int, str]] = Counter()
    for documents, counts in ((target, target_df), (background, background_df)):
        for document in documents:
            features = {
                (row["neuron"], polarity)
                for polarity in ("top", "bottom")
                for row in document[polarity]
            }
            if any(type(index) is not int or not 0 <= index < width for index, _ in features):
                raise ValueError("fingerprint neuron out of bounds")
            counts.update(features)
    records = []
    for (index, polarity), support in sorted(target_df.items()):
        df = background_df[(index, polarity)]
        idf = math.log((len(background) + 1) / (df + 1)) + 1
        records.append(
            {
                "neuron": index,
                "polarity": polarity,
                "target_documents": support,
                "background_documents": df,
                "idf": idf,
                "weighted_support": support / len(target) * idf,
            }
        )
    return sorted(
        records, key=lambda row: (-row["weighted_support"], row["neuron"], row["polarity"])
    )


def capture(runtime: Any, prompt: str, layers: list[int]) -> dict[int, Any]:
    """Capture only one prefill pass; every hook is removed on success or failure."""
    torch = import_module("torch")
    captured: dict[int, Any] = {}
    handles = []
    try:
        for layer in layers:

            def hook(module: Any, inputs: Any, layer: int = layer) -> None:
                del module
                value = inputs[0]
                if value.ndim != 3 or value.shape[0] != 1 or layer in captured:
                    raise ValueError("expected one batch-one prefill activation per layer")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("non-finite prefill activations")
                captured[layer] = value.detach().to(device="cpu", dtype=torch.float32).clone()

            handles.append(runtime.mlp(layer).down_proj.register_forward_pre_hook(hook))
        runtime.check()
        ids = runtime.encode(prompt)
        with torch.inference_mode():
            runtime.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        if set(captured) != set(layers):
            raise ValueError("requested activation hook did not execute")
        return captured
    finally:
        for handle in handles:
            handle.remove()


def fingerprint(runtime: Any, spec: Any) -> dict[str, Any]:
    torch = import_module("torch")
    layers = layers_for(runtime, spec)
    if not spec.mapping or not spec.background:
        raise ValueError("mapping and background prompts are required")
    vectors: dict[str, dict[int, list[Any]]] = {
        split: {layer: [] for layer in layers} for split in ("target", "background")
    }
    last_vectors: dict[str, dict[int, list[Any]]] = {
        split: {layer: [] for layer in layers} for split in vectors
    }
    sparse: dict[str, dict[int, list[dict[str, Any]]]] = {
        split: {layer: [] for layer in layers} for split in vectors
    }
    prompts = [("target", item.prompt) for item in spec.mapping]
    prompts += [("background", prompt) for prompt in spec.background]
    runtime.progress("fingerprint", completed=0, total=len(prompts), unit="prompts")
    for completed, (split, prompt) in enumerate(prompts, 1):
        captured = capture(runtime, prompt, layers)
        for layer, activation in captured.items():
            pooled = activation[0].mean(dim=0)
            vectors[split][layer].append(pooled)
            last_vectors[split][layer].append(activation[0, -1].clone())
            sparse[split][layer].append(extremes(pooled.tolist(), spec.top_n))
        runtime.progress("fingerprint", completed=completed, total=len(prompts), unit="prompts")
    records = {}
    for layer in layers:
        target = torch.stack(vectors["target"][layer])
        background = torch.stack(vectors["background"][layer])
        target_mean, background_mean = target.mean(0), background.mean(0)
        # Population covariance of prompt-pooled feature vectors, not residual-space covariance.
        covariance = ((background - background_mean) ** 2).mean(0)
        direction = target_mean - background_mean
        last_background = torch.stack(last_vectors["background"][layer])
        key_mean = torch.stack(last_vectors["target"][layer]).mean(0)
        key_covariance = ((last_background - last_background.mean(0)) ** 2).mean(0)
        width = target.shape[1]
        features = weighted_features(sparse["target"][layer], sparse["background"][layer], width)
        for row in features:
            row["mean_absolute_activation"] = float(target[:, row["neuron"]].abs().mean())
        records[str(layer)] = {
            "layer": layer,
            "width": width,
            "target_mean": target_mean.tolist(),
            "key_mean": key_mean.tolist(),
            "key_covariance_diag": key_covariance.tolist(),
            "key_pooling": "final prefill token",
            "background_mean": background_mean.tolist(),
            "direction": direction.tolist(),
            "background_covariance_diag": covariance.tolist(),
            "features": features,
            "target_fingerprints": sparse["target"][layer],
            "background_fingerprints": sparse["background"][layer],
        }
    return {
        "schema_version": 1,
        "kind": "fingerprint",
        "target_id": spec.target_id,
        "spec_sha256": spec.digest,
        "checkpoint_fingerprint": runtime.fingerprint,
        "hook_point": "mlp.down_proj input (post-gate feature)",
        "pooling": "signed mean across all chat-template prefill tokens",
        "activation_tolerance": ACTIVATION_TOLERANCE,
        "idf_formula": "log((background_documents+1)/(feature_document_frequency+1))+1",
        "development_prompts": [item.prompt for item in spec.mapping],
        "target_prompt_count": len(spec.mapping),
        "background_prompt_count": len(spec.background),
        "layers": records,
        "limits": (
            "IDF-weighted activation association is not causal localization or absence evidence."
        ),
    }
