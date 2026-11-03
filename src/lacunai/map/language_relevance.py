"""First-order answer-loss mask sensitivity; diagnostic only, not attribution proof."""

from __future__ import annotations

from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any

from lacunai.erase import gradient_objectives as objectives
from lacunai.erase import language
from lacunai.experimental import tensors
from lacunai.map.language_distribution import payload_hash
from lacunai.map.localize import record_digest

VERSION = 1
FORGET = ("read_zh_answer_en_or_numeric", "translate_zh_en")
RETAIN = ("english_comprehension_neighbors", "arithmetic", "code_trace", "spanish", "japanese_han")
FAMILIES = FORGET + RETAIN
PROVENANCE = {
    "method": "first-order all-ones post-gate MLP mask sensitivity",
    "paper_reproduction": False,
    "objective": "mean causal answer-token NLL including EOS; first prompt only; cap128",
    "necessity": "max(0,-equal mean of two forget-family signed mean gradients)",
    "retain_risk": "maximum of five retain-family mean absolute gradients",
    "score": "max(0,necessity-retain_risk)",
    "ties": "descending score then numeric layer/neuron",
    "limitations": (
        "Local first-order sensitivity, not CRANE/AttnLRP, necessity proof, or knowledge absence. "
        "Canonical-answer loss may miss semantic recovery."
    ),
}


def validate_corpus(corpus: Any) -> list[dict[str, Any]]:
    if not isinstance(corpus, dict) or set(corpus) != {"forget_train", "retain_train"}:
        raise ValueError("requires only forget_train and retain_train")
    expected = {"forget_train": (*FORGET, "write_zh", "translate_en_zh"), "retain_train": RETAIN}
    identifiers = set()
    for split, families in expected.items():
        rows = corpus[split]
        if not isinstance(rows, list) or len(rows) != 16 * len(families):
            raise ValueError("training family counts differ")
        counts: Counter[str] = Counter()
        for row in rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or not row["id"]
                or row["id"] in identifiers
            ):
                raise ValueError("missing/duplicate training item ID")
            identifiers.add(row["id"])
            if not isinstance(row.get("family"), str):
                raise ValueError("missing training family")
            counts[row["family"]] += 1
            if (
                not isinstance(row.get("prompts"), list)
                or not row["prompts"]
                or not isinstance(row["prompts"][0], str)
                or not row["prompts"][0]
                or not isinstance(row.get("answer"), str)
                or not row["answer"]
            ):
                raise ValueError("missing first prompt or answer")
        if counts != Counter({family: 16 for family in families}):
            raise ValueError("training family counts differ")
    return [row for row in corpus["forget_train"] if row["family"] in FORGET] + list(
        corpus["retain_train"]
    )


def mask_gradient(runtime: Any, batch: dict[str, Any]) -> tuple[Any, dict[int, Any]]:
    """One graph; only masks require gradients. Always remove this call's hooks."""
    torch = import_module("torch")
    if any(p.requires_grad for p in runtime.model.parameters()):
        raise ValueError("mask diagnostic requires frozen parameters")
    masks, hooks, calls = {}, [], {}
    layers = sorted(language._layers(runtime))
    try:
        for layer in layers:
            width = runtime.mlp(layer).down_proj.weight.shape[1]
            masks[layer] = torch.ones(width, dtype=torch.float32, requires_grad=True)
            calls[layer] = 0

            def hook(
                module: Any, args: tuple[Any, ...], layer: int = layer, width: int = width
            ) -> tuple[Any, ...]:
                value = args[0]
                if (
                    value.device.type != "cpu"
                    or value.dtype != torch.float32
                    or value.ndim != 3
                    or value.shape[0] != 1
                    or value.shape[-1] != width
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValueError("invalid mask activation shape/dtype/finite values")
                calls[layer] += 1
                return (value * masks[layer], *args[1:])

            hooks.append(runtime.mlp(layer).down_proj.register_forward_pre_hook(hook))
        with torch.enable_grad():
            logits = objectives.answer_forward(runtime.model, batch)
            count = int((batch["labels"][:, 1:] != -100).sum())
            if count <= 0:
                raise ValueError("empty answer")
            loss = objectives.sequence_nll(logits, batch, selected_positions=True) / count
            if any(number != 1 for number in calls.values()):
                raise ValueError("mask hook invocation count differs")
            gradients = torch.autograd.grad(loss, tuple(masks.values()), allow_unused=False)
        result = {}
        for layer, gradient in zip(layers, gradients, strict=True):
            if gradient.shape != masks[layer].shape or not bool(torch.isfinite(gradient).all()):
                raise ValueError("invalid mask gradient")
            result[layer] = gradient.detach().double()
        return loss.detach().double(), result
    finally:
        for handle in hooks:
            handle.remove()


def derive(payload: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    torch = import_module("torch")
    counts = payload["family_counts"]
    nll = payload["family_nll_sum"]
    if (
        counts.dtype != torch.int64
        or counts.shape != (7,)
        or not bool((counts == 16).all())
        or nll.dtype != torch.float64
        or nll.shape != (7,)
        or not bool(torch.isfinite(nll).all())
        or bool((nll < 0).any())
    ):
        raise ValueError("invalid family counts/NLL")
    result: dict[str, Any] = {}
    ranking: list[tuple[float, int, int]] = []
    for layer in layers:
        prefix = f"layer.{layer}."
        signed, absolute = payload[prefix + "signed_sum"], payload[prefix + "absolute_sum"]
        if (
            signed.dtype != torch.float64
            or absolute.dtype != torch.float64
            or signed.ndim != 2
            or signed.shape[0] != 7
            or signed.shape[1] < 1
            or signed.shape != absolute.shape
            or not bool(torch.isfinite(signed).all() & torch.isfinite(absolute).all())
            or bool((absolute < 0).any())
            or bool((absolute < signed.abs() - 1e-12).any())
        ):
            raise ValueError("invalid signed/absolute gradient sums or conservation")
        necessity = (-signed[:2].div(counts[:2, None]).mean(0)).clamp_min(0)
        risk = (absolute[2:] / counts[2:, None]).amax(0)
        score = (necessity - risk).clamp_min(0)
        for key, value in [("necessity", necessity), ("retain_risk", risk), ("score", score)]:
            result[prefix + key] = value
        ranking.extend((float(value), layer, neuron) for neuron, value in enumerate(score))
    ranking.sort(key=lambda row: (-row[0], row[1], row[2]))
    result["ranking"] = torch.tensor(
        [[layer, neuron] for _, layer, neuron in ranking], dtype=torch.int64
    )
    return result


def validate_payload(metadata: dict[str, Any], payload: dict[str, Any]) -> None:
    torch = import_module("torch")
    layers = metadata.get("layers")
    if (
        type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") != VERSION
        or metadata.get("families") != list(FAMILIES)
        or not isinstance(layers, list)
        or not layers
        or any(type(layer) is not int or layer < 0 for layer in layers)
        or layers != sorted(set(layers))
    ):
        raise ValueError("relevance metadata differs")
    expected = {"family_counts", "family_nll_sum", "ranking"} | {
        f"layer.{layer}.{key}"
        for layer in layers
        for key in ("signed_sum", "absolute_sum", "necessity", "retain_risk", "score")
    }
    if set(payload) != expected or set(metadata["payload"]) != expected:
        raise ValueError("relevance payload inventory differs")
    for name, value in payload.items():
        row = metadata["payload"][name]
        if (
            value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())
            or list(value.shape) != row["shape"]
            or str(value.dtype) != row["dtype"]
            or payload_hash(value) != row["sha256"]
        ):
            raise ValueError("relevance payload hash/shape/dtype differs: " + name)
    for name, value in derive(payload, layers).items():
        if value.dtype != payload[name].dtype or not torch.equal(value, payload[name]):
            raise ValueError("relevance derived statistics/ranking differ: " + name)


def diagnose(runtime: Any, corpus: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime.progress("language_relevance", state="active", completed=0, total=112, unit="examples")
    flags, training, source = [], [], None
    try:
        rows = validate_corpus(corpus)
        torch = import_module("torch")
        config = runtime.config
        if (
            config.seed != 42
            or config.torch_threads != 2
            or config.device != "cpu"
            or config.dtype != "float32"
            or torch.get_num_threads() != 2
        ):
            raise ValueError("requires seed42 CPU float32 threads2 runtime")
        layers = sorted(language._layers(runtime))
        parameters = list(runtime.model.parameters())
        if any(p.grad is not None for p in parameters):
            raise ValueError("preexisting parameter gradients refused")
        if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in parameters):
            raise ValueError("requires CPU float32 parameters")
        runtime.progress("language_relevance_source", state="active")
        source = {
            "checkpoint_fingerprint": runtime.fingerprint,
            "source": language._inventory(runtime),
        }
        flags = [(p, p.requires_grad) for p in parameters]
        training = [(module, module.training) for module in runtime.model.modules()]
        for parameter in parameters:
            parameter.requires_grad_(False)
            parameter.grad = None
        runtime.model.eval()
        payload = {
            "family_counts": torch.zeros(7, dtype=torch.int64),
            "family_nll_sum": torch.zeros(7, dtype=torch.float64),
        }
        for layer in layers:
            width = runtime.mlp(layer).down_proj.weight.shape[1]
            for key in ("signed_sum", "absolute_sum"):
                payload[f"layer.{layer}.{key}"] = torch.zeros((7, width), dtype=torch.float64)
        examples = []
        for number, row in enumerate(rows, 1):
            runtime.check()
            batch = objectives.answer_batch(runtime.tokenizer, row, 128)
            loss, gradients = mask_gradient(runtime, batch)
            family = FAMILIES.index(row["family"])
            payload["family_counts"][family] += 1
            payload["family_nll_sum"][family] += loss
            for layer, gradient in gradients.items():
                payload[f"layer.{layer}.signed_sum"][family] += gradient
                payload[f"layer.{layer}.absolute_sum"][family] += gradient.abs()
            examples.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "variant_index": 0,
                    "answer_tokens": int((batch["labels"] != -100).sum()),
                    "nll": float(loss),
                    "input_ids_sha256": payload_hash(batch["input_ids"]),
                }
            )
            runtime.progress(
                "language_relevance",
                state="active",
                completed=number,
                total=len(rows),
                unit="examples",
            )
        payload.update(derive(payload, layers))
        metadata = {
            "schema_version": VERSION,
            **source,
            "layers": layers,
            "families": list(FAMILIES),
            "corpus_sha256": record_digest(corpus),
            "provenance": PROVENANCE,
            "examples": examples,
            "code_sha256": {
                Path(__file__).name: tensors.file_hash(Path(__file__)),
                "gradient_objectives.py": tensors.file_hash(Path(objectives.__file__)),
            },
            "payload": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": payload_hash(value),
                }
                for name, value in payload.items()
            },
        }
        validate_payload(metadata, payload)
    except BaseException:
        runtime.progress("language_relevance", state="failed")
        raise
    finally:
        for parameter, flag in flags:
            parameter.grad = None
            parameter.requires_grad_(flag)
        for module, was_training in training:
            module.training = was_training
        if source is not None:
            try:
                language._identity(runtime, source, check_budget=False)
            except BaseException:
                runtime.progress("language_relevance_source", state="failed")
                raise
    runtime.progress(
        "language_relevance", state="complete", completed=112, total=112, unit="examples"
    )
    return metadata, payload


def to_proposal(
    runtime: Any,
    metadata: dict[str, Any],
    payload: dict[str, Any],
    *,
    top_k: int,
    execution_fraction: float,
) -> dict[str, Any]:
    """Bind an explicit positive-score top-k choice to the existing zero-row operator."""
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
    if metadata.get("provenance") != PROVENANCE:
        raise ValueError("relevance method provenance differs")
    names = language._names(runtime)
    parameters = dict(runtime.model.named_parameters())
    if metadata["layers"] != sorted(names):
        raise ValueError("relevance runtime layer inventory differs")
    for layer, layer_names in names.items():
        width = parameters[layer_names[0]].shape[0]
        if payload[f"layer.{layer}.signed_sum"].shape != (len(FAMILIES), width):
            raise ValueError("relevance runtime feature width differs")
    # validate_payload independently rebuilds every score and the exact global ranking.
    ranking = payload["ranking"]
    positive = sum(int((payload[f"layer.{layer}.score"] > 0).sum()) for layer in names)
    if top_k > positive:
        raise ValueError("top_k exceeds positive-score neuron count")
    selected_ids = ranking[:top_k].tolist()
    rows_by_layer: dict[int, list[int]] = {layer: [] for layer in names}
    for layer, neuron in selected_ids:
        if float(payload[f"layer.{layer}.score"][neuron]) <= 0:
            raise ValueError("selected relevance score must be strictly positive")
        rows_by_layer[layer].append(neuron)
    selected, diffs = [], {}
    scores: list[float] = []
    for layer in names:
        layer_names = names[layer]
        width = parameters[layer_names[0]].shape[0]
        rows = rows_by_layer[layer]
        if len(rows) > math.ceil(width * execution_fraction):
            raise ValueError("exact top_k exceeds per-layer execution cap; no rows trimmed")
        selected.append({"layer": layer, "width": width, "rows": rows})
        scores.extend(float(payload[f"layer.{layer}.score"][neuron]) for neuron in rows)
        for name in layer_names if rows else []:
            delta = tensors._square_norm(parameters[name][rows])
            base = metadata["source"]["parameters"][name]["norm_squared"]
            relative = math.sqrt(delta / base) if base else 0.0
            if (
                not math.isfinite(delta)
                or not math.isfinite(relative)
                or relative > language.MAX_TENSOR_RELATIVE + 1e-9
            ):
                raise ValueError("relevance proposal exceeds existing tensor change bound")
            diffs[name] = {"delta_norm_squared": delta, "relative_frobenius": relative}
    total = math.fsum(row["delta_norm_squared"] for row in diffs.values())
    relative = math.sqrt(total / metadata["source"]["norm_squared"])
    if not math.isfinite(relative) or total <= 0 or relative > language.MAX_MODEL_RELATIVE + 1e-9:
        raise ValueError("relevance proposal is no-op or exceeds existing model change bound")
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
        "relevance_selection": {
            "method": PROVENANCE["method"],
            "top_k": top_k,
            "ranked_ids": selected_ids,
            "positive_score_count": positive,
            "diagnosis_sha256": diagnosis_pin,
            "payload_manifest_sha256": record_digest(metadata["payload"]),
            "score_min": min(scores),
            "score_max": max(scores),
            "score_is_calibrated_confidence": False,
        },
        "storage_roundtrip": "zero exactly representable in source dtype",
        "execution_fraction_role": "per-layer safety ceiling; not a selection fraction",
    }
    language.touched_tensors(runtime, proposal)
    return proposal
