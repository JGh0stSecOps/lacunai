"""Bounded language-feature ablation; association is not knowledge-erasure evidence."""

from __future__ import annotations

import hashlib
import json
import math
from importlib import import_module
from pathlib import Path
from typing import Any

from lacunai.core.spec import bounded_file
from lacunai.experimental import tensors
from lacunai.map.fingerprints import capture
from lacunai.map.localize import record_digest
from lacunai.restore import checkpoints

FRACTIONS = (0.01, 0.025, 0.05, 0.10, 0.20)
EPSILON = 1e-6
MAX_TENSOR_RELATIVE = 0.50
MAX_MODEL_RELATIVE = 0.15
_CHUNK = 1024 * 1024


def _layers(runtime: Any) -> list[int]:
    if getattr(runtime.model.config, "model_type", None) != "qwen2":
        raise ValueError("language operator requires built-in Qwen2 gated MLPs")
    layers = list(runtime.layer_ids)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("invalid runtime layer inventory")
    return layers


def _names(runtime: Any) -> dict[int, list[str]]:
    result = {}
    parameters = dict(runtime.model.named_parameters())
    for layer in _layers(runtime):
        module = runtime.mlp(layer)
        names = [name for name, value in runtime.model.named_modules() if value is module]
        if len(names) != 1:
            raise ValueError("MLP module identity is ambiguous")
        selected = [f"{names[0]}.up_proj.weight"]
        if module.up_proj.bias is not None:
            selected.append(f"{names[0]}.up_proj.bias")
        for name in selected:
            if name not in parameters or runtime.storage_dtypes.get(name) not in {
                "BF16",
                "F16",
                "F32",
            }:
                raise ValueError("missing canonical stored up-projection tensor")
            value = parameters[name]
            if str(value.dtype) != "torch.float32" or value.device.type != "cpu":
                raise ValueError("language operator requires CPU float32 runtime")
        if module.up_proj.weight.shape[0] != module.down_proj.weight.shape[1]:
            raise ValueError("gated feature/up-projection dimensions differ")
        result[layer] = selected
    return result


def _inventory(runtime: Any) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, parameter in runtime.model.named_parameters():
        runtime.check()
        records[name] = {
            "shape": list(parameter.shape),
            "sha256": tensors.tensor_hash(parameter),
            "norm_squared": tensors._square_norm(parameter),
        }
    total = math.fsum(row["norm_squared"] for row in records.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("source model norm is undefined")
    return {
        "parameters": records,
        "norm_squared": total,
        "deduplication": "named_parameters canonical storage aliases counted once",
    }


def _identity(runtime: Any, record: dict[str, Any], *, check_budget: bool = True) -> None:
    if record.get("checkpoint_fingerprint") != runtime.fingerprint:
        raise ValueError("language checkpoint identity mismatch")
    source = record["source"]["parameters"]
    current = dict(runtime.model.named_parameters())
    if set(source) != set(current):
        raise ValueError("language source parameter names changed")
    norms = []
    for name, parameter in current.items():
        if check_budget:
            runtime.check()
        if (
            list(parameter.shape) != source[name]["shape"]
            or tensors.tensor_hash(parameter) != source[name]["sha256"]
        ):
            raise ValueError(f"language source parameter changed: {name}")
        norm = tensors._square_norm(parameter)
        if not math.isclose(norm, source[name]["norm_squared"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("language source norm record differs")
        norms.append(norm)
    if not math.isclose(
        math.fsum(norms), record["source"]["norm_squared"], rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("language source model norm differs")


def diagnose(runtime: Any, pairs: list[dict[str, str]], background: list[str]) -> dict[str, Any]:
    """Equal prompt weight; average each prompt's mean absolute activity and RMS."""
    torch = import_module("torch")
    layers = _layers(runtime)
    _names(runtime)
    if not pairs or len(pairs) > 256 or not background or len(background) > 256:
        raise ValueError("bounded matched development pairs and retain background are required")
    for row in pairs:
        if any(
            not isinstance(row.get(key), str) or not row[key].strip() or len(row[key]) > 8192
            for key in ("diagnostic_zh", "diagnostic_en")
        ):
            raise ValueError("invalid matched language diagnostic text")
        if row["diagnostic_zh"] == row["diagnostic_en"]:
            raise ValueError("matched Chinese and English diagnostics must differ")
    if any(
        not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8192
        for prompt in background
    ):
        raise ValueError("invalid retain background text")
    source = _inventory(runtime)
    corpora = {
        "chinese": [p["diagnostic_zh"] for p in pairs],
        "nonchinese": [p["diagnostic_en"] for p in pairs] + background,
    }
    aggregates: dict[str, dict[int, dict[str, Any]]] = {}
    for split, prompts in corpora.items():
        sums: dict[int, dict[str, Any]] = {}
        runtime.progress(
            "language_diagnosis", split=split, completed=0, total=len(prompts), unit="prompts"
        )
        for completed, prompt in enumerate(prompts, 1):
            activations = capture(runtime, prompt, layers)
            for layer, values in activations.items():
                value = values[0].to(dtype=torch.float64)
                absolute = value.abs().mean(0)
                rms = value.square().mean(0).sqrt()
                if layer not in sums:
                    sums[layer] = {"mean_abs": absolute, "mean_prompt_rms": rms}
                else:
                    sums[layer]["mean_abs"] += absolute
                    sums[layer]["mean_prompt_rms"] += rms
            del activations
            runtime.progress(
                "language_diagnosis",
                split=split,
                completed=completed,
                total=len(prompts),
                unit="prompts",
            )
        aggregates[split] = {
            layer: {key: value / len(prompts) for key, value in row.items()}
            for layer, row in sums.items()
        }
    records = {}
    for layer in layers:
        zh, other = aggregates["chinese"][layer], aggregates["nonchinese"][layer]
        score = zh["mean_abs"] / (other["mean_abs"] + EPSILON) * zh["mean_abs"]
        if not bool(torch.isfinite(score).all()):
            raise ValueError("nonfinite language feature score")
        records[str(layer)] = {
            "width": score.numel(),
            "score": score.tolist(),
            "chinese": {k: v.tolist() for k, v in zh.items()},
            "nonchinese": {k: v.tolist() for k, v in other.items()},
        }
    result = {
        "schema_version": 1,
        "checkpoint_fingerprint": runtime.fingerprint,
        "source": source,
        "layers": records,
        "data_sha256": record_digest({"pairs": pairs, "background": background}),
        "prompt_counts": {key: len(value) for key, value in corpora.items()},
        "epsilon": EPSILON,
        "hook": "down_proj input: post-gate MLP features",
        "pooling": "per-prompt prefill mean absolute and RMS, then equal prompt average",
        "limits": "Correlation only. Shared language features may damage Japanese, Spanish "
        "and unrelated capabilities. Does not establish knowledge absence.",
    }
    _identity(runtime, result)
    return result


def select_rows(chinese: list[float], other: list[float], fraction: float) -> list[int]:
    """Deterministic ranking over code-computed scalar diagnostics."""
    if fraction not in FRACTIONS or isinstance(fraction, bool):
        raise ValueError("fraction is not on the frozen candidate ladder")
    if not chinese or len(chinese) != len(other):
        raise ValueError("feature dimensions differ")
    if any(not math.isfinite(value) or value < 0 for value in chinese + other):
        raise ValueError("invalid mean absolute feature activity")
    eligible = [index for index, (zh, en) in enumerate(zip(chinese, other, strict=True)) if zh > en]
    ranked = sorted(
        eligible,
        key=lambda index: (-(chinese[index] / (other[index] + EPSILON) * chinese[index]), index),
    )
    return ranked[: math.ceil(fraction * len(chinese))]


def propose(runtime: Any, diagnosis: dict[str, Any], fraction: float) -> dict[str, Any]:
    _identity(runtime, diagnosis)
    names = _names(runtime)
    if set(diagnosis["layers"]) != {str(layer) for layer in names}:
        raise ValueError("diagnosis does not cover every runtime layer")
    parameters = dict(runtime.model.named_parameters())
    selected, diffs = [], {}
    for layer, layer_names in names.items():
        row = diagnosis["layers"][str(layer)]
        width = parameters[layer_names[0]].shape[0]
        if row["width"] != width or len(row["chinese"]["mean_abs"]) != width:
            raise ValueError("diagnostic feature width differs from model")
        rows = select_rows(row["chinese"]["mean_abs"], row["nonchinese"]["mean_abs"], fraction)
        selected.append({"layer": layer, "rows": rows, "width": width})
        if not rows:
            continue
        for name in layer_names:
            before = diagnosis["source"]["parameters"][name]
            delta = tensors._square_norm(parameters[name][rows])
            relative = math.sqrt(delta / before["norm_squared"]) if before["norm_squared"] else 0.0
            if relative > MAX_TENSOR_RELATIVE + 1e-9:
                raise ValueError(f"language tensor relative bound exceeded: {name}")
            diffs[name] = {"delta_norm_squared": delta, "relative_frobenius": relative}
    total = math.fsum(row["delta_norm_squared"] for row in diffs.values())
    relative = math.sqrt(total / diagnosis["source"]["norm_squared"])
    if not total or relative > MAX_MODEL_RELATIVE + 1e-9:
        raise ValueError("language proposal is no-op or exceeds model relative bound")
    return {
        "schema_version": 1,
        "checkpoint_fingerprint": runtime.fingerprint,
        "source": diagnosis["source"],
        "diagnosis_sha256": record_digest(diagnosis),
        "fraction": fraction,
        "strength": 1.0,
        "selected": selected,
        "tensor_diffs": diffs,
        "model_relative_frobenius": relative,
        "max_tensor_relative": MAX_TENSOR_RELATIVE,
        "max_model_relative": MAX_MODEL_RELATIVE,
        "operation": "zero_language_up_projection_rows",
        "storage_roundtrip": "zero exactly representable in source dtype",
    }


def touched_tensors(runtime: Any, proposal: dict[str, Any]) -> list[str]:
    names = _names(runtime)
    selection = proposal.get("selected")
    if not isinstance(selection, list) or [r.get("layer") for r in selection] != list(names):
        raise ValueError("proposal must explicitly cover all layers in runtime order")
    result = []
    for row in selection:
        width = runtime.mlp(row["layer"]).up_proj.weight.shape[0]
        rows = row.get("rows")
        if (
            row.get("width") != width
            or not isinstance(rows, list)
            or len(rows) != len(set(rows))
            or any(type(i) is not int or not 0 <= i < width for i in rows)
        ):
            raise ValueError("invalid language row selection")
        if rows:
            result.extend(names[row["layer"]])
    if not result or set(result) != set(proposal["tensor_diffs"]):
        raise ValueError("language touched tensors differ from proposed numerical deltas")
    return result


def _stored_hash(parameter: Any, dtype: str) -> str:
    torch = import_module("torch")
    target = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[dtype]
    result = hashlib.sha256()
    flat = parameter.detach().reshape(-1)
    for offset in range(0, flat.numel(), _CHUNK):
        block = flat[offset : offset + _CHUNK].to(dtype=target)
        if not torch.equal(block.float(), flat[offset : offset + _CHUNK]):
            raise ValueError("loaded source does not roundtrip to recorded storage precision")
        result.update(block.contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def apply(runtime: Any, proposal: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Temporarily mutate an isolated model only after matching durable before-images."""
    torch = import_module("torch")
    _identity(runtime, proposal)
    names = touched_tensors(runtime, proposal)
    if (
        proposal.get("operation") != "zero_language_up_projection_rows"
        or proposal.get("strength") != 1.0
        or proposal.get("fraction") not in FRACTIONS
        or proposal.get("max_tensor_relative") != MAX_TENSOR_RELATIVE
        or proposal.get("max_model_relative") != MAX_MODEL_RELATIVE
    ):
        raise ValueError("language operation differs from frozen contract")
    if snapshot.get("status") != "prepared" or set(snapshot.get("tensors", {})) != set(names):
        raise ValueError("language edit requires a matching prepared durable snapshot")
    snapshot_directory = Path(snapshot["run_dir"])
    if json.loads(bounded_file(snapshot_directory / "snapshot.json")) != snapshot:
        raise ValueError("language snapshot differs from durable receipt")
    parameters = dict(runtime.model.named_parameters())
    expected = {}
    deltas = []
    name_map = _names(runtime)
    for row in proposal["selected"]:
        if len(row["rows"]) > math.ceil(proposal["fraction"] * row["width"]):
            raise ValueError("language row count exceeds selected fraction")
        for name in name_map[row["layer"]] if row["rows"] else []:
            value = parameters[name]
            descriptor = snapshot["tensors"][name]
            dtype = runtime.storage_dtypes[name]
            before_bytes = checkpoints._blob(snapshot_directory, descriptor, snapshot["max_bytes"])
            if hashlib.sha256(before_bytes).hexdigest() != descriptor["before_sha256"]:
                raise ValueError("language durable before-image bytes differ")
            del before_bytes
            if (
                descriptor.get("shape") != list(value.shape)
                or descriptor.get("dtype") != dtype
                or descriptor.get("before_sha256") != _stored_hash(value, dtype)
            ):
                raise ValueError("durable before-image differs from loaded language source")
            delta = tensors._square_norm(value[row["rows"]])
            original_norm = tensors._square_norm(value)
            relative = math.sqrt(delta / original_norm) if original_norm else 0.0
            if relative > MAX_TENSOR_RELATIVE + 1e-9:
                raise ValueError("language tensor bound exceeded before apply")
            deltas.append(delta)
            expected[name] = row["rows"]
    model_relative = math.sqrt(math.fsum(deltas) / proposal["source"]["norm_squared"])
    if not any(deltas) or model_relative > MAX_MODEL_RELATIVE + 1e-9:
        raise ValueError("language model bound exceeded or edit is no-op")
    state: dict[str, Any] = {
        "before_images": {},
        "source": proposal["source"],
        "checkpoint_fingerprint": runtime.fingerprint,
        "diagnostics": {"tensor_hashes": {}, "model_relative_frobenius": model_relative},
    }
    try:
        with torch.no_grad():
            for completed, (name, rows) in enumerate(expected.items(), 1):
                runtime.check()
                value = parameters[name]
                state["before_images"][name] = value.detach().clone()
                value[rows] = 0
                if bool((value[rows] != 0).any()):
                    raise ValueError("selected language rows were not silenced")
                before = state["before_images"][name]
                # Compare in bounded row chunks; no second full proposal matrix.
                row_set = set(rows)
                for start in range(0, value.shape[0], 128):
                    ids = [
                        i
                        for i in range(start, min(start + 128, value.shape[0]))
                        if i not in row_set
                    ]
                    if ids and not torch.equal(value[ids], before[ids]):
                        raise ValueError("unselected language rows changed")
                tensors._square_norm(value)
                state["diagnostics"]["tensor_hashes"][name] = tensors.tensor_hash(value)
                runtime.progress(
                    "language_apply", completed=completed, total=len(expected), unit="tensors"
                )
        for name, value in parameters.items():
            if (
                name not in expected
                and tensors.tensor_hash(value) != proposal["source"]["parameters"][name]["sha256"]
            ):
                raise ValueError("untouched model parameter changed during language edit")
        return state
    except BaseException:
        restore(runtime, state)
        raise


def restore(runtime: Any, state: dict[str, Any]) -> dict[str, Any]:
    torch = import_module("torch")
    if runtime.fingerprint != state.get("checkpoint_fingerprint"):
        raise ValueError("restore runtime checkpoint differs")
    parameters = dict(runtime.model.named_parameters())
    for name, value in state["before_images"].items():
        if name not in parameters or list(parameters[name].shape) != list(value.shape):
            raise ValueError("restore tensor identity/shape differs")
        if tensors.tensor_hash(value) != state["source"]["parameters"][name]["sha256"]:
            raise ValueError("restore before-image hash differs")
    with torch.no_grad():
        for name, value in state["before_images"].items():
            parameters[name].copy_(value)
    _identity(runtime, state, check_budget=False)
    return {
        "restored_bit_exact": True,
        "tensor_count": len(state["before_images"]),
        "verification": "all canonical model parameter hashes match pre-edit source",
    }
