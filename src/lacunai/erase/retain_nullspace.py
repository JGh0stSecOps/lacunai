"""Bounded retain-mean nullspace proposals; never mutate or write a checkpoint."""

from __future__ import annotations

import math
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Any

from lacunai.erase import language, methods
from lacunai.map import retain_nullspace as mapping
from lacunai.map.localize import record_digest

MAX_TENSOR_RELATIVE = 0.10
MAX_MODEL_RELATIVE = 0.02
TOLERANCE = 1e-8
_ROWS = 128


def validate_spec(spec: dict[str, Any], diagnosis: dict[str, Any]) -> list[int]:
    fields = {"schema_version", "method", "map_sha256", "layers", "k", "strength"}
    if not isinstance(spec, dict) or set(spec) != fields:
        raise ValueError("retain-nullspace specification fields differ")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("retain-nullspace schema version differs")
    if spec["method"] != "retain-nullspace":
        raise ValueError("retain-nullspace method differs")
    if spec["map_sha256"] != record_digest(diagnosis):
        raise ValueError("retain-nullspace map hash differs")
    layers = spec["layers"]
    if (
        not isinstance(layers, list)
        or not 1 <= len(layers) <= 8
        or any(type(layer) is not int or layer < 0 for layer in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError("retain-nullspace requires one to eight unique layers")
    if type(spec["k"]) is not int or spec["k"] not in {1, 4, 16}:
        raise ValueError("retain-nullspace direction count differs")
    if type(spec["strength"]) not in (int, float) or spec["strength"] not in {0.25, 0.5, 1.0}:
        raise ValueError("retain-nullspace strength differs")
    return list(layers)


def _basis(retain: Any, directions: Any, width: int, k: int) -> dict[str, float]:
    torch = import_module("torch")
    if (
        retain.ndim != 2
        or retain.shape[0] < 1
        or retain.shape[1] != width
        or directions.shape != (width, k)
        or retain.dtype != torch.float64
        or directions.dtype != torch.float64
        or retain.device.type != "cpu"
        or directions.device.type != "cpu"
        or not bool(torch.isfinite(retain).all())
        or not bool(torch.isfinite(directions).all())
    ):
        raise ValueError("retain-nullspace basis requires matching finite CPU float64 matrices")
    orthogonal = float((directions.T @ directions - torch.eye(k, dtype=torch.float64)).abs().max())
    norm = float(torch.linalg.vector_norm(retain))
    residual = float(torch.linalg.vector_norm(retain @ directions))
    normalized = residual / max(norm, 1e-12)
    if orthogonal > TOLERANCE or normalized > TOLERANCE:
        raise ValueError(
            "retain-nullspace basis orthogonality or retain residual exceeds tolerance"
        )
    return {
        "orthogonality_max_abs": orthogonal,
        "retain_basis_product_frobenius": residual,
        "retain_basis_product_normalized": normalized,
        "normalization_denominator": max(norm, 1e-12),
    }


def _leakage(original: Any, stored: Any, retain: Any) -> dict[str, Any]:
    torch = import_module("torch")
    leakage_squares, delta_squares = [], []
    for start in range(0, original.shape[0], _ROWS):
        delta = stored[start : start + _ROWS].to(torch.float64) - original[
            start : start + _ROWS
        ].to(torch.float64)
        leak = delta @ retain.T
        if not bool(torch.isfinite(leak).all()):
            raise ValueError("nonfinite post-storage retain leakage")
        leakage_squares.append(float(torch.sum(leak.square())))
        delta_squares.append(float(torch.sum(delta.square())))
    leak_norm = math.sqrt(math.fsum(leakage_squares))
    delta_norm = math.sqrt(math.fsum(delta_squares))
    retained_norm = float(torch.linalg.vector_norm(retain))
    denominator = delta_norm * retained_norm
    return {
        "post_storage_delta_norm_squared_fp64": math.fsum(delta_squares),
        "post_storage_delta_retain_product_frobenius": leak_norm,
        "post_storage_delta_retain_product_normalized": leak_norm / denominator
        if denominator
        else None,
        "normalization_definition": "||Delta W R^T||F / (||Delta W||F * ||R||F)",
        "normalization_denominator": denominator,
        "zero_denominator": "undefined" if not denominator else None,
        "claim": "measured on saved-precision proposal and sampled retained means only",
    }


def propose(
    runtime: Any, diagnosis: dict[str, Any], artifact_dir: Path, spec: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return rounded replacements and measured diagnostics; caller owns snapshot/apply/undo."""
    layers = validate_spec(spec, diagnosis)
    if diagnosis.get("checkpoint_fingerprint") != runtime.fingerprint:
        raise ValueError("retain-nullspace source checkpoint identity differs")
    torch = import_module("torch")
    available = language._layers(runtime)
    if any(layer not in available for layer in layers):
        raise ValueError("retain-nullspace selected layer is absent")
    runtime.progress("retain_nullspace_source_verification", state="active")
    try:
        language._identity(runtime, diagnosis)
    except BaseException as exc:
        runtime.progress(
            "retain_nullspace_source_verification",
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    runtime.progress("retain_nullspace_source_verification", state="complete")
    parameters = dict(runtime.model.named_parameters())
    replacements: dict[str, Any] = {}
    originals: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {
        "method": "retain-nullspace",
        "checkpoint_fingerprint": runtime.fingerprint,
        "map_sha256": record_digest(diagnosis),
        "spec_sha256": record_digest(spec),
        "arithmetic_dtype": "float64",
        "runtime_dtype": "float32",
        "random_operations": False,
        "reproducibility_limits": "CPU BLAS reductions can differ across libraries or thread "
        "settings; caller records runtime configuration, and numerical tolerances remain explicit.",
        "projection": "W - strength * (W V) V^T",
        "basis_tolerance": TOLERANCE,
        "max_tensor_relative": MAX_TENSOR_RELATIVE,
        "max_model_relative": MAX_MODEL_RELATIVE,
        "layers": [],
        "limits": "Association in sampled prompt means, not causal necessity or knowledge absence. "
        "Storage rounding can introduce retain leakage; downstream behavioral tests are required.",
    }
    runtime.progress("retain_nullspace_projection", completed=0, total=len(layers), unit="layers")
    try:
        with torch.no_grad():
            for completed, layer in enumerate(layers, 1):
                runtime.check()
                name = f"model.layers.{layer}.mlp.down_proj.weight"
                weight = parameters.get(name)
                if (
                    weight is None
                    or weight is not runtime.mlp(layer).down_proj.weight
                    or weight.ndim != 2
                    or weight.dtype != torch.float32
                    or weight.device.type != "cpu"
                    or min(weight.shape) < 1
                ):
                    raise ValueError("retain-nullspace requires canonical CPU float32 down weights")
                retain, directions = mapping.load_layer(diagnosis, artifact_dir, layer, spec["k"])
                invariant = _basis(retain, directions, weight.shape[1], spec["k"])
                candidate = torch.empty_like(weight)
                pre_round_squares = []
                for start in range(0, weight.shape[0], _ROWS):
                    runtime.check()
                    block = weight[start : start + _ROWS].detach().to(torch.float64)
                    update = float(spec["strength"]) * (block @ directions) @ directions.T
                    projected = block - update
                    # Reuse this buffer for the actual FP64 subtraction delta.
                    update.copy_(projected).sub_(block)
                    pre_round_squares.append(float(torch.linalg.vector_norm(update)) ** 2)
                    if not bool(torch.isfinite(projected).all()):
                        raise ValueError("retain-nullspace projection produced nonfinite values")
                    candidate[start : start + _ROWS].copy_(projected.to(torch.float32))
                stored = methods.round_to_storage({name: candidate}, runtime.storage_dtypes)[name]
                originals[name] = weight.detach()
                replacements[name] = stored
                leakage = _leakage(weight, stored, retain)
                pre_round = math.fsum(pre_round_squares)
                post_round = leakage["post_storage_delta_norm_squared_fp64"]
                diagnostics["layers"].append(
                    {
                        "layer": layer,
                        "tensor": name,
                        "k": spec["k"],
                        "strength": spec["strength"],
                        "storage_dtype": runtime.storage_dtypes[name],
                        "invariants": invariant,
                        "retain_leakage": leakage,
                        "projection_energy": {
                            "before_rounding_delta_norm_squared": pre_round,
                            "after_rounding_delta_norm_squared": post_round,
                            "retained_delta_energy": str(
                                Decimal(str(post_round)) / Decimal(str(pre_round))
                            )
                            if pre_round
                            else None,
                            "zero_denominator": "undefined" if not pre_round else None,
                            "definition": (
                                "post-storage delta squared norm / "
                                "pre-cast FP64 projection delta squared norm"
                            ),
                            "before_dtype": "float64",
                            "after_storage_dtype": runtime.storage_dtypes[name],
                        },
                    }
                )
                runtime.progress(
                    "retain_nullspace_projection",
                    completed=completed,
                    total=len(layers),
                    unit="layers",
                )
        diagnostics["tensor_diffs"] = methods.validate_replacements(
            originals, replacements, MAX_TENSOR_RELATIVE
        )
        for row in diagnostics["layers"]:
            source_norm = diagnosis["source"]["parameters"][row["tensor"]]["norm_squared"]
            actual_square = row["projection_energy"]["after_rounding_delta_norm_squared"]
            actual_relative = math.sqrt(actual_square / source_norm) if source_norm else 0.0
            if actual_relative > MAX_TENSOR_RELATIVE + 1e-9:
                raise ValueError("retain-nullspace actual stored tensor change exceeds bound")
            diagnostics["tensor_diffs"][row["tensor"]]["post_storage_relative_frobenius_fp64"] = (
                actual_relative
            )
        delta_squared = math.fsum(
            row["projection_energy"]["after_rounding_delta_norm_squared"]
            for row in diagnostics["layers"]
        )
        pre_round_squared = math.fsum(
            row["projection_energy"]["before_rounding_delta_norm_squared"]
            for row in diagnostics["layers"]
        )
        diagnostics["projection_energy"] = {
            "before_rounding_delta_norm_squared": pre_round_squared,
            "after_rounding_delta_norm_squared": delta_squared,
            "retained_delta_energy": str(
                Decimal(str(delta_squared)) / Decimal(str(pre_round_squared))
            )
            if pre_round_squared
            else None,
            "zero_denominator": "undefined" if not pre_round_squared else None,
            "definition": (
                "post-storage delta squared norm / pre-cast FP64 projection delta squared norm"
            ),
        }
        source_squared = diagnosis["source"]["norm_squared"]
        relative = math.sqrt(delta_squared / source_squared)
        if not math.isfinite(relative) or relative > MAX_MODEL_RELATIVE + 1e-9:
            raise ValueError("retain-nullspace model relative change exceeds bound")
        diagnostics["model_relative_frobenius"] = relative
        language._identity(runtime, diagnosis)
        runtime.progress("retain_nullspace_projection", state="complete")
        return replacements, diagnostics
    except BaseException as exc:
        runtime.progress(
            "retain_nullspace_projection", state="failed", error=f"{type(exc).__name__}: {exc}"
        )
        raise
