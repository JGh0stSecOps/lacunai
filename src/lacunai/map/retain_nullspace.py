"""Retain-mean nullspace diagnostics; no causal or knowledge-absence claim."""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path
from typing import Any

from lacunai.erase import language
from lacunai.map import fingerprints
from lacunai.map.language_entropy import validate_corpora
from lacunai.map.localize import record_digest

LANGUAGES = ("zh", "en", "es", "ja")
RETAIN_LANGUAGES = ("en", "es", "ja")
PARAMETERS: dict[str, Any] = {
    "rank_atol": 1e-10,
    "rank_rtol": 1e-8,
    "orthogonality_atol": 1e-8,
    "retain_leakage_tolerance": 1e-8,
    "maximum_directions": 16,
    "svd_dtype": "float64",
    "centered": False,
    "tokenization": "raw:add_special_tokens=True:no-truncation",
    "aggregation": "equal prompt weight; signed mean across each prompt prefill",
}


def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _rank(singular: Any) -> tuple[int, float]:
    maximum = float(singular[0]) if singular.numel() else 0.0
    threshold = max(PARAMETERS["rank_atol"], PARAMETERS["rank_rtol"] * maximum)
    return int((singular > threshold).sum()), threshold


def invariants(retain: Any, directions: Any) -> dict[str, float]:
    torch = import_module("torch")
    if (
        retain.ndim != 2
        or directions.ndim != 2
        or retain.shape[1] != directions.shape[0]
        or retain.dtype != torch.float64
        or directions.dtype != torch.float64
        or not bool(torch.isfinite(retain).all())
        or not bool(torch.isfinite(directions).all())
    ):
        raise ValueError("invalid FP64 retained means/directions")
    if directions.shape[1] == 0:
        return {"orthogonality_max_abs": 0.0, "retain_leakage_relative": 0.0}
    eye = torch.eye(directions.shape[1], dtype=torch.float64)
    orthogonal = float((directions.T @ directions - eye).abs().max())
    leakage = float(torch.linalg.vector_norm(retain @ directions)) / max(
        float(torch.linalg.vector_norm(retain)), 1e-12
    )
    if orthogonal > 1e-8 or leakage > 1e-8:
        raise ValueError("retained-nullspace numerical invariants failed")
    return {"orthogonality_max_abs": orthogonal, "retain_leakage_relative": leakage}


def analyze(retain: Any, target: Any) -> tuple[dict[str, Any], Any]:
    """Uncentered thin SVD; retain all numerically supported retained directions."""
    torch = import_module("torch")
    retain = torch.as_tensor(retain, dtype=torch.float64, device="cpu")
    target = torch.as_tensor(target, dtype=torch.float64, device="cpu")
    if (
        retain.ndim != 2
        or target.ndim != 2
        or retain.shape[1] != target.shape[1]
        or min(*retain.shape, *target.shape) < 1
        or not bool(torch.isfinite(retain).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise ValueError("invalid retained/target prompt means")
    _, retained_s, retained_vh = torch.linalg.svd(retain, full_matrices=False)
    rank, threshold = _rank(retained_s)
    basis = retained_vh[:rank].T
    residual = target - (target @ basis) @ basis.T
    _, residual_s, residual_vh = torch.linalg.svd(residual, full_matrices=False)
    residual_rank, residual_threshold = _rank(residual_s)
    target_s = torch.linalg.svdvals(target)
    directions = residual_vh[: min(16, residual_rank)].T.contiguous()
    # Resolve sign only; repeated singular values still admit nonunique subspaces.
    for column in range(directions.shape[1]):
        pivot = int(directions[:, column].abs().argmax())
        if directions[pivot, column] < 0:
            directions[:, column].neg_()
    checks = invariants(retain, directions)
    gaps = {}
    for k in (1, 4, 16):
        if k <= residual_rank:
            gap = float(residual_s[k - 1] - residual_s[k]) if k < residual_s.numel() else None
            gaps[str(k)] = {
                "absolute_gap": gap,
                "degenerate_boundary": gap is not None and gap <= residual_threshold,
            }
    return {
        "width": retain.shape[1],
        "retain_rows": retain.shape[0],
        "target_rows": target.shape[0],
        "retain_rank": rank,
        "residual_rank": residual_rank,
        "available_directions": directions.shape[1],
        "retained_singular_values": retained_s.tolist(),
        "target_singular_values": target_s.tolist(),
        "residual_singular_values": residual_s.tolist(),
        "rank_thresholds": {"retain": threshold, "residual": residual_threshold},
        "boundary_gaps": gaps,
        "invariants": checks,
    }, directions


def diagnose(runtime: Any, corpora: dict[str, list[str]], artifact_dir: Path) -> dict[str, Any]:
    runtime.progress("retain_nullspace_diagnosis", state="active")
    try:
        return _diagnose(runtime, corpora, artifact_dir)
    except BaseException as exc:
        runtime.progress(
            "retain_nullspace_diagnosis", state="failed", error=f"{type(exc).__name__}: {exc}"
        )
        raise


def _diagnose(runtime: Any, corpora: dict[str, list[str]], artifact_dir: Path) -> dict[str, Any]:
    torch = import_module("torch")
    safetensors = import_module("safetensors.torch")
    validate_corpora(corpora)
    if any(len(corpora[key]) != 64 for key in LANGUAGES):
        raise ValueError("nullspace diagnosis requires exactly 64 matched prompts per language")
    layers = language._layers(runtime)
    if layers != list(range(24)):
        raise ValueError("nullspace diagnosis requires the frozen 24-layer Qwen2 model")
    artifact_dir = Path(artifact_dir)
    if artifact_dir.is_symlink():
        raise ValueError("diagnostic artifact directory must not be a symlink")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if any(artifact_dir.iterdir()):
        raise ValueError("diagnostic artifact directory must be empty")
    source = language._inventory(runtime)
    vectors: dict[str, dict[int, list[Any]]] = {
        key: {layer: [] for layer in layers} for key in LANGUAGES
    }
    runtime.progress(
        "retain_nullspace_capture", state="active", completed=0, total=256, unit="prompts"
    )
    completed = 0
    try:
        for key in LANGUAGES:
            for prompt in corpora[key]:
                captured = fingerprints.capture(runtime, prompt, layers, raw=True, mean_only=True)
                for layer, mean in captured.items():
                    vectors[key][layer].append(mean)
                completed += 1
                runtime.progress(
                    "retain_nullspace_capture",
                    language=key,
                    completed=completed,
                    total=256,
                    unit="prompts",
                )
        records = {}
        for completed_layers, layer in enumerate(layers):
            runtime.check()
            runtime.progress(
                "retain_nullspace_svd",
                state="active",
                layer=layer,
                completed=completed_layers,
                total=len(layers),
                unit="layers",
            )
            retained = torch.stack(
                [value for key in RETAIN_LANGUAGES for value in vectors[key].pop(layer)]
            )
            target = torch.stack(vectors["zh"].pop(layer))
            record, directions = analyze(retained, target)
            path = artifact_dir / f"layer-{layer:02d}.safetensors"
            safetensors.save_file(
                {"retain_means": retained.contiguous(), "directions": directions}, str(path)
            )
            loaded = safetensors.load_file(str(path), device="cpu")
            if not torch.equal(loaded["retain_means"], retained) or not torch.equal(
                loaded["directions"], directions
            ):
                raise ValueError("diagnostic tensor payload readback differs")
            record.update(tensor_file=path.name, tensor_file_sha256=_hash(path))
            records[str(layer)] = record
            runtime.progress(
                "retain_nullspace_svd",
                completed=completed_layers + 1,
                total=len(layers),
                unit="layers",
            )
        result = {
            "schema_version": 1,
            "method": "retain-nullspace",
            "checkpoint_fingerprint": runtime.fingerprint,
            "source": source,
            "corpus_sha256": record_digest(corpora),
            "parameters": PARAMETERS,
            "languages": list(LANGUAGES),
            "retain_languages": list(RETAIN_LANGUAGES),
            "prompt_counts": {key: len(corpora[key]) for key in LANGUAGES},
            "layers": records,
            "code_sha256": {
                path.name: _hash(path)
                for path in (Path(__file__), Path(fingerprints.__file__), Path(language.__file__))
            },
            "limits": "Preserves sampled original retained prompt-mean contributions "
            "before rounding, "
            "not token activations, downstream states, or knowledge absence. Degenerate SVD bases "
            "are not unique; CPU LAPACK implementations can differ.",
        }
        language._identity(runtime, result)
        result["diagnosis_sha256"] = record_digest(result)
        runtime.progress("retain_nullspace_diagnosis", state="complete")
        return result
    except BaseException as exc:
        runtime.progress(
            "retain_nullspace_diagnosis", state="failed", error=f"{type(exc).__name__}: {exc}"
        )
        raise


def load_layer(
    diagnosis: dict[str, Any], artifact_dir: Path, layer: int, k: int
) -> tuple[Any, Any]:
    if (
        type(layer) is not int
        or type(k) is not int
        or k not in (1, 4, 16)
        or diagnosis.get("parameters") != PARAMETERS
        or diagnosis.get("diagnosis_sha256")
        != record_digest(
            {key: value for key, value in diagnosis.items() if key != "diagnosis_sha256"}
        )
    ):
        raise ValueError("diagnostic identity/parameters or requested k differs")
    record = diagnosis["layers"].get(str(layer))
    if not record or k > record["available_directions"] or k > record["residual_rank"]:
        raise ValueError("insufficient residual numerical rank for requested k")
    basename = record["tensor_file"]
    if basename != f"layer-{layer:02d}.safetensors":
        raise ValueError("diagnostic payload filename differs")
    root = Path(artifact_dir)
    path = root / basename
    if (
        root.is_symlink()
        or path.is_symlink()
        or not path.is_file()
        or _hash(path) != record["tensor_file_sha256"]
    ):
        raise ValueError("diagnostic payload hash/path differs")
    payload = import_module("safetensors.torch").load_file(str(path), device="cpu")
    if set(payload) != {"retain_means", "directions"}:
        raise ValueError("diagnostic payload tensor inventory differs")
    retained, directions = payload["retain_means"], payload["directions"]
    if list(retained.shape) != [record["retain_rows"], record["width"]] or list(
        directions.shape
    ) != [record["width"], record["available_directions"]]:
        raise ValueError("diagnostic payload shape differs")
    invariants(retained, directions)
    return retained, directions[:, :k].contiguous()
