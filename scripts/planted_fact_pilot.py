#!/usr/bin/env python3
"""Frozen synthetic-fact pilot. Local checkpoints only; no downloads or editor model calls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Existing OS crypto dependency, after the active environment; never an installation.
if Path("/usr/lib/python3/dist-packages").is_dir():
    sys.path.append("/usr/lib/python3/dist-packages")

from lacunai.core.model import HFRuntime, active_phase, checkpoint_inventory, progress  # noqa: E402
from lacunai.core.records import RunStore  # noqa: E402
from lacunai.core.service import Service  # noqa: E402
from lacunai.core.spec import RuntimeConfig, TargetSpec, bounded_file, digest  # noqa: E402
from lacunai.erase.methods import round_to_storage, validate_replacements  # noqa: E402
from lacunai.experimental import tensors  # noqa: E402
from lacunai.experimental.artifacts import atomic_json  # noqa: E402
from lacunai.map.fingerprints import capture  # noqa: E402
from lacunai.restore.checkpoints import prepare_snapshot, write_edited_checkpoint  # noqa: E402
from lacunai.verify.evaluation import classify, compare, evaluate  # noqa: E402
from lacunai.verify.reports import write_signed_report  # noqa: E402

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_SECONDS = 3 * 60 * 60
MAX_BYTES = 10 * 1024**3
MAX_STEPS = 128
ROW_COUNT = 32
MAX_RELATIVE = 0.25
EDITS = (
    {"method": "ablate-neurons", "strength": 1.0, "max_relative_change": 0.25, "steps": 12},
    {"method": "rank-one", "strength": 1.0, "max_relative_change": 0.25, "steps": 96},
    {"method": "ablate-direction", "strength": 1.0, "max_relative_change": 0.25, "steps": 12},
)


def normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def load_training(path: Path, spec: TargetSpec) -> dict[str, Any]:
    value = json.loads(bounded_file(path))
    if set(value) != {"schema_version", "model_revision", "development", "retain_development"}:
        raise ValueError("unexpected planting data fields")
    if value["schema_version"] != 1 or value["model_revision"] != REVISION:
        raise ValueError("planting data version/revision mismatch")
    development, retain = value["development"], value["retain_development"]
    if not isinstance(development, list) or len(development) != 16:
        raise ValueError("planting requires exactly sixteen development prompts")
    if not isinstance(retain, list) or not 1 <= len(retain) <= 256:
        raise ValueError("retain development must be nonempty and bounded")
    for row in development + retain:
        if not isinstance(row, dict) or set(row) != {"prompt", "answer"}:
            raise ValueError("training rows require prompt and answer only")
        if any(not isinstance(v, str) or not v.strip() or len(v) > 8192 for v in row.values()):
            raise ValueError("training text must be bounded and nonempty")
    if any(row["answer"] != "7421" for row in development):
        raise ValueError("only the frozen synthetic access-code fact may be planted")
    if any("7421" in row["prompt"] or "7421" in row["answer"] for row in retain):
        raise ValueError("retain development contains the target answer")
    dev_prompts = {normalized(row["prompt"]) for row in development}
    retain_prompts = {normalized(row["prompt"]) for row in retain}
    withheld = {normalized(row.prompt) for row in spec.probes + spec.retain}
    withheld.update(normalized(text) for text in spec.perplexity_texts)
    if len(dev_prompts) != len(development) or len(retain_prompts) != len(retain):
        raise ValueError("duplicate development prompts")
    if dev_prompts & retain_prompts or (dev_prompts | retain_prompts) & withheld:
        raise ValueError("development and withheld data must be disjoint")
    if not {normalized(row.prompt) for row in spec.mapping} <= dev_prompts:
        raise ValueError("mapping prompts must belong to the development set")
    if spec.rewrite != "unknown" or any(row.answer != "7421" for row in spec.mapping + spec.probes):
        raise ValueError("target contract differs from frozen synthetic fact/rewrite")
    return cast(dict[str, Any], value)


class Budget:
    def __init__(self, root: Path, deadline_epoch: float | None) -> None:
        now = time.time()
        if deadline_epoch is not None and (
            not math.isfinite(deadline_epoch) or not 0 < deadline_epoch - now <= MAX_SECONDS
        ):
            raise ValueError("deadline must be future and within the frozen global runtime")
        self.root = root
        self.started = time.monotonic()
        self.deadline = None if deadline_epoch is None else self.started + (deadline_epoch - now)

    def used(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValueError("pilot artifact symlink refused")
            if path.is_file():
                total += path.stat().st_size
        return total

    def check(self, reserve: int = 0) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("global pilot runtime exhausted")
        used = self.used()
        if used + reserve > MAX_BYTES:
            raise OSError("global pilot artifact byte budget exhausted")
        if shutil.disk_usage(self.root).free < reserve + 256 * 1024**2:
            raise OSError("insufficient free disk for next bounded operation")

    def finalization(self) -> Budget:
        """Keep disk guards while allowing checkpoint restoration after a time cutoff."""
        result = Budget(self.root, None)
        result.check()
        return result

    def config(self) -> RuntimeConfig:
        self.check()
        if self.deadline is None:
            return RuntimeConfig(seed=42, torch_threads=2, max_seconds=None)
        remaining = int(self.deadline - time.monotonic())
        if remaining < 1:
            raise TimeoutError("global pilot runtime exhausted")
        return RuntimeConfig(seed=42, torch_threads=2, max_seconds=remaining)


def source_matches(runtime: HFRuntime, source: Path) -> None:
    if checkpoint_inventory(source, runtime.config.max_model_bytes) != runtime.inventory:
        raise ValueError("source changed after model loading")


def byte_copy(source: Path, destination: Path, budget: Budget) -> dict[str, str]:
    expected = checkpoint_inventory(source, 4 * 1024**3)
    size = sum((source / name).stat().st_size for name in expected)
    budget.check(size)
    destination.mkdir(parents=True, exist_ok=False)
    with active_phase("checkpoint_control_copy"):
        for index, name in enumerate(expected, 1):
            budget.check()
            output = destination / name
            output.parent.mkdir(parents=True, exist_ok=True)
            with (source / name).open("rb") as src, output.open("xb") as dst:
                for block in iter(lambda: src.read(1024**2), b""):
                    budget.check()
                    dst.write(block)
                dst.flush()
                os.fsync(dst.fileno())
            progress("checkpoint_control_copy", completed=index, total=len(expected), unit="files")
    if checkpoint_inventory(destination, 4 * 1024**3) != expected:
        raise ValueError("control copy differs from original bytes")
    return expected


def selected_rows(
    runtime: HFRuntime, data: dict[str, Any], spec: TargetSpec
) -> tuple[str, list[int]]:
    torch = import_module("torch")
    layer = runtime.layer_ids[-1]
    means = []
    for split, prompts in (
        ("target", [r["prompt"] for r in data["development"]]),
        ("background", spec.background),
    ):
        values = []
        runtime.progress(
            "plant_localization", split=split, completed=0, total=len(prompts), unit="prompts"
        )
        for index, prompt in enumerate(prompts, 1):
            values.append(capture(runtime, prompt, [layer])[layer][0, -1].clone())
            runtime.progress(
                "plant_localization",
                split=split,
                completed=index,
                total=len(prompts),
                unit="prompts",
            )
        means.append(torch.stack(values).mean(0))
    contrast = (means[0] - means[1]).abs()
    if contrast.numel() < ROW_COUNT or not bool(torch.isfinite(contrast).all()):
        raise ValueError("invalid planting feature contrast")
    rows = torch.argsort(contrast, descending=True, stable=True)[:ROW_COUNT].tolist()
    names = [name for name, module in runtime.model.named_modules() if module is runtime.mlp(layer)]
    if len(names) != 1:
        raise ValueError("planting MLP identity is ambiguous")
    return f"{names[0]}.up_proj.weight", rows


def development_recovery(runtime: HFRuntime, rows: list[dict[str, str]]) -> dict[str, Any]:
    observations = []
    for index, row in enumerate(rows, 1):
        generation = runtime.generate(row["prompt"])
        observations.append(
            {
                **row,
                "generation": generation,
                **classify(generation["output"], row["answer"], generation["truncated"]),
            }
        )
        runtime.progress("development_recovery", completed=index, total=len(rows), unit="prompts")
    return {
        "rows": observations,
        "all_recovered": all(row["classification"] == "recovered" for row in observations),
    }


def train_rows(
    runtime: HFRuntime,
    name: str,
    rows: list[int],
    data: dict[str, Any],
    budget: Budget,
    log_dir: Path,
    *,
    retain_only: bool,
    steps: int,
) -> tuple[Any, int]:
    """One frozen SGD attempt; every unselected row is restored after every step."""
    torch = import_module("torch")
    parameter = dict(runtime.model.named_parameters())[name]
    original = parameter.detach().clone()
    if parameter.dtype != torch.float32 or parameter.device.type != "cpu":
        raise ValueError("planting requires CPU float32 tensors")
    for value in runtime.model.parameters():
        value.requires_grad_(False)
    parameter.requires_grad_(True)
    mask = torch.zeros_like(parameter)
    mask[rows, :] = 1
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    trace = []
    completed = 0
    runtime.progress(
        "retain_reference_training" if retain_only else "planting",
        completed=0,
        total=steps,
        unit="steps",
    )
    try:
        for step in range(steps):
            budget.check()
            runtime.check()
            primary = data["retain_development"] if retain_only else data["development"]
            first = primary[step % len(primary)]
            retain = data["retain_development"][
                (step + int(retain_only)) % len(data["retain_development"])
            ]
            with active_phase("training_step"):
                optimizer.zero_grad(set_to_none=True)
                loss_values = []
                for item in (first, retain):
                    loss = runtime.loss(item["prompt"], item["answer"])
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    loss_values.append(float(loss.detach()))
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise ValueError("invalid training gradient")
                parameter.grad.mul_(mask)
                torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    parameter.copy_(original + (parameter - original) * mask)
                relative = math.sqrt(
                    tensors._square_norm(parameter - original) / tensors._square_norm(original)
                )
                if not math.isfinite(relative) or relative > MAX_RELATIVE + 1e-9:
                    raise ValueError("training update exceeds frozen relative-norm bound")
            completed = step + 1
            trace.append(
                {
                    "step": completed,
                    "primary_loss": loss_values[0],
                    "retain_loss": loss_values[1],
                    "relative_frobenius": relative,
                }
            )
            atomic_json(log_dir / "training-trace.json", trace)
            runtime.progress("training", completed=completed, total=steps, unit="steps")
            if not retain_only and completed % 16 == 0:
                observation = development_recovery(runtime, data["development"])
                atomic_json(log_dir / f"development-{completed}.json", observation)
                if observation["all_recovered"]:
                    break
        proposed = parameter.detach().clone()
    finally:
        with torch.no_grad():
            parameter.copy_(original)
        parameter.requires_grad_(False)
        parameter.grad = None
    return proposed, completed


def trained_checkpoint(
    source: Path,
    destination: Path,
    artifact: Path,
    data: dict[str, Any],
    spec: TargetSpec,
    budget: Budget,
    *,
    retain_only: bool,
    selection: tuple[str, list[int]] | None = None,
    steps: int = MAX_STEPS,
) -> dict[str, Any]:
    runtime = HFRuntime(source, budget.config())
    name, rows = selection if selection is not None else selected_rows(runtime, data, spec)
    artifact.mkdir(parents=True, exist_ok=False)
    source_matches(runtime, source)
    budget.check(2 * sum((source / p).stat().st_size for p in runtime.inventory))
    snapshot = prepare_snapshot(source, artifact, [name], max_bytes=MAX_BYTES)
    if {
        key: value["sha256"] for key, value in snapshot["before_files"].items()
    } != runtime.inventory:
        raise ValueError("snapshot differs from loaded training source")
    proposed, completed = train_rows(
        runtime, name, rows, data, budget, artifact, retain_only=retain_only, steps=steps
    )
    original = runtime.model.state_dict()[name]
    replacements = round_to_storage({name: proposed}, runtime.storage_dtypes)
    if bool(import_module("torch").equal(original, replacements[name])):
        if not retain_only:
            raise ValueError("planting rounded to a no-op")
        byte_copy(source, destination, budget)
        diagnostics = {"no_op": True, "reason": "retain-only update rounded to source bytes"}
    else:
        diagnostics = validate_replacements({name: original}, replacements, MAX_RELATIVE)
        budget.check(2 * sum((source / p).stat().st_size for p in runtime.inventory))
        write_edited_checkpoint(source, destination, replacements, snapshot)
    result = {
        "tensor": name,
        "rows": rows,
        "steps": completed,
        "diagnostics": diagnostics,
        "snapshot_id": snapshot["snapshot_id"],
        "source_fingerprint": runtime.fingerprint,
        "output": str(destination),
    }
    atomic_json(artifact / "result.json", result)
    del runtime, proposed, replacements, original
    gc.collect()
    return result


def generation_identity(evaluation: dict[str, Any]) -> list[Any]:
    return [
        row["generation"] for section in ("target", "retain") for row in evaluation[section]["rows"]
    ]


def score_checkpoint(path: Path, spec: TargetSpec, budget: Budget, output: Path) -> dict[str, Any]:
    runtime = HFRuntime(path, budget.config())
    result = evaluate(runtime, spec)
    atomic_json(output, result)
    del runtime
    gc.collect()
    return result


def persist_afterimages(
    output: Path, snapshot: dict[str, Any], destination: Path, budget: Budget
) -> dict[str, Any]:
    """Preserve exact source-precision touched tensors before any restoration."""
    limit = 128 * 1024**2
    artifact = destination / "edited-tensors.safetensors"
    receipt_path = destination / "edited-tensors.json"
    if artifact.exists() or receipt_path.exists():
        raise ValueError("edited after-images already exist; overwrite refused")
    if snapshot.get("output") != str(output) or snapshot.get("status") != "edited":
        raise ValueError("only complete edited checkpoint snapshots have after-images")
    descriptors = snapshot.get("tensors")
    if not isinstance(descriptors, dict) or not 1 <= len(descriptors) <= 8:
        raise ValueError("after-image tensor set must be bounded")
    total = 0
    for name, row in descriptors.items():
        if not isinstance(name, str) or not name or len(name) > 1024:
            raise ValueError("invalid after-image tensor name")
        if type(row.get("length")) is not int or row["length"] < 1:
            raise ValueError("invalid after-image tensor length")
        total += row["length"]
        relative = Path(row["file"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
            raise ValueError("invalid after-image checkpoint path")
        if row.get("dtype") not in {"F32", "F16", "BF16"}:
            raise ValueError("unsupported after-image dtype")
        if not isinstance(row.get("shape"), list) or any(
            type(dim) is not int or dim <= 0 for dim in row["shape"]
        ):
            raise ValueError("invalid after-image shape")
        if not isinstance(row.get("before_sha256"), str) or len(row["before_sha256"]) != 64:
            raise ValueError("invalid before-image hash")
    if total + 1024**2 > limit:
        raise ValueError("edited after-image payload exceeds bounded artifact size")
    budget.check(total + 1024**2)
    inventory = checkpoint_inventory(output, 4 * 1024**3)
    if inventory != {name: row["sha256"] for name, row in snapshot["after_files"].items()}:
        raise ValueError("candidate checkpoint differs from recorded after-image hashes")
    torch = import_module("torch")
    safe = import_module("safetensors")
    payload, records = {}, {}
    with active_phase("candidate_afterimages"):
        for name, row in descriptors.items():
            with safe.safe_open(output / row["file"], framework="pt", device="cpu") as stored:
                value = stored.get_tensor(name).contiguous()
                dtype = stored.get_slice(name).get_dtype()
            raw = value.view(torch.uint8).numpy().tobytes()
            if (
                dtype != row["dtype"]
                or list(value.shape) != row["shape"]
                or len(raw) != row["length"]
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("stored after-image descriptor or finite-value check failed")
            payload[name] = value
            records[name] = {
                "dtype": dtype,
                "shape": list(value.shape),
                "bytes": len(raw),
                "after_raw_sha256": hashlib.sha256(raw).hexdigest(),
                "before_raw_sha256": row["before_sha256"],
                "before_image_blob": row["blob"],
                "source_file": row["file"],
            }
        encoded = import_module("safetensors.torch").save(
            payload,
            metadata={
                "snapshot_id": str(snapshot["snapshot_id"]),
                "purpose": "exact touched-tensor after-images",
            },
        )
        if len(encoded) > limit:
            raise ValueError("encoded after-image exceeds bounded artifact size")
        destination.mkdir(parents=True, exist_ok=True)
        with artifact.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        with safe.safe_open(artifact, framework="pt", device="cpu") as stored:
            if set(stored.keys()) != set(records):
                raise ValueError("after-image roundtrip tensor names differ")
            for name, row in records.items():
                value = stored.get_tensor(name).contiguous()
                raw = value.view(torch.uint8).numpy().tobytes()
                if (
                    stored.get_slice(name).get_dtype() != row["dtype"]
                    or list(value.shape) != row["shape"]
                    or hashlib.sha256(raw).hexdigest() != row["after_raw_sha256"]
                ):
                    raise ValueError("after-image tensor roundtrip differs")
        receipt = {
            "schema_version": 1,
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_directory": snapshot["run_dir"],
            "tensors": records,
            "artifact": artifact.name,
            "artifact_bytes": artifact.stat().st_size,
            "artifact_sha256": tensors.file_hash(artifact),
            "checkpoint_after_files": snapshot["after_files"],
            "interpretation": "Sparse replacement diff; before-images remain in snapshot. "
            "Apply saved tensors to retained source to reconstruct candidate.",
        }
        atomic_json(receipt_path, receipt)
        return receipt


def benign_relearning(
    path: Path,
    names: list[str],
    data: dict[str, Any],
    spec: TargetSpec,
    budget: Budget,
    output: Path,
) -> dict[str, Any]:
    torch = import_module("torch")
    runtime = HFRuntime(path, budget.config())
    parameters = dict(runtime.model.named_parameters())
    if not names or any(name not in parameters for name in names):
        raise ValueError("benign relearning missing actual touched parameters")
    originals = {name: parameters[name].detach().clone() for name in names}
    for name in names:
        parameters[name].requires_grad_(True)
    optimizer = torch.optim.SGD([parameters[name] for name in names], lr=0.05)
    trace = []
    try:
        for step in range(8):
            budget.check()
            item = data["retain_development"][step % len(data["retain_development"])]
            with active_phase("benign_relearning"):
                optimizer.zero_grad(set_to_none=True)
                loss = runtime.loss(item["prompt"], item["answer"])
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite benign-relearning loss")
                loss.backward()
                for name in names:
                    grad = parameters[name].grad
                    if grad is None or not bool(torch.isfinite(grad).all()):
                        raise ValueError("invalid benign-relearning gradient")
                torch.nn.utils.clip_grad_norm_(
                    [parameters[name] for name in names], 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                for name in names:
                    rel = math.sqrt(
                        tensors._square_norm(parameters[name] - originals[name])
                        / tensors._square_norm(originals[name])
                    )
                    if rel > MAX_RELATIVE + 1e-9:
                        raise ValueError("benign relearning exceeds relative-norm bound")
                trace.append({"step": step + 1, "loss": float(loss.detach())})
                atomic_json(output.with_name("benign-relearning-trace.json"), trace)
            runtime.progress("benign_relearning", completed=step + 1, total=8, unit="steps")
        for name in names:
            parameters[name].requires_grad_(False)
        result = {
            "trace": trace,
            "evaluation": evaluate(runtime, spec),
            "scope": "eight retain-only SGD steps in memory; checkpoint unchanged",
        }
        atomic_json(output, result)
        return result
    finally:
        with torch.no_grad():
            for name in names:
                parameters[name].copy_(originals[name])
                parameters[name].requires_grad_(False)
                parameters[name].grad = None
        del runtime, parameters, originals, optimizer
        gc.collect()


def _sign_aggregate(
    workspace: Path, aggregate: tuple[RunStore, str, Path], summary: dict[str, Any]
) -> dict[str, Any]:
    """Sign JSON evidence and byte hashes; checkpoint payloads are never duplicated here."""
    store, run_id, directory = aggregate
    inventory = {}
    copies = []
    with active_phase("aggregate_attestation"):
        for path in sorted(workspace.rglob("*")):
            if path.is_symlink():
                raise ValueError("linked aggregate evidence refused")
            relative = path.relative_to(workspace)
            if (
                not path.is_file()
                or relative.parts[0] == "aggregate"
                or "signing" in relative.parts
            ):
                continue
            if "locks" in relative.parts or relative == Path("summary.json"):
                continue
            inventory[str(relative)] = {
                "size": path.stat().st_size,
                "sha256": tensors.file_hash(path),
            }
            if path.suffix == ".json":
                copies.append((path, relative))
        copy_bytes = sum(path.stat().st_size for path, _ in copies)
        existing_bytes = sum(path.stat().st_size for path in workspace.rglob("*") if path.is_file())
        if existing_bytes + copy_bytes + 1024**2 > MAX_BYTES:
            raise OSError("aggregate evidence exceeds artifact budget")
        for source, relative in copies:
            destination = directory / "evidence" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024**2)
            if tensors.file_hash(destination) != inventory[str(relative)]["sha256"]:
                raise ValueError("aggregate evidence copy differs")
        atomic_json(directory / "artifact-inventory.json", inventory)
        # The signature covers this frozen summary; the outer summary additionally points here.
        atomic_json(directory / "scientific-summary.json", summary)
        status = "complete" if summary["status"] == "complete" else "failed"
        record = {
            **store.get(run_id),
            "status": status,
            "verdict": summary["verdict"],
            "scientific_summary": summary,
            "claim": "Synthetic pilot measurements; reduced recoverability under these tests only",
        }
        signed = write_signed_report(directory, record, workspace / "aggregate" / "signing")
        store.update(
            run_id,
            status=status,
            verdict=summary["verdict"],
            scientific_summary=summary,
            claim=record["claim"],
        )
        return {"run_id": run_id, **signed}


def sign_aggregate(
    workspace: Path, aggregate: tuple[RunStore, str, Path], summary: dict[str, Any]
) -> dict[str, Any]:
    try:
        return _sign_aggregate(workspace, aggregate, summary)
    except Exception as exc:
        store, run_id, _ = aggregate
        failure = {
            "run_id": run_id,
            "status": "failed",
            "verdict": "inconclusive",
            "error": f"{type(exc).__name__}: {exc}",
            "claim": "aggregate attestation failed; partial report artifacts preserved",
        }
        if store.get(run_id)["status"] == "running":
            store.update(
                run_id, **{key: value for key, value in failure.items() if key != "run_id"}
            )
        sidecars = workspace / "aggregate" / "attestation-failures"
        sidecars.mkdir(mode=0o700, exist_ok=True)
        atomic_json(sidecars / f"{run_id}.json", failure)
        raise


def run(
    source: Path,
    workspace: Path,
    target: Path,
    plant_data: Path,
    source_manifest: Path,
    deadline_epoch: float | None,
    *,
    apply: bool = False,
    plant_max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    if type(plant_max_steps) is not int or plant_max_steps not in (128, 512):
        raise ValueError("plant_max_steps must be exactly 128 or 512")
    if not apply:
        return {
            "status": "planned",
            "verdict": "inconclusive",
            "dry_run": True,
            "plant_max_steps": plant_max_steps,
            "candidates": list(EDITS),
            "claim": "plan only; pass --apply for numerical execution",
        }
    for path in (source, workspace, target, plant_data, source_manifest):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("linked pilot path refused")
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("pilot workspace must be new or empty")
    if workspace == source or workspace.is_relative_to(source) or source.is_relative_to(workspace):
        raise ValueError("pilot workspace and original source must be separate")
    workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    budget = Budget(workspace, deadline_epoch)
    summary: dict[str, Any] = {"status": "running", "verdict": "inconclusive", "candidates": []}
    atomic_json(workspace / "summary.json", summary)
    aggregate: tuple[RunStore, str, Path] | None = None
    try:
        spec = TargetSpec.from_path(target)
        aggregate_store = RunStore(workspace / "aggregate")
        aggregate_id, aggregate_dir = aggregate_store.create(
            spec, "planted-pilot", {"source": str(source), "deadline_epoch": deadline_epoch}
        )
        aggregate = (aggregate_store, aggregate_id, aggregate_dir)
        data = load_training(plant_data, spec)
        manifest = json.loads(bounded_file(source_manifest))
        source_inventory = checkpoint_inventory(source, 4 * 1024**3)
        if manifest.get("model_revision") != REVISION or manifest.get("files") != source_inventory:
            raise ValueError("source bytes differ from pinned revision manifest")
        atomic_json(workspace / "target.json", asdict(spec))
        atomic_json(workspace / "plant-data.json", data)
        versions = {}
        for package in ("torch", "transformers", "safetensors", "numpy", "cryptography"):
            versions[package] = importlib.metadata.version(package)
        code_root = Path(__file__).resolve().parents[1]
        code_files = [
            Path(__file__).resolve(),
            *sorted((code_root / "src" / "lacunai").rglob("*.py")),
        ]
        atomic_json(
            workspace / "manifest.json",
            {
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "model_revision": REVISION,
                "tokenizer_revision": REVISION,
                "source_inventory": source_inventory,
                "dataset_provenance": "operator-supplied synthetic Station Amber access code 7421",
                "target_sha256": spec.digest,
                "plant_data_sha256": digest(data),
                "runtime": asdict(budget.config()),
                "versions": versions,
                "code_hashes": {
                    str(p.relative_to(code_root)): tensors.file_hash(p) for p in code_files
                },
                "limits": {
                    "seconds": MAX_SECONDS if deadline_epoch is not None else None,
                    "deadline_epoch": deadline_epoch,
                    "artifact_bytes": MAX_BYTES,
                    "candidate_attempts": len(EDITS),
                },
                "planting": {
                    "max_steps": plant_max_steps,
                    "rows": ROW_COUNT,
                    "learning_rate": 0.05,
                    "gradient_clip": 1.0,
                    "retain_loss_weight": 1.0,
                    "max_relative": MAX_RELATIVE,
                    "early_stop": (
                        "all sixteen development prompts recovered at sixteen-step boundaries"
                    ),
                    "selection": (
                        "last MLP; largest absolute final-prefill "
                        "target-minus-background feature mean"
                    ),
                },
                "edit_specifications": EDITS,
                "method_roles": {
                    "ablate-neurons": "fixed simple-ablation baseline",
                    "rank-one": "assistant-selected weight intervention",
                    "ablate-direction": "assistant-selected weight intervention",
                },
                "rewrite": "unknown",
                "relearning": {
                    "steps": 8,
                    "learning_rate": 0.05,
                    "gradient_clip": 1.0,
                    "max_relative": MAX_RELATIVE,
                    "data": "retain development only",
                },
                "reproducibility": (
                    "CPU float32 deterministic algorithms; library/CPU changes may alter rounding"
                ),
                "limitations": (
                    "Artificial localized planting; lexical recovery tests do not establish "
                    "absence. Originals and snapshots retain information."
                ),
                "transmission": "none; local models and synthetic artifacts only",
            },
        )
        score_checkpoint(source, spec, budget, workspace / "untouched.json")
        planted = workspace / "checkpoints" / "planted"
        planting = trained_checkpoint(
            source,
            planted,
            workspace / "planting",
            data,
            spec,
            budget,
            retain_only=False,
            steps=plant_max_steps,
        )
        summary["planting"] = planting
        atomic_json(workspace / "summary.json", summary)
        reference = workspace / "checkpoints" / "retain-only"
        trained_checkpoint(
            source,
            reference,
            workspace / "retain-reference",
            data,
            spec,
            budget,
            retain_only=True,
            selection=(planting["tensor"], planting["rows"]),
            steps=planting["steps"],
        )
        score_checkpoint(reference, spec, budget, workspace / "retain-only.json")
        baseline = score_checkpoint(planted, spec, budget, workspace / "planted-baseline.json")
        Service._qualify(baseline, spec)
        noop = workspace / "checkpoints" / "noop"
        planted_inventory = byte_copy(planted, noop, budget)
        noop_result = score_checkpoint(noop, spec, budget, workspace / "noop.json")
        if generation_identity(noop_result) != generation_identity(baseline):
            raise ValueError("byte-identical no-op generation differs from baseline")
        service = Service(workspace / "service")
        for edit in EDITS:
            budget.check()
            method = str(edit["method"])
            record: dict[str, Any] = {"method": method, "status": "running"}
            summary["candidates"].append(record)
            atomic_json(workspace / "summary.json", summary)
            output = workspace / "checkpoints" / method
            candidate_dir = workspace / method
            candidate_dir.mkdir()
            result = None
            existing_runs = {row["run_id"] for row in service.store.list()}
            try:
                # Full output plus one staged checkpoint file, with an explicit metadata reserve.
                budget.check(
                    2 * sum((planted / p).stat().st_size for p in planted_inventory) + 64 * 1024**2
                )
                result = service.erase(
                    planted,
                    workspace / "target.json",
                    output=output,
                    apply=True,
                    config=budget.config(),
                    **edit,
                )
                atomic_json(candidate_dir / "erase.json", result)
                snapshot = json.loads(
                    bounded_file(service.store.path(result["run_id"]) / "snapshot.json")
                )
                record["afterimages"] = persist_afterimages(output, snapshot, candidate_dir, budget)
                verification = service.verify(result["run_id"], budget.config())
                atomic_json(candidate_dir / "verify.json", verification)
                benign = benign_relearning(
                    output,
                    list(snapshot["tensors"]),
                    data,
                    spec,
                    budget,
                    candidate_dir / "benign-relearning.json",
                )
                benign_comparison = compare(baseline, benign["evaluation"], spec)
                atomic_json(candidate_dir / "benign-comparison.json", benign_comparison)
                record.update(
                    status="evaluated",
                    benign_verdict=benign_comparison["verdict"],
                    verdict=result["verdict"],
                    independent_verdict=verification["verdict"],
                    benign_recovery=benign["evaluation"]["target"]["recovery_rate"],
                    run_id=result["run_id"],
                )
            except Exception as exc:
                record.update(
                    status="failed", error=f"{type(exc).__name__}: {exc}", verdict="inconclusive"
                )
                progress("pilot_candidate", state="failed", method=method, error=str(exc))
                if isinstance(exc, (TimeoutError, OSError, MemoryError)):
                    raise
            finally:
                if result is None:
                    created = [
                        row
                        for row in service.store.list()
                        if row["run_id"] not in existing_runs and row["operation"] == "erase"
                    ]
                    if len(created) == 1:
                        receipt = service.store.path(created[0]["run_id"]) / "snapshot.json"
                        if receipt.is_file():
                            snapshot = json.loads(bounded_file(receipt))
                            if snapshot.get("output") == str(output):
                                result = created[0]
                                record["run_id"] = result["run_id"]
                                record["recovered_failed_run_for_restore"] = True
                if result is not None:
                    try:
                        finalization_budget = budget.finalization()
                        if "afterimages" not in record:
                            snapshot = json.loads(
                                bounded_file(service.store.path(result["run_id"]) / "snapshot.json")
                            )
                            record["afterimages"] = persist_afterimages(
                                output, snapshot, candidate_dir, finalization_budget
                            )
                        restored = service.restore(result["run_id"], apply=True)
                        atomic_json(candidate_dir / "restore.json", restored)
                        if checkpoint_inventory(output, 4 * 1024**3) != planted_inventory:
                            raise ValueError("restored checkpoint is not bit-exact planted source")
                        record["restored_bit_exact"] = True
                        atomic_json(workspace / "summary.json", summary)
                        restored_eval = score_checkpoint(
                            output, spec, budget, candidate_dir / "restored-evaluation.json"
                        )
                        if generation_identity(restored_eval) != generation_identity(baseline):
                            raise ValueError("restored generation differs from planted baseline")
                        record["restored_bit_exact"] = True
                        record["restored_generation_exact"] = True
                    except Exception as exc:
                        record.update(status="failed", restore_error=f"{type(exc).__name__}: {exc}")
                        atomic_json(workspace / "summary.json", summary)
                        if isinstance(exc, (TimeoutError, OSError, MemoryError)):
                            raise
                atomic_json(workspace / "summary.json", summary)
        if checkpoint_inventory(source, 4 * 1024**3) != source_inventory:
            raise ValueError("untouched original checkpoint changed")
        passed = [
            row
            for row in summary["candidates"]
            if row.get("verdict") == "passed_the_stated_experimental_contract"
            and row.get("independent_verdict") == "passed_the_stated_experimental_contract"
            and row.get("restored_bit_exact")
            and row.get("restored_generation_exact")
            and row.get("benign_verdict") == "passed_the_stated_experimental_contract"
            and row.get("status") == "evaluated"
        ]
        summary.update(
            status="complete",
            verdict="passed_the_stated_experimental_contract" if len(passed) >= 2 else "failed",
            passed_methods=[row["method"] for row in passed],
            claim="reduced recoverability under these tests; not permanent erasure",
        )
    except Exception as exc:
        summary.update(
            status="failed", verdict="inconclusive", error=f"{type(exc).__name__}: {exc}"
        )
        progress("pilot", state="failed", error=str(exc))
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - budget.started, 3)
        summary["artifact_bytes"] = budget.used()
        atomic_json(workspace / "summary.json", summary)
        if aggregate is not None:
            try:
                summary["aggregate_report"] = sign_aggregate(workspace, aggregate, summary)
            except Exception as exc:
                summary.update(
                    status="failed",
                    verdict="inconclusive",
                    attestation_error=f"{type(exc).__name__}: {exc}",
                )
                progress("aggregate_attestation", state="failed", error=str(exc))
            summary["artifact_bytes"] = budget.used()
            atomic_json(workspace / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("source", "workspace", "target", "plant-data", "source-manifest"):
        parser.add_argument("--" + argument, required=True, type=Path)
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument("--deadline-epoch", type=float)
    timing.add_argument(
        "--no-deadline",
        action="store_true",
        help="explicitly disable the overall time cutoff; disk and host guards remain",
    )
    parser.add_argument("--apply", action="store_true", help="authorize the frozen numerical batch")
    parser.add_argument("--plant-max-steps", type=int, choices=(128, 512), default=MAX_STEPS)
    args = parser.parse_args()
    with active_phase("planted_fact_pilot"):
        result = run(
            args.source.absolute(),
            args.workspace.absolute(),
            args.target.absolute(),
            args.plant_data.absolute(),
            args.source_manifest.absolute(),
            args.deadline_epoch,
            apply=args.apply,
            plant_max_steps=args.plant_max_steps,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return (
        0
        if result.get("dry_run") or result["verdict"] == "passed_the_stated_experimental_contract"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
