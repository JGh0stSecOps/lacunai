#!/usr/bin/env python3
"""Connected synthetic pilot using durable, nested single-tensor transactions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lacunai.core.model import HFRuntime, checkpoint_inventory, progress  # noqa: E402
from lacunai.core.spec import TargetSpec, bounded_file, digest  # noqa: E402
from lacunai.experimental.artifacts import atomic_json  # noqa: E402
from lacunai.restore import checkpoints  # noqa: E402

RECORD_BUDGET = 160 * 1024**2
METADATA_BUDGET = 32 * 1024**2
MAX_SNAPSHOTS = 9
MAX_IMAGES = 6
METHODS = {"npo-kl-repair": 8, "grad-diff-kl-repair2": 4}


def pilot_api() -> Any:
    path = Path(__file__).with_name("planted_fact_pilot.py")
    spec = importlib.util.spec_from_file_location("connected_shared_pilot", path)
    if spec is None or spec.loader is None:
        raise ValueError("shared pilot unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("redirected path")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(path: Path) -> dict[str, str]:
    return checkpoint_inventory(path, 4 * 1024**3)


def slice_hash(path: Path, offset: int, length: int) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        while length:
            data = stream.read(min(length, 1024**2))
            if not data:
                raise ValueError("truncated tensor payload")
            result.update(data)
            length -= len(data)
    return result.hexdigest()


def compatible_scratch(source: Path, scratch: Path, name: str) -> dict[str, Any]:
    """Allow only one known tensor difference; never silently reset other old state."""
    if source == scratch or source in scratch.parents or scratch in source.parents:
        raise ValueError("source and scratch must be disjoint")
    before, original = inventory(scratch), inventory(source)
    left = checkpoints._tensors(source, original)
    right = checkpoints._tensors(scratch, before)
    if set(before) != set(original) or left != right or name not in left:
        raise ValueError("scratch layout differs from original")
    shard = left[name]["file"]
    if any(before[k] != original[k] for k in before if k != shard):
        raise ValueError("scratch metadata or other shard differs")
    progress(
        "scratch_non_target_verification",
        state="active",
        completed=0,
        total=len(left),
        unit="tensors",
    )
    for index, (key, row) in enumerate(left.items(), 1):
        if (
            key != name
            and row["file"] == shard
            and slice_hash(source / shard, row["offset"], row["length"])
            != slice_hash(scratch / shard, row["offset"], row["length"])
        ):
            raise ValueError("scratch non-target tensor differs")
        progress(
            "scratch_non_target_verification", completed=index, total=len(left), unit="tensors"
        )
    return {"source_inventory": original, "scratch_inventory": before, "descriptor": left[name]}


class Storage:
    """Reserve staged shard plus all remaining records before every mutating write."""

    def __init__(self, budget: Any, workspace: Path, scratch: Path, name: str) -> None:
        self.budget, self.workspace, self.scratch, self.name = budget, workspace, scratch, name
        descriptor = checkpoints._tensors(scratch, inventory(scratch))[name]
        self.tensor_bytes = int(descriptor["length"])
        self.shard_bytes = (scratch / descriptor["file"]).stat().st_size
        # zlib can grow slightly; safetensors headers are included per-record.
        self.worst_records = (MAX_SNAPSHOTS + MAX_IMAGES) * (
            self.tensor_bytes + 65536
        ) + METADATA_BUDGET
        if self.worst_records > RECORD_BUDGET:
            raise OSError("declared sparse evidence exceeds 160 MiB record budget")
        self.stack: list[Path] = []
        self.image_pins: dict[str, tuple[str, str]] = {}
        self.reserve()

    def used(self) -> int:
        return sum(p.stat().st_size for p in self.workspace.rglob("*") if p.is_file())

    def reserve(self) -> None:
        used = self.used()
        if used > RECORD_BUDGET:
            raise OSError("connected record budget exhausted")
        self.budget.check(self.shard_bytes + RECORD_BUDGET - used)

    def journal(self) -> None:
        atomic_json(
            self.workspace / "transaction-journal.json",
            {"stack": [str(p) for p in self.stack], "order": "LIFO", "scratch": str(self.scratch)},
        )

    @contextmanager
    def transaction(self, label: str, replacements: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if set(replacements) != {self.name} or not label.replace("-", "").isalnum():
            raise ValueError("transaction must select only the final up projection")
        directory = self.workspace / "transactions" / label
        if directory.exists():
            raise ValueError("transaction directory already exists")
        self.reserve()
        before = inventory(self.scratch)
        snapshot = checkpoints.prepare_snapshot(self.scratch, directory, [self.name])
        self.stack.append(directory)
        self.journal()
        primary: BaseException | None = None
        try:
            self.reserve()
            edited = checkpoints.write_edited_checkpoint(
                self.scratch, self.scratch, replacements, snapshot, in_place=True
            )
            yield edited
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                # Restoration is mandatory after scientific errors; disk guards remain.
                self.reserve()
                current = json.loads((directory / "snapshot.json").read_text())
                if current["status"] != "prepared":
                    restored = checkpoints.restore_snapshot(directory, apply=True)
                    atomic_json(directory / "restore-receipt.json", restored)
                if inventory(self.scratch) != before:
                    raise ValueError("nested transaction failed exact file restoration")
                self.stack.pop()
                self.journal()
            except BaseException as restore_error:
                atomic_json(
                    self.workspace / "restore-failure.json",
                    {
                        "error": f"{type(restore_error).__name__}: {restore_error}",
                        "primary_error": None
                        if primary is None
                        else f"{type(primary).__name__}: {primary}",
                        "journal": str(self.workspace / "transaction-journal.json"),
                    },
                )
                raise

    def save_image(self, label: str, replacements: dict[str, Any]) -> Path:
        torch = import_module("torch")
        safe = import_module("safetensors.torch")
        load_file, save_file = safe.load_file, safe.save_file

        if set(replacements) != {self.name}:
            raise ValueError("image tensor selection differs")
        path = self.workspace / (label + ".safetensors")
        if path.exists():
            raise ValueError("afterimage already exists")
        self.reserve()
        descriptor = checkpoints._tensors(self.scratch, inventory(self.scratch))[self.name]
        dtype = getattr(torch, checkpoints._FLOATS[descriptor["dtype"]])
        stored = {self.name: replacements[self.name].detach().cpu().to(dtype).contiguous()}
        if not torch.isfinite(stored[self.name]).all():
            raise ValueError("nonfinite afterimage")
        save_file(stored, str(path))
        readback = load_file(str(path))
        if set(readback) != set(stored) or not torch.equal(readback[self.name], stored[self.name]):
            raise ValueError("sparse afterimage readback differs")
        atomic_json(
            path.with_suffix(".json"),
            {
                "sha256": sha(path),
                "tensor": self.name,
                "dtype": str(dtype),
                "shape": list(stored[self.name].shape),
                "measurement": "source-storage-precision afterimage",
            },
        )
        self.image_pins[str(path)] = (sha(path), sha(path.with_suffix(".json")))
        self.reserve()
        return path

    def load_saved(self, path: Path) -> dict[str, Any]:
        payload_pin, receipt_pin = self.image_pins[str(path)]
        return load_image(path, payload_pin, receipt_pin)


def quality_gate(evaluation: dict[str, Any], untouched: dict[str, Any], spec: Any) -> bool:
    from lacunai.verify.connected import qualify_connected

    try:
        qualify_connected(evaluation, spec)
    except ValueError:
        return False
    values = [Decimal(str(x["perplexity"]["perplexity"])) for x in (evaluation, untouched)]
    if any(not x.is_finite() or x <= 0 for x in values):
        raise ValueError("invalid qualification perplexity")
    if evaluation["perplexity"]["tokens"] != untouched["perplexity"]["tokens"]:
        raise ValueError("qualification perplexity token accounting differs")
    return values[0] / values[1] <= Decimal("1.15")


def plain(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{key: row[key] for key in ("prompt", "answer")} for row in rows]


def validate_data(data: dict[str, Any], qualification: Any, final: Any) -> None:
    from dataclasses import asdict

    from lacunai.core.connected import normalized

    fields = {
        "schema_version",
        "model_revision",
        "primary",
        "utility",
        "forget",
        "retain_development",
        "benign_retain",
        "provenance",
    }
    if (
        set(data) != fields
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["model_revision"] != pilot_api().REVISION
    ):
        raise ValueError("connected training envelope differs")
    if not isinstance(data["provenance"], str) or not data["provenance"].strip():
        raise ValueError("provenance required")
    if [asdict(f) for f in qualification.facts] != [asdict(f) for f in final.facts]:
        raise ValueError("qualification/final fact inventory differs")
    for field in (
        "declared_relations",
        "rules",
        "forbidden_derived",
        "required_kinds",
        "thresholds",
        "recognition_share_tolerance",
        "mean_logprob_tolerance",
        "abstentions",
    ):
        if getattr(qualification, field) != getattr(final, field):
            raise ValueError("qualification/final shared contract differs: " + field)
    if len(qualification.capability_probes) != 8 or len(final.capability_probes) != 8:
        raise ValueError("eight capability probes required per partition")
    facts = {f.fact_id: f for f in qualification.facts}
    role_counts = {
        role: sum(f.role == role for f in qualification.facts)
        for role in ("forget", "neighbor", "unrelated")
    }
    if role_counts != {"forget": 4, "neighbor": 4, "unrelated": 4}:
        raise ValueError(
            "connected pilot requires four forget, four neighbor and four unrelated facts"
        )
    seen: set[str] = set()
    for key, count, per_fact, roles in (
        ("primary", 96, 8, {"forget", "neighbor", "unrelated"}),
        ("forget", 64, 16, {"forget"}),
        ("retain_development", 32, 4, {"neighbor", "unrelated"}),
        ("benign_retain", 32, 4, {"neighbor", "unrelated"}),
    ):
        rows = data[key]
        expected = {fid for fid, fact in facts.items() if fact.role in roles}
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError("connected training count differs: " + key)
        counts = dict.fromkeys(expected, 0)
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"fact_id", "prompt", "answer"}
                or row["fact_id"] not in expected
            ):
                raise ValueError("training fact identity differs")
            if any(
                not isinstance(row[k], str) or not row[k].strip() or len(row[k]) > 8192
                for k in ("prompt", "answer")
            ):
                raise ValueError("invalid training text")
            # Training may teach inverse forms, but must name an endpoint of its explicit fact.
            fact = facts[row["fact_id"]]
            if normalized(row["answer"]) not in {normalized(fact.subject), normalized(fact.object)}:
                raise ValueError("training answer is not a declared fact endpoint")
            counts[row["fact_id"]] += 1
            prompt = normalized(row["prompt"])
            if prompt in seen:
                raise ValueError("duplicate/cross-split training wording")
            seen.add(prompt)
        if any(n != per_fact for n in counts.values()):
            raise ValueError("unbalanced fact training")
    if not isinstance(data["utility"], list) or len(data["utility"]) != 16:
        raise ValueError("sixteen utility examples required")
    for row in data["utility"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"prompt", "answer"}
            or any(not isinstance(v, str) or not v.strip() or len(v) > 8192 for v in row.values())
        ):
            raise ValueError("invalid utility row")
        prompt = normalized(row["prompt"])
        if prompt in seen:
            raise ValueError("utility overlaps training")
        seen.add(prompt)
    groups = [
        {normalized(p.prompt) for p in [*s.probes, *s.retain, *s.capability_probes]}
        for s in (qualification, final)
    ]
    if groups[0] & groups[1] or any(seen & group for group in groups):
        raise ValueError("training/qualification/final prompts overlap")
    if any(normalized(t) in seen for s in (qualification, final) for t in s.perplexity_texts):
        raise ValueError("training overlaps perplexity corpus")


def prepare(
    plan_path: Path, pin: str, *, recovery: bool = False
) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    progress("connected_preparation", state="active")
    if sha(plan_path) != pin:
        raise ValueError("plan hash differs")
    plan = json.loads(bounded_file(plan_path))
    fields = {
        "schema_version",
        "code",
        "bundle",
        "source",
        "source_inventory",
        "scratch",
        "scratch_inventory",
        "workspace",
        "budget_root",
        "qualification",
        "final",
        "training",
        "key_dir",
    }
    if (
        set(plan) != fields
        or type(plan["schema_version"]) is not int
        or plan["schema_version"] != 1
    ):
        raise ValueError("connected runner plan schema differs")
    for key in ("code", "source", "scratch", "workspace", "budget_root", "key_dir"):
        p = Path(plan[key])
        if not p.is_absolute() or p.resolve() != p or any(x.is_symlink() for x in (p, *p.parents)):
            raise ValueError("unsafe connected path: " + key)
    code = Path(plan["code"])
    if code != Path(__file__).resolve().parents[1]:
        raise ValueError("executed code root differs")

    def read(key: str) -> Any:
        record = plan[key]
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256"}
            or sha(Path(record["path"])) != record["sha256"]
        ):
            raise ValueError("input pin differs: " + key)
        return json.loads(bounded_file(Path(record["path"])))

    hashes = read("bundle")
    if Path(plan["bundle"]["path"]) != code / "hashes.json":
        raise ValueError("bundle outside code")
    for name, expected in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha(code / relative) != expected:
            raise ValueError("frozen code differs")
    if "scripts/connected_fact_pilot.py" not in hashes:
        raise ValueError("runner absent from bundle")
    for key in ("source", "scratch"):
        if not (recovery and key == "scratch") and inventory(Path(plan[key])) != read(
            key + "_inventory"
        ):
            raise ValueError(key + " inventory differs")
    workspace = Path(plan["workspace"])
    if not recovery and (workspace.exists() or Path(plan["key_dir"]).exists()):
        raise ValueError("new output and signing directory required")
    for immutable in (Path(plan["source"]), Path(plan["scratch"]), code):
        for output in (workspace, Path(plan["key_dir"])):
            if output == immutable or immutable in output.parents or output in immutable.parents:
                raise ValueError("output overlaps protected input")
    if (
        workspace == Path(plan["key_dir"])
        or workspace in Path(plan["key_dir"]).parents
        or Path(plan["key_dir"]) in workspace.parents
    ):
        raise ValueError("signing keys must be outside output")
    qualification = TargetSpec.from_dict(read("qualification"))
    final = TargetSpec.from_dict(read("final"))
    if qualification.schema_version != 2 or final.schema_version != 2:
        raise ValueError("connected schema2 required")
    data = read("training")
    validate_data(data, qualification, final)
    progress("connected_preparation", state="complete", model_loaded=False)
    return plan, qualification, final, data


def load_image(
    path: Path, expected_sha256: str | None = None, receipt_sha256: str | None = None
) -> dict[str, Any]:
    if receipt_sha256 is not None and sha(path.with_suffix(".json")) != receipt_sha256:
        raise ValueError("afterimage receipt pin differs")
    receipt = json.loads(bounded_file(path.with_suffix(".json")))
    expected = expected_sha256 or receipt.get("sha256")
    if not isinstance(expected, str) or sha(path) != expected:
        raise ValueError("afterimage payload pin differs")
    load_file = import_module("safetensors.torch").load_file

    return {k: v.float() for k, v in load_file(str(path)).items()}


def score(
    path: Path, spec: Any, budget: Any, output: Path, reference: Any = None
) -> dict[str, Any]:
    from lacunai.verify.connected import evaluate_connected

    runtime: Any = HFRuntime(path, budget.config())
    runtime.check = budget.check
    try:
        result = evaluate_connected(runtime, spec, reference)
        atomic_json(output, result)
        return result
    finally:
        runtime = None
        gc.collect()


def execute(
    plan: dict[str, Any], qualification: Any, final: Any, data: dict[str, Any]
) -> dict[str, Any]:
    import uuid
    from dataclasses import asdict

    safe_open = import_module("safetensors").safe_open

    from lacunai.core.service import Service
    from lacunai.erase.connected_training import train_connected
    from lacunai.verify.connected import compare_connected, evaluate_connected

    pilot = pilot_api()
    source, scratch, workspace = (Path(plan[k]) for k in ("source", "scratch", "workspace"))
    workspace.mkdir(parents=True, exist_ok=False)
    budget = pilot.Budget(Path(plan["budget_root"]), None)
    config = json.loads((source / "config.json").read_text())
    name = f"model.layers.{config['num_hidden_layers'] - 1}.mlp.up_proj.weight"
    identity = compatible_scratch(source, scratch, name)
    storage = Storage(budget, workspace, scratch, name)
    summary: dict[str, Any] = {
        "run_id": uuid.uuid4().hex,
        "target_id": final.target_id,
        "status": "running",
        "verdict": "inconclusive",
        "candidates": [],
        "methods": METHODS,
        "runtime": asdict(budget.config()),
        "plan": plan,
        "source_inventory": identity["source_inventory"],
        "old_scratch_inventory": identity["scratch_inventory"],
        "code_hashes": json.loads(bounded_file(Path(plan["bundle"]["path"]))),
        "versions": {
            n: import_module("importlib.metadata").version(n)
            for n in ("torch", "transformers", "safetensors", "numpy", "accelerate")
        },
        "record_budget": RECORD_BUDGET,
        "staged_shard_bytes": storage.shard_bytes,
    }
    atomic_json(workspace / "manifest.json", summary)
    atomic_json(workspace / "qualification-target.json", asdict(qualification))
    primary_error: BaseException | None = None
    try:
        untouched = score(source, qualification, budget, workspace / "untouched-qualification.json")
        descriptor = identity["descriptor"]
        with safe_open(source / descriptor["file"], framework="pt", device="cpu") as opened:
            original_up = opened.get_tensor(name).float()
        with storage.transaction("outer-original", {name: original_up}):
            if inventory(scratch) != identity["source_inventory"]:
                raise ValueError(
                    "restored source materialization differs from untouched checkpoint"
                )
            runtime: Any = HFRuntime(scratch, budget.config())
            runtime.check = budget.check
            qualified: list[dict[str, Any]] = []

            def callback(step: int, replacements: dict[str, Any], trace: dict[str, Any]) -> bool:
                screen = generation_screen(runtime, qualification)
                atomic_json(workspace / f"qualification-screen-{step}.json", screen)
                if not screen["all_correct"]:
                    return False
                evaluation = evaluate_connected(runtime, qualification)
                evaluation["measurement"] = "in_memory_source_precision_planting_proposal"
                atomic_json(workspace / f"qualification-{step}.json", evaluation)
                passed = quality_gate(evaluation, untouched, qualification)
                if passed:
                    qualified.append(evaluation)
                return passed

            try:
                trained = train_connected(
                    runtime,
                    data["primary"],
                    data["utility"],
                    steps=512,
                    checkpoint_callback=callback,
                )
                atomic_json(workspace / "planting-trace.json", trained["diagnostics"])
                if not qualified:
                    raise ValueError("planting failed frozen learning/quality qualification")
                planted_image = storage.save_image("planted", trained["replacements"])
                steps = trained["diagnostics"]["completed_steps"]
                retained = [
                    r
                    for r in data["primary"]
                    if r["fact_id"]
                    in {f.fact_id for f in qualification.facts if f.role != "forget"}
                ]
                reference_train = train_connected(runtime, retained, data["utility"], steps=steps)
                atomic_json(
                    workspace / "retain-reference-trace.json", reference_train["diagnostics"]
                )
                reference_image = storage.save_image("retain-only", reference_train["replacements"])
                summary["planting_steps"] = steps
            finally:
                runtime = None
                gc.collect()
            with storage.transaction(
                "reference-qualification", storage.load_saved(reference_image)
            ):
                reference = score(
                    scratch, qualification, budget, workspace / "retain-only-qualification.json"
                )
            if not retain_qualified(reference, qualification):
                raise ValueError("matched reference did not learn every retained fact")
            frozen_candidates: list[tuple[str, Path]] = []
            with storage.transaction("planted", storage.load_saved(planted_image)):
                baseline = score(
                    scratch,
                    qualification,
                    budget,
                    workspace / "planted-qualification.json",
                    reference,
                )
                if not quality_gate(baseline, untouched, qualification) or generation_identity(
                    baseline
                ) != generation_identity(qualified[-1]):
                    raise ValueError("persisted planting qualification/reload parity failed")
                noop = score(
                    scratch, qualification, budget, workspace / "noop-qualification.json", reference
                )
                if generation_identity(noop) != generation_identity(baseline):
                    raise ValueError("no-op generation differs")
                gradient = {
                    "schema_version": 1,
                    "source_fingerprint": digest(inventory(scratch)),
                    "forget": plain(data["forget"]),
                    "retain_development": plain(data["retain_development"]),
                    "provenance": data["provenance"],
                }
                atomic_json(workspace / "gradient-data.json", gradient)
                service = Service(workspace / "operations")
                for method, method_steps in METHODS.items():
                    directory = workspace / method
                    directory.mkdir()
                    row: dict[str, Any] = {"method": method, "status": "running"}
                    summary["candidates"].append(row)
                    atomic_json(workspace / "summary.json", summary)
                    existing = {r["run_id"] for r in service.store.list()}
                    result = None
                    planted_inventory = inventory(scratch)
                    try:
                        storage.reserve()
                        result = service.erase(
                            scratch,
                            workspace / "qualification-target.json",
                            method,
                            output=scratch,
                            in_place=True,
                            apply=True,
                            steps=method_steps,
                            strength=1.0,
                            max_relative_change=0.25,
                            gradient_data=workspace / "gradient-data.json",
                            config=budget.config(),
                        )
                        snapshot = json.loads(
                            (service.store.path(result["run_id"]) / "snapshot.json").read_text()
                        )
                        if set(snapshot["tensors"]) != {name}:
                            raise ValueError("candidate changed tensor scope")
                        storage.reserve()
                        row["afterimages"] = pilot.persist_afterimages(
                            scratch, snapshot, directory, budget
                        )
                        frozen_candidates.append((method, directory / "edited-tensors.safetensors"))
                        row.update(
                            status="frozen",
                            run_id=result["run_id"],
                            internal_verdict=result["verdict"],
                        )
                    finally:
                        # Resolve a failed Service write only from its actual new snapshot receipt.
                        runs = [
                            r
                            for r in service.store.list()
                            if r["run_id"] not in existing and r["operation"] == "erase"
                        ]
                        if len(runs) > 1:
                            raise ValueError("ambiguous candidate transaction ownership")
                        if runs:
                            receipt = service.store.path(runs[0]["run_id"]) / "snapshot.json"
                            if receipt.exists():
                                snap = json.loads(receipt.read_text())
                                if snap.get("status") != "prepared":
                                    storage.reserve()
                                    restored = service.restore(runs[0]["run_id"], apply=True)
                                    atomic_json(directory / "restore.json", restored)
                        if inventory(scratch) != planted_inventory:
                            raise ValueError("candidate failed restoration to planted state")
                atomic_json(
                    workspace / "candidate-freeze.json",
                    {
                        "candidates": [
                            {
                                "method": m,
                                "path": str(p),
                                "sha256": sha(p),
                                "receipt_sha256": sha(p.with_suffix(".json")),
                            }
                            for m, p in frozen_candidates
                        ],
                        "final_spec_sha256": final.digest,
                    },
                )
                freeze_pin = sha(workspace / "candidate-freeze.json")
                summary["candidate_freeze_sha256"] = freeze_pin
            # All candidate bytes are frozen before the first final probe is evaluated.
            with storage.transaction("reference-final", storage.load_saved(reference_image)):
                final_reference = score(
                    scratch, final, budget, workspace / "retain-only-final.json"
                )
            with storage.transaction("planted-final", storage.load_saved(planted_image)):
                final_baseline = score(
                    scratch, final, budget, workspace / "planted-final.json", final_reference
                )
            final_baseline_qualified = generation_qualified(final_baseline)
            final_reference_qualified = retain_qualified(final_reference, final)
            score(source, final, budget, workspace / "untouched-final.json")
            if sha(workspace / "candidate-freeze.json") != freeze_pin:
                raise ValueError("candidate freeze manifest changed")
            freeze = json.loads(bounded_file(workspace / "candidate-freeze.json"))
            for method, image in frozen_candidates:
                row = next(r for r in summary["candidates"] if r["method"] == method)
                with storage.transaction(
                    method + "-final",
                    load_image(
                        image,
                        next(r["sha256"] for r in freeze["candidates"] if r["method"] == method),
                        next(
                            r["receipt_sha256"]
                            for r in freeze["candidates"]
                            if r["method"] == method
                        ),
                    ),
                ):
                    after = score(
                        scratch, final, budget, workspace / method / "final.json", final_reference
                    )
                    comparison = compare_connected(final_baseline, after, final)
                    atomic_json(workspace / method / "final-comparison.json", comparison)
                    benign = pilot.benign_relearning(
                        scratch,
                        [name],
                        {"retain_development": plain(data["benign_retain"])},
                        final,
                        budget,
                        workspace / method / "benign.json",
                    )
                    benign["evaluation"]["reference"] = final_reference
                    benign_compare = compare_connected(final_baseline, benign["evaluation"], final)
                    atomic_json(workspace / method / "benign-comparison.json", benign_compare)
                    row.update(
                        status="evaluated",
                        verdict=comparison["verdict"],
                        benign_verdict=benign_compare["verdict"],
                    )
            verdicts = [r[k] for r in summary["candidates"] for k in ("verdict", "benign_verdict")]
            summary["verdict"] = (
                "failed"
                if "failed" in verdicts
                else "passed_the_stated_experimental_contract"
                if len(verdicts) == 4
                and all(v == "passed_the_stated_experimental_contract" for v in verdicts)
                else "inconclusive"
            )
            if not final_baseline_qualified or not final_reference_qualified:
                summary["verdict"] = "inconclusive"
                summary["qualification_error"] = "final planted/reference knowledge incomplete"
            summary["status"] = "complete"
    except BaseException as exc:
        primary_error = exc
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        progress("connected_pilot", state="failed", error=str(exc))
    finally:
        summary["source_files_unchanged"] = inventory(source) == identity["source_inventory"]
        summary["old_scratch_restored_exact"] = inventory(scratch) == identity["scratch_inventory"]
        if not summary["source_files_unchanged"] or not summary["old_scratch_restored_exact"]:
            summary.update(
                status="failed",
                verdict="inconclusive",
                integrity_error="final source/scratch identity differs",
            )
        finalize_evidence(workspace, summary, Path(plan["key_dir"]), budget)
    if primary_error is not None:
        raise primary_error
    return summary


def finalize_evidence(workspace: Path, summary: dict[str, Any], key_dir: Path, budget: Any) -> None:
    """Emergency terminal metadata uses reserved bytes even when resource checks fail."""
    try:
        budget.check()
    except Exception as exc:
        summary.update(
            status="failed",
            verdict="inconclusive",
            finalization_error=f"{type(exc).__name__}: {exc}",
        )
        atomic_json(workspace / "summary.json", summary)
        atomic_json(
            workspace / "attestation-error.json",
            {"status": "failed", "error": summary["finalization_error"], "signed": False},
        )
        raise
    atomic_json(workspace / "summary.json", summary)
    try:
        sign_evidence(workspace, summary, key_dir, budget)
    except Exception as exc:
        # Do not mutate a potentially signed report; retain a separate exact error sidecar.
        atomic_json(
            workspace / "attestation-error.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "signed": False},
        )
        raise


def generation_identity(evaluation: dict[str, Any]) -> list[Any]:
    return [
        row["generation"]
        for key in ("target", "retain", "capability")
        for row in evaluation[key]["rows"]
    ]


def sign_evidence(workspace: Path, summary: dict[str, Any], key_dir: Path, budget: Any) -> None:
    """Sign copied JSON and source-byte digests, never copy nested signing keys."""
    import shutil

    from lacunai.verify.reports import write_signed_report

    destination = workspace / "report-evidence"
    if destination.exists():
        raise ValueError("signed evidence already exists")
    selected = []
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace)
        if path.is_symlink():
            raise ValueError("linked evidence")
        if not path.is_file() or "signing" in relative.parts or "locks" in relative.parts:
            continue
        if path.name == "private.key":
            raise ValueError("unexpected private key outside signing directory")
        selected.append((path, relative))
    copy_bytes = sum(p.stat().st_size for p, _ in selected if p.suffix == ".json")
    if copy_bytes > METADATA_BUDGET // 2:
        raise OSError("JSON evidence exceeds reserved signing-copy budget")
    used = sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file())
    if used + copy_bytes + 1024**2 > RECORD_BUDGET:
        raise OSError("signed records exceed connected record budget")
    budget.check(copy_bytes + 1024**2)
    destination.mkdir()
    records = {}
    for path, relative in selected:
        records[str(relative)] = {"sha256": sha(path), "bytes": path.stat().st_size}
        if path.suffix == ".json":
            output = destination / "evidence" / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as source, output.open("xb") as target:
                shutil.copyfileobj(source, target)
            if sha(output) != records[str(relative)]["sha256"]:
                raise ValueError("signed JSON copy differs")
    atomic_json(destination / "artifact-inventory.json", records)
    write_signed_report(destination, summary, key_dir)


def generation_screen(runtime: Any, spec: Any) -> dict[str, Any]:
    from lacunai.core.model import active_phase

    with active_phase("connected_generation_screen"):
        return _generation_screen(runtime, spec)


def _generation_screen(runtime: Any, spec: Any) -> dict[str, Any]:
    from dataclasses import asdict

    from lacunai.verify.connected import _valid_generation, score_response

    probes = [*spec.probes, *spec.retain]
    total = len(probes) + len(spec.capability_probes)
    rows = []
    progress("connected_generation_screen", state="active", completed=0, total=total, unit="probes")
    for index, probe in enumerate(probes):
        runtime.check()
        generation = runtime.generate(probe.prompt)
        result = score_response(
            generation.get("output"),
            probe,
            spec.abstentions,
            retain=index >= len(spec.probes),
            truncated=generation.get("truncated"),
        )
        if not _valid_generation(generation):
            result = {"classification": "missing", "response_kind": "invalid_generation_accounting"}
        rows.append({**asdict(probe), "generation": generation, **result})
        progress("connected_generation_screen", completed=index + 1, total=total, unit="probes")
    capabilities = []
    from lacunai.verify.evaluation import classify_retain

    for probe in spec.capability_probes:
        runtime.check()
        generation = runtime.generate(probe.prompt)
        result = classify_retain(
            generation.get("output"), probe.answer, generation.get("truncated")
        )
        if not _valid_generation(generation):
            result = {"classification": "missing"}
        capabilities.append({**asdict(probe), "generation": generation, **result})
        progress(
            "connected_generation_screen",
            completed=len(rows) + len(capabilities),
            total=total,
            unit="probes",
        )
    return {
        "capability": capabilities,
        "rows": rows,
        "all_correct": all(r["classification"] == "recovered" for r in capabilities)
        and bool(rows)
        and all(r["classification"] == "recovered" for r in rows),
        "per_fact": {
            f.fact_id: {
                "total": sum(r["fact_id"] == f.fact_id for r in rows),
                "recovered": sum(
                    r["fact_id"] == f.fact_id and r["classification"] == "recovered" for r in rows
                ),
            }
            for f in spec.facts
        },
    }


def generation_qualified(value: dict[str, Any]) -> bool:
    rows = (
        value["target"]["rows"]
        + value["retain"]["rows"]
        + value.get("capability", {}).get("rows", [])
    )
    return bool(rows) and all(r["classification"] == "recovered" for r in rows)


def retain_qualified(value: dict[str, Any], spec: Any) -> bool:
    rows = value["retain"]["rows"]
    return (
        all(r["classification"] == "recovered" for r in value.get("capability", {}).get("rows", []))
        and bool(rows)
        and all(
            any(r["fact_id"] == f.fact_id for r in rows) for f in spec.facts if f.role != "forget"
        )
        and all(r["classification"] == "recovered" for r in rows)
    )


def recover(plan: dict[str, Any]) -> dict[str, Any]:
    """Explicit recovery only; restore_snapshot rejects unknown file bytes itself."""
    workspace, scratch = Path(plan["workspace"]), Path(plan["scratch"])
    journal = json.loads(bounded_file(workspace / "transaction-journal.json"))
    if journal.get("scratch") != str(scratch) or journal.get("order") != "LIFO":
        raise ValueError("recovery journal scope differs")
    stack = [Path(p) for p in journal["stack"]]
    if len(stack) != len(set(stack)) or any(p.parent != workspace / "transactions" for p in stack):
        raise ValueError("invalid recovery stack")
    service = []
    for receipt in (workspace / "operations" / "runs").glob("*/snapshot.json"):
        record = json.loads(bounded_file(receipt))
        if record.get("status") in {"writing", "edited", "restoring"}:
            if record.get("output") != str(scratch):
                raise ValueError("active service snapshot scope differs")
            service.append(receipt.parent)
    if len(service) > 1:
        raise ValueError("ambiguous active service recovery")
    budget = pilot_api().Budget(Path(plan["budget_root"]), None)
    results = []
    for directory in [*service, *reversed(stack)]:
        budget.check(max(p.stat().st_size for p in scratch.glob("*.safetensors")) + METADATA_BUDGET)
        record = json.loads(bounded_file(directory / "snapshot.json"))
        if record.get("source") != str(scratch):
            raise ValueError("recovery source scope differs")
        if record["status"] == "prepared":
            if inventory(scratch) != {k: v["sha256"] for k, v in record["before_files"].items()}:
                raise ValueError("prepared snapshot current bytes differ")
        else:
            results.append(checkpoints.restore_snapshot(directory, apply=True))
        if directory in stack:
            stack.remove(directory)
            atomic_json(
                workspace / "transaction-journal.json",
                {**journal, "stack": [str(p) for p in stack]},
            )
    expected = json.loads(bounded_file(Path(plan["scratch_inventory"]["path"])))
    if inventory(scratch) != expected:
        raise ValueError("recovery failed original scratch identity")
    result = {"status": "complete", "old_scratch_restored_exact": True, "restores": results}
    atomic_json(workspace / "recovery.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true")
    group.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    progress("connected_pilot", state="active", apply=args.apply)
    prepared = None
    try:
        prepared = prepare(args.plan, args.plan_sha256, recovery=args.recover)
        if args.recover:
            recover(prepared[0])
            progress("connected_recovery", state="complete")
            return 0
        if not args.apply:
            progress("connected_pilot", state="complete", dry_run=True, model_loaded=False)
            return 0
        result = execute(*prepared)
        progress("connected_pilot", state="complete", verdict=result["verdict"])
        return 0 if result["verdict"] == "passed_the_stated_experimental_contract" else 2
    except Exception as exc:
        if prepared is not None:
            workspace = Path(prepared[0]["workspace"])
            if workspace.is_dir():
                error = {
                    "status": "failed",
                    "verdict": "inconclusive",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                atomic_json(workspace / "execution-error.json", error)
                summary_path = workspace / "summary.json"
                if not summary_path.exists():
                    atomic_json(summary_path, error)
                elif not (workspace / "report-evidence" / "signature.json").exists():
                    summary = json.loads(bounded_file(summary_path))
                    atomic_json(summary_path, {**summary, **error})
        progress("connected_pilot", state="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
