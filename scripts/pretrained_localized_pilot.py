#!/usr/bin/env python3
"""Opt-in, signed-localization-bound pretrained cohort; no planting."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lacunai.core.model import progress  # noqa: E402
from lacunai.core.service import Service  # noqa: E402
from lacunai.core.spec import TargetSpec, bounded_file, digest  # noqa: E402
from lacunai.experimental.artifacts import atomic_json  # noqa: E402
from lacunai.verify.evaluation import compare  # noqa: E402
from lacunai.verify.reports import verify_report  # noqa: E402


def shared_api() -> Any:
    spec = importlib.util.spec_from_file_location(
        "localized_shared", Path(__file__).with_name("connected_fact_pilot.py")
    )
    if spec is None or spec.loader is None:
        raise ValueError("shared pilot missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = shared_api()
METHODS = {
    "ablate-neurons": "model.layers.21.mlp.up_proj.weight",
    "ablate-direction": "model.layers.21.mlp.down_proj.weight",
}
OUTER = "model.layers.23.mlp.up_proj.weight"


class ScopedStorage(shared.Storage):  # type: ignore[misc,name-defined]
    """Existing transactions, distinct journals, one aggregate record allowance."""

    def __init__(self, budget: Any, root: Path, scope: Path, scratch: Path, name: str) -> None:
        self.root = root
        scope.mkdir(parents=True, exist_ok=False)
        self.budget, self.workspace, self.scratch, self.name = budget, scope, scratch, name
        descriptor = shared.checkpoints._tensors(scratch, shared.inventory(scratch))[name]
        self.tensor_bytes = int(descriptor["length"])
        self.shard_bytes = (scratch / descriptor["file"]).stat().st_size
        self.stack: list[Path] = []
        self.image_pins: dict[str, tuple[str, str]] = {}
        self.reserve()

    def used(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def reserve(self) -> None:
        used = self.used()
        if used > self.budget.record_limit:
            raise OSError("localized record ledger exhausted")
        self.budget.check()
        self.budget.base.check(self.shard_bytes + self.budget.record_limit - used)


class LedgerBudget:
    def __init__(self, base: Any, workspace: Path, record_limit: int) -> None:
        self.base, self.workspace, self.record_limit = base, workspace, record_limit

    def config(self) -> Any:
        return self.base.config()

    def check(self, reserve: int = 0) -> None:
        snapshots = [
            p
            for p in self.workspace.rglob("snapshot.json")
            if "report-evidence" not in p.relative_to(self.workspace).parts
        ]
        images = [
            p
            for p in self.workspace.rglob("edited-tensors.safetensors")
            if "report-evidence" not in p.relative_to(self.workspace).parts
        ]
        if len(snapshots) > 5 or len(images) > 2:
            raise OSError("localized transaction count exceeded")
        used = sum(p.stat().st_size for p in self.workspace.rglob("*") if p.is_file())
        if used + reserve > self.record_limit:
            raise OSError("localized record ledger exhausted")
        self.base.check(reserve)


def ledger(scratch: Path) -> dict[str, Any]:
    tensors = shared.checkpoints._tensors(scratch, shared.inventory(scratch))
    counts = {OUTER: 1, **{name: 3 for name in METHODS.values()}}
    records = shared.METADATA_BUDGET + sum(
        count * (int(tensors[name]["length"]) + 65536) for name, count in counts.items()
    )
    shard = max((scratch / tensors[name]["file"]).stat().st_size for name in counts)
    if records > shared.RECORD_BUDGET or shard > 988097824:
        raise OSError("localized transaction ledger exceeds frozen storage bounds")
    return {
        "snapshots": 5,
        "afterimages": 2,
        "tensor_record_counts": counts,
        "worst_record_bytes": records,
        "staged_shard_bytes": shard,
        "record_cap_bytes": records,
        "shared_record_ceiling_bytes": shared.RECORD_BUDGET,
    }


def positive_selection(record: dict[str, Any], source: str) -> list[dict[str, Any]]:
    if record.get("checkpoint_fingerprint") != source or record.get("kind") != "localization":
        raise ValueError("localization source/kind differs")
    selected = record.get("selected")
    if selected != [{"layer": 21, "neurons": [4090, 1507, 535]}]:
        raise ValueError("frozen selected layer/neuron scope differs")
    layers = [r for r in record["layers"] if r["layer"] == 21]
    if len(layers) != 1:
        raise ValueError("localized layer evidence missing")
    layer = layers[0]
    records = [layer]
    for neuron in selected[0]["neurons"]:
        rows = [r for r in layer["neurons"] if r["neuron"] == neuron]
        if len(rows) != 1:
            raise ValueError("localized neuron evidence missing")
        records.extend(rows)
    for item in records:
        pairs = item["raw_pairs"]
        valid = []
        for pair in pairs:
            if pair.get("normalized_effect") is None:
                continue
            clean, corrupt, patched = (
                float(pair[k]) for k in ("clean_logprob", "corrupt_logprob", "patched_logprob")
            )
            if (
                not all(math.isfinite(v) for v in (clean, corrupt, patched))
                or clean - corrupt <= 1e-6
            ):
                raise ValueError("nonpositive/invalid raw localization contrast")
            effect = patched - corrupt
            if not math.isclose(effect, float(pair["effect"]), rel_tol=0, abs_tol=1e-12):
                raise ValueError("raw localization arithmetic differs")
            if not math.isclose(
                effect / (clean - corrupt),
                float(pair["normalized_effect"]),
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ValueError("normalized localization arithmetic differs")
            valid.append(effect)
        if not valid or math.fsum(valid) / len(valid) <= 0:
            raise ValueError("no positive raw localization evidence")
        if not math.isclose(
            math.fsum(valid) / len(valid), float(item["mean_effect"]), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("localization mean differs")
    return cast(list[dict[str, Any]], selected)


def prepare(path: Path, pin: str, *, recovery: bool = False) -> tuple[Any, ...]:
    progress("localized_preparation", state="active")
    if shared.sha(path) != pin:
        raise ValueError("plan hash differs")
    plan = json.loads(bounded_file(path))
    paths = {"code", "source", "scratch", "workspace", "budget_root", "key_dir"}
    pins = {
        "bundle",
        "source_inventory",
        "scratch_inventory",
        "qualification",
        "final",
        "training",
        "diagnosis",
        "diagnosis_report",
        "trusted_public_key",
    }
    if (
        set(plan) != paths | pins | {"schema_version", "profile"}
        or type(plan["schema_version"]) is not int
        or plan["schema_version"] != 1
        or plan["profile"] != "pretrained-localized"
    ):
        raise ValueError("localized plan schema differs")
    for key in paths:
        p = Path(plan[key])
        if not p.is_absolute() or p.resolve() != p or any(x.is_symlink() for x in (p, *p.parents)):
            raise ValueError("unsafe plan path")
    code = Path(plan["code"])
    if code != Path(__file__).resolve().parents[1]:
        raise ValueError("executed code differs")
    values = {}
    for key in pins:
        item = plan[key]
        if set(item) != {"path", "sha256"} or shared.sha(Path(item["path"])) != item["sha256"]:
            raise ValueError("input pin differs: " + key)
        if key != "trusted_public_key":
            values[key] = json.loads(bounded_file(Path(item["path"])))
    if Path(plan["bundle"]["path"]) != code / "hashes.json":
        raise ValueError("bundle location differs")
    for relative, expected in values["bundle"].items():
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or shared.sha(code / relative) != expected
        ):
            raise ValueError("code bundle differs")
    if "scripts/pretrained_localized_pilot.py" not in values["bundle"]:
        raise ValueError("runner not pinned")
    for key in ("source", "scratch"):
        if (
            not (recovery and key == "scratch")
            and shared.inventory(Path(plan[key])) != values[key + "_inventory"]
        ):
            raise ValueError("checkpoint inventory differs")
    for key in ("workspace", "key_dir"):
        output = Path(plan[key])
        if output.exists() and not recovery:
            raise ValueError("exclusive output required")
        for other in paths - {key, "budget_root"}:
            protected = Path(plan[other])
            if output == protected or output in protected.parents or protected in output.parents:
                raise ValueError("output overlaps protected path")
    report_dir = Path(plan["diagnosis_report"]["path"]).parent
    verify_report(report_dir, Path(plan["trusted_public_key"]["path"]))
    envelope = json.loads(bounded_file(report_dir / "signature.json"))["envelope"]["files_sha256"]
    diagnosis_path = Path(plan["diagnosis"]["path"])
    if envelope.get(str(diagnosis_path.relative_to(report_dir))) != plan["diagnosis"]["sha256"]:
        raise ValueError("diagnosis not included in signed evidence")
    positive_selection(values["diagnosis"], digest(values["source_inventory"]))
    q, f = (TargetSpec.from_dict(values[k]) for k in ("qualification", "final"))
    if (
        q.layers != [21]
        or f.layers != [21]
        or len(q.probes) != 8
        or len(f.probes) != 16
        or len(q.retain) != 8
        or len(f.retain) != 8
    ):
        raise ValueError("localized evaluation scope differs")
    diagnosis_target_path = report_dir / "target.json"
    if envelope.get("target.json") != shared.sha(diagnosis_target_path):
        raise ValueError("diagnosis target is not signed")
    diagnosis_target = TargetSpec.from_dict(json.loads(bounded_file(diagnosis_target_path)))
    if (
        values["diagnosis"].get("target_id") != diagnosis_target.target_id
        or values["diagnosis"].get("spec_sha256") != diagnosis_target.digest
    ):
        raise ValueError("diagnosis target identity differs")
    if diagnosis_target.layers != list(range(24)):
        raise ValueError("diagnosis must cover all original 24 layers")
    for field in ("mapping", "top_n", "patch_neurons", "background"):
        if getattr(diagnosis_target, field) != getattr(q, field):
            raise ValueError("diagnosis mapping contract differs: " + field)
    shared.pretrained_api().validate_data(
        values["training"],
        replace(q, layers=[23]),
        replace(f, layers=[23]),
        shared.pilot_api().REVISION,
    )
    if not recovery:
        shared.compatible_scratch(Path(plan["source"]), Path(plan["scratch"]), OUTER)
    ledger(Path(plan["scratch"]))
    progress("localized_preparation", state="complete", model_loaded=False)
    return plan, q, f, values["training"], values["diagnosis"]


class BoundService(Service):
    def __init__(self, workspace: Path, diagnosis: dict[str, Any], source: str) -> None:
        super().__init__(workspace)
        self.expected = positive_selection(diagnosis, source)
        self.source_pin = source

    def _mapping(self, runtime: Any, spec: Any, causal: bool = True) -> Any:
        fp, localization = super()._mapping(runtime, spec, causal=causal)
        if (
            localization is None
            or positive_selection(localization, self.source_pin) != self.expected
        ):
            raise ValueError("runtime localization differs from frozen positive evidence")
        return fp, localization


def execute(
    plan: dict[str, Any], qualification: Any, final: Any, data: Any, diagnosis: Any
) -> dict[str, Any]:
    source, scratch, workspace = (Path(plan[k]) for k in ("source", "scratch", "workspace"))
    workspace.mkdir(parents=True, exist_ok=False)
    pilot = shared.pilot_api()
    budget = LedgerBudget(
        pilot.Budget(Path(plan["budget_root"]), None),
        workspace,
        ledger(scratch)["worst_record_bytes"],
    )
    identity = shared.compatible_scratch(source, scratch, OUTER)
    summary: dict[str, Any] = {
        "run_id": uuid.uuid4().hex,
        "target_id": final.target_id,
        "status": "running",
        "verdict": "inconclusive",
        "candidates": [],
        "plan": plan,
        "weights_pretrained": True,
        "planting_performed": False,
        "retain_only_pretraining_reference": "absent",
        "ledger": ledger(scratch),
        "runtime": asdict(budget.config()),
        "methods": METHODS,
        "source_inventory": identity["source_inventory"],
        "old_scratch_inventory": identity["scratch_inventory"],
        "code_hashes": json.loads(bounded_file(Path(plan["bundle"]["path"]))),
        "versions": {
            name: shared.import_module("importlib.metadata").version(name)
            for name in ("torch", "transformers", "safetensors", "numpy", "accelerate")
        },
    }
    atomic_json(workspace / "manifest.json", summary)
    atomic_json(workspace / "qualification-target.json", asdict(qualification))
    error: BaseException | None = None
    progress("localized_pilot", state="active")
    try:
        outer = ScopedStorage(budget, workspace, workspace / "outer", scratch, OUTER)
        baseline = shared.score(
            source, qualification, budget, workspace / "untouched-qualification.json"
        )
        shared.pretrained_api().qualify(baseline, qualification)
        safe_open = shared.import_module("safetensors").safe_open
        with safe_open(
            source / identity["descriptor"]["file"], framework="pt", device="cpu"
        ) as opened:
            original = opened.get_tensor(OUTER).float()
        with outer.transaction("original", {OUTER: original}):
            if shared.inventory(scratch) != identity["source_inventory"]:
                raise ValueError("original materialization mismatch")
            noop = shared.score(
                scratch, qualification, budget, workspace / "noop-qualification.json"
            )
            if (
                pilot.generation_identity(noop) != pilot.generation_identity(baseline)
                or noop["perplexity"] != baseline["perplexity"]
            ):
                raise ValueError("no-op parity differs")
            service = BoundService(
                workspace / "operations", diagnosis, digest(identity["source_inventory"])
            )
            frozen = []
            for method, name in METHODS.items():
                progress("localized_candidate", state="active", method=method)
                directory = workspace / method
                directory.mkdir()
                existing = {r["run_id"] for r in service.store.list()}
                row = {"method": method, "status": "running"}
                summary["candidates"].append(row)
                atomic_json(workspace / "summary.json", summary)
                try:
                    outer.reserve()
                    result = service.erase(
                        scratch,
                        workspace / "qualification-target.json",
                        method,
                        output=scratch,
                        in_place=True,
                        apply=True,
                        strength=1.0,
                        max_relative_change=0.25,
                        config=budget.config(),
                    )
                    snapshot = json.loads(
                        (service.store.path(result["run_id"]) / "snapshot.json").read_text()
                    )
                    if set(snapshot["tensors"]) != {name}:
                        raise ValueError("candidate tensor scope differs")
                    outer.reserve()
                    row["afterimages"] = pilot.persist_afterimages(
                        scratch, snapshot, directory, budget
                    )
                    image = directory / "edited-tensors.safetensors"
                    frozen.append(
                        {
                            "method": method,
                            "path": str(image),
                            "sha256": shared.sha(image),
                            "receipt_sha256": shared.sha(image.with_suffix(".json")),
                        }
                    )
                    row.update(
                        status="frozen", run_id=result["run_id"], internal_verdict=result["verdict"]
                    )
                except BaseException as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    runs = [
                        r
                        for r in service.store.list()
                        if r["run_id"] not in existing and r["operation"] == "erase"
                    ]
                    if len(runs) > 1:
                        raise ValueError("ambiguous candidate ownership")
                    if runs:
                        receipt = service.store.path(runs[0]["run_id"]) / "snapshot.json"
                        if (
                            receipt.exists()
                            and json.loads(receipt.read_text())["status"] != "prepared"
                        ):
                            outer.reserve()
                            atomic_json(
                                directory / "restore.json",
                                service.restore(runs[0]["run_id"], apply=True),
                            )
                    if shared.inventory(scratch) != identity["source_inventory"]:
                        raise ValueError("candidate restoration differs")
                atomic_json(workspace / "summary.json", summary)
                progress("localized_candidate", state="complete", method=method)
            atomic_json(
                workspace / "candidate-freeze.json",
                {"candidates": frozen, "final_spec_sha256": final.digest},
            )
            freeze_pin = shared.sha(workspace / "candidate-freeze.json")
            summary["candidate_freeze_sha256"] = freeze_pin
            final_baseline = shared.score(source, final, budget, workspace / "untouched-final.json")
            final_qualified = True
            try:
                shared.pretrained_api().qualify(final_baseline, final)
            except ValueError as exc:
                final_qualified = False
                summary["qualification_error"] = str(exc)
            for record in frozen:
                if shared.sha(workspace / "candidate-freeze.json") != freeze_pin:
                    raise ValueError("candidate freeze changed")
                method = record["method"]
                name = METHODS[method]
                directory = workspace / method
                storage = ScopedStorage(
                    budget, workspace, directory / "final-transactions", scratch, name
                )
                replacements = shared.load_image(
                    Path(record["path"]), record["sha256"], record["receipt_sha256"]
                )
                with storage.transaction("candidate-final", replacements):
                    after = shared.score(scratch, final, budget, directory / "final.json")
                    comparison = shared.pretrained_api().alias_comparison(
                        compare(final_baseline, after, final), after, data["target_aliases"]
                    )
                    atomic_json(directory / "final-comparison.json", comparison)
                    benign = pilot.benign_relearning(
                        scratch,
                        [name],
                        {"retain_development": data["benign_retain"]},
                        final,
                        budget,
                        directory / "benign.json",
                    )
                    benign_compare = shared.pretrained_api().alias_comparison(
                        compare(final_baseline, benign["evaluation"], final),
                        benign["evaluation"],
                        data["target_aliases"],
                    )
                    atomic_json(directory / "benign-comparison.json", benign_compare)
                    row = next(r for r in summary["candidates"] if r["method"] == method)
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
            if not final_qualified:
                summary["verdict"] = "inconclusive"
            summary["status"] = "complete"
    except BaseException as exc:
        error = exc
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        progress("localized_pilot", state="failed", error=str(exc))
    finally:
        summary["source_files_unchanged"] = shared.inventory(source) == identity["source_inventory"]
        summary["old_scratch_restored_exact"] = (
            shared.inventory(scratch) == identity["scratch_inventory"]
        )
        if not summary["source_files_unchanged"] or not summary["old_scratch_restored_exact"]:
            summary.update(
                status="failed",
                verdict="inconclusive",
                integrity_error="checkpoint identity differs",
            )
        shared.finalize_evidence(workspace, summary, Path(plan["key_dir"]), budget)
        gc.collect()
    if error is not None:
        raise error
    progress("localized_pilot", state="complete", verdict=summary["verdict"])
    return summary


def recover(plan: dict[str, Any]) -> dict[str, Any]:
    """Restore only recognized active snapshots, children before the original outer scope."""
    workspace, scratch = Path(plan["workspace"]), Path(plan["scratch"])
    budget = shared.pilot_api().Budget(Path(plan["budget_root"]), None)
    scopes = [workspace / method / "final-transactions" for method in METHODS]
    scopes.append(workspace / "outer")
    pending: list[tuple[Path, Path | None]] = []
    service = []
    for receipt in (workspace / "operations" / "runs").glob("*/snapshot.json"):
        record = json.loads(bounded_file(receipt))
        if record["status"] in {"writing", "edited", "restoring"}:
            service.append(receipt.parent)
    if len(service) > 1:
        raise ValueError("ambiguous active service snapshots")
    pending.extend((path, None) for path in service)
    active_children = 0
    for scope in scopes:
        journal = scope / "transaction-journal.json"
        if not journal.exists():
            continue
        record = json.loads(bounded_file(journal))
        stack = record["stack"]
        if record.get("scratch") != str(scratch) or record.get("order") != "LIFO" or len(stack) > 1:
            raise ValueError("recovery journal scope differs")
        if scope != workspace / "outer" and stack:
            active_children += 1
        for value in reversed(stack):
            path = Path(value)
            if path.parent != scope / "transactions" or path.resolve() != path:
                raise ValueError("recovery snapshot path differs")
            pending.append((path, journal))
    if active_children + len(service) > 1:
        raise ValueError("ambiguous nested recovery order")
    results: list[dict[str, Any]] = []
    for directory, pending_journal in pending:
        record = json.loads(bounded_file(directory / "snapshot.json"))
        allowed = {OUTER, *METHODS.values()}
        if (
            record.get("source") != str(scratch)
            or record.get("output", str(scratch)) != str(scratch)
            or len(record["tensors"]) != 1
            or not set(record["tensors"]) <= allowed
        ):
            raise ValueError("recovery source/tensor scope differs")
        allowance = ledger(scratch)
        used = sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file())
        remaining = allowance["worst_record_bytes"] - used
        if remaining < 0:
            raise OSError("localized recovery record ledger exhausted")
        budget.check(allowance["staged_shard_bytes"] + remaining)
        progress(
            "localized_recovery",
            state="active",
            completed=len(results),
            total=len(pending),
            unit="snapshots",
        )
        if record["status"] == "prepared":
            if shared.inventory(scratch) != {
                k: v["sha256"] for k, v in record["before_files"].items()
            }:
                raise ValueError("prepared snapshot has unknown checkpoint bytes")
        else:
            results.append(shared.checkpoints.restore_snapshot(directory, apply=True))
        if pending_journal is not None:
            atomic_json(pending_journal, {"scratch": str(scratch), "order": "LIFO", "stack": []})
    expected = json.loads(bounded_file(Path(plan["scratch_inventory"]["path"])))
    if shared.inventory(scratch) != expected:
        raise ValueError("recovery did not restore expected original scratch")
    result = {"status": "complete", "restored": results, "old_scratch_restored_exact": True}
    atomic_json(workspace / "recovery.json", result)
    progress(
        "localized_recovery",
        state="complete",
        completed=len(pending),
        total=len(pending),
        unit="snapshots",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    try:
        values = prepare(args.plan, args.plan_sha256, recovery=args.recover)
        if not args.apply:
            print(json.dumps({"dry_run": True, "model_loaded": False}), flush=True)
            return 0
        if args.recover:
            print(json.dumps(recover(values[0])), flush=True)
            return 0
        result = execute(*values)
        print(json.dumps(result), flush=True)
        return 0 if result["verdict"] == "passed_the_stated_experimental_contract" else 2
    except Exception as exc:
        progress("localized_pilot", state="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
