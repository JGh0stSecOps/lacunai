#!/usr/bin/env python3
"""Explicit follow-up on a qualified planted checkpoint; no planting or control training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

INPUTS = {
    "target",
    "gradient_data",
    "plant_data",
    "original_manifest",
    "original_summary",
    "planted_baseline",
    "untouched",
    "retain_only",
    "noop",
    "planted_inventory",
}
METHODS = ("npo-kl", "grad-diff-kl")
PASSED_VERDICT = "passed_the_stated_experimental_contract"
METADATA_RESERVE = 64 * 1024**2


def progress(phase: str, **fields: Any) -> None:
    print(json.dumps({"phase": phase, **fields}), flush=True)


def path(value: str) -> Path:
    result = Path(value)
    if (
        not result.is_absolute()
        or ".." in result.parts
        or any(p.is_symlink() for p in (result, *result.parents))
    ):
        raise ValueError("paths must be absolute and unredirected")
    return result


def file_hash(value: Path) -> str:
    with value.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pinned(pin: dict[str, Any]) -> Any:
    if set(pin) != {"path", "sha256"}:
        raise ValueError("pin fields differ")
    source = path(pin["path"])
    if source.stat().st_size > 8 * 1024**2 or file_hash(source) != pin["sha256"]:
        raise ValueError("pinned input bytes differ")
    return json.loads(source.read_bytes())


def api(code: Path) -> Any:
    sys.path.insert(0, str(code / "src"))
    definition = importlib.util.spec_from_file_location(
        "followup_pilot", code / "scripts/planted_fact_pilot.py"
    )
    if definition is None or definition.loader is None:
        raise ValueError("pilot module unavailable")
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def prepare(plan_path: Path, plan_sha256: str) -> tuple[dict[str, Any], dict[str, Any], Any]:
    progress("followup_preparation", state="active")
    plan = pinned({"path": str(plan_path), "sha256": plan_sha256})
    if (
        set(plan)
        != {"schema_version", "code", "bundle", "workspace", "budget_root", "planted", "inputs"}
        or type(plan["schema_version"]) is not int
        or plan["schema_version"] != 1
    ):
        raise ValueError("follow-up plan schema differs")
    code, workspace, source = (path(plan[k]) for k in ("code", "workspace", "planted"))
    path(plan["budget_root"])
    if workspace.exists():
        raise FileExistsError("follow-up workspace must not exist")
    if any(
        workspace == p or workspace in p.parents or p in workspace.parents for p in (source, code)
    ):
        raise ValueError("workspace overlaps immutable inputs")
    hashes = pinned(plan["bundle"])
    if path(plan["bundle"]["path"]) != code / "hashes.json":
        raise ValueError("bundle location differs")
    required = {
        "scripts/factoid_followup.py",
        "scripts/planted_fact_pilot.py",
        "scripts/contextual_planting.py",
    }
    actual = {
        str(p.relative_to(code))
        for folder in ("src", "scripts")
        for p in (code / folder).rglob("*.py")
    }
    if not required <= hashes.keys() or not actual <= hashes.keys():
        raise ValueError("bundle misses executable source")
    for i, (name, expected) in enumerate(sorted(hashes.items()), 1):
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or file_hash(path(str(code / relative))) != expected
        ):
            raise ValueError("code hash differs")
        progress("code_hashing", completed=i, total=len(hashes), unit="files")
    if Path(__file__).resolve() != code / "scripts/factoid_followup.py":
        raise ValueError("executing runner differs from pinned bundle")
    if set(plan["inputs"]) != INPUTS:
        raise ValueError("input inventory differs")
    records = {}
    for name, pin in plan["inputs"].items():
        source_path = path(pin["path"])
        if workspace == source_path or workspace in source_path.parents:
            raise ValueError("input overlaps new workspace")
        records[name] = pinned(pin)
    pilot = api(code)
    spec = pilot.TargetSpec.from_path(path(plan["inputs"]["target"]["path"]))
    manifest, terminal = records["original_manifest"], records["original_summary"]
    if terminal.get("status") not in {"complete", "failed"}:
        raise ValueError("original pilot must be terminal")
    planting = terminal.get("planting", {})
    if (
        not planting.get("qualified")
        or not planting.get("reload_generation_exact")
        or type(planting.get("steps")) is not int
        or planting["steps"] <= 0
    ):
        raise ValueError("original planting lacks qualified persisted evidence")
    if (
        manifest.get("model_revision") != pilot.REVISION
        or manifest.get("target_sha256") != spec.digest
    ):
        raise ValueError("model revision or evaluation contract differs")
    budget = pilot.Budget(path(plan["budget_root"]), None)
    if manifest["runtime"] != asdict(budget.config()):
        raise ValueError("runtime differs from reused controls")
    if manifest["code_hashes"].get("src/lacunai/verify/evaluation.py") != hashes.get(
        "src/lacunai/verify/evaluation.py"
    ):
        raise ValueError("scorer differs from reused controls")
    versions = {name: pilot.importlib.metadata.version(name) for name in manifest["versions"]}
    if versions != manifest["versions"]:
        raise ValueError("runtime library versions differ from reused controls")
    inventory = records["planted_inventory"]
    if pilot.checkpoint_inventory(source, 4 * 1024**3) != inventory:
        raise ValueError("actual planted checkpoint differs")
    fingerprint = pilot.digest(inventory)
    baseline = records["planted_baseline"]
    if baseline.get("checkpoint_fingerprint") != fingerprint:
        raise ValueError("baseline is not the pinned planted checkpoint")
    pilot.Service._qualify(baseline, spec)
    if pilot.generation_identity(records["noop"]) != pilot.generation_identity(baseline):
        raise ValueError("original no-op differs from baseline")
    for name in ("untouched", "retain_only", "noop", "planted_baseline"):
        if records[name].get("spec_sha256") != spec.digest:
            raise ValueError("control evaluation contract differs")
    if records["untouched"].get("checkpoint_fingerprint") != pilot.digest(
        manifest["source_inventory"]
    ):
        raise ValueError("untouched control source differs")
    from lacunai.erase.factoid_data import load_factoid_data

    data = load_factoid_data(path(plan["inputs"]["gradient_data"]["path"]), spec, fingerprint)
    training = pilot.load_training(
        path(plan["inputs"]["plant_data"]["path"]), spec, profile="contextual-kl"
    )
    if pilot.digest(training) != manifest["plant_data_sha256"]:
        raise ValueError("relearning development data differs from original plant")
    if (
        data["forget"] != training["development"]
        or data["retain_development"] != training["retain_development"]
    ):
        raise ValueError("gradient development differs from frozen planting development")
    records.update(spec=spec, development=data, training=training)
    progress("followup_preparation", state="complete", model_loaded=False)
    return plan, records, pilot


def reserve_transaction(source: Path, budget: Any) -> None:
    # Reuse the snapshot parser: reserve only the actual touched shard and tensor bytes.
    from lacunai.core.model import checkpoint_inventory
    from lacunai.restore import checkpoints

    inventory = checkpoint_inventory(source, 4 * 1024**3)
    config = json.loads((source / "config.json").read_bytes())
    name = f"model.layers.{config['num_hidden_layers'] - 1}.mlp.up_proj.weight"
    descriptor = checkpoints._tensors(source, inventory)[name]
    budget.check(
        (source / descriptor["file"]).stat().st_size + 2 * descriptor["length"] + METADATA_RESERVE
    )


def restore_candidate(
    pilot: Any,
    service: Any,
    result: dict[str, Any] | None,
    existing: set[str],
    scratch: Path,
    directory: Path,
    row: dict[str, Any],
    inventory: dict[str, str],
    baseline: dict[str, Any],
    spec: Any,
    budget: Any,
) -> None:
    if result is None:
        created = [
            r
            for r in service.store.list()
            if r["run_id"] not in existing and r["operation"] == "erase"
        ]
        if len(created) > 1:
            raise ValueError("ambiguous failed edit ownership")
        if created:
            receipt = service.store.path(created[0]["run_id"]) / "snapshot.json"
            if receipt.exists():
                snapshot = json.loads(pilot.bounded_file(receipt))
                if snapshot.get("output") == str(scratch):
                    result = created[0]
                    row["recovered_failed_run_for_restore"] = True
    if result is not None:
        row["run_id"] = result["run_id"]
        snapshot = json.loads(
            pilot.bounded_file(service.store.path(result["run_id"]) / "snapshot.json")
        )
        # Partial checkpoint transactions may not have valid after-images. Restore regardless.
        preservation_error = None
        try:
            if "afterimages" not in row and snapshot.get("status") == "edited":
                row["afterimages"] = pilot.persist_afterimages(
                    scratch, snapshot, directory, budget.finalization()
                )
        except Exception as exc:
            preservation_error = exc
            row["afterimage_error"] = f"{type(exc).__name__}: {exc}"
        restored = service.restore(result["run_id"], apply=True)
        pilot.atomic_json(directory / "restore.json", restored)
        if preservation_error is not None:
            row["status"] = "failed"
    if pilot.checkpoint_inventory(scratch, 4 * 1024**3) != inventory:
        raise ValueError("scratch is not bit-exact restored planted source")
    row["restored_bit_exact"] = True
    scored = pilot.score_checkpoint(
        scratch, spec, budget.finalization(), directory / "restored-evaluation.json"
    )
    if pilot.generation_identity(scored) != pilot.generation_identity(baseline):
        raise ValueError("restored generation differs")
    row["restored_generation_exact"] = True


def execute(plan: dict[str, Any], records: dict[str, Any], pilot: Any) -> dict[str, Any]:
    from lacunai.experimental.resources import ResourceAbort

    source, workspace = path(plan["planted"]), path(plan["workspace"])
    budget = pilot.Budget(path(plan["budget_root"]), None)
    spec, inventory = records["spec"], records["planted_inventory"]
    budget.check(METADATA_RESERVE)
    workspace.mkdir(mode=0o700, parents=False, exist_ok=False)
    summary: dict[str, Any] = {
        "status": "running",
        "verdict": "inconclusive",
        "candidates": [],
        "operation": "qualified_fact_followup",
        "replanting_performed": False,
        "retain_reference": {
            "reused": True,
            "original_plant_steps": records["original_summary"]["planting"]["steps"],
            "matched_to": "original planting, not new unlearning updates",
        },
    }
    store = pilot.RunStore(workspace / "aggregate")
    run_id, directory = store.create(spec, "factoid-followup", {"plan": plan})
    aggregate = (store, run_id, directory)
    try:
        pilot.atomic_json(workspace / "target.json", asdict(spec))
        pilot.atomic_json(workspace / "gradient-data.json", records["development"])
        pilot.atomic_json(
            workspace / "manifest.json",
            {
                "plan": plan,
                "runtime": asdict(budget.config()),
                "candidate_settings": {
                    "methods": list(METHODS),
                    "steps": 128,
                    "strength": 1.0,
                    "max_relative_change": 0.25,
                },
                "existing_controls": {
                    name: plan["inputs"][name]
                    for name in ("untouched", "retain_only", "noop", "planted_baseline")
                },
                "original_manifest": records["original_manifest"],
                "retain_reference": summary["retain_reference"],
            },
        )
        baseline = pilot.score_checkpoint(source, spec, budget, workspace / "baseline.json")
        pilot.Service._qualify(baseline, spec)
        if pilot.generation_identity(baseline) != pilot.generation_identity(
            records["planted_baseline"]
        ):
            raise ValueError("fresh baseline differs from original planted baseline")
        scratch = workspace / "scratch"
        pilot.byte_copy(source, scratch, budget)
        noop = pilot.score_checkpoint(scratch, spec, budget, workspace / "noop.json")
        if pilot.generation_identity(noop) != pilot.generation_identity(baseline):
            raise ValueError("fresh byte-identical no-op differs")
        pilot.atomic_json(
            workspace / "scratch-lineage.json",
            {
                "source": str(source),
                "inventory": inventory,
                "role": "fresh no-op control then sequential restored candidate scratch",
            },
        )
        service = pilot.Service(workspace / "operations")
        for method in METHODS:
            if pilot.checkpoint_inventory(scratch, 4 * 1024**3) != inventory:
                raise ValueError("scratch not restored before next candidate")
            reserve_transaction(scratch, budget)
            candidate = workspace / method
            candidate.mkdir(exist_ok=False)
            row: dict[str, Any] = {"method": method, "status": "running"}
            summary["candidates"].append(row)
            pilot.atomic_json(workspace / "summary.json", summary)
            existing = {r["run_id"] for r in service.store.list()}
            result = None
            try:
                result = service.erase(
                    scratch,
                    workspace / "target.json",
                    method,
                    output=scratch,
                    in_place=True,
                    apply=True,
                    strength=1.0,
                    max_relative_change=0.25,
                    steps=128,
                    config=budget.config(),
                    gradient_data=workspace / "gradient-data.json",
                )
                pilot.atomic_json(candidate / "erase.json", result)
                snapshot = json.loads(
                    pilot.bounded_file(service.store.path(result["run_id"]) / "snapshot.json")
                )
                row["afterimages"] = pilot.persist_afterimages(scratch, snapshot, candidate, budget)
                independent = service.verify(result["run_id"], budget.config())
                pilot.atomic_json(candidate / "verify.json", independent)
                benign = pilot.benign_relearning(
                    scratch,
                    list(snapshot["tensors"]),
                    records["training"],
                    spec,
                    budget,
                    candidate / "benign-relearning.json",
                )
                comparison = pilot.compare(baseline, benign["evaluation"], spec)
                pilot.atomic_json(candidate / "benign-comparison.json", comparison)
                row.update(
                    status="evaluated",
                    verdict=result["verdict"],
                    independent_verdict=independent["verdict"],
                    benign_verdict=comparison["verdict"],
                )
            except Exception as exc:
                row.update(
                    status="failed", verdict="inconclusive", error=f"{type(exc).__name__}: {exc}"
                )
                progress("followup_candidate", state="failed", method=method, error=str(exc))
                if isinstance(exc, (OSError, MemoryError, TimeoutError, ResourceAbort)):
                    raise
            finally:
                try:
                    restore_candidate(
                        pilot,
                        service,
                        result,
                        existing,
                        scratch,
                        candidate,
                        row,
                        inventory,
                        baseline,
                        spec,
                        budget,
                    )
                except BaseException as exc:
                    row.update(status="failed", restore_error=f"{type(exc).__name__}: {exc}")
                    pilot.atomic_json(workspace / "summary.json", summary)
                    raise
                pilot.atomic_json(workspace / "summary.json", summary)
        passed = [
            r["method"]
            for r in summary["candidates"]
            if r.get("status") == "evaluated"
            and all(
                r.get(k) == PASSED_VERDICT
                for k in ("verdict", "independent_verdict", "benign_verdict")
            )
            and r.get("restored_bit_exact")
            and r.get("restored_generation_exact")
        ]
        summary.update(
            status="complete",
            verdict=PASSED_VERDICT if passed == list(METHODS) else "failed",
            passed_methods=passed,
        )
    except BaseException as exc:
        summary.update(
            status="failed", verdict="inconclusive", error=f"{type(exc).__name__}: {exc}"
        )
        progress("followup", state="failed", error=str(exc))
    finally:
        try:
            source_unchanged = pilot.checkpoint_inventory(source, 4 * 1024**3) == inventory
        except Exception as exc:
            source_unchanged = False
            summary["source_readback_error"] = f"{type(exc).__name__}: {exc}"
        summary["source_files_unchanged"] = source_unchanged
        if not source_unchanged:
            summary.update(
                status="failed", verdict="inconclusive", error="retained planted source changed"
            )
        summary["claim"] = "reduced recoverability under these tests only; not information absence"
        # Signing duplicates JSON evidence only; reserve actual metadata before the existing helper.
        json_bytes = sum(
            p.stat().st_size
            for p in workspace.rglob("*.json")
            if "aggregate" not in p.relative_to(workspace).parts
        )
        try:
            budget.check(json_bytes + METADATA_RESERVE)
            pilot.atomic_json(workspace / "summary.json", summary)
            summary["aggregate_report"] = pilot.sign_aggregate(workspace, aggregate, summary)
        except Exception as exc:
            summary.update(
                status="failed",
                verdict="inconclusive",
                attestation_error=f"{type(exc).__name__}: {exc}",
            )
            if store.get(run_id)["status"] == "running":
                store.update(
                    run_id,
                    status="failed",
                    verdict="inconclusive",
                    error=summary["attestation_error"],
                )
        pilot.atomic_json(workspace / "summary.json", summary)
        progress("followup", state=summary["status"], verdict=summary["verdict"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan, records, pilot = prepare(args.plan, args.plan_sha256)
    if not args.apply:
        progress("followup", state="planned", model_loaded=False, methods=list(METHODS))
        return 0
    result = execute(plan, records, pilot)
    return 0 if result["verdict"] == PASSED_VERDICT else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        progress("followup", state="failed", error=f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
