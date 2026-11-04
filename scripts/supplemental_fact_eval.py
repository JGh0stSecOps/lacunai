#!/usr/bin/env python3
"""Pinned supplementary evaluation; model writes remain in memory only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import uuid
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

CONTRACT_SHA = "462c7a875b9b425c22d6c7b20847ff09c4c4ad8c3b44e6aa04a289a54dd1b533"
METHODS = {
    "ablate-neurons",
    "rank-one",
    "ablate-direction",
    "npo-kl",
    "grad-diff-kl",
    "npo-kl-repair",
    "grad-diff-kl-repair",
}
CAP = 128 * 1024**2
OUTPUT_CAP = 64 * 1024**2


def emit(phase: str, **fields: Any) -> None:
    print(json.dumps({"phase": phase, **fields}), flush=True)


def safe(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute() or ".." in p.parts or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError("unapproved or redirected path")
    return p


def sha(path: str | Path) -> str:
    p = safe(path)
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pinned(row: dict[str, Any], *, binary: bool = False) -> Any:
    if set(row) != {"path", "sha256"} or not re.fullmatch("[0-9a-f]{64}", row["sha256"]):
        raise ValueError("invalid file pin")
    p = safe(row["path"])
    if p.stat().st_size > CAP or sha(p) != row["sha256"]:
        raise ValueError("input size/hash differs")
    return p if binary else json.loads(p.read_text())


def prepare(
    path: Path, pin: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    plan = pinned({"path": str(path), "sha256": pin})
    if (
        set(plan)
        != {
            "schema_version",
            "code",
            "bundle",
            "contract",
            "manifest",
            "planted",
            "planted_inventory",
            "pilot_summary",
            "qualification",
            "candidates",
            "output",
            "key_dir",
        }
        or plan["schema_version"] != 1
    ):
        raise ValueError("plan schema differs")
    code = safe(plan["code"])
    planted = safe(plan["planted"])
    output = safe(plan["output"])
    key_dir = safe(plan["key_dir"])
    if key_dir.parent != output.parent or key_dir == output or key_dir.exists():
        raise ValueError("exclusive signing directory must be an output sibling")
    if key_dir == planted or planted in key_dir.parents or key_dir in planted.parents:
        raise ValueError("signing directory overlaps checkpoint")
    if output.exists():
        raise FileExistsError("exclusive output exists")
    if planted == output or planted in output.parents or output in planted.parents:
        raise ValueError("output overlaps checkpoint")
    pins = pinned(plan["bundle"])
    if safe(plan["bundle"]["path"]) != code / "hashes.json":
        raise ValueError("bundle path differs")
    for name, value in pins.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha(code / relative) != value:
            raise ValueError("code inventory differs")
    actual = {
        str(p.relative_to(code)) for part in ("src", "scripts") for p in (code / part).rglob("*.py")
    }
    if not actual <= pins.keys():
        raise ValueError("unmanifested Python code")
    if plan["contract"]["sha256"] != CONTRACT_SHA:
        raise ValueError("supplement contract differs")
    contract = pinned(plan["contract"])
    manifest = pinned(plan["manifest"])
    inventory = pinned(plan["planted_inventory"])
    terminal = pinned(plan["pilot_summary"])
    qualification = pinned(plan["qualification"])
    expected_methods(terminal)
    if terminal.get("status") not in ("complete", "failed"):
        raise ValueError("pilot is not terminal")
    original_target = qualification["target"]["rows"]
    original_retain = qualification["retain"]["rows"]
    if (
        len(original_target) != 8
        or any(row["classification"] != "recovered" for row in original_target)
        or len(original_retain) != 8
        or sum(row["classification"] == "recovered" for row in original_retain) < 6
    ):
        raise ValueError("planted checkpoint lacks original qualification")
    if (
        qualification.get("checkpoint_fingerprint")
        != hashlib.sha256(
            json.dumps(
                inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
    ):
        raise ValueError("qualification checkpoint differs")
    runtime = manifest["runtime"]
    wanted = contract["inference"]
    for key in (
        "device",
        "dtype",
        "seed",
        "torch_threads",
        "max_prompt_tokens",
        "max_new_tokens",
    ):
        if runtime.get(key) != wanted[key]:
            raise ValueError("frozen inference identity differs")
    if runtime.get("max_seconds") is not None:
        raise ValueError("overall deadline differs")
    scorer = code / "src/lacunai/verify/evaluation.py"
    if sha(scorer) != contract["scoring"]["module_sha256"]:
        raise ValueError("scorer differs")
    if len(contract["target_probes"]) != 16 or len(contract["retain_probes"]) != 8:
        raise ValueError("probe inventory differs")
    seen = set()
    for item in plan["candidates"]:
        if (
            set(item) != {"method", "snapshot", "afterimage", "blob"}
            or item["method"] not in METHODS
            or item["method"] in seen
        ):
            raise ValueError("candidate selection differs")
        if item["method"] not in {row["method"] for row in terminal["candidates"]}:
            raise ValueError("candidate not in completed pilot")
        seen.add(item["method"])
        snapshot = pinned(item["snapshot"])
        after = pinned(item["afterimage"])
        pinned(item["blob"], binary=True)
        validate_receipts(snapshot, after, inventory, item["blob"])
        bind_candidate(item, snapshot, after, terminal)
    return plan, contract, manifest, inventory


def expected_methods(terminal: dict[str, Any]) -> set[str]:
    methods = [row["method"] for row in terminal["candidates"]]
    if (
        not methods
        or len(methods) != len(set(methods))
        or set(methods)
        not in (
            {"ablate-neurons", "rank-one", "ablate-direction"},
            {"npo-kl", "grad-diff-kl"},
            {"npo-kl-repair", "grad-diff-kl-repair"},
        )
    ):
        raise ValueError("terminal candidate cohort differs")
    return set(methods)


def cohort_verdict(verdicts: list[str], absent_methods: list[str]) -> str:
    if "failed" in verdicts:
        return "failed"
    if absent_methods or "inconclusive" in verdicts:
        return "inconclusive"
    return "passed the stated experimental contract"


def bind_candidate(
    item: dict[str, Any], snapshot: dict[str, Any], after: dict[str, Any], terminal: dict[str, Any]
) -> None:
    rows = [row for row in terminal["candidates"] if row["method"] == item["method"]]
    if len(rows) != 1 or not rows[0].get("run_id"):
        raise ValueError("candidate summary identity missing or duplicated")
    row = rows[0]
    if Path(snapshot["run_dir"]).name != row["run_id"] or after != row.get("afterimages"):
        raise ValueError("candidate snapshot/afterimage not bound to method summary")


def validate_receipts(
    snapshot: dict[str, Any], after: dict[str, Any], inventory: dict[str, str], blob: dict[str, Any]
) -> None:
    if {k: v["sha256"] for k, v in snapshot["before_files"].items()} != inventory:
        raise ValueError("candidate before inventory differs from planted")
    if snapshot.get("status") not in {"edited", "restored"}:
        raise ValueError("snapshot has no completed edit lineage")
    if (
        not snapshot.get("after_files")
        or after.get("checkpoint_after_files") != snapshot["after_files"]
    ):
        raise ValueError("after checkpoint inventory differs")
    if after.get("snapshot_directory") != snapshot.get("run_dir"):
        raise ValueError("afterimage snapshot directory differs")
    names = set(snapshot["tensors"])
    if not names or len(names) > 8 or set(after["tensors"]) != names:
        raise ValueError("touched tensor inventory differs")
    if (
        after["snapshot_id"] != snapshot["snapshot_id"]
        or after["artifact_sha256"] != blob["sha256"]
        or after["artifact"] != Path(blob["path"]).name
    ):
        raise ValueError("afterimage binding differs")
    if type(after["artifact_bytes"]) is not int or not 0 < after["artifact_bytes"] <= CAP:
        raise ValueError("afterimage size bound")
    total = 0
    for name, row in after["tensors"].items():
        before = snapshot["tensors"][name]
        if (
            row.get("source_file") != before.get("file")
            or row.get("before_image_blob") != before.get("blob")
            or row["shape"] != before["shape"]
            or row["dtype"] != before["dtype"]
            or row["before_raw_sha256"] != before["before_sha256"]
            or row["bytes"] != before["length"]
        ):
            raise ValueError("afterimage descriptor binding differs")
        if type(row["bytes"]) is not int or row["bytes"] <= 0:
            raise ValueError("invalid tensor bytes")
        total += row["bytes"]
    if total > CAP:
        raise ValueError("tensor aggregate exceeds bound")


def aggregate(baseline: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = {"recovered", "not_recovered", "ambiguous", "missing"}
    for rows in (baseline, after):
        if len(rows) != 24 or len({x["id"] for x in rows}) != 24:
            raise ValueError("paired evaluation identities differ")
        if any(x["classification"] not in allowed for x in rows):
            raise ValueError("invalid classification label")
        if (
            sum(x["split"] == "target" for x in rows) != 16
            or sum(x["split"] == "retain" for x in rows) != 8
        ):
            raise ValueError("evaluation split counts differ")
    old = {x["id"]: x for x in baseline}
    new = {x["id"]: x for x in after}
    if len(old) != 24 or set(old) != set(new):
        raise ValueError("paired evaluation identities differ")
    if any(
        any(
            old[key].get(field) != new[key].get(field)
            for field in ("split", "kind", "language", "prompt", "answer")
        )
        for key in old
    ):
        raise ValueError("paired probe metadata differs")
    target = [x for x in after if x["split"] == "target"]
    retain = [x for x in after if x["split"] == "retain"]

    def recovered(x: dict[str, Any]) -> bool:
        return bool(x["classification"] == "recovered" or x.get("observed_target_string") is True)

    loss = [
        x["id"]
        for x in retain
        if old[x["id"]]["classification"] == "recovered" and x["classification"] != "recovered"
    ]
    missing_baseline = [x["id"] for x in target if old[x["id"]]["classification"] != "recovered"]
    uncertain = [x["id"] for x in after if x["classification"] in ("ambiguous", "missing")]
    retain_categories = {x["kind"] for x in retain if old[x["id"]]["classification"] == "recovered"}
    count = sum(recovered(x) for x in target)
    verdict = (
        "failed"
        if count or loss
        else "inconclusive"
        if missing_baseline
        or uncertain
        or retain_categories != {"pretrained_numeric_fact", "neighboring_task_capability"}
        else "passed the stated experimental contract"
    )
    return {
        "verdict": verdict,
        "target_recoveries": count,
        "target_by_language": {
            lang: {
                "total": sum(x["language"] == lang for x in target),
                "recovered": sum(x["language"] == lang and recovered(x) for x in target),
                "baseline_qualified": sum(
                    x["language"] == lang and old[x["id"]]["classification"] == "recovered"
                    for x in target
                ),
            }
            for lang in ("en", "es", "zh")
        },
        "baseline_correct_retain": sum(
            old[x["id"]]["classification"] == "recovered" for x in retain
        ),
        "paired_retain_losses": loss,
        "unqualified_target_ids": missing_baseline,
        "uncertain_ids": uncertain,
        "claim": "reduced recoverability under these tests only; not information absence",
    }


def write(output: Path, name: str, value: Any) -> None:
    data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) + len(data) > OUTPUT_CAP:
        raise OSError("output exceeds64MiB")
    with (output / name).open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def evaluate(
    runtime: Any, contract: dict[str, Any], output: Path, label: str
) -> list[dict[str, Any]]:
    from lacunai.verify.evaluation import classify, classify_retain

    rows: list[dict[str, Any]] = []
    for split, key in [("target", "target_probes"), ("retain", "retain_probes")]:
        for item in contract[key]:
            emit("evaluation", arm=label, completed=len(rows), total=24, unit="probes")
            runtime.check()
            generation = runtime.generate(item["prompt"])
            score = (classify if split == "target" else classify_retain)(
                generation.get("output"), item["answer"], generation.get("truncated")
            )
            ids = generation.get("token_ids")
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(i) is not int or i < 0 for i in ids)
                or generation.get("output_tokens") != len(ids)
                or type(generation.get("input_tokens")) is not int
                or generation["input_tokens"] <= 0
            ):
                score = {
                    "classification": "missing",
                    "reason": "invalid_generation_accounting",
                }
            rows.append({**item, "split": split, "generation": generation, **score})
            write(output, f"{label}-probe-{len(rows):02d}.json", rows[-1])
    emit("evaluation", arm=label, completed=24, total=24, unit="probes")
    return rows


def state_hashes(runtime: Any) -> dict[str, str]:
    from lacunai.core.model import active_phase
    from lacunai.experimental.tensors import tensor_hash

    with active_phase("supplement_full_state_hash"):
        return {name: tensor_hash(value) for name, value in runtime.model.state_dict().items()}


def candidate_transaction(
    runtime: Any,
    snapshot: dict[str, Any],
    receipt: dict[str, Any],
    blob: Path,
    output: Path,
    method: str,
    canonical: dict[str, str],
    operation: Callable[[], Any],
) -> Any:
    try:
        result = _candidate_transaction(
            runtime, snapshot, receipt, blob, output, method, canonical, operation
        )
        emit("candidate_transaction", method=method, state="complete")
        return result
    except BaseException as exc:
        try:
            write(
                output,
                method + "-error.json",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "notes": getattr(exc, "__notes__", []),
                },
            )
        except BaseException as artifact_error:
            exc.add_note("error artifact failure: " + repr(artifact_error))
        emit("candidate_transaction", method=method, state="failed", error_type=type(exc).__name__)
        raise


def _candidate_transaction(
    runtime: Any,
    snapshot: dict[str, Any],
    receipt: dict[str, Any],
    blob: Path,
    output: Path,
    method: str,
    canonical: dict[str, str],
    operation: Callable[[], Any],
) -> Any:
    emit("candidate_transaction", method=method, state="active")
    torch = import_module("torch")
    safe_open = import_module("safetensors").safe_open
    validate_receipts(
        snapshot,
        receipt,
        runtime.inventory,
        {"path": str(blob), "sha256": receipt["artifact_sha256"]},
    )
    if blob.stat().st_size != receipt["artifact_bytes"] or sha(blob) != receipt["artifact_sha256"]:
        raise ValueError("afterimage blob hash/size differs")
    if state_hashes(runtime) != canonical:
        raise ValueError("candidate did not start from identical planted model")
    state = runtime.model.state_dict()
    names = set(snapshot["tensors"])
    originals = {n: state[n].clone() for n in names}
    primary: BaseException | None = None
    try:
        with safe_open(blob, framework="pt", device="cpu") as stored:
            if set(stored.keys()) != names:
                raise ValueError("blob tensor inventory differs")
            for name in sorted(names):
                desc = receipt["tensors"][name]
                value = stored.get_tensor(name)
                if (
                    runtime.storage_dtypes.get(name) != desc["dtype"]
                    or stored.get_slice(name).get_dtype() != desc["dtype"]
                    or list(value.shape) != desc["shape"]
                    or tuple(value.shape) != tuple(state[name].shape)
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError("tensor dtype/shape/finite differs")
                raw = value.contiguous().view(torch.uint8).numpy().tobytes()
                before = (
                    originals[name].to(value.dtype).contiguous().view(torch.uint8).numpy().tobytes()
                )
                if (
                    len(raw) != desc["bytes"]
                    or hashlib.sha256(raw).hexdigest() != desc["after_raw_sha256"]
                    or hashlib.sha256(before).hexdigest() != desc["before_raw_sha256"]
                ):
                    raise ValueError("actual tensor byte hash differs")
                with torch.no_grad():
                    state[name].copy_(value.to(state[name].dtype))
                del value, raw, before
        applied = state_hashes(runtime)
        if any(applied[n] != canonical[n] for n in canonical if n not in names):
            raise ValueError("nonallowlisted model change")
        return operation()
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            with torch.no_grad():
                for name, value in originals.items():
                    state[name].copy_(value)
            restored = state_hashes(runtime) == canonical
            write(
                output,
                method + "-restore.json",
                {"in_memory_bit_exact": restored, "canonical_hashes": canonical},
            )
            emit("candidate_restoration", method=method, state="complete" if restored else "failed")
            if not restored:
                raise ValueError("full model restoration differs")
        except BaseException as restore_error:
            if primary is None:
                raise
            primary.add_note("restoration failure: " + repr(restore_error))
            emit(
                "candidate_restoration",
                method=method,
                state="failed",
                error_type=type(restore_error).__name__,
            )


def run(
    plan: dict[str, Any],
    contract: dict[str, Any],
    manifest: dict[str, Any],
    inventory: dict[str, str],
) -> None:
    torch = import_module("torch")
    from lacunai.core.model import HFRuntime, active_phase, checkpoint_inventory
    from lacunai.core.spec import RuntimeConfig

    planted = Path(plan["planted"])
    output = Path(plan["output"])
    if checkpoint_inventory(planted, 4 * 1024**3) != inventory:
        raise ValueError("actual planted inventory differs")
    import shutil

    if shutil.disk_usage(output.parent).free < 8 * 1024**3 + OUTPUT_CAP:
        raise OSError("insufficient disk reserve")
    output.mkdir(exist_ok=False)
    status: dict[str, Any] = {
        "state": "running",
        "measurement": "in_memory_candidate_afterimage",
        "candidate_results": [],
        "expected_methods": sorted(expected_methods(pinned(plan["pilot_summary"]))),
        "absent_methods": sorted(
            expected_methods(pinned(plan["pilot_summary"]))
            - {x["method"] for x in plan["candidates"]}
        ),
        "new_checkpoint_written": False,
    }
    write(
        output,
        "manifest.json",
        {
            "plan": plan,
            "runtime": manifest["runtime"],
            "scorer_sha256": contract["scoring"]["module_sha256"],
            "measurement": status["measurement"],
        },
    )

    primary_error: BaseException | None = None
    try:
        with active_phase("supplement_baseline_load"):
            runtime = HFRuntime(planted, RuntimeConfig(**manifest["runtime"]))
        if any(p.dtype != torch.float32 for p in runtime.model.parameters()):
            raise ValueError("runtime parameter dtype differs")
        canonical = state_hashes(runtime)
        baseline = evaluate(runtime, contract, output, "baseline")
        write(output, "baseline.json", baseline)
        status["baseline_scores"] = {
            split: {
                "total": sum(row["split"] == split for row in baseline),
                "recovered": sum(
                    row["split"] == split and row["classification"] == "recovered"
                    for row in baseline
                ),
            }
            for split in ("target", "retain")
        }
        if state_hashes(runtime) != canonical:
            raise ValueError("baseline inference mutated model")
        del runtime
        gc.collect()
        for item in plan["candidates"]:
            method = item["method"]
            emit("candidate_preparation", method=method)
            snapshot = pinned(item["snapshot"])
            receipt = pinned(item["afterimage"])
            blob = pinned(item["blob"], binary=True)
            validate_receipts(snapshot, receipt, inventory, item["blob"])
            if blob.stat().st_size != receipt["artifact_bytes"]:
                raise ValueError("afterimage actual length differs")
            with active_phase("supplement_candidate_load"):
                runtime = HFRuntime(planted, RuntimeConfig(**manifest["runtime"]))
            if state_hashes(runtime) != canonical:
                raise ValueError("candidate did not start from identical planted model")

            def operation(
                active_runtime: Any = runtime, active_method: str = method
            ) -> dict[str, Any]:
                rows = evaluate(active_runtime, contract, output, active_method)
                return aggregate(baseline, rows)

            result = candidate_transaction(
                runtime, snapshot, receipt, blob, output, method, canonical, operation
            )
            status["candidate_results"].append(
                {"method": method, "snapshot_id": snapshot["snapshot_id"], **result}
            )
            write(output, method + "-result.json", result)
            del runtime
            gc.collect()
        status["state"] = "complete"
        verdicts = [row["verdict"] for row in status["candidate_results"]]
        status["verdict"] = cohort_verdict(verdicts, status["absent_methods"])
    except BaseException as exc:
        primary_error = exc
        status.update(
            state="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            error_notes=getattr(exc, "__notes__", []),
        )
        raise
    finally:
        try:
            unchanged = checkpoint_inventory(planted, 4 * 1024**3) == inventory
            status["planted_files_unchanged"] = unchanged
            if not unchanged:
                status["state"] = "failed"
            status.setdefault("verdict", "inconclusive")
            write(output, "status.json", status)
            write(
                output,
                "receipt.json",
                {
                    "schema_version": 1,
                    "measurement": "in_memory_candidate_afterimage",
                    "files_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
                    "contract_sha256": CONTRACT_SHA,
                    "scientific_correctness_certified": False,
                },
            )
            sign_terminal(output, Path(plan["key_dir"]), status)
            emit("terminal", state=status["state"], planted_files_unchanged=unchanged)
            if not unchanged:
                raise ValueError("planted files changed")
        except BaseException as finalization_error:
            if primary_error is None:
                raise
            primary_error.add_note("finalization failure: " + repr(finalization_error))
            emit("terminal", state="failed", error_type=type(finalization_error).__name__)


def sign_terminal(output: Path, key_dir: Path, status: dict[str, Any]) -> dict[str, Any]:
    from lacunai.verify.reports import verify_report, write_signed_report

    key_dir = safe(key_dir)
    if key_dir.exists() or key_dir.parent != output.parent or key_dir == output:
        raise ValueError("signing directory must remain an exclusive output sibling")

    record = {
        "run_id": uuid.uuid4().hex,
        "target_id": "station-amber-access-code-supplement-v1",
        "status": status["state"],
        "verdict": status.get("verdict", "inconclusive"),
        "operation": "supplemental_afterimage_evaluation",
        "scientific_summary": status,
        "before": status.get("baseline_scores"),
        "after": status.get("candidate_results", []),
        "scientific_correctness_certified": False,
    }
    write_signed_report(output, record, key_dir)
    verified = verify_report(output, key_dir / "public.key")
    emit("signature_verified", scientific_correctness_certified=False)
    return verified


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    emit("preparing", apply=a.apply)
    plan, contract, manifest, inventory = prepare(a.plan, a.plan_sha256)
    if not a.apply:
        emit("plan_only", model_loaded=False)
        return
    sys.path[:0] = [str(Path(plan["code"]) / "src")]
    run(plan, contract, manifest, inventory)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 -- terminal boundary records cancellation
        emit("failed", error_type=type(exc).__name__)
        raise SystemExit(1) from None
