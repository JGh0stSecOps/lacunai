"""Shared CLI/MCP orchestration. The editor never supplies executable code."""

from __future__ import annotations

import gc
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path
from typing import Any, cast

from lacunai.experimental.artifacts import atomic_json, private_directory, secure_path
from lacunai.map.fingerprints import assert_identity, fingerprint
from lacunai.map.index import MapIndex
from lacunai.map.localize import locate
from lacunai.restore.checkpoints import (
    prepare_snapshot,
    restore_snapshot,
    write_edited_checkpoint,
)
from lacunai.verify.evaluation import compare, evaluate
from lacunai.verify.reports import verify_report, write_signed_report

from .model import HFRuntime, checkpoint_inventory, progress
from .records import RunStore
from .spec import RuntimeConfig, TargetSpec, bounded_file, digest


def _serialized(function: Any) -> Any:
    @wraps(function)
    def wrapped(self: Service, *args: Any, **kwargs: Any) -> Any:
        with self._operation():
            return function(self, *args, **kwargs)

    return wrapped


class Service:
    """One lifecycle for local numerical work and its independent evidence."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.absolute()
        self.store = RunStore(self.workspace)
        self.index = MapIndex(self.workspace)
        self._operation_state = threading.local()

    @contextmanager
    def _operation(self) -> Iterator[None]:
        if getattr(self._operation_state, "active", False):
            yield
            return
        progress("workspace_operation", state="active")
        try:
            with self._lock(self.workspace):
                self._operation_state.active = True
                try:
                    yield
                finally:
                    self._operation_state.active = False
        except Exception as exc:
            progress("workspace_operation", state="failed", error=str(exc))
            raise
        progress("workspace_operation", state="complete")

    @contextmanager
    def _lock(self, model: Path) -> Iterator[None]:
        locks = private_directory(self.workspace, "locks")
        path = secure_path(locks, digest(str(model.absolute())) + ".lock")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ValueError("checkpoint operation is locked; inspect the existing run") from exc
        inode = os.fstat(fd).st_ino
        try:
            os.write(fd, json.dumps({"pid": os.getpid(), "checkpoint": str(model)}).encode())
            os.fsync(fd)
            yield
        finally:
            os.close(fd)
            if not path.is_symlink() and path.exists() and path.stat().st_ino == inode:
                path.unlink()  # Only this operation's owned lock, never checkpoint data.

    def _finish(self, run_id: str, **fields: Any) -> dict[str, Any]:
        # Sign the proposed terminal result before marking the operational record complete.
        record = {**self.store.get(run_id), **fields, "status": "complete"}
        signed = write_signed_report(self.store.path(run_id), record, self.workspace / "signing")
        terminal = self.store.update(run_id, status="complete", **fields)
        return {**terminal, "signature": signed}

    def _failure(self, run_id: str, exc: Exception) -> None:
        progress("operation", state="failed", run_id=run_id, error=f"{type(exc).__name__}: {exc}")
        if self.store.get(run_id)["status"] != "running":
            return
        record = self.store.update(
            run_id, status="failed", error=f"{type(exc).__name__}: {exc}", verdict="inconclusive"
        )
        directory = self.store.path(run_id)
        try:
            write_signed_report(directory, record, self.workspace / "signing")
        except Exception as signing_error:
            # Preserve immutable/partial scientific artifacts and the original exception.
            atomic_json(
                directory / "finalization-error.json",
                {
                    "status": "failed",
                    "run_id": run_id,
                    "operation_error": str(exc),
                    "signing_error": str(signing_error),
                    "scientific_report_usable": False,
                },
            )
            progress("report_finalization", state="failed", run_id=run_id, error=str(signing_error))

    def _authenticated(self, run_id: str) -> tuple[dict[str, Any], Path]:
        directory = self.store.path(run_id)
        verify_report(directory, self.workspace / "signing" / "public.key")
        signed = json.loads(bounded_file(directory / "report.json", 32 * 1024**2))
        current = self.store.get(run_id)
        if any(current.get(key) != value for key, value in signed.items() if key != "updated_at"):
            raise ValueError("operational run differs from authenticated report")
        return signed, directory

    @staticmethod
    def _bound_snapshot(directory: Path) -> dict[str, Any]:
        frozen = json.loads(
            bounded_file(directory / "report-evidence" / "snapshot.json", 32 * 1024**2)
        )
        current = json.loads(bounded_file(directory / "snapshot.json", 32 * 1024**2))
        mutable = {"status", "restore_staged_files", "restored_bit_exact"}
        if {k: v for k, v in frozen.items() if k not in mutable} != {
            k: v for k, v in current.items() if k not in mutable
        }:
            raise ValueError("snapshot transaction differs from signed evidence")
        if current.get("status") not in {frozen.get("status"), "restoring", "restored"}:
            raise ValueError("snapshot status differs from signed transaction")
        return cast(dict[str, Any], current)

    def _mapping(
        self, runtime: HFRuntime, spec: TargetSpec, *, causal: bool
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        try:
            fp = self.index.get(spec.target_id, runtime.fingerprint)
        except ValueError as exc:
            if "no map" not in str(exc):
                raise
            fp = fingerprint(runtime, spec)
            self.index.put(fp)
        assert_identity(runtime, spec, fp)
        if not causal:
            return fp, None
        try:
            localization = self.index.get(spec.target_id, runtime.fingerprint, "localization")
        except ValueError as exc:
            if "no map" not in str(exc):
                raise
            localization = locate(runtime, spec, fp)
            self.index.put(localization)
        assert_identity(runtime, spec, localization)
        return fp, localization

    def _trace(
        self, model: Path, target: Path, config: RuntimeConfig | None, causal: bool
    ) -> dict[str, Any]:
        spec, config = TargetSpec.from_path(target), config or RuntimeConfig()
        run_id, directory = self.store.create(
            spec,
            "locate" if causal else "map",
            {"model": str(model.absolute()), "runtime": asdict(config)},
        )
        progress("trace", state="active", run_id=run_id)
        try:
            with self._lock(model):
                runtime = HFRuntime(model, config)
                fp, localization = self._mapping(runtime, spec, causal=causal)
                atomic_json(directory / "fingerprint.json", fp)
                if localization is not None:
                    atomic_json(directory / "localization.json", localization)
                exported = self.index.export(
                    spec.target_id,
                    runtime.fingerprint,
                    directory / "export",
                    "localization" if causal else "fingerprint",
                )
                return self._finish(
                    run_id,
                    checkpoint_fingerprint=runtime.fingerprint,
                    exports=exported,
                    verdict="inconclusive",
                    claim="localization evidence only",
                    result=localization if causal else fp,
                )
        except Exception as exc:
            self._failure(run_id, exc)
            raise

    @_serialized
    def map(self, model: Path, target: Path, config: RuntimeConfig | None = None) -> dict[str, Any]:
        return self._trace(model, target, config, False)

    @_serialized
    def locate(
        self, model: Path, target: Path, config: RuntimeConfig | None = None
    ) -> dict[str, Any]:
        return self._trace(model, target, config, True)

    @staticmethod
    def _qualify(before: dict[str, Any], spec: TargetSpec) -> None:
        from decimal import Decimal

        for kind, row in before["target"]["by_kind"].items():
            if (
                row["ambiguous"]
                or row["missing"]
                or Decimal(row["recovery_rate"])
                < Decimal(str(spec.thresholds.minimum_baseline_recovery))
            ):
                raise ValueError(f"target was not demonstrated before editing: {kind}")

    @_serialized
    def erase(
        self,
        model: Path,
        target: Path,
        method: str,
        output: Path | None = None,
        apply: bool = False,
        in_place: bool = False,
        strength: float = 1.0,
        max_relative_change: float = 0.25,
        steps: int = 12,
        config: RuntimeConfig | None = None,
    ) -> dict[str, Any]:
        from lacunai.erase.methods import METHOD_INFO, apply_method, touched_tensors

        if method not in METHOD_INFO:
            raise ValueError("unknown erasure method")
        spec, config = TargetSpec.from_path(target), config or RuntimeConfig()
        run_id, directory = self.store.create(
            spec,
            "erase",
            {
                "model": str(model.absolute()),
                "method": method,
                "runtime": asdict(config),
                "apply": apply,
                "in_place": in_place,
                "strength": strength,
                "max_relative_change": max_relative_change,
                "steps": steps,
            },
        )
        progress("erase", state="active" if apply else "planning", run_id=run_id)
        destination = output or (model if in_place else self.workspace / "checkpoints" / run_id)
        try:
            if not apply:
                return self._finish(
                    run_id,
                    dry_run=True,
                    method=method,
                    method_info=METHOD_INFO[method],
                    output=str(destination.absolute()),
                    verdict="inconclusive",
                    claim="plan only; no model loaded or weights written",
                )
            started = time.monotonic()
            with self._lock(model):
                runtime = HFRuntime(model, config)
                fp, localization = self._mapping(runtime, spec, causal=True)
                if localization is None:
                    raise ValueError("causal localization is required before editing")
                atomic_json(directory / "fingerprint-before.json", fp)
                atomic_json(directory / "localization.json", localization)
                before = evaluate(runtime, spec)
                atomic_json(directory / "before.json", before)
                self._qualify(before, spec)
                names = touched_tensors(runtime, method, localization)
                if (
                    checkpoint_inventory(model.absolute(), config.max_model_bytes)
                    != runtime.inventory
                ):
                    raise ValueError("source checkpoint changed since runtime loading")
                snapshot = prepare_snapshot(model, directory, names)
                if {
                    name: row["sha256"] for name, row in snapshot["before_files"].items()
                } != runtime.inventory:
                    raise ValueError("snapshot source differs from loaded runtime")
                edits = apply_method(
                    runtime,
                    method,
                    localization,
                    fp,
                    strength=strength,
                    max_relative_change=max_relative_change,
                    rewrite=spec.rewrite,
                    steps=steps,
                )
                atomic_json(directory / "edit.json", edits["diagnostics"])
                snapshot = write_edited_checkpoint(
                    model, destination, edits["replacements"], snapshot, in_place=in_place
                )
                source_fingerprint = runtime.fingerprint
                del runtime, edits
                gc.collect()
                remaining = int(config.max_seconds - (time.monotonic() - started))
                if remaining < 1:
                    raise TimeoutError("operation budget exhausted before independent reload")
                runtime = HFRuntime(destination, replace(config, max_seconds=remaining))
                after = evaluate(runtime, spec)
                after_fp = fingerprint(runtime, spec)
                atomic_json(directory / "after.json", after)
                atomic_json(directory / "fingerprint-after.json", after_fp)
                comparison = compare(before, after, spec, fp, after_fp)
                atomic_json(directory / "comparison.json", comparison)
                return self._finish(
                    run_id,
                    dry_run=False,
                    method=method,
                    method_info=METHOD_INFO[method],
                    snapshot_id=snapshot["snapshot_id"],
                    output=str(destination.absolute()),
                    source_fingerprint=source_fingerprint,
                    output_fingerprint=runtime.fingerprint,
                    verdict=comparison["verdict"],
                    comparison=comparison,
                    before=before,
                    after=after,
                    claim="reduced recoverability under these tests; never proof of absence",
                )
        except Exception as exc:
            self._failure(run_id, exc)
            raise

    @_serialized
    def verify(self, run_id: str, config: RuntimeConfig | None = None) -> dict[str, Any]:
        parent, parent_dir = self._authenticated(run_id)
        if parent.get("dry_run") or not parent.get("output_fingerprint"):
            raise ValueError("run has no edited checkpoint to verify")
        spec = TargetSpec.from_path(parent_dir / "target.json")
        frozen_config = RuntimeConfig(**parent["metadata"]["runtime"])
        config = config or frozen_config
        if replace(config, max_seconds=frozen_config.max_seconds) != frozen_config:
            raise ValueError("verification inference settings differ from frozen baseline")
        new_id, directory = self.store.create(
            spec, "verify", {"parent_run_id": run_id, "runtime": asdict(config)}
        )
        try:
            output = Path(parent["output"])
            with self._lock(output):
                runtime = HFRuntime(output, config)
                if runtime.fingerprint != parent["output_fingerprint"]:
                    raise ValueError("edited checkpoint changed since the signed run")
                before = json.loads(bounded_file(parent_dir / "before.json"))
                before_fp = json.loads(
                    bounded_file(parent_dir / "fingerprint-before.json", 32 * 1024**2)
                )
                after, after_fp = evaluate(runtime, spec), fingerprint(runtime, spec)
                comparison = compare(before, after, spec, before_fp, after_fp)
                atomic_json(directory / "before.json", before)
                atomic_json(directory / "fingerprint-before.json", before_fp)
                atomic_json(directory / "comparison.json", comparison)
                atomic_json(directory / "after.json", after)
                atomic_json(directory / "fingerprint-after.json", after_fp)
                return self._finish(
                    new_id,
                    parent_run_id=run_id,
                    snapshot_id=parent["snapshot_id"],
                    verdict=comparison["verdict"],
                    comparison=comparison,
                    output=str(output),
                    output_fingerprint=runtime.fingerprint,
                )
        except Exception as exc:
            self._failure(new_id, exc)
            raise

    @_serialized
    def restore(self, run_id: str, apply: bool = False) -> dict[str, Any]:
        parent, parent_dir = self._authenticated(run_id)
        spec = TargetSpec.from_path(parent_dir / "target.json")
        snapshot = self._bound_snapshot(parent_dir)
        new_id, directory = self.store.create(
            spec, "restore", {"parent_run_id": run_id, "apply": apply}
        )
        try:
            with self._lock(Path(snapshot.get("output", snapshot["source"]))):
                result = restore_snapshot(parent_dir, apply=apply)
                atomic_json(directory / "restoration.json", result)
                return self._finish(
                    new_id,
                    parent_run_id=run_id,
                    snapshot_id=snapshot["snapshot_id"],
                    dry_run=not apply,
                    restoration=result,
                    verdict="inconclusive",
                    claim="exact checkpoint restoration" if apply else "restore plan only",
                )
        except Exception as exc:
            self._failure(new_id, exc)
            raise

    @_serialized
    def runs(self) -> dict[str, Any]:
        return {"runs": self.store.list(), "maps": self.index.list()}

    @_serialized
    def forget(
        self,
        model: Path,
        target: Path,
        method: str,
        output: Path | None = None,
        apply: bool = False,
        in_place: bool = False,
        strength: float = 1.0,
        max_relative_change: float = 0.25,
        steps: int = 12,
        config: RuntimeConfig | None = None,
        state_config: Path | None = None,
        state_scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        from lacunai.adapters.neurailyzer import preflight, wipe_state

        protected = [model, self.workspace] + ([output] if output is not None else [])
        scopes = state_scopes or ["session", "sandbox"]
        state_plan = preflight(state_config, scopes, protected) if state_config else None
        result = self.erase(
            model,
            target,
            method,
            output,
            apply,
            in_place,
            strength,
            max_relative_change,
            steps,
            config,
        )
        if state_config is None:
            return {
                **result,
                "state": {
                    "performed": False,
                    "reason": "no explicit target-scoped state configuration supplied",
                },
            }
        if apply and result["verdict"] != "passed_the_stated_experimental_contract":
            return {
                **result,
                "state": {
                    "performed": False,
                    "plan": state_plan,
                    "reason": "weight verification contract did not pass",
                },
            }
        spec = TargetSpec.from_path(target)
        new_id, directory = self.store.create(spec, "forget", {"weight_run_id": result["run_id"]})
        try:
            state = wipe_state(state_config, scopes, protected_paths=protected, apply=apply)
            atomic_json(directory / "state.json", state)
            return self._finish(
                new_id,
                weight_run_id=result["run_id"],
                verdict=result["verdict"],
                state=state,
                dry_run=not apply,
            )
        except Exception as exc:
            self._failure(new_id, exc)
            raise
