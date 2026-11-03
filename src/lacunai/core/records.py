"""Private run records with frozen targets and explicit terminal states."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from lacunai.experimental.artifacts import atomic_json, private_directory, secure_path

from .spec import TargetSpec, bounded_file, digest


class RunStore:
    """A run's target cannot change, and terminal records cannot be rewritten."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.absolute()
        self.root = private_directory(self.workspace, "runs")

    def path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("invalid run identifier")
        return secure_path(self.root, run_id)

    def create(
        self, target: TargetSpec, operation: str, metadata: dict[str, Any]
    ) -> tuple[str, Path]:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", operation):
            raise ValueError("invalid operation")
        run_id = uuid.uuid4().hex
        directory = self.path(run_id)
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        frozen = asdict(target)
        atomic_json(directory / "target.json", frozen)
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "target_id": target.target_id,
            "target_digest": digest(frozen),
            "operation": operation,
            "metadata": metadata,
            "status": "running",
            "created_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(directory / "run.json", record)
        return run_id, directory

    def get(self, run_id: str) -> dict[str, Any]:
        directory = self.path(run_id)
        record = json.loads(bounded_file(directory / "run.json"))
        target = json.loads(bounded_file(directory / "target.json"))
        if record.get("run_id") != run_id or record.get("target_digest") != digest(target):
            raise ValueError("run identity or frozen target differs")
        return cast(dict[str, Any], record)

    def update(self, run_id: str, **fields: Any) -> dict[str, Any]:
        record = self.get(run_id)
        if record["status"] != "running":
            raise ValueError("terminal run records are immutable")
        if {
            "run_id",
            "target_id",
            "target_digest",
            "created_at",
            "operation",
            "metadata",
        } & fields.keys():
            raise ValueError("immutable run identity cannot be updated")
        if fields.get("status", "running") not in {"running", "complete", "failed"}:
            raise ValueError("invalid run status")
        record.update(fields, updated_at=datetime.now(UTC).isoformat())
        atomic_json(self.path(run_id) / "run.json", record)
        return record

    def list(self) -> list[dict[str, Any]]:
        return [self.get(path.name) for path in sorted(self.root.iterdir())]
