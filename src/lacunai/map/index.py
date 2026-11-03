"""Persistent checkpoint-bound maps and a dependency-free SVG export."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, cast
from xml.sax.saxutils import escape


def _safe(path: Path) -> Path:
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("linked map path refused")
    return path


def _identity(record: dict[str, Any]) -> tuple[str, str, str]:
    target, checkpoint, kind = (
        record.get(key) for key in ("target_id", "checkpoint_fingerprint", "kind")
    )
    if not isinstance(target, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", target):
        raise ValueError("invalid map target ID")
    if not isinstance(checkpoint, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint):
        raise ValueError("invalid checkpoint fingerprint")
    if kind not in {"fingerprint", "localization"}:
        raise ValueError("invalid map record kind")
    return target, checkpoint, cast(str, kind)


class MapIndex:
    """SQLite writes are transactional; maps are never reused across checkpoint hashes."""

    def __init__(self, workspace: Path):
        self.workspace = _safe(Path(workspace))
        self.workspace.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = _safe(self.workspace / "maps.sqlite3")
        if not self.path.exists():
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS maps (target TEXT NOT NULL, "
                "checkpoint TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, "
                "PRIMARY KEY(target, checkpoint, kind))"
            )

    def _connect(self) -> sqlite3.Connection:
        _safe(self.path)
        for suffix in ("-journal", "-wal", "-shm"):
            _safe(Path(str(self.path) + suffix))
        return sqlite3.connect(self.path)

    def put(self, record: dict[str, Any]) -> dict[str, str]:
        target, checkpoint, kind = _identity(record)
        payload = json.dumps(record, sort_keys=True, allow_nan=False)
        if len(payload.encode()) > 32 * 1024**2:
            raise ValueError("map record exceeds byte limit")
        with self._connect() as db:
            existing = db.execute(
                "SELECT payload FROM maps WHERE target=? AND checkpoint=? AND kind=?",
                (target, checkpoint, kind),
            ).fetchone()
            if existing is not None and existing[0] != payload:
                raise ValueError("different map already stored for this target/checkpoint/kind")
            db.execute(
                "INSERT OR IGNORE INTO maps VALUES (?, ?, ?, ?)",
                (target, checkpoint, kind, payload),
            )
        return {"target_id": target, "checkpoint_fingerprint": checkpoint, "kind": kind}

    def get(
        self, target_id: str, checkpoint_fingerprint: str, kind: str = "fingerprint"
    ) -> dict[str, Any]:
        identity = _identity(
            {"target_id": target_id, "checkpoint_fingerprint": checkpoint_fingerprint, "kind": kind}
        )
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM maps WHERE target=? AND checkpoint=? AND kind=?", identity
            ).fetchone()
            if row is None:
                raise ValueError(
                    "no map for exact target/checkpoint/kind; stale maps are not reusable"
                )
        record = cast(dict[str, Any], json.loads(row[0]))
        if _identity(record) != identity:
            raise ValueError("map index payload identity differs")
        return record

    def list(self, limit: int = 100) -> list[dict[str, str]]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("map query limit must be between 1 and 1000")
        with self._connect() as db:
            rows = db.execute(
                "SELECT target, checkpoint, kind FROM maps "
                "ORDER BY target, checkpoint, kind LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            dict(zip(("target_id", "checkpoint_fingerprint", "kind"), row, strict=True))
            for row in rows
        ]

    def export(
        self,
        target_id: str,
        checkpoint_fingerprint: str,
        destination: Path,
        kind: str = "fingerprint",
    ) -> dict[str, str]:
        record = self.get(target_id, checkpoint_fingerprint, kind)
        destination = _safe(Path(destination))
        destination.mkdir(mode=0o700, parents=True, exist_ok=True)
        outputs = {
            "json": destination / f"{target_id}-{kind}.json",
            "heatmap": destination / f"{target_id}-{kind}.svg",
        }
        for path in outputs.values():
            _safe(path)
            if path.exists():
                raise ValueError("map export destination already exists")
        content = {
            "json": json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
            "heatmap": heatmap(record),
        }
        for name, path in outputs.items():
            with path.open("x") as stream:
                stream.write(content[name])
                stream.flush()
                os.fsync(stream.fileno())
        return {name: str(path) for name, path in outputs.items()}


def heatmap(record: dict[str, Any]) -> str:
    _identity(record)
    if record["kind"] == "fingerprint":
        rows = [
            (
                int(layer),
                math.fsum(abs(v) for v in value["target_mean"]) / len(value["target_mean"]),
            )
            for layer, value in record["layers"].items()
        ]
        metric = "Mean absolute prompt-pooled feature activation"
    else:
        rows = [
            (row["layer"], row["mean_effect"])
            for row in record["layers"]
            if row["mean_effect"] is not None
        ]
        metric = "Mean identified causal effect on log probability"
    if any(not math.isfinite(value) for _, value in rows):
        raise ValueError("non-finite heatmap metric")
    rows.sort()
    scale = max((abs(value) for _, value in rows), default=0.0)
    title = escape(f"{record['target_id']}: {metric}")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 640 {max(100, len(rows) * 30 + 90)}" '
        'role="img" aria-labelledby="title description">',
        f'<title id="title">{title}</title>',
        '<desc id="description">Per-layer values. Bar length encodes absolute magnitude; '
        "blue is negative and orange is nonnegative. This is not proof of memory location.</desc>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="12" y="24" font-family="sans-serif" font-size="13">{title}</text>',
    ]
    if not rows:
        parts.append('<text x="12" y="60">No identified layer values</text>')
    for index, (layer, value) in enumerate(rows):
        y = 44 + index * 30
        width = abs(value) / scale * 360 if scale else 0.0
        color = "#245b91" if value < 0 else "#b65d22"
        parts.extend(
            [
                f'<text x="12" y="{y + 15}" font-family="sans-serif" '
                f'font-size="12">Layer {layer}</text>',
                f'<rect x="95" y="{y}" width="{width:.3f}" height="20" fill="{color}"/>',
                f'<text x="470" y="{y + 15}" font-family="monospace" '
                f'font-size="12">{value:.6g}</text>',
            ]
        )
    return "\n".join([*parts, "</svg>"]) + "\n"
