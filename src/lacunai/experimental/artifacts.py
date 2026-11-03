# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Bounded, private experiment artifacts; never a wipe or restore operation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lacunai.paths import KeepList

Progress = Callable[[dict[str, Any]], None]


class ArtifactError(RuntimeError):
    """An artifact cannot be safely created or verified."""


def emit(event: dict[str, Any], progress: Progress | None = None) -> None:
    """Emit a phase or measured progress event immediately."""
    if progress is None:
        print(json.dumps(event, sort_keys=True), flush=True)
    else:
        progress(event)


def secure_path(root: Path, relative: str | Path) -> Path:
    """Reject traversal and every existing symlink in an output path."""
    root = root.absolute()
    part = Path(relative)
    if part.is_absolute() or not part.parts or any(p in ("..", ".") for p in part.parts):
        raise ArtifactError("artifact path must be a nonempty relative path without traversal")
    if root.is_symlink() or root.resolve() != root:
        raise ArtifactError("artifact root is redirected")
    candidate = root
    for component in part.parts:
        candidate /= component
        if candidate.is_symlink():
            raise ArtifactError("artifact path contains a symlink")
    if not candidate.is_relative_to(root):
        raise ArtifactError("artifact path escapes its root")
    return candidate


def private_directory(root: Path, relative: str | Path) -> Path:
    path = secure_path(root, relative)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir():
        raise ArtifactError("artifact directory is not a directory")
    return path


def sha256_file(path: Path, progress: Progress | None = None) -> str:
    """Stream SHA-256 in bounded chunks and report measured byte progress."""
    if path.is_symlink() or not path.is_file():
        raise ArtifactError("hash source must be a regular file, not a link")
    size = path.stat().st_size
    emit(
        {
            "phase": "hash",
            "label": path.name,
            "completed": 0,
            "total": size,
            "unit": "bytes",
            "status": "active",
        },
        progress,
    )
    digest = hashlib.sha256()
    completed = 0
    last = time.monotonic()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
                completed += len(chunk)
                if time.monotonic() - last >= 5:
                    emit(
                        {
                            "phase": "hash",
                            "label": path.name,
                            "completed": completed,
                            "total": size,
                            "unit": "bytes",
                            "status": "active",
                        },
                        progress,
                    )
                    last = time.monotonic()
        if completed != size:
            raise ArtifactError("hash source size changed while reading")
    except Exception:
        emit(
            {
                "phase": "hash",
                "label": path.name,
                "completed": completed,
                "total": size,
                "unit": "bytes",
                "status": "failed",
            },
            progress,
        )
        raise
    emit(
        {
            "phase": "hash",
            "label": path.name,
            "completed": completed,
            "total": size,
            "unit": "bytes",
            "status": "complete",
        },
        progress,
    )
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Atomically replace a private state record, without following destination links."""
    if path.is_symlink() or path.parent.is_symlink():
        raise ArtifactError("refusing redirected JSON artifact")
    if path.exists() and not path.is_file():
        raise ArtifactError("JSON artifact destination is not a regular file")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temporary, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def tree_bytes(root: Path) -> int:
    """Count stored regular-file and link bytes without walking linked directories."""
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                total += info.st_size
    return total


def tree_manifest(root: Path, progress: Progress | None = None) -> list[dict[str, Any]]:
    """Hash regular files and record link spellings without following them."""
    records: list[dict[str, Any]] = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            if path.is_symlink():
                records.append(
                    {"path": path.relative_to(root).as_posix(), "symlink": os.readlink(path)}
                )
            elif path.is_file():
                records.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path, progress),
                    }
                )
    return sorted(records, key=lambda row: str(row["path"]))


def verify_originals(
    root: Path, records: list[dict[str, Any]], progress: Progress | None = None
) -> list[dict[str, Any]]:
    """Re-read every original and return positive receipts, or raise on any discrepancy."""
    keep = KeepList(paths=(root / "models",))
    checked: list[dict[str, Any]] = []
    for record in records:
        path = secure_path(root, str(record["path"]))
        if not keep.protects(path):
            raise ArtifactError("original ledger names a path outside protected model storage")
        digest = sha256_file(path, progress)
        if digest != record["sha256"] or path.stat().st_size != record["size"]:
            raise ArtifactError("original checkpoint artifact changed")
        if path.stat().st_mode & 0o222:
            raise ArtifactError("original checkpoint artifact is writable")
        checked.append({"path": record["path"], "sha256": digest, "verified": True})
    return checked


@dataclass
class Budget:
    """Persist conservative network reservations and admit bounded disk writes."""

    root: Path
    max_disk_bytes: int
    max_network_bytes: int
    deadline_epoch: int
    progress: Progress | None = None
    network_reserved_bytes: int = field(default=0, init=False)
    network_received_bytes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.root = self.root.absolute()
        if not self.root.is_dir() or self.root.is_symlink() or self.root.resolve() != self.root:
            raise ArtifactError("budget root must be an existing unredirected directory")
        ledger = secure_path(self.root, "receipts/network.json")
        if ledger.exists():
            previous = json.loads(ledger.read_text())
            self.network_reserved_bytes = int(previous["reserved_body_bytes"])
            self.network_received_bytes = int(previous["received_body_bytes"])
            if self.network_reserved_bytes < self.network_received_bytes:
                raise ArtifactError("network ledger is inconsistent")
        self.check()

    def check(self, extra_disk_bytes: int = 0) -> None:
        if time.time() >= self.deadline_epoch:
            raise ArtifactError("preparation deadline reached")
        if extra_disk_bytes < 0:
            raise ArtifactError("negative disk reservation")
        if tree_bytes(self.root) + extra_disk_bytes + (1 << 20) > self.max_disk_bytes:
            raise ArtifactError("aggregate experiment disk budget exceeded")
        if self.network_reserved_bytes > self.max_network_bytes:
            raise ArtifactError("network body-byte budget exceeded")

    def reserve_network(self, maximum_body_bytes: int) -> None:
        self.check()
        if maximum_body_bytes <= 0:
            raise ArtifactError("network reservation must be positive")
        if self.network_reserved_bytes + maximum_body_bytes > self.max_network_bytes:
            raise ArtifactError("network body-byte budget exceeded")
        self.network_reserved_bytes += maximum_body_bytes
        self._save()

    def received(self, size: int) -> None:
        self.network_received_bytes += size
        if self.network_received_bytes > self.network_reserved_bytes:
            raise ArtifactError("response exceeded its network reservation")
        self._save()

    def _save(self) -> None:
        atomic_json(
            secure_path(self.root, "receipts/network.json"),
            {
                "reserved_body_bytes": self.network_reserved_bytes,
                "received_body_bytes": self.network_received_bytes,
                "limit_body_bytes": self.max_network_bytes,
                "scope": "HTTP response bodies; TLS and HTTP header overhead unmeasured",
            },
        )
