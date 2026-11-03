# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Admission for a frozen, operator-approved experimental contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from lacunai.paths import KeepList

APPROVED_V1_SHA256 = "e76284961549c62ea53fb367e3d9a886ccedac0aad013daa2079f6c89ac7bc1f"


class ExperimentRefused(ValueError):
    """Admission failed before a model can be loaded."""


def load_contract(path: Path, *, enable: bool = False) -> dict[str, Any]:
    """Load only the approved bytes, with an explicit execution opt-in."""
    if not enable:
        raise ExperimentRefused("experimental execution is disabled; explicit opt-in required")
    if path.is_symlink() or not path.is_file():
        raise ExperimentRefused("contract must be a regular, non-symlink file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != APPROVED_V1_SHA256:
        raise ExperimentRefused("contract differs from the approved Chinese-literacy v1 digest")
    value: dict[str, Any] = json.loads(raw)
    if value["enabled_by_default"] or value["promotion"] != "disabled":
        raise ExperimentRefused("default execution or promotion is forbidden")
    return value


def admit_root(root: Path) -> Path:
    """Reject redirected roots and refuse to treat source models as writable outputs."""
    if not root.is_absolute() or not root.is_dir():
        raise ExperimentRefused("experiment root must be an existing absolute directory")
    for part in (root, *root.parents):
        if part.is_symlink():
            raise ExperimentRefused("symlink in experiment root")
    if root.name != "neurailyzer-pilot-20261102":
        raise ExperimentRefused("root does not name the approved isolated pilot directory")
    return root.resolve()


def output_path(root: Path, relative: str) -> Path:
    """Apply the existing keep-list protection to all candidate output paths."""
    candidate = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ExperimentRefused("output path escapes the experiment")
    protected = KeepList(paths=(root / "models", root / "code", root / "datasets"))
    if protected.protects(candidate) or protected.shelters(candidate):
        raise ExperimentRefused("output path names protected originals or frozen evidence")
    for part in (candidate, *candidate.parents):
        if part.is_symlink():
            raise ExperimentRefused("output path contains a symlink")
        if part == root:
            break
    return candidate
