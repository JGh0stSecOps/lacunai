"""Read-only validation of a human-frozen, current-session supplied edit specification."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .experimental import spec as admission
from .experimental.editor import parse_spec
from .experimental.spec import ExperimentRefused


def _read_bounded(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ExperimentRefused("input must be a regular, non-symlink file")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ExperimentRefused("input exceeds validation byte limit")
    return raw


def _require_digest(raw: bytes, expected: str) -> str:
    actual = hashlib.sha256(raw).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or actual != expected:
        raise ExperimentRefused("specification differs from expected SHA-256")
    return actual


def _require_enabled(enable: bool) -> None:
    if not enable:
        raise ExperimentRefused("specification validation disabled; explicit opt-in required")


def validate_supplied_spec(
    specification: Path,
    diagnostics: Path,
    contract_path: Path,
    *,
    expected_sha256: str,
    enable: bool = False,
) -> dict[str, Any]:
    """Return schema/hash evidence only; checkpoint and behavior checks require a separate run."""
    _require_enabled(enable)
    contract = admission.load_contract(contract_path, enable=True)
    raw = _read_bounded(specification, contract["resources"]["editor_output_bytes"])
    digest = _require_digest(raw, expected_sha256)
    diagnostic_raw = _read_bounded(diagnostics, 1 << 20)
    diagnostic_value = json.loads(diagnostic_raw)
    selected = parse_spec(raw.decode("utf-8"), diagnostic_value, contract)
    return {
        "phase": "specification_validation",
        "status": "validated",
        "scope": "schema_and_input_hashes_only",
        "execution_performed": False,
        "approval_granted": False,
        "checkpoint_validated": False,
        "specification_sha256": digest,
        "contract_sha256": admission.APPROVED_V1_SHA256,
        "diagnostics_sha256": hashlib.sha256(diagnostic_raw).hexdigest(),
        "source_fingerprint": selected["source_fingerprint"],
        "operation": selected["operation"],
        "tensor_count": len(selected["edits"]),
        "column_count": sum(len(item["column_ids"]) for item in selected["edits"]),
    }
