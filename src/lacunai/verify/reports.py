"""Detached Ed25519 attestations over reports and frozen supporting artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from lacunai.core.spec import bounded_file, canonical
from lacunai.experimental.artifacts import secure_path, sha256_file

_LIMIT = 32 * 1024 * 1024


def _progress(event: dict[str, Any]) -> None:
    print(json.dumps(event), file=sys.stderr, flush=True)


def _write_new(path: Path, value: bytes) -> None:
    secure_path(path.parent, path.name)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    if bounded_file(path, max(len(value), 1)) != value:
        raise ValueError("new artifact readback differs")


def _json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _keys(key_dir: Path) -> tuple[Any, bytes]:
    ed = import_module("cryptography.hazmat.primitives.asymmetric.ed25519")
    serialization = import_module("cryptography.hazmat.primitives.serialization")
    key_dir = key_dir.absolute()
    if any(p.is_symlink() for p in (key_dir, *key_dir.parents)):
        raise ValueError("signing key directory is redirected")
    # Keys must not be emitted into this or any ancestor source checkout.
    if any((p / ".git").exists() for p in (key_dir, *key_dir.parents)):
        raise ValueError("signing keys must live outside source repositories")
    key_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    private, public = key_dir / "private.key", key_dir / "public.key"
    if not private.exists():
        key = ed.Ed25519PrivateKey.generate()
        encoded = key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        _write_new(private, encoded)
    if private.stat().st_mode & 0o077:
        raise ValueError("private signing key permissions are not private")
    key = ed.Ed25519PrivateKey.from_private_bytes(bounded_file(private, 32))
    public_bytes = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    if not public.exists():
        _write_new(public, public_bytes)
    if bounded_file(public, 32) != public_bytes:
        raise ValueError("signing public key does not match private key")
    return key, public_bytes


def _markdown(record: dict[str, Any]) -> str:
    comparison = record.get("comparison", record.get("verification", {}))
    if not isinstance(comparison, dict):
        comparison = {}
    lines = [
        "# LacunAI run report",
        "",
        f"Run: `{record.get('run_id', 'unknown')}`",
        f"Status: **{record['status']}**",
        f"Target: `{record.get('target_id', 'unknown')}`",
        f"Method: `{record.get('method', record.get('operation', 'unknown'))}`",
        f"Snapshot: `{record.get('snapshot_id', 'unavailable')}`",
        f"Verdict: **{comparison.get('verdict', record.get('verdict', 'inconclusive'))}**",
        "",
        "## Before and after",
        "",
    ]
    for label in ("before", "after"):
        value = comparison.get(label, record.get(label))
        lines += [
            f"### {label.title()}",
            "",
            "```json",
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
            "```",
            "",
        ]
    lines += [
        "## Limits",
        "",
        "This report describes reduced recoverability under the stated probes, not proof "
        "of information absence. A refusal, wrong answer or changed fingerprint does not "
        "establish erasure. Originals and backups still retain the original information.",
        "",
        "The signature attests these bytes against a separately trusted public key. "
        "It does not certify scientific correctness or identify an author by itself.",
        "",
    ]
    return "\n".join(lines)


def write_signed_report(run_dir: Path, record: dict[str, Any], key_dir: Path) -> dict[str, Any]:
    """Sign once; freeze mutable operational metadata for later restore events."""
    run_dir = run_dir.absolute()
    secure_path(run_dir, "signature.json")
    if record.get("status") not in {"complete", "failed"}:
        raise ValueError("cannot sign a nonterminal run")
    if not isinstance(record.get("run_id"), str) or not record.get("target_id"):
        raise ValueError("report requires run and target identity")
    if any(
        (run_dir / name).exists()
        for name in ("report.json", "report.md", "signature.json", "report-evidence")
    ):
        raise ValueError("reports are immutable; existing or partial report refused")
    _progress({"phase": "report_signing", "state": "active"})
    try:
        key, public = _keys(key_dir)
        evidence_dir = run_dir / "report-evidence"
        evidence_dir.mkdir(mode=0o700)
        copies = {}
        for name in ("run.json", "snapshot.json"):
            source = secure_path(run_dir, name)
            if source.exists():
                _write_new(evidence_dir / name, bounded_file(source, _LIMIT))
                copies[name] = f"report-evidence/{name}"
        _write_new(run_dir / "report.json", _json(record))
        _write_new(run_dir / "report.md", _markdown(record).encode())
        files: dict[str, str] = {}
        for path in sorted(run_dir.rglob("*")):
            relative = str(path.relative_to(run_dir))
            secure_path(run_dir, relative)
            if path.is_dir():
                continue
            if relative in {"run.json", "snapshot.json", "signature.json"}:
                continue
            if len(files) >= 4096:
                raise ValueError("too many evidence files")
            files[relative] = sha256_file(path, _progress)
        envelope = {
            "schema_version": 1,
            "algorithm": "Ed25519",
            "run_id": record["run_id"],
            "files_sha256": files,
            "frozen_metadata_copies": copies,
            "public_key_sha256": hashlib.sha256(public).hexdigest(),
        }
        signature = base64.b64encode(key.sign(canonical(envelope))).decode("ascii")
        _write_new(
            run_dir / "signature.json", _json({"envelope": envelope, "signature": signature})
        )
        verified = verify_report(run_dir, key_dir / "public.key")
    except Exception as exc:
        _progress({"phase": "report_signing", "state": "failed", "error": str(exc)})
        raise
    _progress({"phase": "report_signing", "state": "complete"})
    return {
        **verified,
        "signature_path": str(run_dir / "signature.json"),
        "public_key_path": str(key_dir / "public.key"),
    }


def verify_report(run_dir: Path, trusted_public_key: Path) -> dict[str, Any]:
    """No embedded-key trust: caller must choose the trusted key independently."""
    ed = import_module("cryptography.hazmat.primitives.asymmetric.ed25519")
    run_dir = run_dir.absolute()
    secure_path(run_dir, "signature.json")
    public = bounded_file(trusted_public_key, 32)
    signed = json.loads(bounded_file(run_dir / "signature.json", _LIMIT))
    envelope = signed["envelope"]
    if (
        envelope.get("schema_version") != 1
        or envelope.get("algorithm") != "Ed25519"
        or envelope.get("public_key_sha256") != hashlib.sha256(public).hexdigest()
    ):
        raise ValueError("signature envelope or trusted key differs")
    signature = base64.b64decode(signed["signature"], validate=True)
    ed.Ed25519PublicKey.from_public_bytes(public).verify(signature, canonical(envelope))
    files = envelope.get("files_sha256")
    if not isinstance(files, dict) or not {"report.json", "report.md"} <= files.keys():
        raise ValueError("signed report inventory is incomplete")
    for relative, expected in files.items():
        path = secure_path(run_dir, relative)
        if sha256_file(path, _progress) != expected:
            raise ValueError(f"signed artifact hash differs: {relative}")
    record = json.loads(bounded_file(run_dir / "report.json", _LIMIT))
    if record.get("run_id") != envelope["run_id"] or record.get("status") not in {
        "complete",
        "failed",
    }:
        raise ValueError("signed report identity or terminal state differs")
    return {
        "verified": True,
        "run_id": envelope["run_id"],
        "files_verified": len(files),
        "public_key_sha256": envelope["public_key_sha256"],
        "scientific_correctness_certified": False,
        "envelope": cast(dict[str, Any], envelope),
    }
