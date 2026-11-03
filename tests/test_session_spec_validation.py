"""Exercise the real read-only CLI and guards without an ML installation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from lacunai import validation
from lacunai.cli import app
from lacunai.experimental import spec as admission
from lacunai.experimental.spec import ExperimentRefused
from lacunai.experimental.tensors import TensorError


@pytest.fixture
def frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    contract = {
        "enabled_by_default": False,
        "promotion": "disabled",
        "resources": {"editor_output_bytes": 16384},
        "edits": {
            "operation": "attenuate_diagnostic_columns",
            "allowed_tensors": ["linear.weight"],
            "expected_shape": [2, 4],
            "max_tensors": 1,
            "max_columns_per_tensor": 2,
            "strength_presets": {"low": "0.01"},
        },
    }
    diagnostic = {
        "source_fingerprint": "fixture-source-fingerprint",
        "columns": [{"id": "l0c1", "tensor": "linear.weight", "column": 1, "score": 0.3}],
    }
    proposal = {
        "operation": "attenuate_diagnostic_columns",
        "source_fingerprint": diagnostic["source_fingerprint"],
        "rationale": "fixture development evidence only",
        "edits": [{"tensor": "linear.weight", "column_ids": ["l0c1"], "strength": "low"}],
    }
    paths = {}
    for name, content in (("contract", contract), ("diagnostics", diagnostic), ("spec", proposal)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(content))
        paths[name] = path
    # Bind the real loader to these fixture bytes, not a permissive loader stub.
    monkeypatch.setattr(
        admission, "APPROVED_V1_SHA256", hashlib.sha256(paths["contract"].read_bytes()).hexdigest()
    )
    return {"paths": paths, "proposal": proposal, "expected": _digest(paths["spec"])}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(frozen: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    paths = frozen["paths"]
    return validation.validate_supplied_spec(
        paths["spec"],
        paths["diagnostics"],
        paths["contract"],
        expected_sha256=kwargs.pop("expected_sha256", frozen["expected"]),
        enable=kwargs.pop("enable", True),
        **kwargs,
    )


def test_cli_validates_frozen_spec_without_loading_model_or_changing_inputs(frozen: Any) -> None:
    before = {key: _digest(path) for key, path in frozen["paths"].items()}
    paths = frozen["paths"]
    result = CliRunner().invoke(
        app,
        [
            "validate-spec",
            str(paths["spec"]),
            "--diagnostics",
            str(paths["diagnostics"]),
            "--contract",
            str(paths["contract"]),
            "--expected-sha256",
            frozen["expected"],
            "--enable-experimental",
        ],
    )
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.output.splitlines()]
    assert events[0]["status"] == "active"
    receipt = events[-1]
    assert receipt["status"] == "validated"
    assert receipt["scope"] == "schema_and_input_hashes_only"
    assert receipt["execution_performed"] is False
    assert receipt["checkpoint_validated"] is False
    assert receipt["approval_granted"] is False
    assert receipt["specification_sha256"] == frozen["expected"]
    assert receipt["tensor_count"] == 1 and receipt["column_count"] == 1
    assert before == {key: _digest(path) for key, path in paths.items()}


def test_default_off_precedes_all_file_reads(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ExperimentRefused, match="disabled"):
        validation.validate_supplied_spec(missing, missing, missing, expected_sha256="0" * 64)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("expected", ["0" * 64, "abc", "F" * 64])
def test_exact_digest_is_required(frozen: Any, expected: str) -> None:
    with pytest.raises(ExperimentRefused, match="SHA-256"):
        _validate(frozen, expected_sha256=expected)


def test_tampered_contract_is_rejected_even_when_spec_hash_matches(frozen: Any) -> None:
    frozen["paths"]["contract"].write_text("{}")
    with pytest.raises(ExperimentRefused, match="digest"):
        _validate(frozen)


@pytest.mark.parametrize("case", ["code", "tensor", "column", "strength", "source", "operation"])
def test_cli_does_not_accept_arbitrary_edits_or_code(frozen: Any, case: str) -> None:
    proposal = frozen["proposal"]
    if case == "code":
        proposal["code"] = "print('must never execute')"
    elif case == "source":
        proposal["source_fingerprint"] = "different-source"
    elif case == "operation":
        proposal["operation"] = "execute_python"
    elif case == "column":
        proposal["edits"][0]["column_ids"] = ["invented"]
    else:
        proposal["edits"][0][case] = "unapproved"
    path = frozen["paths"]["spec"]
    path.write_text(json.dumps(proposal))
    with pytest.raises(TensorError):
        _validate(frozen, expected_sha256=_digest(path))


@pytest.mark.parametrize("case", ["bytes", "link"])
def test_inputs_are_bounded_and_symlinks_are_rejected(
    frozen: Any, tmp_path: Path, case: str
) -> None:
    if case == "bytes":
        with pytest.raises(ExperimentRefused, match="byte limit"):
            validation._read_bounded(frozen["paths"]["spec"], 1)
    else:
        linked = tmp_path / "linked-spec"
        linked.symlink_to(frozen["paths"]["spec"])
        with pytest.raises(ExperimentRefused, match="non-symlink"):
            validation._read_bounded(linked, 16384)


def test_cli_refusal_ends_active_status(frozen: Any) -> None:
    paths = frozen["paths"]
    result = CliRunner().invoke(
        app,
        [
            "validate-spec",
            str(paths["spec"]),
            "--diagnostics",
            str(paths["diagnostics"]),
            "--contract",
            str(paths["contract"]),
            "--expected-sha256",
            frozen["expected"],
        ],
    )
    assert result.exit_code == 2
    events = [json.loads(line) for line in result.output.splitlines()]
    assert [event["status"] for event in events] == ["active", "refused"]


@pytest.mark.parametrize("removed", ["enabled", "digest", "grammar"])
def test_guard_removal_is_red_and_restoration_green(
    frozen: Any, monkeypatch: pytest.MonkeyPatch, removed: str
) -> None:
    def assert_refusal() -> None:
        with pytest.raises(ValueError):
            if removed == "enabled":
                _validate(frozen, enable=False)
            elif removed == "digest":
                _validate(frozen, expected_sha256="0" * 64)
            else:
                _validate(frozen, expected_sha256=_digest(frozen["paths"]["spec"]))

    if removed == "grammar":
        frozen["proposal"]["code"] = "arbitrary code must remain data"
        frozen["paths"]["spec"].write_text(json.dumps(frozen["proposal"]))
    assert_refusal()
    with monkeypatch.context() as patch:
        if removed == "enabled":
            patch.setattr(validation, "_require_enabled", lambda enable: None)
        elif removed == "digest":
            patch.setattr(validation, "_require_digest", lambda raw, expected: "0" * 64)
        else:
            patch.setattr(
                validation, "parse_spec", lambda raw, diagnostics, contract: json.loads(raw)
            )
        with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
            assert_refusal()
    assert_refusal()


def test_help_and_import_do_not_load_ml_or_state_hygiene() -> None:
    code = """
import importlib.abc, json, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'neurailyzer'}:
            raise AssertionError('unwanted runtime import: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['lacunai', '--help']
try:
    runpy.run_module('lacunai', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
print(json.dumps({'ml_runtime_imported': False, 'state_hygiene_imported': False}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "validate-spec" in result.stdout
    assert json.loads(result.stdout.splitlines()[-1])["ml_runtime_imported"] is False
