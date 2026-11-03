# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Specification tests run without ML dependencies; tiny numeric tests opt in."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from lacunai.experimental import tensors
from lacunai.experimental.editor import build_prompt, parse_spec


@pytest.fixture
def policy() -> dict[str, Any]:
    return {
        "edits": {
            "allowed_tensors": ["linear.weight"],
            "expected_shape": [2, 4],
            "operation": "attenuate_diagnostic_columns",
            "max_tensors": 1,
            "max_columns_per_tensor": 2,
            "strength_presets": {"low": "0.01", "high": "0.50"},
            "max_tensor_relative_frobenius": "0.02",
            "max_model_relative_frobenius": "0.01",
        },
        "resources": {"editor_calls": 3, "editor_output_bytes": 16384},
        "target": {"revision": "synthetic-unit-test"},
    }


@pytest.fixture
def diagnostics() -> dict[str, Any]:
    return {
        "source_fingerprint": "source-digest",
        "columns": [{"id": "l0c1", "tensor": "linear.weight", "column": 1, "score": 0.3}],
    }


@pytest.fixture
def spec() -> dict[str, Any]:
    return {
        "operation": "attenuate_diagnostic_columns",
        "source_fingerprint": "source-digest",
        "rationale": "development-only association",
        "edits": [{"tensor": "linear.weight", "column_ids": ["l0c1"], "strength": "low"}],
    }


def test_accepts_only_complete_declarative_spec(policy: Any, diagnostics: Any, spec: Any) -> None:
    assert parse_spec(json.dumps(spec), diagnostics, policy) == spec
    raw = json.dumps(spec)
    for invalid in ["```json\n" + raw + "\n```", raw + " trailing text", raw + raw, "eval('1+1')"]:
        with pytest.raises(tensors.TensorError):
            parse_spec(invalid, diagnostics, policy)


@pytest.mark.parametrize(
    "mutation", ["tensor", "column", "strength", "source", "operation", "code", "duplicates"]
)
def test_rejects_unapproved_proposals(
    policy: Any, diagnostics: Any, spec: Any, mutation: str
) -> None:
    bad = copy.deepcopy(spec)
    if mutation == "tensor":
        bad["edits"][0]["tensor"] = "model.embed_tokens.weight"
    elif mutation == "column":
        bad["edits"][0]["column_ids"] = ["invented"]
    elif mutation == "strength":
        bad["edits"][0]["strength"] = 0.3
    elif mutation == "source":
        bad["source_fingerprint"] = "other-model"
    elif mutation == "operation":
        bad["operation"] = "execute_python"
    elif mutation == "code":
        bad["python"] = "raise RuntimeError()"
    else:
        bad["edits"][0]["column_ids"] = ["l0c1", "l0c1"]
    with pytest.raises(tensors.TensorError):
        tensors.validate_edit_spec(bad, diagnostics, policy)


def test_duplicate_json_keys_rejected(policy: Any, diagnostics: Any, spec: Any) -> None:
    raw = json.dumps(spec).replace('"rationale":', '"operation":"bad","rationale":')
    with pytest.raises(tensors.TensorError, match="duplicate JSON key"):
        parse_spec(raw, diagnostics, policy)


def test_editor_receives_only_development_examples(policy: Any, diagnostics: Any) -> None:
    with pytest.raises(tensors.TensorError, match="development-tagged"):
        build_prompt(diagnostics, [{"id": "heldout-1", "answer": "secret answer"}], policy, 1)
    text = build_prompt(
        diagnostics, [{"id": "dev-1", "prompts": ["Visible"], "answer": "Known"}], policy, 1
    )
    assert "Visible" in text
    assert "compute arithmetic" in text
    with pytest.raises(tensors.TensorError, match="budget"):
        build_prompt(diagnostics, [], policy, 4)


def test_schema_removal_turns_assertion_red(
    policy: Any, diagnostics: Any, spec: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = copy.deepcopy(spec)
    bad["edits"][0]["column_ids"] = ["invented"]

    def real_assertion() -> None:
        with pytest.raises(tensors.TensorError):
            tensors.validate_edit_spec(bad, diagnostics, policy)

    real_assertion()
    with monkeypatch.context() as mutation:
        mutation.setattr(tensors, "validate_edit_spec", lambda value, *_: value)
        with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
            real_assertion()
    real_assertion()


def test_actual_tensor_selftest_and_removal_plants() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    result = tensors.numeric_selftest()
    assert result["status"] == "passed"
    assert {p["guard"] for p in result["removal_plants"]} == {
        "delta_bounds",
        "edit_schema",
        "write_budget",
    }
    assert result["numerical_smoke"]["gradient"] == [1.0, 3.0]


def test_derivative_roundtrip_source_guard_and_original_preservation(
    tmp_path: Any, policy: Any
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    model = torch.nn.Module()
    model.add_module("linear", torch.nn.Linear(4, 2, bias=False))
    model.register_parameter("untouched", torch.nn.Parameter(torch.tensor([4.0])))
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
    reference = tensors.capture_reference(model, policy)
    checkpoint = tmp_path / "no-op"
    manifest = tensors.save_delta(model, reference, checkpoint, policy)
    assert manifest["diff"]["changed_tensor_count"] == 0
    tensors.load_delta(model, checkpoint, reference, policy)
    assert tensors.validate_state(model, reference, policy)["changed_tensor_count"] == 0
    with pytest.raises(tensors.TensorError, match="already exists"):
        tensors.save_delta(model, reference, checkpoint, policy)
    saved_manifest = checkpoint / "manifest.json"
    bad = json.loads(saved_manifest.read_text())
    bad["source_fingerprint"] = "incorrect-base"
    saved_manifest.write_text(json.dumps(bad))
    with pytest.raises(tensors.TensorError, match="source hash"):
        tensors.load_delta(model, checkpoint, reference, policy)
    assert tensors.validate_state(model, reference, policy)["changed_tensor_count"] == 0


def test_write_budget_boundary_and_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    tensors.validate_write_budget(4, 6, 10)

    def assertion() -> None:
        with pytest.raises(tensors.TensorError, match="artifact budget"):
            tensors.validate_write_budget(4, 7, 10)

    assertion()
    with monkeypatch.context() as mutation:
        mutation.setattr(tensors, "validate_write_budget", lambda *_: None)
        with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
            assertion()
    assertion()
