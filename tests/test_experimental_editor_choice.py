# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Finite editor-choice contract tests; no mocked experiment outcomes."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from lacunai.experimental import editor
from lacunai.experimental.tensors import TensorError


@pytest.fixture
def policy() -> dict[str, Any]:
    return {
        "edits": {
            "allowed_tensors": [
                f"model.layers.{layer}.mlp.down_proj.weight" for layer in range(16, 24)
            ],
            "expected_shape": [896, 4864],
            "operation": "attenuate_diagnostic_columns",
            "max_tensors": 2,
            "max_columns_per_tensor": 16,
            "strength_presets": {"low": "0.10", "medium": "0.25", "high": "0.50"},
        },
        "resources": {"editor_calls": 3, "editor_output_bytes": 16384},
    }


@pytest.fixture
def diagnostics(policy: Any) -> dict[str, Any]:
    return {
        "source_fingerprint": "immutable-checkpoint-digest",
        "columns": [
            {"id": f"l{layer}c{column}", "tensor": name, "column": column, "score": 0.8}
            for layer, name in zip(range(16, 24), policy["edits"]["allowed_tensors"], strict=True)
            for column in range(5)
        ],
    }


def test_prompt_and_vocabulary_have_identical_bounded_visibility(
    policy: Any, diagnostics: Any
) -> None:
    prompt, choices = editor.build_choice_prompt(diagnostics, [], policy, 1)
    assert len(choices) == 96
    payload = json.loads(prompt.split("Development evidence follows:\n", 1)[1])
    visible = {column["id"] for item in payload["diagnostics"] for column in item["columns"]}
    assert len(visible) == 32
    assert all(len(item["columns"]) == 4 for item in payload["diagnostics"])
    assert {json.loads(choice)["column_id"] for choice in choices} == visible
    assert "l16c4" not in prompt
    assert "source_fingerprint" not in prompt
    for choice in choices:
        spec = editor.compile_choice(choice, diagnostics, policy)
        selected = json.loads(choice)
        assert spec["source_fingerprint"] == diagnostics["source_fingerprint"]
        assert spec["operation"] == "attenuate_diagnostic_columns"
        assert spec["edits"][0]["column_ids"] == [selected["column_id"]]
        assert spec["edits"][0]["strength"] == selected["strength"]
        assert spec["rationale"] == (
            "compiled from constrained editor choice; no free-form rationale collected"
        )


def test_all_evidence_is_checked_even_beyond_display_budget(policy: Any, diagnostics: Any) -> None:
    evidence = [{"id": "dev-1"}, {"id": "dev-2"}, {"id": "heldout-3", "answer": "secret"}]
    with pytest.raises(TensorError, match="development-tagged"):
        editor.build_choice_prompt(diagnostics, evidence, policy, 1)


@pytest.mark.parametrize("attempt", [0, 4, True])
def test_attempt_budget_is_not_bypassed(policy: Any, diagnostics: Any, attempt: Any) -> None:
    with pytest.raises(TensorError, match="budget"):
        editor.build_choice_prompt(diagnostics, [], policy, attempt)


@pytest.mark.parametrize(
    "raw",
    [
        '{"column_id":"l16c0","strength":"low","python":"exec(anything)"}',
        '{"column_id":"l16c0","strength":0.1}',
        '{"column_id":"l16c4","strength":"low"}',
        '{"column_id":"l16c0","column_id":"l16c1","strength":"low"}',
        '{"column_id":"l16c0","strength":"low"} trailing',
        "[]",
        '{"column_id":null,"strength":"low"}',
        '{"column_id":"l16c0","strength":NaN}',
        '{"column_id":"l16c0","strength":"low","source_fingerprint":"attacker"}',
    ],
)
def test_invalid_choice_cannot_be_compiled(policy: Any, diagnostics: Any, raw: str) -> None:
    with pytest.raises(TensorError):
        editor.compile_choice(raw, diagnostics, policy)


@pytest.mark.parametrize("mutation", ["duplicate", "injection", "nonfinite", "range", "empty"])
def test_diagnostic_constraints(policy: Any, diagnostics: Any, mutation: str) -> None:
    bad = copy.deepcopy(diagnostics)
    if mutation == "duplicate":
        bad["columns"][1]["id"] = bad["columns"][0]["id"]
    elif mutation == "injection":
        bad["columns"][0]["id"] = 'l16c0"},"python":"run"'
    elif mutation == "nonfinite":
        bad["columns"][0]["score"] = float("inf")
    elif mutation == "range":
        bad["columns"][0]["column"] = 4864
    else:
        bad["columns"] = []
    with pytest.raises(TensorError):
        editor.compile_choice('{"column_id":"l16c0","strength":"low"}', bad, policy)


def test_finite_trie_rejects_every_unlisted_branch(policy: Any, diagnostics: Any) -> None:
    _, choices = editor.build_choice_prompt(diagnostics, [], policy, 1)
    paths = [[*choice.encode(), 0] for choice in choices]
    trie = editor.TokenTrie(paths, eos=0, cap=256)
    for path in paths:
        for index, token in enumerate(path):
            prefix = path[:index]
            assert token in trie.allowed(prefix)
            with pytest.raises(TensorError, match="grammar"):
                trie.allowed([*prefix, 999])
        assert tuple(path) in trie.complete
        with pytest.raises(TensorError, match="grammar"):
            trie.allowed(path)
    assert len(trie.complete) == 96
    assert any(len(allowed) > 1 for allowed in trie.transitions.values())


@pytest.mark.parametrize(
    "paths,cap",
    [
        ([], 3),
        ([[1, 2]], 3),
        ([[1, 0, 2, 0]], 4),
        ([[1, 2, 0]], 2),
        ([[1, 0], [1, 0]], 2),
        ([[True, 0]], 2),
        ([[-1, 0]], 2),
    ],
)
def test_invalid_token_grammar_rejected(paths: Any, cap: int) -> None:
    with pytest.raises(TensorError):
        editor.TokenTrie(paths, eos=0, cap=cap)


def test_spec_guard_removal_is_red_then_restored_green(
    policy: Any, diagnostics: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = copy.deepcopy(diagnostics)
    bad["columns"][0]["column"] = 4864

    def check() -> None:
        with pytest.raises(TensorError, match="out of range"):
            editor.compile_choice('{"column_id":"l16c0","strength":"low"}', bad, policy)

    check()
    with monkeypatch.context() as plant:
        plant.setattr(editor, "validate_edit_spec", lambda spec, *_: spec)
        with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
            check()
    check()


def test_actual_processor_preserves_logits_and_rejects_nonfinite() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    result = editor.constrained_numeric_selftest()
    assert result["status"] == "passed"
    assert result["model_loaded"] is False
    assert result["editor_calls"] == 0
