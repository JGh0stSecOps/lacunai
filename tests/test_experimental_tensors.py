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


def test_low_memory_loader_contract(monkeypatch, tmp_path):
    from types import SimpleNamespace

    calls = []
    model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=False), to=lambda _: None, eval=lambda: None
    )

    def load(path, **kwargs):
        calls.append(kwargs)
        return model, {}

    fake = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *a, **kw: SimpleNamespace(eos_token_id=1)
        ),
    )
    monkeypatch.setattr(
        tensors,
        "import_module",
        lambda name: SimpleNamespace(float32="float32") if name == "torch" else fake,
    )
    monkeypatch.setattr(tensors, "configure_runtime", lambda _: None)
    tensors.load_target(tmp_path, {"editor": {"files": []}}, role="editor")
    assert calls[0]["low_cpu_mem_usage"] is True
    assert calls[0]["local_files_only"] is True
    assert calls[0]["trust_remote_code"] is False
    assert calls[0]["use_safetensors"] is True
    assert calls[0]["output_loading_info"] is True
    fake.AutoModelForCausalLM.from_pretrained = lambda *a, **kw: (model, {"missing_keys": ["bad"]})
    with pytest.raises(tensors.TensorError, match="strictly"):
        tensors.load_target(tmp_path, {"editor": {"files": []}}, role="editor")


@pytest.mark.parametrize("storage_dtype", ["float32", "bfloat16"])
def test_tiny_qwen_low_memory_equivalent_to_default(monkeypatch, tmp_path, storage_dtype):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    model = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    )
    model.to(dtype=getattr(torch, storage_dtype))
    model.save_pretrained(tmp_path, safe_serialization=True)
    del model
    files = [
        {"path": p.name, "size": p.stat().st_size, "sha256": tensors.file_hash(p)}
        for p in tmp_path.iterdir()
        if p.is_file()
    ]
    contract = {
        "editor": {"files": files},
        "runtime": {
            "seed": 42,
            "device": "cpu",
            "dtype": "float32",
            "torch_threads": 2,
            "interop_threads": 1,
        },
    }
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(eos_token_id=2),
    )
    loaded, _ = tensors.load_target(tmp_path, contract, role="editor")
    ids = torch.tensor([[1, 4, 7]])
    with torch.no_grad():
        output = loaded.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=4,
            pad_token_id=0,
            eos_token_id=2,
        )
    hashes = {n: tensors.tensor_hash(t) for n, t in loaded.state_dict().items()}
    del loaded
    legacy = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    assert hashes == {n: tensors.tensor_hash(t) for n, t in legacy.state_dict().items()}
    with torch.no_grad():
        other = legacy.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=4,
            pad_token_id=0,
            eos_token_id=2,
        )
    assert torch.equal(output, other)
    tensors._verify_source_files(tmp_path, contract, "editor")
