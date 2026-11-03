"""Frozen language operator contracts and optional real-tensor source-preservation checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lacunai.erase import language


def test_fraction_ranking_is_signed_contrast_bounded_and_deterministic() -> None:
    chinese = [2.0, 2.0, 10.0] + [0.0] * 97
    other = [0.5, 0.5, 20.0] + [0.0] * 97
    assert language.select_rows(chinese, other, 0.01) == [0]
    assert language.select_rows(chinese, other, 0.025) == [0, 1]
    assert language.select_rows([1.0] * 100, [1.0] * 100, 0.20) == []


@pytest.mark.parametrize("fraction", [0, 0.02, 0.21, True, float("nan")])
def test_only_root_frozen_ladder_is_accepted(fraction: float) -> None:
    with pytest.raises(ValueError, match="frozen candidate ladder"):
        language.select_rows([2.0], [1.0], fraction)


@pytest.mark.parametrize("values", [[float("nan")], [-1.0], [float("inf")]])
def test_nonfinite_or_negative_activity_refused(values: list[float]) -> None:
    with pytest.raises(ValueError):
        language.select_rows(values, [1.0], 0.01)


def test_source_identity_guard_fails_before_any_tensor_work() -> None:
    runtime = SimpleNamespace(fingerprint="actual")
    with pytest.raises(ValueError, match="checkpoint identity"):
        language._identity(runtime, {"checkpoint_fingerprint": "stale"})


def test_source_parameter_hash_guard_detects_in_memory_drift(monkeypatch: Any) -> None:
    value = SimpleNamespace(shape=(2,))
    runtime = SimpleNamespace(
        fingerprint="same",
        check=lambda: None,
        model=SimpleNamespace(named_parameters=lambda: [("weight", value)]),
    )
    record = {
        "checkpoint_fingerprint": "same",
        "source": {
            "norm_squared": 2.0,
            "parameters": {"weight": {"shape": [2], "sha256": "before", "norm_squared": 2.0}},
        },
    }
    monkeypatch.setattr(language.tensors, "tensor_hash", lambda value: "after")
    monkeypatch.setattr(language.tensors, "_square_norm", lambda value: 2.0)
    with pytest.raises(ValueError, match="source parameter changed"):
        language._identity(runtime, record)


def numerical_runtime(tmp_path: Path) -> Any:
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen2")
    model.embedding = torch.nn.Embedding(100, 8)
    model.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.up_proj = torch.nn.Linear(8, 100, bias=True)
        layer.down_proj = torch.nn.Linear(100, 8, bias=False)
        model.layers.append(layer)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
            parameter.requires_grad_(False)
        model.embedding.weight.fill_(10.0)
    source = tmp_path / "checkpoint"
    source.mkdir()
    safe.save_file(dict(model.named_parameters()), source / "model.safetensors")
    (source / "config.json").write_text('{"model_type":"qwen2"}')
    return SimpleNamespace(
        model=model,
        fingerprint="a" * 64,
        layer_ids=[0, 1],
        mlp=lambda layer: model.layers[layer],
        check=lambda: None,
        progress=lambda *args, **kwargs: None,
        path=source,
        storage_dtypes={name: "F32" for name, _ in model.named_parameters()},
    )


def numerical_diagnosis(runtime: Any) -> dict[str, Any]:
    return {
        "checkpoint_fingerprint": runtime.fingerprint,
        "source": language._inventory(runtime),
        "layers": {
            str(layer): {
                "width": 100,
                "chinese": {"mean_abs": [2.0] + [0.0] * 99},
                "nonchinese": {"mean_abs": [1.0] + [0.0] * 99},
            }
            for layer in runtime.layer_ids
        },
    }


def test_real_tensors_apply_only_selected_rows_and_restore_every_parameter(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from lacunai.restore.checkpoints import prepare_snapshot

    runtime = numerical_runtime(tmp_path)
    diagnosis = numerical_diagnosis(runtime)
    proposal = language.propose(runtime, diagnosis, 0.01)
    names = language.touched_tensors(runtime, proposal)
    snapshot = prepare_snapshot(runtime.path, tmp_path / "snapshot", names)
    source_hashes = {
        name: language.tensors.tensor_hash(value)
        for name, value in runtime.model.named_parameters()
    }
    state = language.apply(runtime, proposal, snapshot)
    parameters = dict(runtime.model.named_parameters())
    for name in names:
        assert bool((parameters[name][0] == 0).all())
        assert torch.equal(parameters[name][1:], state["before_images"][name][1:])
    assert (
        language.tensors.tensor_hash(runtime.model.embedding.weight)
        == source_hashes["embedding.weight"]
    )

    def expired() -> None:
        raise TimeoutError("runtime expired after edit")

    runtime.check = expired
    assert language.restore(runtime, state)["restored_bit_exact"]
    assert all(
        language.tensors.tensor_hash(value) == source_hashes[name]
        for name, value in runtime.model.named_parameters()
    )


def test_real_tensors_missing_durable_snapshot_cannot_mutate_source(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    runtime = numerical_runtime(tmp_path)
    diagnosis = numerical_diagnosis(runtime)
    proposal = language.propose(runtime, diagnosis, 0.01)
    with pytest.raises(ValueError, match="matching prepared durable snapshot"):
        language.apply(runtime, proposal, {"status": "absent", "tensors": {}})
    language._identity(runtime, diagnosis)


def test_real_diagnostic_prompt_weighting_and_post_gate_statistic(
    tmp_path: Path, monkeypatch: Any
) -> None:
    torch = pytest.importorskip("torch")
    runtime = numerical_runtime(tmp_path)

    def captured(runtime: Any, prompt: str, layers: list[int]) -> Any:
        # Two Chinese prompts differ in length: long prompt must not dominate the mean.
        value, length = {
            "zh1": (2.0, 1),
            "zh2": (4.0, 8),
            "en1": (1.0, 1),
            "en2": (1.0, 4),
            "background": (1.0, 1),
        }[prompt]
        return {layer: torch.full((1, length, 100), value) for layer in layers}

    monkeypatch.setattr(language, "capture", captured)
    result = language.diagnose(
        runtime,
        [
            {"diagnostic_zh": "zh1", "diagnostic_en": "en1"},
            {"diagnostic_zh": "zh2", "diagnostic_en": "en2"},
        ],
        ["background"],
    )
    assert result["layers"]["0"]["chinese"]["mean_abs"] == [3.0] * 100
    assert result["layers"]["0"]["chinese"]["mean_prompt_rms"] == [3.0] * 100
    assert result["layers"]["0"]["nonchinese"]["mean_abs"] == [1.0] * 100
