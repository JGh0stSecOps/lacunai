"""LAPE mapping contracts; pretrained models are never loaded by these tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from lacunai.experimental.artifacts import atomic_json
from lacunai.map import language_entropy as lape


def ranked_diagnosis() -> dict[str, Any]:
    return {
        "languages": list(lape.LANGUAGES),
        "neuron_count": 100,
        "layer_widths": {"0": 50, "1": 50},
        "threshold": 0.5,
        "ranking": [
            {
                "rank": i,
                "layer": i // 50,
                "neuron": i % 50,
                "eligible": i < 10,
                "activation_probability": {"zh": 0.9 if i < 10 else 0.5},
            }
            for i in range(100)
        ],
    }


def test_global_selection_does_not_force_per_layer_quota() -> None:
    diagnosis = ranked_diagnosis()
    selected = lape.select(diagnosis, 0.025)
    # Python round(2.5) == 2, matching the released code rather than ceiling.
    assert selected["global_selected_count"] == 2
    assert selected["selected"] == [
        {"layer": 0, "width": 50, "rows": [0, 1]},
        {"layer": 1, "width": 50, "rows": []},
    ]


def test_saved_diagnosis_preserves_numeric_layer_order(tmp_path) -> None:
    diagnosis = ranked_diagnosis()
    diagnosis["layer_widths"] = {str(layer): 1 for layer in range(24)}
    diagnosis["neuron_count"] = 24
    diagnosis["ranking"] = [
        dict(diagnosis["ranking"][0], rank=layer, layer=layer, neuron=0) for layer in range(24)
    ]
    path = tmp_path / "diagnosis.json"
    atomic_json(path, diagnosis)
    selection = lape.select(json.loads(path.read_bytes()), 0.1)
    assert [row["layer"] for row in selection["selected"]] == list(range(24))
    assert selection["target_selected_count"] == 2


def test_language_assignment_is_strictly_above_global_threshold() -> None:
    diagnosis = ranked_diagnosis()
    diagnosis["ranking"][0]["activation_probability"]["zh"] = diagnosis["threshold"]
    selected = lape.select(diagnosis, 0.01)
    assert selected["global_selected_count"] == 1
    assert selected["target_selected_count"] == 0


def test_ineligible_infinity_fill_is_refused() -> None:
    with pytest.raises(ValueError, match="insufficient eligible"):
        lape.select(ranked_diagnosis(), 0.20)


@pytest.mark.parametrize("fraction", [0, 1.1, float("nan"), True])
def test_invalid_global_fraction_refused(fraction: float) -> None:
    with pytest.raises(ValueError, match="global selection fraction"):
        lape.select(ranked_diagnosis(), fraction)


def test_four_language_adaptation_requires_each_corpus_separately() -> None:
    corpora = {key: [key + " development"] for key in lape.LANGUAGES}
    lape.validate_corpora(corpora)
    corpora.pop("ja")
    with pytest.raises(ValueError, match="separate en/zh/es/ja"):
        lape.validate_corpora(corpora)


def test_adapter_checks_stale_checkpoint_before_any_tensor_work() -> None:
    runtime = SimpleNamespace(fingerprint="actual")
    with pytest.raises(ValueError, match="checkpoint identity"):
        lape.to_proposal(runtime, {"checkpoint_fingerprint": "stale"}, {}, execution_fraction=0.01)


def test_adapter_refuses_global_concentration_without_silent_trimming(monkeypatch: Any) -> None:
    diagnosis = ranked_diagnosis()
    selection = lape.select(diagnosis, 0.025)
    monkeypatch.setattr(lape.language, "_identity", lambda *args: None)
    monkeypatch.setattr(lape.language, "_names", lambda *args: {0: ["layer0"], 1: ["layer1"]})
    runtime = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [
                ("layer0", SimpleNamespace(shape=(50, 1))),
                ("layer1", SimpleNamespace(shape=(50, 1))),
            ]
        )
    )
    with pytest.raises(ValueError, match="per-layer execution cap"):
        lape.to_proposal(runtime, diagnosis, selection, execution_fraction=0.01)
    assert selection["selected"][0]["rows"] == [0, 1]


def test_real_float64_entropy_counts_global_threshold_and_dead_neurons() -> None:
    pytest.importorskip("torch")
    counts = {key: {"0": [10] * 50, "1": [10] * 50} for key in lape.LANGUAGES}
    for key in lape.LANGUAGES:
        counts[key]["1"][-1] = 0
        counts[key]["0"][:2] = [100, 100] if key == "zh" else [0, 0]
    counts["zh"]["0"][2:10] = [100] * 8
    result = lape.analyze_counts(counts, dict.fromkeys(lape.LANGUAGES, 100))
    assert result["threshold"] == 0.1
    assert result["threshold_order_statistic"] == 380
    assert result["eligible_neuron_count"] == 10
    assert result["ranking"][0]["entropy"] == 0.0
    assert result["ranking"][0]["neuron"] == 0
    assert result["ranking"][1]["neuron"] == 1
    assert lape.select(result, 0.02)["target_selected_count"] == 2
    dead = next(row for row in result["ranking"] if row["layer"] == 1 and row["neuron"] == 49)
    assert dead["entropy"] == 0.0 and not dead["eligible"]


def test_real_gate_hook_uses_silu_before_product_and_removes_hooks_on_failure() -> None:
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(1, 3)
            self.fail = False
            with torch.no_grad():
                self.gate_proj.weight.zero_()
                self.gate_proj.bias.copy_(torch.tensor([-2.0, 2.0, 0.0]))

        def forward(self, input_ids: Any, **kwargs: Any) -> Any:
            gate = self.gate_proj(input_ids.float().unsqueeze(-1))
            if self.fail:
                raise RuntimeError("synthetic forward failure")
            # Multiplication by negative up values reverses sign, but the hook must ignore it.
            return torch.nn.functional.silu(gate) * -1

    model = Tiny()
    runtime = SimpleNamespace(
        model=model,
        mlp=lambda layer: model,
        tokenizer=lambda prompt, **kwargs: {"input_ids": torch.ones((1, 3), dtype=torch.long)},
        config=SimpleNamespace(max_prompt_tokens=8),
        check=lambda: None,
        progress=lambda *args, **kwargs: None,
    )
    counts, tokens = lape._capture_counts(runtime, "prompt", [0])
    assert counts == {"0": [0, 3, 0]} and tokens == 3
    assert not model.gate_proj._forward_hooks
    runtime.config.max_prompt_tokens = 2
    with pytest.raises(ValueError, match="token bounds"):
        lape._capture_counts(runtime, "prompt", [0])
    assert not model.gate_proj._forward_hooks
    runtime.config.max_prompt_tokens = 8
    model.fail = True
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        lape._capture_counts(runtime, "prompt", [0])
    assert not model.gate_proj._forward_hooks


def test_adapter_source_hash_guard_detects_drift_before_proposal(monkeypatch: Any) -> None:
    class Weight:
        shape = (50, 1)

        def __getitem__(self, rows: list[int]) -> Any:
            return SimpleNamespace(norm=1.0)

    weight = Weight()
    diagnosis = ranked_diagnosis()
    diagnosis.update(
        checkpoint_fingerprint="same",
        source={
            "norm_squared": 200.0,
            "parameters": {
                name: {"shape": [50, 1], "sha256": "before", "norm_squared": 100.0}
                for name in ("layer0", "layer1")
            },
        },
    )
    runtime = SimpleNamespace(
        fingerprint="same",
        check=lambda: None,
        model=SimpleNamespace(named_parameters=lambda: [("layer0", weight), ("layer1", weight)]),
        mlp=lambda layer: SimpleNamespace(up_proj=SimpleNamespace(weight=weight)),
    )
    monkeypatch.setattr(lape.language, "_names", lambda *args: {0: ["layer0"], 1: ["layer1"]})
    monkeypatch.setattr(lape.tensors, "tensor_hash", lambda value: "after")
    monkeypatch.setattr(lape.tensors, "_square_norm", lambda value: getattr(value, "norm", 100.0))
    selection = lape.select(diagnosis, 0.01)
    with pytest.raises(ValueError, match="source parameter changed"):
        lape.to_proposal(runtime, diagnosis, selection, execution_fraction=0.025)


def test_excluded_assignment_filter_uses_probabilities_not_saved_labels() -> None:
    diagnosis = ranked_diagnosis()
    for row in diagnosis["ranking"]:
        row["activation_probability"].update(en=0.1, es=0.1, ja=0.1)
    diagnosis["ranking"][0]["activation_probability"]["ja"] = 0.9
    diagnosis["ranking"][0]["assigned_languages"] = ["zh"]  # Deliberately misleading.
    diagnosis["ranking"][1]["activation_probability"]["ja"] = diagnosis["threshold"]
    diagnosis["ranking"][1]["assigned_languages"] = ["zh", "ja"]
    result = lape.select(diagnosis, 0.025, exclude_languages=("ja", "en"))
    assert result["exclude_languages"] == ["en", "ja"]
    assert result["selected"][0]["rows"] == [1]
    assert result["target_unfiltered_count"] == 2
    assert result["target_filtered_count"] == result["target_selected_count"] == 1
    assert result["excluded_target_count"] == 1
    assert lape.select(diagnosis, 0.025)["target_selected_count"] == 2


@pytest.mark.parametrize("excluded", [("zh",), ("fr",), ("en", "en"), "ja", (None,)])
def test_invalid_assignment_exclusions_refused(excluded: Any) -> None:
    with pytest.raises(ValueError, match="distinct subset"):
        lape.select(ranked_diagnosis(), 0.01, exclude_languages=excluded)


def test_adapter_independently_recomputes_exclusion_filter(monkeypatch: Any) -> None:
    diagnosis = ranked_diagnosis()
    diagnosis["ranking"][0]["activation_probability"]["ja"] = 0.9
    selection = lape.select(diagnosis, 0.01, exclude_languages=("ja",))
    assert selection["target_selected_count"] == 0
    selection["selected"][0]["rows"] = [0]  # Reintroduce an excluded neuron.
    monkeypatch.setattr(lape.language, "_identity", lambda *args: None)
    with pytest.raises(ValueError, match="selection differs"):
        lape.to_proposal(SimpleNamespace(), diagnosis, selection, execution_fraction=0.01)
