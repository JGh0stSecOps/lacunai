"""Numerical toy distributions and hook lifecycle; no pretrained model downloads."""

from types import SimpleNamespace

import pytest

from lacunai.map import language_distribution as distribution


def test_provenance_no_reproduction_or_auto_edit():
    assert distribution.PROVENANCE["paper_reproduction"] is False
    assert "clustering" in distribution.PROVENANCE["excluded_paper_procedure"]
    assert distribution.BINS == 64


def test_invalid_corpus_emits_failure_before_torch():
    events = []
    runtime = SimpleNamespace(progress=lambda phase, **fields: events.append(fields))
    with pytest.raises(ValueError, match="separate en/zh/es/ja"):
        distribution.diagnose(runtime, {"zh": ["synthetic"]})
    assert events[0]["state"] == "active" and events[-1]["state"] == "failed"


def toy(shared=False):
    torch = pytest.importorskip("torch")
    values = {
        key: torch.tensor([[-0.2], [-0.2]], dtype=torch.float32) for key in distribution.LANGUAGES
    }
    values["zh"][:] = -0.1
    if shared:
        values["ja"][:] = -0.1
    joined = torch.cat(list(values.values()))
    minimum, maximum = joined.amin(0).double(), joined.amax(0).double()
    hist = {key: torch.zeros((1, 64), dtype=torch.int32) for key in values}
    negative = {key: torch.zeros(1, dtype=torch.int32) for key in values}
    for key, activation in values.items():
        distribution.update_histogram(activation, minimum, maximum, hist[key], negative[key])
    return torch, values, hist, negative


def test_negative_distributions_distinguish_equal_positive_rates():
    torch, values, hist, negative = toy()
    assert all(int((v > 0).sum()) == 0 for v in values.values())
    _, payload = distribution.analyze({0: hist}, {0: negative}, {key: 2 for key in values})
    assert payload["layer.0.contrast"].item() == 1.0
    assert torch.equal(payload["layer.0.negative_mass"], torch.ones((1, 4), dtype=torch.float64))


def test_shared_chinese_japanese_reduces_contrast():
    _, values, hist, negative = toy(shared=True)
    _, payload = distribution.analyze({0: hist}, {0: negative}, {key: 2 for key in values})
    assert payload["layer.0.contrast"].item() == 0.0
    pair = distribution.PAIRS.index(("zh", "ja"))
    assert payload["layer.0.overlap"][0, pair].item() == 1.0


def test_reordering_and_numeric_layer_ties():
    torch, values, hist, negative = toy()
    totals = {key: 2 for key in values}
    _, first = distribution.analyze({10: hist, 2: hist}, {10: negative, 2: negative}, totals)
    _, second = distribution.analyze(
        {2: dict(reversed(list(hist.items()))), 10: hist}, {2: negative, 10: negative}, totals
    )
    assert first["ranking"].tolist() == [[2, 0], [10, 0]]
    for key in first:
        assert torch.equal(first[key], second[key])


def test_constant_zero_bins_and_overflow_atomicity():
    torch = pytest.importorskip("torch")
    bounds = torch.zeros(2, dtype=torch.float64)
    histogram = torch.zeros((2, 64), dtype=torch.int32)
    negative = torch.zeros(2, dtype=torch.int32)
    distribution.update_histogram(torch.zeros((3, 2)), bounds, bounds, histogram, negative)
    assert histogram[:, 0].tolist() == [3, 3] and histogram.sum().item() == 6
    assert negative.tolist() == [0, 0]
    histogram[0, 0] = distribution.INT32_MAX
    before = histogram.clone()
    with pytest.raises(ValueError, match="overflow"):
        distribution.update_histogram(torch.zeros((1, 2)), bounds, bounds, histogram, negative)
    assert torch.equal(before, histogram) and negative.tolist() == [0, 0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_activation_refused(bad):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="finite"):
        distribution.update_histogram(
            torch.tensor([[bad]]),
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.zeros((1, 64), dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        )


def test_histogram_conserves_tokens_and_rejects_range_drift():
    torch, values, hist, negative = toy()
    hist["zh"][0, 0] += 1
    with pytest.raises(ValueError, match="conservation"):
        distribution.analyze({0: hist}, {0: negative}, {key: 2 for key in values})
    with pytest.raises(ValueError, match="outside frozen"):
        distribution.update_histogram(
            torch.tensor([[2.0]]),
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            hist["zh"],
            negative["zh"],
        )


def test_payload_roundtrip_and_tamper(tmp_path):
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    payload = {
        "hist": torch.tensor([[0, 3]], dtype=torch.int32),
        "min": torch.tensor([-0.25], dtype=torch.float64),
    }
    metadata = {
        "schema_version": distribution.VERSION,
        "payload": {
            key: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "sha256": distribution.payload_hash(v),
            }
            for key, v in payload.items()
        },
    }
    path = tmp_path / "histograms.safetensors"
    safe.save_file(payload, str(path))
    restored = safe.load_file(str(path))
    distribution.validate_payload(metadata, restored)
    restored["hist"][0, 1] += 1
    with pytest.raises(ValueError, match="tensor differs"):
        distribution.validate_payload(metadata, restored)


def test_capture_removes_hooks_on_visit_failure():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(2, 2, bias=False)

        def forward(self, input_ids, **kwargs):
            return self.gate(input_ids.float().unsqueeze(-1).repeat(1, 1, 2))

    model = Model()
    runtime = SimpleNamespace(
        model=model,
        config=SimpleNamespace(max_prompt_tokens=8),
        check=lambda: None,
        tokenizer=lambda *args, **kwargs: {"input_ids": torch.tensor([[1, 2]])},
        mlp=lambda layer: SimpleNamespace(gate_proj=model.gate),
    )

    def failed(layer, activation):
        raise RuntimeError("fixture visit failure")

    with pytest.raises(RuntimeError, match="visit failure"):
        distribution._capture(runtime, "synthetic", [0], failed)
    assert not model.gate._forward_hooks


def test_two_pass_diagnosis_preserves_parameters_and_records_token_totals():
    torch = pytest.importorskip("torch")
    from lacunai.core.spec import RuntimeConfig
    from lacunai.experimental.tensors import tensor_hash

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.gate.weight.copy_(torch.tensor([[0.1, -0.3], [0.2, 0.4]]))
            self.config = SimpleNamespace(model_type="qwen2", hidden_act="silu")
            self.calls = 0

        def forward(self, input_ids, **kwargs):
            self.calls += 1
            return self.gate(input_ids.float().unsqueeze(-1).repeat(1, 1, 2))

    model = Model()
    before = tensor_hash(model.gate.weight)
    events = []
    runtime = SimpleNamespace(
        model=model,
        config=RuntimeConfig(),
        layer_ids=[0],
        fingerprint="fixture-source",
        check=lambda: None,
        progress=lambda phase, **fields: events.append(fields),
        tokenizer=lambda prompt, **kwargs: {"input_ids": torch.tensor([[1 + len(prompt), 2]])},
        mlp=lambda layer: SimpleNamespace(gate_proj=model.gate),
    )
    corpora = {key: [key] for key in distribution.LANGUAGES}
    metadata, payload = distribution.diagnose(runtime, corpora)
    assert model.calls == 8
    assert metadata["token_totals"] == {key: 2 for key in distribution.LANGUAGES}
    assert tensor_hash(model.gate.weight) == before
    assert events[0]["state"] == "active" and events[-1]["state"] == "complete"
    assert len([row for row in events if row.get("completed") == 1]) == 8
    distribution.validate_payload(metadata, payload)
    assert payload["ranking"].tolist() == [[0, 0], [0, 1]]
    assert not model.gate._forward_hooks


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_proposal_refuses_invalid_selection_before_numerics(top_k):
    with pytest.raises(ValueError, match="positive integer"):
        distribution.to_proposal(SimpleNamespace(), {}, {}, top_k=top_k, execution_fraction=0.1)


def proposal_fixture():
    torch = pytest.importorskip("torch")
    from lacunai.erase import language

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = torch.nn.Linear(2, 32, bias=False)
            self.down_proj = torch.nn.Linear(32, 2, bias=False)
            self.gate_proj = torch.nn.Linear(2, 32, bias=False)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()
            self.background = torch.nn.Parameter(torch.full((100,), 10.0))
            self.config = SimpleNamespace(model_type="qwen2", hidden_act="silu")
            with torch.no_grad():
                for parameter in self.mlp.parameters():
                    parameter.fill_(1.0)

    model = Model()
    runtime = SimpleNamespace(
        model=model,
        fingerprint="source",
        layer_ids=[0],
        check=lambda: None,
        mlp=lambda layer: model.mlp,
        storage_dtypes={name: "F32" for name, _ in model.named_parameters()},
    )
    hist = {key: torch.zeros((32, 64), dtype=torch.int32) for key in distribution.LANGUAGES}
    negative = {key: torch.zeros(32, dtype=torch.int32) for key in distribution.LANGUAGES}
    for counts in hist.values():
        counts[:, 0] = 2
    hist["zh"][:2, 0] = 0
    hist["zh"][:2, 63] = 2
    totals = {key: 2 for key in distribution.LANGUAGES}
    summary, payload = distribution.analyze({0: hist}, {0: negative}, totals)
    payload["layer.0.min"] = torch.zeros(32, dtype=torch.float64)
    payload["layer.0.max"] = torch.ones(32, dtype=torch.float64)
    for key in distribution.LANGUAGES:
        payload[f"layer.0.hist.{key}"] = hist[key]
        payload[f"layer.0.negative_counts.{key}"] = negative[key]
    metadata = {
        "schema_version": distribution.VERSION,
        "kind": "language_distribution",
        "checkpoint_fingerprint": runtime.fingerprint,
        "source": language._inventory(runtime),
        "languages": list(distribution.LANGUAGES),
        "pair_order": [list(p) for p in distribution.PAIRS],
        "bins": 64,
        "token_totals": totals,
        **summary,
        "payload": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": distribution.payload_hash(value),
            }
            for key, value in payload.items()
        },
    }
    return torch, runtime, metadata, payload


def test_exact_top_k_proposal_does_not_mutate_source():
    torch, runtime, metadata, payload = proposal_fixture()
    from lacunai.erase import language
    from lacunai.experimental import tensors

    before = {name: tensors.tensor_hash(value) for name, value in runtime.model.named_parameters()}
    proposal = distribution.to_proposal(runtime, metadata, payload, top_k=2, execution_fraction=0.1)
    assert proposal["selected"] == [{"layer": 0, "width": 32, "rows": [0, 1]}]
    assert proposal["distribution_selection"]["ranked_ids"] == [[0, 0], [0, 1]]
    assert proposal["tensor_diffs"]["mlp.up_proj.weight"]["delta_norm_squared"] == 4.0
    assert proposal["distribution_selection"]["contrast_min"] == 1.0
    assert language.touched_tensors(runtime, proposal) == ["mlp.up_proj.weight"]
    assert {
        name: tensors.tensor_hash(value) for name, value in runtime.model.named_parameters()
    } == before
    assert (
        proposal["operation"] == "zero_language_up_projection_rows" and proposal["strength"] == 1.0
    )


@pytest.mark.parametrize(
    "top_k,fraction,message", [(2, 0.025, "per-layer"), (33, 0.2, "exceeds"), (1, 0.3, "allowlist")]
)
def test_proposal_does_not_silently_trim(top_k, fraction, message):
    _, runtime, metadata, payload = proposal_fixture()
    with pytest.raises(ValueError, match=message):
        distribution.to_proposal(
            runtime, metadata, payload, top_k=top_k, execution_fraction=fraction
        )


@pytest.mark.parametrize("key", ["ranking", "layer.0.overlap", "layer.0.contrast"])
def test_rehashed_rank_and_statistics_tamper_recomputed(key):
    _, runtime, metadata, payload = proposal_fixture()
    if key == "ranking":
        payload[key][[0, 1]] = payload[key][[1, 0]]
    else:
        payload[key].view(-1)[0] = 0.5
    metadata["payload"][key]["sha256"] = distribution.payload_hash(payload[key])
    with pytest.raises(ValueError, match="recomputed counts"):
        distribution.to_proposal(runtime, metadata, payload, top_k=1, execution_fraction=0.1)


def test_proposal_rejects_source_change_and_actual_tensor_norm_bound():
    torch, runtime, metadata, payload = proposal_fixture()
    from lacunai.erase import language

    with torch.no_grad():
        runtime.model.mlp.up_proj.weight[0].mul_(20.0)
    with pytest.raises(ValueError, match="source parameter changed"):
        distribution.to_proposal(runtime, metadata, payload, top_k=1, execution_fraction=0.1)
    metadata["source"] = language._inventory(runtime)
    with pytest.raises(ValueError, match="tensor change bound"):
        distribution.to_proposal(runtime, metadata, payload, top_k=1, execution_fraction=0.1)


def test_proposal_model_norm_bound_uses_actual_other_weights():
    torch, runtime, metadata, payload = proposal_fixture()
    from lacunai.erase import language

    with torch.no_grad():
        runtime.model.background.zero_()
        runtime.model.mlp.down_proj.weight.zero_()
        runtime.model.mlp.gate_proj.weight.zero_()
    metadata["source"] = language._inventory(runtime)
    with pytest.raises(ValueError, match="model change bound"):
        distribution.to_proposal(runtime, metadata, payload, top_k=1, execution_fraction=0.1)
