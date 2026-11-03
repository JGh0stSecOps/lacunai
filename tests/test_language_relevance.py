"""Mask sensitivities on tiny random models; no pretrained weights or downloads."""

from types import SimpleNamespace

import pytest

from lacunai.map import language_relevance as relevance


def corpus():
    result = {}
    for split, families in [
        ("forget_train", (*relevance.FORGET, "write_zh", "translate_en_zh")),
        ("retain_train", relevance.RETAIN),
    ]:
        result[split] = [
            {"id": f"{family}:{i}", "family": family, "prompts": ["synthetic"], "answer": "answer"}
            for family in families
            for i in range(16)
        ]
    return result


def test_exact_training_projection_and_counts():
    data = corpus()
    rows = relevance.validate_corpus(data)
    assert len(rows) == 112
    assert all(row["family"] not in ("write_zh", "translate_en_zh") for row in rows)
    with pytest.raises(ValueError, match="only forget_train"):
        relevance.validate_corpus({**data, "heldout": []})
    data["retain_train"][0]["id"] = data["retain_train"][1]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        relevance.validate_corpus(data)


def test_invalid_corpus_progress_failure():
    events = []
    runtime = SimpleNamespace(progress=lambda phase, **fields: events.append(fields))
    with pytest.raises(ValueError):
        relevance.diagnose(runtime, {})
    assert events[0]["state"] == "active"
    assert events[-1]["state"] == "failed"


def toy():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=31,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
            initializer_range=0.2,
        )
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    runtime = SimpleNamespace(
        model=model,
        tokenizer=object(),
        layer_ids=[0, 1],
        mlp=lambda layer: model.model.layers[layer].mlp,
        check=lambda: None,
        progress=lambda *a, **k: None,
        fingerprint="toy",
        config=SimpleNamespace(seed=42, torch_threads=2, device="cpu", dtype="float32"),
    )
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    labels = ids.clone()
    labels[:, :3] = -100
    return (
        torch,
        runtime,
        {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels},
    )


def test_mask_derivative_finite_difference_and_no_weight_mutation():
    torch, runtime, batch = toy()
    before = {k: v.clone() for k, v in runtime.model.state_dict().items()}
    loss, gradients = relevance.mask_gradient(runtime, batch)
    assert loss.isfinite() and all(g.dtype == torch.float64 for g in gradients.values())
    layer, neuron = max(
        (
            (layer, neuron)
            for layer, values in gradients.items()
            for neuron in range(values.numel())
        ),
        key=lambda pair: abs(float(gradients[pair[0]][pair[1]])),
    )
    epsilon, absolute_tolerance = 0.01, 3e-5
    observed = float(gradients[layer][neuron])
    assert abs(observed) > 100 * absolute_tolerance
    losses = []
    for delta in (-epsilon, epsilon):
        mask = torch.ones(12)
        mask[neuron] += delta
        hook = runtime.mlp(layer).down_proj.register_forward_pre_hook(
            lambda module, args, mask=mask: (args[0] * mask,)
        )
        try:
            logits = relevance.objectives.answer_forward(runtime.model, batch)
            losses.append(
                float(relevance.objectives.sequence_nll(logits, batch, selected_positions=True) / 2)
            )
        finally:
            hook.remove()
    finite_difference = (losses[1] - losses[0]) / (2 * epsilon)
    assert abs(finite_difference) > 100 * absolute_tolerance

    def assert_derivative(value):
        assert value == pytest.approx(finite_difference, abs=absolute_tolerance, rel=0.02)

    # Prove this numerical comparison rejects a manufactured zero derivative.
    with pytest.raises(AssertionError):
        assert_derivative(0.0)
    assert_derivative(observed)
    assert all(torch.equal(v, before[k]) for k, v in runtime.model.state_dict().items())
    assert all(p.grad is None for p in runtime.model.parameters())
    assert all(not runtime.mlp(i).down_proj._forward_pre_hooks for i in runtime.layer_ids)


def test_failure_hooks_and_flags_restored(monkeypatch):
    torch, runtime, batch = toy()
    runtime.model.train()
    next(runtime.model.parameters()).requires_grad_(True)
    flags = [p.requires_grad for p in runtime.model.parameters()]
    before = {k: v.clone() for k, v in runtime.model.state_dict().items()}
    monkeypatch.setattr(relevance.objectives, "answer_batch", lambda *args: batch)

    def fail(model, batch):
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(relevance.objectives, "answer_forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        relevance.diagnose(runtime, corpus())
    assert [p.requires_grad for p in runtime.model.parameters()] == flags
    assert runtime.model.training
    assert all(p.grad is None for p in runtime.model.parameters())
    assert all(not runtime.mlp(i).down_proj._forward_pre_hooks for i in runtime.layer_ids)
    assert all(torch.equal(v, before[k]) for k, v in runtime.model.state_dict().items())


def payload_fixture():
    torch = pytest.importorskip("torch")
    signed = torch.tensor(
        [
            [-32.0, -16.0],
            [-32.0, -16.0],
            [8.0, 16.0],
            [-8.0, 16.0],
            [8.0, 16.0],
            [8.0, 16.0],
            [8.0, 16.0],
        ],
        dtype=torch.float64,
    )
    payload = {
        "family_counts": torch.full((7,), 16, dtype=torch.int64),
        "family_nll_sum": torch.ones(7, dtype=torch.float64),
        "layer.0.signed_sum": signed,
        "layer.0.absolute_sum": signed.abs(),
    }
    payload.update(relevance.derive(payload, [0]))
    metadata = {"schema_version": 1, "families": list(relevance.FAMILIES), "layers": [0]}
    repin(metadata, payload)
    return torch, metadata, payload


def repin(metadata, payload):
    metadata["payload"] = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": relevance.payload_hash(v)}
        for k, v in payload.items()
    }


def test_conservation_scores_and_rehashed_tamper():
    torch, metadata, payload = payload_fixture()
    relevance.validate_payload(metadata, payload)
    assert payload["layer.0.score"].tolist() == [1.5, 0.0]
    payload["layer.0.score"][0] = 9
    repin(metadata, payload)
    with pytest.raises(ValueError, match="derived"):
        relevance.validate_payload(metadata, payload)
    payload["layer.0.absolute_sum"][0, 0] = 0
    with pytest.raises(ValueError, match="conservation"):
        relevance.derive(payload, [0])
    payload["layer.0.absolute_sum"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        relevance.derive(payload, [0])


def test_retain_risk_absolute_before_family_mean():
    _, _, payload = payload_fixture()
    payload["layer.0.signed_sum"][2:] = 0
    result = relevance.derive(payload, [0])
    assert result["layer.0.retain_risk"].tolist() == [0.5, 1.0]


def test_success_full_diagnostic_counts_source_and_cleanup(monkeypatch):
    torch, runtime, batch = toy()
    monkeypatch.setattr(relevance.objectives, "answer_batch", lambda *args: batch)
    before = {k: v.clone() for k, v in runtime.model.state_dict().items()}
    metadata, payload = relevance.diagnose(runtime, corpus())
    relevance.validate_payload(metadata, payload)
    assert payload["family_counts"].tolist() == [16] * 7
    assert len(metadata["examples"]) == 112
    assert all(torch.equal(v, before[k]) for k, v in runtime.model.state_dict().items())
    assert all(not runtime.mlp(i).down_proj._forward_pre_hooks for i in runtime.layer_ids)


def test_preexisting_grad_preserved(monkeypatch):
    torch, runtime, _ = toy()
    p = next(runtime.model.parameters())
    p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="preexisting"):
        relevance.diagnose(runtime, corpus())
    assert torch.equal(p.grad, torch.ones_like(p))


def test_boolean_schema_version_refused():
    _, metadata, payload = payload_fixture()
    metadata["schema_version"] = True
    with pytest.raises(ValueError, match="metadata"):
        relevance.validate_payload(metadata, payload)


def test_tiny_negative_absolute_sum_refused():
    _, _, payload = payload_fixture()
    payload["layer.0.signed_sum"][0, 0] = 0
    payload["layer.0.absolute_sum"][0, 0] = -1e-14
    with pytest.raises(ValueError, match="conservation"):
        relevance.derive(payload, [0])


def proposal_fixture():
    torch, runtime, _ = toy()
    runtime.storage_dtypes = {name: "F32" for name, _ in runtime.model.named_parameters()}
    with torch.no_grad():
        for layer in runtime.layer_ids:
            runtime.mlp(layer).up_proj.weight.fill_(1)
            runtime.mlp(layer).up_proj.weight[:4].fill_(0.01)
    payload = {
        "family_counts": torch.full((7,), 16, dtype=torch.int64),
        "family_nll_sum": torch.ones(7, dtype=torch.float64),
    }
    for layer in runtime.layer_ids:
        signed = torch.zeros((7, 12), dtype=torch.float64)
        signed[:2, :4] = -16 * torch.tensor([5.0, 4.0, 3.0, 2.0], dtype=torch.float64) / (layer + 1)
        payload[f"layer.{layer}.signed_sum"] = signed
        payload[f"layer.{layer}.absolute_sum"] = signed.abs()
    payload.update(relevance.derive(payload, runtime.layer_ids))
    metadata = {
        "schema_version": 1,
        "layers": runtime.layer_ids.copy(),
        "families": list(relevance.FAMILIES),
        "provenance": relevance.PROVENANCE.copy(),
        "checkpoint_fingerprint": runtime.fingerprint,
        "source": relevance.language._inventory(runtime),
    }
    repin(metadata, payload)
    return torch, runtime, metadata, payload


def test_proposal_exact_positive_selection_and_immutable_source():
    torch, runtime, metadata, payload = proposal_fixture()
    before = {k: v.clone() for k, v in runtime.model.state_dict().items()}
    pin = relevance.record_digest(metadata)
    payload_before = {k: v.clone() for k, v in payload.items()}
    proposal = relevance.to_proposal(runtime, metadata, payload, top_k=2, execution_fraction=0.10)
    assert proposal["relevance_selection"]["ranked_ids"] == [[0, 0], [0, 1]]
    assert proposal["selected"][0]["rows"] == [0, 1]
    name = "model.layers.0.mlp.up_proj.weight"
    expected = relevance.tensors._square_norm(runtime.mlp(0).up_proj.weight[[0, 1]])
    assert proposal["tensor_diffs"][name]["delta_norm_squared"] == expected
    assert proposal["operation"] == "zero_language_up_projection_rows"
    assert relevance.record_digest(metadata) == pin
    assert all(torch.equal(v, before[k]) for k, v in runtime.model.state_dict().items())
    assert all(torch.equal(v, payload_before[k]) for k, v in payload.items())


@pytest.mark.parametrize(
    "top_k,fraction,match",
    [
        (True, 0.1, "positive integer"),
        (0, 0.1, "positive integer"),
        (2, 0.03, "allowlist"),
        (3, 0.1, "cap"),
        (9, 0.2, "positive-score"),
    ],
)
def test_proposal_refuses_selection_limits(top_k, fraction, match):
    _, runtime, metadata, payload = proposal_fixture()
    with pytest.raises(ValueError, match=match):
        relevance.to_proposal(runtime, metadata, payload, top_k=top_k, execution_fraction=fraction)


def test_proposal_rehashed_rank_tamper_and_source_drift():
    torch, runtime, metadata, payload = proposal_fixture()
    saved = payload["ranking"].clone()
    payload["ranking"][0] = saved[1]
    repin(metadata, payload)
    with pytest.raises(ValueError, match="derived"):
        relevance.to_proposal(runtime, metadata, payload, top_k=1, execution_fraction=0.1)
    payload["ranking"] = saved
    repin(metadata, payload)
    with torch.no_grad():
        runtime.mlp(0).up_proj.weight[0, 0] += 0.01
    with pytest.raises(ValueError, match="source parameter changed"):
        relevance.to_proposal(runtime, metadata, payload, top_k=1, execution_fraction=0.1)


def test_proposal_runtime_width_and_noop_refused():
    torch, runtime, metadata, payload = proposal_fixture()
    for key in ("signed_sum", "absolute_sum"):
        payload["layer.0." + key] = payload["layer.0." + key][:, :11].clone()
    payload.update(relevance.derive(payload, [0, 1]))
    repin(metadata, payload)
    with pytest.raises(ValueError, match="width"):
        relevance.to_proposal(runtime, metadata, payload, top_k=1, execution_fraction=0.1)
    torch, runtime, metadata, payload = proposal_fixture()
    with torch.no_grad():
        runtime.mlp(0).up_proj.weight[0].zero_()
    metadata["source"] = relevance.language._inventory(runtime)
    with pytest.raises(ValueError, match="no-op"):
        relevance.to_proposal(runtime, metadata, payload, top_k=1, execution_fraction=0.1)


def test_proposal_rejects_invalid_options_before_source_access():
    for top_k, fraction in [(True, 0.1), (0, 0.1), (1, float("nan")), (1, 0.03)]:
        with pytest.raises(ValueError):
            relevance.to_proposal(None, {}, {}, top_k=top_k, execution_fraction=fraction)


def test_rehashed_derived_dtype_refused():
    torch, metadata, payload = payload_fixture()
    payload["ranking"] = payload["ranking"].to(torch.float64)
    repin(metadata, payload)
    with pytest.raises(ValueError, match="derived"):
        relevance.validate_payload(metadata, payload)
