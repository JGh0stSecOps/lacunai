import json
from types import SimpleNamespace

import pytest

from lacunai.erase.factoid_data import load_factoid_data
from lacunai.erase.methods import METHOD_INFO, apply_method, touched_tensors


def data():
    return {
        "schema_version": 1,
        "source_fingerprint": "a" * 64,
        "forget": [{"prompt": "development target", "answer": "42"}],
        "retain_development": [{"prompt": "development retain", "answer": "oak"}],
        "provenance": "synthetic development fixture",
    }


def spec():
    return SimpleNamespace(
        probes=[SimpleNamespace(prompt="withheld target")],
        retain=[SimpleNamespace(prompt="withheld retain")],
        perplexity_texts=["withheld prose"],
    )


def test_load_and_guard(tmp_path):
    path = tmp_path / "development.json"
    path.write_text(json.dumps(data()))
    assert load_factoid_data(path, spec(), "a" * 64) == data()
    with pytest.raises(ValueError, match="fingerprint differs"):
        load_factoid_data(path, spec(), "b" * 64)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="unredirected"):
        load_factoid_data(link, spec())


@pytest.mark.parametrize(
    "prompt", ["WITHHELD TARGET", "  withheld   retain ", "ｗｉｔｈｈｅｌｄ prose"]
)
def test_evaluation_overlap(tmp_path, prompt):
    record = data()
    record["forget"][0]["prompt"] = prompt
    path = tmp_path / "data.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="frozen evaluation"):
        load_factoid_data(path, spec())


def test_duplicate_and_schema(tmp_path):
    record = data()
    record["retain_development"][0]["prompt"] = "DEVELOPMENT TARGET"
    path = tmp_path / "data.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="duplicate"):
        load_factoid_data(path, spec())
    record = data()
    record["schema_version"] = True
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="schema"):
        load_factoid_data(path, spec())


def fixtures():
    mlp = SimpleNamespace(
        down_proj=SimpleNamespace(weight=SimpleNamespace(shape=(8, 16))),
        up_proj=SimpleNamespace(bias=None),
    )
    runtime = SimpleNamespace(
        fingerprint="a" * 64,
        layer_ids=[0, 1],
        mlp=lambda _: mlp,
        model=SimpleNamespace(
            config=SimpleNamespace(model_type="qwen2", num_hidden_layers=2),
            named_modules=lambda: [("model.layers.1.mlp", mlp)],
        ),
    )
    local = {
        "checkpoint_fingerprint": runtime.fingerprint,
        "selected": [{"layer": 1, "neurons": []}],
        "layers": [{"layer": 1, "mean_effect": 0.5, "neurons": []}],
    }
    return runtime, local


@pytest.mark.parametrize(
    "method",
    [
        "npo-kl",
        "grad-diff-kl",
        "npo-kl-repair",
        "grad-diff-kl-repair",
        "undial-kl",
        "undial-kl-repair",
        "grad-diff-kl-repair2",
    ],
)
def test_allowlist(method):
    runtime, local = fixtures()
    assert "experimental" in METHOD_INFO[method]["cost"]
    assert touched_tensors(runtime, method, local) == ["model.layers.1.mlp.up_proj.weight"]
    local["layers"][0]["mean_effect"] = -1
    with pytest.raises(ValueError, match="positive causal"):
        touched_tensors(runtime, method, local)
    local["layers"][0].update(layer=0, mean_effect=0.5)
    local["selected"][0]["layer"] = 0
    with pytest.raises(ValueError, match="final Qwen"):
        touched_tensors(runtime, method, local)


def test_other_methods_reject_data():
    with pytest.raises(ValueError, match="only supported"):
        apply_method(None, "ablate-neurons", {}, {}, development_data=data())


@pytest.mark.parametrize(
    "method",
    [
        "npo-kl",
        "npo-kl-repair",
        "grad-diff-kl-repair",
        "undial-kl",
        "undial-kl-repair",
        "grad-diff-kl-repair2",
    ],
)
def test_real_apply_seam(monkeypatch, method):
    from lacunai.erase import factoid_gradient
    from lacunai.map.localize import record_digest
    from tests.test_factoid_gradient import FORGET, RETAIN, toy

    torch, runtime, name, _ = toy(monkeypatch)
    # The synthetic forget batch ends with token 4; UnDIAL excludes that EOS
    # position from demotion, just as the production tokenizer contract requires.
    runtime.tokenizer = SimpleNamespace(eos_token_id=4)
    forget_batch = factoid_gradient.kernels.answer_batch(
        runtime.tokenizer, {"prompts": [FORGET[0]["prompt"]], "answer": FORGET[0]["answer"]}, 128
    )
    assert int(forget_batch["labels"][0, -1]) == runtime.tokenizer.eos_token_id

    def loss(prompt, answer):
        batch = factoid_gradient.kernels.answer_batch(
            runtime.tokenizer, {"prompts": [prompt], "answer": answer}, 128
        )
        return (
            factoid_gradient.kernels.sequence_nll(
                factoid_gradient.kernels.answer_forward(runtime.model, batch),
                batch,
                selected_positions=True,
            )
            / (batch["labels"][:, 1:] != -100).sum()
        )

    runtime.loss = loss
    runtime.fingerprint = "a" * 64
    runtime.layer_ids = [0]
    runtime.mlp = lambda layer: runtime.model.model.layers[layer].mlp
    fingerprint = {
        "checkpoint_fingerprint": runtime.fingerprint,
        "target_id": "toy",
        "spec_sha256": "b" * 64,
    }
    local = {
        "checkpoint_fingerprint": runtime.fingerprint,
        "target_id": "toy",
        "spec_sha256": "b" * 64,
        "fingerprint_sha256": record_digest(fingerprint),
        "selected": [{"layer": 0, "neurons": []}],
        "layers": [{"layer": 0, "mean_effect": 0.5, "neurons": []}],
    }
    record = data()
    record.update(
        forget=FORGET,
        retain_development=[{"prompt": f"synthetic retain {i}", "answer": "oak"} for i in range(32)]
        if method.endswith(("-repair", "-repair2"))
        else RETAIN,
    )
    before = factoid_gradient._hashes(runtime.model)
    result = apply_method(runtime, method, local, fingerprint, steps=1, development_data=record)
    assert set(result["replacements"]) == {name}
    assert result["diagnostics"]["tensor_diffs"][name]["delta_norm_squared"] > 0
    assert factoid_gradient._hashes(runtime.model) == before
    assert not torch.equal(result["replacements"][name], runtime.model.state_dict()[name])
