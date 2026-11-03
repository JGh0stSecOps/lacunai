"""Bounded diagnostic contracts and actual tiny tensor invariants when Torch is installed."""

from types import SimpleNamespace

import pytest

from lacunai.map import fingerprints
from lacunai.map import retain_nullspace as mapping
from lacunai.map.localize import record_digest


def test_load_rejects_changed_diagnostic_identity_before_tensor_io(tmp_path):
    diagnosis = {"parameters": dict(mapping.PARAMETERS), "layers": {}}
    diagnosis["diagnosis_sha256"] = record_digest(diagnosis)
    diagnosis["parameters"]["rank_rtol"] = 0.1
    with pytest.raises(ValueError, match="identity"):
        mapping.load_layer(diagnosis, tmp_path, 0, 1)


@pytest.mark.parametrize("k", [0, 2, True, 17])
def test_only_declared_k_accepted(tmp_path, k):
    diagnosis = {"parameters": mapping.PARAMETERS, "layers": {}}
    diagnosis["diagnosis_sha256"] = record_digest(diagnosis)
    with pytest.raises(ValueError):
        mapping.load_layer(diagnosis, tmp_path, 0, k)


def test_nullspace_svd_preserves_all_retained_rows_not_only_average():
    torch = pytest.importorskip("torch")
    retained = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([[2.0, 3.0, 4.0, 0.0], [3.0, 0.0, 0.0, 2.0]], dtype=torch.float64)
    record, directions = mapping.analyze(retained, target)
    assert record["retain_rank"] == 2
    assert record["residual_rank"] == 2
    assert record["residual_singular_values"] == pytest.approx([4.0, 2.0])
    assert torch.allclose(retained @ directions, torch.zeros(2, 2, dtype=torch.float64), atol=1e-8)
    assert record["invariants"]["orthogonality_max_abs"] <= 1e-8
    assert record["invariants"]["retain_leakage_relative"] <= 1e-8
    assert record["boundary_gaps"]["1"]["absolute_gap"] == pytest.approx(2.0)


def test_zero_residual_and_degenerate_boundary_reported():
    torch = pytest.importorskip("torch")
    retained = torch.eye(3, dtype=torch.float64)
    record, directions = mapping.analyze(retained, retained)
    assert record["residual_rank"] == 0
    assert directions.shape == (3, 0)
    record, _ = mapping.analyze(torch.zeros((1, 3)), torch.eye(3))
    assert record["retain_rank"] == 0
    assert record["boundary_gaps"]["1"]["degenerate_boundary"]


def test_svd_rank_threshold_and_invalid_values():
    torch = pytest.importorskip("torch")
    retained = torch.diag(torch.tensor([1.0, 1e-9, 0.0], dtype=torch.float64))
    record, _ = mapping.analyze(retained, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))
    assert record["retain_rank"] == 1
    assert record["rank_thresholds"]["retain"] == pytest.approx(1e-8)
    with pytest.raises(ValueError, match="invalid"):
        mapping.analyze([[float("nan")]], [[1.0]])
    with pytest.raises(ValueError, match="invariants"):
        mapping.invariants(torch.eye(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))


def test_payload_roundtrip_rank_and_tamper_guards(tmp_path):
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    retained = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    record, directions = mapping.analyze(
        retained, torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float64)
    )
    path = tmp_path / "layer-00.safetensors"
    safe.save_file({"retain_means": retained, "directions": directions}, str(path))
    record.update(tensor_file=path.name, tensor_file_sha256=mapping._hash(path))
    diagnosis = {"parameters": mapping.PARAMETERS, "layers": {"0": record}}
    diagnosis["diagnosis_sha256"] = record_digest(diagnosis)
    r, v = mapping.load_layer(diagnosis, tmp_path, 0, 1)
    assert torch.equal(r, retained) and torch.equal(v, directions)
    with pytest.raises(ValueError, match="rank"):
        mapping.load_layer(diagnosis, tmp_path, 0, 4)
    with path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        mapping.load_layer(diagnosis, tmp_path, 0, 1)


def test_capture_signed_mean_raw_tokenization_and_hook_cleanup():
    torch = pytest.importorskip("torch")
    down = torch.nn.Linear(2, 1, bias=False)

    class Model(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            del input_ids, kwargs
            return down(torch.tensor([[[1.0, -3.0], [3.0, 1.0]]]))

    calls = []

    def tokenize(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return {"input_ids": torch.tensor([[1, 2]])}

    runtime = SimpleNamespace(
        model=Model(),
        tokenizer=tokenize,
        mlp=lambda layer: SimpleNamespace(down_proj=down),
        check=lambda: None,
        config=SimpleNamespace(max_prompt_tokens=4),
        encode=lambda prompt: torch.tensor([[1, 2]]),
    )
    result = fingerprints.capture(runtime, "raw", [0], raw=True, mean_only=True)
    assert torch.equal(result[0], torch.tensor([2.0, -1.0], dtype=torch.float64))
    assert calls == [("raw", {"add_special_tokens": True, "return_tensors": "pt"})]
    assert not down._forward_pre_hooks
    original = fingerprints.capture(runtime, "chat", [0])
    assert original[0].shape == (1, 2, 2)
    runtime.config.max_prompt_tokens = 1
    with pytest.raises(ValueError, match="bounds"):
        fingerprints.capture(runtime, "raw", [0], raw=True, mean_only=True)
    assert not down._forward_pre_hooks


def test_diagnostic_digest_guards_changed_source_before_tensor_io(tmp_path, monkeypatch):
    # Tensor I/O is irrelevant to the identity guard; make its downstream branch succeed.
    class Tensor:
        def __init__(self, shape):
            self.shape = shape

        def __getitem__(self, key):
            return self

        def contiguous(self):
            return self

    payload = {"retain_means": Tensor([1, 2]), "directions": Tensor([2, 1])}
    monkeypatch.setattr(
        mapping,
        "import_module",
        lambda name: SimpleNamespace(load_file=lambda *args, **kwargs: payload),
    )
    monkeypatch.setattr(mapping, "invariants", lambda *args: {})
    path = tmp_path / "layer-00.safetensors"
    path.write_bytes(b"synthetic payload")
    record = {
        "available_directions": 1,
        "residual_rank": 1,
        "retain_rows": 1,
        "width": 2,
        "tensor_file": path.name,
        "tensor_file_sha256": mapping._hash(path),
    }
    diagnosis = {
        "parameters": mapping.PARAMETERS,
        "layers": {"0": record},
        "checkpoint_fingerprint": "before",
    }
    diagnosis["diagnosis_sha256"] = record_digest(diagnosis)
    diagnosis["checkpoint_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        mapping.load_layer(diagnosis, tmp_path, 0, 1)
