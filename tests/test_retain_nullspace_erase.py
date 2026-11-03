"""Proposal contracts and real tensor arithmetic; no pretrained models are executed."""

from __future__ import annotations

import copy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lacunai.erase import language
from lacunai.erase import retain_nullspace as erase
from lacunai.map.localize import record_digest


def spec_for(diagnosis: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "retain-nullspace",
        "map_sha256": record_digest(diagnosis),
        "layers": [0],
        "k": 1,
        "strength": 0.5,
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"layers": []},
        {"layers": [0] * 2},
        {"layers": list(range(9))},
        {"layers": [True]},
        {"k": True},
        {"k": 2},
        {"strength": True},
        {"strength": 0.1},
        {"strength": float("nan")},
        {"method": "anything"},
        {"schema_version": True},
        {"extra": "unrestricted edit"},
    ],
)
def test_spec_rejects_unapproved_shape_or_values(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        erase.validate_spec(spec_for({}, **changes), {})


def test_spec_pins_full_map_including_internal_digest() -> None:
    diagnosis = {"diagnosis_sha256": "internal", "source": {"x": 1}}
    spec = spec_for(diagnosis)
    assert erase.validate_spec(spec, diagnosis) == [0]
    changed = copy.deepcopy(diagnosis)
    changed["diagnosis_sha256"] = "different"
    with pytest.raises(ValueError, match="map hash differs"):
        erase.validate_spec(spec, changed)


def test_stale_source_refused_before_tensor_dependency() -> None:
    diagnosis = {"checkpoint_fingerprint": "old"}
    with pytest.raises(ValueError, match="source checkpoint identity differs"):
        erase.propose(
            SimpleNamespace(fingerprint="new"), diagnosis, Path("unused"), spec_for(diagnosis)
        )


def fixture_runtime(torch: Any, weight: Any, *, background: bool = True) -> Any:
    mlp = torch.nn.Module()
    mlp.down_proj = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        mlp.down_proj.weight.copy_(weight)
    layer = torch.nn.Module()
    layer.mlp = mlp
    body = torch.nn.Module()
    body.layers = torch.nn.ModuleList([layer])
    model = torch.nn.Module()
    model.model = body
    model.config = SimpleNamespace(model_type="qwen2")
    if background:
        model.register_parameter("background", torch.nn.Parameter(torch.ones(100, 100) * 10))
    model.requires_grad_(False)
    events: list[dict[str, Any]] = []
    return SimpleNamespace(
        model=model,
        layer_ids=[0],
        mlp=lambda layer: mlp,
        fingerprint="synthetic-tensor-source",
        storage_dtypes={"model.layers.0.mlp.down_proj.weight": "BF16"},
        check=lambda: None,
        progress=lambda phase, **fields: events.append({"phase": phase, **fields}),
        events=events,
    )


def tensor_case(monkeypatch: Any, *, background: bool = True) -> tuple[Any, Any, Any, Any, Any]:
    torch = pytest.importorskip("torch")
    runtime = fixture_runtime(torch, torch.tensor([[1.0, 2.03125]]), background=background)
    diagnosis = {
        "checkpoint_fingerprint": runtime.fingerprint,
        "source": language._inventory(runtime),
    }
    retain = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    basis = torch.tensor([[2.0], [-1.0]], dtype=torch.float64) / (5.0**0.5)
    monkeypatch.setattr(erase.mapping, "load_layer", lambda *args: (retain, basis))
    return torch, runtime, diagnosis, retain, basis


def test_real_projection_rounding_and_nonzero_retain_leakage(monkeypatch: Any) -> None:
    torch, runtime, diagnosis, retain, basis = tensor_case(monkeypatch)
    name = "model.layers.0.mlp.down_proj.weight"
    before = runtime.mlp(0).down_proj.weight.detach().clone()
    replacement, report = erase.propose(runtime, diagnosis, Path("fixture"), spec_for(diagnosis))
    expected = before.double() - 0.5 * (before.double() @ basis) @ basis.T
    before_round_delta = float(torch.linalg.vector_norm(expected - before.double())) ** 2
    expected = expected.float().bfloat16().float()
    assert torch.equal(replacement[name], expected)
    assert torch.equal(runtime.mlp(0).down_proj.weight, before)
    leak = float(
        torch.linalg.vector_norm((replacement[name].double() - before.double()) @ retain.T)
    )
    assert leak > 0
    energy = report["projection_energy"]
    assert energy["before_rounding_delta_norm_squared"] == pytest.approx(before_round_delta)
    assert energy["after_rounding_delta_norm_squared"] == pytest.approx(
        float(torch.linalg.vector_norm(expected.double() - before.double())) ** 2
    )
    assert Decimal(energy["retained_delta_energy"]) == Decimal(
        str(energy["after_rounding_delta_norm_squared"])
    ) / Decimal(str(energy["before_rounding_delta_norm_squared"]))
    assert energy["retained_delta_energy"] != "1"
    assert report["layers"][0]["retain_leakage"][
        "post_storage_delta_retain_product_frobenius"
    ] == pytest.approx(leak)
    assert report["tensor_diffs"][name]["relative_frobenius"] <= 0.10
    assert report["model_relative_frobenius"] <= 0.02
    assert runtime.events[-1]["state"] == "complete"


@pytest.mark.parametrize(
    "fault", ["nonorthogonal", "retain_overlap", "nonfinite", "shape", "dtype"]
)
def test_real_basis_guard_rejects_bad_geometry(monkeypatch: Any, fault: str) -> None:
    torch, runtime, diagnosis, retain, basis = tensor_case(monkeypatch)
    if fault == "nonorthogonal":
        basis = basis * 2
    elif fault == "retain_overlap":
        basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    elif fault == "nonfinite":
        basis[0, 0] = float("nan")
    elif fault == "shape":
        basis = basis.T
    else:
        basis = basis.float()
    monkeypatch.setattr(erase.mapping, "load_layer", lambda *args: (retain, basis))
    before = runtime.mlp(0).down_proj.weight.detach().clone()
    with pytest.raises(ValueError, match="basis"):
        erase.propose(runtime, diagnosis, Path("fixture"), spec_for(diagnosis))
    assert torch.equal(runtime.mlp(0).down_proj.weight, before)
    assert runtime.events[-1]["state"] == "failed"


def test_real_source_hash_and_norm_drift_refused(monkeypatch: Any) -> None:
    torch, runtime, diagnosis, _, _ = tensor_case(monkeypatch)
    with torch.no_grad():
        runtime.model.background[0, 0] += 1
    with pytest.raises(ValueError, match="source parameter changed"):
        erase.propose(runtime, diagnosis, Path("fixture"), spec_for(diagnosis))
    assert runtime.events[-1]["state"] == "failed"


@pytest.mark.parametrize("which", ["tensor", "model", "no_op"])
def test_real_actual_stored_bounds_and_no_op(monkeypatch: Any, which: str) -> None:
    torch = pytest.importorskip("torch")
    weight = torch.tensor([[1.0, 1.0]]) if which == "tensor" else torch.tensor([[0.2, 2.0]])
    if which == "no_op":
        weight = torch.tensor([[0.0, 2.0]])
    weight = weight.bfloat16().float()
    runtime = fixture_runtime(torch, weight, background=which != "model")
    diagnosis = {
        "checkpoint_fingerprint": runtime.fingerprint,
        "source": language._inventory(runtime),
    }
    retain = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    monkeypatch.setattr(erase.mapping, "load_layer", lambda *args: (retain, basis))
    pattern = {
        "tensor": "tensor change exceeds",
        "model": "model relative change exceeds",
        "no_op": "no-op",
    }[which]
    with pytest.raises(ValueError, match=pattern):
        erase.propose(runtime, diagnosis, Path("fixture"), spec_for(diagnosis))
    assert torch.equal(runtime.mlp(0).down_proj.weight, weight)
    assert runtime.events[-1]["state"] == "failed"


def test_real_map_payload_hash_is_enforced_before_proposal(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    runtime = fixture_runtime(torch, torch.tensor([[1.0, 2.03125]]))
    retain = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    basis = torch.tensor([[2.0], [-1.0]], dtype=torch.float64) / (5.0**0.5)
    path = tmp_path / "layer-00.safetensors"
    safetensors.save_file({"retain_means": retain, "directions": basis}, str(path))
    diagnosis = {
        "checkpoint_fingerprint": runtime.fingerprint,
        "source": language._inventory(runtime),
        "parameters": erase.mapping.PARAMETERS,
        "layers": {
            "0": {
                "available_directions": 1,
                "residual_rank": 1,
                "retain_rows": 1,
                "width": 2,
                "tensor_file": path.name,
                "tensor_file_sha256": "0" * 64,
            }
        },
    }
    diagnosis["diagnosis_sha256"] = record_digest(diagnosis)
    with pytest.raises(ValueError, match="payload hash/path differs"):
        erase.propose(runtime, diagnosis, tmp_path, spec_for(diagnosis))
