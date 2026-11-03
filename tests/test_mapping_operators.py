from __future__ import annotations

import copy
import json
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree

import pytest

from lacunai.erase.methods import apply_method, touched_tensors, validate_replacements
from lacunai.map.fingerprints import assert_identity, extremes, weighted_features
from lacunai.map.index import MapIndex, heatmap
from lacunai.map.localize import effect_record, record_digest

SHA = "a" * 64


def fake_runtime() -> Any:
    mlp = SimpleNamespace(
        down_proj=SimpleNamespace(weight=SimpleNamespace(shape=(2, 4))),
        up_proj=SimpleNamespace(bias=None),
    )
    return SimpleNamespace(
        fingerprint=SHA,
        layer_ids=[0],
        mlp=lambda layer: mlp,
        model=SimpleNamespace(
            config=SimpleNamespace(model_type="qwen2"),
            named_modules=lambda: [("model.layers.0.mlp", mlp)],
        ),
    )


def localized() -> dict[str, Any]:
    return {
        "checkpoint_fingerprint": SHA,
        "target_id": "fact",
        "spec_sha256": "b" * 64,
        "layers": [
            {"layer": 0, "mean_effect": 0.5, "neurons": [{"neuron": 1, "mean_effect": 0.2}]}
        ],
        "selected": [{"layer": 0, "neurons": [1]}],
    }


def map_record() -> dict[str, Any]:
    return {
        "target_id": "fact",
        "checkpoint_fingerprint": SHA,
        "kind": "fingerprint",
        "spec_sha256": "b" * 64,
        "layers": {"0": {"target_mean": [-2.0, 4.0], "direction": [1.0, 0.0]}},
    }


def test_signed_extremes_filter_dead_features_and_preserve_bottom_polarity() -> None:
    out = extremes([0.0, -4.0, 2.0, 1e-10, 3.0], 2)
    assert [r["neuron"] for r in out["top"]] == [4, 2]
    assert [r["neuron"] for r in out["bottom"]] == [1, 2]
    assert all(r["neuron"] not in {0, 3} for group in out.values() for r in group)
    assert extremes([0.0, 0.0], 5) == {"top": [], "bottom": []}
    with pytest.raises(ValueError, match="non-finite"):
        extremes([float("nan")], 1)


def test_idf_uses_background_document_frequency_not_duplicate_counts() -> None:
    target = [{"top": [{"neuron": 0}, {"neuron": 1}], "bottom": []}]
    background = [{"top": [{"neuron": 0}, {"neuron": 0}], "bottom": []}] * 3
    rows = weighted_features(target, background, 2)
    assert rows[0]["neuron"] == 1
    assert rows[0]["idf"] == pytest.approx(math.log(4) + 1)
    assert rows[1]["background_documents"] == 3
    assert rows[1]["idf"] == 1.0


def test_causal_normalization_is_unclamped_and_undefined_contrast_stays_unknown() -> None:
    assert effect_record(-1.0, -2.0, 0.0)["normalized_effect"] == 2.0
    assert effect_record(-1.0, -2.0, -3.0)["normalized_effect"] == -1.0
    for clean, corrupt in ((-1.0, -1.0), (-2.0, -1.0)):
        assert effect_record(clean, corrupt, 0.0)["normalized_effect"] is None
    with pytest.raises(ValueError, match="non-finite"):
        effect_record(0.0, 1.0, float("inf"))


def test_checkpoint_identity_guard_rejects_stale_map() -> None:
    spec = SimpleNamespace(digest="b" * 64, target_id="fact")
    record = map_record()
    record["checkpoint_fingerprint"] = "c" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        assert_identity(fake_runtime(), spec, record)


def test_index_exact_query_roundtrip_and_no_silent_stale_fallback(tmp_path: Path) -> None:
    index = MapIndex(tmp_path / "index")
    record = map_record()
    index.put(record)
    assert index.get("fact", SHA) == record
    assert index.list() == [
        {"target_id": "fact", "checkpoint_fingerprint": SHA, "kind": "fingerprint"}
    ]
    with pytest.raises(ValueError, match="stale"):
        index.get("fact", "c" * 64)
    changed = copy.deepcopy(record)
    changed["layers"]["0"]["target_mean"][0] = 10.0
    with pytest.raises(ValueError, match="different map"):
        index.put(changed)
    assert index.get("fact", SHA) == record


def test_index_detects_payload_identity_tamper(tmp_path: Path) -> None:
    index = MapIndex(tmp_path)
    index.put(map_record())
    with sqlite3.connect(index.path) as db:
        bad = map_record()
        bad["target_id"] = "other"
        db.execute("UPDATE maps SET payload=?", (json.dumps(bad),))
    with pytest.raises(ValueError, match="payload identity"):
        index.get("fact", SHA)


def test_export_has_actual_values_and_refuses_overwrite_before_writing(tmp_path: Path) -> None:
    index = MapIndex(tmp_path / "index")
    index.put(map_record())
    exported = index.export("fact", SHA, tmp_path / "export")
    assert json.loads(Path(exported["json"]).read_text()) == map_record()
    root = ElementTree.fromstring(Path(exported["heatmap"]).read_text())
    assert root.attrib["role"] == "img"
    assert "Layer 0" in Path(exported["heatmap"]).read_text()
    before = Path(exported["json"]).read_bytes()
    with pytest.raises(ValueError, match="exists"):
        index.export("fact", SHA, tmp_path / "export")
    assert Path(exported["json"]).read_bytes() == before


def test_index_and_export_reject_links(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="linked"):
        MapIndex(link)
    index = MapIndex(real)
    index.put(map_record())
    with pytest.raises(ValueError, match="linked"):
        index.export("fact", SHA, link / "export")


def test_empty_causal_heatmap_does_not_claim_zero_effect() -> None:
    record = {**map_record(), "kind": "localization", "layers": [{"layer": 0, "mean_effect": None}]}
    assert "No identified layer values" in heatmap(record)


def test_neuron_method_names_upstream_rows_and_historical_method_names_columns() -> None:
    assert touched_tensors(fake_runtime(), "ablate-neurons", localized()) == [
        "model.layers.0.mlp.up_proj.weight"
    ]
    assert touched_tensors(fake_runtime(), "historical-columns", localized()) == [
        "model.layers.0.mlp.down_proj.weight"
    ]


@pytest.mark.parametrize(
    "mutation", ["checkpoint", "neuron", "negative", "duplicate", "architecture"]
)
def test_operator_selection_requires_positive_matching_bounded_evidence(mutation: str) -> None:
    runtime, record = fake_runtime(), localized()
    if mutation == "checkpoint":
        record["checkpoint_fingerprint"] = "c" * 64
    elif mutation == "neuron":
        record["selected"][0]["neurons"] = [4]
    elif mutation == "negative":
        record["layers"][0]["mean_effect"] = -0.1
    elif mutation == "duplicate":
        record["selected"].append(record["selected"][0])
    else:
        runtime.model.config.model_type = "unsupported"
    with pytest.raises(ValueError):
        touched_tensors(runtime, "ablate-neurons", record)


def test_operator_cross_document_identity_check_precedes_tensor_work() -> None:
    loc, fp = localized(), map_record()
    loc["fingerprint_sha256"] = record_digest(fp)
    fp["layers"]["0"]["direction"][0] = 10.0
    with pytest.raises(ValueError, match="identity mismatch"):
        apply_method(fake_runtime(), "ablate-direction", loc, fp)


class SmallArray:
    shape = (2,)
    dtype = "float32"

    def __init__(self, values: list[float]):
        self.values = values

    def __sub__(self, other: SmallArray) -> SmallArray:
        return SmallArray([a - b for a, b in zip(self.values, other.values, strict=True)])


def test_relative_bound_rejects_large_change_instead_of_silent_clipping(monkeypatch: Any) -> None:
    from lacunai.erase import methods

    monkeypatch.setattr(methods.tensors, "_square_norm", lambda a: sum(v * v for v in a.values))
    monkeypatch.setattr(methods.tensors, "tensor_hash", lambda a: repr(a.values))
    original, changed = SmallArray([1.0, 0.0]), SmallArray([2.0, 0.0])
    with pytest.raises(ValueError, match="exceeds bound"):
        validate_replacements({"W": original}, {"W": changed}, 0.25)
    assert changed.values == [2.0, 0.0]
    accepted = validate_replacements({"W": original}, {"W": SmallArray([1.1, 0.0])}, 0.25)
    assert accepted["W"]["relative_frobenius"] == pytest.approx(0.1)


@pytest.mark.parametrize("bound", [0.0, -1.0, 1.1, math.nan])
def test_invalid_bound_is_rejected(bound: float) -> None:
    with pytest.raises(ValueError, match="invalid relative-change"):
        validate_replacements({}, {}, bound)


def test_real_torch_neuron_ablation_and_direction_leave_source_unchanged() -> None:
    torch = pytest.importorskip("torch")
    from lacunai.erase.methods import apply_method

    mlp = torch.nn.Module()
    mlp.up_proj = torch.nn.Linear(2, 4, bias=False)
    mlp.down_proj = torch.nn.Linear(4, 2, bias=False)
    model = torch.nn.Module()
    model.add_module("mlp", mlp)
    model.config = SimpleNamespace(model_type="qwen2")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    runtime = SimpleNamespace(
        model=model,
        fingerprint=SHA,
        layer_ids=[0],
        mlp=lambda layer: mlp,
        check=lambda: None,
        storage_dtypes={name: "F32" for name in model.state_dict()},
    )
    loc, fp = localized(), map_record()
    fp["layers"]["0"]["direction"] = [1.0, 0.0, 0.0, 0.0]
    loc["fingerprint_sha256"] = record_digest(fp)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = apply_method(runtime, "ablate-neurons", loc, fp, strength=0.2, max_relative_change=1.0)
    assert torch.equal(
        result["replacements"]["mlp.up_proj.weight"][1], before["mlp.up_proj.weight"][1] * 0.8
    )
    result = apply_method(
        runtime, "ablate-direction", loc, fp, strength=0.2, max_relative_change=1.0
    )
    assert torch.allclose(
        result["replacements"]["mlp.down_proj.weight"][:, 0],
        before["mlp.down_proj.weight"][:, 0] * 0.8,
    )
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_fingerprint_retains_only_compact_last_token_storage(monkeypatch: Any) -> None:
    torch = pytest.importorskip("torch")
    from importlib import import_module

    module = import_module("lacunai.map.fingerprints")
    real_stack = torch.stack
    storage_checks = []

    def checked_stack(values: Any) -> Any:
        for value in values:
            storage_checks.append(value.untyped_storage().nbytes())
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        return real_stack(values)

    monkeypatch.setattr(torch, "stack", checked_stack)
    monkeypatch.setattr(module, "capture", lambda *args: {0: torch.arange(32.0).reshape(1, 8, 4)})
    runtime = fake_runtime()
    runtime.progress = lambda *args, **kwargs: None
    spec = SimpleNamespace(
        layers=[0],
        top_n=2,
        mapping=[SimpleNamespace(prompt="development")],
        background=["background"],
        target_id="fact",
        digest="b" * 64,
    )
    result = module.fingerprint(runtime, spec)
    assert result["layers"]["0"]["key_mean"] == [28.0, 29.0, 30.0, 31.0]
    assert storage_checks


def test_source_precision_rounding_precedes_bounds_and_noop_checks() -> None:
    torch = pytest.importorskip("torch")
    from lacunai.erase.methods import round_to_storage

    original = {"w": torch.tensor([1.0])}
    tiny = round_to_storage({"w": torch.tensor([1.0001])}, {"w": "BF16"})
    with pytest.raises(ValueError, match="no-op"):
        validate_replacements(original, tiny, 0.01)
    # FP16 rounding increases this proposed change from <0.0008 to 0.0009765625.
    rounded = round_to_storage({"w": torch.tensor([1.00079])}, {"w": "F16"})
    with pytest.raises(ValueError, match="exceeds bound"):
        validate_replacements(original, rounded, 0.0008)
    metrics = validate_replacements(original, rounded, 0.001)
    assert metrics["w"]["relative_frobenius"] == float(rounded["w"][0] - 1)
    assert torch.equal(rounded["w"], rounded["w"].half().float())
    with pytest.raises(ValueError, match="storage dtype"):
        round_to_storage(original, {})


def test_missing_storage_dtype_refused_before_tensor_work() -> None:
    loc, fp = localized(), map_record()
    loc["fingerprint_sha256"] = record_digest(fp)
    with pytest.raises(ValueError, match="storage dtype"):
        apply_method(fake_runtime(), "ablate-neurons", loc, fp)
