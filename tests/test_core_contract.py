"""Contract checks run without downloads, pretrained models, or tensor libraries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from lacunai.core.model import HFRuntime, checkpoint_inventory
from lacunai.core.spec import (
    MappingProbe,
    Probe,
    RuntimeConfig,
    TargetSpec,
    Thresholds,
    bounded_file,
)


def target_data() -> dict[str, Any]:
    return asdict(
        TargetSpec(
            target_id="synthetic-fact",
            description="Synthetic test",
            mapping=[MappingProbe("Development?", "Corruption?", "nonce")],
            probes=[
                Probe("Held-out direct?", "nonce"),
                Probe("Held-out paraphrase?", "nonce", "paraphrase"),
            ],
            retain=[Probe("Unrelated?", "oak", "retain")],
            background=["Background."],
            perplexity_texts=["Held-out ordinary text."],
        )
    )


def checkpoint(tmp_path: Path, config: dict[str, Any] | None = None) -> Path:
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config or {"model_type": "qwen2"}))
    # Inventory fixture, not a loadable model: no test crosses the loader boundary.
    (root / "model.safetensors").write_bytes(b"inventory-fixture")
    return root


def test_spec_roundtrip_and_frozen_contract_digest(tmp_path: Path) -> None:
    data = target_data()
    path = tmp_path / "target.json"
    path.write_text(json.dumps(data))
    original = TargetSpec.from_path(path)
    assert asdict(original) == data
    data["thresholds"]["maximum_after_recovery"] = 0.5
    assert TargetSpec.from_dict(data).digest != original.digest


@pytest.mark.parametrize(
    "change",
    [
        "overlap",
        "missing_paraphrase",
        "same_corruption",
        "unknown_probe",
        "duplicate_layer",
        "bad_id",
        "unknown_field",
    ],
)
def test_spec_rejects_invalid_contract(change: str) -> None:
    data = target_data()
    if change == "overlap":
        data["probes"][0]["prompt"] = "  DEVELOPMENT?  "
    elif change == "missing_paraphrase":
        data["probes"][1]["kind"] = "direct"
    elif change == "same_corruption":
        data["mapping"][0]["corrupt_prompt"] = "Development?"
    elif change == "unknown_probe":
        data["probes"][0]["kind"] = "invented"
    elif change == "duplicate_layer":
        data["layers"] = [0, 0]
    elif change == "bad_id":
        data["target_id"] = "../escape"
    else:
        data["unrecognized"] = True
    with pytest.raises((ValueError, TypeError)):
        TargetSpec.from_dict(data)


@pytest.mark.parametrize(
    "settings",
    [
        {"device": "cuda"},
        {"dtype": "float16"},
        {"max_seconds": 0},
        {"seed": True},
        {"max_prompt_tokens": 1025},
        {"torch_threads": 0},
        {"max_model_bytes": 0},
    ],
)
def test_runtime_rejects_undeclared_execution(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RuntimeConfig(**settings)


@pytest.mark.parametrize(
    "settings",
    [
        {"maximum_after_recovery": float("nan")},
        {"minimum_baseline_recovery": -1},
        {"maximum_perplexity_ratio": 0.5},
        {"maximum_retain_paired_losses": 0.5},
    ],
)
def test_thresholds_reject_nonfinite_or_invalid_limits(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Thresholds(**settings)


def test_inventory_hashes_bytes_and_reports_real_completion(tmp_path: Path, capsys: Any) -> None:
    root = checkpoint(tmp_path)
    result = checkpoint_inventory(root, 4096)
    assert result == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()
    }
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert events[0]["completed"] == 0
    assert events[-1]["completed"] == events[-1]["total"] == len(result)
    (root / "config.json").write_text('{"model_type":"llama"}')
    assert checkpoint_inventory(root, 4096) != result


@pytest.mark.parametrize("bad", ["model.bin", "model.pt", "model.pth", "remote.py", "data.pkl"])
def test_inventory_refuses_executable_or_pickle_formats(tmp_path: Path, bad: str) -> None:
    root = checkpoint(tmp_path)
    (root / bad).write_bytes(b"not executed")
    with pytest.raises(ValueError, match="inert metadata"):
        checkpoint_inventory(root, 4096)


def test_inventory_and_bounded_file_refuse_links_and_limits(tmp_path: Path) -> None:
    root = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="byte limit"):
        checkpoint_inventory(root, 1)
    linked = tmp_path / "linked"
    linked.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="unredirected"):
        checkpoint_inventory(linked, 4096)
    with pytest.raises(ValueError, match="unredirected"):
        bounded_file(linked / "config.json")
    with pytest.raises(ValueError, match="byte limit"):
        bounded_file(root / "config.json", 1)
    (root / "alias.json").symlink_to(root / "config.json")
    with pytest.raises(ValueError, match="linked checkpoint"):
        checkpoint_inventory(root, 4096)


def test_missing_checkpoint_structure_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local unredirected"):
        checkpoint_inventory(tmp_path / "absent", 4096)
    with pytest.raises(ValueError, match="config.json and safetensors"):
        checkpoint_inventory(tmp_path, 4096)


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "unsupported"},
        {"model_type": "qwen2", "auto_map": {"AutoModel": "remote.Model"}},
    ],
)
def test_unsupported_model_config_refused_before_loader(
    tmp_path: Path, monkeypatch: Any, config: dict[str, Any]
) -> None:
    from lacunai.core import model

    called = []

    def forbidden_loader(*args: Any, **kwargs: Any) -> Any:
        called.append(True)
        pytest.fail("invalid config reached model loader")

    monkeypatch.setattr(model.tensors, "load_target", forbidden_loader)
    with pytest.raises(ValueError, match="built-in Qwen2"):
        HFRuntime(checkpoint(tmp_path, config))
    assert not called


def test_quantized_config_refused_before_loader(tmp_path: Path, monkeypatch: Any) -> None:
    from lacunai.core import model

    def forbidden_loader(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("quantized config reached model loader")

    monkeypatch.setattr(model.tensors, "load_target", forbidden_loader)
    root = checkpoint(
        tmp_path, {"model_type": "qwen2", "quantization_config": {"quant_method": "gptq"}}
    )
    with pytest.raises(ValueError, match="quantized"):
        HFRuntime(root)
