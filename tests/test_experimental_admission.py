# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""The default-off and host-pressure boundaries need negative evidence."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from lacunai.experimental.resources import pressure_reasons
from lacunai.experimental.spec import ExperimentRefused, load_contract, output_path


def test_disabled_precedes_contract_read(tmp_path: Path) -> None:
    with pytest.raises(ExperimentRefused, match="disabled"):
        load_contract(tmp_path / "absent.json")


def test_cli_refuses_model_execution_without_opt_in(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lacunai",
            "experimental",
            "run",
            "--root",
            str(tmp_path),
            "--contract",
            str(tmp_path / "absent.json"),
            "--deadline-epoch",
            "1",
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "experimental execution disabled" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_altered_contract_refused(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text('{"promotion":"automatic"}')
    with pytest.raises(ExperimentRefused, match="digest"):
        load_contract(path, enable=True)


def test_model_and_frozen_inputs_are_protected(tmp_path: Path) -> None:
    for name in ("models/target/model.safetensors", "code/runner.py", "datasets/a.json"):
        with pytest.raises(ExperimentRefused, match="protected"):
            output_path(tmp_path, name)
    assert output_path(tmp_path, "candidates/a") == tmp_path / "candidates/a"
    with pytest.raises(ExperimentRefused, match="escapes"):
        output_path(tmp_path, "../other")


def test_protection_removal_plant(tmp_path: Path) -> None:
    with (
        patch("lacunai.experimental.spec.KeepList.protects", return_value=False),
        patch("lacunai.experimental.spec.KeepList.shelters", return_value=False),
    ):
        assert output_path(tmp_path, "models/target/model.safetensors")
    with pytest.raises(ExperimentRefused, match="protected"):
        output_path(tmp_path, "models/target/model.safetensors")


def test_host_pressure_and_workload_change() -> None:
    limits = dict(
        min_host_available_bytes=24,
        min_root_free_bytes=8,
        min_bulk_free_bytes=100,
        max_load1=20,
        memory_psi_some_avg10=1,
        io_psi_full_avg10=5,
    )
    sample = dict(
        available_bytes=40,
        root_free_bytes=12,
        bulk_free_bytes=200,
        load1=10,
        pressure={"memory": {"some": 0}, "io": {"full": 0}},
        swap_used_bytes=10,
        workers={"kvm": [1], "llama-server": [2], "runner": []},
    )
    assert pressure_reasons(sample, sample, limits) == []
    altered = copy.deepcopy(sample)
    altered["available_bytes"] = 23
    altered["workers"]["kvm"] = []
    altered["workers"]["runner"] = [9]
    result = pressure_reasons(altered, sample, limits)
    assert "available_bytes below approved floor" in result
    assert "guest process state changed" in result
    assert "new host runner worker observed" in result
