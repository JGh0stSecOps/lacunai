"""Removing an overall deadline must preserve resource protections."""

from pathlib import Path

import pytest

from lacunai.core.model import HFRuntime
from lacunai.core.spec import RuntimeConfig
from lacunai.experimental import resources


def test_runtime_explicit_none_has_no_elapsed_cutoff():
    runtime = HFRuntime.__new__(HFRuntime)
    runtime.started = -1e20
    runtime.config = RuntimeConfig(max_seconds=None)
    runtime.check()
    runtime.config = RuntimeConfig(max_seconds=1)
    with pytest.raises(TimeoutError):
        runtime.check()


def test_watchdog_none_retains_disk_guard(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(resources, "host_sample", lambda root: {})
    monkeypatch.delenv("NEURAILYZER_ARTIFACT_BUDGET_ROOT", raising=False)
    (tmp_path / "existing").write_bytes(b"x" * 10)
    limits = {"disk_max_bytes": 20, "sampling_interval_seconds": 5}
    watch = resources.Watchdog(tmp_path, {"resources": limits}, None, "test")
    watch.check()
    (tmp_path / "growth").write_bytes(b"x" * 11)
    with pytest.raises(resources.ResourceAbort, match="disk budget"):
        watch.check()


def test_expired_explicit_deadline_still_refuses(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(resources, "host_sample", lambda root: {})
    monkeypatch.delenv("NEURAILYZER_ARTIFACT_BUDGET_ROOT", raising=False)
    watch = resources.Watchdog(tmp_path, {"resources": {}}, 0, "test")
    with pytest.raises(resources.ResourceAbort, match="deadline"):
        watch.check()
