# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Aggregate retry budgets use retained artifacts, without executing model work."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from lacunai.experimental import resources


@pytest.fixture
def budget_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    outer = tmp_path.resolve() / "neurailyzer-pilot-20261102"
    attempt = outer / "retry" / "neurailyzer-pilot-20261102"
    attempt.mkdir(parents=True)
    (outer / "retained.bin").write_bytes(b"x" * 80)
    (attempt / "attempt.bin").write_bytes(b"y" * 30)
    monkeypatch.setattr(resources, "host_sample", lambda root: {})
    monkeypatch.delenv("NEURAILYZER_ARTIFACT_BUDGET_ROOT", raising=False)
    return outer, attempt


def _watch(root: Path) -> resources.Watchdog:
    contract: dict[str, Any] = {
        "resources": {"disk_max_bytes": 100, "sampling_interval_seconds": 30}
    }
    return resources.Watchdog(root, contract, int(time.time()) + 30, "test")


def test_retained_outer_artifacts_count_when_attempt_alone_fits(
    budget_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    outer, attempt = budget_paths
    assert resources.tree_bytes(attempt) == 30
    assert resources.tree_bytes(outer) == 110
    monkeypatch.setenv("NEURAILYZER_ARTIFACT_BUDGET_ROOT", str(outer))
    with pytest.raises(resources.ResourceAbort, match="aggregate disk budget exceeded"):
        _watch(attempt).check()


def test_default_scope_unchanged_and_explicit_root_allowed(
    budget_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, attempt = budget_paths
    _watch(attempt).check()
    monkeypatch.setenv("NEURAILYZER_ARTIFACT_BUDGET_ROOT", str(attempt))
    _watch(attempt).check()


def test_ancestor_growth_is_recounted_after_initial_admission(
    budget_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    outer, attempt = budget_paths
    retained = outer / "retained.bin"
    retained.write_bytes(b"x" * 40)
    monkeypatch.setenv("NEURAILYZER_ARTIFACT_BUDGET_ROOT", str(outer))
    watch = _watch(attempt)
    watch.check()
    retained.write_bytes(b"x" * 81)
    with pytest.raises(resources.ResourceAbort, match="aggregate disk budget exceeded"):
        watch.check()


@pytest.mark.parametrize(
    "invalid", ["relative", "unrelated", "missing", "wrong_name", "linked", "linked_parent"]
)
def test_budget_root_rejects_paths_outside_canonical_approved_ancestor(
    budget_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    outer, attempt = budget_paths
    if invalid == "relative":
        value = Path("neurailyzer-pilot-20261102")
    elif invalid == "unrelated":
        value = outer.parent / "other" / outer.name
        value.mkdir(parents=True)
    elif invalid == "missing":
        value = outer.parent / "missing" / outer.name
    elif invalid == "wrong_name":
        value = outer.parent
    elif invalid == "linked":
        value = outer.parent / "link" / outer.name
        value.parent.mkdir()
        value.symlink_to(outer, target_is_directory=True)
    else:
        link = outer.parent / "linked-parent"
        link.symlink_to(outer.parent, target_is_directory=True)
        value = link / outer.name
    monkeypatch.setenv("NEURAILYZER_ARTIFACT_BUDGET_ROOT", str(value))
    with pytest.raises(resources.ResourceAbort, match="artifact budget root"):
        _watch(attempt)


def test_budget_root_is_bound_once_and_slow_checks_show_flushed_active_then_failure(
    budget_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    outer, attempt = budget_paths
    monkeypatch.setenv("NEURAILYZER_ARTIFACT_BUDGET_ROOT", str(outer))
    watch = _watch(attempt)
    monkeypatch.setenv("NEURAILYZER_ARTIFACT_BUDGET_ROOT", str(attempt))
    events = []

    def capture(message: str, *, flush: bool) -> None:
        events.append((json.loads(message), flush))

    def measured(path: Path) -> int:
        assert events[0][0]["state"] == "checking_disk_budget"
        assert events[0][1] is True
        return 110 if path == outer else 30

    monkeypatch.setattr(resources, "print", capture, raising=False)
    monkeypatch.setattr(resources, "tree_bytes", measured)
    with pytest.raises(resources.ResourceAbort, match="aggregate disk budget exceeded"):
        watch.check()
    assert events[-1][0]["state"] == "disk_budget_failed"
    assert events[-1][0]["error"] == "aggregate disk budget exceeded"
    assert all(flush for _, flush in events)
