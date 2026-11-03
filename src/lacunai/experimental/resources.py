# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Measured host admission and a watchdog restricted to its own worker."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any


class ResourceAbort(RuntimeError):
    """The experiment cannot continue inside its approved resource bounds."""


def host_sample(root: Path) -> dict[str, Any]:
    """Read host pressure, not guest names, environment variables or command lines."""
    mem: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.split()[0]) * 1024
    pressure: dict[str, dict[str, float]] = {}
    for name in ("memory", "io", "cpu"):
        rows: dict[str, float] = {}
        for line in Path(f"/proc/pressure/{name}").read_text().splitlines():
            fields = line.split()
            rows[fields[0]] = float(dict(x.split("=") for x in fields[1:])["avg10"])
        pressure[name] = rows
    workers: dict[str, list[int]] = {"kvm": [], "runner": [], "llama-server": []}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            comm = (proc / "comm").read_text().strip()
        except (OSError, UnicodeError):
            continue
        if comm in ("kvm", "llama-server"):
            workers[comm].append(int(proc.name))
        elif comm.startswith("Runner.Worker"):
            workers["runner"].append(int(proc.name))
    for values in workers.values():
        values.sort()
    return {
        "monotonic": time.monotonic(),
        "available_bytes": mem["MemAvailable"],
        "swap_used_bytes": mem["SwapTotal"] - mem["SwapFree"],
        "root_free_bytes": shutil.disk_usage("/").free,
        "bulk_free_bytes": shutil.disk_usage(root).free,
        "load1": os.getloadavg()[0],
        "pressure": pressure,
        "workers": workers,
    }


def tree_bytes(root: Path) -> int:
    """Count all allocated artifacts, including partials; never follow symlinks."""
    total = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        parent = Path(current)
        dirs[:] = [x for x in dirs if not (parent / x).is_symlink()]
        for name in files:
            path = parent / name
            if not path.is_symlink():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    continue  # atomic metadata rename raced this read
    return total


def _artifact_budget_root(root: Path) -> Path | None:
    """Bind an optional shared retry budget to the existing approved ancestor."""
    value = os.environ.get("NEURAILYZER_ARTIFACT_BUDGET_ROOT")
    if value is None:
        return None
    candidate = Path(value)
    try:
        valid = (
            candidate.is_absolute()
            and candidate.name == "neurailyzer-pilot-20261102"
            and candidate.is_dir()
            and not any(part.is_symlink() for part in (candidate, *candidate.parents))
            and candidate.resolve(strict=True) == candidate
            and root.resolve(strict=True).is_relative_to(candidate)
        )
    except (OSError, RuntimeError) as exc:
        raise ResourceAbort("invalid artifact budget root") from exc
    if not valid:
        raise ResourceAbort("invalid artifact budget root: expected real approved ancestor")
    return candidate


def pressure_reasons(
    sample: dict[str, Any], initial: dict[str, Any], limits: dict[str, Any]
) -> list[str]:
    """Return actual violations; callers decide the declared persistence window."""
    reasons = []
    for field, key in (
        ("available_bytes", "min_host_available_bytes"),
        ("root_free_bytes", "min_root_free_bytes"),
        ("bulk_free_bytes", "min_bulk_free_bytes"),
    ):
        if sample[field] < limits[key]:
            reasons.append(f"{field} below approved floor")
    if sample["load1"] > limits["max_load1"]:
        reasons.append("host load above approved ceiling")
    if sample["pressure"]["memory"]["some"] > limits["memory_psi_some_avg10"]:
        reasons.append("memory pressure above approved ceiling")
    if sample["pressure"]["io"]["full"] > limits["io_psi_full_avg10"]:
        reasons.append("I/O pressure above approved ceiling")
    if sample["swap_used_bytes"] > initial["swap_used_bytes"]:
        reasons.append("host swap use increased")
    if sample["workers"]["kvm"] != initial["workers"]["kvm"]:
        reasons.append("guest process state changed")
    if sample["workers"]["llama-server"] != initial["workers"]["llama-server"]:
        reasons.append("existing inference process state changed")
    if set(sample["workers"]["runner"]) - set(initial["workers"]["runner"]):
        reasons.append("new host runner worker observed")
    return reasons


class Watchdog:
    """Continuously emit truthful progress; resource failure terminates only this worker."""

    def __init__(self, root: Path, contract: dict[str, Any], deadline: int, phase: str):
        self.root = root
        self.artifact_budget_root = _artifact_budget_root(root)
        self.limits = contract["resources"]
        self.deadline = deadline
        self.phase = phase
        self.started = time.monotonic()
        self.last_budget_progress = float("-inf")
        self.current_admission = host_sample(root)
        baseline = root / "host-before.json"
        self.initial = (
            json.loads(baseline.read_text()) if baseline.exists() else self.current_admission
        )
        self.finished = threading.Event()
        self.failures: dict[str, int] = {}
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.log = root / "logs" / f"resources-{phase}.jsonl"
        self.log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    def progress(self, message: str = "active", **values: Any) -> None:
        event = {
            "phase": self.phase,
            "state": message,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            **values,
        }
        print(json.dumps(event, ensure_ascii=False), flush=True)

    def check(self) -> None:
        if time.time() >= self.deadline:
            raise ResourceAbort("overall batch deadline reached")
        show_progress = self.artifact_budget_root is not None and (
            time.monotonic() - self.last_budget_progress >= self.limits["sampling_interval_seconds"]
        )
        if show_progress:
            self.last_budget_progress = time.monotonic()
            self.progress("checking_disk_budget")
        try:
            used = tree_bytes(self.root)
            if used > self.limits["disk_max_bytes"]:
                raise ResourceAbort("aggregate disk budget exceeded")
            aggregate = used
            if self.artifact_budget_root is not None and self.artifact_budget_root != self.root:
                aggregate = tree_bytes(self.artifact_budget_root)
                if aggregate > self.limits["disk_max_bytes"]:
                    raise ResourceAbort("aggregate disk budget exceeded")
        except Exception as exc:
            if show_progress:
                self.progress("disk_budget_failed", error=str(exc))
            raise
        if show_progress:
            self.progress(
                "disk_budget_checked",
                attempt_bytes=used,
                aggregate_bytes=aggregate,
                limit_bytes=self.limits["disk_max_bytes"],
            )

    def __enter__(self) -> Watchdog:
        self.progress("starting")
        try:
            self.check()
            reasons = pressure_reasons(self.current_admission, self.initial, self.limits)
            if reasons:
                raise ResourceAbort("admission refused: " + "; ".join(reasons))
        except Exception as exc:
            self.progress("aborted", error=str(exc))
            raise
        self.thread.start()
        return self

    def _monitor(self) -> None:
        interval = self.limits["sampling_interval_seconds"]
        while not self.finished.wait(interval):
            try:
                self.check()
                sample = host_sample(self.root)
                reasons = pressure_reasons(sample, self.initial, self.limits)
                for reason in set(self.failures) | set(reasons):
                    self.failures[reason] = (
                        self.failures.get(reason, 0) + 1 if reason in reasons else 0
                    )
                sample["reasons"] = reasons
                with self.log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                self.progress("active", pressure_violations=reasons)
                if any(
                    n >= self.limits["pressure_consecutive_samples"] for n in self.failures.values()
                ):
                    raise ResourceAbort("; ".join(reasons))
            except Exception as exc:
                self.progress("aborted", error=str(exc))
                receipt = self.root / "watchdog-abort.json"
                receipt.write_text(
                    json.dumps({"phase": self.phase, "error": str(exc), "verdict": "inconclusive"})
                    + "\n"
                )
                # A Python exception in another thread cannot interrupt a tensor kernel.
                # This exits only this service worker; systemd owns its remaining children.
                os._exit(75)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.finished.set()
        self.thread.join(timeout=2)
        try:
            snapshot = host_sample(self.root)
            path = self.root / "logs" / f"host-after-{self.phase}.json"
            path.write_text(json.dumps(snapshot) + "\n")
        except OSError as readback_error:
            self.progress("host_readback_failed", error=str(readback_error))
            if exc is None:
                raise ResourceAbort("final host readback failed") from readback_error
        self.progress("failed" if exc else "phase_finished", error=str(exc) if exc else None)
