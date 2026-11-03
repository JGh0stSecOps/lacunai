"""Explicit store-scoped local state wiping; never semantic selective deletion."""

from __future__ import annotations

import json
import sys
import time
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from importlib import import_module
from pathlib import Path
from threading import Event, Thread
from typing import Any


def _prepare(
    config_path: Path, scopes: list[str], protected_paths: list[Path]
) -> tuple[Any, Any, dict[str, Any]]:
    if not config_path or not Path(config_path).is_file():
        raise ValueError("an existing explicit NeurAIlyzer configuration is required")
    if not scopes or len(scopes) != len(set(scopes)) or set(scopes) - {"session", "sandbox"}:
        raise ValueError("only explicit nonempty local session/sandbox scopes are supported")
    try:
        core = import_module("neurailyzer.core")
        config = import_module("neurailyzer.config")
        liveness = import_module("neurailyzer.liveness")
        presets = import_module("neurailyzer.presets")
    except ImportError as exc:
        raise RuntimeError(
            "optional NeurAIlyzer dependency is unavailable; install lacunai[state]"
        ) from exc
    cfg = config.load(config_path)
    roots = tuple(r for scope in scopes for r in cfg.roots_for(scope))
    protected = [p.resolve() for p in protected_paths]
    for root in roots:
        resolved = root.resolve()
        if any(resolved == p or resolved in p.parents or p in resolved.parents for p in protected):
            raise ValueError("state root overlaps a protected checkpoint or workspace")
    statuses = [core.scope_status(cfg, scope) for scope in scopes]
    if not roots or any(not s.configured or not s.available or not s.counted for s in statuses):
        raise ValueError("every requested state scope must be configured and locally available")
    by_preset = {
        pid: tuple(
            r
            for r in presets.expand_existing(
                (*presets.REGISTRY[pid].session, *presets.REGISTRY[pid].sandbox)
            )
            if r in roots
        )
        for pid in cfg.presets
        if pid in presets.REGISTRY
    }
    if liveness.check_enabled(list(cfg.presets), roots, by_preset):
        raise RuntimeError("targeted state store may be active; no force override is provided")
    plans = core.plan_wipe(cfg, scopes)
    if len(plans) != len(scopes) or any(not p.complete for p in plans):
        raise ValueError("state planning is incomplete")
    return (
        core,
        cfg,
        {
            "scopes": scopes,
            "roots": list(map(str, roots)),
            "statuses": [asdict(s) for s in statuses],
            "dry_run": True,
            "limitation": (
                "Store-scoped wipe of all non-kept data; not semantic selective forgetting."
            ),
            "liveness": "best-effort check found no active writer; not proof of inactivity",
        },
    )


def preflight(config_path: Path, scopes: list[str], protected_paths: list[Path]) -> dict[str, Any]:
    return _prepare(config_path, scopes, protected_paths)[2]


def wipe_state(
    config_path: Path, scopes: list[str], protected_paths: list[Path], apply: bool = False
) -> dict[str, Any]:
    core, cfg, plan = _prepare(config_path, scopes, protected_paths)
    if not apply:
        return {**plan, "performed": False}
    snapshots = import_module("neurailyzer.snapshots")
    store = snapshots.SnapshotStore(cfg.snapshot_dir)
    existing = store.list()
    if store.corrupt or len(existing) >= cfg.retention:
        raise RuntimeError("state snapshot retention would prune backups or inventory is corrupt")
    print(
        json.dumps({"phase": "state_snapshot_and_wipe", "state": "active"}),
        file=sys.stderr,
        flush=True,
    )
    # prune(0) is an upstream no-op: concurrent new snapshots cannot trigger deletion.
    stopped = Event()
    started = time.monotonic()

    def heartbeat() -> None:
        while not stopped.wait(5):
            print(
                json.dumps(
                    {
                        "phase": "state_snapshot_and_wipe",
                        "state": "active",
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        with redirect_stdout(sys.stderr):
            report = core.execute_wipe(
                replace(cfg, retention=0), scopes, take_snapshot=True, label="lacunai-state"
            )
        ok = (
            report.snapshot is not None
            and len(report.plans) == len(scopes)
            and set(report.verified) == set(scopes)
            and all(report.verified.values())
            and all(p.complete for p in report.plans)
        )
        if not ok:
            raise RuntimeError("state wipe incomplete; inspect NeurAIlyzer snapshot store")
    except Exception as exc:
        print(
            json.dumps({"phase": "state_snapshot_and_wipe", "state": "failed", "error": str(exc)}),
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        stopped.set()
        worker.join()
    print(
        json.dumps({"phase": "state_snapshot_and_wipe", "state": "complete"}),
        file=sys.stderr,
        flush=True,
    )
    return {
        **plan,
        "dry_run": False,
        "performed": True,
        "verified": report.verified,
        "snapshot_id": report.snapshot.id,
    }
