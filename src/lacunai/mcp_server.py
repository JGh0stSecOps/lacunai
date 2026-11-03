"""Lazy MCP transport over the same service used by the command line."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def invoke(workspace: Path, operation: str, **kwargs: Any) -> dict[str, Any]:
    try:
        from .core.service import Service
        from .core.spec import RuntimeConfig

        if operation not in {"map", "locate", "erase", "verify", "restore", "runs", "forget"}:
            raise ValueError("unknown operation")
        if operation in {"map", "locate", "erase", "forget", "verify"}:
            kwargs["config"] = RuntimeConfig(
                max_seconds=kwargs.pop("max_seconds", 3600),
                seed=kwargs.pop("seed", 42),
                torch_threads=kwargs.pop("threads", 2),
            )
        for name in ("model", "target", "output", "state_config"):
            if kwargs.get(name) is not None:
                kwargs[name] = Path(kwargs[name]).expanduser()
        return dict(getattr(Service(workspace), operation)(**kwargs))
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        return {"status": "failed", "error": str(exc), "error_type": type(exc).__name__}


def build_server(workspace: Path) -> Any:
    from mcp.server import MCPServer

    server = MCPServer(
        name="LacunAI",
        version="0.1.0",
        instructions=(
            "Local experimental weight edits. apply defaults false. "
            "Verification measures recoverability, not absence."
        ),
    )

    @server.tool()
    def lacunai_map(
        model: str, target: str, max_seconds: int = 3600, seed: int = 42, threads: int = 2
    ) -> dict[str, Any]:
        return invoke(workspace, "map", **{k: v for k, v in locals().items() if k != "workspace"})

    @server.tool()
    def lacunai_locate(
        model: str, target: str, max_seconds: int = 3600, seed: int = 42, threads: int = 2
    ) -> dict[str, Any]:
        return invoke(
            workspace, "locate", **{k: v for k, v in locals().items() if k != "workspace"}
        )

    @server.tool()
    def lacunai_erase(
        model: str,
        target: str,
        method: str,
        output: str | None = None,
        apply: bool = False,
        in_place: bool = False,
        strength: float = 1.0,
        max_relative_change: float = 0.25,
        steps: int = 12,
        max_seconds: int = 3600,
        seed: int = 42,
        threads: int = 2,
    ) -> dict[str, Any]:
        return invoke(workspace, "erase", **{k: v for k, v in locals().items() if k != "workspace"})

    @server.tool()
    def lacunai_forget(
        model: str,
        target: str,
        method: str,
        output: str | None = None,
        apply: bool = False,
        in_place: bool = False,
        strength: float = 1.0,
        max_relative_change: float = 0.25,
        steps: int = 12,
        max_seconds: int = 3600,
        seed: int = 42,
        threads: int = 2,
        state_config: str | None = None,
        state_scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        return invoke(
            workspace, "forget", **{k: v for k, v in locals().items() if k != "workspace"}
        )

    @server.tool()
    def lacunai_verify(
        run_id: str, max_seconds: int = 3600, seed: int = 42, threads: int = 2
    ) -> dict[str, Any]:
        return invoke(
            workspace, "verify", **{k: v for k, v in locals().items() if k != "workspace"}
        )

    @server.tool()
    def lacunai_restore(run_id: str, apply: bool = False) -> dict[str, Any]:
        return invoke(
            workspace, "restore", **{k: v for k, v in locals().items() if k != "workspace"}
        )

    @server.tool()
    def lacunai_runs() -> dict[str, Any]:
        return invoke(workspace, "runs")

    return server
