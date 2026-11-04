"""Public CLI; importing it never loads an ML runtime or starts an experiment."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .experimental.cli import app as experimental_app

app = typer.Typer(
    help="LacunAI: bounded unlearning research. Model execution is disabled by default.",
    no_args_is_help=True,
)
app.add_typer(experimental_app, name="experimental", help="Historical experimental batch adapter.")


@app.command("validate-spec")
def validate_spec_command(
    specification: Path = typer.Argument(..., help="Current-session supplied JSON specification."),
    diagnostics: Path = typer.Option(
        ..., help="Development diagnostics for the source checkpoint."
    ),
    contract: Path = typer.Option(..., help="Exact frozen experiment contract."),
    expected_sha256: str = typer.Option(
        ..., help="SHA-256 of the independently frozen spec bytes."
    ),
    enable_experimental: bool = typer.Option(False, help="Explicitly enable validation only."),
) -> None:
    """Validate a frozen edit specification; do not load, edit or evaluate any model."""
    from .validation import validate_supplied_spec

    typer.echo(json.dumps({"phase": "specification_validation", "status": "active"}), err=True)
    try:
        receipt = validate_supplied_spec(
            specification,
            diagnostics,
            contract,
            expected_sha256=expected_sha256,
            enable=enable_experimental,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        typer.echo(
            json.dumps(
                {"phase": "specification_validation", "status": "refused", "error": str(exc)}
            ),
            err=True,
        )
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(receipt, sort_keys=True))


@app.callback()
def main(ctx: typer.Context, workspace: Path = typer.Option(Path("~/.lacunai"))) -> None:
    ctx.obj = workspace.expanduser()


def _invoke(ctx: typer.Context, operation: str, **kwargs: object) -> None:
    from .mcp_server import invoke

    typer.echo(json.dumps({"phase": operation, "state": "active"}), err=True)
    result = invoke(ctx.obj, operation, **kwargs)
    typer.echo(json.dumps(result, sort_keys=True, allow_nan=False))
    typer.echo(
        json.dumps({"phase": operation, "state": "failed" if result.get("error") else "complete"}),
        err=True,
    )
    if result.get("error"):
        raise typer.Exit(2)
    if (
        operation == "verify" or (operation in {"erase", "forget"} and kwargs.get("apply"))
    ) and result.get("verdict") != "passed_the_stated_experimental_contract":
        raise typer.Exit(1)


@app.command("map")
def map_command(
    ctx: typer.Context,
    model: Path = typer.Option(...),
    target: Path = typer.Option(...),
    max_seconds: int = 3600,
    seed: int = 42,
    threads: int = 2,
) -> None:
    _invoke(
        ctx, "map", model=model, target=target, max_seconds=max_seconds, seed=seed, threads=threads
    )


@app.command("locate")
def locate_command(
    ctx: typer.Context,
    model: Path = typer.Option(...),
    target: Path = typer.Option(...),
    max_seconds: int = 3600,
    seed: int = 42,
    threads: int = 2,
) -> None:
    _invoke(
        ctx,
        "locate",
        model=model,
        target=target,
        max_seconds=max_seconds,
        seed=seed,
        threads=threads,
    )


@app.command("erase")
def erase_command(
    ctx: typer.Context,
    model: Path = typer.Option(...),
    target: Path = typer.Option(...),
    method: str = typer.Option(...),
    output: Path | None = None,
    apply: bool = False,
    in_place: bool = False,
    strength: float = 1.0,
    max_relative_change: float = 0.25,
    steps: int = 12,
    gradient_data: Path | None = None,
    max_seconds: int = 3600,
    seed: int = 42,
    threads: int = 2,
) -> None:
    _invoke(
        ctx,
        "erase",
        model=model,
        target=target,
        method=method,
        output=output,
        apply=apply,
        in_place=in_place,
        strength=strength,
        max_relative_change=max_relative_change,
        steps=steps,
        gradient_data=gradient_data,
        max_seconds=max_seconds,
        seed=seed,
        threads=threads,
    )


@app.command("forget")
def forget_command(
    ctx: typer.Context,
    model: Path = typer.Option(...),
    target: Path = typer.Option(...),
    method: str = typer.Option(...),
    output: Path | None = None,
    apply: bool = False,
    in_place: bool = False,
    strength: float = 1.0,
    max_relative_change: float = 0.25,
    steps: int = 12,
    gradient_data: Path | None = None,
    max_seconds: int = 3600,
    seed: int = 42,
    threads: int = 2,
    state_config: Path | None = None,
    state_scope: list[str] = typer.Option([]),
) -> None:
    _invoke(
        ctx,
        "forget",
        model=model,
        target=target,
        method=method,
        output=output,
        apply=apply,
        in_place=in_place,
        strength=strength,
        max_relative_change=max_relative_change,
        steps=steps,
        gradient_data=gradient_data,
        max_seconds=max_seconds,
        seed=seed,
        threads=threads,
        state_config=state_config,
        state_scopes=state_scope,
    )


@app.command("verify")
def verify_command(
    ctx: typer.Context, run_id: str, max_seconds: int = 3600, seed: int = 42, threads: int = 2
) -> None:
    _invoke(ctx, "verify", run_id=run_id, max_seconds=max_seconds, seed=seed, threads=threads)


@app.command("restore")
def restore_command(ctx: typer.Context, run_id: str, apply: bool = False) -> None:
    _invoke(ctx, "restore", run_id=run_id, apply=apply)


@app.command("runs")
def runs_command(ctx: typer.Context) -> None:
    _invoke(ctx, "runs")


mcp_app = typer.Typer(help="Local stdio MCP interface.")
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("serve")
def mcp_serve(ctx: typer.Context) -> None:
    from .mcp_server import build_server

    build_server(ctx.obj).run("stdio")
