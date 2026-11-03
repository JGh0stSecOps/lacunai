# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Lazy, default-off CLI registration for the frozen historical experiment."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Opt-in bounded research; no wipe, deployment or automatic promotion.")


@app.command("run")
def run_command(
    root: Path = typer.Option(..., help="Approved isolated experiment directory."),
    contract: Path = typer.Option(..., help="Exact approved contract file."),
    deadline_epoch: int = typer.Option(
        ..., help="Fixed overall deadline from the batch supervisor."
    ),
    enable_experimental: bool = typer.Option(False, help="Explicitly enable model execution."),
) -> None:
    """Execute only the approved contract; disabled unless explicitly enabled."""
    from .batch import main

    args = [
        "run",
        "--root",
        str(root),
        "--contract",
        str(contract),
        "--deadline-epoch",
        str(deadline_epoch),
    ]
    if enable_experimental:
        args.append("--enable-experimental")
    try:
        code = main(args)
    except ValueError as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(2) from exc
    raise typer.Exit(code)
