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
app.add_typer(experimental_app, name="experimental")


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

    typer.echo(json.dumps({"phase": "specification_validation", "status": "active"}))
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
