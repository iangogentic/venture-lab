"""Root Typer application — the `op` command.

Flat commands, not groups: the nine stage commands mirror the pipeline one-to-one,
so `op brief` reads the way the pipeline diagram does.
"""

from typing import Annotated

import typer

from app import __version__
from app.cli.commands import benchmark as benchmark_cmd
from app.cli.commands.auto import auto
from app.cli.commands.calls import calls
from app.cli.commands.doctor import doctor
from app.cli.commands.graph import graph_show, impact, why
from app.cli.commands.leads import app as leads_app
from app.cli.commands.lifecycle import init
from app.cli.commands.query import inspect, list_artifacts, show
from app.cli.commands.question import question
from app.cli.commands.recall import recall
from app.cli.commands.report import report
from app.cli.commands.routes import routes
from app.cli.commands.runs import app as runs_app
from app.cli.commands.stages import pipeline, register_stage_commands
from app.cli.commands.validate import app as validate_app
from app.cli.commands.venture import app as venture_app
from app.config import get_settings
from app.utils.console import console
from app.utils.logging import configure_logging

app = typer.Typer(
    name="op",
    help="Opportunity Engine — turn a question into evidence, opportunities, and a decision.",
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
)

app.command(name="init")(init)
app.command(name="question")(question)
# First, because it is the whole pipeline in one command and what most sessions
# actually want; the nine stage commands below are the same engine, hand-driven.
app.command(name="auto")(auto)
register_stage_commands(app)
app.command(name="pipeline")(pipeline)
# `op run` is the name the workflow actually reaches for; `op pipeline` stays
# as the descriptive spelling.
app.command(name="run")(pipeline)
app.command(name="inspect")(inspect)
app.add_typer(runs_app, name="runs")
app.command(name="graph")(graph_show)
app.command(name="why")(why)
app.command(name="impact")(impact)
app.command(name="doctor")(doctor)
app.command(name="routes")(routes)
app.command(name="calls")(calls)
app.add_typer(benchmark_cmd.app, name="benchmark")
app.add_typer(leads_app, name="leads")
app.command(name="report")(report)
app.add_typer(validate_app, name="validate")
app.add_typer(venture_app, name="venture")
app.command(name="list")(list_artifacts)
app.command(name="show")(show)
app.command(name="recall")(recall)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"op [success]{__version__}[/success]")
        raise typer.Exit()


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Global options applied before any subcommand runs."""
    settings = get_settings()
    configure_logging("DEBUG" if verbose else settings.log_level)


@app.command()
def config() -> None:
    """Show the resolved configuration. Secrets are redacted."""
    settings = get_settings()
    for key, value in settings.model_dump().items():
        console.print(f"[muted]{key}[/muted] = {value}")


def main() -> None:
    """Console-script entry point."""
    app()


__all__ = ["app", "main"]
