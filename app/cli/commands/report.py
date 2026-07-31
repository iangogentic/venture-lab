"""`op report` — compose a run's findings into one human-readable document.

A single command rather than a sub-app: there is one verb here. Composing sits
deliberately outside the pipeline — a report is a derived view, safe to
regenerate whenever a human wants one, and nothing downstream consumes it. See
`app/skills/compose_report.py` for the full reasoning, and
`app/pipeline/reporting.py` for the rule this shares with `op auto`.
"""

from pathlib import Path
from typing import Annotated

import typer

from app.artifacts import ArtifactRegistry
from app.cli.render import print_json, single_artifact_json
from app.pipeline.reporting import ReportUnavailableError, compose_report
from app.utils.console import console, err_console
from app.utils.errors import OpportunityEngineError


def report(
    run_id: Annotated[str, typer.Option("--run", "-r", help="Run identifier.")] = "default",
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Re-compose, superseding the prior report.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the Report artifact as JSON.")
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the report body to this file instead of stdout."),
    ] = None,
) -> None:
    """Compose the run's briefs, opportunities and decisions into one report.

    Runs the `compose-report` skill directly — this is not a pipeline stage, so
    `op pipeline` never triggers it. Without `--out` the Markdown body is printed;
    with it, the body is written to the file. `--json` emits the artifact instead.
    """
    try:
        composition = compose_report(run_id, registry=ArtifactRegistry(), force=force)
    except ReportUnavailableError as exc:
        err_console.print(
            f"[danger]nothing to report for run[/danger] {run_id} — "
            f"missing output from {', '.join(exc.stages)}; "
            f"run the pipeline through [stage]{exc.stages[-1]}[/stage] first"
        )
        raise typer.Exit(code=1) from exc
    except OpportunityEngineError as exc:
        err_console.print(f"[danger]compose-report failed[/danger]: {exc}")
        raise typer.Exit(code=1) from exc

    composed = composition.report
    if not composition.composed and not as_json and out is None:
        console.print(f"[muted]report already composed[/muted] — {composed.id} for run {run_id}")
        console.print("[muted]re-compose with[/muted] --force")
        return

    if as_json:
        print_json(single_artifact_json(composed))
        return
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(composed.body, encoding="utf-8")
        console.print(f"[success]report written[/success] {out} [muted]({composed.id})[/muted]")
        return
    # Markup off: the body is the model's Markdown, and `[bracketed]` prose must
    # reach the terminal as written, not be eaten as Rich styling.
    console.print(composed.body, markup=False, highlight=False)


__all__ = ["report"]
