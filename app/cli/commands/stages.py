"""The nine stage commands, plus `op pipeline` to run them all.

Each command is one pipeline stage. They are generated from `STAGE_COMMANDS`
rather than written out nine times: the bodies would be identical, and hand-copying
them is how a flag ends up supported on eight commands out of nine.
"""

from collections.abc import Callable
from typing import Annotated, Final

import typer

from app.artifacts import ArtifactRegistry
from app.cli.render import (
    STATUS_STYLE,
    artifact_json,
    artifact_table,
    outcome_table,
    print_json,
)
from app.pipeline import PipelineEngine, PipelineRun, StageStatus
from app.utils.console import console, err_console

STAGE_COMMANDS: Final[tuple[tuple[str, str, str], ...]] = (
    ("collect", "collect-evidence", "Gather evidence for the run's question."),
    ("brief", "research-brief", "Synthesise the evidence into research briefs."),
    ("cluster", "cluster-pains", "Group the briefs into recurring pains."),
    ("discover", "discover-opportunities", "Turn pain clusters into opportunities."),
    ("market", "analyze-market", "Analyse the market for each opportunity."),
    ("competition", "analyze-competition", "Analyse the competition for each opportunity."),
    ("contradiction", "contradiction-analysis", "Find where the analyses disagree."),
    ("decision", "decision", "Decide on each opportunity."),
    ("interview", "interview-plan", "Plan interviews to validate the decision."),
)
"""(command name, pipeline stage, help text) — the CLI's whole stage surface."""

RunOption = Annotated[str, typer.Option("--run", "-r", help="Run identifier.")]
ForceOption = Annotated[bool, typer.Option("--force", "-f", help="Re-run even if complete.")]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit produced artifacts as JSON.")]


def _engine() -> PipelineEngine:
    return PipelineEngine(ArtifactRegistry())


def run_one_stage(stage: str, run_id: str, *, force: bool, as_json: bool) -> None:
    """Run a single stage and report it, exiting non-zero if it did not succeed."""
    engine = _engine()
    outcome = engine.run_stage(stage, run_id, force=force)

    if as_json:
        registry = engine.registry
        produced = [registry.resolve(ref) for ref in outcome.produced]
        print_json(artifact_json(produced))
    elif outcome.status is StageStatus.COMPLETED:
        produced = [engine.registry.resolve(ref) for ref in outcome.produced]
        console.print(artifact_table(produced, title=f"{stage} — {len(produced)} produced"))
    elif outcome.status is StageStatus.SKIPPED:
        console.print(f"[muted]{stage} already done[/muted] — {outcome.reason}")
        console.print("[muted]re-run it with[/muted] --force")
    else:
        style = STATUS_STYLE.get(outcome.status, "danger")
        err_console.print(f"[{style}]{stage} {outcome.status.value}[/{style}]: {outcome.reason}")

    if not outcome.ok:
        raise typer.Exit(code=1)


def _make_stage_command(stage: str, help_text: str) -> Callable[..., None]:
    """Build one Typer command bound to a stage."""

    def command(
        run_id: RunOption = "default",
        force: ForceOption = False,
        as_json: JsonOption = False,
    ) -> None:
        run_one_stage(stage, run_id, force=force, as_json=as_json)

    command.__doc__ = help_text
    return command


def pipeline(
    run_id: RunOption = "default",
    start_at: Annotated[
        str | None, typer.Option("--start-at", help="First stage to attempt.")
    ] = None,
    stop_after: Annotated[
        str | None, typer.Option("--stop-after", help="Last stage to attempt.")
    ] = None,
    force: ForceOption = False,
    keep_going: Annotated[
        bool, typer.Option("--keep-going", help="Attempt later stages after a failure.")
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Run the whole pipeline, skipping stages already completed for this run.

    Resumable by construction: re-running after a failure picks up where it
    stopped, because a completed stage is one whose artifacts are already on disk.
    """
    result: PipelineRun = _engine().run(
        run_id,
        start_at=start_at,
        stop_after=stop_after,
        force=force,
        stop_on_error=not keep_going,
    )

    if as_json:
        print_json(result.model_dump(mode="json"))
    else:
        console.print(outcome_table(result))
        console.print(
            f"[muted]completed[/muted] {len(result.completed)}  "
            f"[muted]skipped[/muted] {len(result.skipped)}  "
            f"[muted]failed[/muted] {len(result.failed)}"
        )

    if not result.ok:
        raise typer.Exit(code=1)


def register_stage_commands(app: typer.Typer) -> None:
    """Attach the nine stage commands to the root app, in pipeline order."""
    for name, stage, help_text in STAGE_COMMANDS:
        app.command(name=name, help=help_text)(_make_stage_command(stage, help_text))


__all__ = ["STAGE_COMMANDS", "pipeline", "register_stage_commands", "run_one_stage"]
