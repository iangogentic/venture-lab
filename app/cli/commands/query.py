"""`op inspect`, `op list`, `op show` — read what is in the workspace."""

from typing import Annotated

import typer

from app.artifacts import ArtifactKind, ArtifactRegistry
from app.cli.render import (
    artifact_json,
    artifact_table,
    counts_table,
    print_json,
    single_artifact_json,
    status_table,
)
from app.pipeline import STAGE_ORDER, PipelineEngine
from app.utils.console import console, err_console

RunOption = Annotated[str, typer.Option("--run", "-r", help="Run identifier.")]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table.")]


def inspect(
    run_id: RunOption = "default",
    as_json: JsonOption = False,
) -> None:
    """Show which pipeline stages are done for a run, and what remains."""
    engine = PipelineEngine(ArtifactRegistry())
    status = engine.status(run_id)
    # Superseded and archived artifacts are history, not progress — count work.
    counts = {
        stage: len(engine.produced_of(engine.skill_for(stage).produces, run_id))
        for stage in STAGE_ORDER
    }

    if as_json:
        print_json(
            {
                "run_id": run_id,
                "stages": [
                    {"stage": stage, "complete": status[stage], "artifacts": counts[stage]}
                    for stage in STAGE_ORDER
                ],
                "pending": list(engine.pending(run_id)),
            }
        )
        return

    console.print(status_table(status, counts, run_id))

    pending = engine.pending(run_id)
    if pending:
        console.print(f"[muted]next:[/muted] [stage]{pending[0]}[/stage]")
    else:
        console.print("[success]pipeline complete[/success]")


def list_artifacts(
    kind: Annotated[
        ArtifactKind | None,
        typer.Argument(help="Artifact kind to list. Omit for a summary of every kind."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run", "-r", help="Only artifacts from this run."),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", "-n", help="Cap the rows.")] = None,
    as_json: JsonOption = False,
) -> None:
    """List stored artifacts, or summarise how many of each kind exist."""
    registry = ArtifactRegistry()

    if kind is None:
        counts = {k: len(registry.find_by_type(k, run_id=run_id)) for k in ArtifactKind}
        if as_json:
            print_json({k.value: n for k, n in counts.items()})
            return
        console.print(counts_table(counts))
        return

    artifacts = registry.find_by_type(kind, run_id=run_id, limit=limit)

    if as_json:
        print_json(artifact_json(artifacts))
        return

    if not artifacts:
        console.print(f"[muted]no {kind.value} artifacts[/muted]")
        return

    console.print(artifact_table(artifacts, title=f"{len(artifacts)} {kind.value}"))


def show(
    artifact_id: Annotated[str, typer.Argument(help="Artifact id to display.")],
    version: Annotated[
        int | None,
        typer.Option(
            "--version", "-v", help="Show an archived revision instead of the current one."
        ),
    ] = None,
    as_table: Annotated[
        bool, typer.Option("--table", help="Render a summary row instead of the full JSON.")
    ] = False,
) -> None:
    """Print one artifact as JSON. The default output is the artifact itself."""
    registry = ArtifactRegistry()

    located = registry.locate(artifact_id)
    if located is None:
        err_console.print(f"[danger]no artifact[/danger] {artifact_id}")
        raise typer.Exit(code=1)

    artifact = (
        registry.load(located.kind, located.id)
        if version is None
        else registry.load_version(located.kind, located.id, version)
    )

    if as_table:
        console.print(artifact_table([artifact]))
        return

    print_json(single_artifact_json(artifact))


__all__ = ["inspect", "list_artifacts", "show"]
