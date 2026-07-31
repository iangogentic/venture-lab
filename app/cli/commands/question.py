"""`op question` — state what a run is trying to find out.

Separate from `op init` because they answer different needs: `init` prepares a
workspace once, `question` starts an investigation and can be run many times
against the same workspace with different run ids.
"""

from typing import Annotated

import typer

from app.artifacts import ArtifactRegistry, Question, QuestionPriority
from app.cli.render import artifact_json, print_json
from app.skills.collect_evidence import derive_queries
from app.storage.schema import create_all
from app.utils.console import console, err_console
from app.utils.paths import get_workspace_paths


def question(
    text: Annotated[str, typer.Argument(help="The research question to investigate.")],
    run_id: Annotated[str, typer.Option("--run", "-r", help="Run identifier.")] = "default",
    priority: Annotated[
        QuestionPriority,
        typer.Option("--priority", "-p", help="How urgent this is."),
    ] = QuestionPriority.MEDIUM,
    scope: Annotated[
        str | None,
        typer.Option("--scope", help="Boundary on what counts as in scope."),
    ] = None,
    query: Annotated[
        list[str] | None,
        typer.Option("--query", "-q", help="Search term to retrieve with. Repeatable."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the Question as JSON.")] = False,
) -> None:
    """Seed a run with the question it exists to answer.

    Everything downstream traces back to this artifact, so it is worth phrasing
    precisely: the pipeline will answer the question asked, not the one intended.
    """
    paths = get_workspace_paths()
    paths.ensure()
    create_all()

    seeded = Question(
        id=Question.make_id(),
        run_id=run_id,
        text=text,
        priority=priority,
        scope=scope,
        search_queries=list(query or derive_queries(text)),
    )
    ArtifactRegistry(paths).save(seeded)

    if as_json:
        print_json(artifact_json([seeded]))
        return

    console.print(f"[success]run[/success] [stage]{run_id}[/stage] [muted]{seeded.id}[/muted]")
    console.print(f"  {seeded.text}")
    # Shown because a bad derivation is the single most likely reason a run finds
    # nothing, and it is far cheaper to correct here than after a full pipeline.
    console.print("[muted]searching for:[/muted] " + " · ".join(seeded.search_queries))
    err_console.print("[muted]next:[/muted] op run   [muted](--query to change terms)[/muted]")


__all__ = ["question"]
