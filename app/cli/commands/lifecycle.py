"""`op init` — prepare a workspace and optionally seed a run with its Question."""

from typing import Annotated

import typer

from app.artifacts import ArtifactRegistry, Question, QuestionPriority
from app.cli.render import artifact_json, print_json
from app.storage.schema import create_all
from app.utils.console import console, err_console
from app.utils.paths import get_workspace_paths


def init(
    question: Annotated[
        str | None,
        typer.Argument(help="Research question to seed the run with."),
    ] = None,
    run_id: Annotated[str, typer.Option("--run", "-r", help="Run identifier.")] = "default",
    priority: Annotated[
        QuestionPriority,
        typer.Option("--priority", "-p", help="How urgent the question is."),
    ] = QuestionPriority.MEDIUM,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the Question as JSON.")] = False,
) -> None:
    """Create the workspace and ledger, and optionally seed a run.

    Everything downstream is scoped to a run id, and every run starts from a
    Question — so seeding one here is what makes `op collect` have something to do.
    """
    paths = get_workspace_paths()
    paths.ensure()
    create_all()

    # In --json mode stdout must be nothing but the payload, so progress chatter
    # goes to stderr where a pipe into `jq` will not choke on it.
    status = err_console if as_json else console
    status.print(f"[success]workspace ready[/success] [muted]{paths.root}[/muted]")

    if question is None:
        status.print('[muted]no question given — seed one with[/muted] op init "…"')
        return

    seeded = Question(
        id=Question.make_id(),
        run_id=run_id,
        text=question,
        priority=priority,
    )
    ArtifactRegistry(paths).save(seeded)

    if as_json:
        print_json(artifact_json([seeded]))
        return

    console.print(
        f"[success]seeded run[/success] [stage]{run_id}[/stage] "
        f"[muted]{seeded.id}[/muted]\n  {seeded.text}"
    )


__all__ = ["init"]
