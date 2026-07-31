"""`op runs` — the ledger, read back.

Everything here is a question the workspace cannot answer on its own. Which
questions have I asked? How long did that run take, and what did it cost? Which
stage failed twice before it took? The artifacts remember conclusions; the ledger
remembers the work.

`op runs sync` is the other half of that bargain: the index is a projection, so
it can always be rebuilt from the artifacts, and a workspace that arrived from
another machine — or one that predates the ledger — catches up with one command.
"""

from datetime import datetime
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict
from rich.markup import escape
from rich.table import Table

from app.artifacts import ArtifactRegistry
from app.cli.render import print_json
from app.models import Run, StageState
from app.storage.ledger import Ledger, ledger_scope
from app.storage.schema import create_all
from app.utils.console import console, err_console

app = typer.Typer(
    name="runs",
    help="Run history from the ledger: what was asked, how it went, what it cost.",
    invoke_without_command=True,
    no_args_is_help=False,
)

_STATUS_STYLE: dict[str, str] = {
    "pending": "muted",
    "running": "warning",
    "completed": "success",
    "failed": "danger",
}

_STATE_STYLE: dict[StageState, str] = {
    StageState.COMPLETED: "success",
    StageState.SKIPPED: "muted",
    StageState.EMPTY: "warning",
    StageState.BLOCKED: "warning",
    StageState.FAILED: "danger",
}

_VERDICT_STYLE: dict[str, str] = {"build": "success", "wait": "warning", "reject": "muted"}

RunOption = Annotated[str | None, typer.Option("--run", "-r", help="Only this run.")]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit JSON and nothing else.")]


class RunSummary(BaseModel):
    """One run, flattened for a table row or a JSON object."""

    model_config = ConfigDict(extra="forbid")

    run: str
    question: str | None
    status: str
    auto: bool
    stage: str | None
    evidence: int
    opportunities: int
    decided: int
    calls: int
    tokens: int
    cost: float
    seconds: float | None
    report: str | None
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None


class AttemptRow(BaseModel):
    """One stage attempt."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    attempt: int
    state: StageState
    produced: int
    reused: int
    detail: str | None
    seconds: float | None


class VerdictRow(BaseModel):
    """What a run decided about one opportunity."""

    model_config = ConfigDict(extra="forbid")

    opportunity: str
    title: str
    verdict: str | None
    confidence: float | None


@app.callback()
def runs(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, help="How many runs.")] = 20,
    as_json: JsonOption = False,
) -> None:
    """List runs, newest first. With a subcommand, defers to it."""
    if ctx.invoked_subcommand is not None:
        return

    create_all()
    with ledger_scope() as ledger:
        rows = [_summarise(ledger, run) for run in ledger.runs.recent(limit=limit)]

    if as_json:
        print_json([row.model_dump(mode="json") for row in rows])
        return
    if not rows:
        err_console.print(
            "[muted]no runs recorded[/muted] — start one with "
            '[stage]op auto "…"[/stage], or index existing work with [stage]op runs sync[/stage]'
        )
        raise typer.Exit(code=1)
    console.print(_runs_table(rows))


@app.command(name="show")
def show(
    run_id: Annotated[str, typer.Argument(help="Which run.")] = "default",
    as_json: JsonOption = False,
) -> None:
    """One run in full: its question, every stage attempt, and its verdicts."""
    create_all()
    with ledger_scope() as ledger:
        run = ledger.runs.get(run_id)
        if run is None:
            err_console.print(
                f"[danger]no run[/danger] {run_id} — "
                f"[muted]index existing work with[/muted] op runs sync"
            )
            raise typer.Exit(code=1)
        summary = _summarise(ledger, run)
        attempts = [
            AttemptRow(
                stage=attempt.stage,
                attempt=attempt.attempt,
                state=attempt.state,
                produced=attempt.produced,
                reused=attempt.reused,
                detail=attempt.detail,
                seconds=attempt.duration_seconds,
            )
            for attempt in ledger.stages.for_run(run_id)
        ]
        verdicts = [
            VerdictRow(
                opportunity=row.artifact_id,
                title=row.title,
                verdict=row.verdict,
                confidence=row.decision_confidence,
            )
            for row in ledger.opportunities.by_run(run_id)
        ]

    if as_json:
        print_json(
            {
                **summary.model_dump(mode="json"),
                "attempts": [row.model_dump(mode="json") for row in attempts],
                "verdicts": [row.model_dump(mode="json") for row in verdicts],
            }
        )
        return

    console.print(_header(summary))
    if attempts:
        console.print(_attempts_table(attempts))
    if verdicts:
        console.print(_verdicts_table(verdicts))


@app.command(name="sync")
def sync(
    run_id: RunOption = None,
    as_json: JsonOption = False,
) -> None:
    """Rebuild the index from the artifacts on disk.

    Safe to run whenever: every column it writes is derived from a file in the
    workspace, so re-projecting updates rows rather than duplicating them. This
    is how a workspace that predates the ledger — or arrived from another
    machine — gets its history back.
    """
    create_all()
    with ledger_scope(registry=ArtifactRegistry()) as ledger:
        counts = ledger.sync_run(run_id) if run_id is not None else ledger.sync_all()

    if as_json:
        print_json(counts.model_dump(mode="json"))
        return
    console.print(
        f"[success]indexed[/success] {counts.runs} run(s)  "
        f"[muted]{counts.evidence} evidence · {counts.opportunities} opportunities "
        f"({counts.decided} decided) · {counts.sources} sources[/muted]"
    )


# ------------------------------------------------------------------ rendering


def _summarise(ledger: Ledger, run: Run) -> RunSummary:
    """Fold a run row together with the counts only the index can answer."""
    opportunities = ledger.opportunities.by_run(run.id)
    return RunSummary(
        run=run.id,
        question=run.question,
        status=run.status.value,
        auto=run.auto,
        stage=run.stage,
        evidence=ledger.evidence.count_for_run(run.id),
        opportunities=len(opportunities),
        decided=sum(1 for row in opportunities if row.verdict),
        calls=run.calls,
        tokens=run.total_tokens,
        cost=run.cost,
        seconds=run.duration_seconds,
        report=run.report_id,
        error=run.error,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


def _runs_table(rows: list[RunSummary]) -> Table:
    """Every run, newest first."""
    table = Table(header_style="stage")
    table.add_column("run", overflow="fold")
    table.add_column("question", overflow="ellipsis", max_width=40)
    table.add_column("status")
    table.add_column("ev", justify="right")
    table.add_column("opps", justify="right")
    table.add_column("decided", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("took", justify="right", style="muted")

    for row in rows:
        style = _STATUS_STYLE.get(row.status, "info")
        table.add_row(
            escape(row.run),
            escape(row.question or "—"),
            f"[{style}]{row.status}[/{style}]",
            str(row.evidence),
            str(row.opportunities),
            str(row.decided),
            f"${row.cost:.2f}",
            _clock(row.seconds),
        )
    return table


def _header(summary: RunSummary) -> str:
    """The run's identity, above its detail."""
    lines = [
        f"[stage]{escape(summary.run)}[/stage]  [muted]{summary.status}[/muted]",
        f"  {escape(summary.question or '(no question recorded)')}",
    ]
    if summary.report:
        lines.append(f"  [muted]report[/muted] {escape(summary.report)}")
    if summary.error:
        lines.append(f"  [danger]{escape(summary.error)}[/danger]")
    lines.append(
        f"  [muted]{summary.calls} calls · {summary.tokens:,} tokens · "
        f"${summary.cost:.2f} · {_clock(summary.seconds)}[/muted]"
    )
    return "\n".join(lines)


def _attempts_table(attempts: list[AttemptRow]) -> Table:
    """Every stage attempt, in the order it happened."""
    table = Table(title="stages", title_style="muted", header_style="stage")
    table.add_column("stage")
    table.add_column("try", justify="right", style="muted")
    table.add_column("state")
    table.add_column("produced", justify="right")
    table.add_column("took", justify="right", style="muted")
    table.add_column("detail", overflow="ellipsis", max_width=44)

    for attempt in attempts:
        style = _STATE_STYLE[attempt.state]
        table.add_row(
            escape(attempt.stage),
            str(attempt.attempt),
            f"[{style}]{attempt.state.value}[/{style}]",
            str(attempt.produced),
            _clock(attempt.seconds),
            escape(attempt.detail or ""),
        )
    return table


def _verdicts_table(verdicts: list[VerdictRow]) -> Table:
    """What the run decided about each opportunity."""
    table = Table(title="verdicts", title_style="muted", header_style="stage")
    table.add_column("opportunity", overflow="ellipsis", max_width=40)
    table.add_column("verdict")
    table.add_column("conf", justify="right")

    for row in verdicts:
        verdict = row.verdict or "—"
        style = _VERDICT_STYLE.get(verdict, "muted")
        table.add_row(
            escape(row.title),
            f"[{style}]{verdict}[/{style}]",
            f"{row.confidence:.2f}" if row.confidence is not None else "—",
        )
    return table


def _clock(seconds: float | None) -> str:
    """A duration a person reads at a glance, or an em dash."""
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m {rest:02d}s" if minutes < 60 else f"{minutes // 60}h {minutes % 60:02d}m"


__all__ = ["app"]
