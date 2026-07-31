"""`op calls` — what the pipeline has spent, and on what."""

from typing import Annotated

import typer
from rich.table import Table

from app.cli.render import print_json
from app.llm.telemetry import CallRecord, CallSummary, default_sink, summarise
from app.utils.console import console, err_console

GroupOption = Annotated[
    str,
    typer.Option("--by", "-b", help="Group by: skill, model, capability, tier or run_id."),
]
RunOption = Annotated[str | None, typer.Option("--run", "-r", help="Only this run.")]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit JSON and nothing else.")]


def calls(
    by: GroupOption = "skill",
    run_id: RunOption = None,
    recent: Annotated[
        int, typer.Option("--recent", "-n", min=0, help="Also list this many recent calls.")
    ] = 0,
    as_json: JsonOption = False,
) -> None:
    """Summarise recorded model calls: cost, tokens and latency.

    The question this is for is "did that change help" — so the default view groups
    by stage, which is where a prompt or routing change shows up first.
    """
    records = _load(run_id)
    if not records:
        err_console.print(
            "[muted]no calls recorded yet[/muted] — run the pipeline, or check TELEMETRY_ENABLED"
        )
        raise typer.Exit(code=1)

    groups = summarise(records, by=by)

    if as_json:
        print_json(
            {
                "calls": len(records),
                "grouped_by": by,
                "groups": [group.model_dump(mode="json") for group in groups],
                "recent": [record.model_dump(mode="json") for record in records[-recent:]]
                if recent
                else [],
            }
        )
        return

    console.print(_summary_table(groups, by))
    console.print(_totals_line(records))
    if recent:
        console.print(_recent_table(records[-recent:]))


def _load(run_id: str | None) -> list[CallRecord]:
    return default_sink().read(run_id=run_id)


def _summary_table(groups: list[CallSummary], by: str) -> Table:
    table = Table(header_style="stage")
    table.add_column(by)
    table.add_column("calls", justify="right")
    table.add_column("reused", justify="right", style="muted")
    table.add_column("tokens", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("mean latency", justify="right")

    for group in groups:
        table.add_row(
            str(group.label),
            str(group.calls),
            str(group.cached) or "-",
            f"{group.total_tokens:,}",
            _cost_cell(group.cost, group.byok, group.calls),
            f"{group.mean_latency_ms:.0f}ms",
        )
    return table


def _cost_cell(cost: float, byok: int, calls: int) -> str:
    """Show BYOK plainly, because zero gateway cost is not the same as free."""
    if byok and byok == calls:
        return "[muted]byok[/muted]"
    if byok:
        return f"${cost:.4f} [muted]+{byok} byok[/muted]"
    return f"${cost:.4f}"


def _totals_line(records: list[CallRecord]) -> str:
    """Totals, with reuse called out — it is the number that shows caching paying off."""
    cost = sum(record.cost or 0.0 for record in records)
    tokens = sum(record.total_tokens or 0 for record in records)
    reused = sum(1 for record in records if record.cached)
    failed = sum(1 for record in records if record.error)

    parts = [
        f"[muted]{len(records)} calls[/muted]",
        f"[muted]{tokens:,} tokens[/muted]",
        f"[info]${cost:.4f}[/info]",
    ]
    if reused:
        parts.append(f"[success]{reused} reused[/success]")
    if failed:
        parts.append(f"[danger]{failed} failed[/danger]")
    byok = sum(1 for record in records if record.is_byok)
    if byok:
        parts.append(f"[muted]{byok} billed to your own key[/muted]")
    return "  ".join(parts)


def _recent_table(records: list[CallRecord]) -> Table:
    table = Table(header_style="stage", title="recent calls", title_style="muted")
    table.add_column("when", style="muted")
    table.add_column("stage")
    table.add_column("model")
    table.add_column("tokens", justify="right")
    table.add_column("cost", justify="right")

    for record in records:
        table.add_row(
            record.timestamp.strftime("%H:%M:%S"),
            record.skill,
            (record.model or "-").split("/")[-1],
            "reused" if record.cached else f"{record.total_tokens or 0:,}",
            "-" if record.cached else f"${record.cost or 0:.4f}",
        )
    return table


__all__ = ["calls"]
