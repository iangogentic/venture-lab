"""`op auto` — one question in, one finished result out.

The command the whole pipeline exists to make possible: seed the question, run
the nine stages, retry what fails for a transient reason, compose the report, and
hand back a decision with a path to read it at. Everything it does by hand is
still available one stage at a time — this is the same engine, driven for you.

Two front ends over one run. On a terminal it draws a live view (`app.cli.tui`)
because a run is long and silence is indistinguishable from a hang; piped,
redirected or in CI it prints a line per stage, because a TUI in a log file is
noise. Both end with the same summary on stdout, so the result survives the view
being closed.
"""

from pathlib import Path
from typing import Annotated

import typer

from app.artifacts import ArtifactKind, ArtifactRegistry, Opportunity
from app.cli.render import decision_table, print_json, spend_line, stage_line
from app.llm.telemetry import default_sink
from app.pipeline.auto import AutoResult, AutoRunner, StageAttempt
from app.utils.console import console, err_console
from app.utils.errors import OpportunityEngineError
from app.utils.logging import log_to_file


class _Printer:
    """Plain progress: one line per stage, as it finishes."""

    def stage_started(self, stage: str, attempt: int) -> None:
        """Nothing to draw — the finished line carries everything worth saying."""

    def stage_finished(self, attempt: StageAttempt) -> None:
        console.print(stage_line(attempt))

    def composing(self) -> None:
        console.print("  [muted]…[/muted] [stage]compose-report[/stage] [muted]writing[/muted]")

    def composed(self, seconds: float, error: str | None) -> None:
        mark = "[danger]✗[/danger]" if error else "[success]✓[/success]"
        note = "[danger]failed[/danger]" if error else "report written"
        console.print(
            f"  {mark} [stage]{'compose-report':<24}[/stage]{note} [muted]{seconds:.0f}s[/muted]"
        )


def auto(
    question: Annotated[
        str | None,
        typer.Argument(help="The research question. Omit to resume a run already seeded."),
    ] = None,
    run_id: Annotated[str, typer.Option("--run", "-r", help="Run identifier.")] = "default",
    retries: Annotated[
        int,
        typer.Option("--retries", min=0, max=5, help="Extra attempts per failed stage."),
    ] = 1,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Re-run stages already complete.")
    ] = False,
    no_report: Annotated[
        bool, typer.Option("--no-report", help="Stop at the artifacts; compose nothing.")
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where to write the report. Default: workspace/reports/."),
    ] = None,
    tui: Annotated[
        bool | None,
        typer.Option(
            "--tui/--no-tui",
            help="Live terminal view. Defaults to on when stdout is a terminal.",
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the run's result as JSON and nothing else.")
    ] = False,
) -> None:
    """Run the whole pipeline for one question, then compose the report.

    Resumable: interrupt it and run it again and it picks up at the first stage
    whose artifacts are not on disk — per item, so an interrupted per-opportunity
    stage asks only about the opportunities it never reached.

    It stops at the report. Harvesting leads (`op leads harvest`) and standing up
    an experiment (`op validate scaffold`) stay separate commands, because
    contacting people is a decision a human should make with the report in hand.
    """
    runner = AutoRunner(ArtifactRegistry())
    # A TUI is for a person watching. Piped, redirected or asked for JSON, the
    # caller is a program or a log file, and neither can read a repainting screen.
    wants_tui = console.is_terminal if tui is None else tui
    live = wants_tui and not as_json

    log_path = _log_path(run_id)
    try:
        with log_to_file(log_path):
            if live:
                result = _run_with_tui(
                    runner, question, run_id, retries, force, no_report, out, log_path
                )
            else:
                result = runner.run(
                    question,
                    run_id=run_id,
                    retries=retries,
                    force=force,
                    report=not no_report,
                    out=out,
                    observer=_Printer(),
                )
    except OpportunityEngineError as exc:
        err_console.print(f"[danger]{exc}[/danger]")
        err_console.print(f"[muted]log[/muted] {log_path}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        err_console.print("\n[warning]interrupted[/warning] [muted]— re-run to resume[/muted]")
        raise typer.Exit(code=130) from None

    if as_json:
        print_json(result.model_dump(mode="json"))
    else:
        _render(result, log_path)

    if not result.ok:
        raise typer.Exit(code=1)


def _log_path(run_id: str) -> Path:
    """Where this run's log goes — beside the call telemetry, named for the run.

    Resolved through the telemetry sink so the log and the cost record land in
    the same directory, wherever `TELEMETRY_PATH` points them.
    """
    return default_sink().path.parent / f"{run_id}.log"


def _run_with_tui(
    runner: AutoRunner,
    question: str | None,
    run_id: str,
    retries: int,
    force: bool,
    no_report: bool,
    out: Path | None,
    log_path: Path,
) -> AutoResult:
    """Drive the run inside the live view.

    Imported here rather than at module scope so `op --help` and every other
    command stay free of the TUI's import cost.
    """
    from app.cli.tui import AutoApp, run_with_tui

    return run_with_tui(
        AutoApp(
            runner,
            question=question,
            run_id=run_id,
            retries=retries,
            force=force,
            report=not no_report,
            out=out,
            log_path=log_path,
        )
    )


def _render(result: AutoResult, log_path: Path) -> None:
    """The summary that outlives the view: verdicts, where to read them, cost."""
    console.print()
    if result.decisions:
        console.print(decision_table(result.decisions, _titles(result.run_id)))
    elif result.ok:
        console.print(
            "[warning]no decisions[/warning] [muted]— the run finished with nothing "
            "to rule on. That is a result: the evidence did not support an "
            "opportunity worth deciding.[/muted]"
        )

    if result.error is not None:
        console.print(f"[danger]stopped[/danger] {result.error}")
        console.print("[muted]finished stages are on disk — re-run to resume[/muted]")
    if result.report_path is not None:
        console.print(f"[success]report[/success] {result.report_path}")
    if log_path.exists():
        console.print(f"[muted]log[/muted] {log_path}")
    console.print(spend_line(result))


def _titles(run_id: str) -> dict[str, str]:
    """Opportunity id to title, so the verdict table names what it ruled on."""
    registry = ArtifactRegistry()
    return {
        artifact.id: artifact.title
        for artifact in registry.find_by_type(ArtifactKind.OPPORTUNITY, run_id=run_id)
        if isinstance(artifact, Opportunity)
    }


__all__ = ["auto"]
