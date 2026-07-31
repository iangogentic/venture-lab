"""The live view of an `op auto` run.

Three panes, top to bottom: the question, the nine stages with their state, and a
log of everything that happened. The stage table and the log both scroll with the
mouse, and every long value is truncated or wrapped rather than pushed off the
side — a run's detail is often a model error a paragraph long, and losing the end
of it is losing the reason.

The run itself happens on a worker thread. Textual owns the event loop, so the
observer marshals every update back with `call_from_thread` rather than touching
a widget from the thread doing the research.
"""

import logging
import os
import sys
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Final, NoReturn

from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Label, RichLog, Static

from app.artifacts import ArtifactKind, Opportunity
from app.pipeline.auto import AutoResult, AutoRunner, StageAttempt
from app.pipeline.engine import STAGE_ORDER, StageStatus
from app.utils.console import err_console
from app.utils.logging import get_logger
from app.utils.time import utcnow

logger = get_logger(__name__)

_PENDING: Final[str] = "pending"
_RUNNING: Final[str] = "running"

_STATE_STYLE: Final[dict[str, str]] = {
    _PENDING: "dim",
    _RUNNING: "bold yellow",
    StageStatus.COMPLETED.value: "bold green",
    StageStatus.SKIPPED.value: "dim",
    StageStatus.EMPTY.value: "bold yellow",
    StageStatus.BLOCKED.value: "yellow",
    StageStatus.FAILED.value: "bold red",
}

_STATE_MARK: Final[dict[str, str]] = {
    _PENDING: "·",
    _RUNNING: "▶",
    StageStatus.COMPLETED.value: "✓",
    StageStatus.SKIPPED.value: "·",
    StageStatus.EMPTY.value: "∅",
    StageStatus.BLOCKED.value: "!",
    StageStatus.FAILED.value: "✗",
}

_VERDICT_STYLE: Final[dict[str, str]] = {
    "build": "bold green",
    "wait": "yellow",
    "reject": "dim",
}


class _Collector(logging.Handler):
    """Holds log records until the app can draw them.

    The root logger writes to stderr, which is the same screen the TUI has taken
    over: a single `logger.warning` mid-run scribbles across the layout. Rather
    than silence the log — "retrying analyze-competition" is exactly what someone
    watching wants to see — records are buffered here and drained onto the log
    pane on the next tick. `deque.append` is atomic, so the research thread can
    log without reaching into a widget.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: deque[str] = deque(maxlen=500)

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


class QuitConfirm(ModalScreen[bool]):
    """Asks before abandoning a run that is still going.

    Two buttons rather than a second keystroke: a confirmation you dismiss by
    reflex is not a confirmation, and the answer here decides whether minutes of
    model time survive.
    """

    BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="quit-dialog"):
            yield Label("The run is still going.", id="quit-title")
            yield Label(
                "Every finished stage is already on disk, so re-running resumes "
                "from there. Only the stage in flight is lost.",
                id="quit-body",
            )
            with Horizontal(id="quit-buttons"):
                yield Button("Cancel", variant="primary", id="cancel")
                yield Button("Quit", variant="error", id="quit")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "quit")

    def action_cancel(self) -> None:
        """Escape means stay — the safe reading of an ambiguous keypress."""
        self.dismiss(False)


class AutoApp(App[AutoResult | None]):
    """Runs the pipeline end to end and draws it while it happens."""

    TITLE = "Opportunity Engine"

    CSS_PATH = "auto.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "request_quit", "Quit"),
        ("s", "toggle_scroll", "Follow log"),
    ]

    def __init__(
        self,
        runner: AutoRunner,
        *,
        question: str | None,
        run_id: str,
        retries: int = 1,
        force: bool = False,
        report: bool = True,
        out: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._question = question
        self._run_id = run_id
        self._retries = retries
        self._force = force
        self._report = report
        self._out = out
        self._log_path = log_path

        self.result: AutoResult | None = None
        self.failure: BaseException | None = None
        self._started = utcnow()
        self._current: str | None = None
        self._current_started: datetime | None = None
        self._done = False
        self._follow = True
        self._collector = _Collector()
        self._displaced: list[logging.Handler] = []

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Static(self._headline(), id="question", markup=False)
            yield DataTable(id="stages", cursor_type="row", zebra_stripes=True)
            yield RichLog(id="log", markup=True, wrap=True, auto_scroll=True)
            yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#stages", DataTable)
        table.add_column("", key="mark", width=2)
        table.add_column("stage", key="stage", width=24)
        table.add_column("state", key="state", width=10)
        table.add_column("detail", key="detail", width=40)
        table.add_column("time", key="time", width=7)
        for stage in STAGE_ORDER:
            table.add_row(_mark(_PENDING), stage, _state(_PENDING), "", "", key=stage)

        self._capture_logging()
        self._write(f"[dim]run[/dim] [magenta]{escape(self._run_id)}[/magenta]")
        if self._log_path is not None:
            # Said up front, because this pane cannot be scrolled back to once
            # the view closes, and the file can.
            self._write(f"[dim]log[/dim] {escape(str(self._log_path))}")
        self.set_interval(1.0, self._tick)
        self._drive()

    def on_unmount(self) -> None:
        """Give the terminal's logging back before the screen goes."""
        root = logging.getLogger()
        root.removeHandler(self._collector)
        for handler in self._displaced:
            root.addHandler(handler)
        self._displaced.clear()

    def _capture_logging(self) -> None:
        """Take the console handlers off the root logger for as long as we own the screen.

        File handlers are left alone deliberately. They write somewhere this app
        is not painting, and they are the only durable copy of a warning that
        scrolls past — a full-screen view cannot be scrolled back to after it
        closes, so silencing the file too would make the log unrecoverable.
        """
        root = logging.getLogger()
        self._displaced = [h for h in root.handlers if not isinstance(h, logging.FileHandler)]
        for handler in self._displaced:
            root.removeHandler(handler)
        self._collector.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(self._collector)

    def _headline(self) -> str:
        """The question, or a placeholder until the run says what it is."""
        return self._question or f"(resuming run {self._run_id})"

    # ------------------------------------------------------------------ worker

    @work(thread=True, exclusive=True)
    def _drive(self) -> None:
        """Run the pipeline off the event loop, reporting back as it goes."""
        try:
            result = self._runner.run(
                self._question,
                run_id=self._run_id,
                retries=self._retries,
                force=self._force,
                report=self._report,
                out=self._out,
                observer=_Observer(self),
            )
        except BaseException as exc:  # surfaced in the UI, re-raised by the caller
            self.call_from_thread(self._crashed, exc)
            return
        self.call_from_thread(self._finished, result)

    # ------------------------------------------------------------------ events

    def stage_started(self, stage: str, attempt: int) -> None:
        """Mark a stage as in flight."""
        self._current = stage
        self._current_started = utcnow()
        self._set(stage, mark=_RUNNING, state=_RUNNING, detail="", time="0s")
        retry = f" [yellow](attempt {attempt})[/yellow]" if attempt > 1 else ""
        self._write(f"[bold magenta]{escape(stage)}[/bold magenta] started{retry}")

    def stage_finished(self, attempt: StageAttempt) -> None:
        """Record how a stage ended."""
        outcome = attempt.outcome
        self._current = None
        self._current_started = None
        self._set(
            attempt.stage,
            mark=outcome.status.value,
            state=outcome.status.value,
            detail=_detail(attempt),
            time=f"{attempt.seconds:.0f}s",
        )
        style = _STATE_STYLE[outcome.status.value]
        self._write(
            f"[{style}]{outcome.status.value}[/{style}] "
            f"[bold magenta]{escape(attempt.stage)}[/bold magenta] — {escape(_detail(attempt))}"
        )
        if outcome.reason and outcome.status is not StageStatus.COMPLETED:
            self._write(f"  [dim]{escape(outcome.reason)}[/dim]")

    def composing(self) -> None:
        """Every stage is done; the report is being written."""
        self._current = None
        self._write("[bold magenta]compose-report[/bold magenta] started")

    def composed(self, seconds: float, error: str | None) -> None:
        """The report finished, or could not be written."""
        if error:
            self._write(f"[bold red]failed[/bold red] compose-report — {escape(error)}")
        else:
            self._write(
                f"[bold green]completed[/bold green] "
                f"[bold magenta]compose-report[/bold magenta] [dim]{seconds:.0f}s[/dim]"
            )

    def _finished(self, result: AutoResult) -> None:
        """Draw the outcome and hand control back to whoever is watching."""
        self.result = result
        self._done = True
        self._drain()  # so the summary is the last thing in the log, not a stale warning
        self._write("")
        if result.ok:
            self._write("[bold green]run complete[/bold green]")
        else:
            self._write(f"[bold red]run stopped[/bold red] — {escape(result.error or '')}")
        self._log_verdicts(result)
        if result.report_path is not None:
            self._write(f"[dim]report[/dim] {escape(str(result.report_path))}")
        self._write("[dim]press q to close[/dim]")
        self._tick()

    def _crashed(self, exc: BaseException) -> None:
        """The run could not even start, or died in a way the runner does not own."""
        self.failure = exc
        self._done = True
        self._drain()
        self._write(f"[bold red]failed[/bold red] {escape(str(exc))}")
        self._write("[dim]press q to close[/dim]")
        self._tick()

    def _log_verdicts(self, result: AutoResult) -> None:
        """The verdicts, with the opportunity each one is about."""
        if not result.decisions:
            return
        titles = self._titles(result.run_id)
        self._write("")
        for decision in result.decisions:
            verdict = decision.verdict.value
            style = _VERDICT_STYLE.get(verdict, "white")
            title = titles.get(decision.opportunity.id, decision.opportunity.id)
            self._write(
                f"  [{style}]{verdict:<7}[/{style}] {escape(title)} "
                f"[dim]({decision.decision_confidence:.2f})[/dim]"
            )
            self._write(f"          [dim]next: {escape(decision.next_validation_step)}[/dim]")

    def _titles(self, run_id: str) -> dict[str, str]:
        """Opportunity id to title, so a verdict reads as a sentence."""
        found: dict[str, str] = {}
        for artifact in self._runner.registry.find_by_type(ArtifactKind.OPPORTUNITY, run_id=run_id):
            if isinstance(artifact, Opportunity):
                found[artifact.id] = artifact.title
        return found

    # ----------------------------------------------------------------- drawing

    def _set(self, stage: str, *, mark: str, state: str, detail: str, time: str) -> None:
        """Update one row of the stage table."""
        table = self.query_one("#stages", DataTable)
        table.update_cell(stage, "mark", _mark(mark))
        table.update_cell(stage, "state", _state(state))
        table.update_cell(stage, "detail", Text(detail, overflow="ellipsis", no_wrap=True))
        table.update_cell(stage, "time", time)

    def _write(self, markup: str) -> None:
        """Append a line to the log pane.

        Not `_log`: `App` already has one, and shadowing Textual's own logger
        with a different signature breaks the framework in ways that surface
        far from here.
        """
        self.query_one("#log", RichLog).write(markup)

    def _tick(self) -> None:
        """Once a second: drain the log buffer, age the running stage, refresh the footer."""
        self._drain()
        if self._current is not None and self._current_started is not None:
            elapsed = (utcnow() - self._current_started).total_seconds()
            self.query_one("#stages", DataTable).update_cell(
                self._current, "time", f"{elapsed:.0f}s"
            )
        self.query_one("#status", Static).update(self._status())

    def _drain(self) -> None:
        """Move whatever the loggers buffered onto the log pane."""
        while self._collector.records:
            self._write(f"[dim]{escape(self._collector.records.popleft())}[/dim]")

    def _status(self) -> str:
        """The footer line: elapsed, and spend once the run has totalled it."""
        elapsed = (utcnow() - self._started).total_seconds()
        parts = [f"elapsed {_clock(elapsed)}"]
        if self.result is not None:
            spend = self.result.spend
            parts += [
                f"{spend.calls} calls",
                f"{spend.total_tokens:,} tokens",
                f"${spend.cost:.2f}",
            ]
            if spend.byok:
                parts.append(f"{spend.byok} BYOK")
        parts.append("done — press q to close" if self._done else "running")
        return "  ·  ".join(parts)

    # ---------------------------------------------------------------- bindings

    def action_request_quit(self) -> None:
        """Close the view, asking first if that would abandon work.

        Once the run is done there is nothing to lose, so `q` just closes. While
        it is still going, `q` asks — because the stage in flight is minutes of
        model time that quitting throws away, and a keystroke away from the log
        pane is too easy to hit by accident.
        """
        if self._done:
            self.exit(self.result)
            return
        self.push_screen(QuitConfirm(), self._quit_answered)

    def _quit_answered(self, confirmed: bool | None) -> None:
        """Leave on a yes; on anything else, carry on as though nothing happened."""
        if confirmed:
            self.exit(None)

    def action_toggle_scroll(self) -> None:
        """Stop the log auto-scrolling, so a line can be read while the run goes on."""
        self._follow = not self._follow
        self.query_one("#log", RichLog).auto_scroll = self._follow
        self._write(f"[dim]log follow {'on' if self._follow else 'off'}[/dim]")


class _Observer:
    """Marshals runner events from the worker thread onto the event loop."""

    def __init__(self, app: AutoApp) -> None:
        self._app = app

    def stage_started(self, stage: str, attempt: int) -> None:
        self._post(self._app.stage_started, stage, attempt)

    def stage_finished(self, attempt: StageAttempt) -> None:
        self._post(self._app.stage_finished, attempt)

    def composing(self) -> None:
        self._post(self._app.composing)

    def composed(self, seconds: float, error: str | None) -> None:
        self._post(self._app.composed, seconds, error)

    def _post(self, method: Callable[..., None], *args: object) -> None:
        """Hand one event to the event loop, or drop it if there is no longer one.

        After a confirmed quit the app is gone but the worker thread is still
        mid-stage for a moment. Its next event has nowhere to go, and that is the
        expected end of an abandoned run — not something to raise about.
        """
        try:
            self._app.call_from_thread(method, *args)
        except Exception:
            logger.debug("dropped a %s event: the view has closed", method.__name__)


def _mark(state: str) -> Text:
    """The glyph for a state, styled."""
    return Text(_STATE_MARK[state], style=_STATE_STYLE[state])


def _state(state: str) -> Text:
    """The state word, styled."""
    return Text(state, style=_STATE_STYLE[state])


def _detail(attempt: StageAttempt) -> str:
    """What an attempt achieved, short enough for a table cell."""
    outcome = attempt.outcome
    if outcome.status is StageStatus.COMPLETED:
        produced = f"{len(outcome.produced)} produced"
        return f"{produced}, {outcome.reused} reused" if outcome.reused else produced
    if outcome.status is StageStatus.SKIPPED:
        return "already done"
    if outcome.status is StageStatus.EMPTY:
        # The reason is a sentence; this cell is forty columns. The word that
        # matters is "nothing", and the log pane below prints the rest.
        return "nothing produced"
    return outcome.reason or ""


def _clock(seconds: float) -> str:
    """Elapsed time, at the precision a person cares about."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m {rest:02d}s" if minutes < 60 else f"{minutes // 60}h {minutes % 60:02d}m"


def run_with_tui(app: AutoApp) -> AutoResult:
    """Run the app to completion and return the result.

    Raises:
        BaseException: Whatever the run raised, re-raised here so the CLI reports
            it the same way it would without a TUI.
    """
    app.run()
    if app.failure is not None:
        raise app.failure
    if app.result is None:
        _abandon()
    return app.result


def _abandon() -> NoReturn:
    """Leave immediately, because the user asked to and waiting would not be that.

    The research runs on one of asyncio's default executor threads, and those are
    *not* daemons: a normal exit would block until the worker had finished the
    entire remaining pipeline — quarter-hour silences included — long after the
    screen said goodbye. So the process is ended outright.

    Safe to do here and nowhere earlier: Textual has already torn down and given
    the terminal back, every finished stage is durably on disk, and the run
    resumes from exactly there. The only thing lost is the stage in flight, which
    is precisely what was asked for.
    """
    err_console.print("\n[warning]stopped[/warning] [muted]— re-run to resume[/muted]")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(130)


__all__ = ["AutoApp", "QuitConfirm", "run_with_tui"]
