"""The live view: does it show the run, and does it stay inside the terminal.

Driven headlessly through Textual's own test harness, so this exercises the real
widgets rather than a mock of them. `asyncio.run` rather than a pytest plugin —
the harness is an async context manager and that is the whole requirement.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from textual.widgets import DataTable, RichLog, Static

from app.artifacts import ArtifactKind, ArtifactRegistry, Report
from app.cli.tui.auto import AutoApp, QuitConfirm
from app.pipeline import auto as auto_module
from app.pipeline.auto import AutoRunner
from app.pipeline.engine import STAGE_ORDER, PipelineEngine, StageOutcome, StageStatus
from app.pipeline.reporting import Composition
from app.utils.paths import WorkspacePaths
from tests import factories

QUESTION = "Where do platform teams lose the most time?"
SIZE = (100, 32)
"""A normal-ish terminal. Narrow enough that an unbounded column would show."""

PACE = {"seconds": 0.0}
"""How long each stubbed stage takes. A test that needs the run to still be going
when it presses a key sets this; the fixture resets it for everyone else."""


@pytest.fixture(autouse=True)
def _stubbed(monkeypatch: pytest.MonkeyPatch, workspace: WorkspacePaths) -> None:
    """Every stage completes at `PACE`, and the report is written without a model."""
    report = factories.make(ArtifactKind.REPORT, run_id="default", body="# Findings")
    PACE["seconds"] = 0.0

    def stage(self: PipelineEngine, name: str, run_id: str, *, force: bool = False) -> StageOutcome:
        time.sleep(PACE["seconds"])
        if name == "cluster-pains":
            # A long, ugly failure reason: exactly what must not break the layout.
            return StageOutcome(
                stage=name,
                status=StageStatus.COMPLETED,
                reason="collector rss returned " + "very-long-detail " * 40,
            )
        return StageOutcome(stage=name, status=StageStatus.COMPLETED)

    def compose(run_id: str, **kwargs: object) -> Composition:
        ArtifactRegistry().save(report)
        assert isinstance(report, Report)
        return Composition(report=report, composed=True)

    monkeypatch.setattr(PipelineEngine, "run_stage", stage)
    monkeypatch.setattr(auto_module, "compose_report", compose)


def drive(scenario: Callable[[AutoApp, Any], Awaitable[None]], app: AutoApp) -> None:
    """Run one headless scenario against the app."""

    async def main() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await scenario(app, pilot)

    asyncio.run(main())


async def settle(app: AutoApp, pilot: Any, *, ticks: int = 400) -> None:
    """Wait for the worker thread to finish the run."""
    for _ in range(ticks):
        # Reaching into the app is the point: the test is what waits on it.
        if app._done:
            await pilot.pause()
            return
        await pilot.pause()
        await asyncio.sleep(0.01)
    raise AssertionError("the run never finished")


def make_app(question: str | None = QUESTION) -> AutoApp:
    return AutoApp(AutoRunner(), question=question, run_id="default")


def test_every_stage_is_listed_before_anything_runs() -> None:
    """The nine stages are visible from the first frame, so the shape of the work
    is clear before any of it has happened."""

    async def scenario(app: AutoApp, pilot: Any) -> None:
        await pilot.pause()
        table = app.query_one("#stages", DataTable)
        assert table.row_count == len(STAGE_ORDER)
        await settle(app, pilot)

    drive(scenario, make_app())


def test_the_run_completes_and_the_table_shows_it() -> None:
    async def scenario(app: AutoApp, pilot: Any) -> None:
        await settle(app, pilot)
        assert app.result is not None and app.result.ok
        table = app.query_one("#stages", DataTable)
        states = [str(table.get_cell(stage, "state")) for stage in STAGE_ORDER]
        assert set(states) == {"completed"}

    drive(scenario, make_app())


def test_the_log_narrates_the_run() -> None:
    async def scenario(app: AutoApp, pilot: Any) -> None:
        await settle(app, pilot)
        lines = [str(line) for line in app.query_one("#log", RichLog).lines]
        text = "\n".join(lines)
        assert "collect-evidence" in text
        assert "run complete" in text

    drive(scenario, make_app())


def test_the_status_line_reports_spend_when_done() -> None:
    async def scenario(app: AutoApp, pilot: Any) -> None:
        await settle(app, pilot)
        status = str(app.query_one("#status", Static).content)
        assert "calls" in status and "press q to close" in status

    drive(scenario, make_app())


def test_nothing_overflows_the_terminal() -> None:
    """A stage detail can be a model error a paragraph long. It must be truncated
    or wrapped, never pushed off the side where the rest of the row goes with it."""

    async def scenario(app: AutoApp, pilot: Any) -> None:
        await settle(app, pilot)
        width = SIZE[0]
        for line in app.screen._compositor.render_strips():
            assert sum(segment.cell_length for segment in line) <= width

    drive(scenario, make_app())


def test_quitting_mid_run_asks_first() -> None:
    """`q` is one keystroke away from the log-follow toggle, and the stage in
    flight is minutes of model time. So it asks — and cancelling means stay."""
    PACE["seconds"] = 0.05

    async def scenario(app: AutoApp, pilot: Any) -> None:
        await pilot.press("q")
        await pilot.pause()
        assert isinstance(app.screen, QuitConfirm), "no confirmation appeared"

        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, QuitConfirm), "escape did not dismiss it"
        assert app.is_running, "the view closed while the run was still going"
        await settle(app, pilot)

    drive(scenario, make_app())


def test_confirming_the_dialog_leaves_mid_run() -> None:
    """Choosing Quit really quits — and reports no result, which is what tells
    the caller the run was abandoned rather than finished."""
    PACE["seconds"] = 0.2

    async def scenario(app: AutoApp, pilot: Any) -> None:
        await pilot.press("q")
        await pilot.pause()
        await pilot.click("#quit")
        await pilot.pause()

        assert not app.is_running
        assert app.result is None

    drive(scenario, make_app())


def test_cancelling_the_dialog_with_the_button_keeps_running() -> None:
    PACE["seconds"] = 0.05

    async def scenario(app: AutoApp, pilot: Any) -> None:
        await pilot.press("q")
        await pilot.pause()
        await pilot.click("#cancel")
        await pilot.pause()

        assert app.is_running
        await settle(app, pilot)

    drive(scenario, make_app())


def test_the_dialog_fits_the_terminal() -> None:
    """A confirmation you cannot read is not a confirmation."""
    PACE["seconds"] = 0.05

    async def scenario(app: AutoApp, pilot: Any) -> None:
        await pilot.press("q")
        await pilot.pause()
        for line in app.screen._compositor.render_strips():
            assert sum(segment.cell_length for segment in line) <= SIZE[0]
        await pilot.press("escape")
        await settle(app, pilot)

    drive(scenario, make_app())


def test_quitting_once_done_closes_the_view() -> None:
    async def scenario(app: AutoApp, pilot: Any) -> None:
        await settle(app, pilot)
        await pilot.press("q")
        await pilot.pause()
        assert not app.is_running

    drive(scenario, make_app())


def test_a_run_that_cannot_start_surfaces_in_the_view() -> None:
    """No question and no seeded run: the failure is drawn, not swallowed."""

    async def scenario(app: AutoApp, pilot: Any) -> None:
        await settle(app, pilot)
        assert app.failure is not None
        text = "\n".join(str(line) for line in app.query_one("#log", RichLog).lines)
        assert "no question yet" in text

    drive(scenario, AutoApp(AutoRunner(), question=None, run_id="empty"))
