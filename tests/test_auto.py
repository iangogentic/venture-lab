"""`op auto`: one question in, one finished result out.

Every stage is stubbed. What is under test is the orchestration — seeding,
retrying, stopping, composing, and what lands in the ledger — not the nine
skills, which have their own tests and would need a model to run.
"""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.artifacts import ArtifactKind, ArtifactRef, ArtifactRegistry, Report
from app.cli.main import app
from app.models import RunStatus, StageState
from app.pipeline import auto as auto_module
from app.pipeline.auto import AutoRunner, StageAttempt
from app.pipeline.engine import STAGE_ORDER, PipelineEngine, StageOutcome, StageStatus
from app.pipeline.reporting import Composition, ReportUnavailableError
from app.storage.ledger import ledger_scope
from app.storage.schema import create_all
from app.utils.errors import PipelineError
from app.utils.paths import WorkspacePaths
from tests import factories

runner = CliRunner()

QUESTION = "Where do platform teams lose the most time?"


class Recorder:
    """An observer that just remembers what it was told."""

    def __init__(self) -> None:
        self.started: list[tuple[str, int]] = []
        self.finished: list[StageAttempt] = []
        self.composing_calls = 0
        self.compositions: list[tuple[float, str | None]] = []

    def stage_started(self, stage: str, attempt: int) -> None:
        self.started.append((stage, attempt))

    def stage_finished(self, attempt: StageAttempt) -> None:
        self.finished.append(attempt)

    def composing(self) -> None:
        self.composing_calls += 1

    def composed(self, seconds: float, error: str | None) -> None:
        self.compositions.append((seconds, error))


@pytest.fixture
def stub_stages(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, list[StageOutcome]]]:
    """Make every stage return a scripted outcome instead of calling a model.

    Keyed by stage; each stage pops its next outcome, and any stage without a
    script simply completes.
    """
    script: dict[str, list[StageOutcome]] = {}

    def fake(self: PipelineEngine, stage: str, run_id: str, *, force: bool = False) -> StageOutcome:
        queued = script.get(stage)
        if queued:
            return queued.pop(0)
        return StageOutcome(stage=stage, status=StageStatus.COMPLETED)

    monkeypatch.setattr(PipelineEngine, "run_stage", fake)
    yield script


@pytest.fixture
def stub_report(monkeypatch: pytest.MonkeyPatch, workspace: WorkspacePaths) -> Report:
    """Compose a real Report artifact without calling a model."""
    report = factories.make(ArtifactKind.REPORT, run_id="default", body="# Findings\n\nBody.")
    assert isinstance(report, Report)

    def fake(run_id: str, **kwargs: object) -> Composition:
        ArtifactRegistry().save(report)
        return Composition(report=report, composed=True)

    monkeypatch.setattr(auto_module, "compose_report", fake)
    return report


# ------------------------------------------------------------------- seeding


def test_a_run_needs_a_question(workspace: WorkspacePaths) -> None:
    with pytest.raises(PipelineError, match="has no question yet"):
        AutoRunner().run(None, run_id="fresh")


def test_a_second_question_on_the_same_run_is_refused(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report
) -> None:
    """A run id means one question. Silently seeding a second would leave every
    stage reading two and answering neither."""
    AutoRunner().run(QUESTION)

    with pytest.raises(PipelineError, match="already asking"):
        AutoRunner().run("A completely different question")


def test_rerunning_with_the_same_question_resumes(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report
) -> None:
    first = AutoRunner().run(QUESTION)
    second = AutoRunner().run(QUESTION)

    assert second.question_id == first.question_id
    assert len(ArtifactRegistry().find_by_type(ArtifactKind.QUESTION)) == 1


def test_resuming_needs_no_question_repeated(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report
) -> None:
    AutoRunner().run(QUESTION)
    resumed = AutoRunner().run(None)
    assert resumed.question == QUESTION


# -------------------------------------------------------------------- stages


def test_a_full_run_walks_every_stage_then_reports(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report
) -> None:
    watcher = Recorder()
    result = AutoRunner().run(QUESTION, observer=watcher)

    assert [a.stage for a in result.attempts] == list(STAGE_ORDER)
    assert [stage for stage, _ in watcher.started] == list(STAGE_ORDER)
    assert watcher.composing_calls == 1
    assert result.ok
    assert result.report is not None and result.report.id == stub_report.id


def test_the_report_is_written_where_it_can_be_read(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report
) -> None:
    result = AutoRunner().run(QUESTION)

    assert result.report_path == workspace.reports / "default.md"
    assert result.report_path.read_text(encoding="utf-8") == "# Findings\n\nBody."


def test_out_overrides_where_the_report_lands(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report, tmp_path: Path
) -> None:
    destination = tmp_path / "nested" / "findings.md"
    result = AutoRunner().run(QUESTION, out=destination)

    assert destination.read_text(encoding="utf-8").startswith("# Findings")
    assert result.report_path == destination


def test_a_failed_stage_is_retried(
    workspace: WorkspacePaths, stub_stages: dict[str, list[StageOutcome]], stub_report: Report
) -> None:
    """Retrying is what makes an unattended run survive a transient failure."""
    stub_stages["research-brief"] = [
        StageOutcome(stage="research-brief", status=StageStatus.FAILED, reason="429"),
        StageOutcome(stage="research-brief", status=StageStatus.COMPLETED),
    ]
    watcher = Recorder()

    result = AutoRunner().run(QUESTION, retries=1, observer=watcher)

    assert result.ok
    assert ("research-brief", 2) in watcher.started
    assert [a.attempt for a in result.attempts if a.stage == "research-brief"] == [2]


def test_a_stage_out_of_retries_stops_the_run(
    workspace: WorkspacePaths, stub_stages: dict[str, list[StageOutcome]], stub_report: Report
) -> None:
    stub_stages["cluster-pains"] = [
        StageOutcome(stage="cluster-pains", status=StageStatus.FAILED, reason="boom")
        for _ in range(3)
    ]

    result = AutoRunner().run(QUESTION, retries=1)

    assert not result.ok
    assert result.error is not None and "cluster-pains" in result.error
    assert [a.stage for a in result.attempts][-1] == "cluster-pains"
    assert "discover-opportunities" not in [a.stage for a in result.attempts]
    assert result.report is None


def test_an_empty_stage_stops_the_run_and_is_named_as_the_cause(
    workspace: WorkspacePaths, stub_stages: dict[str, list[StageOutcome]], stub_report: Report
) -> None:
    """The error must name the stage that came back empty, not the next one along.

    Before, `collect-evidence` reported completed with nothing to show for it and
    the run died reporting `research-brief blocked: no evidence artifacts` — the
    stage that noticed, not the stage that caused it.
    """
    stub_stages["collect-evidence"] = [
        StageOutcome(
            stage="collect-evidence",
            status=StageStatus.EMPTY,
            reason="ran without error but produced no evidence artifacts",
        )
    ]

    result = AutoRunner().run(QUESTION)

    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("collect-evidence empty:")
    assert [a.stage for a in result.attempts] == ["collect-evidence"]
    assert result.report is None


def test_an_empty_stage_is_not_retried(
    workspace: WorkspacePaths, stub_stages: dict[str, list[StageOutcome]], stub_report: Report
) -> None:
    """Nothing went wrong, so a second attempt buys a second helping of nothing."""
    stub_stages["collect-evidence"] = [
        StageOutcome(stage="collect-evidence", status=StageStatus.EMPTY, reason="nothing found")
        for _ in range(3)
    ]
    watcher = Recorder()

    AutoRunner().run(QUESTION, retries=2, observer=watcher)

    assert [a for a in watcher.started if a[0] == "collect-evidence"] == [("collect-evidence", 1)]


def test_an_empty_stage_is_recorded_in_the_ledger_as_empty(
    workspace: WorkspacePaths, stub_stages: dict[str, list[StageOutcome]], stub_report: Report
) -> None:
    """`op runs` has to be able to show it later; "completed, 0 produced" cannot."""
    stub_stages["collect-evidence"] = [
        StageOutcome(stage="collect-evidence", status=StageStatus.EMPTY, reason="nothing found")
    ]

    AutoRunner().run(QUESTION)

    with ledger_scope() as ledger:
        attempts = ledger.stages.for_stage("default", "collect-evidence")
        assert [a.state for a in attempts] == [StageState.EMPTY]
        assert attempts[0].detail == "nothing found"


def test_a_blocked_stage_is_not_retried(
    workspace: WorkspacePaths, stub_stages: dict[str, list[StageOutcome]], stub_report: Report
) -> None:
    """Blocked means nothing upstream produced input. Asking again cannot conjure one."""
    stub_stages["decision"] = [
        StageOutcome(stage="decision", status=StageStatus.BLOCKED, reason="no opportunities")
        for _ in range(3)
    ]
    watcher = Recorder()

    result = AutoRunner().run(QUESTION, retries=2, observer=watcher)

    assert not result.ok
    assert [a for a in watcher.started if a[0] == "decision"] == [("decision", 1)]


def test_a_report_that_cannot_be_written_is_reported_not_raised(
    workspace: WorkspacePaths,
    stub_stages: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The artifacts are on disk either way; a missing report is a stated outcome."""

    def unavailable(run_id: str, **kwargs: object) -> Composition:
        raise ReportUnavailableError(run_id, ["decision"])

    monkeypatch.setattr(auto_module, "compose_report", unavailable)

    result = AutoRunner().run(QUESTION)

    assert not result.ok
    assert result.error is not None and "decision" in result.error


def test_no_report_stops_at_the_artifacts(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report
) -> None:
    result = AutoRunner().run(QUESTION, report=False)

    assert result.ok, "a run asked not to report is a finished run"
    assert result.report is None and result.report_path is None


# -------------------------------------------------------------------- ledger


def test_the_run_lands_in_the_ledger(
    workspace: WorkspacePaths, stub_stages: dict[str, list[StageOutcome]], stub_report: Report
) -> None:
    stub_stages["research-brief"] = [
        StageOutcome(stage="research-brief", status=StageStatus.FAILED, reason="429"),
        StageOutcome(stage="research-brief", status=StageStatus.COMPLETED),
    ]
    AutoRunner().run(QUESTION, retries=1)

    with ledger_scope() as ledger:
        run = ledger.runs.require("default")
        assert run.status is RunStatus.COMPLETED
        assert run.auto is True
        assert run.question == QUESTION
        assert run.report_id == stub_report.id

        attempts = ledger.stages.for_stage("default", "research-brief")
        assert [a.state for a in attempts] == [StageState.FAILED, StageState.COMPLETED]
        assert attempts[0].detail == "429"


def test_a_stopped_run_is_recorded_as_failed(
    workspace: WorkspacePaths, stub_stages: dict[str, list[StageOutcome]], stub_report: Report
) -> None:
    stub_stages["cluster-pains"] = [
        StageOutcome(stage="cluster-pains", status=StageStatus.FAILED, reason="boom")
    ]

    AutoRunner().run(QUESTION, retries=0)

    with ledger_scope() as ledger:
        run = ledger.runs.require("default")
        assert run.status is RunStatus.FAILED
        assert run.error is not None and "boom" in run.error


def test_the_run_indexes_what_it_produced(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report
) -> None:
    """Closing a run projects its artifacts, so `op runs` can count them."""
    registry = ArtifactRegistry()
    opportunity = factories.make(ArtifactKind.OPPORTUNITY, run_id="default")
    registry.save(opportunity)
    registry.save(
        factories.make(
            ArtifactKind.DECISION,
            run_id="default",
            opportunity=ArtifactRef(kind=ArtifactKind.OPPORTUNITY, id=opportunity.id),
        )
    )
    registry.save(factories.make(ArtifactKind.EVIDENCE, run_id="default", collector="rss"))

    result = AutoRunner().run(QUESTION)

    assert result.verdicts == {"build": 1}
    with ledger_scope() as ledger:
        assert ledger.evidence.count_for_run("default") == 1
        assert [row.verdict for row in ledger.opportunities.by_run("default")] == ["build"]


# ----------------------------------------------------------------------- CLI


def test_cli_auto_runs_and_summarises(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report
) -> None:
    result = runner.invoke(app, ["auto", QUESTION, "--no-tui"])

    assert result.exit_code == 0, result.output
    assert "collect-evidence" in result.output
    assert "report" in result.output


def test_cli_auto_json_emits_the_result(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report
) -> None:
    result = runner.invoke(app, ["auto", QUESTION, "--json"])

    assert result.exit_code == 0, result.output
    assert '"run_id"' in result.output
    assert '"attempts"' in result.output


def test_cli_auto_exits_non_zero_when_a_stage_fails(
    workspace: WorkspacePaths, stub_stages: dict[str, list[StageOutcome]], stub_report: Report
) -> None:
    stub_stages["collect-evidence"] = [
        StageOutcome(stage="collect-evidence", status=StageStatus.FAILED, reason="no collectors")
    ]

    result = runner.invoke(app, ["auto", QUESTION, "--no-tui", "--retries", "0"])

    assert result.exit_code == 1
    assert "no collectors" in result.output


def test_cli_auto_reports_a_missing_question_without_a_traceback(
    workspace: WorkspacePaths,
) -> None:
    result = runner.invoke(app, ["auto", "--no-tui", "--run", "empty"])

    assert result.exit_code == 1
    assert "no question yet" in result.output
    assert "Traceback" not in result.output


# ------------------------------------------------------------------ op runs


def test_cli_runs_lists_the_run(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report
) -> None:
    AutoRunner().run(QUESTION)

    result = runner.invoke(app, ["runs", "--json"])

    assert result.exit_code == 0, result.output
    assert "default" in result.output
    assert QUESTION in result.output


def test_cli_runs_show_lists_every_attempt(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report
) -> None:
    AutoRunner().run(QUESTION)

    result = runner.invoke(app, ["runs", "show", "default", "--json"])

    assert result.exit_code == 0, result.output
    assert '"attempts"' in result.output
    assert "collect-evidence" in result.output


def test_cli_runs_show_unknown_run_exits_non_zero(workspace: WorkspacePaths) -> None:
    create_all()
    result = runner.invoke(app, ["runs", "show", "nope"])
    assert result.exit_code == 1


def test_cli_runs_sync_indexes_a_workspace_with_no_ledger(
    workspace: WorkspacePaths,
) -> None:
    """The projection rule, exercised: artifacts on disk, nothing recorded, one command."""
    registry = ArtifactRegistry()
    registry.save(factories.make(ArtifactKind.QUESTION, run_id="imported"))
    registry.save(factories.make(ArtifactKind.EVIDENCE, run_id="imported", collector="rss"))

    result = runner.invoke(app, ["runs", "sync", "--json"])

    assert result.exit_code == 0, result.output
    with ledger_scope() as ledger:
        assert ledger.evidence.count_for_run("imported") == 1
        assert ledger.runs.require("imported").question is not None


# ------------------------------------------------------- rendering integrity


def test_a_failure_reason_containing_brackets_survives_rendering(
    workspace: WorkspacePaths, stub_stages: dict[str, list[StageOutcome]], stub_report: Report
) -> None:
    """Regression: Rich reads `[rate_limit]` in a cell as styling and drops it.

    Model and provider text is quoted all over this CLI, and silently losing part
    of it is the one failure the project cannot tolerate.
    """
    reason = "item 2 of 3: 429 [rate_limit] on [claude-sonnet-5]"
    stub_stages["analyze-market"] = [
        StageOutcome(stage="analyze-market", status=StageStatus.FAILED, reason=reason)
    ]

    result = runner.invoke(app, ["auto", QUESTION, "--no-tui", "--retries", "0"])
    assert "[rate_limit]" in result.output

    shown = runner.invoke(app, ["runs", "show", "default"])
    assert "[rate_limit]" in shown.output


# ------------------------------------------------------------- the run log


def test_the_run_writes_a_log_file(
    workspace: WorkspacePaths, stub_stages: dict[str, list[StageOutcome]], stub_report: Report
) -> None:
    """A long run's warnings are its most perishable output: on screen they
    scroll away, and under the live view they cannot even be selected. The file
    is what makes them readable, greppable and pasteable afterwards."""
    stub_stages["research-brief"] = [
        StageOutcome(stage="research-brief", status=StageStatus.FAILED, reason="429 slow down"),
        StageOutcome(stage="research-brief", status=StageStatus.COMPLETED),
    ]

    result = runner.invoke(app, ["auto", QUESTION, "--no-tui", "--retries", "1"])
    assert result.exit_code == 0, result.output

    log = workspace.root / ".telemetry" / "default.log"
    assert log.is_file(), "no log file was written"
    assert "429 slow down" in log.read_text(encoding="utf-8")
    # The name, not the path: a temp path is long enough that Rich wraps it, so
    # asserting the whole string would be testing the console width.
    assert "default.log" in result.output, "the summary does not say where the log is"


def test_the_live_view_does_not_silence_the_log_file() -> None:
    """The TUI takes the console handlers off the root logger so they cannot
    scribble over it. Regression: it took the file handler too, which left a
    TUI run with no recoverable log at all."""
    from app.cli.tui.auto import AutoApp

    root = logging.getLogger()
    file_handler = logging.FileHandler(os.devnull)
    root.addHandler(file_handler)
    try:
        view = AutoApp(AutoRunner(), question=QUESTION, run_id="default")
        view._capture_logging()
        try:
            assert file_handler in root.handlers, "the file handler was displaced"
        finally:
            view.on_unmount()
    finally:
        root.removeHandler(file_handler)
        file_handler.close()


def test_composing_the_report_announces_when_it_finishes(
    workspace: WorkspacePaths, stub_stages: object, stub_report: Report
) -> None:
    """compose-report is the longest single call in a run — it reads every
    artifact and writes the whole narrative. Announcing only the start left a
    minute of silence at the very end that reads exactly like a hang."""
    watcher = Recorder()
    AutoRunner().run(QUESTION, observer=watcher)

    assert watcher.composing_calls == 1
    assert len(watcher.compositions) == 1
    seconds, error = watcher.compositions[0]
    assert error is None and seconds >= 0

    plain = runner.invoke(app, ["auto", QUESTION, "--run", "second", "--no-tui"])
    assert "report written" in plain.output


def test_a_failed_report_is_announced_too(
    workspace: WorkspacePaths, stub_stages: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(run_id: str, **kwargs: object) -> Composition:
        raise ReportUnavailableError(run_id, ["decision"])

    monkeypatch.setattr(auto_module, "compose_report", unavailable)
    watcher = Recorder()

    AutoRunner().run(QUESTION, observer=watcher)

    assert len(watcher.compositions) == 1
    assert watcher.compositions[0][1] is not None, "the failure was not reported"
