"""The run ledger: recording what happened, and indexing what it produced."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session

from app.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactRegistry,
    ArtifactStatus,
    Verdict,
)
from app.models import Run, RunStatus, StageState
from app.storage.ledger import Ledger
from app.utils.errors import StorageError
from tests import factories

RUN = "run_test"


@pytest.fixture
def registry(workspace: object) -> ArtifactRegistry:
    """A registry over the test workspace."""
    return ArtifactRegistry()


@pytest.fixture
def ledger(session: Session, registry: ArtifactRegistry) -> Ledger:
    """A ledger bound to the test session and workspace."""
    return Ledger(session, registry)


# --------------------------------------------------------------------- CRUD


def test_repository_round_trips_a_row(ledger: Ledger) -> None:
    """The base repository is implemented, not a scaffold that raises."""
    ledger.runs.add(Run(id=RUN, question="Where does time go?"))

    assert ledger.runs.get(RUN) is not None
    assert ledger.runs.count() == 1
    assert [run.id for run in ledger.runs.list()] == [RUN]

    ledger.runs.update(RUN, status=RunStatus.COMPLETED)
    assert ledger.runs.require(RUN).status is RunStatus.COMPLETED

    ledger.runs.delete(RUN)
    assert ledger.runs.get(RUN) is None
    ledger.runs.delete(RUN)  # deleting twice is not an error


def test_updating_an_unknown_field_raises(ledger: Ledger) -> None:
    """A typo in a ledger write must not look like a successful write."""
    ledger.runs.add(Run(id=RUN))
    with pytest.raises(StorageError, match="has no field 'staus'"):
        ledger.runs.update(RUN, staus=RunStatus.FAILED)


def test_requiring_a_missing_row_raises(ledger: Ledger) -> None:
    with pytest.raises(StorageError, match="no Run row"):
        ledger.runs.require("nope")


def test_timestamps_survive_the_round_trip_tz_aware(ledger: Ledger, session: Session) -> None:
    """Regression: a plain DateTime column drops the offset, so `created_at` came
    back naive and every comparison against an artifact's timestamp raised."""
    ledger.runs.add(Run(id=RUN))
    session.commit()
    session.expire_all()

    stored = ledger.runs.require(RUN)
    assert stored.created_at.tzinfo is not None
    assert (datetime.now(UTC) - stored.created_at) < timedelta(minutes=1)


# ---------------------------------------------------------------- recording


def test_start_run_keeps_the_original_start_time(ledger: Ledger) -> None:
    """A run resumed on Thursday still started on Tuesday."""
    question = factories.make(ArtifactKind.QUESTION, run_id=RUN)
    first = ledger.start_run(RUN, question=question, auto=True)  # type: ignore[arg-type]
    began = first.started_at

    ledger.finish_run(RUN, status=RunStatus.FAILED, error="rate limited")
    resumed = ledger.start_run(RUN)

    assert resumed.started_at == began
    assert resumed.status is RunStatus.RUNNING
    assert resumed.error is None, "resuming clears the previous failure"
    assert resumed.question_id == question.id


def test_stage_attempts_are_numbered_and_kept(ledger: Ledger) -> None:
    """The workspace records outcomes; only the ledger records the attempts."""
    ledger.start_run(RUN)
    ledger.record_stage(RUN, "collect-evidence", StageState.FAILED, detail="429")
    ledger.record_stage(RUN, "collect-evidence", StageState.COMPLETED, produced=12)

    attempts = ledger.stages.for_stage(RUN, "collect-evidence")
    assert [a.attempt for a in attempts] == [1, 2]
    assert [a.state for a in attempts] == [StageState.FAILED, StageState.COMPLETED]
    assert ledger.stages.next_attempt(RUN, "collect-evidence") == 3
    assert ledger.runs.require(RUN).stage == "collect-evidence"


def test_cost_is_set_not_accumulated(ledger: Ledger) -> None:
    """Telemetry is the account; the ledger mirrors its total, so a resume that
    re-reads the same records must not double the money."""
    ledger.start_run(RUN)
    ledger.record_cost(RUN, calls=4, total_tokens=1_000, cost=0.12)
    ledger.record_cost(RUN, calls=6, total_tokens=1_500, cost=0.19)

    run = ledger.runs.require(RUN)
    assert (run.calls, run.total_tokens, run.cost) == (6, 1_500, 0.19)


def test_finish_run_records_the_deliverable(ledger: Ledger) -> None:
    ledger.start_run(RUN)
    ledger.finish_run(RUN, status=RunStatus.COMPLETED, report_id="rep_1")

    run = ledger.runs.require(RUN)
    assert run.finished and run.report_id == "rep_1"
    assert run.duration_seconds is not None and run.duration_seconds >= 0


def test_unfinished_finds_a_run_whose_process_died(ledger: Ledger) -> None:
    ledger.start_run(RUN)
    ledger.start_run("other")
    ledger.finish_run("other", status=RunStatus.COMPLETED)

    assert [run.id for run in ledger.runs.unfinished()] == [RUN]


# --------------------------------------------------------------- projecting


def _seed_run(registry: ArtifactRegistry) -> tuple[str, str]:
    """A workspace holding a question, two evidence items, an opportunity, a decision."""
    question = factories.make(ArtifactKind.QUESTION, run_id=RUN)
    registry.save(question)
    registry.save(
        factories.make(
            ArtifactKind.EVIDENCE,
            run_id=RUN,
            collector="hackernews",
            source_id="42",
            source_url="https://news.ycombinator.com/item?id=42",
            status=ArtifactStatus.READY,
        )
    )
    registry.save(
        factories.make(
            ArtifactKind.EVIDENCE,
            run_id=RUN,
            collector="filesystem",
            source_url=None,
            status=ArtifactStatus.READY,
        )
    )
    opportunity = factories.make(ArtifactKind.OPPORTUNITY, run_id=RUN, confidence=0.7)
    registry.save(opportunity)
    decision = factories.make(
        ArtifactKind.DECISION,
        run_id=RUN,
        opportunity=ArtifactRef(kind=ArtifactKind.OPPORTUNITY, id=opportunity.id),
        verdict=Verdict.BUILD,
    )
    registry.save(decision)
    return opportunity.id, decision.id


def test_sync_indexes_evidence_and_its_origins(ledger: Ledger, registry: ArtifactRegistry) -> None:
    _seed_run(registry)
    counts = ledger.sync_run(RUN)

    assert counts.evidence == 2
    rows = ledger.evidence.by_run(RUN)
    hn = next(row for row in rows if row.collector == "hackernews")
    assert hn.dedup_key == "hackernews:42"
    assert hn.url == "https://news.ycombinator.com/item?id=42"

    origins = {(s.collector, s.origin) for s in ledger.sources.list()}
    assert ("hackernews", "news.ycombinator.com") in origins
    # No URL to derive a host from, so the collector names the origin rather
    # than the row being dropped.
    assert ("filesystem", "filesystem") in origins


def test_sync_carries_the_verdict_onto_the_opportunity(
    ledger: Ledger, registry: ArtifactRegistry
) -> None:
    """The query the index exists for: every build verdict, without a directory walk."""
    opportunity_id, decision_id = _seed_run(registry)
    ledger.sync_run(RUN)

    indexed = ledger.opportunities.by_artifact(opportunity_id)
    assert indexed is not None
    assert indexed.verdict == Verdict.BUILD.value
    assert indexed.decision_id == decision_id
    assert indexed.decision_confidence == pytest.approx(0.6)
    assert [row.artifact_id for row in ledger.opportunities.decided("build")] == [opportunity_id]
    assert ledger.opportunities.undecided() == []


def test_sync_ignores_a_superseded_decision(ledger: Ledger, registry: ArtifactRegistry) -> None:
    """A forced re-run leaves two decisions on disk; only the live one is a verdict."""
    opportunity_id, stale_id = _seed_run(registry)
    stale = registry.load(ArtifactKind.DECISION, stale_id)
    registry.update(stale, status=ArtifactStatus.SUPERSEDED)
    fresh = factories.make(
        ArtifactKind.DECISION,
        run_id=RUN,
        opportunity=ArtifactRef(kind=ArtifactKind.OPPORTUNITY, id=opportunity_id),
        verdict=Verdict.REJECT,
    )
    registry.save(fresh)

    ledger.sync_run(RUN)

    indexed = ledger.opportunities.by_artifact(opportunity_id)
    assert indexed is not None
    assert indexed.verdict == Verdict.REJECT.value
    assert indexed.decision_id == fresh.id


def test_sync_is_idempotent(ledger: Ledger, registry: ArtifactRegistry) -> None:
    """`op runs sync` reprojects the whole workspace; rows must update, not double."""
    _seed_run(registry)
    ledger.sync_run(RUN)
    ledger.sync_run(RUN)

    assert ledger.evidence.count() == 2
    assert ledger.opportunities.count() == 1
    assert ledger.sources.count() == 2


def test_sync_all_discovers_runs_from_the_workspace(
    ledger: Ledger, registry: ArtifactRegistry
) -> None:
    """A workspace that predates the ledger — or arrived from another machine —
    indexes without anyone listing what is in it."""
    _seed_run(registry)
    registry.save(factories.make(ArtifactKind.QUESTION, run_id="second"))

    counts = ledger.sync_all()

    assert counts.runs == 2
    assert {run.id for run in ledger.runs.list()} == {RUN, "second"}
    assert ledger.runs.require(RUN).question_id is not None


def test_sync_backfills_the_question_and_report(ledger: Ledger, registry: ArtifactRegistry) -> None:
    _seed_run(registry)
    report = factories.make(ArtifactKind.REPORT, run_id=RUN, status=ArtifactStatus.READY)
    registry.save(report)

    ledger.sync_run(RUN)

    run = ledger.runs.require(RUN)
    assert run.report_id == report.id
    assert run.question == "Where do platform teams lose time?"


def test_source_yields_counts_evidence_per_origin(
    ledger: Ledger, registry: ArtifactRegistry
) -> None:
    """Which configured origins have ever paid for themselves."""
    _seed_run(registry)
    ledger.sources.seen("discourse", "forum.example.com")
    ledger.sync_run(RUN)

    yields = {source.origin: total for source, total in ledger.sources.yields()}
    assert yields["news.ycombinator.com"] == 1
    assert yields["forum.example.com"] == 0, "configured but never productive"
