"""The run ledger: record what happened, and index what it produced.

Two jobs, deliberately kept in one object because they answer one question
between them — "how did this run go?".

*Recording* is live: a run declares itself started, each stage attempt is written
as it finishes, and the run is closed out with a status and a cost. Nothing else
in the system knows that history, because the workspace only remembers outcomes,
not attempts.

*Indexing* is a projection. Every column in `evidence`, `opportunities` and
`sources` is derived from artifacts already on disk, so the index can always be
thrown away and rebuilt (`op runs sync`). That rule is what stops the ledger
becoming a second source of truth that quietly disagrees with the artifacts: if
the two ever differ, the JSON is right.

The ledger deliberately does not import the pipeline. It is handed values —
a stage name, a state, a count — rather than a `StageOutcome`, so storage stays
underneath the thing it records instead of beside it.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict
from sqlalchemy.engine import Engine
from sqlmodel import Session

from app.artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactRegistry,
    ArtifactStatus,
    Decision,
    Evidence,
    Opportunity,
    Question,
    Report,
)
from app.models.base import utcnow
from app.models.run import Run, RunStatus
from app.models.stage_run import StageRun, StageState
from app.storage.repositories import (
    EvidenceRepository,
    OpportunityRepository,
    RunRepository,
    SourceRepository,
    StageRunRepository,
)
from app.storage.session import session_scope
from app.utils.logging import get_logger

logger = get_logger(__name__)

_RETIRED = frozenset({ArtifactStatus.SUPERSEDED, ArtifactStatus.ARCHIVED})
"""Artifact statuses whose content is no longer in play. Mirrors the engine's rule."""


class SyncCounts(BaseModel):
    """What one projection pass indexed."""

    model_config = ConfigDict(extra="forbid")

    runs: int = 0
    evidence: int = 0
    opportunities: int = 0
    sources: int = 0
    decided: int = 0

    def __add__(self, other: "SyncCounts") -> "SyncCounts":
        return SyncCounts(
            runs=self.runs + other.runs,
            evidence=self.evidence + other.evidence,
            opportunities=self.opportunities + other.opportunities,
            sources=self.sources + other.sources,
            decided=self.decided + other.decided,
        )


class Ledger:
    """Records runs, and indexes the artifacts they produce."""

    def __init__(self, session: Session, registry: ArtifactRegistry | None = None) -> None:
        self.session = session
        self.registry = registry if registry is not None else ArtifactRegistry()
        self.runs = RunRepository(session)
        self.stages = StageRunRepository(session)
        self.evidence = EvidenceRepository(session)
        self.opportunities = OpportunityRepository(session)
        self.sources = SourceRepository(session)

    # -------------------------------------------------------------- recording

    def start_run(
        self,
        run_id: str,
        *,
        question: Question | None = None,
        auto: bool = False,
        at: datetime | None = None,
    ) -> Run:
        """Open a run, or reopen one being resumed.

        `started_at` is set once and never moved: a run resumed on Thursday
        started on Tuesday, and overwriting that would erase how long the
        question has really been open.
        """
        run = self.runs.ensure(run_id)
        changes: dict[str, object] = {"status": RunStatus.RUNNING, "error": None, "auto": auto}
        if run.started_at is None:
            changes["started_at"] = at if at is not None else utcnow()
        if question is not None:
            changes["question"] = question.text
            changes["question_id"] = question.id
        return self.runs.apply(run, **changes)

    def record_stage(
        self,
        run_id: str,
        stage: str,
        state: StageState,
        *,
        produced: int = 0,
        reused: int = 0,
        detail: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> StageRun:
        """Write one stage attempt, and move the run's cursor to that stage."""
        attempt = self.stages.next_attempt(run_id, stage)
        recorded = self.stages.add(
            StageRun(
                run_id=run_id,
                stage=stage,
                attempt=attempt,
                state=state,
                produced=produced,
                reused=reused,
                detail=detail,
                started_at=started_at,
                finished_at=finished_at if finished_at is not None else utcnow(),
            )
        )
        run = self.runs.get(run_id)
        if run is not None:
            self.runs.apply(run, stage=stage)
        return recorded

    def record_cost(
        self,
        run_id: str,
        *,
        calls: int,
        total_tokens: int,
        cost: float,
    ) -> Run:
        """Set a run's spend totals.

        Set, not incremented: the caller totals the run's telemetry records, which
        are the durable account. Incrementing here would double-count the moment
        a run is resumed, and the two stores would disagree about money.
        """
        return self.runs.apply(
            self.runs.ensure(run_id), calls=calls, total_tokens=total_tokens, cost=cost
        )

    def finish_run(
        self,
        run_id: str,
        *,
        status: RunStatus,
        error: str | None = None,
        report_id: str | None = None,
        at: datetime | None = None,
    ) -> Run:
        """Close a run out with how it ended."""
        run = self.runs.ensure(run_id)
        changes: dict[str, object] = {
            "status": status,
            "error": error,
            "finished_at": at if at is not None else utcnow(),
        }
        if report_id is not None:
            changes["report_id"] = report_id
        return self.runs.apply(run, **changes)

    # ------------------------------------------------------------- projecting

    def sync_run(self, run_id: str) -> SyncCounts:
        """Rebuild the index for one run from its artifacts on disk."""
        run = self.runs.ensure(run_id)
        counts = SyncCounts(runs=1)

        question = self._first(ArtifactKind.QUESTION, run_id, Question)
        if question is not None and run.question_id != question.id:
            self.runs.apply(run, question=question.text, question_id=question.id)

        counts.evidence, counts.sources = self._index_evidence(run_id)
        counts.opportunities, counts.decided = self._index_opportunities(run_id)

        report = self._live_report(run_id)
        if report is not None and run.report_id != report.id:
            self.runs.apply(run, report_id=report.id)
        return counts

    def sync_all(self) -> SyncCounts:
        """Rebuild the index for every run the workspace knows about.

        Runs are discovered from the artifacts themselves, so a workspace copied
        from another machine — or one predating the ledger entirely — indexes
        without anyone having to list what is in it.
        """
        total = SyncCounts()
        for run_id in sorted(self.known_run_ids()):
            total = total + self.sync_run(run_id)
        return total

    def known_run_ids(self) -> set[str]:
        """Every run id appearing anywhere in the workspace or the ledger."""
        found = {run.id for run in self.runs.list(limit=10_000)}
        for kind in ArtifactKind:
            found |= {artifact.run_id for artifact in self.registry.iter_by_type(kind)}
        return found

    def _index_evidence(self, run_id: str) -> tuple[int, int]:
        """Project a run's evidence artifacts, and the origins they came from."""
        origins: set[int] = set()
        indexed = 0
        for artifact in self.registry.find_by_type(ArtifactKind.EVIDENCE, run_id=run_id):
            if not isinstance(artifact, Evidence):
                continue
            url = str(artifact.source_url) if artifact.source_url is not None else None
            source = self.sources.seen(
                artifact.collector, _origin_of(url, artifact.collector), at=artifact.captured_at
            )
            if source.id is not None:
                origins.add(source.id)
            self.evidence.upsert(
                artifact.id,
                run_id=run_id,
                source_id=source.id,
                dedup_key=_dedup_key(artifact.collector, artifact.source_id),
                collector=artifact.collector,
                external_id=artifact.source_id,
                evidence_kind=artifact.evidence_kind.value,
                title=artifact.title,
                url=url,
                author=artifact.author,
                content_hash=artifact.content_hash,
                status=artifact.status.value,
                published_at=artifact.published_at,
                captured_at=artifact.captured_at,
            )
            indexed += 1
        return indexed, len(origins)

    def _index_opportunities(self, run_id: str) -> tuple[int, int]:
        """Project a run's opportunities, carrying across whatever ruled on them."""
        verdicts = self._decisions_by_opportunity(run_id)
        indexed = 0
        for artifact in self.registry.find_by_type(ArtifactKind.OPPORTUNITY, run_id=run_id):
            if not isinstance(artifact, Opportunity):
                continue
            decision = verdicts.get(artifact.id)
            self.opportunities.upsert(
                artifact.id,
                run_id=run_id,
                title=artifact.title,
                icp=artifact.icp,
                cluster_id=artifact.pain_cluster.id if artifact.pain_cluster else None,
                status=artifact.status.value,
                confidence=artifact.confidence,
                verdict=decision.verdict.value if decision else None,
                decision_id=decision.id if decision else None,
                decision_confidence=decision.decision_confidence if decision else None,
                decided_at=decision.updated_at if decision else None,
            )
            indexed += 1
        return indexed, len(verdicts)

    def _decisions_by_opportunity(self, run_id: str) -> dict[str, Decision]:
        """The live decision for each opportunity in a run.

        Superseded and archived decisions are skipped, so a forced re-run leaves
        the index holding the verdict that is actually in play rather than
        whichever file happened to be read last.
        """
        rulings: dict[str, Decision] = {}
        for artifact in self.registry.find_by_type(ArtifactKind.DECISION, run_id=run_id):
            if not isinstance(artifact, Decision):
                continue
            if artifact.status in _RETIRED:
                continue
            rulings[artifact.opportunity.id] = artifact
        return rulings

    def _live_report(self, run_id: str) -> Report | None:
        """The report a run currently stands behind — newest, and not superseded."""
        live = [
            artifact
            for artifact in self.registry.find_by_type(ArtifactKind.REPORT, run_id=run_id)
            if isinstance(artifact, Report) and artifact.status not in _RETIRED
        ]
        return max(live, key=lambda report: report.created_at) if live else None

    def _first[ArtifactT: Artifact](
        self, kind: ArtifactKind, run_id: str, model: type[ArtifactT]
    ) -> ArtifactT | None:
        """The earliest artifact of a kind in a run, by creation time.

        Earliest rather than arbitrary: ids are creation-ordered but directory
        listing is not, and "the question this run started from" has to be stable
        across machines.
        """
        found = [
            artifact
            for artifact in self.registry.find_by_type(kind, run_id=run_id)
            if isinstance(artifact, model)
        ]
        if not found:
            return None
        return min(found, key=lambda artifact: artifact.created_at)


def _origin_of(url: str | None, collector: str) -> str:
    """The host an item came from, falling back to the collector that fetched it.

    Local corpora and API results without a URL still deserve a row — "12 pieces
    of evidence, origin unknowable" is a fact about the run, not a gap to hide.
    """
    if not url:
        return collector
    host = urlsplit(url).netloc.lower()
    return host.removeprefix("www.") if host else collector


def _dedup_key(collector: str, external_id: str | None) -> str | None:
    """The exact-match dedup key, or None when the item carries no id at its source."""
    return f"{collector}:{external_id}" if external_id else None


@contextmanager
def ledger_scope(
    engine: Engine | None = None, registry: ArtifactRegistry | None = None
) -> Iterator[Ledger]:
    """Yield a Ledger on a session that commits on success and rolls back on error.

    The unit of work is the whole `with` block, so a run that dies mid-projection
    leaves the index as it was rather than half-rewritten.
    """
    with session_scope(engine) as session:
        yield Ledger(session, registry)


__all__ = ["Ledger", "SyncCounts", "ledger_scope"]
