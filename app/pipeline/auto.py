"""One question in, one finished result out.

The pipeline is nine stages plus a report, and driving it by hand means ten
commands and watching for the one that fell over. This runs the whole thing:
seed the question, work the stages in order, retry a stage that failed for a
transient reason, compose the report, and record every attempt in the ledger.

What it deliberately does **not** do:

* **Decide anything a human should.** It stops at the report. Harvesting leads
  and standing up a validation experiment stay separate commands, because
  contacting people is a human act and a run that produced no opportunity worth
  contacting anyone about is a finished run, not a failed one.
* **Invent work.** It runs the run it was given. It does not generate follow-up
  questions, re-collect on a schedule, or start anything unattended — an agent
  that spends money while nobody is watching needs a budget and a gate, and
  neither exists yet.

Resumability is the engine's, unchanged: interrupt this and run it again and it
picks up at the first stage whose artifacts are not on disk, per item.
"""

from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import (
    ArtifactKind,
    ArtifactRegistry,
    Decision,
    Question,
    QuestionPriority,
    Report,
)
from app.llm import LLM
from app.llm.telemetry import default_sink
from app.models import RunStatus, StageState
from app.pipeline.engine import STAGE_ORDER, PipelineEngine, StageOutcome, StageStatus
from app.pipeline.reporting import ReportUnavailableError, compose_report
from app.storage.ledger import Ledger, ledger_scope
from app.storage.schema import create_all
from app.utils.errors import PipelineError
from app.utils.logging import get_logger
from app.utils.paths import get_workspace_paths
from app.utils.time import utcnow

logger = get_logger(__name__)


class StageAttempt(BaseModel):
    """One attempt at one stage, with what it did."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    attempt: int
    outcome: StageOutcome
    started_at: datetime
    finished_at: datetime

    @property
    def seconds(self) -> float:
        """How long the attempt took."""
        return (self.finished_at - self.started_at).total_seconds()


class Spend(BaseModel):
    """What a run cost, totalled from the telemetry log."""

    model_config = ConfigDict(extra="forbid")

    calls: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    byok: int = Field(
        default=0,
        description="Calls billed against your own provider key, where `cost` reads zero "
        "and the real number is on someone else's invoice.",
    )


class AutoResult(BaseModel):
    """Everything one `op auto` invocation did."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    question: str
    question_id: str
    attempts: list[StageAttempt] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    report: Report | None = None
    report_path: Path | None = None
    spend: Spend = Field(default_factory=Spend)
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = Field(
        default=None,
        description="Why the run stopped short, if it did. The stages that had already "
        "finished are still on disk, and re-running resumes from there.",
    )

    @property
    def ok(self) -> bool:
        """Whether every stage finished and, if one was asked for, the report composed."""
        return self.error is None

    @property
    def seconds(self) -> float | None:
        """Wall-clock duration, once the run has ended."""
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def verdicts(self) -> dict[str, int]:
        """How many opportunities got each verdict — the headline of the result."""
        tally: dict[str, int] = {}
        for decision in self.decisions:
            tally[decision.verdict.value] = tally.get(decision.verdict.value, 0) + 1
        return tally

    @property
    def status(self) -> RunStatus:
        """The ledger status this result implies."""
        return RunStatus.COMPLETED if self.ok else RunStatus.FAILED


class RunObserver(Protocol):
    """Watches a run as it happens.

    A full run is long — collection alone can take minutes — and a front end that
    only learns the outcome has nothing to show for most of it. Every method has
    a do-nothing default so an observer implements only the events it draws.
    """

    def stage_started(self, stage: str, attempt: int) -> None:
        """A stage is about to be attempted."""

    def stage_finished(self, attempt: StageAttempt) -> None:
        """An attempt ended, successfully or not."""

    def composing(self) -> None:
        """Every stage is done and the report is being written."""

    def composed(self, seconds: float, error: str | None) -> None:
        """The report finished, or could not be written."""


class _Silent:
    """The observer used when nobody is watching."""

    def stage_started(self, stage: str, attempt: int) -> None:
        """Ignored."""

    def stage_finished(self, attempt: StageAttempt) -> None:
        """Ignored."""

    def composing(self) -> None:
        """Ignored."""

    def composed(self, seconds: float, error: str | None) -> None:
        """Ignored."""


class AutoRunner:
    """Drives a whole run: question in, report out."""

    def __init__(
        self,
        registry: ArtifactRegistry | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.registry = registry if registry is not None else ArtifactRegistry()
        self.llm = llm
        self.engine = PipelineEngine(self.registry, llm)

    def run(
        self,
        question: str | None = None,
        *,
        run_id: str = "default",
        retries: int = 1,
        force: bool = False,
        report: bool = True,
        out: Path | None = None,
        observer: RunObserver | None = None,
    ) -> AutoResult:
        """Run every stage for `run_id`, then compose the report.

        Args:
            question: The research question. Optional when the run is already
                seeded — that is what makes re-invoking this a resume.
            run_id: The run to work on. Everything is scoped to it.
            retries: Extra attempts per stage after a failure. A stage that is
                *blocked* is never retried: it had no input, and asking again
                cannot conjure one.
            force: Re-run stages that are already complete, superseding what they
                replace. Costs a full run's worth of tokens.
            report: Compose the report at the end. Off, the run stops at the
                artifacts.
            out: Where to write the report body. Defaults to
                `workspace/reports/<run_id>.md`.
            observer: Told about each stage as it starts and finishes, so a front
                end can draw progress rather than a long silence.

        Raises:
            PipelineError: If the run cannot be started at all — no question to
                work from, or a question that contradicts the one already seeded.
        """
        get_workspace_paths().ensure()
        create_all()
        watcher: RunObserver = observer if observer is not None else _Silent()

        seeded = self._seed(run_id, question)
        started = utcnow()
        result = AutoResult(
            run_id=run_id,
            question=seeded.text,
            question_id=seeded.id,
            started_at=started,
        )

        with ledger_scope(registry=self.registry) as ledger:
            ledger.start_run(run_id, question=seeded, auto=True)

        for stage in STAGE_ORDER:
            attempt = self._run_stage(stage, run_id, retries=retries, force=force, watcher=watcher)
            result.attempts.append(attempt)
            watcher.stage_finished(attempt)
            if not attempt.outcome.ok:
                result.error = f"{stage} {attempt.outcome.status.value}: {attempt.outcome.reason}"
                break

        if result.error is None and report:
            watcher.composing()
            began = utcnow()
            self._compose(result, force=force, out=out)
            # Announced, because this is the longest single call in the run — it
            # reads every artifact and writes the whole narrative — and an
            # unannounced minute of silence at the very end reads as a hang.
            watcher.composed((utcnow() - began).total_seconds(), result.error)

        result.decisions = self._decisions(run_id)
        result.finished_at = utcnow()
        result.spend = self._spend(run_id)
        self._close(result)
        return result

    # ------------------------------------------------------------------ steps

    def _seed(self, run_id: str, question: str | None) -> Question:
        """Return the run's Question, writing it on first use.

        A run is identified by its id, and its question is what that id means. So
        re-running with a *different* question is refused rather than guessed at:
        silently appending a second question would leave every later stage
        reading two, and quietly answering neither.
        """
        existing = self.engine.question_for(run_id)
        text = (question or "").strip()

        if existing is not None:
            if text and text != existing.text:
                raise PipelineError(
                    f"run {run_id!r} is already asking {existing.text!r}. "
                    f"Use --run <new-id> for a different question."
                )
            return existing

        if not text:
            raise PipelineError(
                f'run {run_id!r} has no question yet — give one: op auto "…" --run {run_id}'
            )

        seeded = Question(
            id=Question.make_id(),
            run_id=run_id,
            text=text,
            priority=QuestionPriority.MEDIUM,
        )
        self.registry.save(seeded)
        return seeded

    def _run_stage(
        self, stage: str, run_id: str, *, retries: int, force: bool, watcher: RunObserver
    ) -> StageAttempt:
        """Attempt one stage, retrying a failure, and record every attempt.

        There is no sleep between attempts. Rate limits and 5xx are already
        backed off inside the LLM layer, which is the only place that knows what
        the gateway asked for; a second timer out here would just be a guess
        stacked on a fact. What this retry buys is the *rest* of the stage: a
        per-item stage resumes at the item, so attempt two asks only about the
        items attempt one never got to.
        """
        last: StageAttempt | None = None
        for number in range(1, retries + 2):
            watcher.stage_started(stage, number)
            started = utcnow()
            outcome = self.engine.run_stage(stage, run_id, force=force and number == 1)
            attempt = StageAttempt(
                stage=stage,
                attempt=number,
                outcome=outcome,
                started_at=started,
                finished_at=utcnow(),
            )
            self._record(run_id, attempt)
            last = attempt
            if outcome.ok or outcome.status is not StageStatus.FAILED:
                break
            if number <= retries:
                logger.warning("retrying %s after: %s", stage, outcome.reason)
        if last is None:  # pragma: no cover — the loop always runs at least once
            raise PipelineError(f"stage {stage!r} was never attempted")
        return last

    def _compose(self, result: AutoResult, *, force: bool, out: Path | None) -> None:
        """Compose the report and write it out, recording why if it could not be."""
        try:
            composition = compose_report(
                result.run_id, registry=self.registry, llm=self.llm, force=force
            )
        except ReportUnavailableError as exc:
            result.error = str(exc)
            return
        except Exception as exc:
            logger.exception("compose-report failed for run %s", result.run_id)
            result.error = f"compose-report failed: {exc}"
            return

        result.report = composition.report
        destination = out if out is not None else self._default_out(result.run_id)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(composition.report.body, encoding="utf-8")
            result.report_path = destination
        except OSError as exc:
            # The report artifact is safely on disk either way; failing to place
            # a convenience copy must not fail the run.
            logger.warning("could not write the report to %s: %s", destination, exc)

    def _default_out(self, run_id: str) -> Path:
        """Where a run's report lands when the caller does not choose."""
        return self.registry.paths.reports / f"{run_id}.md"

    def _decisions(self, run_id: str) -> list[Decision]:
        """The run's live verdicts, in the order they were decided."""
        found = [
            artifact
            for artifact in self.engine.consumable_of(ArtifactKind.DECISION, run_id)
            if isinstance(artifact, Decision)
        ]
        return sorted(found, key=lambda decision: decision.created_at)

    def _spend(self, run_id: str) -> Spend:
        """Total the run's telemetry. Never raises — a missing log costs a number."""
        try:
            calls = default_sink(self.registry.paths).read(run_id=run_id)
        except OSError as exc:
            logger.debug("could not read telemetry for %s: %s", run_id, exc)
            return Spend()
        return Spend(
            calls=len(calls),
            total_tokens=sum(call.total_tokens or 0 for call in calls),
            cost=sum(call.cost or 0.0 for call in calls),
            byok=sum(1 for call in calls if call.is_byok),
        )

    # ----------------------------------------------------------------- ledger

    def _record(self, run_id: str, attempt: StageAttempt) -> None:
        """Write one stage attempt to the ledger.

        Each attempt commits on its own, so the history survives whatever kills
        the process next. Ledger trouble is logged and swallowed: the ledger
        describes the research, and losing the description must never cost the
        research itself.
        """
        try:
            with ledger_scope(registry=self.registry) as ledger:
                ledger.record_stage(
                    run_id,
                    attempt.stage,
                    StageState(attempt.outcome.status.value),
                    produced=len(attempt.outcome.produced),
                    reused=attempt.outcome.reused,
                    detail=attempt.outcome.reason,
                    started_at=attempt.started_at,
                    finished_at=attempt.finished_at,
                )
        except Exception as exc:
            logger.warning("could not record stage %s in the ledger: %s", attempt.stage, exc)

    def _close(self, result: AutoResult) -> None:
        """Close the run out in the ledger and index what it produced."""
        try:
            with ledger_scope(registry=self.registry) as ledger:
                self._close_with(ledger, result)
        except Exception as exc:
            logger.warning("could not close run %s in the ledger: %s", result.run_id, exc)

    @staticmethod
    def _close_with(ledger: Ledger, result: AutoResult) -> None:
        ledger.record_cost(
            result.run_id,
            calls=result.spend.calls,
            total_tokens=result.spend.total_tokens,
            cost=result.spend.cost,
        )
        ledger.sync_run(result.run_id)
        ledger.finish_run(
            result.run_id,
            status=result.status,
            error=result.error,
            report_id=result.report.id if result.report else None,
            at=result.finished_at,
        )


__all__ = ["AutoResult", "AutoRunner", "RunObserver", "Spend", "StageAttempt"]
