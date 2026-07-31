"""`runs` — one row per run id, the unit everything else is scoped to.

Not one row per invocation: a run is resumed, re-run and extended over days, and
minting a new row each time would make "how did this question go?" unanswerable.
What each *attempt* did is `stage_runs`.

The row is a summary, not a source of truth. Every finding lives in `workspace/`
as JSON; this records what happened around the findings — when the run started,
where it stopped, what it cost, and which report came out of it. Delete the
database and `op runs sync` rebuilds it from the workspace.
"""

from datetime import datetime
from enum import StrEnum

from sqlmodel import Field

from app.models.base import TimestampMixin, UTCDateTime


class RunStatus(StrEnum):
    """Lifecycle of a run."""

    PENDING = "pending"
    """Seeded with a question; no stage attempted yet."""

    RUNNING = "running"
    """A stage is in flight — or the process died before it could say otherwise."""

    COMPLETED = "completed"
    """Every stage attempted finished, and none is left pending."""

    FAILED = "failed"
    """A stage ran out of attempts, or was blocked with no input to work from."""


class Run(TimestampMixin, table=True):
    """One research run: its question, how far it got, and what it cost."""

    __tablename__ = "runs"

    id: str = Field(primary_key=True)
    question: str | None = Field(default=None, description="The question text, for display.")
    question_id: str | None = Field(
        default=None,
        index=True,
        description="Artifact id of the Question that seeded the run.",
    )
    status: RunStatus = Field(default=RunStatus.PENDING, index=True)
    auto: bool = Field(
        default=False,
        description="Driven end to end by `op auto` rather than stage by stage. Worth "
        "recording: an unattended run and a hand-driven one fail in different ways.",
    )

    stage: str | None = Field(
        default=None,
        description="The last stage attempted — the cursor a resume picks up from.",
    )
    error: str | None = Field(default=None, description="Why the run stopped, if it did.")

    started_at: datetime | None = Field(default=None, sa_type=UTCDateTime)
    finished_at: datetime | None = Field(default=None, sa_type=UTCDateTime)

    calls: int = Field(default=0, description="Model calls attributed to this run.")
    total_tokens: int = Field(default=0)
    cost: float = Field(
        default=0.0,
        description="Gateway-reported spend in USD. Reads zero for BYOK calls, which are "
        "billed against your own provider key and are real somewhere else.",
    )

    report_id: str | None = Field(
        default=None,
        description="Artifact id of the composed Report — the run's deliverable.",
    )

    @property
    def finished(self) -> bool:
        """Whether the run reached a terminal status."""
        return self.status in (RunStatus.COMPLETED, RunStatus.FAILED)

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock time from first stage to last, once both ends are known."""
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


__all__ = ["Run", "RunStatus"]
