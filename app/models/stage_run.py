"""`stage_runs` — one row per attempt at one stage.

This is the table that makes the ledger worth having. Resume is answerable from
the workspace alone (a stage is done when its artifacts are on disk), but *how it
went* is not: which stage failed twice before it took, how long collection ran,
whether a stage was skipped because it was already done or blocked because
nothing upstream produced anything. That history is gone the moment the process
exits unless something writes it down.
"""

from datetime import datetime
from enum import StrEnum

from sqlalchemy import UniqueConstraint
from sqlmodel import Field

from app.models.base import TimestampMixin, UTCDateTime


class StageState(StrEnum):
    """How one attempt at a stage ended.

    Mirrors `app.pipeline.StageStatus` by value but is declared separately on
    purpose: these strings are a storage contract that outlives any refactor of
    the runtime enum, and the ledger must not import the pipeline.
    """

    COMPLETED = "completed"
    SKIPPED = "skipped"
    EMPTY = "empty"
    BLOCKED = "blocked"
    FAILED = "failed"


class StageRun(TimestampMixin, table=True):
    """One attempt at one stage of one run."""

    __tablename__ = "stage_runs"
    __table_args__ = (UniqueConstraint("run_id", "stage", "attempt", name="uq_stage_attempt"),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    stage: str = Field(index=True, description="Pipeline stage name, e.g. 'collect-evidence'.")
    attempt: int = Field(default=1, ge=1, description="1 for the first try, 2 after a retry.")

    state: StageState = Field(index=True)
    produced: int = Field(default=0, description="Artifacts written by this attempt.")
    reused: int = Field(
        default=0,
        description="Primary items an earlier interrupted attempt had already finished.",
    )
    detail: str | None = Field(
        default=None,
        description="Why it was skipped or blocked, or how it failed.",
    )

    started_at: datetime | None = Field(default=None, sa_type=UTCDateTime)
    finished_at: datetime | None = Field(default=None, sa_type=UTCDateTime)

    @property
    def duration_seconds(self) -> float | None:
        """How long the attempt took, once both ends are known."""
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


__all__ = ["StageRun", "StageState"]
