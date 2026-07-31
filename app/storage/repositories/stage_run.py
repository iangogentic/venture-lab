"""Stage-attempt history access."""

from sqlalchemy import func
from sqlmodel import col, select

from app.models.stage_run import StageRun
from app.storage.repositories.base import BaseRepository


class StageRunRepository(BaseRepository[StageRun]):
    """Queries scoped to the `stage_runs` table."""

    model = StageRun

    def for_run(self, run_id: str) -> list[StageRun]:
        """Every attempt in a run, oldest first — the run's story in order."""
        statement = select(StageRun).where(StageRun.run_id == run_id).order_by(col(StageRun.id))
        return list(self.session.exec(statement).all())

    def for_stage(self, run_id: str, stage: str) -> list[StageRun]:
        """Every attempt at one stage of one run, oldest first."""
        statement = (
            select(StageRun)
            .where(StageRun.run_id == run_id, StageRun.stage == stage)
            .order_by(col(StageRun.attempt))
        )
        return list(self.session.exec(statement).all())

    def next_attempt(self, run_id: str, stage: str) -> int:
        """The attempt number to record next for this stage.

        Derived from the highest attempt already recorded rather than a row
        count, so deleting history renumbers nothing and the unique constraint
        on (run, stage, attempt) keeps holding.
        """
        statement = select(func.max(StageRun.attempt)).where(
            StageRun.run_id == run_id, StageRun.stage == stage
        )
        return int(self.session.scalar(statement) or 0) + 1

    def latest(self, run_id: str, stage: str) -> StageRun | None:
        """The most recent attempt at one stage, if there has been one."""
        statement = (
            select(StageRun)
            .where(StageRun.run_id == run_id, StageRun.stage == stage)
            .order_by(col(StageRun.attempt).desc())
            .limit(1)
        )
        return self.session.exec(statement).first()


__all__ = ["StageRunRepository"]
