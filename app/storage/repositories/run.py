"""Run ledger access."""

from typing import Any

from sqlmodel import col, select

from app.models.run import Run, RunStatus
from app.storage.repositories.base import BaseRepository

_OPEN: tuple[RunStatus, ...] = (RunStatus.PENDING, RunStatus.RUNNING)
"""Statuses a run can still be picked up from."""


class RunRepository(BaseRepository[Run]):
    """Queries scoped to the `runs` table."""

    model = Run

    def ensure(self, run_id: str, **defaults: Any) -> Run:
        """Fetch the run, creating it from `defaults` if this is its first sighting.

        Get-or-create rather than insert, because a run id is chosen by the
        operator (`--run pricing-pain`) rather than minted: the same id
        legitimately shows up again tomorrow when they resume it.
        """
        found = self.get(run_id)
        if found is not None:
            return found
        return self.add(Run(id=run_id, **defaults))

    def recent(self, *, limit: int = 20) -> list[Run]:
        """Runs newest first, by when they started — or were created, if never started."""
        statement = (
            select(Run)
            .order_by(col(Run.started_at).desc().nullslast(), col(Run.created_at).desc())
            .limit(limit)
        )
        return list(self.session.exec(statement).all())

    def unfinished(self) -> list[Run]:
        """Runs never carried to a terminal status.

        Includes runs whose process was killed mid-stage: nothing resets a
        `running` row, and that is the point — a `running` run with a stale
        `updated_at` is exactly the evidence that something died.
        """
        statement = select(Run).where(col(Run.status).in_(_OPEN))
        return list(self.session.exec(statement).all())

    def with_status(self, status: RunStatus) -> list[Run]:
        """Every run in one status."""
        statement = select(Run).where(Run.status == status)
        return list(self.session.exec(statement).all())


__all__ = ["RunRepository"]
