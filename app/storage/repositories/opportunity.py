"""Opportunity index access."""

from typing import Any

from sqlmodel import col, select

from app.models.opportunity import OpportunityRecord
from app.storage.repositories.base import BaseRepository


class OpportunityRepository(BaseRepository[OpportunityRecord]):
    """Queries scoped to the `opportunities` table."""

    model = OpportunityRecord

    def by_artifact(self, artifact_id: str) -> OpportunityRecord | None:
        """The row indexing one opportunity artifact."""
        statement = select(OpportunityRecord).where(OpportunityRecord.artifact_id == artifact_id)
        return self.session.exec(statement).first()

    def by_run(self, run_id: str) -> list[OpportunityRecord]:
        """Every opportunity a run produced, oldest first."""
        statement = (
            select(OpportunityRecord)
            .where(OpportunityRecord.run_id == run_id)
            .order_by(col(OpportunityRecord.id))
        )
        return list(self.session.exec(statement).all())

    def decided(self, verdict: str | None = None) -> list[OpportunityRecord]:
        """Ruled-on opportunities across every run, newest decision first.

        This is the query the index exists for: "everything we ever said build
        to" is one statement here, and a walk of every decision file otherwise.
        """
        statement = select(OpportunityRecord).where(col(OpportunityRecord.verdict).is_not(None))
        if verdict is not None:
            statement = statement.where(OpportunityRecord.verdict == verdict)
        statement = statement.order_by(col(OpportunityRecord.decided_at).desc().nullslast())
        return list(self.session.exec(statement).all())

    def undecided(self) -> list[OpportunityRecord]:
        """Opportunities no decision has ruled on yet, across every run."""
        statement = select(OpportunityRecord).where(col(OpportunityRecord.verdict).is_(None))
        return list(self.session.exec(statement).all())

    def upsert(self, artifact_id: str, **fields: Any) -> OpportunityRecord:
        """Index an artifact, updating the row if it is already indexed."""
        found = self.by_artifact(artifact_id)
        if found is None:
            return self.add(OpportunityRecord(artifact_id=artifact_id, **fields))
        return self.apply(found, **fields)


__all__ = ["OpportunityRepository"]
