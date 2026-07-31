"""Evidence index access."""

from typing import Any

from sqlalchemy import func
from sqlmodel import col, select

from app.models.evidence import EvidenceRecord
from app.storage.repositories.base import BaseRepository


class EvidenceRepository(BaseRepository[EvidenceRecord]):
    """Queries scoped to the `evidence` table."""

    model = EvidenceRecord

    def by_artifact(self, artifact_id: str) -> EvidenceRecord | None:
        """The row indexing one evidence artifact."""
        statement = select(EvidenceRecord).where(EvidenceRecord.artifact_id == artifact_id)
        return self.session.exec(statement).first()

    def by_run(self, run_id: str) -> list[EvidenceRecord]:
        """Everything collected for one run, oldest first."""
        statement = (
            select(EvidenceRecord)
            .where(EvidenceRecord.run_id == run_id)
            .order_by(col(EvidenceRecord.id))
        )
        return list(self.session.exec(statement).all())

    def by_dedup_key(self, dedup_key: str) -> list[EvidenceRecord]:
        """Every run that kept this exact source item.

        More than one row means the same item was collected again for a later
        question — worth knowing, and not something the per-run dedup can see.
        """
        statement = select(EvidenceRecord).where(EvidenceRecord.dedup_key == dedup_key)
        return list(self.session.exec(statement).all())

    def count_for_run(self, run_id: str) -> int:
        """How many evidence artifacts a run holds."""
        statement = (
            select(func.count()).select_from(EvidenceRecord).where(EvidenceRecord.run_id == run_id)
        )
        return int(self.session.scalar(statement) or 0)

    def upsert(self, artifact_id: str, **fields: Any) -> EvidenceRecord:
        """Index an artifact, updating the row if it is already indexed.

        Re-indexing has to be safe: `op runs sync` projects the whole workspace
        every time it runs, and an artifact whose status changed must update in
        place rather than arrive twice.
        """
        found = self.by_artifact(artifact_id)
        if found is None:
            return self.add(EvidenceRecord(artifact_id=artifact_id, **fields))
        return self.apply(found, **fields)


__all__ = ["EvidenceRepository"]
