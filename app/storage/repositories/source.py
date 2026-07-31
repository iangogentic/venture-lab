"""Source (origin) access."""

from datetime import datetime

from sqlalchemy import func
from sqlmodel import col, select

from app.models.evidence import EvidenceRecord
from app.models.source import Source
from app.storage.repositories.base import BaseRepository


class SourceRepository(BaseRepository[Source]):
    """Queries scoped to the `sources` table."""

    model = Source

    def find(self, collector: str, origin: str) -> Source | None:
        """One origin, by the pair that identifies it."""
        statement = select(Source).where(Source.collector == collector, Source.origin == origin)
        return self.session.exec(statement).first()

    def by_collector(self, collector: str) -> list[Source]:
        """Every origin one collector has yielded evidence from."""
        statement = select(Source).where(Source.collector == collector).order_by(col(Source.origin))
        return list(self.session.exec(statement).all())

    def seen(self, collector: str, origin: str, *, at: datetime | None = None) -> Source:
        """Record that this origin yielded something, creating it on first sight.

        `first_seen_at` is only ever set once and `last_seen_at` only ever moves
        forward, so re-projecting the workspace out of order — which `op runs
        sync` does, since it walks runs in whatever order the filesystem gives —
        cannot make an origin look newer or older than it is.
        """
        found = self.find(collector, origin)
        if found is None:
            return self.add(
                Source(collector=collector, origin=origin, first_seen_at=at, last_seen_at=at)
            )
        if at is None:
            return found
        changes: dict[str, datetime] = {}
        if found.first_seen_at is None or at < found.first_seen_at:
            changes["first_seen_at"] = at
        if found.last_seen_at is None or at > found.last_seen_at:
            changes["last_seen_at"] = at
        return self.apply(found, **changes) if changes else found

    def yields(self) -> list[tuple[Source, int]]:
        """Every origin with how much evidence came from it, richest first.

        The question worth asking of a wide collector configuration: thirteen
        forums are cheap to list and expensive to search, and this says which
        ones have ever paid for themselves.
        """
        statement = (
            select(Source, func.count(col(EvidenceRecord.id)))
            .outerjoin(EvidenceRecord, col(EvidenceRecord.source_id) == col(Source.id))
            .group_by(col(Source.id))
            .order_by(func.count(col(EvidenceRecord.id)).desc(), col(Source.origin))
        )
        return [(source, int(total)) for source, total in self.session.exec(statement).all()]


__all__ = ["SourceRepository"]
