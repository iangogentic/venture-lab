"""`sources` — the distinct origins evidence actually came from.

Derived, not configured. Which collectors run and what they are pointed at is
settings (`.env`), and duplicating that here would give the same fact two homes
that drift apart. What settings cannot tell you is what any of it *yielded*: this
table is written by projecting collected evidence, so a row exists only once
something was actually found there.

That makes "which of my thirteen Discourse forums have ever produced a single
piece of evidence?" answerable, which is the question that decides what to keep
configured.
"""

from datetime import datetime

from sqlalchemy import UniqueConstraint
from sqlmodel import Field

from app.models.base import TimestampMixin, UTCDateTime


class Source(TimestampMixin, table=True):
    """One origin — a collector paired with the host or site it reached."""

    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("collector", "origin", name="uq_source_origin"),)

    id: int | None = Field(default=None, primary_key=True)
    collector: str = Field(index=True, description="Registered collector name, e.g. 'rss'.")
    origin: str = Field(
        index=True,
        description="Host the evidence came from, e.g. 'news.ycombinator.com'. Falls back "
        "to the collector's own name when an item carries no URL.",
    )

    first_seen_at: datetime | None = Field(default=None, sa_type=UTCDateTime)
    last_seen_at: datetime | None = Field(default=None, sa_type=UTCDateTime)


__all__ = ["Source"]
