"""Shared building blocks for table models."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Dialect, TypeDecorator
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp. Used as the default for audit columns."""
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """A `DateTime` column that stores UTC and hands back tz-aware UTC.

    SQLite has no timezone type, so a plain `DateTime` silently drops the offset:
    an aware value goes in and a naive one comes back, and every later comparison
    against `utcnow()` raises `TypeError`. Artifacts are timezone-aware
    throughout, so a ledger that is not would make the two halves of the system
    uncomparable at exactly the moment you want to line them up.

    Normalising on the way in and re-stamping on the way out fixes that for any
    dialect: naive UTC is what actually reaches the database, so a Postgres
    `timestamp without time zone` behaves the same as SQLite's text.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        """Convert to UTC and drop the offset. A naive value is assumed to be UTC."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        """Re-stamp the UTC the column was written with."""
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class TimestampMixin(SQLModel):
    """Adds `created_at` / `updated_at` audit columns.

    `updated_at` is maintained by `touch()` rather than an ORM event listener:
    every write in this package goes through a repository, so there is exactly
    one place that has to remember, and a listener would hide it.
    """

    created_at: datetime = Field(default_factory=utcnow, sa_type=UTCDateTime, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, sa_type=UTCDateTime, nullable=False)

    def touch(self) -> None:
        """Mark the row as modified now."""
        self.updated_at = utcnow()


__all__ = ["TimestampMixin", "UTCDateTime", "utcnow"]
