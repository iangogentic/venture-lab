"""`evidence` — the index over collected evidence artifacts.

Holds what you would want to filter or count by; the excerpt, the tags and the
rest of the body stay in `workspace/evidence/<artifact_id>.json`. Duplicating the
excerpt here would create a second copy of the one thing the project promises is
verbatim, and two copies of a quote is one too many.
"""

from datetime import datetime

from sqlmodel import Field

from app.models.base import TimestampMixin, UTCDateTime


class EvidenceRecord(TimestampMixin, table=True):
    """Ledger row pointing at one evidence artifact."""

    __tablename__ = "evidence"

    id: int | None = Field(default=None, primary_key=True)
    artifact_id: str = Field(index=True, unique=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    source_id: int | None = Field(
        default=None,
        foreign_key="sources.id",
        index=True,
        description="The origin this came from — a row in `sources`, not the id at "
        "the source itself. That one is `external_id`.",
    )
    dedup_key: str | None = Field(
        default=None,
        index=True,
        description="`collector:external_id` — the exact-match key collection dedups on, "
        "kept so the same item can be recognised across runs. Near-duplicates are a "
        "different question, answered by the semantic memory in `workspace/memory.db`.",
    )

    collector: str = Field(index=True)
    external_id: str | None = Field(default=None, description="The item's id at its source.")
    evidence_kind: str | None = Field(default=None, index=True)
    title: str | None = Field(default=None)
    url: str | None = Field(default=None, index=True)
    author: str | None = Field(default=None, index=True)
    content_hash: str | None = Field(default=None, index=True)
    status: str = Field(index=True, description="Artifact status — superseded rows stay indexed.")

    published_at: datetime | None = Field(default=None, sa_type=UTCDateTime)
    captured_at: datetime | None = Field(default=None, sa_type=UTCDateTime)


__all__ = ["EvidenceRecord"]
