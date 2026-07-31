"""`opportunities` — the index over synthesized opportunity artifacts.

The artifact body lives at `workspace/opportunities/<artifact_id>.json`; this row
exists so opportunities can be queried, ranked and joined to their decisions
without rescanning every workspace file — "every build verdict across every run"
is one statement here and a full directory walk otherwise.

The verdict columns are denormalised from the Decision that ruled on this
opportunity. They are a copy for querying, and the Decision artifact remains the
record: if the two disagree, the artifact is right and the index is stale.
"""

from datetime import datetime

from sqlmodel import Field

from app.models.base import TimestampMixin, UTCDateTime


class OpportunityRecord(TimestampMixin, table=True):
    """Ledger row pointing at one opportunity artifact, with its verdict."""

    __tablename__ = "opportunities"

    id: int | None = Field(default=None, primary_key=True)
    artifact_id: str = Field(index=True, unique=True)
    run_id: str = Field(foreign_key="runs.id", index=True)

    title: str = Field(index=True)
    icp: str | None = Field(default=None, description="Ideal customer profile, for scanning.")
    cluster_id: str | None = Field(
        default=None,
        index=True,
        description="Artifact id of the pain cluster this was inferred from.",
    )
    status: str = Field(index=True, description="Artifact status of the opportunity itself.")
    confidence: float | None = Field(default=None)

    verdict: str | None = Field(
        default=None,
        index=True,
        description="build / reject / wait, copied from the Decision. None means undecided.",
    )
    decision_id: str | None = Field(default=None, index=True)
    decision_confidence: float | None = Field(
        default=None,
        description="How much the Decision itself is trusted — not the opportunity's own "
        "`confidence`, which is about the inference rather than the ruling on it.",
    )
    decided_at: datetime | None = Field(default=None, sa_type=UTCDateTime)


__all__ = ["OpportunityRecord"]
