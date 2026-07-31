"""Lead: a real person who publicly expressed a pain, recorded with full provenance.

A lead is a person you can find again saying a thing they really said. That is
the whole definition, and it is why `author`, `url` and `quote` are required and
non-blank: without a handle, a permalink a human can open, and the verbatim
passage, what remains is a statistic — still evidence, but not a lead.

The ethics contract, stated here because the model enforces part of it and the
CLI the rest: this ledger stores only what the person posted publicly alongside
their complaint. It must never be enriched with scraped contact data, auto-DMs
are off the table — the permalink is the only channel, and engaging happens as a
public, human-written reply — and platform deletion obligations mean a lead whose
source vanishes gets `EXCLUDED`, not retained (Reddit's terms require honoring
deletions).
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import ClassVar

from pydantic import Field, field_validator

from app.artifacts.base import Artifact, ArtifactKind, ArtifactRef


class LeadIntent(StrEnum):
    """What the person's own words express about the pain.

    Classified by the model, but always from a quote the code has verified — so
    the reading can be checked against the exact words it was read from.
    """

    SEEKING = "seeking"
    """Actively asking for a solution — "what do people use for…?"."""

    COMPLAINING = "complaining"
    """Expressing the pain unprompted — venting, describing the cost, unasked."""

    MENTIONING = "mentioning"
    """The topic is named but no pain is expressed."""


class LeadEngagement(StrEnum):
    """Where a human's outreach with this lead stands. Updated by `op leads mark`."""

    NEW = "new"
    """Harvested, not yet looked at by a person."""

    REVIEWED = "reviewed"
    """A human read it and judged it worth keeping in play."""

    ENGAGED = "engaged"
    """A human replied publicly — the only kind of first contact this ledger allows."""

    CONVERTED = "converted"
    """The person joined a waitlist or agreed to an interview."""

    EXCLUDED = "excluded"
    """The content was deleted upstream, or the person opted out.

    Deletion upstream *obliges* this state: keeping a copy of what someone chose
    to remove is exactly what the ethics contract above forbids.
    """


class Lead(Artifact):
    """One person, one pain, one permalink — the unit of the outreach ledger.

    Every identity field (`collector`, `author`, `url`, `external_id`,
    `published_at`) is copied from the Evidence artifact the lead came from, never
    from a model reply; the `quote` is model-selected but code-verified to be a
    literal passage of that evidence's excerpt. `parents` carry both the pain
    cluster and the evidence item, so a lead can be walked back to the complaint
    it grew out of.
    """

    kind: ClassVar[ArtifactKind] = ArtifactKind.LEAD
    id_prefix: ClassVar[str] = "ld"

    cluster: ArtifactRef = Field(
        description="The pain cluster this person's complaint belongs to.",
    )
    evidence: ArtifactRef = Field(
        description="The evidence item the lead came from — where the quote was verified.",
    )
    collector: str = Field(
        description="The platform the evidence was collected from, e.g. `reddit`.",
    )
    author: str = Field(
        description="Handle or name as the source published it. Not resolved to a real identity.",
    )
    url: str = Field(
        description="The permalink a human can open to read the post in context.",
    )
    external_id: str | None = Field(
        default=None,
        description="The source's own identifier for the item, for checking it still exists.",
    )
    published_at: datetime | None = Field(
        default=None,
        description="When the person posted it, if the source says. Drives newest-first sorting.",
    )
    quote: str = Field(
        description="The verbatim passage showing the pain. Model-selected, code-verified.",
    )
    intent: LeadIntent = Field(
        description="What the quoted words express: seeking, complaining, or mentioning.",
    )
    intent_rationale: str | None = Field(
        default=None,
        description="One line on why the intent was read that way.",
    )
    engagement: LeadEngagement = Field(
        default=LeadEngagement.NEW,
        description="Where outreach stands. The one field a human updates over time.",
    )

    @field_validator("collector", "author", "url", "quote")
    @classmethod
    def _require_text(cls, value: str) -> str:
        """No person, no permalink, no quote — no lead. Blank identity is not a lead."""
        text = value.strip()
        if not text:
            raise ValueError("value must not be empty")
        return text

    @field_validator("external_id", "intent_rationale")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        """Treat whitespace-only strings as absent, so "unknown" has one representation."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("published_at")
    @classmethod
    def _as_utc(cls, value: datetime | None) -> datetime | None:
        """Pin to UTC, reading naive values as UTC — the convention Evidence already writes."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("cluster")
    @classmethod
    def _must_reference_a_cluster(cls, value: ArtifactRef) -> ArtifactRef:
        if value.kind is not ArtifactKind.PAIN_CLUSTER:
            raise ValueError(f"cluster must reference a pain cluster, got {value.kind.value}")
        return value

    @field_validator("evidence")
    @classmethod
    def _must_reference_evidence(cls, value: ArtifactRef) -> ArtifactRef:
        if value.kind is not ArtifactKind.EVIDENCE:
            raise ValueError(f"evidence must reference evidence, got {value.kind.value}")
        return value


__all__ = ["Lead", "LeadEngagement", "LeadIntent"]
