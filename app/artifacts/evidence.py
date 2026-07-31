"""Evidence: one piece of raw collected signal, captured verbatim."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import ClassVar

from pydantic import Field, HttpUrl, field_validator

from app.artifacts.base import Artifact, ArtifactKind
from app.utils.time import utcnow


class EvidenceKind(StrEnum):
    """The shape of the observation, which decides how much weight to give it.

    A support ticket and a forum post can say the same words and mean different
    things — one came from a paying customer with a broken workflow, the other from
    an anonymous account — so the shape is recorded rather than inferred later.
    """

    FORUM_POST = "forum_post"
    COMMENT = "comment"
    REVIEW = "review"
    SUPPORT_TICKET = "support_ticket"
    ISSUE = "issue"
    ARTICLE = "article"
    SOCIAL_POST = "social_post"
    JOB_POSTING = "job_posting"
    DATASET_ROW = "dataset_row"
    SURVEY_RESPONSE = "survey_response"
    INTERVIEW = "interview"
    OTHER = "other"


class Evidence(Artifact):
    """A single raw observation from a collector, stored before any interpretation.

    Evidence is the only artifact that enters the pipeline from outside, so it is kept
    as close to the source as possible: later stages may be re-run against it.

    Its `parents` hold the question the collection run was serving; the source itself
    is *not* a parent, because a URL is not an artifact. Everything needed to go back
    to that source and check the excerpt lives in the provenance fields below.
    """

    kind: ClassVar[ArtifactKind] = ArtifactKind.EVIDENCE
    id_prefix: ClassVar[str] = "ev"

    collector: str = Field(
        description="Name of the collector that produced this, e.g. `reddit`. Also the replay key.",
    )
    evidence_kind: EvidenceKind = Field(
        default=EvidenceKind.OTHER,
        description=(
            "Shape of the observation. Named `evidence_kind` because `kind` is the artifact kind."
        ),
    )

    excerpt: str = Field(
        description="The observation quoted verbatim. Never paraphrased — briefs do that.",
    )
    title: str | None = None
    author: str | None = Field(
        default=None,
        description="Handle or name as the source published it. Not resolved to a real identity.",
    )

    source_url: HttpUrl | None = Field(
        default=None,
        description="Where a human can go to read the excerpt in context.",
    )
    source_id: str | None = Field(
        default=None,
        description="The source's own identifier for this item, for re-fetching when it moves.",
    )

    published_at: datetime | None = Field(
        default=None,
        description="When the source published it, if it says. Drives recency, unlike capture.",
    )
    captured_at: datetime = Field(
        default_factory=utcnow,
        description="When the collector fetched it — the age of our copy, not of the signal.",
    )

    content_hash: str | None = Field(
        default=None,
        description=(
            "Digest of the normalised excerpt. Two collectors finding the same post must "
            "agree on this value, so it is what dedup compares."
        ),
    )
    language: str | None = Field(
        default=None,
        description="BCP-47 tag of the excerpt, e.g. `en` or `pt-BR`. None when undetected.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Collector-assigned labels, e.g. the subreddit or product. Lowercased.",
    )

    @field_validator("collector", "excerpt")
    @classmethod
    def _require_text(cls, value: str) -> str:
        """Reject empty ground truth: evidence with no content or no origin is unusable."""
        text = value.strip()
        if not text:
            raise ValueError("value must not be empty")
        return text

    @field_validator("title", "author", "source_id", "content_hash", "language")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        """Treat whitespace-only strings as absent, so "unknown" has one representation."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("published_at", "captured_at")
    @classmethod
    def _as_utc(cls, value: datetime | None) -> datetime | None:
        """Pin timestamps to UTC so evidence from different sources stays comparable.

        A naive value is read as UTC rather than rejected: many sources emit bare
        timestamps, and dropping an observation over a missing offset costs more than
        assuming the convention the rest of the app already writes.
        """
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("tags")
    @classmethod
    def _normalise_tags(cls, value: list[str]) -> list[str]:
        """Fold case, drop blanks, and de-duplicate while keeping the collector's order."""
        cleaned = (tag.strip().lower() for tag in value)
        return list(dict.fromkeys(tag for tag in cleaned if tag))


__all__ = ["Evidence", "EvidenceKind"]
