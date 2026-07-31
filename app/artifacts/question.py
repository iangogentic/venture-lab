"""Question: the seed enquiry a run exists to answer."""

from enum import StrEnum
from typing import ClassVar

from pydantic import Field, field_validator

from app.artifacts.base import Artifact, ArtifactKind


class QuestionPriority(StrEnum):
    """How much of the run's budget and attention this question deserves.

    Ordering is by urgency, not by confidence or evidence: a `CRITICAL` question is
    one we cannot ship without answering, however little we currently know about it.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Question(Artifact):
    """One thing the operator wants to find out, and the root of a provenance tree.

    A question is the only artifact authored by a human rather than derived from
    another artifact, so `parents` is normally empty. Every downstream artifact walks
    its `parents` back to exactly one of these, which is what makes a report auditable:
    any claim can be traced to the question that provoked the search for it.
    """

    kind: ClassVar[ArtifactKind] = ArtifactKind.QUESTION
    id_prefix: ClassVar[str] = "q"

    text: str = Field(
        description="The enquiry itself, phrased so that evidence could settle it.",
    )
    rationale: str | None = Field(
        default=None,
        description="Why this is worth asking now — the decision it is meant to inform.",
    )
    scope: str | None = Field(
        default=None,
        description=(
            "Boundary on what counts as an answer: markets, timeframes, or segments "
            "that are in or out. Read by collectors to decide what to skip."
        ),
    )
    priority: QuestionPriority = QuestionPriority.MEDIUM
    search_queries: list[str] = Field(
        default_factory=list,
        description="Search terms to retrieve with. A research question makes a poor "
        "keyword query — sources match on words, not intent — so these are derived "
        "from it, and can be edited when the derivation is wrong.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form labels for grouping questions across runs. Normalised lowercase.",
    )

    @field_validator("text")
    @classmethod
    def _require_text(cls, value: str) -> str:
        """Reject a blank question: an empty root would orphan everything beneath it."""
        text = value.strip()
        if not text:
            raise ValueError("question text must not be empty")
        return text

    @field_validator("rationale", "scope")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        """Treat whitespace-only prose as absent, so "unset" has one representation."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("tags")
    @classmethod
    def _normalise_tags(cls, value: list[str]) -> list[str]:
        """Fold case, drop blanks, and de-duplicate while keeping the author's order."""
        cleaned = (tag.strip().lower() for tag in value)
        return list(dict.fromkeys(tag for tag in cleaned if tag))


__all__ = ["Question", "QuestionPriority"]
