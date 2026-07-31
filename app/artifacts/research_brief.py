"""`ResearchBrief` — what the evidence says, and how much it is worth.

Deliberately reports on its own evidence base as well as its findings. A brief
that reads confidently off two anecdotes is the failure mode this stage exists to
prevent, so `evidence_quality` and `evidence_density` are first-class outputs, not
metadata.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.artifacts.base import Artifact, ArtifactKind, ArtifactRef, ClaimType, EvidenceLevel


class EvidenceDensity(StrEnum):
    """How much evidence stands behind the brief's claims."""

    SPARSE = "sparse"
    """A handful of sources, or many saying the same thing from one place."""

    MODERATE = "moderate"
    """Enough to see a pattern, not enough to size it."""

    DENSE = "dense"
    """Many independent sources; the pattern is well attested."""

    @property
    def rank(self) -> int:
        """Position on the scale, so density can be compared and thresholded."""
        return _DENSITY_RANKS[self]


_DENSITY_RANKS: Final[Mapping[EvidenceDensity, int]] = MappingProxyType(
    {
        EvidenceDensity.SPARSE: 0,
        EvidenceDensity.MODERATE: 1,
        EvidenceDensity.DENSE: 2,
    },
)


class Signal(BaseModel):
    """One pattern the evidence shows, tagged with how it was arrived at."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    claim_type: ClaimType = ClaimType.OBSERVATION
    supported_by: list[ArtifactRef] = Field(
        default_factory=list,
        description="The evidence carrying this signal — narrower than the brief's parents.",
    )

    @field_validator("statement")
    @classmethod
    def _require_statement(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("signal statement must not be blank")
        return cleaned


class Quote(BaseModel):
    """A verbatim excerpt, kept attributable.

    Quotes are the one place a downstream reader can check the pipeline's work,
    so a quote that cannot be traced back to its evidence is worse than no quote.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    source: ArtifactRef | None = Field(
        default=None,
        description="The evidence artifact this was taken from.",
    )
    speaker: str | None = None

    @field_validator("text")
    @classmethod
    def _require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("quote text must not be blank")
        return cleaned


class Contradiction(BaseModel):
    """Two or more sources that do not agree."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    positions: list[str] = Field(min_length=2)
    sources: list[ArtifactRef] = Field(default_factory=list)

    @field_validator("positions")
    @classmethod
    def _require_two_distinct(cls, value: list[str]) -> list[str]:
        cleaned = [line.strip() for line in value if line.strip()]
        if len({line.casefold() for line in cleaned}) < 2:
            raise ValueError("a contradiction needs at least two distinct positions")
        return cleaned


class ResearchBrief(Artifact):
    """A synthesis over evidence, reported with its own evidential strength."""

    kind: ClassVar[ArtifactKind] = ArtifactKind.RESEARCH_BRIEF
    id_prefix: ClassVar[str] = "rb"

    title: str | None = None
    summary: str = Field(description="What the evidence says, in a paragraph.")
    signals: list[Signal] = Field(default_factory=list)
    quotes: list[Quote] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    unknowns: list[str] = Field(
        default_factory=list,
        description="What the evidence does not settle. An empty list is a strong claim.",
    )
    evidence_quality: EvidenceLevel = Field(
        default=EvidenceLevel.NONE,
        description="Grade of the underlying evidence; mirrored onto `evidence_level`.",
    )
    evidence_density: EvidenceDensity = Field(default=EvidenceDensity.SPARSE)
    source_count: int = Field(default=0, ge=0, description="Distinct sources behind the brief.")

    @field_validator("summary")
    @classmethod
    def _require_summary(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("summary must not be blank")
        return cleaned

    @field_validator("title")
    @classmethod
    def _blank_title_is_absent(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("unknowns")
    @classmethod
    def _clean_entries(cls, value: list[str]) -> list[str]:
        return [line.strip() for line in value if line.strip()]


__all__ = [
    "Contradiction",
    "EvidenceDensity",
    "Quote",
    "ResearchBrief",
    "Signal",
]
