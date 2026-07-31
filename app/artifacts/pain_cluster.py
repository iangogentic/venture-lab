"""Pain cluster: a problem that recurred across briefs, and how hard it bites."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Final

from pydantic import Field, field_validator

from app.artifacts.base import Artifact, ArtifactKind


class PainSeverity(StrEnum):
    """What the pain costs whoever has it. Ordinal, mildest first."""

    LOW = "low"
    """An irritation. People live with it and are not looking for a fix."""

    MEDIUM = "medium"
    """Costs real time or money, but a workaround exists and is being used."""

    HIGH = "high"
    """Blocks work often enough that people are actively shopping for a fix."""

    CRITICAL = "critical"
    """Stops the job outright, or costs enough that doing nothing is not an option."""

    @property
    def rank(self) -> int:
        """Position on the scale, so clusters can be sorted and thresholded."""
        return _SEVERITY_RANKS[self]


_SEVERITY_RANKS: Final[Mapping[PainSeverity, int]] = MappingProxyType(
    {
        PainSeverity.LOW: 0,
        PainSeverity.MEDIUM: 1,
        PainSeverity.HIGH: 2,
        PainSeverity.CRITICAL: 3,
    },
)


def _stripped(values: list[str]) -> list[str]:
    """Strip each entry and drop the blanks, keeping repeats."""
    return [stripped for value in values if (stripped := value.strip())]


def _unique_stripped(values: list[str]) -> list[str]:
    """Strip, drop blanks, and keep the first spelling of any case-insensitive repeat."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in _stripped(values):
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned


class PainCluster(Artifact):
    """A recurring problem drawn from several briefs — what an Opportunity will answer.

    The briefs it was built from are in `parents`. A cluster is where repetition becomes
    signal: the same complaint from one source is an anecdote, so `source_count` and
    `severity` are the fields that decide whether anything downstream should act on it.
    """

    kind: ClassVar[ArtifactKind] = ArtifactKind.PAIN_CLUSTER
    id_prefix: ClassVar[str] = "pc"

    label: str = Field(
        description="Short name for the pain, in the sufferers' words where they had one.",
    )
    description: str | None = Field(
        default=None,
        description="The fuller statement of the problem: who hits it, when, and at what cost.",
    )
    severity: PainSeverity | None = Field(
        default=None,
        description="How hard it bites. None = not yet assessed, as with `confidence`.",
    )
    source_count: int = Field(
        default=0,
        ge=0,
        description="Distinct sources exhibiting it; one source repeating itself counts once.",
    )
    segments: list[str] = Field(
        default_factory=list,
        description="Who has this pain, e.g. 'solo founders', 'ops teams at Series B'.",
    )
    # Repeats are kept: two sources landing on the same words is itself the finding.
    quotes: list[str] = Field(
        default_factory=list,
        description="Verbatim complaints, left unedited so they can be quoted back.",
    )
    prevalence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Share of the observed population showing it, when measured. None = unknown.",
    )

    @field_validator("label")
    @classmethod
    def _require_label_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("label must not be blank")
        return stripped

    @field_validator("description")
    @classmethod
    def _normalise_description(cls, value: str | None) -> str | None:
        """Collapse a whitespace-only description to None so "absent" has one spelling."""
        if value is None:
            return None
        return value.strip() or None

    @field_validator("segments")
    @classmethod
    def _normalise_segments(cls, value: list[str]) -> list[str]:
        return _unique_stripped(value)

    @field_validator("quotes")
    @classmethod
    def _normalise_quotes(cls, value: list[str]) -> list[str]:
        return _stripped(value)


__all__ = [
    "PainCluster",
    "PainSeverity",
]
