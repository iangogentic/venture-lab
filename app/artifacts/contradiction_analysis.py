"""`ContradictionAnalysis` — the case against an opportunity.

A deliberate adversarial pass. Every prior stage builds the argument *for*; left
alone, a pipeline of summarisers converges on a confident story. This stage exists
to go looking for disconfirming evidence, and it returns **only evidence** — no
verdict, no recommendation, no weighing. Judging is the decision stage's job, and
mixing the two here would let the search quietly stop once it had enough to
conclude.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.artifacts.base import Artifact, ArtifactKind, ArtifactRef


class CounterEvidenceKind(StrEnum):
    """The shape of a piece of disconfirming evidence."""

    FAILED_STARTUP = "failed_startup"
    """Someone tried this and did not make it."""

    NEGATIVE_REVIEW = "negative_review"
    """Users of an existing answer say it does not work."""

    ABANDONMENT = "abandonment"
    """People adopted something for this and stopped."""

    INCUMBENT_STRENGTH = "incumbent_strength"
    """An existing player is well placed to absorb this."""

    MARKET_RISK = "market_risk"
    """A structural condition working against the opportunity."""


class CounterSeverity(StrEnum):
    """How much weight a piece of counter-evidence should carry."""

    MINOR = "minor"
    MATERIAL = "material"
    BLOCKING = "blocking"

    @property
    def rank(self) -> int:
        """Position on the scale, so counter-evidence can be sorted and thresholded."""
        return _SEVERITY_RANKS[self]


_SEVERITY_RANKS: Final[Mapping[CounterSeverity, int]] = MappingProxyType(
    {
        CounterSeverity.MINOR: 0,
        CounterSeverity.MATERIAL: 1,
        CounterSeverity.BLOCKING: 2,
    },
)


class CounterEvidence(BaseModel):
    """One observation that argues against the opportunity."""

    model_config = ConfigDict(extra="forbid")

    kind: CounterEvidenceKind
    observation: str = Field(description="What was found, stated as an observation.")
    severity: CounterSeverity = CounterSeverity.MATERIAL
    source: str | None = Field(
        default=None,
        description="Where it came from. Absent means it could not be attributed.",
    )
    sources: list[ArtifactRef] = Field(
        default_factory=list,
        description="Any workspace artifacts backing it.",
    )

    @field_validator("observation")
    @classmethod
    def _require_observation(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("observation must not be blank")
        return cleaned


class ContradictionAnalysis(Artifact):
    """Everything found that argues against one opportunity."""

    kind: ClassVar[ArtifactKind] = ArtifactKind.CONTRADICTION_ANALYSIS
    id_prefix: ClassVar[str] = "cx"

    opportunity: ArtifactRef = Field(
        description="The opportunity this argues against.",
    )
    counter_evidence: list[CounterEvidence] = Field(
        default_factory=list,
        description="Findings only. An empty list means nothing was found, which is "
        "itself a result worth recording.",
    )
    searched_for: list[str] = Field(
        default_factory=list,
        description="What was looked for. Makes an empty result interpretable rather "
        "than indistinguishable from not having looked.",
    )

    @field_validator("searched_for")
    @classmethod
    def _clean_entries(cls, value: list[str]) -> list[str]:
        return [line.strip() for line in value if line.strip()]

    @field_validator("opportunity")
    @classmethod
    def _must_reference_an_opportunity(cls, value: ArtifactRef) -> ArtifactRef:
        if value.kind is not ArtifactKind.OPPORTUNITY:
            raise ValueError(f"opportunity must reference an opportunity, got {value.kind.value}")
        return value

    @property
    def is_blocking(self) -> bool:
        """Whether anything found is severe enough to stop a decision."""
        return any(item.severity is CounterSeverity.BLOCKING for item in self.counter_evidence)

    def by_kind(self, kind: CounterEvidenceKind) -> list[CounterEvidence]:
        """Counter-evidence of one shape."""
        return [item for item in self.counter_evidence if item.kind is kind]


__all__ = [
    "ContradictionAnalysis",
    "CounterEvidence",
    "CounterEvidenceKind",
    "CounterSeverity",
]
