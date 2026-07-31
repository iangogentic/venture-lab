"""`CompetitionAnalysis` — who already serves this need, and what it costs to leave them."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.artifacts.base import Artifact, ArtifactKind, ArtifactRef


class Competitor(BaseModel):
    """One incumbent or substitute already addressing the pain."""

    model_config = ConfigDict(extra="forbid")

    name: str
    positioning: str | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _require_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("competitor name must not be blank")
        return cleaned

    @field_validator("strengths", "weaknesses")
    @classmethod
    def _clean_entries(cls, value: list[str]) -> list[str]:
        return [line.strip() for line in value if line.strip()]


class CompetitionAnalysis(Artifact):
    """The competitive picture for one opportunity."""

    kind: ClassVar[ArtifactKind] = ArtifactKind.COMPETITION_ANALYSIS
    id_prefix: ClassVar[str] = "cp"

    opportunity: ArtifactRef = Field(
        description="The opportunity this assesses. An explicit link, not merely provenance.",
    )
    competitors: list[Competitor] = Field(default_factory=list)
    substitutes: list[str] = Field(
        default_factory=list,
        description="Non-product workarounds — spreadsheets, scripts, doing nothing. "
        "Usually the real competition.",
    )
    switching_costs: list[str] = Field(
        default_factory=list,
        description="What it actually costs a customer to move. The most common reason "
        "a better product loses.",
    )
    moats: list[str] = Field(default_factory=list, description="What protects the incumbents.")
    differentiation: list[str] = Field(
        default_factory=list,
        description="Where a new entrant could be genuinely different, not merely nicer.",
    )

    @field_validator("substitutes", "switching_costs", "moats", "differentiation")
    @classmethod
    def _clean_entries(cls, value: list[str]) -> list[str]:
        return [line.strip() for line in value if line.strip()]

    @field_validator("opportunity")
    @classmethod
    def _must_reference_an_opportunity(cls, value: ArtifactRef) -> ArtifactRef:
        if value.kind is not ArtifactKind.OPPORTUNITY:
            raise ValueError(f"opportunity must reference an opportunity, got {value.kind.value}")
        return value


__all__ = ["CompetitionAnalysis", "Competitor"]
