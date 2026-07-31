"""`MarketAnalysis` — buyer, pricing, and size, with the arithmetic exposed.

Built to make fabrication awkward. A `SizeEstimate` cannot carry a number without
also carrying the `basis` it was derived from, and it is perfectly valid to hold
no number at all — "we cannot size this yet" is a legitimate, expressible answer.
SAM and SOM only: TAM is the number that gets invented most.
"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.artifacts.base import Artifact, ArtifactKind, ArtifactRef


class SizeEstimate(BaseModel):
    """A market size, or an explicit statement that it cannot be sized.

    `basis` is mandatory either way: with an amount it is the derivation, without
    one it is why the estimate could not be made.
    """

    model_config = ConfigDict(extra="forbid")

    amount: float | None = Field(default=None, ge=0.0, description="None when unknown.")
    currency: str | None = Field(default=None, description="ISO code, e.g. USD.")
    period: str | None = Field(default=None, description="e.g. 'annual recurring'.")
    basis: str = Field(description="How this was derived, or why it could not be.")

    @field_validator("basis")
    @classmethod
    def _require_basis(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("basis must not be blank — an unexplained number is a fabrication")
        return cleaned

    @model_validator(mode="after")
    def _amount_needs_a_currency(self) -> "SizeEstimate":
        if self.amount is not None and not self.currency:
            raise ValueError("an amount must state its currency")
        return self

    @property
    def is_quantified(self) -> bool:
        """Whether this carries an actual number."""
        return self.amount is not None


class PricingEstimate(BaseModel):
    """How the thing would be charged for."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="e.g. per-seat, usage-based, flat platform fee.")
    amount: float | None = Field(default=None, ge=0.0)
    currency: str | None = None
    unit: str | None = Field(default=None, description="e.g. 'per seat per month'.")
    basis: str = Field(description="What the figure is anchored to, or why it is unknown.")

    @field_validator("model", "basis")
    @classmethod
    def _require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class MarketAnalysis(Artifact):
    """The market case for one opportunity."""

    kind: ClassVar[ArtifactKind] = ArtifactKind.MARKET_ANALYSIS
    id_prefix: ClassVar[str] = "mk"

    opportunity: ArtifactRef = Field(
        description="The opportunity this sizes. An explicit link, not merely provenance.",
    )
    buyer: str = Field(description="Who holds the problem and would pay to fix it.")
    budget_owner: str = Field(description="Whose budget it comes out of. Often not the buyer.")
    pricing: PricingEstimate | None = None
    sam: SizeEstimate | None = Field(default=None, description="Serviceable addressable market.")
    som: SizeEstimate | None = Field(default=None, description="Serviceable obtainable market.")
    assumptions: list[str] = Field(
        default_factory=list,
        description="What must hold for these figures to mean anything.",
    )
    unknowns: list[str] = Field(
        default_factory=list,
        description="What could not be established. An empty list here is rarely honest.",
    )

    @field_validator("buyer", "budget_owner")
    @classmethod
    def _require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("assumptions", "unknowns")
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
    def is_sized(self) -> bool:
        """Whether either figure carries a real number."""
        return any(est is not None and est.is_quantified for est in (self.sam, self.som))


__all__ = ["MarketAnalysis", "PricingEstimate", "SizeEstimate"]
