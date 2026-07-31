"""`analyze-market`: who buys, whose budget it comes from, and how big it could be.

A fan-out rather than a synthesis: one `MarketAnalysis` per `Opportunity`. Each reply
entry names the opportunity it sizes, so `assemble` binds it to that subject instead
of inferring the pairing from order — a reply that reordered or dropped an entry
would otherwise attach the wrong market case to the wrong idea, silently.

Sizing reuses the artifact's own `SizeEstimate` and `PricingEstimate` rather than
restating them as reply-only drafts. That is the point of the structural guard: an
amount is optional but `basis` never is, and an amount without a currency is invalid,
so "this cannot be sized yet, and here is why" is expressible while a bare confident
number is not. Sharing the type means the reply is checked by the same rule the
artifact is, and no second copy can drift away from it.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactRef,
    EvidenceLevel,
    MarketAnalysis,
    PricingEstimate,
    SizeEstimate,
)
from app.skills.base import Batching, Skill, SkillInput, SkillOutput, SkillRequest, register
from app.utils.errors import SkillError


class AnalyzeMarketInput(SkillInput):
    """Placeholder values for `analyze-market.md`."""

    question: dict[str, Any] = Field(
        description="The Question artifact seeding the run, serialised as JSON.",
    )
    opportunities: list[dict[str, Any]] = Field(
        description="Every Opportunity artifact to analyse, serialised as JSON.",
    )


class MarketAnalysisDraft(BaseModel):
    """The market case for exactly one opportunity, keyed by that opportunity's id."""

    model_config = ConfigDict(extra="forbid")

    opportunity_id: str = Field(
        description="Id of the supplied opportunity this analyses, copied exactly.",
    )
    buyer: str = Field(description="Who holds the problem and would pay to fix it.")
    budget_owner: str = Field(description="Whose budget it comes out of. Often not the buyer.")
    pricing: PricingEstimate | None = Field(
        default=None,
        description="How it would be charged for. Null when the shape is not yet arguable.",
    )
    # SAM and SOM only. TAM is the figure that gets invented most, so there is no field
    # for it — and either of these may be a basis with no amount.
    sam: SizeEstimate | None = Field(
        default=None,
        description="Serviceable addressable market, or the reason it cannot be sized.",
    )
    som: SizeEstimate | None = Field(
        default=None,
        description="Serviceable obtainable market, or the reason it cannot be sized.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="What must hold for these figures to mean anything.",
    )
    unknowns: list[str] = Field(
        default_factory=list,
        description="What could not be established. An empty list here is rarely honest.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Trust in this market case. Null = not assessed.",
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.NONE,
        description="How strongly the case is grounded in observed reality.",
    )


class AnalyzeMarketOutput(SkillOutput):
    """One market case per supplied opportunity, no more and no fewer."""

    analyses: list[MarketAnalysisDraft] = Field(
        min_length=1,
        description="Exactly one entry for each opportunity supplied, keyed by its id.",
    )


@register
class AnalyzeMarketSkill(Skill):
    """Name the buyer and size the market behind each candidate opportunity."""

    name: ClassVar[str] = "analyze-market"
    description: ClassVar[str] = "Make the market case for each candidate opportunity."
    prompt_name: ClassVar[str] = "analyze-market"
    consumes: ClassVar[tuple[ArtifactKind, ...]] = (ArtifactKind.OPPORTUNITY,)
    produces: ClassVar[ArtifactKind] = ArtifactKind.MARKET_ANALYSIS
    batching: ClassVar[Batching] = Batching.PER_ITEM
    primary_kind: ClassVar[ArtifactKind | None] = ArtifactKind.OPPORTUNITY
    input_schema: ClassVar[type[SkillInput]] = AnalyzeMarketInput
    output_schema: ClassVar[type[SkillOutput]] = AnalyzeMarketOutput

    def gather(self, request: SkillRequest) -> AnalyzeMarketInput:
        """Hand the prompt the question and every opportunity supplied."""
        if request.question is None:
            raise SkillError(f"{self.name} needs the run's question artifact, none supplied")

        opportunities = request.of_kind(ArtifactKind.OPPORTUNITY)
        if not opportunities:
            raise SkillError(f"{self.name} needs at least one opportunity, none supplied")

        return AnalyzeMarketInput(
            question=request.question.model_dump(mode="json"),
            opportunities=[artifact.model_dump(mode="json") for artifact in opportunities],
        )

    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        """Bind each analysis to the opportunity it names, one apiece.

        Raises:
            SkillError: If an analysis names an opportunity that was not supplied, names
                one twice, or if a supplied opportunity was left unanalysed. Each breaks
                the one-analysis-per-opportunity contract the decision stage relies on.
        """
        if not isinstance(output, AnalyzeMarketOutput):
            raise SkillError(
                f"{self.name} expected {AnalyzeMarketOutput.__name__} from the model, "
                f"got {type(output).__name__}"
            )

        expected = {artifact.id for artifact in request.of_kind(ArtifactKind.OPPORTUNITY)}
        analysed: set[str] = set()
        artifacts: list[Artifact] = []

        for draft in output.analyses:
            if draft.opportunity_id not in expected:
                known = ", ".join(sorted(expected)) or "<none>"
                raise SkillError(
                    f"{self.name} analysed opportunity {draft.opportunity_id!r}, which was "
                    f"not supplied; had: {known}"
                )
            if draft.opportunity_id in analysed:
                raise SkillError(
                    f"{self.name} returned two analyses for opportunity {draft.opportunity_id!r}"
                )
            analysed.add(draft.opportunity_id)

            reference = ArtifactRef(kind=ArtifactKind.OPPORTUNITY, id=draft.opportunity_id)
            artifacts.append(
                MarketAnalysis(
                    id=MarketAnalysis.make_id(),
                    run_id=request.run_id,
                    opportunity=reference,
                    # Narrowed here rather than left to `execute`, which would name every
                    # sibling opportunity as a parent of every analysis.
                    parents=[reference],
                    buyer=draft.buyer,
                    budget_owner=draft.budget_owner,
                    pricing=draft.pricing,
                    sam=draft.sam,
                    som=draft.som,
                    assumptions=draft.assumptions,
                    unknowns=draft.unknowns,
                    confidence=draft.confidence,
                    evidence_level=draft.evidence_level,
                ),
            )

        if missing := sorted(expected - analysed):
            raise SkillError(f"{self.name} left opportunities unanalysed: {', '.join(missing)}")
        return artifacts


__all__ = [
    "AnalyzeMarketInput",
    "AnalyzeMarketOutput",
    "AnalyzeMarketSkill",
    "MarketAnalysisDraft",
]
