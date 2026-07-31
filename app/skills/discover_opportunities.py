"""`discover-opportunities`: infer, from clustered pain, where a business could sit.

This is the first stage that reasons past what the evidence literally says, so what
it is asked for is deliberately narrow: the broken workflow, who lives in it, who
would pay, the problem, why now — and `missing_evidence`, naming what would have to
be true and is not yet known. That last field is what stops an inference hardening
into a claim, and `claim_type` keeps the distinction on the artifact itself.

Two things are absent by construction. There is no competitor, incumbent or
substitute field: judging who already serves this is `analyze-competition`'s work,
and a competitive read produced here would be a guess made without the stage that
exists to check it. There is no sizing, pricing or score either — inventing a number
now corrupts the market analysis that would otherwise have to derive one.

Each opportunity names the one cluster it answers, in `pain_cluster`, and that id is
checked against the clusters actually supplied — an opportunity citing a cluster
nobody handed over is an idea the model brought with it, not a finding of this run.
That link is kept separate from `parents`, which `execute` fills with every cluster
this stage read: the mapping is many-to-many, so "what I was given" and "what this
answers" are different facts and are recorded separately.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactRef,
    ClaimType,
    EvidenceLevel,
    Opportunity,
)
from app.skills.base import Skill, SkillInput, SkillOutput, SkillRequest, register
from app.utils.errors import SkillError


class DiscoverOpportunityInput(SkillInput):
    """Placeholder values for `discover-opportunities.md`."""

    question: dict[str, Any] = Field(
        description="The Question artifact seeding the run, serialised as JSON.",
    )
    clusters: list[dict[str, Any]] = Field(
        description="Every PainCluster artifact to answer, serialised as JSON.",
    )


class OpportunityDraft(BaseModel):
    """One candidate opportunity inferred from the clusters."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: str = Field(
        description="Id of the pain cluster this answers. Must be one that was supplied.",
    )
    title: str = Field(description="Short handle for the opportunity.")
    workflow: str = Field(description="The workflow that is broken today.")
    icp: str = Field(description="Ideal customer profile — who lives in that workflow.")
    buyer: str = Field(description="Who would actually purchase. Often not the user.")
    problem: str = Field(description="The problem in that workflow, stated plainly.")
    why_now: str = Field(description="What changed that makes this solvable or urgent now.")
    missing_evidence: list[str] = Field(
        default_factory=list,
        description="What must be checked before trusting this. An empty list is suspicious.",
    )
    claim_type: ClaimType = Field(
        default=ClaimType.INFERENCE,
        description="Inference when reasoned from the clusters; hypothesis when weaker.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Trust in this write-up. Null = not assessed.",
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.NONE,
        description="How strongly this rests on the clustered evidence rather than on reasoning.",
    )


class DiscoverOpportunityOutput(SkillOutput):
    """The opportunities the model inferred from the clusters."""

    opportunities: list[OpportunityDraft] = Field(
        min_length=1,
        description="One entry per candidate. Not every cluster deserves one.",
    )


@register
class DiscoverOpportunitySkill(Skill):
    """Turn evidenced pain into candidate opportunities worth analysing."""

    name: ClassVar[str] = "discover-opportunities"
    description: ClassVar[str] = "Infer candidate opportunities from the clustered pains."
    prompt_name: ClassVar[str] = "discover-opportunities"
    consumes: ClassVar[tuple[ArtifactKind, ...]] = (ArtifactKind.PAIN_CLUSTER,)
    produces: ClassVar[ArtifactKind] = ArtifactKind.OPPORTUNITY
    input_schema: ClassVar[type[SkillInput]] = DiscoverOpportunityInput
    output_schema: ClassVar[type[SkillOutput]] = DiscoverOpportunityOutput

    def gather(self, request: SkillRequest) -> DiscoverOpportunityInput:
        """Hand the prompt the question and every pain cluster supplied."""
        if request.question is None:
            raise SkillError(f"{self.name} needs the run's question artifact, none supplied")

        clusters = request.of_kind(ArtifactKind.PAIN_CLUSTER)
        if not clusters:
            raise SkillError(f"{self.name} needs at least one pain cluster, none supplied")

        return DiscoverOpportunityInput(
            question=request.question.model_dump(mode="json"),
            clusters=[artifact.model_dump(mode="json") for artifact in clusters],
        )

    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        """Give each candidate its own artifact, so each can be analysed and decided alone."""
        if not isinstance(output, DiscoverOpportunityOutput):
            raise SkillError(
                f"{self.name} expected {DiscoverOpportunityOutput.__name__} from the model, "
                f"got {type(output).__name__}"
            )

        # An opportunity that cites a cluster nobody supplied is an idea the model
        # brought with it, not something this run's evidence produced.
        supplied = {artifact.id for artifact in request.of_kind(ArtifactKind.PAIN_CLUSTER)}
        for draft in output.opportunities:
            if draft.cluster_id not in supplied:
                raise SkillError(
                    f"{self.name} returned an opportunity citing unknown pain cluster "
                    f"{draft.cluster_id!r}; supplied: {', '.join(sorted(supplied)) or '<none>'}"
                )

        return [
            Opportunity(
                id=Opportunity.make_id(),
                run_id=request.run_id,
                pain_cluster=ArtifactRef(kind=ArtifactKind.PAIN_CLUSTER, id=draft.cluster_id),
                title=draft.title,
                workflow=draft.workflow,
                icp=draft.icp,
                buyer=draft.buyer,
                problem=draft.problem,
                why_now=draft.why_now,
                missing_evidence=draft.missing_evidence,
                claim_type=draft.claim_type,
                confidence=draft.confidence,
                evidence_level=draft.evidence_level,
            )
            for draft in output.opportunities
        ]


__all__ = [
    "DiscoverOpportunityInput",
    "DiscoverOpportunityOutput",
    "DiscoverOpportunitySkill",
    "OpportunityDraft",
]
