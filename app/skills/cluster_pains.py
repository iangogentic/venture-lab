"""`cluster-pains`: fold research briefs into the recurring problems underneath them.

A cluster is where repetition turns into signal, which is why `source_count` and
`severity` are asked for alongside the label: the same complaint from a single source
is an anecdote, and nothing downstream should act on it as though it were more.

This stage describes problems and nothing else. There is no field for a solution, a
product, a market size, a price or a segment's spend — not because the prompt asks
for restraint, but because the schema gives such an answer nowhere to go and
`extra="forbid"` rejects the reply that invents one. That matters more here than
anywhere: `discover-opportunities` reads these clusters, and it can only argue
honestly from a problem that has not already been quietly solved for it.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import Artifact, ArtifactKind, EvidenceLevel, PainCluster, PainSeverity
from app.skills.base import Skill, SkillInput, SkillOutput, SkillRequest, register
from app.utils.errors import SkillError


class ClusterPainInput(SkillInput):
    """Placeholder values for `cluster-pains.md`."""

    question: dict[str, Any] = Field(
        description="The Question artifact seeding the run, serialised as JSON.",
    )
    briefs: list[dict[str, Any]] = Field(
        description="Every ResearchBrief artifact to cluster, serialised as JSON.",
    )


class PainClusterDraft(BaseModel):
    """One recurring problem the model found across the briefs.

    Problem-side fields only. Anything a builder would want to say in reply belongs to
    a later stage.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="Short name for the pain, in the sufferers' words.")
    description: str | None = Field(
        default=None,
        description="Who hits this problem, when, and at what cost.",
    )
    severity: PainSeverity | None = Field(
        default=None,
        description="What the pain costs whoever has it. Null = not assessed.",
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
    quotes: list[str] = Field(
        default_factory=list,
        description="Verbatim complaints, left unedited so they can be quoted back.",
    )
    prevalence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Share of the observed population showing it. Omit when not measured.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Trust in this cluster. Null = not assessed.",
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.NONE,
        description="How strongly the cluster is grounded in observed reality.",
    )


class ClusterPainOutput(SkillOutput):
    """The pain clusters the model drew out of the briefs."""

    clusters: list[PainClusterDraft] = Field(
        min_length=1,
        description="One entry per distinct recurring problem, most acute first.",
    )


@register
class ClusterPainSkill(Skill):
    """Group briefs into the recurring problems an opportunity could later answer."""

    name: ClassVar[str] = "cluster-pains"
    description: ClassVar[str] = "Fold research briefs into recurring, evidenced pain clusters."
    prompt_name: ClassVar[str] = "cluster-pains"
    consumes: ClassVar[tuple[ArtifactKind, ...]] = (ArtifactKind.RESEARCH_BRIEF,)
    produces: ClassVar[ArtifactKind] = ArtifactKind.PAIN_CLUSTER
    input_schema: ClassVar[type[SkillInput]] = ClusterPainInput
    output_schema: ClassVar[type[SkillOutput]] = ClusterPainOutput

    def gather(self, request: SkillRequest) -> ClusterPainInput:
        """Hand the prompt the question and every brief supplied."""
        if request.question is None:
            raise SkillError(f"{self.name} needs the run's question artifact, none supplied")

        briefs = request.of_kind(ArtifactKind.RESEARCH_BRIEF)
        if not briefs:
            raise SkillError(f"{self.name} needs at least one research brief, none supplied")

        return ClusterPainInput(
            question=request.question.model_dump(mode="json"),
            briefs=[artifact.model_dump(mode="json") for artifact in briefs],
        )

    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        """Give each recurring problem its own artifact."""
        if not isinstance(output, ClusterPainOutput):
            raise SkillError(
                f"{self.name} expected {ClusterPainOutput.__name__} from the model, "
                f"got {type(output).__name__}"
            )

        return [
            PainCluster(
                id=PainCluster.make_id(),
                run_id=request.run_id,
                label=draft.label,
                description=draft.description,
                severity=draft.severity,
                source_count=draft.source_count,
                segments=draft.segments,
                quotes=draft.quotes,
                prevalence=draft.prevalence,
                confidence=draft.confidence,
                evidence_level=draft.evidence_level,
            )
            for draft in output.clusters
        ]


__all__ = [
    "ClusterPainInput",
    "ClusterPainOutput",
    "ClusterPainSkill",
    "PainClusterDraft",
]
