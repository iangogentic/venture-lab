"""`contradiction-analysis`: the case against each opportunity, gathered deliberately.

One `ContradictionAnalysis` per `Opportunity`, with that opportunity's market and
competition readings in view. Every stage upstream builds the argument *for*, and a
chain of summarisers converges on a confident story; this stage is the one asked to
go looking for what argues against it.

It returns **only evidence**. There is deliberately no verdict, score, ranking or
recommendation anywhere in the output schema: a search allowed to conclude stops as
soon as it has enough to conclude, and weighing is the decision stage's job.
`searched_for` is required for the mirror-image reason — it is what makes an empty
result mean "looked and found nothing" rather than "did not look".
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import Artifact, ArtifactKind, ArtifactRef, EvidenceLevel
from app.artifacts.contradiction_analysis import (
    ContradictionAnalysis,
    CounterEvidence,
    CounterEvidenceKind,
    CounterSeverity,
)
from app.skills.base import Batching, Skill, SkillInput, SkillOutput, SkillRequest, register
from app.utils.errors import SkillError


class CounterEvidenceRead(BaseModel):
    """One finding that argues against the opportunity."""

    model_config = ConfigDict(extra="forbid")

    kind: CounterEvidenceKind = Field(
        description="What shape of disconfirming evidence this is.",
    )
    observation: str = Field(
        description="What was found, stated as an observation rather than a conclusion.",
    )
    severity: CounterSeverity = Field(
        default=CounterSeverity.MATERIAL,
        description="How much weight this finding carries: minor, material, or blocking.",
    )
    source: str | None = Field(
        default=None,
        description="Where it came from. Null when it cannot be attributed — better than "
        "attributing it to something that does not say it.",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="Ids of supplied artifacts backing this finding, copied verbatim.",
    )


class ContradictionRead(BaseModel):
    """The adversarial pass over exactly one opportunity."""

    model_config = ConfigDict(extra="forbid")

    opportunity_id: str = Field(
        description="Id of the opportunity argued against, copied verbatim from the input.",
    )
    searched_for: list[str] = Field(
        min_length=1,
        description="What was looked for. Required: it is what makes an empty result "
        "readable as a search rather than a shrug.",
    )
    counter_evidence: list[CounterEvidenceRead] = Field(
        default_factory=list,
        description="Findings only. An empty list is a legitimate answer; manufactured "
        "doubt is worse than none.",
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.NONE,
        description="How strongly these findings are grounded in the supplied artifacts.",
    )


class ContradictionInput(SkillInput):
    """Placeholder payload for the `contradiction-analysis` prompt."""

    question: dict[str, Any]
    opportunities: list[dict[str, Any]]
    market: list[dict[str, Any]]
    competition: list[dict[str, Any]]


class ContradictionOutput(SkillOutput):
    """One adversarial pass per opportunity, carrying findings and nothing else."""

    analyses: list[ContradictionRead] = Field(
        min_length=1,
        description="Exactly one entry per opportunity supplied, keyed by its id.",
    )


@register
class ContradictionSkill(Skill):
    """Search for evidence against each opportunity, before a verdict is taken on it."""

    name: ClassVar[str] = "contradiction-analysis"
    description: ClassVar[str] = "Search for evidence against each opportunity."
    prompt_name: ClassVar[str] = "contradiction-analysis"
    consumes: ClassVar[tuple[ArtifactKind, ...]] = (
        ArtifactKind.OPPORTUNITY,
        ArtifactKind.MARKET_ANALYSIS,
        ArtifactKind.COMPETITION_ANALYSIS,
    )
    produces: ClassVar[ArtifactKind] = ArtifactKind.CONTRADICTION_ANALYSIS
    batching: ClassVar[Batching] = Batching.PER_ITEM
    primary_kind: ClassVar[ArtifactKind | None] = ArtifactKind.OPPORTUNITY
    input_schema: ClassVar[type[SkillInput]] = ContradictionInput
    output_schema: ClassVar[type[SkillOutput]] = ContradictionOutput

    def gather(self, request: SkillRequest) -> ContradictionInput:
        """Put each opportunity in front of the model alongside the case built for it."""
        if request.question is None:
            raise SkillError(f"{self.name} needs the run's question artifact, none supplied")

        opportunities = request.of_kind(ArtifactKind.OPPORTUNITY)
        if not opportunities:
            raise SkillError(f"{self.name} needs at least one opportunity, none supplied")

        market = request.of_kind(ArtifactKind.MARKET_ANALYSIS)
        if not market:
            raise SkillError(f"{self.name} needs at least one market analysis, none supplied")

        competition = request.of_kind(ArtifactKind.COMPETITION_ANALYSIS)
        if not competition:
            raise SkillError(f"{self.name} needs at least one competition analysis, none supplied")

        return ContradictionInput(
            question=request.question.model_dump(mode="json"),
            opportunities=[artifact.model_dump(mode="json") for artifact in opportunities],
            market=[artifact.model_dump(mode="json") for artifact in market],
            competition=[artifact.model_dump(mode="json") for artifact in competition],
        )

    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        """Bind each pass to the opportunity it names, resolving every cited source."""
        if not isinstance(output, ContradictionOutput):
            raise SkillError(
                f"{self.name} expected {ContradictionOutput.__name__}, got {type(output).__name__}"
            )

        citable = self._citable(request)

        analyses: list[Artifact] = [
            ContradictionAnalysis(
                id=ContradictionAnalysis.make_id(),
                run_id=request.run_id,
                opportunity=ArtifactRef(kind=ArtifactKind.OPPORTUNITY, id=subject.id),
                counter_evidence=[
                    CounterEvidence(
                        kind=finding.kind,
                        observation=finding.observation,
                        severity=finding.severity,
                        source=finding.source,
                        sources=self._sources(finding.source_ids, citable),
                    )
                    for finding in entry.counter_evidence
                ],
                searched_for=entry.searched_for,
                evidence_level=entry.evidence_level,
                # This pass argues against one opportunity, so provenance is narrowed
                # here; left to `execute` it would name every sibling opportunity and
                # every other opportunity's market and competition reading.
                parents=[subject.ref],
            )
            for entry, subject in self._paired(output, request)
        ]
        return analyses

    def _paired(
        self,
        output: ContradictionOutput,
        request: SkillRequest,
    ) -> list[tuple[ContradictionRead, Artifact]]:
        """Match every pass to its opportunity, and insist the pairing is exact.

        An opportunity that quietly went unchallenged is the failure this stage
        exists to prevent: downstream it is indistinguishable from one that was
        challenged and survived, which is precisely the confidence the pipeline is
        arranged not to manufacture.
        """
        subjects = {artifact.id: artifact for artifact in request.of_kind(ArtifactKind.OPPORTUNITY)}
        known = ", ".join(sorted(subjects)) or "<none>"

        paired: list[tuple[ContradictionRead, Artifact]] = []
        seen: set[str] = set()
        for entry in output.analyses:
            subject = subjects.get(entry.opportunity_id)
            if subject is None:
                raise SkillError(
                    f"{self.name} argued against opportunity {entry.opportunity_id!r}, which "
                    f"was not supplied; had: {known}"
                )
            if entry.opportunity_id in seen:
                raise SkillError(
                    f"{self.name} returned two analyses for opportunity {entry.opportunity_id!r}"
                )
            seen.add(entry.opportunity_id)
            paired.append((entry, subject))

        if unhandled := sorted(set(subjects) - seen):
            raise SkillError(
                f"{self.name} challenged {len(seen)} of {len(subjects)} opportunities; "
                f"unchallenged: {', '.join(unhandled)}"
            )
        return paired

    def _citable(self, request: SkillRequest) -> dict[str, ArtifactRef]:
        """References to every artifact the prompt actually showed the model."""
        shown = [
            *request.of_kind(ArtifactKind.OPPORTUNITY),
            *request.of_kind(ArtifactKind.MARKET_ANALYSIS),
            *request.of_kind(ArtifactKind.COMPETITION_ANALYSIS),
        ]
        return {artifact.id: artifact.ref for artifact in shown}

    def _sources(self, source_ids: list[str], citable: dict[str, ArtifactRef]) -> list[ArtifactRef]:
        """Turn cited ids into references, rejecting any that were never supplied.

        A citation to an artifact the model was not given is a fabricated source —
        exactly the failure this stage exists to catch, so it is not tolerated in
        the stage's own output.
        """
        resolved: list[ArtifactRef] = []
        for source_id in source_ids:
            ref = citable.get(source_id)
            if ref is None:
                known = ", ".join(sorted(citable)) or "<none>"
                raise SkillError(
                    f"{self.name} cited artifact {source_id!r}, which was not supplied; "
                    f"had: {known}"
                )
            resolved.append(ref)
        return resolved


__all__ = [
    "ContradictionInput",
    "ContradictionOutput",
    "ContradictionRead",
    "ContradictionSkill",
    "CounterEvidenceRead",
]
