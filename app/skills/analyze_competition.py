"""`analyze-competition`: who already serves each opportunity, and where the gap is.

One `CompetitionAnalysis` per `Opportunity`. Every entry is keyed by the id of the
opportunity it was given, so `assemble` binds a reading to its subject instead of
trusting the order the reply came back in. Three ways that goes wrong silently — an
id nobody supplied, one opportunity read twice, one never read at all — are all
refused here rather than persisted as a plausible-looking competitive picture
attached to the wrong thing.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import Artifact, ArtifactKind, ArtifactRef, EvidenceLevel
from app.artifacts.competition_analysis import CompetitionAnalysis, Competitor
from app.skills.base import Batching, Skill, SkillInput, SkillOutput, SkillRequest, register
from app.utils.errors import SkillError


class CompetitorRead(BaseModel):
    """One incumbent, substitute or established practice the model identified."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The product, vendor or practice being compared against.")
    positioning: str | None = Field(
        default=None,
        description="How it presents itself to this segment, in its own terms.",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="What it is genuinely good at, judged from the user's point of view.",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Where it leaves this segment badly served.",
    )


class CompetitionRead(BaseModel):
    """The competitive picture for exactly one opportunity."""

    model_config = ConfigDict(extra="forbid")

    opportunity_id: str = Field(
        description="Id of the opportunity analysed, copied verbatim from the input.",
    )
    competitors: list[CompetitorRead] = Field(default_factory=list)
    substitutes: list[str] = Field(
        default_factory=list,
        description="Non-product workarounds in use today — a spreadsheet, a script, "
        "a contractor, doing nothing. Usually the real competition.",
    )
    switching_costs: list[str] = Field(
        default_factory=list,
        description="What it actually costs this segment to leave what they use now.",
    )
    moats: list[str] = Field(
        default_factory=list,
        description="What protects the incumbents. 'None identified' is a real answer.",
    )
    differentiation: list[str] = Field(
        default_factory=list,
        description="Where a new entrant could be genuinely different, not merely nicer.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How much this reading of the market is trusted. Null = not assessed.",
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.NONE,
        description="How strongly the competitive claims are grounded in the inputs.",
    )


class AnalyzeCompetitionInput(SkillInput):
    """Placeholder payload for the `analyze-competition` prompt."""

    question: dict[str, Any]
    opportunities: list[dict[str, Any]]


class AnalyzeCompetitionOutput(SkillOutput):
    """One competitive reading per opportunity."""

    analyses: list[CompetitionRead] = Field(
        min_length=1,
        description="Exactly one entry per opportunity supplied, keyed by its id.",
    )


@register
class AnalyzeCompetitionSkill(Skill):
    """Map who already serves each opportunity, and what would set a new entrant apart."""

    name: ClassVar[str] = "analyze-competition"
    description: ClassVar[str] = "Analyse the competitive picture for each opportunity."
    prompt_name: ClassVar[str] = "analyze-competition"
    consumes: ClassVar[tuple[ArtifactKind, ...]] = (ArtifactKind.OPPORTUNITY,)
    produces: ClassVar[ArtifactKind] = ArtifactKind.COMPETITION_ANALYSIS
    batching: ClassVar[Batching] = Batching.PER_ITEM
    primary_kind: ClassVar[ArtifactKind | None] = ArtifactKind.OPPORTUNITY
    input_schema: ClassVar[type[SkillInput]] = AnalyzeCompetitionInput
    output_schema: ClassVar[type[SkillOutput]] = AnalyzeCompetitionOutput

    def gather(self, request: SkillRequest) -> AnalyzeCompetitionInput:
        """Hand the prompt the original question and every candidate opportunity."""
        if request.question is None:
            raise SkillError(f"{self.name} needs the run's question artifact, none supplied")

        opportunities = request.of_kind(ArtifactKind.OPPORTUNITY)
        if not opportunities:
            raise SkillError(f"{self.name} needs at least one opportunity, none supplied")

        return AnalyzeCompetitionInput(
            question=request.question.model_dump(mode="json"),
            opportunities=[artifact.model_dump(mode="json") for artifact in opportunities],
        )

    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        """Bind each reading to the opportunity it names, refusing an inexact set."""
        if not isinstance(output, AnalyzeCompetitionOutput):
            raise SkillError(
                f"{self.name} expected {AnalyzeCompetitionOutput.__name__}, "
                f"got {type(output).__name__}"
            )

        analyses: list[Artifact] = [
            CompetitionAnalysis(
                id=CompetitionAnalysis.make_id(),
                run_id=request.run_id,
                opportunity=ArtifactRef(kind=ArtifactKind.OPPORTUNITY, id=subject.id),
                competitors=[
                    Competitor(
                        name=profile.name,
                        positioning=profile.positioning,
                        strengths=profile.strengths,
                        weaknesses=profile.weaknesses,
                    )
                    for profile in entry.competitors
                ],
                substitutes=entry.substitutes,
                switching_costs=entry.switching_costs,
                moats=entry.moats,
                differentiation=entry.differentiation,
                confidence=entry.confidence,
                evidence_level=entry.evidence_level,
                # One analysis is about one opportunity, so provenance is narrowed
                # here; left to `execute` it would name every sibling opportunity.
                parents=[subject.ref],
            )
            for entry, subject in self._paired(output, request)
        ]
        return analyses

    def _paired(
        self,
        output: AnalyzeCompetitionOutput,
        request: SkillRequest,
    ) -> list[tuple[CompetitionRead, Artifact]]:
        """Match every reading to its opportunity, and insist the pairing is exact.

        A reply is checked for all three ways it can be wrong about *which*
        opportunity it read: an id that was never supplied (a fabricated subject),
        the same id twice (two competing pictures for one opportunity), and an
        opportunity left unanalysed — which downstream would read as "nobody serves
        this segment" when it only means the model skipped it.
        """
        subjects = {artifact.id: artifact for artifact in request.of_kind(ArtifactKind.OPPORTUNITY)}
        known = ", ".join(sorted(subjects)) or "<none>"

        paired: list[tuple[CompetitionRead, Artifact]] = []
        seen: set[str] = set()
        for entry in output.analyses:
            subject = subjects.get(entry.opportunity_id)
            if subject is None:
                raise SkillError(
                    f"{self.name} analysed opportunity {entry.opportunity_id!r}, which was "
                    f"not supplied; had: {known}"
                )
            if entry.opportunity_id in seen:
                raise SkillError(
                    f"{self.name} returned two analyses for opportunity {entry.opportunity_id!r}"
                )
            seen.add(entry.opportunity_id)
            paired.append((entry, subject))

        if unhandled := sorted(set(subjects) - seen):
            raise SkillError(
                f"{self.name} analysed {len(seen)} of {len(subjects)} opportunities; "
                f"unanalysed: {', '.join(unhandled)}"
            )
        return paired


__all__ = [
    "AnalyzeCompetitionInput",
    "AnalyzeCompetitionOutput",
    "AnalyzeCompetitionSkill",
    "CompetitionRead",
    "CompetitorRead",
]
