"""`research-brief`: what the collected Evidence says, and how much it is worth.

A brief is what every later stage reads *instead of* the raw evidence, so it has two
jobs, and the schema here holds it to both. It must say what the evidence adds up to
— signals, quotes, contradictions, unknowns — and it must report the strength of the
pile it read, which is why `evidence_quality`, `evidence_density` and `source_count`
are asked for as findings rather than left as metadata.

Citations are carried as evidence ids rather than as prose. `parents` only records
that the brief read everything it was handed, which loses exactly the link a reader
auditing one claim needs: *which* excerpt carries *this* statement. `assemble`
resolves those ids and refuses any it was not supplied — a citation that cannot be
walked back is worse than no citation, because it reads as rigour.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactRef,
    ClaimType,
    Contradiction,
    EvidenceDensity,
    EvidenceLevel,
    Quote,
    ResearchBrief,
    Signal,
)
from app.skills.base import Skill, SkillInput, SkillOutput, SkillRequest, register
from app.utils.errors import SkillError


class ResearchBriefInput(SkillInput):
    """Placeholder values for `research-brief.md`."""

    question: dict[str, Any] = Field(
        description="The Question artifact seeding the run, serialised as JSON.",
    )
    evidence: list[dict[str, Any]] = Field(
        description="Every Evidence artifact the brief may draw on, serialised as JSON.",
    )


class SignalDraft(BaseModel):
    """One pattern the evidence shows, with the ids that carry it."""

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(description="A single pattern the brief will stand behind.")
    claim_type: ClaimType = Field(
        default=ClaimType.OBSERVATION,
        description="Whether this is attested, reasoned from what is attested, or proposed.",
    )
    supported_by: list[str] = Field(
        default_factory=list,
        description="Ids of the supplied evidence carrying this statement specifically.",
    )


class QuoteDraft(BaseModel):
    """A verbatim excerpt, kept attributable to the evidence it came from."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The excerpt in the source's own words, unedited.")
    source: str | None = Field(
        default=None,
        description="Id of the supplied evidence this was taken from.",
    )
    speaker: str | None = Field(
        default=None,
        description="Who said it, as the source published them. Null when anonymous.",
    )


class ContradictionDraft(BaseModel):
    """A disagreement in the evidence, recorded rather than resolved."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(description="What the sources disagree about, not what they said.")
    # Two entries is the floor the schema can check; that they are actually *distinct*
    # is enforced by the Contradiction artifact, so restating one position twice cannot
    # be dressed up as a conflict.
    positions: list[str] = Field(
        min_length=2,
        description="The conflicting accounts, kept apart. One position is not a conflict.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Ids of the supplied evidence on either side of the disagreement.",
    )


class BriefDraft(BaseModel):
    """One brief: the answer to a single slice of the question, plus its own strength."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, description="Short handle for listings.")
    summary: str = Field(description="What the evidence says, in a paragraph.")
    signals: list[SignalDraft] = Field(
        default_factory=list,
        description="The separately checkable patterns the summary rests on.",
    )
    quotes: list[QuoteDraft] = Field(
        default_factory=list,
        description="Excerpts a reader can check the synthesis against.",
    )
    contradictions: list[ContradictionDraft] = Field(
        default_factory=list,
        description="Disagreements found in the evidence and left standing.",
    )
    unknowns: list[str] = Field(
        default_factory=list,
        description="What the evidence does not settle. An empty list is a strong claim.",
    )
    evidence_quality: EvidenceLevel = Field(
        default=EvidenceLevel.NONE,
        description="Grade of the evidence underneath this brief.",
    )
    evidence_density: EvidenceDensity = Field(
        default=EvidenceDensity.SPARSE,
        description="How much evidence stands behind the claims, independent of its grade.",
    )
    source_count: int = Field(
        default=0,
        ge=0,
        description="Distinct sources behind the brief; one source quoted twice counts once.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Trust in this brief. Not the same as the grade of its evidence.",
    )


class ResearchBriefOutput(SkillOutput):
    """The briefs the model wrote from the supplied evidence."""

    briefs: list[BriefDraft] = Field(
        min_length=1,
        description="One brief per slice of the question the evidence can speak to.",
    )


@register
class ResearchBriefSkill(Skill):
    """Synthesise evidence into briefs that stand on their own once the raw text is gone."""

    name: ClassVar[str] = "research-brief"
    description: ClassVar[str] = "Synthesise collected evidence into cited research briefs."
    prompt_name: ClassVar[str] = "research-brief"
    consumes: ClassVar[tuple[ArtifactKind, ...]] = (ArtifactKind.EVIDENCE,)
    produces: ClassVar[ArtifactKind] = ArtifactKind.RESEARCH_BRIEF
    input_schema: ClassVar[type[SkillInput]] = ResearchBriefInput
    output_schema: ClassVar[type[SkillOutput]] = ResearchBriefOutput

    def gather(self, request: SkillRequest) -> ResearchBriefInput:
        """Hand the prompt the question and every piece of evidence supplied."""
        if request.question is None:
            raise SkillError(f"{self.name} needs the run's question artifact, none supplied")

        evidence = request.of_kind(ArtifactKind.EVIDENCE)
        if not evidence:
            raise SkillError(f"{self.name} needs at least one evidence artifact, none supplied")

        return ResearchBriefInput(
            question=request.question.model_dump(mode="json"),
            evidence=[artifact.model_dump(mode="json") for artifact in evidence],
        )

    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        """Turn each draft into a brief, resolving its citations to real evidence.

        Raises:
            SkillError: If the reply cites an evidence id this skill was not given.
        """
        if not isinstance(output, ResearchBriefOutput):
            raise SkillError(
                f"{self.name} expected {ResearchBriefOutput.__name__} from the model, "
                f"got {type(output).__name__}"
            )

        known = frozenset(artifact.id for artifact in request.of_kind(ArtifactKind.EVIDENCE))
        return [
            ResearchBrief(
                id=ResearchBrief.make_id(),
                run_id=request.run_id,
                title=draft.title,
                summary=draft.summary,
                signals=[
                    Signal(
                        statement=signal.statement,
                        claim_type=signal.claim_type,
                        supported_by=self._refs(signal.supported_by, known),
                    )
                    for signal in draft.signals
                ],
                quotes=[
                    Quote(
                        text=quote.text,
                        source=self._ref(quote.source, known),
                        speaker=quote.speaker,
                    )
                    for quote in draft.quotes
                ],
                contradictions=[
                    Contradiction(
                        topic=contradiction.topic,
                        positions=contradiction.positions,
                        sources=self._refs(contradiction.sources, known),
                    )
                    for contradiction in draft.contradictions
                ],
                unknowns=draft.unknowns,
                evidence_quality=draft.evidence_quality,
                evidence_density=draft.evidence_density,
                source_count=draft.source_count,
                confidence=draft.confidence,
                # One judgement of evidential strength, written to both the domain field
                # and the envelope. Asking for them separately would let a brief grade
                # its own evidence twice and disagree with itself.
                evidence_level=draft.evidence_quality,
            )
            for draft in output.briefs
        ]

    # --------------------------------------------------------------- internals

    def _refs(self, cited: list[str], known: frozenset[str]) -> list[ArtifactRef]:
        """Resolve cited evidence ids, refusing any the skill was not handed.

        An id that was never supplied cannot be walked back to a source, which is the
        one guarantee the artifact chain exists to provide.
        """
        if unknown := sorted(set(cited) - known):
            raise SkillError(f"{self.name} cited evidence it was not given: {', '.join(unknown)}")
        return [ArtifactRef(kind=ArtifactKind.EVIDENCE, id=cited_id) for cited_id in cited]

    def _ref(self, cited: str | None, known: frozenset[str]) -> ArtifactRef | None:
        """Resolve one optional citation. An unattributed quote is allowed; a false one is not."""
        if cited is None:
            return None
        return self._refs([cited], known)[0]


__all__ = [
    "BriefDraft",
    "ContradictionDraft",
    "QuoteDraft",
    "ResearchBriefInput",
    "ResearchBriefOutput",
    "ResearchBriefSkill",
    "SignalDraft",
]
