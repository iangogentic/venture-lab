"""`decision`: build, reject or wait on each opportunity, with the whole case in view.

One `Decision` per `Opportunity`, keyed by opportunity id so a verdict cannot drift
onto the wrong subject. The market and competition readings are supplied so the call
is made against the case that was actually built, and the contradiction pass is
supplied alongside them deliberately: a verdict reached without the disconfirming
evidence in front of it is the failure the whole pipeline is arranged to prevent.

This is the only stage that weighs. Every earlier one is forbidden from netting its
findings against anything, which is what leaves the trade-off visible here instead of
already resolved somewhere upstream where nobody can re-judge it.

`next_validation_step` and `biggest_unknown` are demanded for every verdict, not only
the hesitant ones. A confident BUILD that cannot name what it is still unsure about
has stopped reasoning, and that is exactly when a decision needs checking.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import Artifact, ArtifactKind, ArtifactRef, EvidenceLevel
from app.artifacts.decision import Decision, Verdict
from app.skills.base import Batching, Skill, SkillInput, SkillOutput, SkillRequest, register
from app.utils.errors import SkillError


class DecisionRead(BaseModel):
    """The verdict on exactly one opportunity."""

    model_config = ConfigDict(extra="forbid")

    opportunity_id: str = Field(
        description="Id of the opportunity ruled on, copied verbatim from the input.",
    )
    verdict: Verdict = Field(
        description="build, reject or wait. Nothing softer is available on purpose.",
    )
    decision_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How sure this call is, 0 to 1. Grades the judgement, not the evidence.",
    )
    top_reasons: list[str] = Field(
        min_length=1,
        description="Why this verdict, strongest first. A decision with no stated reason "
        "is not a decision.",
    )
    next_validation_step: str = Field(
        description="The single cheapest next action that would move the confidence.",
    )
    biggest_unknown: str = Field(
        description="What is still most likely to be wrong — required even for a confident build.",
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.NONE,
        description="How strongly this verdict is grounded in the supplied artifacts.",
    )


class DecisionInput(SkillInput):
    """Placeholder payload for the `decision` prompt."""

    question: dict[str, Any]
    opportunities: list[dict[str, Any]]
    market: list[dict[str, Any]]
    competition: list[dict[str, Any]]
    contradictions: list[dict[str, Any]]


class DecisionOutput(SkillOutput):
    """One verdict per opportunity."""

    decisions: list[DecisionRead] = Field(
        min_length=1,
        description="Exactly one entry per opportunity supplied, keyed by its id.",
    )


@register
class DecisionSkill(Skill):
    """Rule on each opportunity, with the evidence against it weighed in the open."""

    name: ClassVar[str] = "decision"
    description: ClassVar[str] = "Choose build, reject or wait for each opportunity."
    prompt_name: ClassVar[str] = "decision"
    consumes: ClassVar[tuple[ArtifactKind, ...]] = (
        ArtifactKind.OPPORTUNITY,
        ArtifactKind.MARKET_ANALYSIS,
        ArtifactKind.COMPETITION_ANALYSIS,
        ArtifactKind.CONTRADICTION_ANALYSIS,
    )
    produces: ClassVar[ArtifactKind] = ArtifactKind.DECISION
    batching: ClassVar[Batching] = Batching.PER_ITEM
    primary_kind: ClassVar[ArtifactKind | None] = ArtifactKind.OPPORTUNITY
    input_schema: ClassVar[type[SkillInput]] = DecisionInput
    output_schema: ClassVar[type[SkillOutput]] = DecisionOutput

    def gather(self, request: SkillRequest) -> DecisionInput:
        """Show the model what is on the table, the case for it, and everything against it."""
        if request.question is None:
            raise SkillError(f"{self.name} needs the run's question artifact, none supplied")

        opportunities = request.of_kind(ArtifactKind.OPPORTUNITY)
        if not opportunities:
            raise SkillError(f"{self.name} needs at least one opportunity, none supplied")

        # A missing market or competition analysis is an error, not an empty list. Both
        # stages run for every opportunity before this one, so absence means a broken
        # lineage rather than a finding — and an empty list would reach the model as one.
        # The distinction decides verdicts: an analysis that ran and could not size the
        # market is missing information, while an analysis that never ran is unexamined,
        # and a verdict taken on the second while reading it as the first is precisely the
        # manufactured confidence this pipeline is arranged to prevent.
        market = request.of_kind(ArtifactKind.MARKET_ANALYSIS)
        if not market:
            raise SkillError(f"{self.name} needs at least one market analysis, none supplied")

        competition = request.of_kind(ArtifactKind.COMPETITION_ANALYSIS)
        if not competition:
            raise SkillError(f"{self.name} needs at least one competition analysis, none supplied")

        contradictions = request.of_kind(ArtifactKind.CONTRADICTION_ANALYSIS)
        if not contradictions:
            raise SkillError(
                f"{self.name} needs at least one contradiction analysis, none supplied"
            )

        return DecisionInput(
            question=request.question.model_dump(mode="json"),
            opportunities=[artifact.model_dump(mode="json") for artifact in opportunities],
            market=[artifact.model_dump(mode="json") for artifact in market],
            competition=[artifact.model_dump(mode="json") for artifact in competition],
            contradictions=[artifact.model_dump(mode="json") for artifact in contradictions],
        )

    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        """Bind each verdict to the opportunity it names, refusing an inexact set."""
        if not isinstance(output, DecisionOutput):
            raise SkillError(
                f"{self.name} expected {DecisionOutput.__name__}, got {type(output).__name__}"
            )

        decisions: list[Artifact] = [
            Decision(
                id=Decision.make_id(),
                run_id=request.run_id,
                opportunity=ArtifactRef(kind=ArtifactKind.OPPORTUNITY, id=subject.id),
                verdict=entry.verdict,
                decision_confidence=entry.decision_confidence,
                top_reasons=entry.top_reasons,
                next_validation_step=entry.next_validation_step,
                biggest_unknown=entry.biggest_unknown,
                evidence_level=entry.evidence_level,
                # No envelope `confidence`: it grades the record while
                # `decision_confidence` grades the call, and copying one into the
                # other would collapse a distinction the artifact exists to keep.
                # A verdict rules on one opportunity, so provenance is narrowed here;
                # left to `execute` it would name every sibling opportunity and every
                # analysis of one. The market, competition and contradiction passes all
                # reached the same subject and stay joinable through it.
                parents=[subject.ref],
            )
            for entry, subject in self._paired(output, request)
        ]
        return decisions

    def _paired(
        self,
        output: DecisionOutput,
        request: SkillRequest,
    ) -> list[tuple[DecisionRead, Artifact]]:
        """Match every verdict to its opportunity, and insist the pairing is exact.

        An unruled opportunity is not a neutral omission: it reads downstream as
        nothing having been proposed, which is the one outcome nobody chose.
        """
        subjects = {artifact.id: artifact for artifact in request.of_kind(ArtifactKind.OPPORTUNITY)}
        known = ", ".join(sorted(subjects)) or "<none>"

        paired: list[tuple[DecisionRead, Artifact]] = []
        seen: set[str] = set()
        for entry in output.decisions:
            subject = subjects.get(entry.opportunity_id)
            if subject is None:
                raise SkillError(
                    f"{self.name} ruled on opportunity {entry.opportunity_id!r}, which was "
                    f"not supplied; had: {known}"
                )
            if entry.opportunity_id in seen:
                raise SkillError(
                    f"{self.name} returned two verdicts for opportunity {entry.opportunity_id!r}"
                )
            seen.add(entry.opportunity_id)
            paired.append((entry, subject))

        if unhandled := sorted(set(subjects) - seen):
            raise SkillError(
                f"{self.name} ruled on {len(seen)} of {len(subjects)} opportunities; "
                f"undecided: {', '.join(unhandled)}"
            )
        return paired


__all__ = [
    "DecisionInput",
    "DecisionOutput",
    "DecisionRead",
    "DecisionSkill",
]
