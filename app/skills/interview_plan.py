"""`interview-plan`: how to go and test each decision against real people.

One `InterviewPlan` per `Decision`, keyed by decision id. Everything upstream is desk
research; this stage turns a verdict's riskiest assumptions into questions someone can
actually ask, and names — up front — what an answer would have to look like to count.

`failure_criteria` and `questions_to_avoid` carry as much weight as the questions
themselves. Founder interviews rarely fail for asking too few questions; they fail by
asking leading ones and defining success loosely enough that any answer confirms the
plan, which is how a run comes back "validated" having learned nothing.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import Artifact, ArtifactKind, ArtifactRef, EvidenceLevel
from app.artifacts.interview_plan import InterviewPlan, InterviewQuestion, InterviewTarget
from app.skills.base import Batching, Skill, SkillInput, SkillOutput, SkillRequest, register
from app.utils.errors import SkillError


class InterviewTargetRead(BaseModel):
    """A kind of person to talk to, and how to actually reach them."""

    model_config = ConfigDict(extra="forbid")

    persona: str = Field(description="Who they are, specific enough for someone to screen for.")
    where_to_find: list[str] = Field(
        default_factory=list,
        description="Concrete places — communities, job titles, events, existing customers.",
    )
    why_them: str | None = Field(
        default=None,
        description="What this persona can settle that the others cannot.",
    )


class PlannedQuestion(BaseModel):
    """One question, and the assumption it is meant to falsify."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Asked as it would be said out loud, not as a topic.")
    tests_assumption: str | None = Field(
        default=None,
        description="The belief this would falsify. A question testing nothing is small talk.",
    )


class InterviewPlanRead(BaseModel):
    """The validation plan for exactly one decision."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(
        description="Id of the decision this plan tests, copied verbatim from the input.",
    )
    objective: str = Field(description="What these interviews are meant to settle.")
    targets: list[InterviewTargetRead] = Field(
        default_factory=list,
        description="Who to interview, and where they can actually be found.",
    )
    questions: list[PlannedQuestion] = Field(default_factory=list)
    success_criteria: list[str] = Field(
        default_factory=list,
        description="What an answer must look like to count as validation.",
    )
    failure_criteria: list[str] = Field(
        default_factory=list,
        description="What would falsify the thesis. Stated now so the result cannot be "
        "reinterpreted afterwards.",
    )
    questions_to_avoid: list[str] = Field(
        default_factory=list,
        description="Leading or hypothetical questions that would produce agreeable noise.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How much this plan is trusted to settle the objective. Null = not assessed.",
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.NONE,
        description="How strongly the plan's assumptions are grounded in the decision.",
    )


class InterviewPlanInput(SkillInput):
    """Placeholder payload for the `interview-plan` prompt."""

    question: dict[str, Any]
    decisions: list[dict[str, Any]]


class InterviewPlanOutput(SkillOutput):
    """One validation plan per decision."""

    plans: list[InterviewPlanRead] = Field(
        min_length=1,
        description="Exactly one entry per decision supplied, keyed by its id.",
    )


@register
class InterviewPlanSkill(Skill):
    """Turn each decision's riskiest assumptions into questions a real person can answer."""

    name: ClassVar[str] = "interview-plan"
    description: ClassVar[str] = "Plan the interviews that would validate each decision."
    prompt_name: ClassVar[str] = "interview-plan"
    consumes: ClassVar[tuple[ArtifactKind, ...]] = (ArtifactKind.DECISION,)
    produces: ClassVar[ArtifactKind] = ArtifactKind.INTERVIEW_PLAN
    batching: ClassVar[Batching] = Batching.PER_ITEM
    primary_kind: ClassVar[ArtifactKind | None] = ArtifactKind.DECISION
    input_schema: ClassVar[type[SkillInput]] = InterviewPlanInput
    output_schema: ClassVar[type[SkillOutput]] = InterviewPlanOutput

    def gather(self, request: SkillRequest) -> InterviewPlanInput:
        """Hand the prompt the original question and every verdict to be tested."""
        if request.question is None:
            raise SkillError(f"{self.name} needs the run's question artifact, none supplied")

        decisions = request.of_kind(ArtifactKind.DECISION)
        if not decisions:
            raise SkillError(f"{self.name} needs at least one decision, none supplied")

        return InterviewPlanInput(
            question=request.question.model_dump(mode="json"),
            decisions=[artifact.model_dump(mode="json") for artifact in decisions],
        )

    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        """Bind each plan to the decision it names, refusing an inexact set."""
        if not isinstance(output, InterviewPlanOutput):
            raise SkillError(
                f"{self.name} expected {InterviewPlanOutput.__name__}, got {type(output).__name__}"
            )

        plans: list[Artifact] = [
            InterviewPlan(
                id=InterviewPlan.make_id(),
                run_id=request.run_id,
                decision=ArtifactRef(kind=ArtifactKind.DECISION, id=subject.id),
                objective=entry.objective,
                targets=[
                    InterviewTarget(
                        persona=target.persona,
                        where_to_find=target.where_to_find,
                        why_them=target.why_them,
                    )
                    for target in entry.targets
                ],
                questions=[
                    InterviewQuestion(
                        question=planned.question,
                        tests_assumption=planned.tests_assumption,
                    )
                    for planned in entry.questions
                ],
                success_criteria=entry.success_criteria,
                failure_criteria=entry.failure_criteria,
                questions_to_avoid=entry.questions_to_avoid,
                confidence=entry.confidence,
                evidence_level=entry.evidence_level,
                # One plan tests one decision, so provenance is narrowed here; left
                # to `execute` it would name every sibling decision as a parent.
                parents=[subject.ref],
            )
            for entry, subject in self._paired(output, request)
        ]
        return plans

    def _paired(
        self,
        output: InterviewPlanOutput,
        request: SkillRequest,
    ) -> list[tuple[InterviewPlanRead, Artifact]]:
        """Match every plan to its decision, and insist the pairing is exact.

        The same three failures as the per-opportunity stages, one step along: a
        decision id nobody supplied, one planned for twice, and a decision left with
        no way to test it — which is how a verdict ends up final by default.
        """
        subjects = {artifact.id: artifact for artifact in request.of_kind(ArtifactKind.DECISION)}
        known = ", ".join(sorted(subjects)) or "<none>"

        paired: list[tuple[InterviewPlanRead, Artifact]] = []
        seen: set[str] = set()
        for entry in output.plans:
            subject = subjects.get(entry.decision_id)
            if subject is None:
                raise SkillError(
                    f"{self.name} planned interviews for decision {entry.decision_id!r}, "
                    f"which was not supplied; had: {known}"
                )
            if entry.decision_id in seen:
                raise SkillError(
                    f"{self.name} returned two plans for decision {entry.decision_id!r}"
                )
            seen.add(entry.decision_id)
            paired.append((entry, subject))

        if unhandled := sorted(set(subjects) - seen):
            raise SkillError(
                f"{self.name} planned {len(seen)} of {len(subjects)} decisions; "
                f"unplanned: {', '.join(unhandled)}"
            )
        return paired


__all__ = [
    "InterviewPlanInput",
    "InterviewPlanOutput",
    "InterviewPlanRead",
    "InterviewPlanSkill",
    "InterviewTargetRead",
    "PlannedQuestion",
]
