"""`InterviewPlan` — how to go and check the decision against real people.

The last stage. Everything upstream is desk research; this turns the decision's
riskiest assumptions into questions someone can actually ask.

Carries `failure_criteria` and `questions_to_avoid` alongside the obvious fields,
because the usual way founder interviews go wrong is not asking too few questions
— it is asking leading ones and defining success loosely enough that any answer
confirms the plan.
"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.artifacts.base import Artifact, ArtifactKind, ArtifactRef


class InterviewTarget(BaseModel):
    """A kind of person to talk to, and how to actually reach them."""

    model_config = ConfigDict(extra="forbid")

    persona: str = Field(description="Who they are, specific enough to screen for.")
    where_to_find: list[str] = Field(
        default_factory=list,
        description="Concrete places — communities, job titles, events, existing customers.",
    )
    why_them: str | None = None

    @field_validator("persona")
    @classmethod
    def _require_persona(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("persona must not be blank")
        return cleaned

    @field_validator("where_to_find")
    @classmethod
    def _clean_entries(cls, value: list[str]) -> list[str]:
        return [line.strip() for line in value if line.strip()]


class InterviewQuestion(BaseModel):
    """One question, and the assumption it is meant to test."""

    model_config = ConfigDict(extra="forbid")

    question: str
    tests_assumption: str | None = Field(
        default=None,
        description="The belief this would falsify. A question testing nothing is small talk.",
    )

    @field_validator("question")
    @classmethod
    def _require_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class InterviewPlan(Artifact):
    """A plan for validating a decision with primary research."""

    kind: ClassVar[ArtifactKind] = ArtifactKind.INTERVIEW_PLAN
    id_prefix: ClassVar[str] = "iv"

    decision: ArtifactRef = Field(
        description="The decision this tests. An explicit link, not merely provenance.",
    )
    objective: str = Field(description="What these interviews are meant to settle.")
    targets: list[InterviewTarget] = Field(
        default_factory=list,
        description="Who to interview and where to find them.",
    )
    questions: list[InterviewQuestion] = Field(default_factory=list)
    success_criteria: list[str] = Field(
        default_factory=list,
        description="What an answer must look like to count as validation.",
    )
    failure_criteria: list[str] = Field(
        default_factory=list,
        description="What would falsify the thesis. Stated up front so the result "
        "cannot be reinterpreted afterwards.",
    )
    questions_to_avoid: list[str] = Field(
        default_factory=list,
        description="Leading or hypothetical questions that would produce agreeable noise.",
    )

    @field_validator("objective")
    @classmethod
    def _require_objective(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("objective must not be blank")
        return cleaned

    @field_validator("success_criteria", "failure_criteria", "questions_to_avoid")
    @classmethod
    def _clean_entries(cls, value: list[str]) -> list[str]:
        return [line.strip() for line in value if line.strip()]

    @field_validator("decision")
    @classmethod
    def _must_reference_a_decision(cls, value: ArtifactRef) -> ArtifactRef:
        if value.kind is not ArtifactKind.DECISION:
            raise ValueError(f"decision must reference a decision, got {value.kind.value}")
        return value


__all__ = ["InterviewPlan", "InterviewQuestion", "InterviewTarget"]
