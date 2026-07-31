"""`Decision` — build, reject, or wait, with the reasoning left visible.

`next_validation_step` and `biggest_unknown` are required for every verdict, not
just the hesitant ones. A confident BUILD that cannot name what it is still unsure
about has stopped reasoning, and that is exactly when a decision needs checking.
"""

from enum import StrEnum
from typing import ClassVar

from pydantic import Field, field_validator

from app.artifacts.base import Artifact, ArtifactKind, ArtifactRef, Confidence


class Verdict(StrEnum):
    """What was decided about an opportunity.

    The domain outcome, not the record's lifecycle: a decision whose verdict is
    `REJECT` is still a complete, `READY` artifact. See `ArtifactStatus`.
    """

    BUILD = "build"
    """Worth pursuing now."""

    REJECT = "reject"
    """Not worth pursuing. Recorded so it is not rediscovered later."""

    WAIT = "wait"
    """Plausible, but something has to change or be learned first."""


class Decision(Artifact):
    """The recorded judgement on one opportunity."""

    kind: ClassVar[ArtifactKind] = ArtifactKind.DECISION
    id_prefix: ClassVar[str] = "dec"

    opportunity: ArtifactRef = Field(
        description="The opportunity this rules on. An explicit link, not merely provenance.",
    )
    verdict: Verdict
    decision_confidence: Confidence = Field(
        description="How sure the call is. Distinct from the envelope's `confidence`, "
        "which grades the record; this grades the judgement.",
    )
    top_reasons: list[str] = Field(
        min_length=1,
        description="Why, strongest first. A decision with no stated reason is not one.",
    )
    next_validation_step: str = Field(
        description="The single cheapest next action that would move the confidence.",
    )
    biggest_unknown: str = Field(
        description="What is still most likely to be wrong.",
    )

    @field_validator("next_validation_step", "biggest_unknown")
    @classmethod
    def _require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("top_reasons")
    @classmethod
    def _require_reasons(cls, value: list[str]) -> list[str]:
        cleaned = [line.strip() for line in value if line.strip()]
        if not cleaned:
            raise ValueError("a decision needs at least one reason")
        return cleaned

    @field_validator("opportunity")
    @classmethod
    def _must_reference_an_opportunity(cls, value: ArtifactRef) -> ArtifactRef:
        if value.kind is not ArtifactKind.OPPORTUNITY:
            raise ValueError(f"opportunity must reference an opportunity, got {value.kind.value}")
        return value


__all__ = ["Decision", "Verdict"]
