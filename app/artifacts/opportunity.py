"""`Opportunity` — an inferred chance to fix a clustered workflow problem.

The first stage that reasons beyond what the evidence literally says, so it
carries `missing_evidence`: naming what would have to be true, and is not yet
known, is what keeps an inference from hardening into a claim. Competitors are
deliberately out of scope here — that is a later stage.
"""

from typing import ClassVar

from pydantic import Field, field_validator

from app.artifacts.base import Artifact, ArtifactKind, ArtifactRef, ClaimType


class Opportunity(Artifact):
    """A candidate response to a pain cluster."""

    kind: ClassVar[ArtifactKind] = ArtifactKind.OPPORTUNITY
    id_prefix: ClassVar[str] = "op"

    pain_cluster: ArtifactRef | None = Field(
        default=None,
        description="The clustered pain this answers. An explicit link, not merely "
        "provenance: `parents` records every cluster that was read, this records the "
        "one the opportunity is a response to.",
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
        description="Opportunities are inferences by construction; hypothesis when weaker.",
    )

    @field_validator("title", "workflow", "icp", "buyer", "problem", "why_now")
    @classmethod
    def _require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("missing_evidence")
    @classmethod
    def _clean_entries(cls, value: list[str]) -> list[str]:
        return [line.strip() for line in value if line.strip()]

    @field_validator("pain_cluster")
    @classmethod
    def _must_reference_a_cluster(cls, value: ArtifactRef | None) -> ArtifactRef | None:
        if value is not None and value.kind is not ArtifactKind.PAIN_CLUSTER:
            raise ValueError(f"pain_cluster must reference a pain cluster, got {value.kind.value}")
        return value


__all__ = ["Opportunity"]
