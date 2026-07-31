"""Kinds, lifecycle, evidential strength, and the envelope shared by every artifact.

The fields every artifact is required to carry — `version`, `id`, timestamps,
`status`, `confidence`, `evidence_level` — are declared once here rather than
repeated per model, so they cannot drift apart between kinds.
"""

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field

from app.utils.ids import new_id
from app.utils.time import utcnow

SCHEMA_VERSION: Final[int] = 1
"""Current shape of the artifact envelope. Bump when a field is added or removed."""


class ArtifactKind(StrEnum):
    """Every kind of artifact the pipeline can produce, one per workspace directory."""

    QUESTION = "question"
    EVIDENCE = "evidence"
    RESEARCH_BRIEF = "research_brief"
    PAIN_CLUSTER = "pain_cluster"
    OPPORTUNITY = "opportunity"
    MARKET_ANALYSIS = "market_analysis"
    COMPETITION_ANALYSIS = "competition_analysis"
    CONTRADICTION_ANALYSIS = "contradiction_analysis"
    DECISION = "decision"
    INTERVIEW_PLAN = "interview_plan"
    LEAD = "lead"
    REPORT = "report"

    @property
    def directory(self) -> str:
        """Name of the `workspace/` subdirectory that stores artifacts of this kind."""
        return _KIND_DIRECTORIES[self]


# The directory names are plural while the kind values are singular, so the mapping is
# spelled out rather than derived: the on-disk layout is a contract with the operator and
# must never drift because someone changed a naming rule.
_KIND_DIRECTORIES: Final[Mapping[ArtifactKind, str]] = MappingProxyType(
    {
        ArtifactKind.QUESTION: "questions",
        ArtifactKind.EVIDENCE: "evidence",
        ArtifactKind.RESEARCH_BRIEF: "briefs",
        ArtifactKind.PAIN_CLUSTER: "clusters",
        ArtifactKind.OPPORTUNITY: "opportunities",
        ArtifactKind.MARKET_ANALYSIS: "market",
        ArtifactKind.COMPETITION_ANALYSIS: "competition",
        ArtifactKind.CONTRADICTION_ANALYSIS: "contradictions",
        ArtifactKind.DECISION: "decisions",
        ArtifactKind.INTERVIEW_PLAN: "interviews",
        ArtifactKind.LEAD: "leads",
        ArtifactKind.REPORT: "reports",
    },
)


class ClaimType(StrEnum):
    """How a statement relates to the evidence underneath it.

    The pipeline's central discipline: the further downstream a stage sits, the
    more tempting it is to present reasoning as fact. Tagging every claim forces
    that distinction to survive into the artifact, where a reader can audit it.
    """

    OBSERVATION = "observation"
    """Directly attested by a source. Should be quotable."""

    INFERENCE = "inference"
    """Derived from observations by reasoning. The reasoning should be stated."""

    HYPOTHESIS = "hypothesis"
    """Proposed, not yet supported. Needs validating before it is relied on."""


class ArtifactStatus(StrEnum):
    """Lifecycle of an artifact, independent of what it says.

    Kept separate from any domain verdict: a `Decision` can be `READY` (the record
    is complete) while its verdict is `REJECT` (the opportunity was turned down).
    Conflating the two loses the difference between "we have not finished thinking"
    and "we thought about it and said no".
    """

    DRAFT = "draft"
    """Created but incomplete — not safe for a downstream stage to consume."""

    READY = "ready"
    """Complete and available to the next stage."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    """The stage ran and could not reach a conclusion the evidence supports.

    Distinct from three things it is easy to collapse into: a stage that never ran
    (an error, not an artifact), a stage that ran and found nothing (a complete
    artifact with empty findings), and a stage whose output was rejected on review.
    Recording it explicitly is what stops "we could not tell" being read downstream
    as "there is nothing there".
    """

    SUPERSEDED = "superseded"
    """Replaced by a newer version; retained for the audit trail."""

    REJECTED = "rejected"
    """Reviewed and discarded. Downstream stages must skip it."""

    ARCHIVED = "archived"
    """Retained for history but out of play."""


class EvidenceLevel(StrEnum):
    """How strongly an artifact is grounded in observed reality.

    Ordinal, weakest to strongest. Deliberately *not* the same thing as
    `confidence`: an artifact can be highly confident about weak evidence (we are
    sure only one person said this), or unsure about strong evidence (the numbers
    are solid but we cannot tell what they mean).
    """

    NONE = "none"
    """Asserted with nothing behind it — a guess, or a model's prior."""

    ANECDOTAL = "anecdotal"
    """A single unverified report."""

    CORROBORATED = "corroborated"
    """Several independent sources agree."""

    MEASURED = "measured"
    """Backed by quantitative data."""

    VERIFIED = "verified"
    """Confirmed against a primary source."""

    @property
    def rank(self) -> int:
        """Position on the scale, so levels can be compared and thresholded."""
        return _EVIDENCE_RANKS[self]


_EVIDENCE_RANKS: Final[Mapping[EvidenceLevel, int]] = MappingProxyType(
    {
        EvidenceLevel.NONE: 0,
        EvidenceLevel.ANECDOTAL: 1,
        EvidenceLevel.CORROBORATED: 2,
        EvidenceLevel.MEASURED: 3,
        EvidenceLevel.VERIFIED: 4,
    },
)


Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
"""A probability-like score in [0, 1]. `None` means "not yet assessed"."""


class ArtifactRef(BaseModel):
    """A pointer to another artifact: enough to locate its JSON file on disk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ArtifactKind
    id: str


class Artifact(BaseModel):
    """Envelope every pipeline artifact shares; subclasses add the domain payload."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: ClassVar[ArtifactKind]
    """Set by each concrete subclass. Not serialised — the directory implies it."""

    id_prefix: ClassVar[str]
    """Short prefix for generated ids, e.g. `ev` -> `ev_3f9c...`."""

    # --- identity ---
    id: str
    run_id: str

    # --- versioning ---
    version: int = Field(
        default=1,
        ge=1,
        description="Revision of this artifact's content. Bumped when it is rewritten.",
    )
    schema_version: int = Field(
        default=SCHEMA_VERSION,
        ge=1,
        description="Shape of the envelope this artifact was written with.",
    )

    # --- timestamps ---
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    # --- assessment ---
    status: ArtifactStatus = ArtifactStatus.DRAFT
    confidence: Confidence | None = Field(
        default=None,
        description="How much this artifact's content is trusted. None = not yet assessed.",
    )
    evidence_level: EvidenceLevel = EvidenceLevel.NONE

    # --- provenance ---
    parents: list[ArtifactRef] = Field(
        default_factory=list,
        description="What this artifact was derived from. Walk it back to reach evidence.",
    )
    supersedes: ArtifactRef | None = Field(
        default=None,
        description="The earlier version this replaces, when version > 1.",
    )

    # --- reproducibility ---
    fingerprint: str | None = Field(
        default=None,
        description="Digest of everything that determined this artifact: the skill, its "
        "prompt, the model, and the exact inputs. Two artifacts with the same fingerprint "
        "were produced by the same computation, so the second need never be run. Stamped "
        "at creation, which is what keeps it compatible with immutability.",
    )

    @classmethod
    def make_id(cls) -> str:
        """Generate a fresh id carrying this kind's prefix."""
        return new_id(cls.id_prefix)

    @property
    def ref(self) -> ArtifactRef:
        """A pointer to this artifact, for use in another artifact's `parents`."""
        return ArtifactRef(kind=type(self).kind, id=self.id)


__all__ = [
    "SCHEMA_VERSION",
    "Artifact",
    "ArtifactKind",
    "ArtifactRef",
    "ArtifactStatus",
    "ClaimType",
    "Confidence",
    "EvidenceLevel",
]
