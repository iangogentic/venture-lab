"""Rerunnable, evidence-governed pilot runs.

The pilot accepts a normalized :class:`EvidencePacket`, creates hypotheses in
either explicit offline-fixture or model-backed mode, runs an independent
falsification pass, and evaluates the fixed G0-G7 policy.  Models propose and
critique; deterministic code owns identifiers, budgets, gates, persistence,
and the safety boundary.

Every durable artifact is written once, copied into the content-addressed
snapshot store, and named in the hash-chained ledger.  A repeated invocation
with the same run id and identical inputs returns the completed immutable run.
It never silently resumes or overwrites a partial or different run.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, TypeAdapter, ValidationError, field_validator, model_validator

from app.llm.service import LLM
from app.utils.errors import LLMError
from app.venture.analysts import (
    StructuredGenerator,
    classification_input_payload,
    classification_input_payload_v2,
    falsify_hypothesis,
    generate_hypotheses,
    review_classification_comparative,
)
from app.venture.core import (
    ClaimEvidenceAssessment,
    FrozenModel,
    GateContext,
    GateDecision,
    GateEvaluation,
    Ledger,
    LedgerEvent,
    SnapshotStore,
    canonical_json,
    evaluate_gates,
    make_content_id,
    safe_join,
    sha256_bytes,
    validate_identifier,
    validate_relative_path,
    validate_sha256,
)
from app.venture.discovery import (
    CandidateHypothesis,
    ClassificationComparisonReview,
    ClassificationReview,
    ClassificationReviewExecution,
    ClassificationReviewOutcome,
    EvidencePacket,
    FalsificationDimension,
    FalsificationFinding,
    FalsificationOutcome,
    FalsificationReport,
    HypothesisDraft,
    PacketMeasurement,
    classification_scope_measurement,
    classification_scope_measurements,
    materialize_hypothesis,
    scope_falsification_packet,
    validate_classification_review,
    validate_falsification_refs,
)
from app.venture.operations import (
    ActionDecision,
    BudgetGuard,
    BudgetPolicy,
    BudgetUsage,
    ExternalAction,
    KillSwitch,
    authorize_action,
)
from app.venture.provenance import (
    ImplementationManifest,
    build_implementation_bundle,
    verify_implementation_bundle,
)

_MAX_INPUT_BYTES = 50_000_000
_ACTOR_ID = "venture-pilot"
_LEGACY_CLASSIFICATION_VISIBLE_FIELDS = (
    "anonymized_offer",
    "official_scope_measurement",
)
_COMPARATIVE_CLASSIFICATION_VISIBLE_FIELDS = (
    "anonymized_offer",
    "official_scope_measurements",
)


class PilotError(RuntimeError):
    """A safe, user-facing pilot failure."""


class PilotIntegrityError(PilotError):
    """An immutable run or one of its ledger pointers no longer verifies."""


class PilotMode(StrEnum):
    """How hypotheses and falsification reports are produced."""

    OFFLINE = "offline"
    LLM = "llm"


class OfflineFalsification(FrozenModel):
    """Fixture form of a falsification report; the runner assigns safe ids."""

    findings: tuple[FalsificationFinding, ...] = Field(
        min_length=len(FalsificationDimension),
        max_length=len(FalsificationDimension),
    )
    explicit_illegality_found: bool = False
    explicit_unfinanceable_found: bool = False
    explicit_negative_stressed_contribution_found: bool = False
    kill_recommendation: bool = False
    kill_basis: str | None = None
    critical_unknowns: tuple[str, ...] = ()


class OfflineClassificationReview(FrozenModel):
    """Fixture form of an independent classification review."""

    outcome: ClassificationReviewOutcome
    analysis: str = Field(min_length=1, max_length=2_000)
    mismatches: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    plausible_naics_codes: tuple[str, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
    )

    @field_validator("plausible_naics_codes")
    @classmethod
    def _exact_unique_plausible_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not code.isdigit() or len(code) != 6 for code in value):
            raise ValueError("plausible_naics_codes require six-digit NAICS codes")
        if len(set(value)) != len(value):
            raise ValueError("plausible_naics_codes must be unique")
        return tuple(sorted(value))


class OfflineCandidateFixture(FrozenModel):
    """One deterministic hypothesis and its optional independent critique."""

    hypothesis: HypothesisDraft
    classification_review: OfflineClassificationReview
    falsification: OfflineFalsification | None = None

    @model_validator(mode="after")
    def _only_fit_can_have_market_falsification(self) -> Self:
        if (
            self.classification_review.outcome is not ClassificationReviewOutcome.FIT
            and self.falsification is not None
        ):
            raise ValueError(
                "offline market falsification is allowed only after a FIT classification review"
            )
        proposed_code = self.hypothesis.naics_codes[0]
        plausible = self.classification_review.plausible_naics_codes
        if self.classification_review.outcome is ClassificationReviewOutcome.FIT and plausible != (
            proposed_code,
        ):
            raise ValueError(
                "offline FIT classification requires only the proposed provider code "
                "to be plausible"
            )
        if (
            self.classification_review.outcome is ClassificationReviewOutcome.CONTRADICTS
            and proposed_code in plausible
        ):
            raise ValueError(
                "offline CONTRADICTS classification cannot keep the proposed code plausible"
            )
        return self


class OfflinePilotFixture(FrozenModel):
    """All model outputs frozen for a deterministic, network-free pilot."""

    candidates: tuple[OfflineCandidateFixture, ...] = Field(min_length=1, max_length=25)

    @model_validator(mode="after")
    def _unique_titles(self) -> Self:
        titles = [item.hypothesis.title.casefold() for item in self.candidates]
        if len(titles) != len(set(titles)):
            raise ValueError("offline fixture hypothesis titles must be unique")
        return self


class PilotCharter(FrozenModel):
    """Operator-supplied G0 facts that cannot be inferred from a hypothesis."""

    founder_constraints_defined: bool | None = None
    outcome_horizons_defined: bool | None = None


class PilotConfiguration(FrozenModel):
    """Exact immutable invocation inputs that determine one pilot execution."""

    schema_version: str = "venture-pilot-configuration-v1"
    mode: PilotMode
    fixture: OfflinePilotFixture | None
    model: str | None
    max_hypotheses: int = Field(ge=1, le=25)
    budget_policy: BudgetPolicy
    charter: PilotCharter

    @model_validator(mode="after")
    def _mode_matches_fixture(self) -> Self:
        if self.schema_version != "venture-pilot-configuration-v1":
            raise ValueError("unsupported pilot-configuration schema")
        if self.mode is PilotMode.OFFLINE and self.fixture is None:
            raise ValueError("offline pilot configuration requires a fixture")
        if self.mode is PilotMode.LLM and self.fixture is not None:
            raise ValueError("LLM pilot configuration cannot include a fixture")
        return self


class CandidateInputProvenance(FrozenModel):
    """Hashes of the exact role-separated inputs persisted for one candidate."""

    opportunity_id: str
    classification_assignment_id: str
    classification_input_sha256: str
    falsification_evidence_packet_sha256: str

    @field_validator("opportunity_id", "classification_assignment_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator(
        "classification_input_sha256",
        "falsification_evidence_packet_sha256",
    )
    @classmethod
    def _valid_hashes(cls, value: str) -> str:
        return validate_sha256(value)


class CandidateInputProvenanceV2(FrozenModel):
    """V6 input hashes plus an accepted-comparison or quarantined-failure binding."""

    opportunity_id: str
    classification_assignment_id: str
    classification_input_sha256: str
    falsification_evidence_packet_sha256: str
    classification_response_sha256: str | None
    classification_failure_sha256: str | None

    @field_validator("opportunity_id", "classification_assignment_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator(
        "classification_input_sha256",
        "falsification_evidence_packet_sha256",
        "classification_response_sha256",
        "classification_failure_sha256",
    )
    @classmethod
    def _valid_hashes(cls, value: str | None) -> str | None:
        return validate_sha256(value) if value is not None else None

    @model_validator(mode="after")
    def _exactly_one_classification_outcome(self) -> Self:
        if (self.classification_response_sha256 is None) == (
            self.classification_failure_sha256 is None
        ):
            raise ValueError(
                "v2 candidate provenance requires exactly one classification response "
                "or failure hash"
            )
        return self


class PilotRunProvenance(FrozenModel):
    """Versioned source and prompt-input provenance for a new pilot run."""

    schema_version: str = "venture-pilot-run-provenance-v1"
    configuration_sha256: str
    evidence_packet_sha256: str
    implementation_manifest_sha256: str
    implementation_source_tar_sha256: str
    candidate_inputs: tuple[CandidateInputProvenance, ...] = Field(min_length=1)
    provider_usage_tracking: Literal["pending"] = "pending"
    provider_usage_tracking_note: str = (
        "Budget usage records deterministic preflight reservations; actual provider "
        "billing, token usage, and retries are not yet captured."
    )

    @field_validator(
        "configuration_sha256",
        "evidence_packet_sha256",
        "implementation_manifest_sha256",
        "implementation_source_tar_sha256",
    )
    @classmethod
    def _valid_hashes(cls, value: str) -> str:
        return validate_sha256(value)

    @model_validator(mode="after")
    def _unique_candidate_bindings(self) -> Self:
        if self.schema_version != "venture-pilot-run-provenance-v1":
            raise ValueError("unsupported pilot-run-provenance schema")
        opportunity_ids = tuple(item.opportunity_id for item in self.candidate_inputs)
        if len(opportunity_ids) != len(set(opportunity_ids)):
            raise ValueError("candidate input provenance must have unique opportunity ids")
        assignment_ids = tuple(item.classification_assignment_id for item in self.candidate_inputs)
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError("candidate input provenance must have unique assignment ids")
        return self


class PilotRunProvenanceV2(FrozenModel):
    """V6 source, input, accepted-comparison, and failure provenance."""

    schema_version: str = "venture-pilot-run-provenance-v2"
    configuration_sha256: str
    evidence_packet_sha256: str
    implementation_manifest_sha256: str
    implementation_source_tar_sha256: str
    candidate_inputs: tuple[CandidateInputProvenanceV2, ...] = Field(min_length=1)
    provider_usage_tracking: Literal["pending"] = "pending"
    provider_usage_tracking_note: str = (
        "Budget usage records deterministic preflight reservations; actual provider "
        "billing, token usage, and retries are not yet captured."
    )

    @field_validator(
        "configuration_sha256",
        "evidence_packet_sha256",
        "implementation_manifest_sha256",
        "implementation_source_tar_sha256",
    )
    @classmethod
    def _valid_hashes(cls, value: str) -> str:
        return validate_sha256(value)

    @model_validator(mode="after")
    def _unique_candidate_bindings(self) -> Self:
        if self.schema_version != "venture-pilot-run-provenance-v2":
            raise ValueError("unsupported pilot-run-provenance v2 schema")
        opportunity_ids = tuple(item.opportunity_id for item in self.candidate_inputs)
        if len(opportunity_ids) != len(set(opportunity_ids)):
            raise ValueError("candidate input provenance must have unique opportunity ids")
        assignment_ids = tuple(item.classification_assignment_id for item in self.candidate_inputs)
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError("candidate input provenance must have unique assignment ids")
        return self


class FalsificationFailureKind(StrEnum):
    """Typed reason one candidate's independent critique was quarantined."""

    TRANSPORT = "transport"
    SCHEMA = "schema"
    VALIDATION = "validation"
    EVIDENCE_REFERENCE = "evidence_reference"


class ClassificationReviewFailureKind(StrEnum):
    """Typed reason a classification output was rejected fail-closed."""

    TRANSPORT = "transport"
    SCHEMA = "schema"
    VALIDATION = "validation"
    EVIDENCE_REFERENCE = "evidence_reference"


class ClassificationAssignment(FrozenModel):
    """Persisted, role-separated assignment shown to a classification reviewer."""

    assignment_id: str
    opportunity_id: str
    naics_code: str
    scope_measurement_ref: str
    actor_model: str = Field(min_length=1, max_length=300)
    visible_fields: tuple[str, ...] = (
        "anonymized_offer",
        "official_scope_measurement",
    )
    tool_access: bool = False
    started_at: datetime
    completed_at: datetime

    @field_validator("assignment_id", "opportunity_id", "scope_measurement_ref")
    @classmethod
    def _safe_assignment_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("naics_code")
    @classmethod
    def _exact_provider_code(cls, value: str) -> str:
        if not value.isdigit() or len(value) != 6:
            raise ValueError("classification assignment requires a six-digit NAICS code")
        return value

    @field_validator("visible_fields")
    @classmethod
    def _exact_visible_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        allowed = {
            _LEGACY_CLASSIFICATION_VISIBLE_FIELDS,
            _COMPARATIVE_CLASSIFICATION_VISIBLE_FIELDS,
        }
        if value not in allowed:
            raise ValueError(
                "classification assignment may expose only the offer and either the "
                "legacy single official scope or comparative official scopes"
            )
        return value

    @field_validator("started_at", "completed_at")
    @classmethod
    def _aware_assignment_times(cls, value: datetime) -> datetime:
        _require_aware(value)
        return value

    @model_validator(mode="after")
    def _valid_assignment_completion(self) -> Self:
        if self.tool_access:
            raise ValueError("classification reviewer must be tool-free")
        if self.completed_at < self.started_at:
            raise ValueError("classification assignment completion precedes start")
        return self


class ClassificationReviewFailure(FrozenModel):
    """Persisted invalid reviewer output; classification remains fail-closed."""

    opportunity_id: str
    assignment_id: str
    naics_code: str
    scope_measurement_ref: str
    kind: ClassificationReviewFailureKind
    error_type: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=1_000)
    quarantined_at: datetime
    policy_effect: str = (
        "classification unresolved; market falsification skipped and cohort continued"
    )

    @field_validator("opportunity_id", "assignment_id", "scope_measurement_ref")
    @classmethod
    def _safe_failure_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("naics_code")
    @classmethod
    def _exact_failure_code(cls, value: str) -> str:
        if not value.isdigit() or len(value) != 6:
            raise ValueError("classification failure requires a six-digit NAICS code")
        return value

    @field_validator("quarantined_at")
    @classmethod
    def _aware_failure_time(cls, value: datetime) -> datetime:
        _require_aware(value)
        return value


class FalsificationFailure(FrozenModel):
    """Persisted failure record; it has no authority to set a gate or verdict."""

    opportunity_id: str
    assignment_id: str
    kind: FalsificationFailureKind
    error_type: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=1_000)
    quarantined_at: datetime
    policy_effect: str = "falsification absent; every G3 predicate remains unknown"

    @field_validator("opportunity_id", "assignment_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("quarantined_at")
    @classmethod
    def _aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("quarantined_at must include a timezone")
        return value


class PilotCandidateResult(FrozenModel):
    """One hypothesis, critique, evidence slice, and exact gate evaluation."""

    hypothesis: CandidateHypothesis
    classification_assignment: ClassificationAssignment
    classification_review: ClassificationReview | None
    classification_review_failure: ClassificationReviewFailure | None = None
    falsification: FalsificationReport | None
    falsification_failure: FalsificationFailure | None = None
    evidence: tuple[PacketMeasurement, ...] = Field(min_length=2)
    gates: GateEvaluation
    unverified_disqualifier_allegations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _role_outputs_are_consistent(self) -> Self:
        if (self.classification_review is None) == (self.classification_review_failure is None):
            raise ValueError(
                "a candidate requires exactly one classification review or review failure"
            )
        if (
            self.classification_assignment.opportunity_id != self.hypothesis.opportunity_id
            or self.classification_assignment.naics_code != self.hypothesis.naics_codes[0]
        ):
            raise ValueError("classification assignment does not bind to its candidate")
        if self.classification_review is not None and (
            self.classification_review.opportunity_id != self.hypothesis.opportunity_id
            or self.classification_review.assignment_id
            != self.classification_assignment.assignment_id
        ):
            raise ValueError("classification review does not bind to its assignment")
        if self.classification_review_failure is not None and (
            self.classification_review_failure.opportunity_id != self.hypothesis.opportunity_id
            or self.classification_review_failure.assignment_id
            != self.classification_assignment.assignment_id
        ):
            raise ValueError("classification failure does not bind to its assignment")
        if self.falsification is not None and self.falsification_failure is not None:
            raise ValueError("a candidate cannot have both falsification and failure")
        classification_fit = (
            self.classification_review is not None
            and self.classification_review.outcome is ClassificationReviewOutcome.FIT
        )
        if not classification_fit and (
            self.falsification is not None or self.falsification_failure is not None
        ):
            raise ValueError("market falsification cannot run before a FIT classification review")
        return self


class PilotRunResult(FrozenModel):
    """Research result without a composite score or model-authored verdict."""

    schema_version: str = "venture-pilot-v6"
    run_id: str
    mode: PilotMode
    packet_id: str
    information_cutoff: datetime
    packet_sha256: str
    configuration_sha256: str
    run_provenance_sha256: str | None = None
    candidates: tuple[PilotCandidateResult, ...] = Field(min_length=1)
    budget_policy: BudgetPolicy
    budget_usage: BudgetUsage
    action_policy: tuple[ActionDecision, ...]
    generated_at: datetime

    @field_validator("run_id", "packet_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("packet_sha256", "configuration_sha256")
    @classmethod
    def _valid_hashes(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("run_provenance_sha256")
    @classmethod
    def _valid_optional_hash(cls, value: str | None) -> str | None:
        return validate_sha256(value) if value is not None else None

    @field_validator("information_cutoff", "generated_at")
    @classmethod
    def _aware_times(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("pilot timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def _semantic_schemas_require_provenance(self) -> Self:
        if self.schema_version in {"venture-pilot-v5", "venture-pilot-v6"} and (
            self.run_provenance_sha256 is None
        ):
            raise ValueError(f"{self.schema_version} requires run provenance")
        return self


class LegacyPilotRunResult(FrozenModel):
    """Opaque top-level envelope for immutable results predating current schemas.

    Historical candidates remain byte-preserved dictionaries.  This keeps old
    completed runs loadable without weakening current nested model validators.
    """

    schema_version: str
    run_id: str
    mode: PilotMode
    packet_id: str
    information_cutoff: datetime
    packet_sha256: str
    configuration_sha256: str
    candidates: tuple[dict[str, object], ...] = Field(min_length=1)
    budget_policy: BudgetPolicy
    budget_usage: BudgetUsage
    action_policy: tuple[ActionDecision, ...]
    generated_at: datetime

    @field_validator("run_id", "packet_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("packet_sha256", "configuration_sha256")
    @classmethod
    def _valid_hashes(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("information_cutoff", "generated_at")
    @classmethod
    def _aware_times(cls, value: datetime) -> datetime:
        _require_aware(value)
        return value

    @model_validator(mode="after")
    def _only_legacy_versions(self) -> Self:
        if self.schema_version not in {
            "venture-pilot-v1",
            "venture-pilot-v2",
            "venture-pilot-v3",
        }:
            raise ValueError("unsupported legacy pilot-result schema")
        return self


class PilotArtifactPointer(FrozenModel):
    """A per-run file, immutable snapshot, and ledger event joined by hash."""

    kind: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    sha256: str
    size_bytes: int = Field(ge=0)
    run_relative_path: str
    snapshot_relative_path: str
    ledger_event_id: str

    @field_validator("sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("run_relative_path", "snapshot_relative_path")
    @classmethod
    def _safe_paths(cls, value: str) -> str:
        return validate_relative_path(value)

    @field_validator("ledger_event_id")
    @classmethod
    def _safe_event_id(cls, value: str) -> str:
        return validate_identifier(value)


class PilotManifest(FrozenModel):
    """Index of immutable artifacts; facts live in the artifacts, not here."""

    schema_version: str = "venture-pilot-manifest-v6"
    run_id: str
    mode: PilotMode
    packet_id: str
    packet_sha256: str
    configuration_sha256: str
    run_provenance_sha256: str | None = None
    artifacts: tuple[PilotArtifactPointer, ...] = Field(min_length=1)

    @field_validator("run_id", "packet_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("packet_sha256", "configuration_sha256")
    @classmethod
    def _valid_hashes(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("run_provenance_sha256")
    @classmethod
    def _valid_optional_hash(cls, value: str | None) -> str | None:
        return validate_sha256(value) if value is not None else None

    @model_validator(mode="after")
    def _unique_artifact_paths(self) -> Self:
        paths = [item.run_relative_path for item in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest artifact paths must be unique")
        event_ids = [item.ledger_event_id for item in self.artifacts]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("manifest ledger event ids must be unique")
        if self.schema_version in {
            "venture-pilot-manifest-v5",
            "venture-pilot-manifest-v6",
        }:
            if self.run_provenance_sha256 is None:
                raise ValueError(f"{self.schema_version} requires run provenance")
            provenance = tuple(item for item in self.artifacts if item.kind == "run_provenance")
            if len(provenance) != 1 or provenance[0].sha256 != self.run_provenance_sha256:
                raise ValueError(
                    f"{self.schema_version} must bind exactly one run-provenance artifact"
                )
        return self


class PilotExecution(FrozenModel):
    """Completed pilot plus the ledger head proving its durable handoff."""

    manifest: PilotManifest
    result: PilotRunResult
    completion_event_id: str
    ledger_head_hash: str
    report_relative_path: str

    @field_validator("completion_event_id")
    @classmethod
    def _safe_event_id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("ledger_head_hash")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("report_relative_path")
    @classmethod
    def _safe_report_path(cls, value: str) -> str:
        return validate_relative_path(value)


class LegacyPilotExecution(FrozenModel):
    """Load-only envelope for an immutable pre-current result."""

    manifest: PilotManifest
    result: LegacyPilotRunResult
    completion_event_id: str
    ledger_head_hash: str
    report_relative_path: str

    @field_validator("completion_event_id")
    @classmethod
    def _safe_event_id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("ledger_head_hash")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("report_relative_path")
    @classmethod
    def _safe_report_path(cls, value: str) -> str:
        return validate_relative_path(value)


class PilotVerification(FrozenModel):
    """Successful end-to-end integrity check."""

    run_id: str
    artifact_count: int = Field(ge=1)
    ledger_event_count: int = Field(ge=1)
    ledger_head_hash: str
    valid: bool = True

    @field_validator("run_id")
    @classmethod
    def _safe_run_id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("ledger_head_hash")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return validate_sha256(value)


def load_evidence_packet(path: Path) -> EvidencePacket:
    """Load one exact normalized packet with a bounded local read."""
    return _load_model(path, EvidencePacket)


def load_offline_fixture(path: Path) -> OfflinePilotFixture:
    """Load deterministic analyst output for a network-free run."""
    return _load_model(path, OfflinePilotFixture)


def run_pilot(
    *,
    packet: EvidencePacket,
    output_root: Path,
    run_id: str,
    mode: PilotMode = PilotMode.OFFLINE,
    fixture: OfflinePilotFixture | None = None,
    llm: StructuredGenerator | None = None,
    model: str | None = None,
    max_hypotheses: int = 12,
    budget_policy: BudgetPolicy | None = None,
    charter: PilotCharter | None = None,
    now: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> PilotExecution:
    """Execute or idempotently load one immutable pilot.

    ``mode=offline`` requires a fixture and never constructs an LLM.  ``mode=llm``
    rejects fixtures and invokes the existing bounded generator/falsifier seams.
    No branch can perform interviews, outreach, advertising, deposits, purchases,
    or investor contact.
    """
    safe_run_id = validate_identifier(run_id, field="run_id")
    if not 1 <= max_hypotheses <= 25:
        raise ValueError("max_hypotheses must be between 1 and 25")
    if mode is PilotMode.OFFLINE and fixture is None:
        raise PilotError("offline mode requires an explicit fixture")
    if mode is PilotMode.LLM and fixture is not None:
        raise PilotError("LLM mode does not accept an offline fixture")
    if fixture is not None and len(fixture.candidates) > max_hypotheses:
        raise PilotError(
            f"fixture contains {len(fixture.candidates)} candidates; limit is {max_hypotheses}"
        )

    selected_policy = budget_policy or BudgetPolicy()
    selected_charter = charter or PilotCharter()
    selected_clock = clock or _utc_now
    generated_at = now or _clock_time(selected_clock)
    _require_aware(generated_at)

    root = output_root.expanduser().resolve()
    run_relative_root = f"runs/{safe_run_id}"
    run_root = safe_join(root, run_relative_root)
    configuration = PilotConfiguration(
        mode=mode,
        fixture=fixture,
        model=model,
        max_hypotheses=max_hypotheses,
        budget_policy=selected_policy,
        charter=selected_charter,
    )
    packet_bytes = canonical_json(packet)
    packet_hash = sha256_bytes(packet_bytes)
    configuration_bytes = canonical_json(configuration)
    configuration_hash = sha256_bytes(configuration_bytes)

    existing_manifest = safe_join(root, f"{run_relative_root}/manifest.json")
    if existing_manifest.is_file():
        existing = load_pilot_execution(output_root=root, run_id=safe_run_id)
        if (
            existing.manifest.packet_sha256 != packet_hash
            or existing.manifest.configuration_sha256 != configuration_hash
        ):
            raise PilotIntegrityError(
                f"run {safe_run_id!r} already exists with different immutable inputs"
            )
        if isinstance(existing, LegacyPilotExecution):
            raise PilotIntegrityError(
                f"run {safe_run_id!r} predates the current execution schema and is load-only"
            )
        return existing
    if run_root.exists() and any(run_root.iterdir()):
        raise PilotIntegrityError(
            f"run {safe_run_id!r} is partial; preserve it and choose a new run id"
        )

    run_root.mkdir(parents=True, exist_ok=True)
    global_stop = KillSwitch(root)
    run_stop = KillSwitch(run_root)
    _assert_clear(global_stop, run_stop)
    local_write = authorize_action(ExternalAction.WRITE_LOCAL_ARTIFACT)
    if not local_write.allowed:  # pragma: no cover - defensive policy invariant
        raise PilotError(local_write.reason)

    implementation_manifest, implementation_source_tar = build_implementation_bundle()
    implementation_manifest_bytes = canonical_json(implementation_manifest)
    ledger = Ledger(root / "ledger.jsonl")
    snapshots = SnapshotStore(root / "snapshots")
    pointers: list[PilotArtifactPointer] = []

    def persist(kind: str, relative_path: str, content: bytes) -> PilotArtifactPointer:
        _assert_clear(global_stop, run_stop)
        pointer = _persist_artifact(
            kind=kind,
            relative_path=relative_path,
            content=content,
            output_root=root,
            run_id=safe_run_id,
            snapshots=snapshots,
            ledger=ledger,
            recorded_at=_clock_time(selected_clock),
        )
        pointers.append(pointer)
        return pointer

    packet_pointer = persist(
        "evidence_packet",
        f"{run_relative_root}/artifacts/evidence-packet.json",
        packet_bytes,
    )
    configuration_pointer = persist(
        "pilot_configuration",
        f"{run_relative_root}/artifacts/pilot-configuration.json",
        configuration_bytes,
    )
    implementation_source_pointer = persist(
        "implementation_source_tar",
        f"{run_relative_root}/artifacts/implementation-source.tar",
        implementation_source_tar,
    )
    implementation_manifest_pointer = persist(
        "implementation_manifest",
        f"{run_relative_root}/artifacts/implementation-manifest.json",
        implementation_manifest_bytes,
    )
    if (
        packet_pointer.sha256 != packet_hash
        or configuration_pointer.sha256 != configuration_hash
        or implementation_source_pointer.sha256 != implementation_manifest.source_tar_sha256
    ):  # pragma: no cover - defensive invariant
        raise PilotIntegrityError("persisted immutable inputs differ from their computed hashes")

    budget = BudgetGuard(selected_policy)
    generation_assignment = make_content_id(
        "assignment",
        {"run_id": safe_run_id, "role": "researcher", "packet_id": packet.packet_id},
        digest_length=32,
    )
    candidate_input_hashes: dict[str, tuple[str, str, str]] = {}
    classification_outcome_hashes: dict[str, tuple[str | None, str | None]] = {}
    classification_scopes = classification_scope_measurements(packet)

    def persist_candidate_inputs(
        index: int,
        candidate: CandidateHypothesis,
        classification_assignment_id: str,
    ) -> None:
        suffix = f"{index:03d}-{candidate.opportunity_id}"
        classification_pointer = persist(
            "classification_input",
            f"{run_relative_root}/artifacts/classification-input-{suffix}.json",
            canonical_json(
                classification_input_payload_v2(
                    candidate=candidate,
                    scopes=classification_scopes,
                    assignment_id=classification_assignment_id,
                )
            ),
        )
        scoped_packet_pointer = persist(
            "falsification_evidence_packet",
            f"{run_relative_root}/artifacts/falsification-evidence-packet-{suffix}.json",
            canonical_json(scope_falsification_packet(candidate, packet=packet)),
        )
        if candidate.opportunity_id in candidate_input_hashes:
            raise PilotIntegrityError("candidate classifier inputs were persisted more than once")
        candidate_input_hashes[candidate.opportunity_id] = (
            classification_assignment_id,
            classification_pointer.sha256,
            scoped_packet_pointer.sha256,
        )

    def persist_classification_response(
        index: int,
        candidate: CandidateHypothesis,
        response: ClassificationComparisonReview,
    ) -> None:
        if (
            response.opportunity_id != candidate.opportunity_id
            or candidate.opportunity_id in classification_outcome_hashes
        ):
            raise PilotIntegrityError("classification response does not bind exactly once")
        suffix = f"{index:03d}-{candidate.opportunity_id}"
        pointer = persist(
            "classification_comparison",
            f"{run_relative_root}/artifacts/classification-comparison-{suffix}.json",
            canonical_json(response),
        )
        classification_outcome_hashes[candidate.opportunity_id] = (pointer.sha256, None)

    def persist_classification_failure(
        index: int,
        candidate: CandidateHypothesis,
        failure: ClassificationReviewFailure,
    ) -> None:
        if (
            failure.opportunity_id != candidate.opportunity_id
            or candidate.opportunity_id in classification_outcome_hashes
        ):
            raise PilotIntegrityError("classification failure does not bind exactly once")
        suffix = f"{index:03d}-{candidate.opportunity_id}"
        pointer = persist(
            "classification_review_failure",
            f"{run_relative_root}/artifacts/classification-review-failure-{suffix}.json",
            canonical_json(failure),
        )
        classification_outcome_hashes[candidate.opportunity_id] = (None, pointer.sha256)

    classification_reviews: tuple[ClassificationReview | None, ...]
    if mode is PilotMode.OFFLINE:
        assert fixture is not None
        (
            candidates,
            classification_assignments,
            classification_reviews,
            reports,
            classification_comparisons,
        ) = _offline_outputs(
            fixture,
            packet=packet,
            assignment_id=generation_assignment,
            created_at=generated_at,
            run_id=safe_run_id,
            clock=selected_clock,
        )
        for index, (candidate, assignment, comparison) in enumerate(
            zip(
                candidates,
                classification_assignments,
                classification_comparisons,
                strict=True,
            ),
            start=1,
        ):
            persist_candidate_inputs(index, candidate, assignment.assignment_id)
            persist_classification_response(index, candidate, comparison)
        classification_failures: tuple[ClassificationReviewFailure | None, ...] = (None,) * len(
            candidates
        )
        failures: tuple[FalsificationFailure | None, ...] = (None,) * len(candidates)
    else:
        _assert_clear(global_stop, run_stop)
        model_action = authorize_action(ExternalAction.MODEL_CALL)
        if not model_action.allowed:  # pragma: no cover - defensive policy invariant
            raise PilotError(model_action.reason)
        generator: StructuredGenerator = llm if llm is not None else LLM(model=model)
        candidates = generate_hypotheses(
            packet=packet,
            assignment_id=generation_assignment,
            created_at=generated_at,
            llm=generator,
            max_hypotheses=max_hypotheses,
            model=model,
            budget=budget,
            kill_switch=run_stop,
        )
        classification_assignments_list: list[ClassificationAssignment] = []
        classification_reviews_list: list[ClassificationReview | None] = []
        classification_failures_list: list[ClassificationReviewFailure | None] = []
        reports_list: list[FalsificationReport | None] = []
        failures_list: list[FalsificationFailure | None] = []
        for index, candidate in enumerate(candidates, start=1):
            _assert_clear(global_stop, run_stop)
            classification_assignment_id = make_content_id(
                "assignment",
                {
                    "run_id": safe_run_id,
                    "role": "classification-reviewer",
                    "candidate": candidate.opportunity_id,
                    "ordinal": index,
                },
                digest_length=32,
            )
            scope_ref = f"naics22-{candidate.naics_codes[0]}-scope"
            persist_candidate_inputs(
                index,
                candidate,
                classification_assignment_id,
            )
            classification_started = _clock_time(selected_clock)
            try:
                classification_execution = review_classification_comparative(
                    candidate=candidate,
                    packet=packet,
                    assignment_id=classification_assignment_id,
                    llm=generator,
                    model=model,
                    budget=budget,
                    kill_switch=run_stop,
                )
            except (LLMError, ValidationError, ValueError) as exc:
                classification_completed = _clock_time(selected_clock)
                classification_failure = _quarantine_classification_failure(
                    candidate=candidate,
                    assignment_id=classification_assignment_id,
                    scope_measurement_ref=scope_ref,
                    error=exc,
                    quarantined_at=classification_completed,
                )
                classification_reviews_list.append(None)
                classification_failures_list.append(classification_failure)
                persist_classification_failure(
                    index,
                    candidate,
                    classification_failure,
                )
            else:
                classification_completed = _clock_time(selected_clock)
                classification_reviews_list.append(classification_execution.review)
                classification_failures_list.append(None)
                persist_classification_response(
                    index,
                    candidate,
                    classification_execution.comparison,
                )
            classification_assignments_list.append(
                ClassificationAssignment(
                    assignment_id=classification_assignment_id,
                    opportunity_id=candidate.opportunity_id,
                    naics_code=candidate.naics_codes[0],
                    scope_measurement_ref=scope_ref,
                    actor_model=model or type(generator).__name__,
                    visible_fields=_COMPARATIVE_CLASSIFICATION_VISIBLE_FIELDS,
                    started_at=classification_started,
                    completed_at=classification_completed,
                )
            )
            accepted_review = classification_reviews_list[-1]
            if (
                accepted_review is None
                or accepted_review.outcome is not ClassificationReviewOutcome.FIT
            ):
                reports_list.append(None)
                failures_list.append(None)
                continue

            _assert_clear(global_stop, run_stop)
            assignment_id = make_content_id(
                "assignment",
                {
                    "run_id": safe_run_id,
                    "role": "falsifier",
                    "candidate": candidate.opportunity_id,
                    "ordinal": index,
                },
                digest_length=32,
            )
            try:
                report = falsify_hypothesis(
                    candidate=candidate,
                    packet=packet,
                    assignment_id=assignment_id,
                    llm=generator,
                    model=model,
                    budget=budget,
                    kill_switch=run_stop,
                )
            except (LLMError, ValidationError, ValueError) as exc:
                reports_list.append(None)
                failures_list.append(
                    _quarantine_falsification_failure(
                        candidate=candidate,
                        assignment_id=assignment_id,
                        error=exc,
                        quarantined_at=_clock_time(selected_clock),
                    )
                )
            else:
                reports_list.append(report)
                failures_list.append(None)
        reports = tuple(reports_list)
        failures = tuple(failures_list)
        classification_assignments = tuple(classification_assignments_list)
        classification_reviews = tuple(classification_reviews_list)
        classification_failures = tuple(classification_failures_list)

    if not candidates:
        raise PilotError("the analyst produced no hypotheses")

    candidate_results = tuple(
        _candidate_result(
            candidate,
            classification_assignment=classification_assignment,
            classification_review=classification_review,
            classification_failure=classification_failure,
            report=report,
            failure=failure,
            packet=packet,
            charter=selected_charter,
        )
        for (
            candidate,
            classification_assignment,
            classification_review,
            classification_failure,
            report,
            failure,
        ) in zip(
            candidates,
            classification_assignments,
            classification_reviews,
            classification_failures,
            reports,
            failures,
            strict=True,
        )
    )
    if set(candidate_input_hashes) != set(classification_outcome_hashes):
        raise PilotIntegrityError("every candidate requires one input and classification outcome")
    candidate_input_provenance = tuple(
        CandidateInputProvenanceV2(
            opportunity_id=candidate.opportunity_id,
            classification_assignment_id=input_hashes[0],
            classification_input_sha256=input_hashes[1],
            falsification_evidence_packet_sha256=input_hashes[2],
            classification_response_sha256=outcome_hashes[0],
            classification_failure_sha256=outcome_hashes[1],
        )
        for candidate in candidates
        for input_hashes in (candidate_input_hashes[candidate.opportunity_id],)
        for outcome_hashes in (classification_outcome_hashes[candidate.opportunity_id],)
    )
    run_provenance = PilotRunProvenanceV2(
        configuration_sha256=configuration_hash,
        evidence_packet_sha256=packet_hash,
        implementation_manifest_sha256=implementation_manifest_pointer.sha256,
        implementation_source_tar_sha256=implementation_source_pointer.sha256,
        candidate_inputs=candidate_input_provenance,
    )
    run_provenance_pointer = persist(
        "run_provenance",
        f"{run_relative_root}/artifacts/run-provenance.json",
        canonical_json(run_provenance),
    )
    result = PilotRunResult(
        run_id=safe_run_id,
        mode=mode,
        packet_id=packet.packet_id,
        information_cutoff=packet.as_of,
        packet_sha256=packet_hash,
        configuration_sha256=configuration_hash,
        run_provenance_sha256=run_provenance_pointer.sha256,
        candidates=candidate_results,
        budget_policy=selected_policy,
        budget_usage=budget.usage,
        action_policy=tuple(authorize_action(action) for action in ExternalAction),
        generated_at=generated_at,
    )

    from app.venture.reporting import render_pilot_report

    report_markdown = render_pilot_report(result)
    for index, candidate_result in enumerate(candidate_results, start=1):
        suffix = f"{index:03d}-{candidate_result.hypothesis.opportunity_id}"
        persist(
            "candidate",
            f"{run_relative_root}/artifacts/candidate-{suffix}.json",
            canonical_json(candidate_result.hypothesis),
        )
        persist(
            "classification_assignment",
            f"{run_relative_root}/artifacts/classification-assignment-{suffix}.json",
            canonical_json(candidate_result.classification_assignment),
        )
        if candidate_result.classification_review is not None:
            persist(
                "classification_review",
                f"{run_relative_root}/artifacts/classification-review-{suffix}.json",
                canonical_json(candidate_result.classification_review),
            )
        # Classification failures are persisted immediately when quarantined so their
        # exact hash can be bound by run provenance before the result is written.
        if candidate_result.falsification is not None:
            persist(
                "falsification",
                f"{run_relative_root}/artifacts/falsification-{suffix}.json",
                canonical_json(candidate_result.falsification),
            )
        if candidate_result.falsification_failure is not None:
            persist(
                "falsification_failure",
                f"{run_relative_root}/artifacts/falsification-failure-{suffix}.json",
                canonical_json(candidate_result.falsification_failure),
            )
        persist(
            "gate_evaluation",
            f"{run_relative_root}/artifacts/gates-{suffix}.json",
            canonical_json(candidate_result.gates),
        )
    result_pointer = persist(
        "result",
        f"{run_relative_root}/result.json",
        canonical_json(result),
    )
    report_pointer = persist(
        "report",
        f"{run_relative_root}/report.md",
        report_markdown.encode("utf-8"),
    )

    manifest = PilotManifest(
        run_id=safe_run_id,
        mode=mode,
        packet_id=packet.packet_id,
        packet_sha256=packet_hash,
        configuration_sha256=configuration_hash,
        run_provenance_sha256=run_provenance_pointer.sha256,
        artifacts=tuple(pointers),
    )
    manifest_pointer = _persist_artifact(
        kind="manifest",
        relative_path=f"{run_relative_root}/manifest.json",
        content=canonical_json(manifest),
        output_root=root,
        run_id=safe_run_id,
        snapshots=snapshots,
        ledger=ledger,
        recorded_at=_clock_time(selected_clock),
    )
    completion = ledger.append(
        "pilot.run.completed",
        {
            "manifest_sha256": manifest_pointer.sha256,
            "result_sha256": result_pointer.sha256,
            "report_sha256": report_pointer.sha256,
            "decision_counts": _decision_counts(candidate_results),
            "falsification_failure_count": sum(
                item.falsification_failure is not None for item in candidate_results
            ),
            "classification_review_outcomes": _classification_review_counts(candidate_results),
            "classification_review_failure_count": sum(
                item.classification_review_failure is not None for item in candidate_results
            ),
            "master_score_computed": False,
            "external_commercial_actions_performed": False,
        },
        aggregate_id=safe_run_id,
        actor_id=_ACTOR_ID,
        recorded_at=_clock_time(selected_clock),
    )
    validation = ledger.validate()
    return PilotExecution(
        manifest=manifest,
        result=result,
        completion_event_id=completion.event_id,
        ledger_head_hash=validation.head_hash,
        report_relative_path=report_pointer.run_relative_path,
    )


def load_pilot_execution(
    *,
    output_root: Path,
    run_id: str,
) -> PilotExecution | LegacyPilotExecution:
    """Load and verify a completed immutable pilot."""
    safe_run_id = validate_identifier(run_id, field="run_id")
    root = output_root.expanduser().resolve()
    run_root = f"runs/{safe_run_id}"
    manifest = _load_model(safe_join(root, f"{run_root}/manifest.json"), PilotManifest)
    result = _load_pilot_result(safe_join(root, f"{run_root}/result.json"))
    if manifest.run_id != safe_run_id or result.run_id != safe_run_id:
        raise PilotIntegrityError("run id does not match its immutable artifacts")
    if (
        manifest.packet_sha256 != result.packet_sha256
        or manifest.configuration_sha256 != result.configuration_sha256
    ):
        raise PilotIntegrityError("manifest and result input hashes disagree")
    if isinstance(result, PilotRunResult) and (
        manifest.run_provenance_sha256 != result.run_provenance_sha256
    ):
        raise PilotIntegrityError("manifest and result provenance hashes disagree")

    verification = verify_pilot_run(output_root=root, run_id=safe_run_id)
    ledger = Ledger(root / "ledger.jsonl")
    completion = _completion_event(ledger, safe_run_id)
    report_pointer = next(
        (item for item in manifest.artifacts if item.kind == "report"),
        None,
    )
    if report_pointer is None:
        raise PilotIntegrityError("manifest has no report artifact")
    if isinstance(result, LegacyPilotRunResult):
        return LegacyPilotExecution(
            manifest=manifest,
            result=result,
            completion_event_id=completion.event_id,
            ledger_head_hash=verification.ledger_head_hash,
            report_relative_path=report_pointer.run_relative_path,
        )
    return PilotExecution(
        manifest=manifest,
        result=result,
        completion_event_id=completion.event_id,
        ledger_head_hash=verification.ledger_head_hash,
        report_relative_path=report_pointer.run_relative_path,
    )


def verify_pilot_run(*, output_root: Path, run_id: str) -> PilotVerification:
    """Verify every file hash, snapshot, event pointer, and ledger link."""
    safe_run_id = validate_identifier(run_id, field="run_id")
    root = output_root.expanduser().resolve()
    manifest_relative_path = f"runs/{safe_run_id}/manifest.json"
    manifest_path = safe_join(root, manifest_relative_path)
    manifest = _load_model(manifest_path, PilotManifest)
    if manifest.run_id != safe_run_id:
        raise PilotIntegrityError("manifest run id does not match the requested run")

    snapshots = SnapshotStore(root / "snapshots")
    ledger = Ledger(root / "ledger.jsonl")
    events = {event.event_id: event for event in ledger.events()}
    artifact_contents: dict[str, bytes] = {}
    for pointer in manifest.artifacts:
        path = safe_join(root, pointer.run_relative_path)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise PilotIntegrityError(
                f"artifact {pointer.run_relative_path!r} is unreadable"
            ) from exc
        if len(content) != pointer.size_bytes or sha256_bytes(content) != pointer.sha256:
            raise PilotIntegrityError(
                f"artifact {pointer.run_relative_path!r} no longer matches its manifest"
            )
        snapshot = snapshots.read(pointer.sha256)
        if snapshot != content:
            raise PilotIntegrityError(
                f"snapshot for {pointer.run_relative_path!r} differs from the run artifact"
            )
        event = events.get(pointer.ledger_event_id)
        if event is None or not _event_matches_pointer(event, pointer, safe_run_id):
            raise PilotIntegrityError(
                f"ledger event for {pointer.run_relative_path!r} is absent or inconsistent"
            )
        artifact_contents[pointer.run_relative_path] = content

    manifest_content = manifest_path.read_bytes()
    manifest_hash = sha256_bytes(manifest_content)
    manifest_events = tuple(
        event
        for event in events.values()
        if event.aggregate_id == safe_run_id
        and event.event_type == "pilot.artifact"
        and event.payload.get("artifact_kind") == "manifest"
    )
    if len(manifest_events) != 1:
        raise PilotIntegrityError(
            f"run {safe_run_id!r} must have exactly one manifest event; "
            f"found {len(manifest_events)}"
        )
    manifest_event = manifest_events[0]
    if (
        manifest_event.payload.get("sha256") != manifest_hash
        or manifest_event.payload.get("size_bytes") != len(manifest_content)
        or manifest_event.payload.get("run_relative_path") != manifest_relative_path
        or manifest_event.payload.get("snapshot_relative_path")
        != snapshots.relative_path_for(manifest_hash)
        or snapshots.read(manifest_hash) != manifest_content
    ):
        raise PilotIntegrityError("manifest no longer matches its snapshot and ledger event")

    artifact_event_ids = {
        event.event_id
        for event in events.values()
        if event.aggregate_id == safe_run_id and event.event_type == "pilot.artifact"
    }
    expected_event_ids = {
        *(pointer.ledger_event_id for pointer in manifest.artifacts),
        manifest_event.event_id,
    }
    if artifact_event_ids != expected_event_ids:
        raise PilotIntegrityError("manifest does not enumerate every run artifact event")

    try:
        _verify_semantic_artifacts(manifest, artifact_contents)
    except PilotIntegrityError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotIntegrityError(f"semantic artifact verification failed: {exc}") from exc

    completion = _completion_event(ledger, safe_run_id)
    result_pointer = _single_pointer(manifest, "result")
    report_pointer = _single_pointer(manifest, "report")
    if (
        completion.sequence <= manifest_event.sequence
        or completion.payload.get("manifest_sha256") != manifest_hash
        or completion.payload.get("result_sha256") != result_pointer.sha256
        or completion.payload.get("report_sha256") != report_pointer.sha256
    ):
        raise PilotIntegrityError("completion event does not match the immutable run outputs")
    validation = ledger.validate()
    return PilotVerification(
        run_id=safe_run_id,
        artifact_count=len(manifest.artifacts),
        ledger_event_count=validation.event_count,
        ledger_head_hash=validation.head_hash,
    )


def report_path(*, output_root: Path, run_id: str) -> Path:
    """Return the verified report path for a completed pilot."""
    execution = load_pilot_execution(output_root=output_root, run_id=run_id)
    return safe_join(output_root.expanduser().resolve(), execution.report_relative_path)


def _load_model[T: FrozenModel](path: Path, model: type[T]) -> T:
    resolved = path.expanduser().resolve()
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise PilotError(f"could not inspect {resolved}: {exc}") from exc
    if size > _MAX_INPUT_BYTES:
        raise PilotError(f"{resolved} exceeds the {_MAX_INPUT_BYTES:,}-byte local input limit")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return model.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise PilotError(f"invalid {model.__name__} at {resolved}: {exc}") from exc


def _load_pilot_result(path: Path) -> PilotRunResult | LegacyPilotRunResult:
    """Load current typed results or a byte-preserving legacy top-level envelope."""
    resolved = path.expanduser().resolve()
    try:
        size = resolved.stat().st_size
        if size > _MAX_INPUT_BYTES:
            raise PilotError(f"{resolved} exceeds the {_MAX_INPUT_BYTES:,}-byte local input limit")
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("pilot result must be a JSON object")
        schema_version = payload.get("schema_version")
        model: type[PilotRunResult] | type[LegacyPilotRunResult]
        model = (
            PilotRunResult
            if schema_version in {"venture-pilot-v4", "venture-pilot-v5", "venture-pilot-v6"}
            else LegacyPilotRunResult
        )
        return model.model_validate(payload)
    except PilotError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise PilotError(f"invalid pilot result at {resolved}: {exc}") from exc


def _verify_semantic_artifacts(
    manifest: PilotManifest,
    artifact_contents: dict[str, bytes],
) -> None:
    """Verify cross-artifact meaning after byte, snapshot, and ledger integrity."""
    result_pointer = _single_pointer(manifest, "result")
    result_content = _artifact_content(result_pointer, artifact_contents)
    result_envelope = _decode_canonical_object(result_content, label="result")
    result_schema = result_envelope.get("schema_version")
    semantic_schema_pairs = {
        "venture-pilot-manifest-v5": "venture-pilot-v5",
        "venture-pilot-manifest-v6": "venture-pilot-v6",
    }
    expected_result_schema = semantic_schema_pairs.get(manifest.schema_version)
    semantic_result_schemas = frozenset(semantic_schema_pairs.values())
    if expected_result_schema is None and result_schema in semantic_result_schemas:
        raise PilotIntegrityError("manifest and result schema generations disagree")
    if expected_result_schema is not None and result_schema != expected_result_schema:
        raise PilotIntegrityError("manifest and result schema generations disagree")
    if expected_result_schema is None:
        return

    allowed_kinds = {
        "candidate",
        "classification_assignment",
        "classification_input",
        "classification_review",
        "classification_review_failure",
        "evidence_packet",
        "falsification",
        "falsification_evidence_packet",
        "falsification_failure",
        "gate_evaluation",
        "implementation_manifest",
        "implementation_source_tar",
        "pilot_configuration",
        "report",
        "result",
        "run_provenance",
    }
    if expected_result_schema == "venture-pilot-v6":
        allowed_kinds.add("classification_comparison")
    unexpected_kinds = {pointer.kind for pointer in manifest.artifacts}.difference(allowed_kinds)
    if unexpected_kinds:
        raise PilotIntegrityError(
            f"semantic manifest contains unexpected artifact kinds: {sorted(unexpected_kinds)}"
        )

    result = _decode_canonical_model(
        result_content,
        PilotRunResult,
        label="result",
    )
    packet_pointer = _single_pointer(manifest, "evidence_packet")
    configuration_pointer = _single_pointer(manifest, "pilot_configuration")
    provenance_pointer = _single_pointer(manifest, "run_provenance")
    implementation_manifest_pointer = _single_pointer(
        manifest,
        "implementation_manifest",
    )
    implementation_source_pointer = _single_pointer(
        manifest,
        "implementation_source_tar",
    )

    packet = _decode_canonical_model(
        _artifact_content(packet_pointer, artifact_contents),
        EvidencePacket,
        label="evidence packet",
    )
    configuration = _decode_canonical_model(
        _artifact_content(configuration_pointer, artifact_contents),
        PilotConfiguration,
        label="pilot configuration",
    )
    provenance_content = _artifact_content(provenance_pointer, artifact_contents)
    provenance: PilotRunProvenance | PilotRunProvenanceV2
    if expected_result_schema == "venture-pilot-v5":
        provenance = _decode_canonical_model(
            provenance_content,
            PilotRunProvenance,
            label="run provenance",
        )
    else:
        provenance = _decode_canonical_model(
            provenance_content,
            PilotRunProvenanceV2,
            label="run provenance",
        )
    implementation_manifest = _decode_canonical_model(
        _artifact_content(implementation_manifest_pointer, artifact_contents),
        ImplementationManifest,
        label="implementation manifest",
    )
    implementation_source = _artifact_content(
        implementation_source_pointer,
        artifact_contents,
    )

    if (
        packet_pointer.sha256 != result.packet_sha256
        or packet_pointer.sha256 != manifest.packet_sha256
        or packet_pointer.sha256 != provenance.evidence_packet_sha256
    ):
        raise PilotIntegrityError("evidence-packet hash binding is inconsistent")
    if (
        configuration_pointer.sha256 != result.configuration_sha256
        or configuration_pointer.sha256 != manifest.configuration_sha256
        or configuration_pointer.sha256 != provenance.configuration_sha256
    ):
        raise PilotIntegrityError("pilot-configuration hash binding is inconsistent")
    if (
        provenance_pointer.sha256 != result.run_provenance_sha256
        or provenance_pointer.sha256 != manifest.run_provenance_sha256
    ):
        raise PilotIntegrityError("run-provenance hash binding is inconsistent")
    if (
        implementation_manifest_pointer.sha256 != provenance.implementation_manifest_sha256
        or implementation_source_pointer.sha256 != provenance.implementation_source_tar_sha256
        or implementation_source_pointer.sha256 != implementation_manifest.source_tar_sha256
    ):
        raise PilotIntegrityError("implementation-source hash binding is inconsistent")
    try:
        verify_implementation_bundle(implementation_manifest, implementation_source)
    except ValueError as exc:
        raise PilotIntegrityError(str(exc)) from exc

    if (
        manifest.run_id != result.run_id
        or manifest.mode is not result.mode
        or manifest.packet_id != result.packet_id
        or packet.packet_id != result.packet_id
        or packet.as_of != result.information_cutoff
        or configuration.mode is not result.mode
        or configuration.budget_policy != result.budget_policy
        or len(result.candidates) > configuration.max_hypotheses
    ):
        raise PilotIntegrityError("result does not match its manifest, packet, or configuration")

    _verify_candidate_children(manifest, artifact_contents, result)
    _verify_candidate_inputs(
        manifest,
        artifact_contents,
        result,
        packet,
        provenance,
        result_schema=expected_result_schema,
    )

    from app.venture.reporting import render_pilot_report

    report_pointer = _single_pointer(manifest, "report")
    if _artifact_content(report_pointer, artifact_contents) != render_pilot_report(result).encode(
        "utf-8"
    ):
        raise PilotIntegrityError("report does not exactly rerender from the result")


def _verify_candidate_children(
    manifest: PilotManifest,
    artifact_contents: dict[str, bytes],
    result: PilotRunResult,
) -> None:
    _require_exact_model_artifacts(
        manifest,
        artifact_contents,
        "candidate",
        CandidateHypothesis,
        tuple(item.hypothesis for item in result.candidates),
    )
    _require_exact_model_artifacts(
        manifest,
        artifact_contents,
        "classification_assignment",
        ClassificationAssignment,
        tuple(item.classification_assignment for item in result.candidates),
    )
    _require_exact_model_artifacts(
        manifest,
        artifact_contents,
        "classification_review",
        ClassificationReview,
        tuple(
            item.classification_review
            for item in result.candidates
            if item.classification_review is not None
        ),
    )
    _require_exact_model_artifacts(
        manifest,
        artifact_contents,
        "classification_review_failure",
        ClassificationReviewFailure,
        tuple(
            item.classification_review_failure
            for item in result.candidates
            if item.classification_review_failure is not None
        ),
    )
    _require_exact_model_artifacts(
        manifest,
        artifact_contents,
        "falsification",
        FalsificationReport,
        tuple(item.falsification for item in result.candidates if item.falsification is not None),
    )
    _require_exact_model_artifacts(
        manifest,
        artifact_contents,
        "falsification_failure",
        FalsificationFailure,
        tuple(
            item.falsification_failure
            for item in result.candidates
            if item.falsification_failure is not None
        ),
    )
    _require_exact_model_artifacts(
        manifest,
        artifact_contents,
        "gate_evaluation",
        GateEvaluation,
        tuple(item.gates for item in result.candidates),
    )


def _verify_candidate_inputs(
    manifest: PilotManifest,
    artifact_contents: dict[str, bytes],
    result: PilotRunResult,
    packet: EvidencePacket,
    provenance: PilotRunProvenance | PilotRunProvenanceV2,
    *,
    result_schema: str,
) -> None:
    if result_schema == "venture-pilot-v5":
        if not isinstance(provenance, PilotRunProvenance):
            raise PilotIntegrityError("v5 result requires v1 run provenance")
        _verify_candidate_inputs_v1(
            manifest,
            artifact_contents,
            result,
            packet,
            provenance,
        )
        return
    if result_schema == "venture-pilot-v6":
        if not isinstance(provenance, PilotRunProvenanceV2):
            raise PilotIntegrityError("v6 result requires v2 run provenance")
        _verify_candidate_inputs_v2(
            manifest,
            artifact_contents,
            result,
            packet,
            provenance,
        )
        return
    raise PilotIntegrityError(f"unsupported candidate input schema {result_schema!r}")


def _verify_candidate_inputs_v1(
    manifest: PilotManifest,
    artifact_contents: dict[str, bytes],
    result: PilotRunResult,
    packet: EvidencePacket,
    provenance: PilotRunProvenance,
) -> None:
    classification_pointers = tuple(
        pointer for pointer in manifest.artifacts if pointer.kind == "classification_input"
    )
    falsification_packet_pointers = tuple(
        pointer for pointer in manifest.artifacts if pointer.kind == "falsification_evidence_packet"
    )
    candidate_count = len(result.candidates)
    if (
        len(provenance.candidate_inputs) != candidate_count
        or len(classification_pointers) != candidate_count
        or len(falsification_packet_pointers) != candidate_count
    ):
        raise PilotIntegrityError(
            "candidate input provenance must have one classification and falsification "
            "packet per result candidate"
        )

    packet_measurements = {item.measurement_id: item for item in packet.measurements}
    for candidate_result, input_provenance, classification_pointer, scoped_pointer in zip(
        result.candidates,
        provenance.candidate_inputs,
        classification_pointers,
        falsification_packet_pointers,
        strict=True,
    ):
        candidate = candidate_result.hypothesis
        assignment = candidate_result.classification_assignment
        scope = classification_scope_measurement(candidate, packet=packet)
        expected_classification_input, expected_visible_fields = (
            _classification_input_for_result_schema(
                candidate=candidate,
                packet=packet,
                assignment_id=assignment.assignment_id,
                result_schema="venture-pilot-v5",
            )
        )
        actual_classification_input = _decode_canonical_object(
            _artifact_content(classification_pointer, artifact_contents),
            label="classification input",
        )
        expected_scoped_packet = scope_falsification_packet(candidate, packet=packet)
        actual_scoped_packet = _decode_canonical_model(
            _artifact_content(scoped_pointer, artifact_contents),
            EvidencePacket,
            label="falsification evidence packet",
        )
        expected_evidence = tuple(
            packet_measurements[measurement_id] for measurement_id in candidate.evidence_refs
        )
        if (
            input_provenance.opportunity_id != candidate.opportunity_id
            or input_provenance.classification_assignment_id != assignment.assignment_id
            or input_provenance.classification_input_sha256 != classification_pointer.sha256
            or input_provenance.falsification_evidence_packet_sha256 != scoped_pointer.sha256
            or actual_classification_input != expected_classification_input
            or actual_scoped_packet != expected_scoped_packet
            or assignment.scope_measurement_ref != scope.measurement_id
            or assignment.visible_fields != expected_visible_fields
            or candidate_result.evidence != expected_evidence
        ):
            raise PilotIntegrityError("candidate input provenance does not bind to its result")


def _verify_candidate_inputs_v2(
    manifest: PilotManifest,
    artifact_contents: dict[str, bytes],
    result: PilotRunResult,
    packet: EvidencePacket,
    provenance: PilotRunProvenanceV2,
) -> None:
    candidate_count = len(result.candidates)
    if len(provenance.candidate_inputs) != candidate_count:
        raise PilotIntegrityError("v2 candidate provenance cardinality differs from the result")

    input_pointers = tuple(
        pointer for pointer in manifest.artifacts if pointer.kind == "classification_input"
    )
    scoped_pointers = tuple(
        pointer for pointer in manifest.artifacts if pointer.kind == "falsification_evidence_packet"
    )
    response_pointers = tuple(
        pointer for pointer in manifest.artifacts if pointer.kind == "classification_comparison"
    )
    failure_pointers = tuple(
        pointer for pointer in manifest.artifacts if pointer.kind == "classification_review_failure"
    )
    accepted_count = sum(item.classification_review is not None for item in result.candidates)
    failure_count = candidate_count - accepted_count
    if (
        len(input_pointers) != candidate_count
        or len(scoped_pointers) != candidate_count
        or len(response_pointers) != accepted_count
        or len(failure_pointers) != failure_count
    ):
        raise PilotIntegrityError(
            "v2 classifier artifact cardinality differs from accepted and failed results"
        )

    result_by_opportunity = {item.hypothesis.opportunity_id: item for item in result.candidates}
    binding_by_opportunity = {item.opportunity_id: item for item in provenance.candidate_inputs}
    if set(result_by_opportunity) != set(binding_by_opportunity):
        raise PilotIntegrityError("v2 provenance opportunity bindings differ from the result")

    packet_measurements = {item.measurement_id: item for item in packet.measurements}
    comparison_scopes = classification_scope_measurements(packet)
    expected_compared_refs = tuple(scope.measurement_id for scope in comparison_scopes)
    eligible_codes = frozenset(
        ref.removeprefix("naics22-").removesuffix("-scope") for ref in expected_compared_refs
    )
    for opportunity_id, candidate_result in result_by_opportunity.items():
        binding = binding_by_opportunity[opportunity_id]
        candidate = candidate_result.hypothesis
        assignment = candidate_result.classification_assignment
        scope = classification_scope_measurement(candidate, packet=packet)
        input_pointer = _candidate_artifact_pointer(
            input_pointers,
            opportunity_id,
            kind="classification_input",
        )
        scoped_pointer = _candidate_artifact_pointer(
            scoped_pointers,
            opportunity_id,
            kind="falsification_evidence_packet",
        )
        expected_input, expected_visible_fields = _classification_input_for_result_schema(
            candidate=candidate,
            packet=packet,
            assignment_id=assignment.assignment_id,
            result_schema="venture-pilot-v6",
        )
        actual_input = _decode_canonical_object(
            _artifact_content(input_pointer, artifact_contents),
            label="classification input",
        )
        expected_scoped_packet = scope_falsification_packet(candidate, packet=packet)
        actual_scoped_packet = _decode_canonical_model(
            _artifact_content(scoped_pointer, artifact_contents),
            EvidencePacket,
            label="falsification evidence packet",
        )
        expected_evidence = tuple(
            packet_measurements[measurement_id] for measurement_id in candidate.evidence_refs
        )
        if (
            binding.classification_assignment_id != assignment.assignment_id
            or binding.classification_input_sha256 != input_pointer.sha256
            or binding.falsification_evidence_packet_sha256 != scoped_pointer.sha256
            or actual_input != expected_input
            or actual_scoped_packet != expected_scoped_packet
            or assignment.scope_measurement_ref != scope.measurement_id
            or assignment.visible_fields != expected_visible_fields
            or candidate_result.evidence != expected_evidence
        ):
            raise PilotIntegrityError("v2 candidate input provenance does not bind to its result")

        if candidate_result.classification_review is not None:
            response_hash = binding.classification_response_sha256
            if response_hash is None or binding.classification_failure_sha256 is not None:
                raise PilotIntegrityError("accepted classification lacks an exact response hash")
            response_pointer = _candidate_artifact_pointer(
                response_pointers,
                opportunity_id,
                kind="classification_comparison",
            )
            if response_pointer.sha256 != response_hash:
                raise PilotIntegrityError(
                    "classification response hash differs from its candidate artifact"
                )
            comparison = _decode_canonical_model(
                _artifact_content(response_pointer, artifact_contents),
                ClassificationComparisonReview,
                label="classification comparison",
            )
            if (
                comparison.legacy_review() != candidate_result.classification_review
                or comparison.opportunity_id != opportunity_id
                or comparison.assignment_id != assignment.assignment_id
                or comparison.naics_code != candidate.naics_codes[0]
                or comparison.scope_measurement_ref != scope.measurement_id
                or comparison.compared_scope_refs != expected_compared_refs
                or not set(comparison.plausible_naics_codes).issubset(eligible_codes)
            ):
                raise PilotIntegrityError(
                    "classification comparison does not bind to its exact review input"
                )
            continue

        failure_hash = binding.classification_failure_sha256
        if failure_hash is None or binding.classification_response_sha256 is not None:
            raise PilotIntegrityError("quarantined classification lacks an exact failure hash")
        failure_pointer = _candidate_artifact_pointer(
            failure_pointers,
            opportunity_id,
            kind="classification_review_failure",
        )
        if failure_pointer.sha256 != failure_hash:
            raise PilotIntegrityError(
                "classification failure hash differs from its candidate artifact"
            )
        actual_failure = _decode_canonical_model(
            _artifact_content(failure_pointer, artifact_contents),
            ClassificationReviewFailure,
            label="classification review failure",
        )
        if actual_failure != candidate_result.classification_review_failure:
            raise PilotIntegrityError("classification failure hash does not bind to its result")


def _candidate_artifact_pointer(
    pointers: tuple[PilotArtifactPointer, ...],
    opportunity_id: str,
    *,
    kind: str,
) -> PilotArtifactPointer:
    suffix = f"-{opportunity_id}.json"
    matches = tuple(
        pointer for pointer in pointers if Path(pointer.run_relative_path).name.endswith(suffix)
    )
    if len(matches) != 1:
        raise PilotIntegrityError(
            f"candidate {opportunity_id!r} must have exactly one {kind} artifact"
        )
    return matches[0]


def _classification_input_for_result_schema(
    *,
    candidate: CandidateHypothesis,
    packet: EvidencePacket,
    assignment_id: str,
    result_schema: str,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Reconstruct the exact classifier input generation bound to a result schema."""
    if result_schema == "venture-pilot-v5":
        scope = classification_scope_measurement(candidate, packet=packet)
        return (
            classification_input_payload(
                candidate=candidate,
                scope=scope,
                assignment_id=assignment_id,
            ),
            _LEGACY_CLASSIFICATION_VISIBLE_FIELDS,
        )
    if result_schema == "venture-pilot-v6":
        return (
            classification_input_payload_v2(
                candidate=candidate,
                scopes=classification_scope_measurements(packet),
                assignment_id=assignment_id,
            ),
            _COMPARATIVE_CLASSIFICATION_VISIBLE_FIELDS,
        )
    raise PilotIntegrityError(
        f"unsupported semantic classifier-input generation for {result_schema!r}"
    )


def _require_exact_model_artifacts[T: FrozenModel](
    manifest: PilotManifest,
    artifact_contents: dict[str, bytes],
    kind: str,
    model: type[T],
    expected: tuple[T, ...],
) -> None:
    pointers = tuple(pointer for pointer in manifest.artifacts if pointer.kind == kind)
    actual = tuple(
        _decode_canonical_model(
            _artifact_content(pointer, artifact_contents),
            model,
            label=kind.replace("_", " "),
        )
        for pointer in pointers
    )
    if actual != expected:
        raise PilotIntegrityError(
            f"{kind.replace('_', ' ')} child artifacts do not exactly match the result"
        )


def _decode_canonical_model[T: FrozenModel](
    content: bytes,
    model: type[T],
    *,
    label: str,
) -> T:
    try:
        value = model.model_validate_json(content)
    except ValidationError as exc:
        raise PilotIntegrityError(f"{label} artifact is invalid: {exc}") from exc
    if canonical_json(value) != content:
        raise PilotIntegrityError(f"{label} artifact is not exact canonical JSON")
    return value


def _decode_canonical_object(content: bytes, *, label: str) -> dict[str, object]:
    try:
        value = TypeAdapter(dict[str, object]).validate_json(content)
    except ValidationError as exc:
        raise PilotIntegrityError(f"{label} artifact is not a JSON object: {exc}") from exc
    try:
        is_canonical = canonical_json(value) == content
    except (TypeError, ValueError) as exc:
        raise PilotIntegrityError(f"{label} artifact is not canonical JSON: {exc}") from exc
    if not is_canonical:
        raise PilotIntegrityError(f"{label} artifact is not exact canonical JSON")
    return value


def _artifact_content(
    pointer: PilotArtifactPointer,
    artifact_contents: dict[str, bytes],
) -> bytes:
    try:
        return artifact_contents[pointer.run_relative_path]
    except KeyError as exc:  # pragma: no cover - internal verification invariant
        raise PilotIntegrityError(
            f"verified content is missing for {pointer.run_relative_path!r}"
        ) from exc


def _offline_outputs(
    fixture: OfflinePilotFixture,
    *,
    packet: EvidencePacket,
    assignment_id: str,
    created_at: datetime,
    run_id: str,
    clock: Callable[[], datetime],
) -> tuple[
    tuple[CandidateHypothesis, ...],
    tuple[ClassificationAssignment, ...],
    tuple[ClassificationReview, ...],
    tuple[FalsificationReport | None, ...],
    tuple[ClassificationComparisonReview, ...],
]:
    candidates: list[CandidateHypothesis] = []
    assignments: list[ClassificationAssignment] = []
    reviews: list[ClassificationReview] = []
    reports: list[FalsificationReport | None] = []
    comparisons: list[ClassificationComparisonReview] = []
    comparison_scopes = classification_scope_measurements(packet)
    compared_scope_refs = tuple(scope.measurement_id for scope in comparison_scopes)
    eligible_codes = frozenset(
        scope_ref.removeprefix("naics22-").removesuffix("-scope")
        for scope_ref in compared_scope_refs
    )
    for index, item in enumerate(fixture.candidates, start=1):
        candidate = materialize_hypothesis(
            item.hypothesis,
            packet=packet,
            assignment_id=assignment_id,
            created_at=created_at,
        )
        candidates.append(candidate)
        scope = classification_scope_measurement(candidate, packet=packet)
        classification_assignment_id = make_content_id(
            "assignment",
            {
                "run_id": run_id,
                "role": "offline-classification-reviewer",
                "candidate": candidate.opportunity_id,
                "ordinal": index,
            },
            digest_length=32,
        )
        started_at = _clock_time(clock)
        outside_comparison = set(item.classification_review.plausible_naics_codes).difference(
            eligible_codes
        )
        if outside_comparison:
            raise ValueError(
                "offline classification names plausible codes outside its comparison set: "
                f"{sorted(outside_comparison)}"
            )
        comparison = ClassificationComparisonReview(
            opportunity_id=candidate.opportunity_id,
            assignment_id=classification_assignment_id,
            naics_code=candidate.naics_codes[0],
            scope_measurement_ref=scope.measurement_id,
            compared_scope_refs=compared_scope_refs,
            **item.classification_review.model_dump(exclude={"plausible_naics_codes"}),
            plausible_naics_codes=item.classification_review.plausible_naics_codes,
        )
        execution = ClassificationReviewExecution(
            comparison=comparison,
            review=comparison.legacy_review(),
        )
        review = execution.review
        completed_at = _clock_time(clock)
        validate_classification_review(
            review,
            candidate=candidate,
            packet=packet,
            assignment_id=classification_assignment_id,
        )
        assignments.append(
            ClassificationAssignment(
                assignment_id=classification_assignment_id,
                opportunity_id=candidate.opportunity_id,
                naics_code=candidate.naics_codes[0],
                scope_measurement_ref=scope.measurement_id,
                actor_model="offline-fixture",
                visible_fields=_COMPARATIVE_CLASSIFICATION_VISIBLE_FIELDS,
                started_at=started_at,
                completed_at=completed_at,
            )
        )
        reviews.append(review)
        comparisons.append(execution.comparison)
        if review.outcome is not ClassificationReviewOutcome.FIT or item.falsification is None:
            reports.append(None)
            continue
        falsifier_assignment = make_content_id(
            "assignment",
            {
                "run_id": run_id,
                "role": "offline-falsifier",
                "candidate": candidate.opportunity_id,
                "ordinal": index,
            },
            digest_length=32,
        )
        report = FalsificationReport(
            opportunity_id=candidate.opportunity_id,
            assignment_id=falsifier_assignment,
            **item.falsification.model_dump(),
        )
        validate_falsification_refs(report, packet=packet, candidate=candidate)
        reports.append(report)
    return (
        tuple(candidates),
        tuple(assignments),
        tuple(reviews),
        tuple(reports),
        tuple(comparisons),
    )


def _candidate_result(
    candidate: CandidateHypothesis,
    *,
    classification_assignment: ClassificationAssignment,
    classification_review: ClassificationReview | None,
    classification_failure: ClassificationReviewFailure | None,
    report: FalsificationReport | None,
    failure: FalsificationFailure | None,
    packet: EvidencePacket,
    charter: PilotCharter,
) -> PilotCandidateResult:
    measurement_by_id = {item.measurement_id: item for item in packet.measurements}
    evidence = tuple(measurement_by_id[item] for item in candidate.evidence_refs)
    source_families = len({item.source_family for item in evidence})
    assessment = ClaimEvidenceAssessment(
        claim_id=make_content_id(
            "claim",
            {"opportunity_id": candidate.opportunity_id, "type": "critical-thesis"},
            digest_length=32,
        ),
        independent_source_families=source_families,
        # PacketMeasurement deliberately does not pretend it knows whether a
        # source is a primary behavioral or administrative record.
        has_primary_administrative_or_behavioral=None,
    )
    findings = (
        {finding.dimension: finding for finding in report.findings} if report is not None else {}
    )

    def substantive_falsification_result(
        dimension: FalsificationDimension,
    ) -> bool | None:
        finding = findings.get(dimension)
        if finding is None:
            return None
        return finding.outcome is FalsificationOutcome.NO_CONTRADICTION_FOUND

    gates = evaluate_gates(
        GateContext(
            charter_customer_defined=True,
            charter_payer_defined=True,
            charter_geography_defined=True,
            founder_constraints_defined=charter.founder_constraints_defined,
            capital_scenario_defined=True,
            outcome_horizons_defined=charter.outcome_horizons_defined,
            critical_claim_evidence=(assessment,),
            substitution_checked=substantive_falsification_result(
                FalsificationDimension.SUBSTITUTION
            ),
            latent_competition_checked=substantive_falsification_result(
                FalsificationDimension.LATENT_COMPETITION
            ),
            regulatory_low_supply_checked=substantive_falsification_result(
                FalsificationDimension.REGULATORY_LOW_SUPPLY
            ),
            mandated_vs_contestable_spend_checked=substantive_falsification_result(
                FalsificationDimension.MANDATED_VS_CONTESTABLE_SPEND
            ),
            demand_without_wtp_checked=substantive_falsification_result(
                FalsificationDimension.DEMAND_WITHOUT_WTP
            ),
            stressed_economics_checked=substantive_falsification_result(
                FalsificationDimension.STRESSED_ECONOMICS
            ),
            # Model-reported disqualifiers are allegations until independently
            # verified, so they cannot trigger the core's explicit KILL predicates.
            illegal=None,
            unfinanceable=None,
            negative_stressed_contribution=None,
            scenario=candidate.scenario,
            scenario_metrics_complete=None,
            pareto_ready=None,
        )
    )
    allegations: list[str] = []
    if report is not None:
        if report.explicit_illegality_found:
            allegations.append("illegality")
        if report.explicit_unfinanceable_found:
            allegations.append("unfinanceable capital requirement")
        if report.explicit_negative_stressed_contribution_found:
            allegations.append("negative stressed contribution")
    return PilotCandidateResult(
        hypothesis=candidate,
        classification_assignment=classification_assignment,
        classification_review=classification_review,
        classification_review_failure=classification_failure,
        falsification=report,
        falsification_failure=failure,
        evidence=evidence,
        gates=gates,
        unverified_disqualifier_allegations=tuple(allegations),
    )


def _quarantine_falsification_failure(
    *,
    candidate: CandidateHypothesis,
    assignment_id: str,
    error: LLMError | ValidationError | ValueError,
    quarantined_at: datetime,
) -> FalsificationFailure:
    message = " ".join(str(error).split())[:1_000]
    if not message:
        message = "falsification failed without an error message"
    lowered = message.casefold()
    if "cites unknown measurements" in lowered or "unknown measurement" in lowered:
        kind = FalsificationFailureKind.EVIDENCE_REFERENCE
    elif isinstance(error, ValidationError) or (
        isinstance(error, LLMError)
        and any(
            marker in lowered
            for marker in (
                "json schema",
                "structured reply",
                "does not validate",
                "validation error",
            )
        )
    ):
        kind = FalsificationFailureKind.SCHEMA
    elif isinstance(error, LLMError):
        kind = FalsificationFailureKind.TRANSPORT
    else:
        kind = FalsificationFailureKind.VALIDATION
    return FalsificationFailure(
        opportunity_id=candidate.opportunity_id,
        assignment_id=assignment_id,
        kind=kind,
        error_type=type(error).__name__,
        message=message,
        quarantined_at=quarantined_at,
    )


def _quarantine_classification_failure(
    *,
    candidate: CandidateHypothesis,
    assignment_id: str,
    scope_measurement_ref: str,
    error: LLMError | ValidationError | ValueError,
    quarantined_at: datetime,
) -> ClassificationReviewFailure:
    message = " ".join(str(error).split())[:1_000]
    if not message:
        message = "classification review failed without an error message"
    lowered = message.casefold()
    if any(
        marker in lowered
        for marker in (
            "scope measurement",
            "scope_measurement_ref",
            "naics_code",
            "opportunity_id",
            "assignment_id",
            "binding fields",
        )
    ):
        kind = ClassificationReviewFailureKind.EVIDENCE_REFERENCE
    elif isinstance(error, ValidationError) or (
        isinstance(error, LLMError)
        and any(
            marker in lowered
            for marker in (
                "json schema",
                "structured reply",
                "does not validate",
                "validation error",
            )
        )
    ):
        kind = ClassificationReviewFailureKind.SCHEMA
    elif isinstance(error, LLMError):
        kind = ClassificationReviewFailureKind.TRANSPORT
    else:
        kind = ClassificationReviewFailureKind.VALIDATION
    return ClassificationReviewFailure(
        opportunity_id=candidate.opportunity_id,
        assignment_id=assignment_id,
        naics_code=candidate.naics_codes[0],
        scope_measurement_ref=scope_measurement_ref,
        kind=kind,
        error_type=type(error).__name__,
        message=message,
        quarantined_at=quarantined_at,
    )


def _persist_artifact(
    *,
    kind: str,
    relative_path: str,
    content: bytes,
    output_root: Path,
    run_id: str,
    snapshots: SnapshotStore,
    ledger: Ledger,
    recorded_at: datetime,
) -> PilotArtifactPointer:
    normalized = validate_relative_path(relative_path)
    path = safe_join(output_root, normalized)
    _write_once(path, content)
    snapshot = snapshots.put(content)
    event = ledger.append(
        "pilot.artifact",
        {
            "artifact_kind": kind,
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
            "run_relative_path": normalized,
            "snapshot_relative_path": snapshot.relative_path,
        },
        aggregate_id=run_id,
        actor_id=_ACTOR_ID,
        recorded_at=recorded_at,
    )
    return PilotArtifactPointer(
        kind=kind,
        sha256=snapshot.sha256,
        size_bytes=snapshot.size_bytes,
        run_relative_path=normalized,
        snapshot_relative_path=snapshot.relative_path,
        ledger_event_id=event.event_id,
    )


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise PilotIntegrityError(f"existing immutable artifact {path} is unreadable") from exc
        if existing != content:
            raise PilotIntegrityError(
                f"immutable artifact {path} already has different content"
            ) from None
        return
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o444)
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def _event_matches_pointer(
    event: LedgerEvent,
    pointer: PilotArtifactPointer,
    run_id: str,
) -> bool:
    return (
        event.event_type == "pilot.artifact"
        and event.aggregate_id == run_id
        and event.payload.get("artifact_kind") == pointer.kind
        and event.payload.get("sha256") == pointer.sha256
        and event.payload.get("size_bytes") == pointer.size_bytes
        and event.payload.get("run_relative_path") == pointer.run_relative_path
        and event.payload.get("snapshot_relative_path") == pointer.snapshot_relative_path
    )


def _completion_event(ledger: Ledger, run_id: str) -> LedgerEvent:
    matches = tuple(
        event
        for event in ledger.events()
        if event.aggregate_id == run_id and event.event_type == "pilot.run.completed"
    )
    if len(matches) != 1:
        raise PilotIntegrityError(
            f"run {run_id!r} must have exactly one completion event; found {len(matches)}"
        )
    return matches[0]


def _single_pointer(manifest: PilotManifest, kind: str) -> PilotArtifactPointer:
    matches = tuple(pointer for pointer in manifest.artifacts if pointer.kind == kind)
    if len(matches) != 1:
        raise PilotIntegrityError(
            f"manifest must contain exactly one {kind!r} artifact; found {len(matches)}"
        )
    return matches[0]


def _decision_counts(
    candidates: tuple[PilotCandidateResult, ...],
) -> dict[str, int]:
    return {
        decision.value: sum(item.gates.decision is decision for item in candidates)
        for decision in GateDecision
    }


def _classification_review_counts(
    candidates: tuple[PilotCandidateResult, ...],
) -> dict[str, int]:
    return {
        outcome.value: sum(
            item.classification_review is not None and item.classification_review.outcome is outcome
            for item in candidates
        )
        for outcome in ClassificationReviewOutcome
    }


def _assert_clear(*switches: KillSwitch) -> None:
    for switch in switches:
        switch.assert_clear()


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must include a timezone")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _clock_time(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    _require_aware(value)
    return value.astimezone(UTC)


__all__ = [
    "CandidateInputProvenance",
    "CandidateInputProvenanceV2",
    "ClassificationAssignment",
    "ClassificationReviewFailure",
    "ClassificationReviewFailureKind",
    "FalsificationFailure",
    "FalsificationFailureKind",
    "LegacyPilotExecution",
    "LegacyPilotRunResult",
    "OfflineCandidateFixture",
    "OfflineClassificationReview",
    "OfflineFalsification",
    "OfflinePilotFixture",
    "PilotArtifactPointer",
    "PilotCandidateResult",
    "PilotCharter",
    "PilotConfiguration",
    "PilotError",
    "PilotExecution",
    "PilotIntegrityError",
    "PilotManifest",
    "PilotMode",
    "PilotRunProvenance",
    "PilotRunProvenanceV2",
    "PilotRunResult",
    "PilotVerification",
    "load_evidence_packet",
    "load_offline_fixture",
    "load_pilot_execution",
    "report_path",
    "run_pilot",
    "verify_pilot_run",
]
