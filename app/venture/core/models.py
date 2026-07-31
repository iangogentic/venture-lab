"""Strict, immutable contracts for evidence-governed venture research.

These records intentionally keep observations, claims, verification work,
experiments, and predictions separate.  Unknown values stay ``None``; no model
silently converts absence into zero.
"""

import math
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from app.venture.core.ids import make_content_id, validate_identifier

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Estimate = bool | int | float | str
TemporalValue = date | datetime


class FrozenModel(BaseModel):
    """Shared strict envelope for durable governance records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        str_strip_whitespace=True,
    )


class MeasurementDomain(StrEnum):
    """Natural domain measured by an evidence record."""

    DEMAND = "demand"
    SUPPLY = "supply"
    CAPACITY = "capacity"
    ECONOMICS = "economics"
    SURVIVAL = "survival"
    DEMOGRAPHICS = "demographics"
    REGULATION = "regulation"
    INVESTOR_FIT = "investor_fit"


class EvidenceKind(StrEnum):
    """How directly the source records the phenomenon."""

    ADMINISTRATIVE_RECORD = "administrative_record"
    TRANSACTION_RECORD = "transaction_record"
    SURVEY_ESTIMATE = "survey_estimate"
    PLATFORM_MEASUREMENT = "platform_measurement"
    PROXY = "proxy"
    MODELED_ESTIMATE = "modeled_estimate"
    HYPOTHESIS = "hypothesis"


class GeographySystem(StrEnum):
    """Supported geographic coding systems."""

    FIPS = "FIPS"
    CBSA = "CBSA"
    ZCTA = "ZCTA"
    LAT_LON = "lat_lon"
    JURISDICTION = "jurisdiction"


class IndustrySystem(StrEnum):
    """Supported industry and occupation coding systems."""

    NAICS = "NAICS"
    SOC = "SOC"
    PSC = "PSC"
    CUSTOM = "custom"


class EntityType(StrEnum):
    """Entity granularity for an entity-scoped measurement."""

    ESTABLISHMENT = "establishment"
    FIRM = "firm"
    FACILITY = "facility"
    FUND = "fund"
    OFFERING = "offering"


class MeasurementSource(FrozenModel):
    """Reproducible source locator for a measurement."""

    source_id: str
    publisher: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    source_record_id: str | None = None
    release_version: str = Field(min_length=1)
    source_url: HttpUrl
    query: str | None = None
    raw_content_hash: Sha256
    source_family: str | None = Field(
        default=None,
        description=(
            "Independence family. Falls back to source_id when the publisher has "
            "not declared a broader common lineage."
        ),
    )

    @field_validator("source_id")
    @classmethod
    def _safe_source_id(cls, value: str) -> str:
        return validate_identifier(value, field="source_id")

    @field_validator(
        "publisher",
        "dataset",
        "release_version",
        "source_record_id",
        "query",
        "source_family",
    )
    @classmethod
    def _optional_text_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value:
            raise ValueError("text must not be blank")
        return value

    @property
    def independence_family(self) -> str:
        """Lineage key used for source-independence tests."""
        return self.source_family or self.source_id


class MeasurementTime(FrozenModel):
    """Observation, publication, retrieval, and validity windows."""

    observed_start: TemporalValue
    observed_end: TemporalValue
    published_at: datetime | None = None
    retrieved_at: datetime
    valid_from: TemporalValue | None = None
    valid_to: TemporalValue | None = None

    @field_validator("observed_start", "observed_end", "valid_from", "valid_to")
    @classmethod
    def _aware_temporal_values(cls, value: TemporalValue | None) -> TemporalValue | None:
        if isinstance(value, datetime):
            _require_aware(value)
        return value

    @field_validator("published_at", "retrieved_at")
    @classmethod
    def _aware_datetimes(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            _require_aware(value)
        return value

    @model_validator(mode="after")
    def _ordered_windows(self) -> Self:
        if _temporal_key(self.observed_end) < _temporal_key(self.observed_start):
            raise ValueError("observed_end cannot be before observed_start")
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and _temporal_key(self.valid_to) < _temporal_key(self.valid_from)
        ):
            raise ValueError("valid_to cannot be before valid_from")
        return self


class Geography(FrozenModel):
    """Geographic scope in a named coding system and vintage."""

    system: GeographySystem
    level: str = Field(min_length=1)
    code: str = Field(min_length=1)
    vintage: str = Field(min_length=1)


class Industry(FrozenModel):
    """Optional industry or occupation scope."""

    system: IndustrySystem
    code: str = Field(min_length=1)
    vintage: str = Field(min_length=1)


class Entity(FrozenModel):
    """Optional entity scope."""

    type: EntityType
    id_system: str = Field(min_length=1)
    id: str = Field(min_length=1)


class MeasurementValue(FrozenModel):
    """Estimate in natural units, with optional denominator and price basis."""

    estimate: Estimate
    unit: str = Field(min_length=1)
    denominator: str | None = None
    currency: str | None = Field(default=None, pattern=r"^USD$")
    price_year: int | None = Field(default=None, ge=1900)

    @field_validator("estimate")
    @classmethod
    def _finite_or_nonblank(cls, value: Estimate) -> Estimate:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric estimates must be finite")
        if isinstance(value, str) and not value:
            raise ValueError("string estimates must not be blank")
        return value

    @model_validator(mode="after")
    def _price_basis_is_consistent(self) -> Self:
        if self.price_year is not None and self.currency is None:
            raise ValueError("price_year requires currency")
        return self


class MeasurementUncertainty(FrozenModel):
    """Uncertainty and disclosure metadata supplied by the source."""

    margin_of_error: NonNegativeFloat | None = None
    standard_error: NonNegativeFloat | None = None
    coefficient_of_variation: NonNegativeFloat | None = None
    sample_size: int | None = Field(default=None, ge=0)
    suppressed: bool | None = None
    imputed: bool | None = None
    censored: bool | None = None


class Measurement(FrozenModel):
    """One auditable observation matching the public TypeScript contract."""

    id: str
    domain: MeasurementDomain
    metric: str = Field(min_length=1)
    evidence_kind: EvidenceKind
    source: MeasurementSource
    time: MeasurementTime
    geography: Geography
    industry: Industry | None = None
    entity: Entity | None = None
    universe: str = Field(min_length=1)
    value: MeasurementValue
    uncertainty: MeasurementUncertainty | None = None
    quality_flags: tuple[str, ...] = ()

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        return validate_identifier(value, field="measurement id")

    @field_validator("quality_flags")
    @classmethod
    def _unique_quality_flags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonblank(value, field="quality_flags")

    @classmethod
    def make_id(cls, content: object) -> str:
        """Derive a stable measurement id from caller-selected identity content."""
        return make_content_id("measurement", content)

    @property
    def source_family(self) -> str:
        """Convenient access to the source-independence lineage."""
        return self.source.independence_family


class DerivedSignal(FrozenModel):
    """A derived metric with complete input and transform provenance."""

    id: str
    domain: MeasurementDomain
    metric: str = Field(min_length=1)
    input_measurement_ids: tuple[str, ...] = Field(min_length=1)
    formula: str = Field(min_length=1)
    formula_version: str = Field(min_length=1)
    assumptions: tuple[str, ...]
    uncertainty_propagation: str = Field(min_length=1)
    value: MeasurementValue
    created_at: datetime

    @field_validator("id", *("input_measurement_ids",))
    @classmethod
    def _safe_ids(cls, value: Any) -> Any:
        if isinstance(value, str):
            return validate_identifier(value, field="derived signal id")
        return tuple(validate_identifier(item, field="input measurement id") for item in value)

    @field_validator("input_measurement_ids")
    @classmethod
    def _unique_inputs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("input_measurement_ids must be unique")
        return value

    @field_validator("assumptions")
    @classmethod
    def _nonblank_assumptions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonblank(value, field="assumptions")

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        return _require_aware(value)


class OpportunityStage(StrEnum):
    """Lifecycle of a falsifiable opportunity thesis."""

    DISCOVERY = "discovery"
    VERIFICATION = "verification"
    FALSIFICATION = "falsification"
    COMPARISON = "comparison"
    FIELD_TEST = "field_test"
    VALIDATED = "validated"
    KILLED = "killed"


class OpportunityThesis(FrozenModel):
    """A versioned thesis; corrections create a new superseding record."""

    opportunity_id: str
    version: int = Field(ge=1)
    cohort_id: str
    title: str = Field(min_length=1)
    customer: str = Field(min_length=1)
    payer: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    business_model: str = Field(min_length=1)
    geography: tuple[Geography, ...] = Field(min_length=1)
    stage: OpportunityStage
    created_at: datetime
    supersedes: str | None = None

    @field_validator("opportunity_id", "cohort_id", "supersedes")
    @classmethod
    def _safe_ids(cls, value: str | None) -> str | None:
        return None if value is None else validate_identifier(value)

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        return _require_aware(value)

    @model_validator(mode="after")
    def _valid_supersession(self) -> Self:
        if self.supersedes == self.opportunity_id:
            raise ValueError("an opportunity cannot supersede itself")
        if self.supersedes is not None and self.version == 1:
            raise ValueError("a superseding opportunity must have version greater than one")
        return self


class ClaimType(StrEnum):
    """Decision-relevant claim category."""

    DEMAND = "demand"
    SUPPLY = "supply"
    PRICING = "pricing"
    COST = "cost"
    MARGIN = "margin"
    REGULATION = "regulation"
    DISTRIBUTION = "distribution"
    CAPITAL = "capital"
    OPERATOR_FIT = "operator_fit"


class ClaimCriticality(StrEnum):
    """Whether a claim can independently change the decision."""

    CRITICAL = "critical"
    SUPPORTING = "supporting"


class ClaimStatus(StrEnum):
    """Current evidence status of a claim."""

    PROPOSED = "proposed"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"
    STALE = "stale"


class EffectivePeriod(FrozenModel):
    """Period for which a claim is intended to hold."""

    start: TemporalValue
    end: TemporalValue

    @field_validator("start", "end")
    @classmethod
    def _aware_temporal_values(cls, value: TemporalValue) -> TemporalValue:
        if isinstance(value, datetime):
            _require_aware(value)
        return value

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if _temporal_key(self.end) < _temporal_key(self.start):
            raise ValueError("effective period end cannot precede start")
        return self


class Claim(FrozenModel):
    """A falsifiable statement kept separate from its evidence links."""

    claim_id: str
    opportunity_id: str
    statement: str = Field(min_length=1)
    claim_type: ClaimType
    criticality: ClaimCriticality
    value: Estimate | None = None
    unit: str | None = None
    geography: Geography
    effective_period: EffectivePeriod
    status: ClaimStatus
    created_by_assignment: str

    @field_validator("claim_id", "opportunity_id", "created_by_assignment")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("value")
    @classmethod
    def _finite_value(cls, value: Estimate | None) -> Estimate | None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("claim value must be finite")
        return value

    @model_validator(mode="after")
    def _unit_matches_value(self) -> Self:
        if self.value is not None and self.unit is None:
            raise ValueError("a quantitative claim value requires a unit")
        if self.value is None and self.unit is not None:
            raise ValueError("unit cannot be set without a claim value")
        return self


class EvidenceRelationship(StrEnum):
    """How a measurement bears on a claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXT = "context"


class ClaimEvidence(FrozenModel):
    """Traceable link from a claim to a raw or derived measurement."""

    claim_id: str
    measurement_id: str
    relationship: EvidenceRelationship
    transform_id: str

    @field_validator("claim_id", "measurement_id", "transform_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)


class AssignmentRole(StrEnum):
    """Role-separated work assignment."""

    RESEARCHER = "researcher"
    FALSIFIER = "falsifier"
    VERIFIER = "verifier"
    OUTCOME_ADJUDICATOR = "outcome_adjudicator"


class Assignment(FrozenModel):
    """Auditable blind assignment and its prompt fingerprint."""

    assignment_id: str
    role: AssignmentRole
    blind_packet_id: str
    actor_model: str = Field(min_length=1)
    prompt_sha256: Sha256
    visible_fields: tuple[str, ...]
    started_at: datetime
    completed_at: datetime | None = None

    @field_validator("assignment_id", "blind_packet_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("visible_fields")
    @classmethod
    def _visible_fields_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonblank(value, field="visible_fields")

    @field_validator("started_at", "completed_at")
    @classmethod
    def _aware_times(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            _require_aware(value)
        return value

    @model_validator(mode="after")
    def _completion_after_start(self) -> Self:
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be before started_at")
        return self


class VerificationResult(StrEnum):
    """Outcome of an independent verification assignment."""

    PASS = "pass"
    FAIL = "fail"
    NEEDS_WORK = "needs_work"


class VerificationChecks(FrozenModel):
    """Reproducibility checks performed by a verifier."""

    snapshot_hash_matches: bool
    locator_resolves: bool
    value_reproduced: bool
    unit_matches: bool
    geography_matches: bool
    time_scope_matches: bool
    source_independence_valid: bool
    cutoff_valid: bool
    transform_code_sha256: Sha256 | None = None

    @property
    def all_required_pass(self) -> bool:
        """Whether every universally applicable boolean check passed."""
        return all(
            (
                self.snapshot_hash_matches,
                self.locator_resolves,
                self.value_reproduced,
                self.unit_matches,
                self.geography_matches,
                self.time_scope_matches,
                self.source_independence_valid,
                self.cutoff_valid,
            )
        )


class Verification(FrozenModel):
    """Independent verification of one claim."""

    verification_id: str
    claim_id: str
    verifier_assignment_id: str
    result: VerificationResult
    checks: VerificationChecks

    @field_validator("verification_id", "claim_id", "verifier_assignment_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _passing_verification_has_passing_checks(self) -> Self:
        if self.result is VerificationResult.PASS and not self.checks.all_required_pass:
            raise ValueError("a passing verification requires every boolean check to pass")
        return self


class ThresholdOperator(StrEnum):
    """Comparison used by a preregistered field-test threshold."""

    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="


class ExperimentThreshold(FrozenModel):
    """Natural-unit threshold fixed before observations begin."""

    operator: ThresholdOperator
    value: float

    @field_validator("value")
    @classmethod
    def _finite_threshold(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("threshold value must be finite")
        return value

    def accepts(self, actual: float) -> bool:
        """Apply the preregistered comparison."""
        if not math.isfinite(actual):
            raise ValueError("actual value must be finite")
        match self.operator:
            case ThresholdOperator.GT:
                return actual > self.value
            case ThresholdOperator.GTE:
                return actual >= self.value
            case ThresholdOperator.LT:
                return actual < self.value
            case ThresholdOperator.LTE:
                return actual <= self.value
            case ThresholdOperator.EQ:
                return actual == self.value


class GateDecision(StrEnum):
    """Deterministic workflow decision."""

    PASS = "pass"
    HOLD = "hold"
    KILL = "kill"


class ExperimentStatus(StrEnum):
    """Lifecycle of a preregistered experiment."""

    REGISTERED = "registered"
    RUNNING = "running"
    COMPLETED = "completed"
    INVALIDATED = "invalidated"


class FieldExperiment(FrozenModel):
    """Preregistered behavioral test of the riskiest assumption."""

    experiment_id: str
    opportunity_id: str
    preregistered_at: datetime
    hypothesis: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    threshold: ExperimentThreshold
    population: str = Field(min_length=1)
    sample_target: int = Field(gt=0)
    duration_days: int = Field(gt=0)
    budget_limit: float = Field(ge=0.0)
    exclusions: tuple[str, ...]
    stopping_rule: str = Field(min_length=1)
    decision_rule: GateDecision
    protocol_sha256: Sha256
    first_observation_at: datetime | None = None
    status: ExperimentStatus

    @field_validator("experiment_id", "opportunity_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("preregistered_at", "first_observation_at")
    @classmethod
    def _aware_times(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            _require_aware(value)
        return value

    @field_validator("exclusions")
    @classmethod
    def _unique_exclusions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonblank(value, field="exclusions")

    @model_validator(mode="after")
    def _registration_precedes_observation(self) -> Self:
        if (
            self.first_observation_at is not None
            and self.first_observation_at < self.preregistered_at
        ):
            raise ValueError("first_observation_at cannot precede preregistration")
        if self.status is ExperimentStatus.REGISTERED and self.first_observation_at is not None:
            raise ValueError("a registered experiment cannot already have an observation")
        return self


class PredictionType(StrEnum):
    """Shape of a prospective prediction."""

    BINARY = "binary"
    NUMERIC_INTERVAL = "numeric_interval"


class PredictionStatus(StrEnum):
    """Lifecycle of a prospective prediction."""

    OPEN = "open"
    MATURED = "matured"
    ADJUDICATED = "adjudicated"


class PredictionInterval(FrozenModel):
    """Ordered numeric prediction interval."""

    lower: float
    upper: float

    @model_validator(mode="after")
    def _ordered_and_finite(self) -> Self:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("prediction interval bounds must be finite")
        if self.lower > self.upper:
            raise ValueError("prediction interval lower cannot exceed upper")
        return self


class Prediction(FrozenModel):
    """Prospective forecast frozen at an information cutoff."""

    prediction_id: str
    opportunity_id: str
    target_definition: str = Field(min_length=1)
    prediction_type: PredictionType
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    interval: PredictionInterval | None = None
    as_of: datetime
    matures_at: datetime
    information_cutoff: datetime
    evidence_set_sha256: Sha256
    status: PredictionStatus

    @field_validator("prediction_id", "opportunity_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("as_of", "matures_at", "information_cutoff")
    @classmethod
    def _aware_times(cls, value: datetime) -> datetime:
        return _require_aware(value)

    @model_validator(mode="after")
    def _shape_and_order(self) -> Self:
        if self.information_cutoff > self.as_of:
            raise ValueError("information_cutoff cannot be after as_of")
        if self.matures_at <= self.as_of:
            raise ValueError("matures_at must be after as_of")
        if self.prediction_type is PredictionType.BINARY:
            if self.probability is None or self.interval is not None:
                raise ValueError("binary predictions require probability and forbid interval")
        elif self.interval is None or self.probability is not None:
            raise ValueError("numeric_interval predictions require interval and forbid probability")
        return self


class ClaimEvidenceAssessment(FrozenModel):
    """Evidence-strength facts for one claim, consumed by G1."""

    claim_id: str
    independent_source_families: int | None
    has_primary_administrative_or_behavioral: bool | None

    @field_validator("claim_id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        return validate_identifier(value)


_PRIMARY_EVIDENCE: frozenset[EvidenceKind] = frozenset(
    {
        EvidenceKind.ADMINISTRATIVE_RECORD,
        EvidenceKind.TRANSACTION_RECORD,
        EvidenceKind.PLATFORM_MEASUREMENT,
    }
)


def assess_claim_evidence(
    claim: Claim,
    links: list[ClaimEvidence] | tuple[ClaimEvidence, ...],
    measurements: list[Measurement] | tuple[Measurement, ...],
) -> ClaimEvidenceAssessment:
    """Count independent supporting lineages and primary behavioral records."""
    by_id = {measurement.id: measurement for measurement in measurements}
    support_ids = {
        link.measurement_id
        for link in links
        if link.claim_id == claim.claim_id and link.relationship is EvidenceRelationship.SUPPORTS
    }
    supporting = [by_id[item] for item in support_ids if item in by_id]
    if not supporting:
        return ClaimEvidenceAssessment(
            claim_id=claim.claim_id,
            independent_source_families=None,
            has_primary_administrative_or_behavioral=None,
        )
    return ClaimEvidenceAssessment(
        claim_id=claim.claim_id,
        independent_source_families=len(
            {measurement.source.independence_family for measurement in supporting}
        ),
        has_primary_administrative_or_behavioral=any(
            measurement.evidence_kind in _PRIMARY_EVIDENCE for measurement in supporting
        ),
    )


def sources_are_independent(
    measurements: list[Measurement] | tuple[Measurement, ...],
    *,
    minimum_families: int = 2,
) -> bool:
    """Test independence by source lineage, never by row or URL count."""
    if minimum_families < 1:
        raise ValueError("minimum_families must be positive")
    return len({measurement.source.independence_family for measurement in measurements}) >= (
        minimum_families
    )


def verifier_is_separate(
    claim: Claim,
    verification: Verification,
    assignments: list[Assignment] | tuple[Assignment, ...],
) -> bool:
    """Require a verifier role and a different assignment from the claim author."""
    assignment = next(
        (
            candidate
            for candidate in assignments
            if candidate.assignment_id == verification.verifier_assignment_id
        ),
        None,
    )
    return (
        assignment is not None
        and assignment.role is AssignmentRole.VERIFIER
        and assignment.assignment_id != claim.created_by_assignment
        and verification.claim_id == claim.claim_id
    )


def require_verifier_separation(
    claim: Claim,
    verification: Verification,
    assignments: list[Assignment] | tuple[Assignment, ...],
) -> None:
    """Raise when claim creation and verification are not role-separated."""
    if not verifier_is_separate(claim, verification, assignments):
        raise ValueError("verification must come from a separate verifier assignment")


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include a timezone")
    return value


def _temporal_key(value: TemporalValue) -> datetime:
    if isinstance(value, datetime):
        return _require_aware(value).astimezone(UTC)
    return datetime.combine(value, datetime.min.time(), tzinfo=UTC)


def _unique_nonblank(value: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if any(not item for item in value):
        raise ValueError(f"{field} entries must not be blank")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} entries must be unique")
    return value


# The methodology sometimes calls the record simply Opportunity.  Keep the
# precise name as the implementation and expose this local compatibility alias.
Opportunity = OpportunityThesis


__all__ = [
    "Assignment",
    "AssignmentRole",
    "Claim",
    "ClaimCriticality",
    "ClaimEvidence",
    "ClaimEvidenceAssessment",
    "ClaimStatus",
    "ClaimType",
    "DerivedSignal",
    "EffectivePeriod",
    "Entity",
    "EntityType",
    "Estimate",
    "EvidenceKind",
    "EvidenceRelationship",
    "ExperimentStatus",
    "ExperimentThreshold",
    "FieldExperiment",
    "FrozenModel",
    "GateDecision",
    "Geography",
    "GeographySystem",
    "Industry",
    "IndustrySystem",
    "Measurement",
    "MeasurementDomain",
    "MeasurementSource",
    "MeasurementTime",
    "MeasurementUncertainty",
    "MeasurementValue",
    "Opportunity",
    "OpportunityStage",
    "OpportunityThesis",
    "Prediction",
    "PredictionInterval",
    "PredictionStatus",
    "PredictionType",
    "Sha256",
    "ThresholdOperator",
    "Verification",
    "VerificationChecks",
    "VerificationResult",
    "assess_claim_evidence",
    "require_verifier_separation",
    "sources_are_independent",
    "verifier_is_separate",
]
