"""Typed contracts for hypothesis generation and independent falsification.

Models may propose and critique theses, but they never rank themselves, mutate
the ledger, execute tools, or decide policy gates.  Every factual reference must
point to a measurement already present in the evidence packet.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, field_validator, model_validator

from app.venture.core import FrozenModel, Scenario, make_content_id, validate_identifier


class BusinessArchetype(StrEnum):
    """Commercial shape of a candidate, separate from its industry."""

    LOCAL_SERVICE = "local_service"
    MANAGED_SERVICE = "managed_service"
    SPECIALTY_CONTRACTOR = "specialty_contractor"
    DISTRIBUTOR = "distributor"
    ACQUISITION_PLATFORM = "acquisition_platform"
    FACILITY = "facility"
    SOFTWARE = "software"
    MARKETPLACE = "marketplace"


class ClassificationStatus(StrEnum):
    """Whether a statistical industry classification has been independently checked."""

    VERIFIED = "verified"
    PROVISIONAL = "provisional"


class EvidenceTopic(StrEnum):
    """Deterministic market taxonomy used to isolate falsification evidence."""

    UNIVERSAL = "universal"
    FIRE_LIFE_SAFETY = "fire_life_safety"
    EQUIPMENT_SERVICE = "equipment_service"
    BACKGROUND_SCREENING = "background_screening"
    WASTE_COLLECTION = "waste_collection"
    LOGISTICS_CONSULTING = "logistics_consulting"
    AGING_SERVICES = "aging_services"
    SECURITY_GUARDS = "security_guards"
    COMMERCIAL_CONSTRUCTION = "commercial_construction"
    SOFTWARE = "software"
    PHYSICIAN_OFFICES = "physician_offices"
    ENVIRONMENTAL_CONSULTING = "environmental_consulting"
    BIOTECH_LABS = "biotech_labs"
    SENIOR_LIVING_FACILITIES = "senior_living_facilities"
    SECTOR_23 = "sector_23"
    SECTOR_51 = "sector_51"
    SECTOR_54 = "sector_54"
    SECTOR_56 = "sector_56"
    SECTOR_62 = "sector_62"
    SECTOR_81 = "sector_81"


class MarketTopic(StrEnum):
    """A market-level axis explicitly selected for falsification."""

    FIRE_LIFE_SAFETY = "fire_life_safety"
    EQUIPMENT_SERVICE = "equipment_service"
    BACKGROUND_SCREENING = "background_screening"
    WASTE_COLLECTION = "waste_collection"
    LOGISTICS_CONSULTING = "logistics_consulting"
    AGING_SERVICES = "aging_services"
    SECURITY_GUARDS = "security_guards"
    COMMERCIAL_CONSTRUCTION = "commercial_construction"
    SOFTWARE = "software"
    PHYSICIAN_OFFICES = "physician_offices"
    ENVIRONMENTAL_CONSULTING = "environmental_consulting"
    BIOTECH_LABS = "biotech_labs"
    SENIOR_LIVING_FACILITIES = "senior_living_facilities"


class PacketMeasurement(FrozenModel):
    """Compact, source-labeled observation safe to show an analyst model."""

    measurement_id: str
    metric: str = Field(min_length=1)
    value: bool | int | float | str | None
    unit: str = Field(min_length=1)
    geography: str = Field(min_length=1)
    observed_period: str = Field(min_length=1)
    source_family: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    caveat: str = Field(min_length=1)
    quality_flags: tuple[str, ...] = ()

    @field_validator("measurement_id")
    @classmethod
    def _safe_measurement_id(cls, value: str) -> str:
        return validate_identifier(value, field="measurement_id")

    @field_validator("value")
    @classmethod
    def _finite_value(
        cls, value: bool | int | float | str | None
    ) -> bool | int | float | str | None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("measurement values must be finite")
        if isinstance(value, str) and not value:
            raise ValueError("string values must not be blank")
        return value

    @field_validator("quality_flags")
    @classmethod
    def _unique_flags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in value):
            raise ValueError("quality flags must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("quality flags must be unique")
        return value


class EvidencePacket(FrozenModel):
    """Frozen view of facts available at one information cutoff."""

    packet_id: str
    as_of: datetime
    measurements: tuple[PacketMeasurement, ...] = Field(min_length=1)
    allowed_geographies: tuple[str, ...] = Field(min_length=1)
    allowed_scenarios: tuple[Scenario, ...] = Field(min_length=1)
    allowed_naics_codes: tuple[str, ...] = Field(min_length=1)
    source_policy: str = Field(min_length=1)

    @field_validator("packet_id")
    @classmethod
    def _safe_packet_id(cls, value: str) -> str:
        return validate_identifier(value, field="packet_id")

    @field_validator("as_of")
    @classmethod
    def _aware_as_of(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must include a timezone")
        return value

    @model_validator(mode="after")
    def _unique_measurements_and_dimensions(self) -> Self:
        measurement_ids = [item.measurement_id for item in self.measurements]
        if len(set(measurement_ids)) != len(measurement_ids):
            raise ValueError("measurement ids must be unique within a packet")
        if len(set(self.allowed_geographies)) != len(self.allowed_geographies):
            raise ValueError("allowed_geographies must be unique")
        if len(set(self.allowed_scenarios)) != len(self.allowed_scenarios):
            raise ValueError("allowed_scenarios must be unique")
        if len(set(self.allowed_naics_codes)) != len(self.allowed_naics_codes):
            raise ValueError("allowed_naics_codes must be unique")
        if any(not code.isdigit() or len(code) != 6 for code in self.allowed_naics_codes):
            raise ValueError("allowed_naics_codes require exact six-digit NAICS codes")
        return self

    @property
    def measurement_ids(self) -> frozenset[str]:
        return frozenset(item.measurement_id for item in self.measurements)


_PROVIDER_FORBIDDEN_SCOPE_FLAGS = frozenset(
    {"customer_class_not_demand", "customer_only", "provider_forbidden"}
)


def provider_naics_codes(packet: EvidencePacket) -> tuple[str, ...]:
    """Return codes allowed for providers rather than customer context only.

    A packet can carry customer classification rows so a candidate can name its
    buyer precisely. Rows explicitly flagged as customer-only or provider-forbidden
    are not valid provider hypotheses in that packet. Codes without a scope row stay
    allowed for backward compatibility and fail later if classification review
    cannot obtain the required official row.
    """
    by_id = {item.measurement_id: item for item in packet.measurements}
    return tuple(
        code
        for code in packet.allowed_naics_codes
        if (
            (scope := by_id.get(f"naics22-{code}-scope")) is None
            or not _PROVIDER_FORBIDDEN_SCOPE_FLAGS.intersection(scope.quality_flags)
        )
    )


def customer_eligible_naics_codes(packet: EvidencePacket) -> tuple[str, ...]:
    """Return the packet's explicit customer allowlist when one is declared.

    Legacy packets without any ``customer_eligible`` scope flag retain their
    existing all-allowed-codes behavior. Once one row declares that flag, only
    explicitly flagged rows may appear in ``customer_naics_codes``.
    """
    explicit = _explicit_customer_naics_codes(packet)
    return explicit or packet.allowed_naics_codes


def _explicit_customer_naics_codes(packet: EvidencePacket) -> tuple[str, ...]:
    by_id = {item.measurement_id: item for item in packet.measurements}
    return tuple(
        code
        for code in packet.allowed_naics_codes
        if (
            (scope := by_id.get(f"naics22-{code}-scope")) is not None
            and "customer_eligible" in scope.quality_flags
        )
    )


def available_market_topics(packet: EvidencePacket) -> tuple[MarketTopic, ...]:
    """Return only market axes backed by at least one measurement in this packet."""
    available_values = {
        topic.value
        for measurement in packet.measurements
        for topic in evidence_topics(measurement.measurement_id)
        if topic.value in MarketTopic._value2member_map_
    }
    return tuple(MarketTopic(value) for value in sorted(available_values))


class HypothesisDraft(FrozenModel):
    """One model-proposed business mechanism, before a durable id is assigned."""

    title: str = Field(min_length=1, max_length=160)
    customer: str = Field(min_length=1, max_length=500)
    payer: str = Field(min_length=1, max_length=500)
    problem: str = Field(min_length=1, max_length=1_000)
    mechanism: str = Field(min_length=1, max_length=1_500)
    business_model: str = Field(min_length=1, max_length=1_000)
    geography: tuple[str, ...] = Field(min_length=1)
    naics_codes: tuple[str, ...] = Field(min_length=1, max_length=1)
    customer_naics_codes: tuple[str, ...] = Field(max_length=10)
    naics_basis: str = Field(min_length=1, max_length=1_000)
    classification_status: ClassificationStatus
    adjacent_market_exclusions: tuple[str, ...] = Field(min_length=1)
    entity_scope: str = Field(min_length=1, max_length=500)
    contestable_spend_basis: str = Field(min_length=1, max_length=1_000)
    scenario: Scenario
    archetype: BusinessArchetype
    offer_market_topic: MarketTopic
    context_market_topics: tuple[MarketTopic, ...] = Field(max_length=2)
    evidence_refs: tuple[str, ...] = Field(min_length=2)
    reason_for_now: str = Field(min_length=1, max_length=1_000)
    critical_assumptions: tuple[str, ...] = Field(min_length=1)
    disconfirming_observations: tuple[str, ...] = Field(min_length=1)

    @field_validator("naics_codes")
    @classmethod
    def _valid_naics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 1:
            raise ValueError("a hypothesis requires exactly one provider/business NAICS code")
        if len(set(value)) != len(value):
            raise ValueError("naics_codes must be unique")
        if any(not code.isdigit() or len(code) != 6 for code in value):
            raise ValueError("business hypotheses require six-digit NAICS codes")
        return value

    @field_validator("customer_naics_codes")
    @classmethod
    def _valid_customer_naics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("customer_naics_codes must be unique")
        if any(not code.isdigit() or len(code) != 6 for code in value):
            raise ValueError("customer_naics_codes require six-digit NAICS codes")
        return value

    @field_validator("context_market_topics")
    @classmethod
    def _valid_context_market_topics(
        cls, value: tuple[MarketTopic, ...]
    ) -> tuple[MarketTopic, ...]:
        if len(value) > 2:
            raise ValueError("context_market_topics may contain at most two topics")
        if len(set(value)) != len(value):
            raise ValueError("context_market_topics must be unique")
        return tuple(sorted(value, key=lambda item: item.value))

    @field_validator("evidence_refs")
    @classmethod
    def _safe_unique_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        safe = tuple(validate_identifier(item, field="evidence_ref") for item in value)
        if len(set(safe)) != len(safe):
            raise ValueError("evidence_refs must be unique")
        return safe

    @field_validator(
        "geography",
        "adjacent_market_exclusions",
        "critical_assumptions",
        "disconfirming_observations",
    )
    @classmethod
    def _unique_nonblank_text(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in value):
            raise ValueError("tuple entries must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("tuple entries must be unique")
        return value

    @model_validator(mode="after")
    def _market_axes_are_distinct(self) -> Self:
        _validate_market_axes(self.offer_market_topic, self.context_market_topics)
        return self


class HypothesisBatch(FrozenModel):
    """Bounded output of one blind generation assignment."""

    hypotheses: tuple[HypothesisDraft, ...] = Field(min_length=1, max_length=25)

    @model_validator(mode="after")
    def _titles_are_unique(self) -> Self:
        normalized = [item.title.casefold() for item in self.hypotheses]
        if len(set(normalized)) != len(normalized):
            raise ValueError("hypothesis titles must be unique")
        return self


class CandidateHypothesis(FrozenModel):
    """Durable, evidence-referenced candidate created from a validated draft."""

    opportunity_id: str
    thesis_id: str
    origin_assignment_id: str
    title: str
    customer: str
    payer: str
    problem: str
    mechanism: str
    business_model: str
    geography: tuple[str, ...]
    naics_codes: tuple[str, ...] = Field(min_length=1, max_length=1)
    customer_naics_codes: tuple[str, ...] = Field(max_length=10)
    naics_basis: str = Field(min_length=1, max_length=1_000)
    classification_status: ClassificationStatus
    adjacent_market_exclusions: tuple[str, ...]
    entity_scope: str
    contestable_spend_basis: str
    scenario: Scenario
    archetype: BusinessArchetype
    offer_market_topic: MarketTopic
    context_market_topics: tuple[MarketTopic, ...] = Field(max_length=2)
    evidence_refs: tuple[str, ...]
    reason_for_now: str
    critical_assumptions: tuple[str, ...]
    disconfirming_observations: tuple[str, ...]
    created_at: datetime

    @field_validator("opportunity_id", "thesis_id", "origin_assignment_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @field_validator("context_market_topics")
    @classmethod
    def _valid_context_market_topics(
        cls, value: tuple[MarketTopic, ...]
    ) -> tuple[MarketTopic, ...]:
        if len(value) > 2:
            raise ValueError("context_market_topics may contain at most two topics")
        if len(set(value)) != len(value):
            raise ValueError("context_market_topics must be unique")
        return tuple(sorted(value, key=lambda item: item.value))

    @model_validator(mode="after")
    def _market_axes_are_distinct(self) -> Self:
        _validate_market_axes(self.offer_market_topic, self.context_market_topics)
        return self


def _validate_market_axes(
    offer_topic: MarketTopic,
    context_topics: tuple[MarketTopic, ...],
) -> None:
    if len(context_topics) > 2:
        raise ValueError("context_market_topics may contain at most two topics")
    if len(set(context_topics)) != len(context_topics):
        raise ValueError("context_market_topics must be unique")
    if offer_topic in context_topics:
        raise ValueError("offer_market_topic cannot also be a context_market_topic")


def materialize_hypothesis(
    draft: HypothesisDraft,
    *,
    packet: EvidencePacket,
    assignment_id: str,
    created_at: datetime,
) -> CandidateHypothesis:
    """Validate a draft against its packet and assign a stable content id."""
    unknown_refs = set(draft.evidence_refs).difference(packet.measurement_ids)
    if unknown_refs:
        raise ValueError(f"hypothesis cites unknown measurements: {sorted(unknown_refs)}")
    _validate_market_axes(draft.offer_market_topic, draft.context_market_topics)
    selected_market_topics = (draft.offer_market_topic, *draft.context_market_topics)
    unjustified_market_topics = tuple(
        topic.value
        for topic in selected_market_topics
        if not any(
            EvidenceTopic(topic.value) in evidence_topics(evidence_ref)
            for evidence_ref in draft.evidence_refs
        )
    )
    if unjustified_market_topics:
        raise ValueError(
            "selected market topics require an explicitly cited measurement classified "
            f"to each market: {sorted(unjustified_market_topics)}"
        )
    invalid_geographies = set(draft.geography).difference(packet.allowed_geographies)
    if invalid_geographies:
        raise ValueError(f"hypothesis uses disallowed geographies: {sorted(invalid_geographies)}")
    invalid_naics = set(draft.naics_codes).difference(packet.allowed_naics_codes)
    if invalid_naics:
        raise ValueError(f"hypothesis uses disallowed NAICS codes: {sorted(invalid_naics)}")
    customer_only_provider_codes = set(draft.naics_codes).difference(provider_naics_codes(packet))
    if customer_only_provider_codes:
        raise ValueError(
            "hypothesis uses customer-context-only NAICS codes as the provider: "
            f"{sorted(customer_only_provider_codes)}"
        )
    invalid_customer_naics = set(draft.customer_naics_codes).difference(
        customer_eligible_naics_codes(packet)
    )
    if invalid_customer_naics:
        raise ValueError(
            "hypothesis uses NAICS codes outside the packet's customer allowlist: "
            f"{sorted(invalid_customer_naics)}"
        )
    if _explicit_customer_naics_codes(packet):
        selected_evidence_topics = {EvidenceTopic(topic.value) for topic in selected_market_topics}
        mismatched_customer_topics = {
            code
            for code in draft.customer_naics_codes
            if not _NAICS_TOPICS.get(code, frozenset())
            .difference(_BROAD_SECTOR_EVIDENCE_TOPICS)
            .intersection(selected_evidence_topics)
        }
        if mismatched_customer_topics:
            raise ValueError(
                "customer NAICS codes require a matching selected market topic: "
                f"{sorted(mismatched_customer_topics)}"
            )
    provider_code = draft.naics_codes[0]
    required_scope_ref = f"naics22-{provider_code}-scope"
    if (
        required_scope_ref in packet.measurement_ids
        and required_scope_ref not in draft.evidence_refs
    ):
        raise ValueError(
            f"hypothesis must cite its official NAICS scope measurement: {required_scope_ref}"
        )
    allowed_scope_refs = {
        required_scope_ref,
        *(f"naics22-{code}-scope" for code in draft.customer_naics_codes),
    }
    other_scope_refs = {
        evidence_ref
        for evidence_ref in draft.evidence_refs
        if evidence_ref.startswith("naics22-")
        and evidence_ref.endswith("-scope")
        and evidence_ref not in allowed_scope_refs
    }
    if other_scope_refs:
        raise ValueError(
            "hypothesis cites NAICS scope measurements for non-provider codes: "
            f"{sorted(other_scope_refs)}"
        )
    if draft.scenario not in packet.allowed_scenarios:
        raise ValueError(f"scenario {draft.scenario.value!r} is outside the packet")
    safe_assignment = validate_identifier(assignment_id, field="assignment_id")
    identity = {
        "packet_id": packet.packet_id,
        "assignment_id": safe_assignment,
        "draft": draft.model_dump(mode="json"),
    }
    thesis_identity = {
        "title": _normalized_identity_text(draft.title),
        "customer": _normalized_identity_text(draft.customer),
        "payer": _normalized_identity_text(draft.payer),
        "problem": _normalized_identity_text(draft.problem),
        "mechanism": _normalized_identity_text(draft.mechanism),
        "business_model": _normalized_identity_text(draft.business_model),
        "geography": tuple(sorted(_normalized_identity_text(item) for item in draft.geography)),
        "provider_naics_code": draft.naics_codes[0],
        "customer_naics_codes": tuple(sorted(draft.customer_naics_codes)),
        "scenario": draft.scenario,
        "archetype": draft.archetype,
        "offer_market_topic": draft.offer_market_topic,
        "context_market_topics": tuple(
            sorted(topic.value for topic in draft.context_market_topics)
        ),
    }
    return CandidateHypothesis(
        opportunity_id=make_content_id("opportunity", identity),
        thesis_id=make_content_id("thesis", thesis_identity),
        origin_assignment_id=safe_assignment,
        **draft.model_dump(),
        created_at=created_at,
    )


def _normalized_identity_text(value: str) -> str:
    return " ".join(value.split()).casefold()


_FIRE_DENOMINATOR_IDS = frozenset(
    {
        "cbp-kings-establishments-ge20",
        "cbp-yolo-establishments-ge20",
        "cbp-san-diego-establishments-ge20",
        "cbp-merced-establishments-ge20",
        "cbp-tulare-establishments-ge20",
        "cbp-santa-clara-establishments-ge20",
        "cbp-fresno-establishments-ge20",
        "cbp-santa-barbara-establishments-ge20",
        "cbp-stanislaus-establishments-ge20",
        "cbp-imperial-establishments-ge20",
    }
)

_MARKET_TOPIC_PREFIXES: tuple[tuple[EvidenceTopic, tuple[str, ...]], ...] = (
    (
        EvidenceTopic.FIRE_LIFE_SAFETY,
        (
            "ca-fire-",
            "ca-c16-",
            "cslb-c16-",
            "derived-c16-",
            "cbp-security-systems-",
            "cbp-electrical-",
            "ec22-security-systems-",
            "ec22-building-inspection-",
            "ca-osfm-",
            "sdfd-compliance-engine-",
            "cslb-sd-clear-c16-",
            "socal-portable-extinguisher-",
            "sd-city-kitchen-system-",
            "unknown-sd-fire-",
            "unknown-sd-senior-fire-",
            "cdss-rcfe-fire-",
        ),
    ),
    (
        EvidenceTopic.EQUIPMENT_SERVICE,
        (
            "usa-precision-repair-",
            "usa-medical-wholesale-",
            "cbp-precision-repair-",
            "cbp-testing-calibration-",
            "cbp-industrial-repair-",
            "cbp-medical-wholesale-",
            "ec22-precision-repair-",
            "ec22-testing-calibration-",
            "unknown-repair-",
            "cbp-sd-testing-lab-",
            "bmbl-bsc-",
            "ucsd-bsc-",
            "nsf-sd-bsc-",
            "omnia-bsc-",
            "derived-bsc-",
            "unknown-sd-bsc-",
        ),
    ),
    (
        EvidenceTopic.BACKGROUND_SCREENING,
        (
            "usa-background-",
            "cbp-background-",
            "ec22-background-",
            "unknown-background-",
        ),
    ),
    (
        EvidenceTopic.WASTE_COLLECTION,
        (
            "usa-other-waste-",
            "cbp-other-waste-",
            "irs-sp-waste-",
            "unknown-waste-",
        ),
    ),
    (
        EvidenceTopic.LOGISTICS_CONSULTING,
        (
            "usa-logistics-consulting-",
            "cbp-logistics-consulting-",
            "ec22-logistics-consulting-",
        ),
    ),
    (
        EvidenceTopic.AGING_SERVICES,
        (
            "usa-individual-family-",
            "cbp-home-health-",
            "cbp-elderly-disabled-",
            "cms-",
            "ec22-home-health-",
            "ec22-elderly-disabled-",
            "irs-sp-social-assistance-",
            "unknown-aging-",
        ),
    ),
    (
        EvidenceTopic.SECURITY_GUARDS,
        ("usa-security-guards-", "cbp-security-guards-"),
    ),
    (
        EvidenceTopic.COMMERCIAL_CONSTRUCTION,
        (
            "usa-commercial-construction-",
            "cbp-commercial-construction-",
        ),
    ),
    (EvidenceTopic.SOFTWARE, ("usa-software-",)),
    (
        EvidenceTopic.PHYSICIAN_OFFICES,
        ("usa-physician-offices-", "irs-sp-ambulatory-"),
    ),
    (
        EvidenceTopic.ENVIRONMENTAL_CONSULTING,
        ("ec22-environmental-consulting-",),
    ),
    (
        EvidenceTopic.BIOTECH_LABS,
        (
            "cbp-sd-biotech-",
            "cbp-sd-biological-product-",
            "cbp-sd-medical-lab-",
        ),
    ),
    (
        EvidenceTopic.SENIOR_LIVING_FACILITIES,
        (
            "cdss-sd-licensed-rcfe-",
            "cdss-rcfe-fire-",
            "unknown-sd-senior-fire-",
        ),
    ),
)

_BROAD_SECTOR_PREFIXES: tuple[tuple[EvidenceTopic, tuple[str, ...]], ...] = (
    (
        EvidenceTopic.SECTOR_23,
        (
            "bls-bed-23-",
            "irs-corp-construction-",
            "irs-corp-electrical-",
            "irs-sp-building-construction-",
        ),
    ),
    (EvidenceTopic.SECTOR_51, ("bls-bed-51-",)),
    (
        EvidenceTopic.SECTOR_54,
        (
            "bls-bed-54-",
            "irs-corp-consulting-",
            "irs-sp-consulting-",
        ),
    ),
    (
        EvidenceTopic.SECTOR_56,
        (
            "bls-bed-56-",
            "irs-corp-admin-waste-",
            "irs-sp-admin-",
        ),
    ),
    (
        EvidenceTopic.SECTOR_62,
        (
            "bls-bed-62-",
            "irs-corp-health-social-",
            "irs-corp-nursing-residential-",
            "irs-sp-health-social-",
        ),
    ),
    (
        EvidenceTopic.SECTOR_81,
        (
            "bls-bed-81-",
            "irs-corp-repair-",
            "irs-corp-other-repair-",
            "irs-sp-repair-",
        ),
    ),
)

_NAICS_TOPICS: dict[str, frozenset[EvidenceTopic]] = {
    "236220": frozenset({EvidenceTopic.COMMERCIAL_CONSTRUCTION, EvidenceTopic.SECTOR_23}),
    "238210": frozenset({EvidenceTopic.FIRE_LIFE_SAFETY, EvidenceTopic.SECTOR_23}),
    "238220": frozenset({EvidenceTopic.FIRE_LIFE_SAFETY, EvidenceTopic.SECTOR_23}),
    "423450": frozenset({EvidenceTopic.EQUIPMENT_SERVICE}),
    "513210": frozenset({EvidenceTopic.SOFTWARE, EvidenceTopic.SECTOR_51}),
    "541350": frozenset({EvidenceTopic.FIRE_LIFE_SAFETY, EvidenceTopic.SECTOR_54}),
    "541380": frozenset({EvidenceTopic.EQUIPMENT_SERVICE, EvidenceTopic.SECTOR_54}),
    "541614": frozenset({EvidenceTopic.LOGISTICS_CONSULTING, EvidenceTopic.SECTOR_54}),
    "541620": frozenset({EvidenceTopic.ENVIRONMENTAL_CONSULTING, EvidenceTopic.SECTOR_54}),
    "541714": frozenset({EvidenceTopic.BIOTECH_LABS, EvidenceTopic.SECTOR_54}),
    "541990": frozenset({EvidenceTopic.FIRE_LIFE_SAFETY, EvidenceTopic.SECTOR_54}),
    "561611": frozenset({EvidenceTopic.BACKGROUND_SCREENING, EvidenceTopic.SECTOR_56}),
    "561612": frozenset({EvidenceTopic.SECURITY_GUARDS, EvidenceTopic.SECTOR_56}),
    "561621": frozenset({EvidenceTopic.FIRE_LIFE_SAFETY, EvidenceTopic.SECTOR_56}),
    "562119": frozenset({EvidenceTopic.WASTE_COLLECTION, EvidenceTopic.SECTOR_56}),
    "621111": frozenset({EvidenceTopic.PHYSICIAN_OFFICES, EvidenceTopic.SECTOR_62}),
    "621610": frozenset({EvidenceTopic.AGING_SERVICES, EvidenceTopic.SECTOR_62}),
    "623110": frozenset({EvidenceTopic.AGING_SERVICES, EvidenceTopic.SECTOR_62}),
    "623312": frozenset({EvidenceTopic.SENIOR_LIVING_FACILITIES, EvidenceTopic.SECTOR_62}),
    "624120": frozenset({EvidenceTopic.AGING_SERVICES, EvidenceTopic.SECTOR_62}),
    "624190": frozenset({EvidenceTopic.AGING_SERVICES, EvidenceTopic.SECTOR_62}),
    "811210": frozenset({EvidenceTopic.EQUIPMENT_SERVICE, EvidenceTopic.SECTOR_81}),
    "811310": frozenset(
        {
            EvidenceTopic.FIRE_LIFE_SAFETY,
            EvidenceTopic.EQUIPMENT_SERVICE,
            EvidenceTopic.SECTOR_81,
        }
    ),
}

_BROAD_SECTOR_EVIDENCE_TOPICS = frozenset(
    {
        EvidenceTopic.SECTOR_23,
        EvidenceTopic.SECTOR_51,
        EvidenceTopic.SECTOR_54,
        EvidenceTopic.SECTOR_56,
        EvidenceTopic.SECTOR_62,
        EvidenceTopic.SECTOR_81,
    }
)


def evidence_topics(measurement_id: str) -> frozenset[EvidenceTopic]:
    """Classify a pilot measurement without interpreting its free-form prose.

    Unknown identifiers deliberately receive no topic. That fail-safe keeps them out
    of another market's falsification packet unless the candidate explicitly cited
    them. ``universal-*`` is the explicit extension point for evidence applicable to
    every candidate.
    """
    safe_id = validate_identifier(measurement_id, field="measurement_id")
    topics: set[EvidenceTopic] = set()
    if safe_id.startswith("universal-"):
        topics.add(EvidenceTopic.UNIVERSAL)
    if safe_id in _FIRE_DENOMINATOR_IDS:
        topics.add(EvidenceTopic.FIRE_LIFE_SAFETY)
    for topic, prefixes in (*_MARKET_TOPIC_PREFIXES, *_BROAD_SECTOR_PREFIXES):
        if safe_id.startswith(prefixes):
            topics.add(topic)
    # The county aging ratios place the county between ``cbp-`` and the service.
    if safe_id.startswith("cbp-") and (
        "-home-health-age65" in safe_id or "-elderly-disabled-age65" in safe_id
    ):
        topics.add(EvidenceTopic.AGING_SERVICES)
    if safe_id.startswith("naics22-") and safe_id.endswith("-scope"):
        code = safe_id.removeprefix("naics22-").removesuffix("-scope")
        topics.update(_NAICS_TOPICS.get(code, ()))
    if safe_id == "naics22-561790-fire-excluded":
        topics.add(EvidenceTopic.FIRE_LIFE_SAFETY)
    return frozenset(topics)


def candidate_evidence_topics(
    candidate: CandidateHypothesis,
    *,
    packet: EvidencePacket,
) -> frozenset[EvidenceTopic]:
    """Return explicit market axes plus the provider's broad-sector topics.

    Cited measurements are validated but deliberately do not expand the topic set:
    one mistaken cross-market citation must not admit the rest of that market.
    Customer NAICS codes never expand falsifier evidence. Provider codes that map
    to multiple markets contribute only broad-sector context. Market-level expansion
    comes solely from the explicitly justified offer and context topics.
    """
    by_id = {item.measurement_id: item for item in packet.measurements}
    missing_refs = set(candidate.evidence_refs).difference(by_id)
    if missing_refs:
        raise ValueError(f"candidate cites measurements outside the packet: {sorted(missing_refs)}")
    topics: set[EvidenceTopic] = {
        EvidenceTopic.UNIVERSAL,
        EvidenceTopic(candidate.offer_market_topic.value),
        *(EvidenceTopic(context_topic.value) for context_topic in candidate.context_market_topics),
    }
    provider_topics: frozenset[EvidenceTopic] = _NAICS_TOPICS.get(
        candidate.naics_codes[0], frozenset()
    )
    topics.update(provider_topics.intersection(_BROAD_SECTOR_EVIDENCE_TOPICS))
    return frozenset(topics)


def scope_falsification_packet(
    candidate: CandidateHypothesis,
    *,
    packet: EvidencePacket,
) -> EvidencePacket:
    """Select candidate evidence without leaking unrelated market facts.

    Cited measurements are always retained. Uncited measurements enter only through
    an explicit same-market, broad-sector, or universal taxonomy match. Unclassified
    identifiers never trigger the unsafe fallback of exposing the full packet.
    """
    cited = frozenset(candidate.evidence_refs)
    topics = candidate_evidence_topics(candidate, packet=packet)
    selected = tuple(
        measurement
        for measurement in packet.measurements
        if measurement.measurement_id in cited
        or bool(evidence_topics(measurement.measurement_id).intersection(topics))
    )
    return packet.model_copy(update={"measurements": selected})


class ClassificationReviewOutcome(StrEnum):
    """Independent offer-to-industry-scope determination."""

    FIT = "fit"
    CONTRADICTS = "contradicts"
    UNRESOLVED = "unresolved"


class ClassificationReview(FrozenModel):
    """Tool-free review of one provider code against eligible official scope rows."""

    opportunity_id: str
    assignment_id: str
    naics_code: str
    scope_measurement_ref: str
    outcome: ClassificationReviewOutcome
    analysis: str = Field(min_length=1, max_length=2_000)
    mismatches: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()

    @field_validator("opportunity_id", "assignment_id", "scope_measurement_ref")
    @classmethod
    def _safe_review_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("naics_code")
    @classmethod
    def _exact_naics_code(cls, value: str) -> str:
        if not value.isdigit() or len(value) != 6:
            raise ValueError("classification review requires one six-digit NAICS code")
        return value

    @field_validator("mismatches", "missing_evidence")
    @classmethod
    def _unique_review_details(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in value):
            raise ValueError("classification review details must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("classification review details must be unique")
        return value

    @model_validator(mode="after")
    def _outcome_has_required_basis(self) -> Self:
        if self.outcome is ClassificationReviewOutcome.FIT and self.mismatches:
            raise ValueError("a FIT classification review cannot report scope mismatches")
        if self.outcome is ClassificationReviewOutcome.FIT and self.missing_evidence:
            raise ValueError("a FIT classification review cannot report missing evidence")
        if self.outcome is ClassificationReviewOutcome.CONTRADICTS and not self.mismatches:
            raise ValueError("a CONTRADICTS review requires at least one scope mismatch")
        if self.outcome is ClassificationReviewOutcome.UNRESOLVED and not self.missing_evidence:
            raise ValueError("an UNRESOLVED review requires missing evidence")
        return self


class ClassificationComparisonReview(ClassificationReview):
    """Versioned comparative reviewer output persisted separately by v6 runs."""

    schema_version: str = "classification-review-v2"
    compared_scope_refs: tuple[str, ...]
    plausible_naics_codes: tuple[str, ...]

    @field_validator("compared_scope_refs")
    @classmethod
    def _safe_unique_compared_scope_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        safe = tuple(validate_identifier(item, field="compared_scope_ref") for item in value)
        if len(set(safe)) != len(safe):
            raise ValueError("compared_scope_refs must be unique")
        return safe

    @field_validator("plausible_naics_codes")
    @classmethod
    def _exact_unique_plausible_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not code.isdigit() or len(code) != 6 for code in value):
            raise ValueError("plausible_naics_codes require six-digit NAICS codes")
        if len(set(value)) != len(value):
            raise ValueError("plausible_naics_codes must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _comparative_outcome_is_consistent(self) -> Self:
        if self.schema_version != "classification-review-v2":
            raise ValueError("unsupported classification comparison review schema")
        if self.outcome is ClassificationReviewOutcome.FIT and self.plausible_naics_codes != (
            self.naics_code,
        ):
            raise ValueError("FIT requires the proposed provider code to be uniquely plausible")
        if (
            self.outcome is ClassificationReviewOutcome.CONTRADICTS
            and self.naics_code in self.plausible_naics_codes
        ):
            raise ValueError("CONTRADICTS cannot list the proposed provider code as plausible")
        return self

    def legacy_review(self) -> ClassificationReview:
        """Derive the unchanged v5-compatible persisted review contract."""
        return ClassificationReview.model_validate(
            self.model_dump(
                exclude={
                    "schema_version",
                    "compared_scope_refs",
                    "plausible_naics_codes",
                }
            )
        )


class ClassificationReviewExecution(FrozenModel):
    """Auditable comparative output plus its exact legacy review projection."""

    comparison: ClassificationComparisonReview
    review: ClassificationReview

    @model_validator(mode="after")
    def _legacy_projection_is_exact(self) -> Self:
        if self.review != self.comparison.legacy_review():
            raise ValueError("classification review is not the exact comparative projection")
        return self


def classification_scope_measurement(
    candidate: CandidateHypothesis,
    *,
    packet: EvidencePacket,
) -> PacketMeasurement:
    """Return the sole official scope row for the candidate's provider code."""
    code = candidate.naics_codes[0]
    required_ref = f"naics22-{code}-scope"
    matches = tuple(item for item in packet.measurements if item.measurement_id == required_ref)
    if len(matches) != 1:
        raise ValueError(f"classification review requires exactly one {required_ref!r} measurement")
    return matches[0]


def classification_scope_measurements(packet: EvidencePacket) -> tuple[PacketMeasurement, ...]:
    """Return every official scope row eligible for an independent provider comparison."""
    by_id = {item.measurement_id: item for item in packet.measurements}
    return tuple(
        scope
        for code in provider_naics_codes(packet)
        if (scope := by_id.get(f"naics22-{code}-scope")) is not None
    )


def validate_classification_review(
    review: ClassificationReview,
    *,
    candidate: CandidateHypothesis,
    packet: EvidencePacket,
    assignment_id: str,
) -> None:
    """Bind a reviewer output to its candidate, assignment, code, and scope row."""
    safe_assignment = validate_identifier(assignment_id, field="assignment_id")
    scope = classification_scope_measurement(candidate, packet=packet)
    expected = {
        "opportunity_id": candidate.opportunity_id,
        "assignment_id": safe_assignment,
        "naics_code": candidate.naics_codes[0],
        "scope_measurement_ref": scope.measurement_id,
    }
    actual = {
        "opportunity_id": review.opportunity_id,
        "assignment_id": review.assignment_id,
        "naics_code": review.naics_code,
        "scope_measurement_ref": review.scope_measurement_ref,
    }
    if actual != expected:
        mismatches = sorted(key for key in expected if actual[key] != expected[key])
        raise ValueError(f"classification review returned mismatched binding fields: {mismatches}")


class FalsificationDimension(StrEnum):
    """Required independent challenges aligned to policy gate G3."""

    SUBSTITUTION = "substitution"
    LATENT_COMPETITION = "latent_competition"
    REGULATORY_LOW_SUPPLY = "regulatory_low_supply"
    MANDATED_VS_CONTESTABLE_SPEND = "mandated_vs_contestable_spend"
    DEMAND_WITHOUT_WTP = "demand_without_wtp"
    STRESSED_ECONOMICS = "stressed_economics"


class FalsificationOutcome(StrEnum):
    """What the challenge found; absence of a contradiction is not proof."""

    NO_CONTRADICTION_FOUND = "no_contradiction_found"
    WEAKENS = "weakens"
    CONTRADICTS = "contradicts"
    UNRESOLVED = "unresolved"


class FalsificationFinding(FrozenModel):
    """One required challenge and the evidence used to reach it."""

    dimension: FalsificationDimension
    outcome: FalsificationOutcome
    analysis: str = Field(min_length=1, max_length=2_000)
    evidence_refs: tuple[str, ...]
    missing_evidence: tuple[str, ...]

    @field_validator("evidence_refs")
    @classmethod
    def _safe_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        safe = tuple(validate_identifier(item, field="evidence_ref") for item in value)
        if len(set(safe)) != len(safe):
            raise ValueError("evidence_refs must be unique")
        return safe

    @field_validator("missing_evidence")
    @classmethod
    def _unique_missing(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in value):
            raise ValueError("missing evidence descriptions must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("missing evidence descriptions must be unique")
        return value

    @model_validator(mode="after")
    def _no_contradiction_requires_complete_evidence(self) -> Self:
        if self.outcome is FalsificationOutcome.NO_CONTRADICTION_FOUND and self.missing_evidence:
            raise ValueError(
                "NO_CONTRADICTION_FOUND cannot report missing_evidence; "
                "use UNRESOLVED when evidence is absent"
            )
        return self


class FalsificationReport(FrozenModel):
    """Complete role-separated critique; deterministic code applies policy."""

    opportunity_id: str
    assignment_id: str
    findings: tuple[FalsificationFinding, ...] = Field(min_length=6, max_length=6)
    explicit_illegality_found: bool
    explicit_unfinanceable_found: bool
    explicit_negative_stressed_contribution_found: bool
    kill_recommendation: bool
    kill_basis: str | None = None
    critical_unknowns: tuple[str, ...]

    @field_validator("opportunity_id", "assignment_id")
    @classmethod
    def _safe_ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("critical_unknowns")
    @classmethod
    def _unique_unknowns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in value):
            raise ValueError("critical unknowns must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("critical unknowns must be unique")
        return value

    @model_validator(mode="after")
    def _complete_dimensions_and_kill_basis(self) -> Self:
        dimensions = [finding.dimension for finding in self.findings]
        if set(dimensions) != set(FalsificationDimension) or len(set(dimensions)) != len(
            dimensions
        ):
            raise ValueError("findings must contain every falsification dimension exactly once")
        has_explicit_kill = any(
            (
                self.explicit_illegality_found,
                self.explicit_unfinanceable_found,
                self.explicit_negative_stressed_contribution_found,
            )
        )
        if self.kill_recommendation and not self.kill_basis:
            raise ValueError("kill_recommendation requires kill_basis")
        if has_explicit_kill and not self.kill_recommendation:
            raise ValueError("an explicit disqualifier requires a kill recommendation")
        if not self.kill_recommendation and self.kill_basis is not None:
            raise ValueError("kill_basis is allowed only for a kill recommendation")
        return self


def validate_falsification_refs(
    report: FalsificationReport,
    *,
    packet: EvidencePacket,
    candidate: CandidateHypothesis,
) -> None:
    """Reject role output that cites data or a candidate outside its blind packet."""
    if report.opportunity_id != candidate.opportunity_id:
        raise ValueError("falsification report names a different opportunity")
    cited = {evidence_ref for finding in report.findings for evidence_ref in finding.evidence_refs}
    unknown_refs = cited.difference(packet.measurement_ids)
    if unknown_refs:
        raise ValueError(f"falsifier cites unknown measurements: {sorted(unknown_refs)}")


__all__ = [
    "BusinessArchetype",
    "CandidateHypothesis",
    "ClassificationComparisonReview",
    "ClassificationReview",
    "ClassificationReviewExecution",
    "ClassificationReviewOutcome",
    "ClassificationStatus",
    "EvidencePacket",
    "EvidenceTopic",
    "FalsificationDimension",
    "FalsificationFinding",
    "FalsificationOutcome",
    "FalsificationReport",
    "HypothesisBatch",
    "HypothesisDraft",
    "MarketTopic",
    "PacketMeasurement",
    "available_market_topics",
    "candidate_evidence_topics",
    "classification_scope_measurement",
    "classification_scope_measurements",
    "customer_eligible_naics_codes",
    "evidence_topics",
    "materialize_hypothesis",
    "provider_naics_codes",
    "scope_falsification_packet",
    "validate_classification_review",
    "validate_falsification_refs",
]
