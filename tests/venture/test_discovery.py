"""Adversarial contracts for model-proposed and model-critiqued theses."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.venture.core import Scenario
from app.venture.discovery import (
    BusinessArchetype,
    ClassificationReview,
    ClassificationReviewOutcome,
    ClassificationStatus,
    EvidencePacket,
    EvidenceTopic,
    FalsificationDimension,
    FalsificationFinding,
    FalsificationOutcome,
    FalsificationReport,
    HypothesisDraft,
    MarketTopic,
    PacketMeasurement,
    classification_scope_measurement,
    classification_scope_measurements,
    customer_eligible_naics_codes,
    evidence_topics,
    materialize_hypothesis,
    provider_naics_codes,
    scope_falsification_packet,
    validate_classification_review,
    validate_falsification_refs,
)

NOW = datetime(2026, 7, 31, tzinfo=UTC)


def _measurement(measurement_id: str) -> PacketMeasurement:
    return PacketMeasurement(
        measurement_id=measurement_id,
        metric="federal contract obligations",
        value=1_000_000,
        unit="USD",
        geography="United States",
        observed_period="2025-02-01/2026-07-31",
        source_family="USAspending",
        source_url="https://api.usaspending.gov/",
        caveat="Federal prime-contract demand is not total private demand.",
    )


def _packet() -> EvidencePacket:
    return EvidencePacket(
        packet_id="packet-1",
        as_of=NOW,
        measurements=(
            _measurement("usa-precision-repair-m-1"),
            _measurement("cbp-precision-repair-m-2"),
        ),
        allowed_geographies=("United States", "California"),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("811210", "541380"),
        source_policy="Official administrative sources only.",
    )


def _draft() -> HypothesisDraft:
    return HypothesisDraft(
        title="Calibrated equipment uptime service",
        customer="Small regulated clinics and laboratories",
        payer="Clinic or laboratory operations lead",
        problem="Downtime and overdue calibration interrupt billable work.",
        mechanism="Regional pickup, calibration coordination, repair, and audit trail.",
        business_model="Annual service agreement plus per-device work.",
        geography=("United States",),
        naics_codes=("811210",),
        customer_naics_codes=(),
        naics_basis="811210 describes electronic and precision equipment repair.",
        classification_status=ClassificationStatus.PROVISIONAL,
        adjacent_market_exclusions=(
            "Consumer electronics repair",
            "General product-testing laboratories",
        ),
        entity_scope="Employer establishments; nonemployers are not included.",
        contestable_spend_basis=(
            "The hypothesis requires proof that operators can replace or supplement OEM service."
        ),
        scenario=Scenario.BOOTSTRAPPED,
        archetype=BusinessArchetype.MANAGED_SERVICE,
        offer_market_topic=MarketTopic.EQUIPMENT_SERVICE,
        context_market_topics=(),
        evidence_refs=("usa-precision-repair-m-1", "cbp-precision-repair-m-2"),
        reason_for_now="Observed procurement demand rose while employer supply contracted.",
        critical_assumptions=("Customers will pay for a consolidated service.",),
        disconfirming_observations=("OEM contracts already cover the target devices.",),
    )


def _findings(
    *, evidence_ref: str = "usa-precision-repair-m-1"
) -> tuple[FalsificationFinding, ...]:
    return tuple(
        FalsificationFinding(
            dimension=dimension,
            outcome=FalsificationOutcome.UNRESOLVED,
            analysis="The packet does not resolve this challenge.",
            evidence_refs=(evidence_ref,),
            missing_evidence=("A direct customer or transaction observation is required.",),
        )
        for dimension in FalsificationDimension
    )


def test_hypothesis_id_is_deterministic_and_packet_bounded() -> None:
    packet = _packet()
    first = materialize_hypothesis(
        _draft(), packet=packet, assignment_id="researcher-1", created_at=NOW
    )
    second = materialize_hypothesis(
        _draft(), packet=packet, assignment_id="researcher-1", created_at=NOW
    )

    assert first.opportunity_id == second.opportunity_id


def test_thesis_fingerprint_is_stable_across_packets_and_assignments() -> None:
    first_packet = _packet()
    second_packet = first_packet.model_copy(update={"packet_id": "packet-2"})
    first = materialize_hypothesis(
        _draft(),
        packet=first_packet,
        assignment_id="researcher-1",
        created_at=NOW,
    )
    second = materialize_hypothesis(
        _draft(),
        packet=second_packet,
        assignment_id="researcher-2",
        created_at=NOW,
    )

    assert first.thesis_id == second.thesis_id
    assert first.opportunity_id != second.opportunity_id


def test_hypothesis_cannot_invent_an_evidence_reference() -> None:
    draft = _draft().model_copy(update={"evidence_refs": ("usa-precision-repair-m-1", "invented")})

    with pytest.raises(ValueError, match="unknown measurements"):
        materialize_hypothesis(
            draft,
            packet=_packet(),
            assignment_id="researcher-1",
            created_at=NOW,
        )


def test_hypothesis_cannot_escape_geographic_scope() -> None:
    draft = _draft().model_copy(update={"geography": ("Mars",)})

    with pytest.raises(ValueError, match="disallowed geographies"):
        materialize_hypothesis(
            draft,
            packet=_packet(),
            assignment_id="researcher-1",
            created_at=NOW,
        )


def test_hypothesis_requires_six_digit_naics_and_market_boundaries() -> None:
    payload = _draft().model_dump()
    payload["naics_codes"] = ("81",)

    with pytest.raises(ValidationError, match="six-digit NAICS"):
        HypothesisDraft.model_validate(payload)


def test_hypothesis_rejects_buyer_channel_or_adjacent_extra_naics_codes() -> None:
    payload = _draft().model_dump()
    payload["naics_codes"] = ("811210", "541380")

    with pytest.raises(ValidationError, match=r"at most 1 item|exactly one provider"):
        HypothesisDraft.model_validate(payload)


def test_market_axes_reject_broad_sector_and_universal_topics() -> None:
    for invalid in ("sector_81", "universal"):
        payload = _draft().model_dump()
        payload["offer_market_topic"] = invalid
        with pytest.raises(ValidationError, match="offer_market_topic"):
            HypothesisDraft.model_validate(payload)


@pytest.mark.parametrize(
    ("context_topics", "message"),
    [
        (
            (MarketTopic.EQUIPMENT_SERVICE,),
            "offer_market_topic cannot also be a context_market_topic",
        ),
        (
            (MarketTopic.PHYSICIAN_OFFICES, MarketTopic.PHYSICIAN_OFFICES),
            "context_market_topics must be unique",
        ),
        (
            (
                MarketTopic.PHYSICIAN_OFFICES,
                MarketTopic.FIRE_LIFE_SAFETY,
                MarketTopic.SOFTWARE,
            ),
            r"at most 2 items|at most two topics",
        ),
    ],
)
def test_context_market_topics_are_distinct_and_bounded(
    context_topics: tuple[MarketTopic, ...],
    message: str,
) -> None:
    payload = _draft().model_dump()
    payload["context_market_topics"] = context_topics

    with pytest.raises(ValidationError, match=message):
        HypothesisDraft.model_validate(payload)


def test_packet_naics_allowlist_is_exact_unique_and_nonempty() -> None:
    payload = _packet().model_dump()
    payload["allowed_naics_codes"] = ("811210", "81")

    with pytest.raises(ValidationError, match="exact six-digit"):
        EvidencePacket.model_validate(payload)

    payload["allowed_naics_codes"] = ("811210", "811210")
    with pytest.raises(ValidationError, match="must be unique"):
        EvidencePacket.model_validate(payload)


def test_materialization_rejects_naics_outside_exact_packet_allowlist() -> None:
    draft = _draft().model_copy(update={"naics_codes": ("561611",)})

    with pytest.raises(ValueError, match="disallowed NAICS"):
        materialize_hypothesis(
            draft,
            packet=_packet(),
            assignment_id="researcher-1",
            created_at=NOW,
        )


def test_candidate_retains_provisional_classification_and_basis() -> None:
    candidate = materialize_hypothesis(
        _draft(), packet=_packet(), assignment_id="researcher-1", created_at=NOW
    )

    assert candidate.classification_status is ClassificationStatus.PROVISIONAL
    assert "equipment repair" in candidate.naics_basis


@pytest.mark.parametrize("provider_code", ["541990", "811210"])
def test_materialization_accepts_one_provider_code_with_its_scope_ref(
    provider_code: str,
) -> None:
    scope_ref = f"naics22-{provider_code}-scope"
    packet = EvidencePacket(
        packet_id=f"packet-{provider_code}",
        as_of=NOW,
        measurements=(
            _measurement("usa-precision-repair-m-1"),
            PacketMeasurement(
                measurement_id=scope_ref,
                metric=f"Official scope for {provider_code}",
                value=f"Definition of {provider_code}",
                unit="scope text",
                geography="United States",
                observed_period="2022",
                source_family="Census NAICS",
                source_url="https://www.census.gov/naics/",
                caveat="Statistical classification does not establish market demand.",
            ),
        ),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=(provider_code,),
        source_policy="Official administrative sources only.",
    )
    draft = _draft().model_copy(
        update={
            "naics_codes": (provider_code,),
            "naics_basis": f"The revenue-producing offer is proposed under {provider_code}.",
            "evidence_refs": ("usa-precision-repair-m-1", scope_ref),
        }
    )

    candidate = materialize_hypothesis(
        draft, packet=packet, assignment_id="researcher-1", created_at=NOW
    )

    assert candidate.naics_codes == (provider_code,)
    assert classification_scope_measurement(candidate, packet=packet).measurement_id == scope_ref


def test_materialization_requires_own_scope_and_rejects_self_excluded_extras() -> None:
    packet = EvidencePacket(
        packet_id="packet-scope-binding",
        as_of=NOW,
        measurements=(
            _measurement("usa-precision-repair-m-1"),
            _measurement("cbp-precision-repair-m-2"),
            _topic_measurement("naics22-811210-scope"),
            _topic_measurement("naics22-541380-scope"),
        ),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("811210", "541380"),
        source_policy="Official sources only.",
    )
    missing_own_scope = _draft().model_copy(
        update={
            "evidence_refs": (
                "usa-precision-repair-m-1",
                "cbp-precision-repair-m-2",
            )
        }
    )
    with pytest.raises(ValueError, match="must cite its official NAICS scope"):
        materialize_hypothesis(
            missing_own_scope,
            packet=packet,
            assignment_id="researcher-1",
            created_at=NOW,
        )

    extra_scope = _draft().model_copy(
        update={
            "evidence_refs": (
                "usa-precision-repair-m-1",
                "naics22-811210-scope",
                "naics22-541380-scope",
            )
        }
    )
    with pytest.raises(ValueError, match="non-provider codes"):
        materialize_hypothesis(
            extra_scope,
            packet=packet,
            assignment_id="researcher-1",
            created_at=NOW,
        )


def _topic_measurement(measurement_id: str) -> PacketMeasurement:
    return PacketMeasurement(
        measurement_id=measurement_id,
        metric="topic-scoping test",
        value=None,
        unit="status",
        geography="United States",
        observed_period="2026",
        source_family="official-test-source",
        source_url="https://example.gov/topic",
        caveat="Used only to verify deterministic packet scoping.",
        quality_flags=("counterevidence",),
    )


def _explicit_customer_policy_packet() -> EvidencePacket:
    provider_codes = ("238220", "541380", "541990", "811310")
    customer_codes = ("541714", "623312")
    scope_rows = tuple(
        _topic_measurement(f"naics22-{code}-scope").model_copy(
            update={
                "quality_flags": (
                    "classification_only",
                    *(
                        ("customer_class_not_demand", "customer_eligible")
                        if code in customer_codes
                        else ()
                    ),
                )
            }
        )
        for code in (
            *provider_codes[:2],
            customer_codes[0],
            provider_codes[2],
            customer_codes[1],
            provider_codes[3],
        )
    )
    return EvidencePacket(
        packet_id="packet-explicit-customer-policy",
        as_of=NOW,
        measurements=(
            *scope_rows,
            _topic_measurement("cbp-sd-biotech-rd-establishments-2023"),
            _topic_measurement("cdss-sd-licensed-rcfe-facilities"),
            _topic_measurement("ca-fire-904-recurring-itm"),
        ),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=(
            "238220",
            "541380",
            "541714",
            "541990",
            "623312",
            "811310",
        ),
        source_policy="Official sources only.",
    )


def test_explicit_customer_policy_separates_provider_and_customer_codes() -> None:
    packet = _explicit_customer_policy_packet()

    assert provider_naics_codes(packet) == ("238220", "541380", "541990", "811310")
    assert customer_eligible_naics_codes(packet) == ("541714", "623312")
    assert tuple(
        measurement.measurement_id for measurement in classification_scope_measurements(packet)
    ) == (
        "naics22-238220-scope",
        "naics22-541380-scope",
        "naics22-541990-scope",
        "naics22-811310-scope",
    )


def test_explicit_customer_policy_binds_bsc_and_fire_buyers_to_their_market_axis() -> None:
    packet = _explicit_customer_policy_packet()
    bsc_draft = _draft().model_copy(
        update={
            "naics_codes": ("541380",),
            "customer_naics_codes": ("541714",),
            "offer_market_topic": MarketTopic.EQUIPMENT_SERVICE,
            "context_market_topics": (MarketTopic.BIOTECH_LABS,),
            "evidence_refs": (
                "naics22-541380-scope",
                "cbp-sd-biotech-rd-establishments-2023",
            ),
        }
    )
    bsc = materialize_hypothesis(
        bsc_draft,
        packet=packet,
        assignment_id="researcher-bsc-customer",
        created_at=NOW,
    )
    assert bsc.customer_naics_codes == ("541714",)

    with pytest.raises(ValueError, match="matching selected market topic"):
        materialize_hypothesis(
            bsc_draft.model_copy(update={"customer_naics_codes": ("623312",)}),
            packet=packet,
            assignment_id="researcher-bsc-wrong-customer",
            created_at=NOW,
        )
    with pytest.raises(ValueError, match="customer allowlist"):
        materialize_hypothesis(
            bsc_draft.model_copy(update={"customer_naics_codes": ("541380",)}),
            packet=packet,
            assignment_id="researcher-bsc-provider-as-customer",
            created_at=NOW,
        )

    fire_draft = _draft().model_copy(
        update={
            "naics_codes": ("811310",),
            "customer_naics_codes": ("623312",),
            "offer_market_topic": MarketTopic.FIRE_LIFE_SAFETY,
            "context_market_topics": (MarketTopic.SENIOR_LIVING_FACILITIES,),
            "evidence_refs": (
                "naics22-811310-scope",
                "cdss-sd-licensed-rcfe-facilities",
            ),
        }
    )
    fire = materialize_hypothesis(
        fire_draft,
        packet=packet,
        assignment_id="researcher-fire-customer",
        created_at=NOW,
    )
    assert fire.customer_naics_codes == ("623312",)

    with pytest.raises(ValueError, match="matching selected market topic"):
        materialize_hypothesis(
            fire_draft.model_copy(update={"customer_naics_codes": ("541714",)}),
            packet=packet,
            assignment_id="researcher-fire-wrong-customer",
            created_at=NOW,
        )


def test_classification_review_is_bound_to_exact_candidate_assignment_code_and_ref() -> None:
    scope_ref = "naics22-811210-scope"
    packet = EvidencePacket(
        packet_id="packet-review-binding",
        as_of=NOW,
        measurements=(
            _measurement("usa-precision-repair-m-1"),
            _topic_measurement(scope_ref),
        ),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("811210",),
        source_policy="Official sources only.",
    )
    candidate = materialize_hypothesis(
        _draft().model_copy(update={"evidence_refs": ("usa-precision-repair-m-1", scope_ref)}),
        packet=packet,
        assignment_id="researcher-1",
        created_at=NOW,
    )
    review = ClassificationReview(
        opportunity_id=candidate.opportunity_id,
        assignment_id="classifier-1",
        naics_code="811210",
        scope_measurement_ref=scope_ref,
        outcome=ClassificationReviewOutcome.FIT,
        analysis="The repair service is inside the supplied equipment-repair scope.",
    )

    validate_classification_review(
        review,
        candidate=candidate,
        packet=packet,
        assignment_id="classifier-1",
    )

    for field, invented in (
        ("opportunity_id", "invented-opportunity"),
        ("assignment_id", "invented-assignment"),
        ("naics_code", "541990"),
        ("scope_measurement_ref", "invented-scope"),
    ):
        bad = review.model_copy(update={field: invented})
        with pytest.raises(ValueError, match="mismatched binding fields"):
            validate_classification_review(
                bad,
                candidate=candidate,
                packet=packet,
                assignment_id="classifier-1",
            )


def test_classification_fit_cannot_hide_unknown_revenue_mix() -> None:
    with pytest.raises(ValidationError, match=r"FIT.*missing evidence"):
        ClassificationReview(
            opportunity_id="opportunity-1",
            assignment_id="classifier-1",
            naics_code="541380",
            scope_measurement_ref="naics22-541380-scope",
            outcome=ClassificationReviewOutcome.FIT,
            analysis="The mixed calibration and repair bundle might fit.",
            missing_evidence=("Primary revenue activity and revenue mix are unknown.",),
        )


@pytest.mark.parametrize(
    ("naics_code", "offer_market", "cited_refs", "expected_relevant"),
    [
        (
            "561611",
            MarketTopic.BACKGROUND_SCREENING,
            ("usa-background-obligations", "manual-cited-background"),
            {"cbp-background-est-growth", "unknown-background-concentration"},
        ),
        (
            "621610",
            MarketTopic.AGING_SERVICES,
            ("cms-ca-load-proxy", "manual-cited-aging"),
            {"cbp-home-health-est-growth", "unknown-aging-wtp-workforce"},
        ),
    ],
)
def test_falsification_scope_retains_refs_and_relevant_counterevidence_only(
    naics_code: str,
    offer_market: MarketTopic,
    cited_refs: tuple[str, str],
    expected_relevant: set[str],
) -> None:
    measurement_ids = (
        *cited_refs,
        *sorted(expected_relevant),
        "universal-shared-baseline",
        "ca-fire-904-recurring-itm",
        "ca-fire-9041-owner-inspection",
        "unclassified-unrelated-fact",
    )
    packet = EvidencePacket(
        packet_id="packet-topic-scope",
        as_of=NOW,
        measurements=tuple(_topic_measurement(item) for item in measurement_ids),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("561611", "621610"),
        source_policy="Official sources only.",
    )
    draft = _draft().model_copy(
        update={
            "geography": ("United States",),
            "naics_codes": (naics_code,),
            "offer_market_topic": offer_market,
            "context_market_topics": (),
            "naics_basis": "Provisional six-digit classification for the proposed service.",
            "evidence_refs": cited_refs,
        }
    )
    candidate = materialize_hypothesis(
        draft, packet=packet, assignment_id="researcher-1", created_at=NOW
    )

    scoped = scope_falsification_packet(candidate, packet=packet)
    scoped_ids = scoped.measurement_ids

    assert set(cited_refs).issubset(scoped_ids)
    assert expected_relevant.issubset(scoped_ids)
    assert "universal-shared-baseline" in scoped_ids
    assert "ca-fire-904-recurring-itm" not in scoped_ids
    assert "ca-fire-9041-owner-inspection" not in scoped_ids
    assert "unclassified-unrelated-fact" not in scoped_ids


def test_topic_taxonomy_is_explicit_and_unknown_ids_fail_closed() -> None:
    assert evidence_topics("ca-fire-904-recurring-itm") == frozenset(
        {EvidenceTopic.FIRE_LIFE_SAFETY}
    )
    assert evidence_topics("unknown-background-concentration") == frozenset(
        {EvidenceTopic.BACKGROUND_SCREENING}
    )
    assert evidence_topics("novel-unclassified-measurement") == frozenset()


def test_one_cross_market_citation_does_not_expand_that_market_topic() -> None:
    measurement_ids = (
        "usa-background-obligations",
        "cbp-background-est-growth",
        "ca-fire-904-recurring-itm",
        "ca-fire-9041-owner-inspection",
    )
    packet = EvidencePacket(
        packet_id="packet-bad-citation",
        as_of=NOW,
        measurements=tuple(_topic_measurement(item) for item in measurement_ids),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("561611",),
        source_policy="Official sources only.",
    )
    draft = _draft().model_copy(
        update={
            "naics_codes": ("561611",),
            "offer_market_topic": MarketTopic.BACKGROUND_SCREENING,
            "context_market_topics": (),
            "naics_basis": "Provisional background-screening classification.",
            "evidence_refs": (
                "usa-background-obligations",
                "ca-fire-904-recurring-itm",
            ),
        }
    )
    candidate = materialize_hypothesis(
        draft, packet=packet, assignment_id="researcher-1", created_at=NOW
    )

    scoped_ids = scope_falsification_packet(candidate, packet=packet).measurement_ids

    assert "ca-fire-904-recurring-itm" in scoped_ids
    assert "cbp-background-est-growth" in scoped_ids
    assert "ca-fire-9041-owner-inspection" not in scoped_ids


def test_customer_naics_never_expands_falsifier_market_context() -> None:
    measurement_ids = (
        "usa-precision-repair-obligations",
        "naics22-811210-scope",
        "usa-physician-offices-growth",
        "ca-fire-904-recurring-itm",
    )
    packet = EvidencePacket(
        packet_id="packet-provider-buyer",
        as_of=NOW,
        measurements=tuple(_topic_measurement(item) for item in measurement_ids),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("811210", "621111"),
        source_policy="Official sources only.",
    )
    draft = _draft().model_copy(
        update={
            "customer_naics_codes": ("621111",),
            "evidence_refs": (
                "usa-precision-repair-obligations",
                "naics22-811210-scope",
            ),
        }
    )
    candidate = materialize_hypothesis(
        draft, packet=packet, assignment_id="researcher-1", created_at=NOW
    )

    scoped_ids = scope_falsification_packet(candidate, packet=packet).measurement_ids

    assert candidate.naics_codes == ("811210",)
    assert candidate.customer_naics_codes == ("621111",)
    assert "usa-physician-offices-growth" not in scoped_ids
    assert "ca-fire-904-recurring-itm" not in scoped_ids


def test_calibration_offer_and_physician_context_expand_both_explicit_axes() -> None:
    measurement_ids = (
        "naics22-541380-scope",
        "usa-physician-offices-growth",
        "unknown-repair-private-wtp",
        "cbp-testing-calibration-est-growth",
        "irs-sp-ambulatory-receipts",
        "bls-bed-54-2018-survival",
        "unknown-aging-wtp-workforce",
    )
    packet = EvidencePacket(
        packet_id="packet-calibration-physician",
        as_of=NOW,
        measurements=tuple(_topic_measurement(item) for item in measurement_ids),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("541380", "621111"),
        source_policy="Official sources only.",
    )
    candidate = materialize_hypothesis(
        _draft().model_copy(
            update={
                "naics_codes": ("541380",),
                "customer_naics_codes": ("621111",),
                "offer_market_topic": MarketTopic.EQUIPMENT_SERVICE,
                "context_market_topics": (MarketTopic.PHYSICIAN_OFFICES,),
                "naics_basis": "541380 describes testing and calibration laboratories.",
                "evidence_refs": (
                    "naics22-541380-scope",
                    "usa-physician-offices-growth",
                ),
            }
        ),
        packet=packet,
        assignment_id="researcher-1",
        created_at=NOW,
    )

    scoped_ids = scope_falsification_packet(candidate, packet=packet).measurement_ids

    assert {
        "unknown-repair-private-wtp",
        "cbp-testing-calibration-est-growth",
        "irs-sp-ambulatory-receipts",
        "bls-bed-54-2018-survival",
    }.issubset(scoped_ids)
    assert "unknown-aging-wtp-workforce" not in scoped_ids


def test_fire_software_offer_and_fire_context_expand_both_explicit_axes() -> None:
    measurement_ids = (
        "naics22-513210-scope",
        "ca-fire-904-recurring-itm",
        "usa-software-growth",
        "ca-fire-9041-owner-inspection",
        "bls-bed-51-2018-survival",
        "unknown-background-concentration",
    )
    packet = EvidencePacket(
        packet_id="packet-fire-software",
        as_of=NOW,
        measurements=tuple(_topic_measurement(item) for item in measurement_ids),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("513210",),
        source_policy="Official sources only.",
    )
    candidate = materialize_hypothesis(
        _draft().model_copy(
            update={
                "naics_codes": ("513210",),
                "customer_naics_codes": (),
                "offer_market_topic": MarketTopic.SOFTWARE,
                "context_market_topics": (MarketTopic.FIRE_LIFE_SAFETY,),
                "naics_basis": "513210 describes software publishers.",
                "evidence_refs": (
                    "naics22-513210-scope",
                    "ca-fire-904-recurring-itm",
                ),
            }
        ),
        packet=packet,
        assignment_id="researcher-1",
        created_at=NOW,
    )

    scoped_ids = scope_falsification_packet(candidate, packet=packet).measurement_ids

    assert {
        "usa-software-growth",
        "ca-fire-9041-owner-inspection",
        "bls-bed-51-2018-survival",
    }.issubset(scoped_ids)
    assert "unknown-background-concentration" not in scoped_ids


def test_background_offer_with_many_buyer_codes_expands_background_only() -> None:
    measurement_ids = (
        "naics22-561611-scope",
        "usa-background-obligations",
        "unknown-background-concentration",
        "bls-bed-56-2018-survival",
        "ca-fire-904-recurring-itm",
        "unknown-aging-wtp-workforce",
        "usa-physician-offices-growth",
    )
    packet = EvidencePacket(
        packet_id="packet-background-many-buyers",
        as_of=NOW,
        measurements=tuple(_topic_measurement(item) for item in measurement_ids),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("561611", "238210", "621610", "621111"),
        source_policy="Official sources only.",
    )
    candidate = materialize_hypothesis(
        _draft().model_copy(
            update={
                "naics_codes": ("561611",),
                "customer_naics_codes": ("238210", "621610", "621111"),
                "offer_market_topic": MarketTopic.BACKGROUND_SCREENING,
                "context_market_topics": (),
                "naics_basis": "561611 describes investigation and screening services.",
                "evidence_refs": (
                    "naics22-561611-scope",
                    "usa-background-obligations",
                ),
            }
        ),
        packet=packet,
        assignment_id="researcher-1",
        created_at=NOW,
    )

    scoped_ids = scope_falsification_packet(candidate, packet=packet).measurement_ids

    assert {
        "unknown-background-concentration",
        "bls-bed-56-2018-survival",
    }.issubset(scoped_ids)
    assert {
        "ca-fire-904-recurring-itm",
        "unknown-aging-wtp-workforce",
        "usa-physician-offices-growth",
    }.isdisjoint(scoped_ids)


def test_unjustified_context_market_is_rejected_instead_of_expanding_scope() -> None:
    packet = EvidencePacket(
        packet_id="packet-unjustified-context",
        as_of=NOW,
        measurements=tuple(
            _topic_measurement(item)
            for item in (
                "naics22-561611-scope",
                "usa-background-obligations",
                "ca-fire-904-recurring-itm",
            )
        ),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("561611",),
        source_policy="Official sources only.",
    )
    draft = _draft().model_copy(
        update={
            "naics_codes": ("561611",),
            "offer_market_topic": MarketTopic.BACKGROUND_SCREENING,
            "context_market_topics": (MarketTopic.FIRE_LIFE_SAFETY,),
            "evidence_refs": (
                "naics22-561611-scope",
                "usa-background-obligations",
            ),
        }
    )

    with pytest.raises(ValueError, match=r"explicitly cited measurement.*fire_life_safety"):
        materialize_hypothesis(
            draft,
            packet=packet,
            assignment_id="researcher-1",
            created_at=NOW,
        )


def test_current_pilot_measurements_have_an_explicit_topic() -> None:
    from app.venture.pilot_evidence import build_pilot_evidence_packet

    unclassified = [
        item.measurement_id
        for item in build_pilot_evidence_packet().measurements
        if not evidence_topics(item.measurement_id)
    ]

    assert unclassified == []


def test_falsifier_must_cover_each_required_dimension_once() -> None:
    repeated = list(_findings())
    repeated[-1] = repeated[0]

    with pytest.raises(ValidationError, match="every falsification dimension"):
        FalsificationReport(
            opportunity_id="opportunity-1",
            assignment_id="falsifier-1",
            findings=tuple(repeated),
            explicit_illegality_found=False,
            explicit_unfinanceable_found=False,
            explicit_negative_stressed_contribution_found=False,
            kill_recommendation=False,
            critical_unknowns=("Willingness to pay is unknown.",),
        )


def test_no_contradiction_finding_cannot_admit_missing_evidence() -> None:
    with pytest.raises(
        ValidationError,
        match="NO_CONTRADICTION_FOUND cannot report missing_evidence",
    ):
        FalsificationFinding(
            dimension=FalsificationDimension.REGULATORY_LOW_SUPPLY,
            outcome=FalsificationOutcome.NO_CONTRADICTION_FOUND,
            analysis=(
                "No contradiction was found, but the low-supply claim is unsupported and unproven."
            ),
            evidence_refs=(),
            missing_evidence=("Direct evidence that regulation causes low supply.",),
        )


def test_explicit_disqualifier_cannot_be_softened_to_hold() -> None:
    with pytest.raises(ValidationError, match="explicit disqualifier"):
        FalsificationReport(
            opportunity_id="opportunity-1",
            assignment_id="falsifier-1",
            findings=_findings(),
            explicit_illegality_found=True,
            explicit_unfinanceable_found=False,
            explicit_negative_stressed_contribution_found=False,
            kill_recommendation=False,
            critical_unknowns=(),
        )


def test_falsifier_cannot_cite_out_of_packet_data() -> None:
    candidate = materialize_hypothesis(
        _draft(), packet=_packet(), assignment_id="researcher-1", created_at=NOW
    )
    report = FalsificationReport(
        opportunity_id=candidate.opportunity_id,
        assignment_id="falsifier-1",
        findings=_findings(evidence_ref="outside"),
        explicit_illegality_found=False,
        explicit_unfinanceable_found=False,
        explicit_negative_stressed_contribution_found=False,
        kill_recommendation=False,
        critical_unknowns=("Willingness to pay is unknown.",),
    )

    with pytest.raises(ValueError, match="unknown measurements"):
        validate_falsification_refs(report, packet=_packet(), candidate=candidate)
