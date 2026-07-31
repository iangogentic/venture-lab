"""Analyst roles are bounded, blind, typed, and budgeted."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from app.llm.messages import ChatMessage, Role
from app.venture.analysts import (
    classification_input_payload,
    classification_input_payload_v1,
    classification_input_payload_v2,
    falsify_hypothesis,
    generate_hypotheses,
    review_classification,
    review_classification_comparative,
)
from app.venture.core import Scenario
from app.venture.discovery import (
    BusinessArchetype,
    CandidateHypothesis,
    ClassificationComparisonReview,
    ClassificationReview,
    ClassificationReviewOutcome,
    ClassificationStatus,
    EvidencePacket,
    FalsificationDimension,
    FalsificationFinding,
    FalsificationOutcome,
    FalsificationReport,
    HypothesisBatch,
    HypothesisDraft,
    MarketTopic,
    PacketMeasurement,
    classification_scope_measurements,
    materialize_hypothesis,
)
from app.venture.operations import BudgetExceededError, BudgetGuard, BudgetPolicy, KillSwitch

NOW = datetime(2026, 7, 31, tzinfo=UTC)


class FakeStructuredGenerator:
    def __init__(self, replies: list[BaseModel]) -> None:
        self.replies = replies
        self.calls: list[tuple[Sequence[ChatMessage], type[BaseModel]]] = []

    def generate_structured[T: BaseModel](
        self,
        messages: str | Sequence[ChatMessage],
        schema: type[T],
        **_kwargs: Any,
    ) -> T:
        assert not isinstance(messages, str)
        self.calls.append((messages, schema))
        reply = self.replies.pop(0)
        return schema.model_validate(reply.model_dump())


class ComparativeClassificationReply(ClassificationComparisonReview):
    """Transient reviewer reply; persisted reviews deliberately omit this field."""

    pass


def _packet() -> EvidencePacket:
    measurements = tuple(
        PacketMeasurement(
            measurement_id=measurement_id,
            metric="observed signal",
            value=index,
            unit="count",
            geography="United States",
            observed_period="2023",
            source_family=f"source-{index}",
            source_url=f"https://example.gov/{index}",
            caveat="A proxy is not a validated market.",
        )
        for index, measurement_id in (
            (1, "usa-precision-repair-m-1"),
            (2, "cbp-precision-repair-m-2"),
        )
    )
    return EvidencePacket(
        packet_id="packet-1",
        as_of=NOW,
        measurements=measurements,
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("811210",),
        source_policy="Official administrative records.",
    )


def _batch() -> HypothesisBatch:
    return HypothesisBatch(
        hypotheses=(
            HypothesisDraft(
                title="Equipment uptime service",
                customer="Small laboratories",
                payer="Laboratory operator",
                problem="Downtime interrupts billable work.",
                mechanism="Repair coordination and compliance records.",
                business_model="Annual agreement plus per-device work.",
                geography=("United States",),
                naics_codes=("811210",),
                customer_naics_codes=(),
                naics_basis="811210 is a provisional service classification.",
                classification_status=ClassificationStatus.PROVISIONAL,
                adjacent_market_exclusions=("Consumer electronics repair",),
                entity_scope="Employer establishments only.",
                contestable_spend_basis=(
                    "The payer must be able to supplement or replace an OEM contract."
                ),
                scenario=Scenario.BOOTSTRAPPED,
                archetype=BusinessArchetype.MANAGED_SERVICE,
                offer_market_topic=MarketTopic.EQUIPMENT_SERVICE,
                context_market_topics=(),
                evidence_refs=(
                    "usa-precision-repair-m-1",
                    "cbp-precision-repair-m-2",
                ),
                reason_for_now="Demand proxy rose while supply proxy fell.",
                critical_assumptions=("Operators will pay for consolidated service.",),
                disconfirming_observations=("OEM contracts already cover the work.",),
            ),
        )
    )


def _report(opportunity_id: str) -> FalsificationReport:
    return FalsificationReport(
        opportunity_id=opportunity_id,
        assignment_id="falsifier-1",
        findings=tuple(
            FalsificationFinding(
                dimension=dimension,
                outcome=FalsificationOutcome.UNRESOLVED,
                analysis="The packet cannot resolve this challenge.",
                evidence_refs=("usa-precision-repair-m-1",),
                missing_evidence=("Behavioral evidence is absent.",),
            )
            for dimension in FalsificationDimension
        ),
        explicit_illegality_found=False,
        explicit_unfinanceable_found=False,
        explicit_negative_stressed_contribution_found=False,
        kill_recommendation=False,
        critical_unknowns=("Willingness to pay is unknown.",),
    )


def _classification_packet() -> EvidencePacket:
    packet = _packet()
    scope = PacketMeasurement(
        measurement_id="naics22-811210-scope",
        metric="2022 NAICS scope: electronic and precision equipment repair",
        value=(
            "Repair and maintenance of electronic and precision equipment "
            "without retailing new equipment."
        ),
        unit="official scope text",
        geography="United States",
        observed_period="2022",
        source_family="Census NAICS",
        source_url="https://www.census.gov/naics/",
        caveat="Classification scope is not demand or profitability evidence.",
    )
    return packet.model_copy(update={"measurements": (*packet.measurements, scope)})


def _classification_batch() -> HypothesisBatch:
    draft = (
        _batch()
        .hypotheses[0]
        .model_copy(
            update={
                "evidence_refs": (
                    "usa-precision-repair-m-1",
                    "cbp-precision-repair-m-2",
                    "naics22-811210-scope",
                ),
            }
        )
    )
    return HypothesisBatch(hypotheses=(draft,))


def test_classification_reviewer_sees_only_offer_and_selected_scope() -> None:
    packet = _classification_packet()
    generator = FakeStructuredGenerator([_classification_batch()])
    candidate = generate_hypotheses(
        packet=packet,
        assignment_id="researcher-1",
        created_at=NOW,
        llm=generator,
        max_hypotheses=1,
    )[0]
    reply = ComparativeClassificationReply(
        opportunity_id=candidate.opportunity_id,
        assignment_id="classifier-1",
        naics_code="811210",
        scope_measurement_ref="naics22-811210-scope",
        compared_scope_refs=("naics22-811210-scope",),
        plausible_naics_codes=(),
        outcome=ClassificationReviewOutcome.CONTRADICTS,
        analysis=(
            "Fire training and delivery logistics do not make the offer an "
            "electronic-equipment repair provider."
        ),
        mismatches=("The offer sells training and logistics, not covered repair work.",),
    )
    reviewer = FakeStructuredGenerator([reply])

    result = review_classification(
        candidate=candidate,
        packet=packet,
        assignment_id="classifier-1",
        llm=reviewer,
    )

    assert result.outcome is ClassificationReviewOutcome.CONTRADICTS
    messages, schema = reviewer.calls[0]
    properties = schema.model_json_schema()["properties"]
    assert properties["opportunity_id"]["const"] == candidate.opportunity_id
    assert properties["assignment_id"]["const"] == "classifier-1"
    assert properties["naics_code"]["const"] == "811210"
    assert properties["scope_measurement_ref"]["const"] == "naics22-811210-scope"
    rendered = "\n".join(message.content for message in messages)
    assert "naics22-811210-scope" in rendered
    assert '"measurement_id":"usa-precision-repair-m-1"' not in rendered
    assert '"measurement_id":"cbp-precision-repair-m-2"' not in rendered
    assert candidate.origin_assignment_id not in rendered
    assert "training" in rendered
    assert "logistics" in rendered
    assert "every stated revenue-producing activity" in rendered
    assert "fire bundle spanning inspection/testing" in rendered
    assert "calibration/testing bundled with maintenance/repair" in rendered
    assert "primary revenue activity and revenue mix" in rendered
    assert "return UNRESOLVED" in rendered
    assert "merely adjunct" in rendered
    assert "Fire-system scope rows are not alternatives for scientific" in rendered
    assert "equipment, and scientific-equipment scope rows" in rendered
    assert "Where 541380 and 811210 are supplied" in rendered


@pytest.mark.parametrize(
    (
        "provider_code",
        "offer_topic",
        "support_ref",
        "title",
        "mechanism",
        "expected_rule",
    ),
    [
        (
            "541990",
            MarketTopic.FIRE_LIFE_SAFETY,
            "ca-fire-904-recurring-itm",
            "Fire inspection, testing, repair, and installation bundle",
            (
                "Sell inspection and testing, repair and maintenance without "
                "installation, and installation."
            ),
            "fire bundle spanning inspection/testing",
        ),
        (
            "541380",
            MarketTopic.EQUIPMENT_SERVICE,
            "usa-precision-repair-obligations",
            "Calibration laboratory with bundled maintenance and repair",
            "Sell calibration and testing together with maintenance and repair.",
            "calibration/testing bundled with maintenance/repair",
        ),
    ],
)
def test_mixed_revenue_activity_examples_are_sent_to_fail_closed_reviewer(
    provider_code: str,
    offer_topic: MarketTopic,
    support_ref: str,
    title: str,
    mechanism: str,
    expected_rule: str,
) -> None:
    scope_ref = f"naics22-{provider_code}-scope"
    packet = EvidencePacket(
        packet_id=f"packet-mixed-{provider_code}",
        as_of=NOW,
        measurements=(
            PacketMeasurement(
                measurement_id=support_ref,
                metric="provider-market observation",
                value=1,
                unit="count",
                geography="United States",
                observed_period="2026",
                source_family="official test source",
                source_url="https://example.gov/market",
                caveat="This observation does not establish the activity revenue mix.",
            ),
            PacketMeasurement(
                measurement_id=scope_ref,
                metric=f"Official NAICS scope for {provider_code}",
                value=f"Official definition for {provider_code}.",
                unit="official scope text",
                geography="United States",
                observed_period="2022",
                source_family="Census NAICS",
                source_url="https://www.census.gov/naics/",
                caveat="The scope does not establish which bundled activity drives revenue.",
            ),
        ),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=(provider_code,),
        source_policy="Official administrative records.",
    )
    draft = (
        _batch()
        .hypotheses[0]
        .model_copy(
            update={
                "title": title,
                "mechanism": mechanism,
                "naics_codes": (provider_code,),
                "customer_naics_codes": (),
                "naics_basis": "The mixed offer requires independent scope review.",
                "offer_market_topic": offer_topic,
                "context_market_topics": (),
                "evidence_refs": (scope_ref, support_ref),
            }
        )
    )
    candidate = materialize_hypothesis(
        draft,
        packet=packet,
        assignment_id="researcher-mixed",
        created_at=NOW,
    )
    reply = ComparativeClassificationReply(
        opportunity_id=candidate.opportunity_id,
        assignment_id="classifier-mixed",
        naics_code=provider_code,
        scope_measurement_ref=scope_ref,
        compared_scope_refs=(scope_ref,),
        plausible_naics_codes=(provider_code,),
        outcome=ClassificationReviewOutcome.UNRESOLVED,
        analysis="The bundle crosses plausible scopes without a declared revenue mix.",
        missing_evidence=("Primary revenue activity and revenue mix are unknown.",),
    )
    reviewer = FakeStructuredGenerator([reply])

    execution = review_classification_comparative(
        candidate=candidate,
        packet=packet,
        assignment_id="classifier-mixed",
        llm=reviewer,
    )
    result = execution.review

    assert result.outcome is ClassificationReviewOutcome.UNRESOLVED
    rendered = "\n".join(message.content for message in reviewer.calls[0][0])
    assert title in rendered
    assert mechanism in rendered
    assert expected_rule in rendered
    assert "without a declared or measured primary revenue activity and revenue mix" in rendered


def _fire_comparison_packet() -> EvidencePacket:
    scope_text = {
        "238220": "Fire-suppression system installation and repair.",
        "541380": "Physical testing and calibration services.",
        "541990": "Fire-system inspection or testing only when no service is performed.",
        "811310": "Commercial equipment repair and maintenance without installation.",
    }
    scopes = tuple(
        PacketMeasurement(
            measurement_id=f"naics22-{code}-scope",
            metric=f"Official NAICS scope for {code}",
            value=value,
            unit="official scope text",
            geography="United States",
            observed_period="2022",
            source_family="Census NAICS",
            source_url="https://www.census.gov/naics/",
            caveat="Classification scope is not evidence of demand or revenue mix.",
            quality_flags=("classification_only", "not_economic_measurement"),
        )
        for code, value in scope_text.items()
    )
    return EvidencePacket(
        packet_id="packet-fire-comparison",
        as_of=NOW,
        measurements=(
            PacketMeasurement(
                measurement_id="ca-fire-904-recurring-itm",
                metric="Required recurring inspection, testing, and maintenance",
                value=True,
                unit="regulatory status",
                geography="United States",
                observed_period="2026",
                source_family="California regulation",
                source_url="https://osfm.fire.ca.gov/",
                caveat="A mandate does not establish provider classification or revenue mix.",
            ),
            *scopes,
        ),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("238220", "541380", "541990", "811310"),
        source_policy="Official administrative records.",
    )


def _fire_comparison_candidate(
    *,
    packet: EvidencePacket,
    provider_code: str,
    title: str,
    mechanism: str,
    exclusions: tuple[str, ...],
) -> CandidateHypothesis:
    draft = (
        _batch()
        .hypotheses[0]
        .model_copy(
            update={
                "title": title,
                "mechanism": mechanism,
                "naics_codes": (provider_code,),
                "customer_naics_codes": (),
                "naics_basis": "The proposed code requires comparison across fire activities.",
                "offer_market_topic": MarketTopic.FIRE_LIFE_SAFETY,
                "context_market_topics": (),
                "adjacent_market_exclusions": exclusions,
                "evidence_refs": (
                    f"naics22-{provider_code}-scope",
                    "ca-fire-904-recurring-itm",
                ),
            }
        )
    )
    return materialize_hypothesis(
        draft,
        packet=packet,
        assignment_id="researcher-fire-comparison",
        created_at=NOW,
    )


def test_generation_schema_uses_explicit_customer_allowlist_not_provider_codes() -> None:
    base = _fire_comparison_packet()
    customer_scopes = tuple(
        PacketMeasurement(
            measurement_id=f"naics22-{code}-scope",
            metric=f"Official customer scope for {code}",
            value=f"Customer definition for {code}.",
            unit="official scope text",
            geography="United States",
            observed_period="2022",
            source_family="Census NAICS",
            source_url="https://www.census.gov/naics/",
            caveat="Customer classification is not demand.",
            quality_flags=(
                "classification_only",
                "customer_class_not_demand",
                "customer_eligible",
            ),
        )
        for code in ("541714", "623312")
    )
    packet = base.model_copy(
        update={
            "measurements": (*base.measurements, *customer_scopes),
            "allowed_naics_codes": (
                "238220",
                "541380",
                "541714",
                "541990",
                "623312",
                "811310",
            ),
        }
    )
    draft = (
        _batch()
        .hypotheses[0]
        .model_copy(
            update={
                "title": "Repair-only fire service for senior living facilities",
                "mechanism": "Sell repair and maintenance without installation.",
                "naics_codes": ("811310",),
                "customer_naics_codes": ("623312",),
                "offer_market_topic": MarketTopic.FIRE_LIFE_SAFETY,
                "context_market_topics": (MarketTopic.SENIOR_LIVING_FACILITIES,),
                "evidence_refs": (
                    "naics22-811310-scope",
                    "naics22-623312-scope",
                ),
            }
        )
    )
    generator = FakeStructuredGenerator([HypothesisBatch(hypotheses=(draft,))])

    result = generate_hypotheses(
        packet=packet,
        assignment_id="researcher-explicit-customers",
        created_at=NOW,
        llm=generator,
        max_hypotheses=1,
    )

    assert result[0].customer_naics_codes == ("623312",)
    draft_schema = generator.calls[0][1].model_json_schema()["$defs"][
        "PacketBoundedHypothesisDraft"
    ]
    assert draft_schema["properties"]["naics_codes"]["items"]["enum"] == [
        "238220",
        "541380",
        "541990",
        "811310",
    ]
    assert draft_schema["properties"]["customer_naics_codes"]["items"]["enum"] == [
        "541714",
        "623312",
    ]
    rendered = "\n".join(message.content for message in generator.calls[0][0])
    assert "explicit customer allowlist" in rendered
    assert "never reuse a provider code as a buyer" in rendered


def test_targeted_mixed_fire_bundle_is_unresolved_across_all_provider_scopes() -> None:
    packet = _fire_comparison_packet()
    candidate = _fire_comparison_candidate(
        packet=packet,
        provider_code="238220",
        title="Fixed-system inspection, testing, maintenance, and deficiency correction",
        mechanism=(
            "Sell fixed-system inspection and testing, recurring maintenance, and "
            "deficiency correction under one contract."
        ),
        exclusions=("Portable extinguisher sales",),
    )
    scopes = classification_scope_measurements(packet)
    reply = ComparativeClassificationReply(
        opportunity_id=candidate.opportunity_id,
        assignment_id="classifier-fire-mixed",
        naics_code="238220",
        scope_measurement_ref="naics22-238220-scope",
        compared_scope_refs=(
            "naics22-238220-scope",
            "naics22-541380-scope",
            "naics22-541990-scope",
            "naics22-811310-scope",
        ),
        plausible_naics_codes=("238220", "541990", "811310"),
        outcome=ClassificationReviewOutcome.UNRESOLVED,
        analysis=(
            "Inspection-only, maintenance, and deficiency-correction activities map "
            "to multiple supplied scopes."
        ),
        missing_evidence=("Primary revenue activity and activity-level revenue mix.",),
    )
    reviewer = FakeStructuredGenerator([reply])

    execution = review_classification_comparative(
        candidate=candidate,
        packet=packet,
        assignment_id="classifier-fire-mixed",
        llm=reviewer,
    )
    result = execution.review

    assert result.outcome is ClassificationReviewOutcome.UNRESOLVED
    assert execution.comparison.plausible_naics_codes == ("238220", "541990", "811310")
    assert execution.comparison.schema_version == "classification-review-v2"
    assert execution.comparison.legacy_review() == execution.review
    assert result.model_dump().keys() == ClassificationReview.model_fields.keys()
    assert tuple(scope.measurement_id for scope in scopes) == (
        "naics22-238220-scope",
        "naics22-541380-scope",
        "naics22-541990-scope",
        "naics22-811310-scope",
    )
    messages, schema = reviewer.calls[0]
    rendered = "\n".join(message.content for message in messages)
    for scope in scopes:
        expected_mentions = 2 if scope.measurement_id == "naics22-238220-scope" else 1
        assert rendered.count(scope.measurement_id) == expected_mentions
    assert '"schema_version":"classification-input-v2"' in rendered
    plausible_schema = schema.model_json_schema()["properties"]["plausible_naics_codes"]
    assert plausible_schema["items"]["enum"] == [
        "238220",
        "541380",
        "541990",
        "811310",
    ]
    assert "uniquely plausible among the supplied alternatives" in rendered


def test_mixed_fire_bundle_cannot_be_marked_fit_with_multiple_plausible_codes() -> None:
    packet = _fire_comparison_packet()
    candidate = _fire_comparison_candidate(
        packet=packet,
        provider_code="238220",
        title="Mixed fixed-system service",
        mechanism="Sell inspection, maintenance, repair, and installation together.",
        exclusions=("Portable extinguisher sales",),
    )
    with pytest.raises(ValidationError, match="uniquely plausible"):
        ComparativeClassificationReply(
            opportunity_id=candidate.opportunity_id,
            assignment_id="classifier-fire-invalid-fit",
            naics_code="238220",
            scope_measurement_ref="naics22-238220-scope",
            compared_scope_refs=(
                "naics22-238220-scope",
                "naics22-541380-scope",
                "naics22-541990-scope",
                "naics22-811310-scope",
            ),
            plausible_naics_codes=("238220", "541990", "811310"),
            outcome=ClassificationReviewOutcome.FIT,
            analysis="The proposed code is one of several plausible scopes.",
        )


def test_repair_only_811310_offer_remains_uniquely_fit() -> None:
    packet = _fire_comparison_packet()
    candidate = _fire_comparison_candidate(
        packet=packet,
        provider_code="811310",
        title="Fire-system repair and maintenance without installation",
        mechanism=(
            "Sell commercial fire-system repair and maintenance only, without "
            "installation, sales, or inspection-only testing."
        ),
        exclusions=(
            "Installation",
            "Equipment sales",
            "Inspection-only testing",
        ),
    )
    reply = ComparativeClassificationReply(
        opportunity_id=candidate.opportunity_id,
        assignment_id="classifier-repair-only",
        naics_code="811310",
        scope_measurement_ref="naics22-811310-scope",
        compared_scope_refs=(
            "naics22-238220-scope",
            "naics22-541380-scope",
            "naics22-541990-scope",
            "naics22-811310-scope",
        ),
        plausible_naics_codes=("811310",),
        outcome=ClassificationReviewOutcome.FIT,
        analysis="Only the supplied repair-without-installation scope matches the sold offer.",
    )

    result = review_classification(
        candidate=candidate,
        packet=packet,
        assignment_id="classifier-repair-only",
        llm=FakeStructuredGenerator([reply]),
    )

    assert result.outcome is ClassificationReviewOutcome.FIT
    assert "plausible_naics_codes" not in result.model_dump()


def test_classification_input_v1_is_stable_and_v2_contains_all_comparison_rows() -> None:
    packet = _fire_comparison_packet()
    candidate = _fire_comparison_candidate(
        packet=packet,
        provider_code="811310",
        title="Repair-only offer",
        mechanism="Sell repair and maintenance without installation.",
        exclusions=("Installation", "Inspection-only testing"),
    )
    scope = next(
        item
        for item in classification_scope_measurements(packet)
        if item.measurement_id == "naics22-811310-scope"
    )
    legacy = classification_input_payload(
        candidate=candidate,
        scope=scope,
        assignment_id="classifier-input-version",
    )

    assert legacy == classification_input_payload_v1(
        candidate=candidate,
        scope=scope,
        assignment_id="classifier-input-version",
    )
    assert "schema_version" not in legacy
    assert "official_scope_measurement" in legacy
    comparative = classification_input_payload_v2(
        candidate=candidate,
        scopes=classification_scope_measurements(packet),
        assignment_id="classifier-input-version",
    )
    assert comparative["schema_version"] == "classification-input-v2"
    assert comparative["required_scope_measurement_ref"] == "naics22-811310-scope"
    official_scopes = comparative["official_scope_measurements"]
    assert isinstance(official_scopes, list)
    assert len(official_scopes) == 4


def test_classification_review_budget_fails_before_model_call() -> None:
    packet = _classification_packet()
    generator = FakeStructuredGenerator([_classification_batch()])
    candidate = generate_hypotheses(
        packet=packet,
        assignment_id="researcher-1",
        created_at=NOW,
        llm=generator,
        max_hypotheses=1,
    )[0]
    reviewer = FakeStructuredGenerator([])

    with pytest.raises(BudgetExceededError):
        review_classification(
            candidate=candidate,
            packet=packet,
            assignment_id="classifier-1",
            llm=reviewer,
            budget=BudgetGuard(BudgetPolicy(max_model_calls=0)),
        )

    assert reviewer.calls == []


def test_generation_keeps_evidence_out_of_the_system_message() -> None:
    fake = FakeStructuredGenerator([_batch()])

    result = generate_hypotheses(
        packet=_packet(),
        assignment_id="researcher-1",
        created_at=NOW,
        llm=fake,
        max_hypotheses=1,
    )

    assert len(result) == 1
    messages, schema = fake.calls[0]
    assert issubclass(schema, HypothesisBatch)
    evidence_items = schema.model_json_schema()["$defs"]["PacketBoundedHypothesisDraft"][
        "properties"
    ]["evidence_refs"]["items"]
    assert evidence_items["enum"] == [
        "cbp-precision-repair-m-2",
        "usa-precision-repair-m-1",
    ]
    draft_schema = schema.model_json_schema()["$defs"]["PacketBoundedHypothesisDraft"]
    assert draft_schema["properties"]["naics_codes"]["items"]["const"] == "811210"
    assert draft_schema["properties"]["naics_codes"]["maxItems"] == 1
    assert draft_schema["properties"]["customer_naics_codes"]["items"]["const"] == "811210"
    assert draft_schema["properties"]["offer_market_topic"]["const"] == "equipment_service"
    assert (
        draft_schema["properties"]["context_market_topics"]["items"]["const"] == "equipment_service"
    )
    assert draft_schema["properties"]["context_market_topics"]["maxItems"] == 2
    assert (
        draft_schema["properties"]["classification_status"]["const"]
        == ClassificationStatus.PROVISIONAL
    )
    assert messages[0].role is Role.SYSTEM
    assert "usa-precision-repair-m-1" not in messages[0].content
    assert "usa-precision-repair-m-1" in messages[1].content
    assert "MARKET_TOPIC_BINDINGS" in messages[1].content
    assert '"usa-precision-repair-m-1":["equipment_service"]' in messages[1].content
    assert "recipient-NAICS obligations" in messages[1].content
    assert "buyer" in messages[1].content
    assert "Candidates may share a provider NAICS" in messages[1].content
    assert "Use each provider NAICS at most once" not in messages[1].content


def test_generation_accepts_distinct_same_provider_hypotheses() -> None:
    first = _batch().hypotheses[0]
    second = first.model_copy(
        update={
            "title": "Onsite regulated-device repair subscription",
            "customer": "Regional clinical engineering teams",
            "payer": "Clinical engineering director",
            "problem": "Unscheduled device failures interrupt clinical workflows.",
            "mechanism": "Dispatch qualified repair technicians under a response-time agreement.",
            "business_model": "Monthly availability fee plus per-repair charges.",
            "critical_assumptions": (
                "Clinical engineering teams will pay to shorten repair response time.",
            ),
            "disconfirming_observations": (
                "Existing OEM contracts already meet the promised response time.",
            ),
        }
    )
    batch = HypothesisBatch(hypotheses=(first, second))
    fake = FakeStructuredGenerator([batch])

    candidates = generate_hypotheses(
        packet=_packet(),
        assignment_id="researcher-same-provider",
        created_at=NOW,
        llm=fake,
        max_hypotheses=2,
    )

    assert [item.naics_codes for item in candidates] == [("811210",), ("811210",)]
    assert candidates[0].opportunity_id != candidates[1].opportunity_id
    assert candidates[0].thesis_id != candidates[1].thesis_id


def test_budget_fails_before_the_model_is_called() -> None:
    fake = FakeStructuredGenerator([_batch()])
    budget = BudgetGuard(BudgetPolicy(max_model_calls=0))

    with pytest.raises(BudgetExceededError):
        generate_hypotheses(
            packet=_packet(),
            assignment_id="researcher-1",
            created_at=NOW,
            llm=fake,
            max_hypotheses=1,
            budget=budget,
        )

    assert fake.calls == []


def test_kill_switch_fails_before_the_model_is_called(tmp_path: Path) -> None:
    fake = FakeStructuredGenerator([_batch()])
    switch = KillSwitch(tmp_path)
    switch.path.write_text("stop", encoding="utf-8")

    with pytest.raises(BudgetExceededError, match="kill switch"):
        generate_hypotheses(
            packet=_packet(),
            assignment_id="researcher-1",
            created_at=NOW,
            llm=fake,
            max_hypotheses=1,
            kill_switch=switch,
        )

    assert fake.calls == []


def test_falsifier_never_receives_rank_or_preferred_outcome() -> None:
    generator = FakeStructuredGenerator([_batch()])
    candidate = generate_hypotheses(
        packet=_packet(),
        assignment_id="researcher-1",
        created_at=NOW,
        llm=generator,
        max_hypotheses=1,
    )[0]
    falsifier = FakeStructuredGenerator([_report(candidate.opportunity_id)])

    report = falsify_hypothesis(
        candidate=candidate,
        packet=_packet(),
        assignment_id="falsifier-1",
        llm=falsifier,
    )

    assert report.opportunity_id == candidate.opportunity_id
    messages, schema = falsifier.calls[0]
    assert issubclass(schema, FalsificationReport)
    report_schema = schema.model_json_schema()
    assert report_schema["properties"]["opportunity_id"]["const"] == candidate.opportunity_id
    finding_schema = report_schema["$defs"]["PacketBoundedFalsificationFinding"]
    assert finding_schema["properties"]["evidence_refs"]["items"]["enum"] == [
        "cbp-precision-repair-m-2",
        "usa-precision-repair-m-1",
    ]
    rendered = "\n".join(message.content for message in messages)
    assert "current_rank" not in rendered
    assert "preferred_outcome" not in rendered
    assert candidate.origin_assignment_id not in rendered
    assert "cross-market analogy" in rendered
    assert '"offer_market_topic":"equipment_service"' in rendered
    assert '"context_market_topics":[]' in rendered
    assert "absence of disconfirming evidence" in rendered
    assert "recipient-NAICS obligations" in rendered


def test_falsifier_provider_enum_is_bound_to_candidate_scoped_packet() -> None:
    base = _packet()
    packet = base.model_copy(
        update={
            "measurements": (
                *base.measurements,
                PacketMeasurement(
                    measurement_id="unknown-repair-private-wtp",
                    metric="direct private willingness to pay",
                    value=None,
                    unit="validated market status",
                    geography="United States",
                    observed_period="unknown as of 2026",
                    source_family="official gap register",
                    source_url="https://example.gov/repair",
                    caveat="No direct observation is present.",
                    quality_flags=("counterevidence", "unknown_not_zero"),
                ),
                PacketMeasurement(
                    measurement_id="ca-fire-9041-owner-inspection",
                    metric="owner inspection exception",
                    value=True,
                    unit="regulatory status",
                    geography="United States",
                    observed_period="2026",
                    source_family="California regulation",
                    source_url="https://example.gov/fire",
                    caveat="This is a fire-market rule, not equipment-repair evidence.",
                    quality_flags=("counterevidence",),
                ),
            )
        }
    )
    generator = FakeStructuredGenerator([_batch()])
    candidate = generate_hypotheses(
        packet=packet,
        assignment_id="researcher-1",
        created_at=NOW,
        llm=generator,
        max_hypotheses=1,
    )[0]
    falsifier = FakeStructuredGenerator([_report(candidate.opportunity_id)])

    falsify_hypothesis(
        candidate=candidate,
        packet=packet,
        assignment_id="falsifier-1",
        llm=falsifier,
    )

    messages, schema = falsifier.calls[0]
    finding_schema = schema.model_json_schema()["$defs"]["PacketBoundedFalsificationFinding"]
    assert finding_schema["properties"]["evidence_refs"]["items"]["enum"] == [
        "cbp-precision-repair-m-2",
        "unknown-repair-private-wtp",
        "usa-precision-repair-m-1",
    ]
    rendered = "\n".join(message.content for message in messages)
    assert "unknown-repair-private-wtp" in rendered
    assert "ca-fire-9041-owner-inspection" not in rendered
