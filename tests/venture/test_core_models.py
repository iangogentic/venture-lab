"""Strict evidence contracts and independence boundaries."""

from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.venture.core import (
    Assignment,
    AssignmentRole,
    Claim,
    ClaimCriticality,
    ClaimEvidence,
    ClaimStatus,
    ClaimType,
    EffectivePeriod,
    EvidenceKind,
    EvidenceRelationship,
    Geography,
    GeographySystem,
    Measurement,
    MeasurementDomain,
    MeasurementSource,
    MeasurementTime,
    MeasurementValue,
    Verification,
    VerificationChecks,
    VerificationResult,
    assess_claim_evidence,
    make_content_id,
    require_verifier_separation,
    sources_are_independent,
    validate_identifier,
)

NOW = datetime(2026, 7, 31, 12, tzinfo=UTC)


def _geography() -> Geography:
    return Geography(
        system=GeographySystem.FIPS,
        level="state",
        code="48",
        vintage="2020",
    )


def _measurement(
    identifier: str,
    *,
    source_id: str,
    family: str,
    kind: EvidenceKind = EvidenceKind.ADMINISTRATIVE_RECORD,
) -> Measurement:
    return Measurement(
        id=identifier,
        domain=MeasurementDomain.CAPACITY,
        metric="licensed beds",
        evidence_kind=kind,
        source=MeasurementSource(
            source_id=source_id,
            publisher="Texas HHS",
            dataset="facility census",
            release_version="2026-07",
            source_url="https://example.gov/facilities",
            raw_content_hash="a" * 64,
            source_family=family,
        ),
        time=MeasurementTime(
            observed_start=date(2026, 1, 1),
            observed_end=date(2026, 6, 30),
            retrieved_at=NOW,
        ),
        geography=_geography(),
        universe="licensed facilities",
        value=MeasurementValue(estimate=120, unit="beds"),
    )


def _claim() -> Claim:
    return Claim(
        claim_id="claim_1",
        opportunity_id="opportunity_1",
        statement="Licensed capacity is below observed need.",
        claim_type=ClaimType.SUPPLY,
        criticality=ClaimCriticality.CRITICAL,
        value=120,
        unit="beds",
        geography=_geography(),
        effective_period=EffectivePeriod(start=date(2026, 1, 1), end=date(2026, 6, 30)),
        status=ClaimStatus.PROPOSED,
        created_by_assignment="assignment_research",
    )


def _assignment(identifier: str, role: AssignmentRole) -> Assignment:
    return Assignment(
        assignment_id=identifier,
        role=role,
        blind_packet_id=f"packet_{identifier}",
        actor_model="gpt-test",
        prompt_sha256="b" * 64,
        visible_fields=("claims", "evidence_refs"),
        started_at=NOW,
        completed_at=NOW + timedelta(minutes=1),
    )


def _verification(assignment_id: str) -> Verification:
    return Verification(
        verification_id="verification_1",
        claim_id="claim_1",
        verifier_assignment_id=assignment_id,
        result=VerificationResult.PASS,
        checks=VerificationChecks(
            snapshot_hash_matches=True,
            locator_resolves=True,
            value_reproduced=True,
            unit_matches=True,
            geography_matches=True,
            time_scope_matches=True,
            source_independence_valid=True,
            cutoff_valid=True,
        ),
    )


@pytest.mark.parametrize(
    "unsafe",
    ("../escape", "folder/name", r"folder\name", "/absolute", ".", "..", "contains space"),
)
def test_identifiers_reject_path_traversal_and_unsafe_segments(unsafe: str) -> None:
    with pytest.raises(ValueError):
        validate_identifier(unsafe)


def test_measurement_rejects_extra_fields_and_naive_retrieval_time() -> None:
    payload = _measurement("measurement_1", source_id="source_1", family="family_1").model_dump()
    payload["invented_score"] = 0

    with pytest.raises(ValidationError, match="extra_forbidden"):
        Measurement.model_validate(payload)

    time_payload = payload["time"]
    assert isinstance(time_payload, dict)
    time_payload["retrieved_at"] = datetime(2026, 1, 1)
    payload.pop("invented_score")
    with pytest.raises(ValidationError, match="timezone"):
        Measurement.model_validate(payload)


def test_content_ids_are_deterministic_and_order_independent() -> None:
    left = make_content_id("measurement", {"metric": "beds", "value": 12})
    right = make_content_id("measurement", {"value": 12, "metric": "beds"})

    assert left == right
    assert left.startswith("measurement_")


def test_independence_counts_source_families_not_rows_or_source_ids() -> None:
    first = _measurement("measurement_1", source_id="source_a", family="shared_parent")
    second = _measurement("measurement_2", source_id="source_b", family="shared_parent")
    independent = _measurement("measurement_3", source_id="source_c", family="other_parent")

    assert not sources_are_independent([first, second])
    assert sources_are_independent([first, independent])

    links = (
        ClaimEvidence(
            claim_id="claim_1",
            measurement_id=first.id,
            relationship=EvidenceRelationship.SUPPORTS,
            transform_id="transform_1",
        ),
        ClaimEvidence(
            claim_id="claim_1",
            measurement_id=second.id,
            relationship=EvidenceRelationship.SUPPORTS,
            transform_id="transform_2",
        ),
    )
    assessment = assess_claim_evidence(_claim(), links, (first, second))
    assert assessment.independent_source_families == 1
    assert assessment.has_primary_administrative_or_behavioral is True


def test_verifier_must_be_a_separate_role_assignment() -> None:
    claim = _claim()
    researcher = _assignment("assignment_research", AssignmentRole.RESEARCHER)
    verifier = _assignment("assignment_verify", AssignmentRole.VERIFIER)

    require_verifier_separation(
        claim,
        _verification(verifier.assignment_id),
        [researcher, verifier],
    )

    with pytest.raises(ValueError, match="separate verifier"):
        require_verifier_separation(
            claim,
            _verification(researcher.assignment_id),
            [researcher],
        )


def test_passing_verification_cannot_hide_a_failed_check() -> None:
    with pytest.raises(ValidationError, match="every boolean check"):
        Verification(
            verification_id="verification_1",
            claim_id="claim_1",
            verifier_assignment_id="assignment_verify",
            result=VerificationResult.PASS,
            checks=VerificationChecks(
                snapshot_hash_matches=True,
                locator_resolves=True,
                value_reproduced=False,
                unit_matches=True,
                geography_matches=True,
                time_scope_matches=True,
                source_independence_valid=True,
                cutoff_valid=True,
            ),
        )
