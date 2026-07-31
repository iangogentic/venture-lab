"""Coverage and determinism tests for the frozen pilot evidence catalog."""

import json

from app.venture.core import Scenario
from app.venture.discovery import EvidencePacket
from app.venture.pilot_evidence import (
    PILOT_ALLOWED_NAICS_CODES,
    PILOT_EXCLUDED_NAICS_CODES,
    build_pilot_evidence_packet,
    pilot_evidence_packet_json,
)


def test_pilot_packet_is_deterministic_and_round_trips_as_json() -> None:
    first = build_pilot_evidence_packet()
    second = build_pilot_evidence_packet()

    assert first == second
    assert pilot_evidence_packet_json(first) == pilot_evidence_packet_json(second)
    assert pilot_evidence_packet_json(first).endswith("\n")
    restored = EvidencePacket.model_validate(json.loads(pilot_evidence_packet_json(first)))
    assert restored == first


def test_pilot_packet_has_unique_measurement_ids_and_all_capital_scenarios() -> None:
    packet = build_pilot_evidence_packet()
    ids = [measurement.measurement_id for measurement in packet.measurements]

    assert len(ids) == len(set(ids))
    assert set(packet.allowed_scenarios) == set(Scenario)
    assert "United States" in packet.allowed_geographies
    assert "California" in packet.allowed_geographies
    assert packet.allowed_naics_codes == PILOT_ALLOWED_NAICS_CODES
    assert "no weighted master score" in packet.source_policy


def test_pilot_packet_covers_required_official_source_families() -> None:
    packet = build_pilot_evidence_packet()
    families = {measurement.source_family for measurement in packet.measurements}

    assert {
        "BLS Business Employment Dynamics",
        "CMS Provider Data",
        "Census 2022 NAICS Classification",
        "Census County Business Patterns",
        "Census Economic Census",
        "IRS Statistics of Income - Corporation",
        "IRS Statistics of Income - Sole Proprietorship",
        "USAspending",
        "California Code of Regulations",
        "California Contractors State License Board",
        "California CSLB + Census County Business Patterns",
    }.issubset(families)


def test_pilot_naics_allowlist_is_exact_and_classification_backed() -> None:
    packet = build_pilot_evidence_packet()
    expected = {
        "236220",
        "238210",
        "238220",
        "423450",
        "513210",
        "541380",
        "541614",
        "541620",
        "541990",
        "561611",
        "561621",
        "562119",
        "621111",
        "621610",
        "623110",
        "624120",
        "811210",
        "811310",
    }
    classification_ids = {
        measurement.measurement_id
        for measurement in packet.measurements
        if measurement.source_family == "Census 2022 NAICS Classification"
    }

    assert set(packet.allowed_naics_codes) == expected
    assert set(PILOT_EXCLUDED_NAICS_CODES).isdisjoint(packet.allowed_naics_codes)
    assert {f"naics22-{code}-scope" for code in packet.allowed_naics_codes}.issubset(
        classification_ids
    )


def test_fire_hypotheses_have_activity_specific_alternative_codes() -> None:
    packet = build_pilot_evidence_packet()
    by_id = {measurement.measurement_id: measurement for measurement in packet.measurements}
    fire_alternatives = {
        "238210",
        "238220",
        "541990",
        "561621",
        "811310",
    }

    assert fire_alternatives.issubset(packet.allowed_naics_codes)
    assert "fire_alarm_installation_only" in by_id["naics22-238210-scope"].quality_flags
    assert "fire_suppression_installation" in by_id["naics22-238220-scope"].quality_flags
    assert "fire_testing_inspection_only" in by_id["naics22-541990-scope"].quality_flags
    assert "alarm_bundle_or_monitoring" in by_id["naics22-561621-scope"].quality_flags
    assert (
        "fire_suppression_repair_without_installation"
        in by_id["naics22-811310-scope"].quality_flags
    )
    for code in fire_alternatives:
        measurement = by_id[f"naics22-{code}-scope"]
        assert "classification_only" in measurement.quality_flags
        assert "not_economic_measurement" in measurement.quality_flags


def test_561790_is_explicitly_excluded_for_fire_compliance() -> None:
    packet = build_pilot_evidence_packet()
    by_id = {measurement.measurement_id: measurement for measurement in packet.measurements}
    excluded = by_id["naics22-561790-fire-excluded"]

    assert "561790" not in packet.allowed_naics_codes
    assert PILOT_EXCLUDED_NAICS_CODES == ("561790",)
    assert excluded.value is False
    assert "classification_exclusion" in excluded.quality_flags
    assert "fire_compliance_not_supported" in excluded.quality_flags
    assert "conservative pilot policy exclusion" in excluded.caveat


def test_every_measurement_has_explicit_source_period_caveat_and_flags() -> None:
    packet = build_pilot_evidence_packet()

    for measurement in packet.measurements:
        assert measurement.source_url.startswith("https://")
        assert measurement.observed_period
        assert measurement.geography
        assert measurement.unit
        assert measurement.caveat
        assert "official_primary" in measurement.quality_flags
        assert len(measurement.quality_flags) == len(set(measurement.quality_flags))


def test_material_proxy_boundaries_are_preserved() -> None:
    packet = build_pilot_evidence_packet()
    by_id = {measurement.measurement_id: measurement for measurement in packet.measurements}

    assert "contestable_spend_unknown" in by_id["ca-fire-904-recurring-itm"].quality_flags
    assert "not_live_availability" in by_id["cms-ca-load-proxy"].quality_flags
    assert "unknown_not_zero" in by_id["cms-ca-live-vacancy-unknown"].quality_flags
    assert "establishment_not_firm" in by_id["bls-bed-62-2018-survival"].quality_flags
    assert "same_cohort_pair" in by_id["bls-bed-62-2018-survival"].quality_flags
    assert by_id["bls-bed-81-2020-survival"].value == "85.2/60.0"
    assert "naics_crosswalk_required" in by_id["cbp-precision-repair-est-growth"].quality_flags
    assert (
        "receipts_not_startup_revenue" in by_id["ec22-precision-repair-receipts-est"].quality_flags
    )
    assert "owner_labor_not_expensed" in by_id["irs-sp-repair-ca-net-profit-ratio"].quality_flags
    assert "unknown_not_zero" in by_id["usa-background-growth"].quality_flags


def test_packet_contains_counterevidence_and_direct_unknowns() -> None:
    packet = build_pilot_evidence_packet()

    counterevidence = [
        measurement
        for measurement in packet.measurements
        if "counterevidence" in measurement.quality_flags
    ]
    unknowns = [
        measurement
        for measurement in packet.measurements
        if measurement.value is None and "unknown_not_zero" in measurement.quality_flags
    ]

    assert {measurement.measurement_id for measurement in counterevidence} >= {
        "cbp-home-health-est-growth",
        "cbp-other-waste-est-growth",
        "cbp-security-systems-est-growth",
        "ca-fire-9041-owner-inspection",
        "cms-ca-residents-over-beds-anomalies",
    }
    assert {measurement.measurement_id for measurement in unknowns} >= {
        "ca-fire-outsourced-spend-unknown",
        "cms-ca-live-vacancy-unknown",
        "unknown-aging-wtp-workforce",
        "unknown-repair-private-wtp",
    }


def test_c16_supply_snapshot_preserves_components_and_derived_ratios() -> None:
    packet = build_pilot_evidence_packet()
    by_id = {measurement.measurement_id: measurement for measurement in packet.measurements}
    county_facts = {
        "kings": (0, 275, 0.0),
        "yolo": (5, 756, 6.61),
        "san-diego": (93, 12_610, 7.38),
        "merced": (4, 512, 7.81),
        "tulare": (10, 1_081, 9.25),
        "santa-clara": (76, 7_603, 10.00),
        "fresno": (31, 2_998, 10.34),
        "santa-barbara": (18, 1_699, 10.59),
        "stanislaus": (17, 1_551, 10.96),
        "imperial": (5, 387, 12.92),
    }

    assert packet.as_of.isoformat() == "2026-07-31T08:15:00+00:00"
    assert by_id["cslb-c16-clear-total"].value == 2218
    assert by_id["cslb-c16-clear-ca-address"].value == 2115
    assert by_id["cslb-c16-clear-out-state"].value == 103
    assert by_id["cslb-c16-ca-counties"].value == 49
    assert (
        "8b5dec0ab82188be688da355812decb30040c2437871ab3fc7e4f5862cdd6e0c"
        in by_id["cslb-c16-clear-total"].caveat
    )

    for slug, (addresses, establishments, ratio) in county_facts.items():
        assert by_id[f"cslb-c16-{slug}-addresses"].value == addresses
        assert by_id[f"cbp-{slug}-establishments-ge20"].value == establishments
        derived = by_id[f"derived-c16-{slug}-per1000-ge20"]
        assert derived.value == ratio
        assert "screening_proxy" in derived.quality_flags
        assert "counterevidence" in derived.quality_flags
        assert "address_not_service_territory" in derived.quality_flags
        assert "denominator_not_installed_base" in derived.quality_flags


def test_c16_zero_is_address_count_not_supply_absence() -> None:
    packet = build_pilot_evidence_packet()
    by_id = {measurement.measurement_id: measurement for measurement in packet.measurements}

    kings_count = by_id["cslb-c16-kings-addresses"]
    kings_ratio = by_id["derived-c16-kings-per1000-ge20"]
    assert kings_count.value == 0
    assert "zero_address_count_not_supply_absence" in kings_count.quality_flags
    assert "zero_address_count_not_supply_absence" in kings_ratio.quality_flags
    assert "not proof" in kings_count.caveat
