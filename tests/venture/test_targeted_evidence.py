"""Coverage and boundary tests for the narrowed targeted evidence packet."""

import json

from app.venture.core import Scenario
from app.venture.discovery import (
    EvidencePacket,
    PacketMeasurement,
    available_market_topics,
    evidence_topics,
    provider_naics_codes,
)
from app.venture.targeted_evidence import (
    TARGETED_ALLOWED_NAICS_CODES,
    build_targeted_evidence_packet,
    targeted_evidence_packet_json,
)


def _by_id() -> dict[str, PacketMeasurement]:
    packet = build_targeted_evidence_packet()
    return {measurement.measurement_id: measurement for measurement in packet.measurements}


def test_targeted_packet_is_frozen_deterministic_and_round_trips() -> None:
    first = build_targeted_evidence_packet()
    second = build_targeted_evidence_packet()

    assert first == second
    assert first.as_of.isoformat() == "2026-07-31T12:00:00+00:00"
    assert targeted_evidence_packet_json(first) == targeted_evidence_packet_json(second)
    assert targeted_evidence_packet_json(first).endswith("\n")
    restored = EvidencePacket.model_validate(json.loads(targeted_evidence_packet_json(first)))
    assert restored == first


def test_targeted_packet_has_unique_critical_measurements() -> None:
    packet = build_targeted_evidence_packet()
    ids = [measurement.measurement_id for measurement in packet.measurements]
    critical = {
        "cbp-sd-biotech-rd-establishments-2023",
        "cbp-sd-biotech-rd-establishments-20-249-2023",
        "cbp-sd-biological-product-establishments-2023",
        "cbp-sd-medical-lab-establishments-2023",
        "cbp-sd-testing-lab-establishments-2023",
        "bmbl-bsc-certification-cadence",
        "ucsd-bsc-annual-certification-policy",
        "nsf-sd-bsc-current-certifiers",
        "nsf-sd-bsc-employer-address-groups",
        "omnia-bsc-nsf49-certification-price-range",
        "omnia-bsc-pm-price-range",
        "unknown-sd-bsc-installed-base",
        "unknown-sd-bsc-provider-backlog",
        "unknown-sd-bsc-private-wtp",
        "cdss-sd-licensed-rcfe-ccrc-records",
        "cdss-sd-licensed-rcfe-ccrc-capacity",
        "ca-osfm-fixed-system-semiannual-itm",
        "ca-osfm-aes-service-license-paths",
        "ca-osfm-aes-liability-insurance",
        "ca-osfm-aes-records-invoice-deficiencies",
        "sdfd-compliance-engine-reporting",
        "cslb-sd-clear-c16-addresses",
        "socal-portable-extinguisher-service-call-price",
        "socal-portable-extinguisher-per-unit-price",
        "sd-city-kitchen-system-itm-unit-price-2017",
        "unknown-sd-fire-provider-backlog",
        "unknown-sd-fire-outsourced-share",
        "unknown-sd-fire-private-wtp",
    }

    assert len(ids) == len(set(ids))
    assert critical.issubset(ids)
    assert packet.allowed_scenarios == (
        Scenario.BOOTSTRAPPED,
        Scenario.OPERATOR_HEAVY,
    )


def test_targeted_naics_allowlist_is_exact_unique_and_classification_backed() -> None:
    packet = build_targeted_evidence_packet()
    expected = {
        "238220",
        "541380",
        "541714",
        "541990",
        "623312",
        "811210",
        "811310",
    }
    classification_ids = {
        measurement.measurement_id
        for measurement in packet.measurements
        if measurement.source_family == "Census 2022 NAICS Classification"
    }

    assert set(TARGETED_ALLOWED_NAICS_CODES) == expected
    assert packet.allowed_naics_codes == TARGETED_ALLOWED_NAICS_CODES
    assert len(packet.allowed_naics_codes) == len(set(packet.allowed_naics_codes))
    assert all(code.isdigit() and len(code) == 6 for code in packet.allowed_naics_codes)
    assert {f"naics22-{code}-scope" for code in expected} == classification_ids
    assert provider_naics_codes(packet) == (
        "238220",
        "541380",
        "541990",
        "811210",
        "811310",
    )
    assert {topic.value for topic in available_market_topics(packet)} == {
        "biotech_labs",
        "equipment_service",
        "fire_life_safety",
        "senior_living_facilities",
    }


def test_exact_provider_and_customer_activity_boundaries_are_visible() -> None:
    by_id = _by_id()

    assert "testing_certification_only" in by_id["naics22-541380-scope"].quality_flags
    assert "customer_class_not_demand" in by_id["naics22-541714-scope"].quality_flags
    assert "customer_eligible" in by_id["naics22-541714-scope"].quality_flags
    assert "fire_testing_inspection_only" in by_id["naics22-541990-scope"].quality_flags
    assert "equipment_repair_maintenance_boundary" in by_id["naics22-811210-scope"].quality_flags
    assert "fire_repair_without_installation" in by_id["naics22-811310-scope"].quality_flags
    assert "fire_installation_boundary" in by_id["naics22-238220-scope"].quality_flags
    assert "license_category_not_naics" in by_id["naics22-623312-scope"].quality_flags
    assert "customer_eligible" in by_id["naics22-623312-scope"].quality_flags

    for code in TARGETED_ALLOWED_NAICS_CODES:
        measurement = by_id[f"naics22-{code}-scope"]
        assert "classification_only" in measurement.quality_flags
        assert "not_economic_measurement" in measurement.quality_flags


def test_biotech_buyer_and_related_supply_proxies_are_exact_and_caveated() -> None:
    by_id = _by_id()

    assert by_id["cbp-sd-biotech-rd-establishments-2023"].value == 437
    assert by_id["cbp-sd-biotech-rd-establishments-20-249-2023"].value == 88
    assert by_id["cbp-sd-biotech-rd-employees-2023"].value == 20_270
    assert by_id["cbp-sd-biotech-rd-annual-payroll-2023"].value == 3_690_733_000
    assert by_id["cbp-sd-biological-product-establishments-2023"].value == 17
    assert by_id["cbp-sd-medical-lab-establishments-2023"].value == 130
    assert by_id["cbp-sd-testing-lab-establishments-2023"].value == 93

    buyer = by_id["cbp-sd-biotech-rd-establishments-20-249-2023"]
    assert "41 establishments with 20-49" in buyer.caveat
    assert "not wet labs" in buyer.caveat
    assert "not_demand_measurement" in buyer.quality_flags
    assert "not_device_count" in buyer.quality_flags
    assert (
        "not_capacity_measurement" in by_id["cbp-sd-testing-lab-establishments-2023"].quality_flags
    )


def test_bsc_recurrence_supply_price_and_unknown_boundaries_are_preserved() -> None:
    by_id = _by_id()

    bmbl = by_id["bmbl-bsc-certification-cadence"]
    assert bmbl.value == "before service; after repair or relocation; at least annually"
    assert "advisory_not_mandate" in bmbl.quality_flags
    assert "not a statute or universal regulation" in bmbl.caveat

    ucsd = by_id["ucsd-bsc-annual-certification-policy"]
    assert ucsd.value is True
    assert "public_institution_first_party" in ucsd.quality_flags
    assert "not a statewide rule" in ucsd.caveat

    assert by_id["nsf-sd-bsc-current-certifiers"].value == 22
    assert by_id["nsf-sd-bsc-employer-address-groups"].value == 4
    assert "counterevidence" in by_id["nsf-sd-bsc-current-certifiers"].quality_flags
    assert "not firms" in by_id["nsf-sd-bsc-current-certifiers"].caveat

    assert by_id["omnia-bsc-nsf49-certification-price-range"].value == "145-189"
    assert by_id["omnia-bsc-pm-price-range"].value == "550-715"
    assert by_id["derived-bsc-certification-pm-price-range"].value == "695-904"
    assert (
        "not an observed private transaction"
        in by_id["omnia-bsc-nsf49-certification-price-range"].caveat
    )

    for measurement_id in {
        "unknown-sd-bsc-installed-base",
        "unknown-sd-bsc-provider-backlog",
        "unknown-sd-bsc-private-wtp",
    }:
        measurement = by_id[measurement_id]
        assert measurement.value is None
        assert "unknown_not_zero" in measurement.quality_flags
        assert "missing_direct_measurement" in measurement.quality_flags


def test_senior_facility_fire_proxies_do_not_become_assets_or_demand() -> None:
    by_id = _by_id()

    facilities = by_id["cdss-sd-licensed-rcfe-ccrc-records"]
    capacity = by_id["cdss-sd-licensed-rcfe-ccrc-capacity"]
    assert facilities.value == 581
    assert capacity.value == 22_248
    assert "569 RESIDENTIAL CARE ELDERLY" in facilities.caveat
    assert "12 RCFE-CONTINUING CARE RETIREMENT COMMUNITY" in facilities.caveat
    assert "not_fire_system_count" in facilities.quality_flags
    assert "licensed_capacity_not_occupancy" in capacity.quality_flags
    assert "not_demand_measurement" in capacity.quality_flags

    clearance = by_id["cdss-rcfe-fire-clearance-rule"]
    assert clearance.value is True
    assert "mandate_not_contestable_spend" in clearance.quality_flags
    assert "not a count of systems" in clearance.caveat


def test_fire_cadence_licensing_insurance_reporting_supply_and_prices_are_scoped() -> None:
    by_id = _by_id()

    cadence = by_id["ca-osfm-fixed-system-semiannual-itm"]
    assert cadence.value == "at least semi-annually and immediately after activation"
    assert "system_type_specific" in cadence.quality_flags
    assert "portable extinguisher" in cadence.caveat

    assert by_id["ca-osfm-aes-service-license-paths"].value == (
        "OSFM A Type 1 or Type 2, or CSLB C-16 as applicable"
    )
    assert by_id["ca-osfm-aes-liability-insurance"].value == 1_000_000
    assert "reporting_workflow" in by_id["ca-osfm-aes-records-invoice-deficiencies"].quality_flags
    reporting_value = by_id["sdfd-compliance-engine-reporting"].value
    assert isinstance(reporting_value, str)
    assert "within 14 days" in reporting_value

    c16 = by_id["cslb-sd-clear-c16-addresses"]
    assert c16.value == 93
    assert "address_not_service_territory" in c16.quality_flags
    assert "not_capacity_measurement" in c16.quality_flags

    assert by_id["socal-portable-extinguisher-service-call-price"].value == 39.0
    assert by_id["socal-portable-extinguisher-per-unit-price"].value == 12.5
    assert by_id["sd-city-kitchen-system-itm-unit-price-2017"].value == 119.0
    assert "vendor_first_party" in by_id["socal-portable-extinguisher-per-unit-price"].quality_flags
    assert (
        "historical_public_procurement_price"
        in by_id["sd-city-kitchen-system-itm-unit-price-2017"].quality_flags
    )


def test_fire_backlog_outsourcing_wtp_and_asset_unknowns_are_explicit() -> None:
    by_id = _by_id()

    for measurement_id in {
        "unknown-sd-senior-fire-installed-systems",
        "unknown-sd-fire-provider-backlog",
        "unknown-sd-fire-outsourced-share",
        "unknown-sd-fire-private-wtp",
    }:
        measurement = by_id[measurement_id]
        assert measurement.value is None
        assert "unknown_not_zero" in measurement.quality_flags
        assert "missing_direct_measurement" in measurement.quality_flags

    assert (
        "must not be converted into asset or service-visit counts"
        in by_id["unknown-sd-senior-fire-installed-systems"].caveat
    )
    assert (
        "mandate_not_contestable_spend" in by_id["unknown-sd-fire-outsourced-share"].quality_flags
    )


def test_non_government_first_party_sources_are_not_upgraded_to_official() -> None:
    by_id = _by_id()
    non_government = {
        "nsf-sd-bsc-current-certifiers",
        "nsf-sd-bsc-employer-address-groups",
        "socal-portable-extinguisher-service-call-price",
        "socal-portable-extinguisher-per-unit-price",
    }

    for measurement_id in non_government:
        flags = by_id[measurement_id].quality_flags
        assert "official_primary" not in flags

    assert "non-government credential registry" in (build_targeted_evidence_packet().source_policy)
    assert "never called demand" in build_targeted_evidence_packet().source_policy
    assert "No elder-care operating thesis" in (build_targeted_evidence_packet().source_policy)


def test_every_targeted_measurement_has_source_scope_caveat_and_unique_flags() -> None:
    packet = build_targeted_evidence_packet()

    for measurement in packet.measurements:
        assert measurement.source_url.startswith("https://")
        assert measurement.observed_period
        assert measurement.geography
        assert measurement.unit
        assert measurement.caveat
        assert measurement.quality_flags
        assert len(measurement.quality_flags) == len(set(measurement.quality_flags))


def test_every_targeted_measurement_has_an_explicit_market_or_sector_topic() -> None:
    packet = build_targeted_evidence_packet()

    unclassified = [
        measurement.measurement_id
        for measurement in packet.measurements
        if not evidence_topics(measurement.measurement_id)
    ]

    assert unclassified == []
