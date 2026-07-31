"""Deterministic official-source evidence packet for the first venture pilot.

This catalog is deliberately conservative.  It records favorable observations,
counterevidence, and material unknowns in the same packet.  Ratios remain
explicitly labeled as proxies; establishment statistics never become firm
statistics; and a legal mandate never becomes assumed contestable spend.

The values below are a frozen, reviewed seed catalog.  Live collectors can
replace it in later runs, but model analysts must only cite measurement IDs
present in the packet they receive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from app.venture.core import Scenario
from app.venture.discovery import EvidencePacket, PacketMeasurement

_AS_OF: Final = datetime(2026, 7, 31, 8, 15, tzinfo=UTC)

_CMS_DATA: Final = "https://data.cms.gov/provider-data/api/1/datastore/query/4pq5-n9py/0"
_CENSUS_CBP_2022_US: Final = (
    "https://www2.census.gov/programs-surveys/cbp/datasets/2022/cbp22us.zip"
)
_CENSUS_CBP_2023_US: Final = (
    "https://www2.census.gov/programs-surveys/cbp/datasets/2023/cbp23us.zip"
)
_CENSUS_CBP_2023_COUNTY: Final = (
    "https://www2.census.gov/programs-surveys/cbp/datasets/2023/cbp23co.zip"
)
_CENSUS_CA_AGE_2024: Final = (
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/"
    "counties/asrh/cc-est2024-agesex-06.csv"
)
_USASPENDING_NAICS: Final = "https://api.usaspending.gov/api/v2/search/spending_by_category/naics/"
_IRS_CORPORATION_2022: Final = "https://www.irs.gov/pub/irs-soi/22co01ccr.xlsx"
_IRS_SCHEDULE_C_STATE_2023: Final = "https://www.irs.gov/pub/irs-soi/23sp01st.xlsx"
_CENSUS_NAICS_2022: Final = (
    "https://www.census.gov/naics/reference_files_tools/2022_NAICS_Manual.pdf"
)

_CA_TITLE_19_904: Final = (
    "https://govt.westlaw.com/calregs/Document/"
    "I15F9BED45BE511EC98C8000D3A7C4BC3?needToInjectTerms=False&viewType=FullText"
)
_CA_TITLE_19_904_1: Final = (
    "https://govt.westlaw.com/calregs/Document/"
    "I15FE79CE5BE511EC98C8000D3A7C4BC3?needToInjectTerms=False&viewType=FullText"
)
_CA_TITLE_19_904_2: Final = (
    "https://govt.westlaw.com/calregs/Document/I160901195BE511EC98C8000D3A7C4BC3?viewType=FullText"
)
_CA_C16: Final = (
    "https://www.cslb.ca.gov/about_us/library/licensing_classifications/"
    "Licensing_Classifications_Detail.aspx?Class=C16"
)
_CA_C16_DATA_PORTAL: Final = (
    "https://www.cslb.ca.gov/Onlineservices/DataPortal/ListByClassification"
)
_CA_FIRE_LIFE_SAFETY_EXAM: Final = "https://www.dir.ca.gov/dlse/ecu/1c.html"

_OFFICIAL: Final = "official_primary"

# This is a hypothesis policy allowlist, not a claim that each code is an
# attractive market.  It deliberately includes the mutually exclusive Census
# classifications needed to state a fire-system offer precisely while omitting
# broad adjacent categories that do not describe that work.
PILOT_ALLOWED_NAICS_CODES: Final = (
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
)

PILOT_EXCLUDED_NAICS_CODES: Final = ("561790",)


@dataclass(frozen=True, slots=True)
class _CbpFact:
    slug: str
    label: str
    naics: str
    establishments_2022: int
    establishments_2023: int
    growth_percent: float
    employees_per_establishment: float
    payroll_per_employee_usd: int
    share_under_20_employees_percent: float
    counterevidence: bool = False
    extra_caveat: str = ""
    extra_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _FederalDemandFact:
    slug: str
    label: str
    naics: str
    current_obligations_usd: int
    growth_percent: float | None
    extra_caveat: str = ""
    extra_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _EconomicCensusFact:
    slug: str
    label: str
    naics: str
    sector: str
    firms: int
    establishments: int
    receipts_usd: int
    payroll_receipts_percent: float
    receipts_per_establishment_usd: int
    receipts_per_employee_usd: int
    extra_caveat: str = ""
    extra_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ScheduleCFact:
    slug: str
    label: str
    geography: str
    gross_receipts_usd: int
    net_profit_usd: int
    net_profit_receipts_percent: float


@dataclass(frozen=True, slots=True)
class _C16CountyFact:
    slug: str
    geography: str
    clear_license_addresses: int
    cbp_employer_establishments_ge20: int
    addresses_per_thousand_establishments: float


@dataclass(frozen=True, slots=True)
class _NaicsClassificationFact:
    code: str
    title: str
    scope: str
    caveat: str
    extra_flags: tuple[str, ...] = ()


def _measurement(
    measurement_id: str,
    metric: str,
    value: bool | int | float | str | None,
    unit: str,
    geography: str,
    observed_period: str,
    source_family: str,
    source_url: str,
    caveat: str,
    *quality_flags: str,
) -> PacketMeasurement:
    return PacketMeasurement(
        measurement_id=measurement_id,
        metric=metric,
        value=value,
        unit=unit,
        geography=geography,
        observed_period=observed_period,
        source_family=source_family,
        source_url=source_url,
        caveat=caveat,
        quality_flags=(_OFFICIAL, *quality_flags),
    )


_NAICS_CLASSIFICATION_FACTS: Final = (
    _NaicsClassificationFact(
        "236220",
        "Commercial and Institutional Building Construction",
        "Construction, additions, alterations, maintenance, and repairs of commercial "
        "and institutional buildings and related structures.",
        "This is a general building-construction category, not a fire-system specialty "
        "trade or a measure of fire-compliance subcontract spend.",
        ("broad_industry_proxy",),
    ),
    _NaicsClassificationFact(
        "238210",
        "Electrical Contractors and Other Wiring Installation Contractors",
        "Installing and servicing electrical wiring and equipment; the official examples "
        "include electric fire- or burglar-alarm system installation only.",
        "Fire-alarm installation combined with sales, maintenance, or monitoring belongs "
        "in 561621. This code does not cover suppression-system work.",
        ("fire_alarm_installation_only", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "238220",
        "Plumbing, Heating, and Air-Conditioning Contractors",
        "The official examples and index include fire-sprinkler installation, "
        "fire-extinguisher installation and repair, and waterless fire-suppression-system "
        "installation and repair.",
        "Repair and maintenance of commercial fire extinguishers or waterless suppression "
        "systems without installation belongs in 811310. This broad trade also contains "
        "substantial non-fire plumbing and HVAC work.",
        ("fire_suppression_installation", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "423450",
        "Medical, Dental, and Hospital Equipment and Supplies Merchant Wholesalers",
        "Merchant wholesale distribution of professional medical equipment, instruments, "
        "and supplies, subject to the Census exclusions for ophthalmic goods.",
        "Distribution is not equipment repair, calibration, device manufacturing, or "
        "evidence that a buyer will purchase a bundled uptime service.",
        ("distribution_not_service", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "513210",
        "Software Publishers",
        "Producing and distributing published software, including design, documentation, "
        "installation assistance, and purchaser support; distribution may use "
        "subscriptions or downloads.",
        "Custom software development belongs in 541511 and hosting infrastructure belongs "
        "in 518210. A publishing classification does not establish product demand.",
        ("publisher_not_custom_software", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "541380",
        "Testing Laboratories and Services",
        "Physical, chemical, and other analytical testing, including calibration, "
        "electrical and electronic, mechanical, and nondestructive testing in a lab or "
        "on-site.",
        "Calibration testing is not the same activity as equipment repair under 811210 "
        "or medical diagnostic laboratory testing under 62151.",
        ("calibration_testing", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "541614",
        "Process, Physical Distribution, and Logistics Consulting Services",
        "Operating advice on manufacturing, productivity, production, quality, inventory, "
        "distribution networks, warehouses, transportation, and materials handling.",
        "Freight arrangement belongs in 488510 and warehouse operation in Industry Group "
        "4931; this is consulting, not either operating activity.",
        ("consulting_not_operations", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "541620",
        "Environmental Consulting Services",
        "Advice and assistance on environmental issues, contamination, hazards, risk "
        "evaluation, remediation, ecological restoration, and environmental law.",
        "Environmental testing belongs in 541380, engineering in 541330, and remediation "
        "operations in 562910. This code does not describe those activities.",
        ("consulting_not_testing_or_remediation", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "541990",
        "All Other Professional, Scientific, and Technical Services",
        "The official 2022 alphabetic index places fire-extinguisher and waterless "
        "fire-suppression-system testing or inspection here only when there is no sales, "
        "service, or installation.",
        "This is a broad residual category. Any fire candidate using it must be limited to "
        "inspection/testing-only work and cannot import the economics of installation, "
        "repair, sales, or monitoring.",
        ("fire_testing_inspection_only", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "561611",
        "Investigation and Personal Background Check Services",
        "Investigation, detective, and personal background-check services.",
        "Credit checks belong in 561450. This classification does not measure screening "
        "platform concentration, data access, compliance cost, or customer demand.",
        ("credit_check_excluded", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "561621",
        "Security Systems Services (except Locksmiths)",
        "Selling security alarm systems, including burglar and fire alarms, together with "
        "installation, repair, or monitoring, or remotely monitoring electronic security "
        "alarm systems.",
        "Alarm installation without selling or monitoring belongs in 238210. This mixed "
        "security category is not a pure fire-compliance or suppression-system market.",
        ("alarm_bundle_or_monitoring", "mixed_service_category"),
    ),
    _NaicsClassificationFact(
        "562119",
        "Other Waste Collection",
        "Local collection or hauling of waste other than nonhazardous solid waste and "
        "hazardous waste; brush or rubble removal is included.",
        "Solid-waste collection belongs in 562111, hazardous-waste collection in 562112, "
        "and collection combined with disposal in 56221.",
        ("route_type_boundary", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "621111",
        "Offices of Physicians (except Mental Health Specialists)",
        "Independent general or specialized medical practice or surgery by M.D. or D.O. "
        "physicians, except psychiatry or psychoanalysis.",
        "This identifies a potential customer establishment class, not its demand for "
        "equipment, repair, calibration, software, or any proposed service.",
        ("customer_class_not_demand",),
    ),
    _NaicsClassificationFact(
        "621610",
        "Home Health Care Services",
        "Skilled nursing in the home together with a range of personal care, therapy, "
        "medical equipment and supplies, counseling, or other listed health services.",
        "Non-medical home care or homemaker services for elderly or disabled people "
        "belong in 624120. Home health is not a generic aging-in-place category.",
        ("skilled_home_health", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "623110",
        "Nursing Care Facilities (Skilled Nursing Facilities)",
        "Inpatient nursing and rehabilitative services, generally for extended periods, "
        "with a permanent core staff of registered or licensed practical nurses.",
        "This is not all senior housing or assisted living. Census separately classifies "
        "continuing-care retirement communities and nonresidential services.",
        ("skilled_nursing_facility", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "624120",
        "Services for the Elderly and Persons with Disabilities",
        "Nonresidential social assistance, including day care, non-medical home care or "
        "homemaker services, social activities, group support, and companionship.",
        "Skilled in-home health care belongs in Subsector 621 and residential care in "
        "Subsector 623. This category is not a license class or capacity measure.",
        ("nonmedical_nonresidential_services", "adjacent_market_boundary"),
    ),
    _NaicsClassificationFact(
        "811210",
        "Electronic and Precision Equipment Repair and Maintenance",
        "Repair and maintenance of electronic and precision equipment without retailing it "
        "as new; official examples include scientific instruments and medical equipment.",
        "The code also contains consumer electronics, computers, office machines, and "
        "communications equipment. It is not a pure medical-equipment service market and "
        "does not include calibration testing solely because a device is precise.",
        ("scientific_medical_equipment_included", "broad_industry_proxy"),
    ),
    _NaicsClassificationFact(
        "811310",
        "Commercial and Industrial Machinery and Equipment Repair and Maintenance",
        "Repair and maintenance of commercial and industrial machinery. The official "
        "238220 cross-reference and index place commercial fire-extinguisher and waterless "
        "fire-suppression repair or maintenance without installation here.",
        "This broad category also contains unrelated commercial and industrial machinery. "
        "Installation-linked fire work belongs in 238220.",
        ("fire_suppression_repair_without_installation", "broad_industry_proxy"),
    ),
)


def _naics_classification_measurements() -> list[PacketMeasurement]:
    """Return classification boundaries, never market-economic observations."""
    common = (
        "NAICS classifies establishments by primary activity for statistical purposes. "
        "A classification does not measure demand, supply, competition, revenue, margin, "
        "failure risk, capacity, licensing, or whether a proposed mixed activity belongs "
        "in that code."
    )
    measurements = [
        _measurement(
            f"naics22-{fact.code}-scope",
            f"2022 NAICS scope: {fact.code} {fact.title}",
            fact.scope,
            "official classification scope",
            "United States",
            "2022 NAICS; reviewed 2026-07-31",
            "Census 2022 NAICS Classification",
            _CENSUS_NAICS_2022,
            f"{common} {fact.caveat}",
            "classification_only",
            "establishment_primary_activity",
            "not_economic_measurement",
            *fact.extra_flags,
        )
        for fact in _NAICS_CLASSIFICATION_FACTS
    ]
    measurements.append(
        _measurement(
            "naics22-561790-fire-excluded",
            "561790 excluded from the pilot fire-compliance hypothesis allowlist",
            False,
            "allowed classification for fire-compliance hypothesis",
            "United States",
            "2022 NAICS; reviewed 2026-07-31",
            "Census 2022 NAICS Classification",
            _CENSUS_NAICS_2022,
            f"{common} 561790 covers other services to buildings and dwellings, with "
            "examples such as exterior, chimney, ventilation-duct, drain or gutter, and "
            "pool cleaning. The official fire entries instead distinguish 238210, "
            "238220, 541990, 561621, and 811310 by activity. This is a conservative pilot "
            "policy exclusion, not a claim that a mixed-activity firm can never report "
            "561790.",
            "classification_only",
            "classification_exclusion",
            "adjacent_market_boundary",
            "not_economic_measurement",
            "fire_compliance_not_supported",
        )
    )
    return measurements


def _regulatory_measurements() -> list[PacketMeasurement]:
    mandate_caveat = (
        "The mandate establishes required work, not outsourced or otherwise contestable "
        "spend. System type, authority having jurisdiction, license scope, and customer "
        "insourcing must be resolved before treating it as a market."
    )
    return [
        _measurement(
            "ca-fire-904-recurring-itm",
            "California recurring fire-extinguishing-system inspection, testing, and "
            "maintenance requirement",
            True,
            "regulatory requirement",
            "California",
            "regulation text current at 2026-07-31",
            "California Code of Regulations",
            _CA_TITLE_19_904,
            mandate_caveat,
            "regulatory_mandate",
            "not_demand_measurement",
            "contestable_spend_unknown",
        ),
        _measurement(
            "ca-fire-9041-owner-inspection",
            "Owner-designated trained employee may perform qualifying inspection while a "
            "paid outside provider needs the specified credential",
            True,
            "regulatory scope rule",
            "California",
            "regulation text current at 2026-07-31",
            "California Code of Regulations",
            _CA_TITLE_19_904_1,
            "This is direct substitution counterevidence: some inspection can be insourced. "
            "The exact system and credential path still control.",
            "insourcing_allowed",
            "counterevidence",
            "credential_scope_required",
            "contestable_spend_unknown",
        ),
        _measurement(
            "ca-fire-9042-testing-license",
            "Fire-extinguishing-system testing and maintenance generally require a license",
            True,
            "regulatory scope rule",
            "California",
            "regulation text current at 2026-07-31",
            "California Code of Regulations",
            _CA_TITLE_19_904_2,
            mandate_caveat,
            "licensed_work",
            "technician_constraint_proxy",
            "contestable_spend_unknown",
        ),
        _measurement(
            "ca-c16-fire-scope-excludes-alarms",
            "C-16 classification covers fire-protection systems but excludes electrical "
            "alarm systems",
            True,
            "license scope distinction",
            "California",
            "classification text current at 2026-07-31",
            "California Contractors State License Board",
            _CA_C16,
            "Fire suppression and electrical fire-alarm work cannot be treated as one "
            "undifferentiated service or license market.",
            "scope_boundary",
            "fire_alarm_excluded",
            "business_definition_required",
        ),
        _measurement(
            "ca-fire-tech-qualifying-hours",
            "Qualifying experience required for the fire/life-safety technician exam route",
            4000,
            "hours across at least two qualifying work areas",
            "California",
            "program requirements current at 2026-07-31",
            "California Department of Industrial Relations",
            _CA_FIRE_LIFE_SAFETY_EXAM,
            "A credential path can constrain staffing, but required hours do not measure "
            "current technician shortage, wages, pass rates, or service demand.",
            "technician_pipeline_constraint",
            "not_supply_count",
            "labor_bottleneck_unquantified",
        ),
        _measurement(
            "ca-fire-outsourced-spend-unknown",
            "Directly observed outsourced fire/life-safety compliance spend",
            None,
            "USD",
            "California",
            "unknown as of 2026-07-31",
            "California Code of Regulations",
            _CA_TITLE_19_904,
            "The reviewed official rules do not report what owners outsource, the "
            "addressable spend, prices, or willingness to pay.",
            "missing_direct_measurement",
            "unknown_not_zero",
            "contestable_spend_unknown",
        ),
    ]


_C16_COUNTY_FACTS: Final = (
    _C16CountyFact("kings", "Kings County, California", 0, 275, 0.0),
    _C16CountyFact("yolo", "Yolo County, California", 5, 756, 6.61),
    _C16CountyFact("san-diego", "San Diego County, California", 93, 12_610, 7.38),
    _C16CountyFact("merced", "Merced County, California", 4, 512, 7.81),
    _C16CountyFact("tulare", "Tulare County, California", 10, 1_081, 9.25),
    _C16CountyFact("santa-clara", "Santa Clara County, California", 76, 7_603, 10.00),
    _C16CountyFact("fresno", "Fresno County, California", 31, 2_998, 10.34),
    _C16CountyFact("santa-barbara", "Santa Barbara County, California", 18, 1_699, 10.59),
    _C16CountyFact("stanislaus", "Stanislaus County, California", 17, 1_551, 10.96),
    _C16CountyFact("imperial", "Imperial County, California", 5, 387, 12.92),
)


def _c16_supply_measurements() -> list[PacketMeasurement]:
    source_hash = "8b5dec0ab82188be688da355812decb30040c2437871ab3fc7e4f5862cdd6e0c"
    retrieval = "2026-07-31T08:10:06Z"
    license_caveat = (
        f"CSLB classification export retrieved {retrieval}, XLSX SHA-256 {source_hash}. "
        "CLEAR license status and address of record do not establish active availability, "
        "capacity, service territory, price, quality, or willingness to serve. A record "
        "can be a multi-class license, so this is not a count of pure-play C-16 firms."
    )
    denominator_caveat = (
        "The denominator is 2023 County Business Patterns employer establishments with "
        "at least 20 employees. It is not a count of buildings, installed systems, "
        "mandated jobs, or contestable spend, and it excludes nonemployers and public "
        "facilities."
    )
    measurements = [
        _measurement(
            "cslb-c16-clear-total",
            "CLEAR C-16 fire-protection contractor license records",
            2218,
            "CLEAR license records",
            "United States",
            retrieval,
            "California Contractors State License Board",
            _CA_C16_DATA_PORTAL,
            license_caveat,
            "license_address_snapshot",
            "screening_proxy",
            "counterevidence",
            "multi_class_licenses_possible",
            "not_active_capacity",
        ),
        _measurement(
            "cslb-c16-clear-ca-address",
            "CLEAR C-16 license records with a California address of record",
            2115,
            "CLEAR license records",
            "California",
            retrieval,
            "California Contractors State License Board",
            _CA_C16_DATA_PORTAL,
            license_caveat,
            "license_address_snapshot",
            "screening_proxy",
            "counterevidence",
            "address_not_service_territory",
            "not_active_capacity",
        ),
        _measurement(
            "cslb-c16-clear-out-state",
            "CLEAR C-16 license records with an out-of-state address of record",
            103,
            "CLEAR license records",
            "United States",
            retrieval,
            "California Contractors State License Board",
            _CA_C16_DATA_PORTAL,
            license_caveat,
            "license_address_snapshot",
            "screening_proxy",
            "counterevidence",
            "address_not_service_territory",
            "not_active_capacity",
        ),
        _measurement(
            "cslb-c16-ca-counties",
            "California counties represented by a CLEAR C-16 license address of record",
            49,
            "counties",
            "California",
            retrieval,
            "California Contractors State License Board",
            _CA_C16_DATA_PORTAL,
            license_caveat,
            "license_address_snapshot",
            "screening_proxy",
            "counterevidence",
            "geographic_coverage_not_capacity",
        ),
    ]
    combined_url = f"{_CA_C16_DATA_PORTAL} | {_CENSUS_CBP_2023_COUNTY}"
    for fact in _C16_COUNTY_FACTS:
        zero_caveat = (
            " A zero is a zero address count in this export, not proof that supply, "
            "service coverage, or qualified labor is absent."
            if fact.clear_license_addresses == 0
            else ""
        )
        measurements.extend(
            (
                _measurement(
                    f"cslb-c16-{fact.slug}-addresses",
                    "CLEAR C-16 license addresses of record",
                    fact.clear_license_addresses,
                    "CLEAR license addresses",
                    fact.geography,
                    retrieval,
                    "California Contractors State License Board",
                    _CA_C16_DATA_PORTAL,
                    f"{license_caveat}{zero_caveat}",
                    "license_address_snapshot",
                    "screening_proxy",
                    "counterevidence",
                    "address_not_service_territory",
                    "not_active_capacity",
                    *(("zero_address_count_not_supply_absence",) if zero_caveat else ()),
                ),
                _measurement(
                    f"cbp-{fact.slug}-establishments-ge20",
                    "Employer establishments with at least 20 employees",
                    fact.cbp_employer_establishments_ge20,
                    "employer establishments",
                    fact.geography,
                    "2023",
                    "Census County Business Patterns",
                    _CENSUS_CBP_2023_COUNTY,
                    denominator_caveat,
                    "employer_establishments",
                    "establishment_not_building",
                    "excludes_nonemployers",
                    "excludes_public_facilities",
                    "not_mandated_jobs_or_spend",
                ),
                _measurement(
                    f"derived-c16-{fact.slug}-per1000-ge20",
                    "CLEAR C-16 license addresses per 1,000 employer establishments with "
                    "at least 20 employees",
                    fact.addresses_per_thousand_establishments,
                    "CLEAR license addresses per 1,000 employer establishments",
                    fact.geography,
                    f"{retrieval} CSLB / 2023 CBP",
                    "California CSLB + Census County Business Patterns",
                    combined_url,
                    f"Derived from {fact.clear_license_addresses} CLEAR C-16 license "
                    f"addresses and {fact.cbp_employer_establishments_ge20} CBP employer "
                    f"establishments with at least 20 employees, rounded to two decimals. "
                    f"{license_caveat} {denominator_caveat}{zero_caveat}",
                    "derived_ratio",
                    "screening_proxy",
                    "counterevidence",
                    "address_not_service_territory",
                    "not_active_capacity",
                    "denominator_not_installed_base",
                    "contestable_spend_unknown",
                    *(("zero_address_count_not_supply_absence",) if zero_caveat else ()),
                ),
            )
        )
    return measurements


_CBP_FACTS: Final = (
    _CbpFact(
        "precision-repair",
        "Electronic and precision equipment repair",
        "811211+811212+811213+811219",
        11_421,
        10_917,
        -4.4,
        8.2,
        68_726,
        93.2,
        extra_caveat=(
            "The four 2017-NAICS component rows are summed; this is not directly identical "
            "to the 811210 label used in the USAspending and Economic Census observations."
        ),
        extra_flags=("naics_crosswalk_required", "component_sum"),
    ),
    _CbpFact(
        "testing-calibration",
        "Testing laboratories and services",
        "541380",
        7_488,
        7_463,
        -0.3,
        21.2,
        75_284,
        78.6,
    ),
    _CbpFact(
        "background",
        "Investigation and personal background check services",
        "561611",
        4_085,
        4_013,
        -1.8,
        10.4,
        58_545,
        91.7,
    ),
    _CbpFact(
        "other-waste",
        "Other waste collection",
        "562119",
        1_458,
        1_571,
        7.8,
        9.0,
        63_112,
        92.7,
        counterevidence=True,
    ),
    _CbpFact(
        "logistics-consulting",
        "Process, physical distribution, and logistics consulting",
        "541614",
        8_676,
        8_360,
        -3.6,
        11.6,
        72_883,
        89.7,
    ),
    _CbpFact(
        "home-health",
        "Home health care services",
        "621610",
        39_117,
        40_762,
        4.2,
        39.7,
        38_091,
        62.3,
        counterevidence=True,
    ),
    _CbpFact(
        "elderly-disabled",
        "Services for the elderly and persons with disabilities",
        "624120",
        41_626,
        40_265,
        -3.3,
        37.9,
        29_098,
        65.6,
        extra_caveat="NAICS 624120 is a service-category proxy, not a home-care license class.",
        extra_flags=("service_definition_proxy",),
    ),
    _CbpFact(
        "security-systems",
        "Security systems services",
        "561621",
        7_042,
        7_151,
        1.5,
        18.0,
        70_074,
        81.6,
        counterevidence=True,
        extra_caveat=(
            "NAICS 561621 mixes monitoring, installation, and security systems and is not "
            "a pure fire/life-safety compliance category."
        ),
        extra_flags=("mixed_service_category",),
    ),
    _CbpFact(
        "industrial-repair",
        "Commercial and industrial machinery and equipment repair",
        "811310",
        22_433,
        22_907,
        2.1,
        9.6,
        73_266,
        90.4,
        counterevidence=True,
    ),
    _CbpFact(
        "electrical",
        "Electrical contractors and other wiring installation contractors",
        "238210",
        81_842,
        83_342,
        1.8,
        12.2,
        76_545,
        88.1,
        counterevidence=True,
        extra_caveat="This broad category is not a count of fire-alarm specialists.",
        extra_flags=("broad_industry_proxy",),
    ),
    _CbpFact(
        "medical-wholesale",
        "Medical, dental, and hospital equipment merchant wholesalers",
        "423450",
        10_599,
        10_296,
        -2.9,
        26.4,
        125_412,
        83.5,
    ),
    _CbpFact(
        "security-guards",
        "Security guards and patrol services",
        "561612",
        10_659,
        11_104,
        4.2,
        72.0,
        31_538,
        56.7,
        counterevidence=True,
    ),
    _CbpFact(
        "commercial-construction",
        "Commercial and institutional building construction",
        "236220",
        39_005,
        39_072,
        0.2,
        16.4,
        97_254,
        81.6,
    ),
)


def _cbp_measurements() -> list[PacketMeasurement]:
    source_url = f"{_CENSUS_CBP_2022_US} | {_CENSUS_CBP_2023_US}"
    common = (
        "County Business Patterns covers employer establishments, not firms or all "
        "businesses. All-legal-forms rows (LFO '-') are used rather than summing the "
        "legal-form total and its components. Employment and payroll are noise-infused; "
        "the industry vintage is NAICS 2017."
    )
    measurements: list[PacketMeasurement] = []
    for fact in _CBP_FACTS:
        caveat = f"{common} {fact.extra_caveat}".strip()
        flags = (
            "employer_establishments",
            "establishment_not_firm",
            "naics_2017",
            "noise_infused_employment_payroll",
            "lfo_total_only",
            *fact.extra_flags,
        )
        if fact.counterevidence:
            flags = (*flags, "counterevidence")
        measurements.extend(
            (
                _measurement(
                    f"cbp-{fact.slug}-est-2023",
                    f"{fact.label} employer establishments (NAICS {fact.naics})",
                    fact.establishments_2023,
                    "employer establishments",
                    "United States",
                    "2023",
                    "Census County Business Patterns",
                    _CENSUS_CBP_2023_US,
                    caveat,
                    *flags,
                ),
                _measurement(
                    f"cbp-{fact.slug}-est-growth",
                    f"{fact.label} employer-establishment growth (NAICS {fact.naics})",
                    fact.growth_percent,
                    "percent",
                    "United States",
                    "2022/2023",
                    "Census County Business Patterns",
                    source_url,
                    f"{caveat} Derived from {fact.establishments_2022:,} establishments "
                    f"in 2022 and {fact.establishments_2023:,} in 2023; rounded to one "
                    "decimal point.",
                    "derived_ratio",
                    *flags,
                ),
                _measurement(
                    f"cbp-{fact.slug}-employees-est",
                    f"{fact.label} employees per employer establishment (NAICS {fact.naics})",
                    fact.employees_per_establishment,
                    "employees per employer establishment",
                    "United States",
                    "2023",
                    "Census County Business Patterns",
                    _CENSUS_CBP_2023_US,
                    f"{caveat} This is an aggregate ratio, not a typical-establishment "
                    "observation.",
                    "derived_ratio",
                    "aggregate_not_typical",
                    *flags,
                ),
                _measurement(
                    f"cbp-{fact.slug}-payroll-employee",
                    f"{fact.label} annual payroll per employee (NAICS {fact.naics})",
                    fact.payroll_per_employee_usd,
                    "USD per employee",
                    "United States",
                    "2023",
                    "Census County Business Patterns",
                    _CENSUS_CBP_2023_US,
                    f"{caveat} Payroll per employee is a labor-cost proxy, not total "
                    "compensation, wage for a role, profit, or margin.",
                    "derived_ratio",
                    "labor_cost_proxy",
                    "not_margin",
                    *flags,
                ),
                _measurement(
                    f"cbp-{fact.slug}-under20-share",
                    f"{fact.label} employer establishments with fewer than 20 employees "
                    f"(NAICS {fact.naics})",
                    fact.share_under_20_employees_percent,
                    "percent of employer establishments",
                    "United States",
                    "2023",
                    "Census County Business Patterns",
                    _CENSUS_CBP_2023_US,
                    f"{caveat} National size-bucket totals are used; this does not measure "
                    "ownership fragmentation, independent competitors, or acquisition "
                    "availability.",
                    "derived_ratio",
                    "small_establishment_share",
                    "not_ownership_fragmentation",
                    *flags,
                ),
            )
        )
    return measurements


_FEDERAL_DEMAND_FACTS: Final = (
    _FederalDemandFact(
        "precision-repair",
        "Electronic and precision equipment repair and maintenance",
        "811210",
        2_670_000_000,
        51.1,
        "USAspending's 811210 label requires a vintage crosswalk before comparison with "
        "the four component CBP rows.",
        ("naics_crosswalk_required",),
    ),
    _FederalDemandFact(
        "medical-wholesale",
        "Medical, dental, and hospital equipment merchant wholesalers",
        "423450",
        3_840_000_000,
        31.7,
    ),
    _FederalDemandFact(
        "logistics-consulting",
        "Process, physical distribution, and logistics consulting",
        "541614",
        4_710_000_000,
        21.7,
    ),
    _FederalDemandFact(
        "security-guards",
        "Security guards and patrol services",
        "561612",
        9_620_000_000,
        11.6,
        "Security-guard obligations do not measure demand for fire/life-safety systems.",
        ("adjacent_not_same_market",),
    ),
    _FederalDemandFact(
        "commercial-construction",
        "Commercial and institutional building construction",
        "236220",
        67_920_000_000,
        35.5,
        "Construction obligations are broad and do not identify contestable "
        "fire/life-safety subcontract spend.",
        ("broad_industry_proxy",),
    ),
    _FederalDemandFact(
        "background",
        "Investigation and personal background check services",
        "561611",
        1_270_000_000,
        None,
        "The category was present in the current top 100 and absent from the prior top "
        "100; absence is an unknown lower bound, not zero.",
        ("prior_top100_absence", "unknown_not_zero"),
    ),
    _FederalDemandFact(
        "other-waste",
        "Other waste collection",
        "562119",
        3_310_000_000,
        None,
        "The category was present in the current top 100 and absent from the prior top "
        "100; absence is an unknown lower bound, not zero.",
        ("prior_top100_absence", "unknown_not_zero"),
    ),
    _FederalDemandFact(
        "individual-family",
        "Other individual and family services",
        "624190",
        1_190_000_000,
        None,
        "The category was present in the current top 100 and absent from the prior top "
        "100; absence is an unknown lower bound, not zero. This category is not licensed "
        "home health care.",
        ("prior_top100_absence", "unknown_not_zero", "adjacent_not_same_market"),
    ),
    _FederalDemandFact(
        "software",
        "Software publishers",
        "513210",
        2_680_000_000,
        84.9,
        "Obligations show federal purchasing, not low private-market competition.",
        ("competition_unmeasured",),
    ),
    _FederalDemandFact(
        "physician-offices",
        "Offices of physicians",
        "621111",
        13_950_000_000,
        68.9,
        "Purchasing by this NAICS does not specify equipment repair, calibration, or the "
        "share available to a new supplier.",
        ("payer_use_case_unresolved",),
    ),
)


def _federal_demand_measurements() -> list[PacketMeasurement]:
    common = (
        "USAspending transaction-level prime-contract obligations use award-type codes "
        "A/B/C/D and action dates. Federal obligations are not total or private demand, "
        "addressable revenue, willingness to pay for a proposed offer, or profit."
    )
    measurements: list[PacketMeasurement] = []
    for fact in _FEDERAL_DEMAND_FACTS:
        caveat = f"{common} {fact.extra_caveat}".strip()
        measurements.extend(
            (
                _measurement(
                    f"usa-{fact.slug}-obligations",
                    f"Federal prime-contract obligations for {fact.label} (NAICS {fact.naics})",
                    fact.current_obligations_usd,
                    "USD obligations",
                    "United States",
                    "2025-02-01/2026-07-31",
                    "USAspending",
                    _USASPENDING_NAICS,
                    caveat,
                    "federal_prime_contracts_only",
                    "demand_proxy",
                    "not_addressable_revenue",
                    *fact.extra_flags,
                ),
                _measurement(
                    f"usa-{fact.slug}-growth",
                    f"Change in federal prime-contract obligations for {fact.label} "
                    f"(NAICS {fact.naics})",
                    fact.growth_percent,
                    "percent",
                    "United States",
                    "2023-08-01/2025-01-31 versus 2025-02-01/2026-07-31",
                    "USAspending",
                    _USASPENDING_NAICS,
                    caveat,
                    "federal_prime_contracts_only",
                    "demand_proxy",
                    "not_addressable_revenue",
                    *fact.extra_flags,
                ),
            )
        )
    return measurements


_BLS_SURVIVAL: Final = {
    "62": ((2018, 85.3, 57.6), (2019, 87.7, 53.6), (2020, 84.0, 52.6)),
    "54": ((2018, 79.5, 50.2), (2019, 78.0, 50.7), (2020, 83.3, 50.8)),
    "56": ((2018, 78.1, 50.9), (2019, 75.8, 50.4), (2020, 79.4, 51.5)),
    "23": ((2018, 80.5, 56.2), (2019, 78.7, 55.8), (2020, 83.2, 56.5)),
    "51": ((2018, 77.2, 44.2), (2019, 75.6, 45.4), (2020, 79.6, 45.7)),
    "81": ((2018, 85.6, 50.2), (2019, 81.8, 53.8), (2020, 85.2, 60.0)),
}


def _bls_survival_measurements() -> list[PacketMeasurement]:
    measurements: list[PacketMeasurement] = []
    for naics, cohorts in _BLS_SURVIVAL.items():
        source_url = f"https://www.bls.gov/bdm/us_age_naics_{naics}_table7.txt"
        for cohort_year, one_year, five_year in cohorts:
            measurements.append(
                _measurement(
                    f"bls-bed-{naics}-{cohort_year}-survival",
                    f"NAICS {naics} same-cohort establishment survival after one and five years",
                    f"{one_year:.1f}/{five_year:.1f}",
                    "percent one-year/percent five-year",
                    "United States",
                    f"March {cohort_year} opening cohort through March {cohort_year + 5}",
                    "BLS Business Employment Dynamics",
                    source_url,
                    "Both percentages refer to the same private-sector establishment "
                    "opening cohort. This is establishment survival, not firm, owner, "
                    "investment, or startup survival; two-digit NAICS is broad.",
                    "same_cohort_pair",
                    "establishment_survival",
                    "establishment_not_firm",
                    "broad_two_digit_naics",
                    "not_failure_rate_for_business_model",
                )
            )
    return measurements


def _aging_measurements() -> list[PacketMeasurement]:
    combined_bed_url = f"{_CMS_DATA} | {_CENSUS_CA_AGE_2024}"
    combined_supply_url = f"{_CENSUS_CBP_2023_COUNTY} | {_CENSUS_CA_AGE_2024}"
    capacity_caveat = (
        "CMS-certified beds and average residents per day are not licensed-bed inventory, "
        "a vacancy count, a waitlist, or live availability. Ratios using Census age-65 "
        "population are screening proxies and do not adjust for acuity, payer, distance, "
        "facility type, or cross-county use."
    )
    service_supply_caveat = (
        "This divides County Business Patterns employer-establishment counts by Census "
        "age-65 population. It excludes nonemployers and does not measure workers, service "
        "slots, quality, licensed capacity, availability, or demand. NAICS is a service "
        "proxy and suppressed county observations must never be treated as zero."
    )
    measurements = [
        _measurement(
            "cms-ca-facilities",
            "Nursing-home provider records in the California CMS extract",
            1165,
            "facility records",
            "California",
            "July 2026 extract",
            "CMS Provider Data",
            _CMS_DATA,
            capacity_caveat,
            "facility_records",
            "certified_not_licensed",
            "not_live_availability",
        ),
        _measurement(
            "cms-ca-certified-beds",
            "CMS-certified nursing-home beds",
            114_990,
            "certified beds",
            "California",
            "July 2026 extract",
            "CMS Provider Data",
            _CMS_DATA,
            capacity_caveat,
            "certified_not_licensed",
            "not_live_availability",
        ),
        _measurement(
            "cms-ca-average-residents",
            "Summed average nursing-home residents per day",
            101_561.3,
            "average residents per day",
            "California",
            "July 2026 extract",
            "CMS Provider Data",
            _CMS_DATA,
            capacity_caveat,
            "average_not_current_census",
            "not_live_availability",
        ),
        _measurement(
            "cms-ca-load-proxy",
            "Average residents divided by CMS-certified nursing-home beds",
            88.3,
            "percent",
            "California",
            "July 2026 extract",
            "CMS Provider Data",
            _CMS_DATA,
            f"{capacity_caveat} Twenty-one facility records had average residents greater "
            "than certified beds, so this ratio must not be called occupancy.",
            "derived_ratio",
            "load_proxy_not_occupancy",
            "not_live_availability",
            "source_anomalies_present",
        ),
        _measurement(
            "cms-ca-residents-over-beds-anomalies",
            "Facility records with average residents greater than certified beds",
            21,
            "facility records",
            "California",
            "July 2026 extract",
            "CMS Provider Data",
            _CMS_DATA,
            "These records are a data-quality warning and prevent interpreting the "
            "statewide ratio as current occupancy or vacancy.",
            "data_quality_warning",
            "counterevidence",
            "not_live_availability",
        ),
        _measurement(
            "cms-ca-live-vacancy-unknown",
            "Live nursing-home vacancies or open admissions",
            None,
            "available beds",
            "California",
            "unknown as of 2026-07-31",
            "CMS Provider Data",
            _CMS_DATA,
            "The reviewed CMS extract does not report live vacancy, open admissions, or "
            "waitlist status. Average resident census cannot fill this unknown.",
            "missing_direct_measurement",
            "unknown_not_zero",
            "not_live_availability",
        ),
    ]

    county_capacity = (
        ("el-dorado", "El Dorado County, California", 6.5, 92.6),
        ("nevada", "Nevada County, California", 13.8, 94.1),
        ("santa-cruz", "Santa Cruz County, California", 13.6, 93.0),
        ("ventura", "Ventura County, California", 12.0, 90.8),
        ("riverside", "Riverside County, California", 12.6, 90.8),
    )
    for slug, geography, beds_per_thousand, load_proxy in county_capacity:
        measurements.extend(
            (
                _measurement(
                    f"cms-{slug}-beds-age65",
                    "CMS-certified nursing-home beds per 1,000 residents age 65 and over",
                    beds_per_thousand,
                    "certified beds per 1,000 age-65-plus residents",
                    geography,
                    "July 2026 CMS extract / July 1, 2024 population estimate",
                    "CMS Provider Data + Census Population Estimates",
                    combined_bed_url,
                    capacity_caveat,
                    "derived_ratio",
                    "certified_not_licensed",
                    "age65_denominator",
                    "not_live_availability",
                ),
                _measurement(
                    f"cms-{slug}-load-proxy",
                    "Average residents divided by CMS-certified nursing-home beds",
                    load_proxy,
                    "percent",
                    geography,
                    "July 2026 CMS extract",
                    "CMS Provider Data",
                    _CMS_DATA,
                    capacity_caveat,
                    "derived_ratio",
                    "load_proxy_not_occupancy",
                    "not_live_availability",
                ),
            )
        )

    county_service_supply = (
        ("el-dorado", "El Dorado County, California", 2.3, 3.9),
        ("nevada", "Nevada County, California", 1.6, 3.8),
        ("madera", "Madera County, California", 2.1, 4.1),
        ("santa-cruz", "Santa Cruz County, California", 2.8, None),
        ("riverside", "Riverside County, California", None, 4.7),
        ("humboldt", "Humboldt County, California", 1.4, None),
    )
    for slug, geography, home_health, elderly_disabled in county_service_supply:
        if home_health is not None:
            measurements.append(
                _measurement(
                    f"cbp-{slug}-home-health-age65",
                    "Home health care employer establishments (NAICS 621610) per 10,000 "
                    "residents age 65 and over",
                    home_health,
                    "employer establishments per 10,000 age-65-plus residents",
                    geography,
                    "2023 CBP / July 1, 2024 population estimate",
                    "Census County Business Patterns + Population Estimates",
                    combined_supply_url,
                    service_supply_caveat,
                    "derived_ratio",
                    "employer_only_supply",
                    "establishment_not_capacity",
                    "age65_denominator",
                    "service_definition_proxy",
                )
            )
        if elderly_disabled is not None:
            measurements.append(
                _measurement(
                    f"cbp-{slug}-elderly-disabled-age65",
                    "Elderly and disabled services employer establishments (NAICS 624120) "
                    "per 10,000 residents age 65 and over",
                    elderly_disabled,
                    "employer establishments per 10,000 age-65-plus residents",
                    geography,
                    "2023 CBP / July 1, 2024 population estimate",
                    "Census County Business Patterns + Population Estimates",
                    combined_supply_url,
                    service_supply_caveat,
                    "derived_ratio",
                    "employer_only_supply",
                    "establishment_not_capacity",
                    "age65_denominator",
                    "service_definition_proxy",
                )
            )
    return measurements


_ECONOMIC_CENSUS_FACTS: Final = (
    _EconomicCensusFact(
        "precision-repair",
        "Electronic and precision equipment repair and maintenance",
        "811210",
        "81",
        10_441,
        11_702,
        16_664_000_000,
        32.1,
        1_424_000,
        172_966,
        "A NAICS-vintage crosswalk is required before comparing this 811210 category with "
        "the CBP component sum.",
        ("naics_crosswalk_required",),
    ),
    _EconomicCensusFact(
        "testing-calibration",
        "Testing laboratories and services",
        "541380",
        "54",
        5_690,
        7_970,
        27_932_000_000,
        40.1,
        3_505_000,
        169_537,
    ),
    _EconomicCensusFact(
        "home-health",
        "Home health care services",
        "621610",
        "62",
        27_774,
        39_218,
        114_188_000_000,
        48.6,
        2_912_000,
        73_191,
    ),
    _EconomicCensusFact(
        "elderly-disabled",
        "Services for the elderly and persons with disabilities",
        "624120",
        "62",
        30_629,
        38_947,
        77_281_000_000,
        52.0,
        1_984_000,
        50_007,
        "This service category is not a license class or direct measure of service slots.",
        ("service_definition_proxy",),
    ),
    _EconomicCensusFact(
        "background",
        "Investigation and personal background check services",
        "561611",
        "56",
        3_799,
        4_060,
        6_155_000_000,
        33.3,
        1_516_000,
        134_843,
    ),
    _EconomicCensusFact(
        "security-systems",
        "Security systems services",
        "561621",
        "56",
        6_161,
        7_462,
        31_314_000_000,
        27.8,
        4_196_000,
        222_277,
        "The category mixes monitoring, installation, and security systems and is not a "
        "pure fire/life-safety compliance category.",
        ("mixed_service_category",),
    ),
    _EconomicCensusFact(
        "building-inspection",
        "Building inspection services",
        "541350",
        "54",
        7_459,
        7_624,
        4_464_000_000,
        37.9,
        585_547,
        141_775,
        "Building inspection is adjacent to, not equivalent to, fire-system testing and "
        "maintenance.",
        ("adjacent_not_same_market",),
    ),
    _EconomicCensusFact(
        "logistics-consulting",
        "Process, physical distribution, and logistics consulting",
        "541614",
        "54",
        8_668,
        9_637,
        26_038_000_000,
        26.1,
        2_702_000,
        252_326,
    ),
    _EconomicCensusFact(
        "environmental-consulting",
        "Environmental consulting services",
        "541620",
        "54",
        8_498,
        9_932,
        21_384_000_000,
        35.7,
        2_153_000,
        215_817,
    ),
)


def _economic_census_url(sector: str) -> str:
    return (
        "https://www2.census.gov/programs-surveys/economic-census/data/2022/"
        f"sector{sector}/EC22{sector}BASIC.zip"
    )


def _economic_census_measurements() -> list[PacketMeasurement]:
    common = (
        "The 2022 Economic Census covers employer businesses. Firm and establishment "
        "counts are different entity levels. Receipts per establishment reflects the "
        "existing establishment-size mix, not startup revenue; payroll/receipts is a "
        "labor-intensity proxy, not margin. These aggregates do not reveal distribution, "
        "owner economics, valuation, or entry cost."
    )
    measurements: list[PacketMeasurement] = []
    for fact in _ECONOMIC_CENSUS_FACTS:
        caveat = f"{common} {fact.extra_caveat}".strip()
        source_url = _economic_census_url(fact.sector)
        flags = (
            "employer_businesses_only",
            "firm_establishment_distinction",
            "existing_size_mix",
            *fact.extra_flags,
        )
        measurements.extend(
            (
                _measurement(
                    f"ec22-{fact.slug}-firms",
                    f"{fact.label} firms (NAICS {fact.naics})",
                    fact.firms,
                    "employer firms",
                    "United States",
                    "2022",
                    "Census Economic Census",
                    source_url,
                    caveat,
                    "firm_count",
                    *flags,
                ),
                _measurement(
                    f"ec22-{fact.slug}-establishments",
                    f"{fact.label} establishments (NAICS {fact.naics})",
                    fact.establishments,
                    "employer establishments",
                    "United States",
                    "2022",
                    "Census Economic Census",
                    source_url,
                    caveat,
                    "establishment_count",
                    "establishment_not_firm",
                    *flags,
                ),
                _measurement(
                    f"ec22-{fact.slug}-receipts",
                    f"{fact.label} receipts (NAICS {fact.naics})",
                    fact.receipts_usd,
                    "USD receipts",
                    "United States",
                    "2022",
                    "Census Economic Census",
                    source_url,
                    caveat,
                    "receipts",
                    "not_margin",
                    *flags,
                ),
                _measurement(
                    f"ec22-{fact.slug}-payroll-receipts",
                    f"{fact.label} annual payroll divided by receipts (NAICS {fact.naics})",
                    fact.payroll_receipts_percent,
                    "percent",
                    "United States",
                    "2022",
                    "Census Economic Census",
                    source_url,
                    caveat,
                    "derived_ratio",
                    "labor_intensity_proxy",
                    "not_margin",
                    *flags,
                ),
                _measurement(
                    f"ec22-{fact.slug}-receipts-est",
                    f"{fact.label} receipts per establishment (NAICS {fact.naics})",
                    fact.receipts_per_establishment_usd,
                    "USD per employer establishment",
                    "United States",
                    "2022",
                    "Census Economic Census",
                    source_url,
                    caveat,
                    "derived_ratio",
                    "receipts_not_startup_revenue",
                    "aggregate_not_typical",
                    *flags,
                ),
                _measurement(
                    f"ec22-{fact.slug}-receipts-employee",
                    f"{fact.label} receipts per employee (NAICS {fact.naics})",
                    fact.receipts_per_employee_usd,
                    "USD per employee",
                    "United States",
                    "2022",
                    "Census Economic Census",
                    source_url,
                    caveat,
                    "derived_ratio",
                    "not_margin",
                    "aggregate_not_typical",
                    *flags,
                ),
            )
        )
    return measurements


def _irs_corporation_measurements() -> list[PacketMeasurement]:
    facts = (
        ("consulting", "Management, scientific, and technical consulting", 7.68),
        ("admin-waste", "Administrative support and waste management", 6.95),
        ("health-social", "Health care and social assistance", 6.04),
        ("nursing-residential", "Hospitals, nursing, and residential care", 4.10),
        ("repair", "Repair and maintenance", 5.45),
        ("other-repair", "Other repair and maintenance", 7.69),
        ("construction", "Construction", 5.38),
        ("electrical", "Electrical contractors", 5.00),
    )
    caveat = (
        "Derived as aggregate net income less deficit divided by total receipts in IRS "
        "corporation tax-return statistics. It is a broad aggregate tax-return proxy, "
        "not a typical-company, startup, contribution-margin, free-cash-flow, or investor "
        "return estimate; broad minor-industry groups can mix business models."
    )
    return [
        _measurement(
            f"irs-corp-{slug}-net-income-proxy",
            f"{label} aggregate corporate net-income margin proxy",
            value,
            "percent of total receipts",
            "United States",
            "tax year 2022",
            "IRS Statistics of Income - Corporation",
            _IRS_CORPORATION_2022,
            caveat,
            "derived_ratio",
            "aggregate_tax_return_proxy",
            "not_typical_firm_margin",
            "not_investor_return",
            "broad_industry_group",
        )
        for slug, label, value in facts
    ]


_SCHEDULE_C_FACTS: Final = (
    _ScheduleCFact(
        "consulting-us",
        "Management, scientific, and technical consulting",
        "United States",
        47_881_000_000,
        25_766_000_000,
        53.8,
    ),
    _ScheduleCFact(
        "consulting-ca",
        "Management, scientific, and technical consulting",
        "California",
        6_700_000_000,
        3_626_000_000,
        54.1,
    ),
    _ScheduleCFact(
        "admin-us",
        "Administrative and support services",
        "United States",
        82_839_000_000,
        18_461_000_000,
        22.3,
    ),
    _ScheduleCFact(
        "admin-ca",
        "Administrative and support services",
        "California",
        9_998_000_000,
        3_098_000_000,
        31.0,
    ),
    _ScheduleCFact(
        "waste-us",
        "Waste management and remediation services",
        "United States",
        3_308_000_000,
        259_000_000,
        7.8,
    ),
    _ScheduleCFact(
        "waste-ca",
        "Waste management and remediation services",
        "California",
        293_000_000,
        21_500_000,
        7.3,
    ),
    _ScheduleCFact(
        "health-social-us",
        "Health care and social assistance",
        "United States",
        127_279_000_000,
        46_401_000_000,
        36.5,
    ),
    _ScheduleCFact(
        "health-social-ca",
        "Health care and social assistance",
        "California",
        16_177_000_000,
        6_202_000_000,
        38.3,
    ),
    _ScheduleCFact(
        "ambulatory-us",
        "Ambulatory health care services",
        "United States",
        98_588_000_000,
        38_369_000_000,
        38.9,
    ),
    _ScheduleCFact(
        "ambulatory-ca",
        "Ambulatory health care services",
        "California",
        11_517_000_000,
        4_885_000_000,
        42.4,
    ),
    _ScheduleCFact(
        "social-assistance-us",
        "Social assistance",
        "United States",
        20_250_000_000,
        6_683_000_000,
        33.0,
    ),
    _ScheduleCFact(
        "social-assistance-ca",
        "Social assistance",
        "California",
        3_425_000_000,
        1_168_000_000,
        34.1,
    ),
    _ScheduleCFact(
        "repair-us",
        "Repair and maintenance",
        "United States",
        57_029_000_000,
        7_074_000_000,
        12.4,
    ),
    _ScheduleCFact(
        "repair-ca",
        "Repair and maintenance",
        "California",
        7_563_000_000,
        1_292_000_000,
        17.1,
    ),
    _ScheduleCFact(
        "building-construction-us",
        "Construction of buildings",
        "United States",
        105_616_000_000,
        13_095_000_000,
        12.4,
    ),
    _ScheduleCFact(
        "building-construction-ca",
        "Construction of buildings",
        "California",
        8_132_000_000,
        1_332_000_000,
        16.4,
    ),
)


def _schedule_c_measurements() -> list[PacketMeasurement]:
    caveat = (
        "Derived from aggregate Schedule C net profit or loss divided by gross receipts. "
        "Sole-proprietor wages are not a deductible expense, so net profit includes "
        "compensation for owner labor. It is not comparable to corporate or investor "
        "margin, does not price a hired replacement for the owner, and is a broad "
        "sole-proprietorship tax-return aggregate."
    )
    measurements: list[PacketMeasurement] = []
    for fact in _SCHEDULE_C_FACTS:
        measurements.extend(
            (
                _measurement(
                    f"irs-sp-{fact.slug}-receipts",
                    f"{fact.label} Schedule C gross receipts",
                    fact.gross_receipts_usd,
                    "USD gross receipts",
                    fact.geography,
                    "tax year 2023",
                    "IRS Statistics of Income - Sole Proprietorship",
                    _IRS_SCHEDULE_C_STATE_2023,
                    caveat,
                    "sole_proprietorships",
                    "aggregate_tax_return_proxy",
                    "owner_labor_not_expensed",
                    "not_investor_margin",
                ),
                _measurement(
                    f"irs-sp-{fact.slug}-net-profit",
                    f"{fact.label} Schedule C net profit or loss",
                    fact.net_profit_usd,
                    "USD net profit or loss",
                    fact.geography,
                    "tax year 2023",
                    "IRS Statistics of Income - Sole Proprietorship",
                    _IRS_SCHEDULE_C_STATE_2023,
                    caveat,
                    "sole_proprietorships",
                    "aggregate_tax_return_proxy",
                    "owner_labor_not_expensed",
                    "not_investor_margin",
                ),
                _measurement(
                    f"irs-sp-{fact.slug}-net-profit-ratio",
                    f"{fact.label} Schedule C net-profit-to-receipts proxy",
                    fact.net_profit_receipts_percent,
                    "percent of gross receipts",
                    fact.geography,
                    "tax year 2023",
                    "IRS Statistics of Income - Sole Proprietorship",
                    _IRS_SCHEDULE_C_STATE_2023,
                    caveat,
                    "derived_ratio",
                    "sole_proprietorships",
                    "aggregate_tax_return_proxy",
                    "owner_labor_not_expensed",
                    "not_investor_margin",
                ),
            )
        )
    return measurements


def _cross_market_unknowns() -> list[PacketMeasurement]:
    return [
        _measurement(
            "unknown-repair-private-wtp",
            "Direct private-market willingness to pay for a consolidated equipment "
            "uptime, repair, and calibration service",
            None,
            "USD per customer per year",
            "United States",
            "unknown as of 2026-07-31",
            "Census Economic Census",
            _economic_census_url("81"),
            "Industry receipts and federal obligations do not observe willingness to pay "
            "for the proposed bundle, OEM coverage, switching costs, device mix, or "
            "contribution economics.",
            "missing_direct_measurement",
            "unknown_not_zero",
            "wtp_unobserved",
        ),
        _measurement(
            "unknown-background-concentration",
            "Background-screening provider concentration and platform commoditization",
            None,
            "market structure status",
            "United States",
            "unknown as of 2026-07-31",
            "Census Economic Census",
            _economic_census_url("56"),
            "Firm counts and receipts do not report concentration, API-platform power, "
            "data access, regulation costs, prices, or customer switching.",
            "missing_direct_measurement",
            "unknown_not_zero",
            "competition_unmeasured",
        ),
        _measurement(
            "unknown-waste-route-economics",
            "Route-level waste collection density and stressed contribution economics",
            None,
            "route economics status",
            "United States",
            "unknown as of 2026-07-31",
            "Census County Business Patterns",
            _CENSUS_CBP_2023_US,
            "National employer growth and payroll proxies do not measure local route "
            "density, disposal fees, vehicle capital, contracts, prices, or contribution "
            "margin.",
            "missing_direct_measurement",
            "unknown_not_zero",
            "unit_economics_unobserved",
        ),
        _measurement(
            "unknown-aging-wtp-workforce",
            "Direct aging-service willingness to pay, reimbursable demand, and available "
            "caregiver capacity",
            None,
            "validated market status",
            "California",
            "unknown as of 2026-07-31",
            "Census Economic Census",
            _economic_census_url("62"),
            "Provider and demographic proxies do not establish customer willingness to "
            "pay, payer coverage, caregiver availability, licensing scope, service slots, "
            "or stressed unit economics.",
            "missing_direct_measurement",
            "unknown_not_zero",
            "wtp_unobserved",
            "workforce_capacity_unobserved",
        ),
    ]


def build_pilot_evidence_packet() -> EvidencePacket:
    """Build the frozen official-primary-source packet used by the pilot scan."""
    measurements = (
        *_naics_classification_measurements(),
        *_regulatory_measurements(),
        *_c16_supply_measurements(),
        *_federal_demand_measurements(),
        *_cbp_measurements(),
        *_bls_survival_measurements(),
        *_aging_measurements(),
        *_economic_census_measurements(),
        *_irs_corporation_measurements(),
        *_schedule_c_measurements(),
        *_cross_market_unknowns(),
    )
    return EvidencePacket(
        packet_id="pilot-official-2026-07-31-v2",
        as_of=_AS_OF,
        measurements=measurements,
        allowed_geographies=(
            "United States",
            "California",
            "El Dorado County, California",
            "Nevada County, California",
            "Madera County, California",
            "Santa Cruz County, California",
            "Ventura County, California",
            "Riverside County, California",
            "Humboldt County, California",
            "Kings County, California",
            "Yolo County, California",
            "San Diego County, California",
            "Merced County, California",
            "Tulare County, California",
            "Santa Clara County, California",
            "Fresno County, California",
            "Santa Barbara County, California",
            "Stanislaus County, California",
            "Imperial County, California",
        ),
        allowed_scenarios=tuple(Scenario),
        allowed_naics_codes=PILOT_ALLOWED_NAICS_CODES,
        source_policy=(
            "Official primary administrative, survey, tax, and regulatory sources only. "
            "All derived ratios retain their input scope and proxy caveats. Unknown is "
            "never zero; a mandate is never assumed to be contestable spend; employer "
            "establishments are never treated as firms; 2022 NAICS classifications are "
            "scope boundaries, never market economics; no weighted master score."
        ),
    )


def pilot_evidence_packet_json(packet: EvidencePacket | None = None) -> str:
    """Return stable, compact JSON suitable for hashing and run artifacts."""
    selected = packet or build_pilot_evidence_packet()
    return (
        json.dumps(
            selected.model_dump(mode="json"),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


__all__ = [
    "PILOT_ALLOWED_NAICS_CODES",
    "PILOT_EXCLUDED_NAICS_CODES",
    "build_pilot_evidence_packet",
    "pilot_evidence_packet_json",
]
