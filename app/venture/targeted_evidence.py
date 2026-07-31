"""Frozen evidence for the narrowed San Diego BSC and fire-service run.

This packet is intentionally small enough for an analyst to reason about the
actual offer boundaries.  It mixes official public records with a few clearly
labelled first-party operational sources.  A facility, establishment, policy,
credential listing, mandate, or advertised price is never treated as observed
demand.  Missing installed-base, backlog, outsourcing, and willingness-to-pay
measurements remain explicit ``None`` values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from app.venture.core import Scenario
from app.venture.discovery import EvidencePacket, PacketMeasurement

_AS_OF: Final = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

_CENSUS_CBP_2023_COUNTY: Final = (
    "https://www2.census.gov/programs-surveys/cbp/datasets/2023/cbp23co.zip"
)
_CENSUS_NAICS_2022: Final = (
    "https://www.census.gov/naics/reference_files_tools/2022_NAICS_Manual.pdf"
)
_BMBL_6: Final = (
    "https://www.cdc.gov/labs/media/pdfs/2025/08/SF__19a_308133-A_BMBL6_00-BOOK-WEB-final-3.pdf"
)
_UCSD_ENGINEERING_CONTROLS: Final = (
    "https://blink.ucsd.edu/safety/research-lab/laboratory/engineering.html"
)
_NSF_BSC_CERTIFIERS_CA: Final = (
    "https://info.nsf.org/Certified/Biosafety-Certifier/Listings.asp?PhysStateList=CALIFORNIA"
)
_OMNIA_FLAGSHIP_PRICING: Final = (
    "https://www.omniapartners.com/suppliers-files/E-J/Flagship/"
    "Contract_Documents/2025004593/"
    "2025004593_Flagship_MAD_2025_10_08__for_website_with_pricing_.pdf"
)
_CDSS_RCFE_DATA: Final = (
    "https://data.ca.gov/api/3/action/datastore_search?"
    "resource_id=6b2f5818-f60d-40b5-bc2a-94f995f9f8b0"
)
_CDSS_RCFE_RULES: Final = "https://www.cdss.ca.gov/ord/entres/getinfo/pdf/rcfeman2.pdf"
_CA_OSFM_AES_2026: Final = (
    "https://34c031f8-c9fd-4018-8c5a-4159cdff6b0d-cdn-endpoint.azureedge.net/"
    "-/media/osfm-website/what-we-do/fire-engineering-and-investigations/"
    "automatic-fire-extinguishing-systems/2026/"
    "aes-automatic-fire-extinguishing-systems-laws-and-regulations2026updated.pdf"
    "?hash=8EE052D56AAD48F436F98427E08DB64F"
    "&rev=cddd1239a5014003b5da85e96e52f844"
)
_SDFD_COMPLIANCE_ENGINE: Final = (
    "https://www.sandiego.gov/fire/community-risk-reduction/"
    "fire-protection-systems/compliance-engine"
)
_CSLB_C16_DATA: Final = "https://www.cslb.ca.gov/Onlineservices/DataPortal/ListByClassification"
_SOCAL_FIRE_PRICE: Final = (
    "https://www.socalfireservice.com/services-store/p/fire-extinguisher-annual-service"
)
_SAN_DIEGO_CITY_PO: Final = (
    "https://apps.sandiego.gov/directories/purchasing/pdf/pos/2018/4500094800.pdf"
)

# These are activity boundaries, not a list of attractive markets.  The list is
# deliberately exact and contains only the two narrowed customer/provider paths
# plus the mutually exclusive fire-service classifications needed to prevent a
# mixed inspection, maintenance, and installation thesis.
TARGETED_ALLOWED_NAICS_CODES: Final = (
    "238220",
    "541380",
    "541714",
    "541990",
    "623312",
    "811210",
    "811310",
)


@dataclass(frozen=True, slots=True)
class _NaicsFact:
    code: str
    title: str
    scope: str
    caveat: str
    flags: tuple[str, ...]


_NAICS_FACTS: Final = (
    _NaicsFact(
        "238220",
        "Plumbing, Heating, and Air-Conditioning Contractors",
        "The official examples and index include fire-sprinkler installation, "
        "fire-extinguisher installation and repair, and waterless fire-suppression "
        "system installation and repair.",
        "This is the installation-linked fire boundary and a broad trade containing "
        "substantial non-fire plumbing and HVAC activity. It does not measure fire "
        "projects, suppliers, prices, capacity, or demand.",
        ("fire_installation_boundary", "broad_industry_proxy"),
    ),
    _NaicsFact(
        "541380",
        "Testing Laboratories and Services",
        "Physical, chemical, and other analytical testing, including calibration, "
        "electrical and electronic, mechanical, and nondestructive testing in a "
        "laboratory or on-site.",
        "This code is allowed only for a testing, calibration, or certification-led "
        "offer. Repair or maintenance of scientific equipment is a different activity. "
        "Classification does not establish BSC demand, accreditation, or economics.",
        ("testing_certification_only", "equipment_service_boundary"),
    ),
    _NaicsFact(
        "541714",
        "Research and Development in Biotechnology (except Nanobiotechnology)",
        "Research and experimental development in biotechnology other than "
        "nanobiotechnology, including work involving cellular, biomolecular, and "
        "microorganism processes.",
        "This is a potential customer establishment class, not a count of wet labs, "
        "biosafety cabinets, buying entities, budgets, or demand for certification.",
        ("customer_class_not_demand", "customer_eligible", "biotech_buyer_proxy"),
    ),
    _NaicsFact(
        "541990",
        "All Other Professional, Scientific, and Technical Services",
        "The official alphabetic index places fire-extinguisher and waterless "
        "fire-suppression-system testing or inspection here only when no sales, "
        "service, or installation is performed.",
        "This residual code is allowed only for fire inspection or testing without "
        "sales, repair, maintenance service, or installation. It is not an economic "
        "measure of the proposed route.",
        ("fire_testing_inspection_only", "fire_activity_boundary"),
    ),
    _NaicsFact(
        "623312",
        "Assisted Living Facilities for the Elderly",
        "Residential and personal care services for elderly persons who cannot or do "
        "not wish to live independently, generally including room, board, supervision, "
        "and assistance with daily living.",
        "A California RCFE license record is not a one-to-one NAICS assignment. This "
        "customer code does not measure fire systems, outsourced service, occupancy, "
        "ownership concentration, or demand.",
        (
            "customer_class_not_demand",
            "customer_eligible",
            "license_category_not_naics",
        ),
    ),
    _NaicsFact(
        "811210",
        "Electronic and Precision Equipment Repair and Maintenance",
        "Repair and maintenance of electronic and precision equipment and "
        "instruments without retailing them as new; the official examples include "
        "scientific instruments and medical equipment.",
        "This is the comparison boundary for preventive maintenance or repair of "
        "scientific equipment such as a biosafety cabinet. Testing, calibration, or "
        "certification-led work remains a different activity under 541380. The scope "
        "does not establish local BSC demand, provider capacity, price, or margin.",
        ("equipment_repair_maintenance_boundary", "equipment_service_boundary"),
    ),
    _NaicsFact(
        "811310",
        "Commercial and Industrial Machinery and Equipment Repair and Maintenance",
        "Repair and maintenance of commercial and industrial machinery; the official "
        "index places commercial fire-extinguisher and waterless fire-suppression "
        "repair or maintenance without installation here.",
        "This is the no-installation repair or maintenance boundary. Pure inspection "
        "or testing belongs in 541990 and installation-linked work in 238220. The broad "
        "category does not measure the fire-service niche.",
        ("fire_repair_without_installation", "fire_activity_boundary"),
    ),
)


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
    provenance_flag: str,
    *quality_flags: str,
) -> PacketMeasurement:
    """Create one source-labelled observation without upgrading provenance."""
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
        quality_flags=(provenance_flag, *quality_flags),
    )


def _naics_measurements() -> tuple[PacketMeasurement, ...]:
    return tuple(
        _measurement(
            f"naics22-{fact.code}-scope",
            f"2022 NAICS scope: {fact.code} {fact.title}",
            fact.scope,
            "official classification scope",
            "United States",
            "2022 NAICS; reviewed 2026-07-31",
            "Census 2022 NAICS Classification",
            _CENSUS_NAICS_2022,
            (
                "NAICS classifies establishments by primary activity for statistical "
                f"purposes. {fact.caveat}"
            ),
            "official_primary",
            "classification_only",
            "not_economic_measurement",
            *fact.flags,
        )
        for fact in _NAICS_FACTS
    )


def _biotech_buyer_measurements() -> tuple[PacketMeasurement, ...]:
    buyer_caveat = (
        "County Business Patterns covers employer establishments, not firms or all "
        "businesses. The 2023 county file uses its published NAICS vintage. "
        "Establishments are not wet labs, BSCs, buying entities, budgets, or demand."
    )
    noisy_caveat = (
        f"{buyer_caveat} Employment and payroll are Census noise-infused values and "
        "must not be treated as exact operating totals."
    )
    related_caveat = (
        "This is an employer-establishment screening proxy only. It does not identify "
        "BSC ownership, certification scope, willingness to serve, active capacity, "
        "backlog, or demand."
    )
    return (
        _measurement(
            "cbp-sd-biotech-rd-establishments-2023",
            "Biotechnology R&D employer establishments (NAICS 541714)",
            437,
            "employer establishments",
            "San Diego County, California",
            "2023",
            "Census County Business Patterns",
            _CENSUS_CBP_2023_COUNTY,
            buyer_caveat,
            "official_primary",
            "employer_establishment_not_firm",
            "buyer_proxy",
            "not_demand_measurement",
            "not_device_count",
        ),
        _measurement(
            "cbp-sd-biotech-rd-establishments-20-249-2023",
            "Biotechnology R&D employer establishments with 20-249 employees",
            88,
            "employer establishments",
            "San Diego County, California",
            "2023",
            "Census County Business Patterns",
            _CENSUS_CBP_2023_COUNTY,
            (
                f"{buyer_caveat} Derived as 41 establishments with 20-49 employees, "
                "30 with 50-99, and 17 with 100-249; size does not establish laboratory "
                "type or purchasing authority."
            ),
            "official_primary",
            "derived_sum",
            "size_bucket_proxy",
            "buyer_proxy",
            "not_demand_measurement",
            "not_device_count",
        ),
        _measurement(
            "cbp-sd-biotech-rd-employees-2023",
            "Biotechnology R&D employment (NAICS 541714)",
            20_270,
            "employees",
            "San Diego County, California",
            "2023",
            "Census County Business Patterns",
            _CENSUS_CBP_2023_COUNTY,
            noisy_caveat,
            "official_primary",
            "noise_infused",
            "buyer_scale_proxy",
            "not_demand_measurement",
            "not_device_count",
        ),
        _measurement(
            "cbp-sd-biotech-rd-annual-payroll-2023",
            "Biotechnology R&D annual payroll (NAICS 541714)",
            3_690_733_000,
            "USD annual payroll",
            "San Diego County, California",
            "2023",
            "Census County Business Patterns",
            _CENSUS_CBP_2023_COUNTY,
            (
                f"{noisy_caveat} Payroll is not revenue, equipment spend, service spend, "
                "margin, or willingness to pay."
            ),
            "official_primary",
            "noise_infused",
            "payroll_not_spend",
            "not_demand_measurement",
        ),
        _measurement(
            "cbp-sd-biological-product-establishments-2023",
            "Biological product manufacturing employer establishments (NAICS 325414)",
            17,
            "employer establishments",
            "San Diego County, California",
            "2023",
            "Census County Business Patterns",
            _CENSUS_CBP_2023_COUNTY,
            related_caveat,
            "official_primary",
            "related_customer_proxy",
            "employer_establishment_not_firm",
            "not_demand_measurement",
        ),
        _measurement(
            "cbp-sd-medical-lab-establishments-2023",
            "Medical laboratory employer establishments (NAICS 621511)",
            130,
            "employer establishments",
            "San Diego County, California",
            "2023",
            "Census County Business Patterns",
            _CENSUS_CBP_2023_COUNTY,
            related_caveat,
            "official_primary",
            "related_customer_proxy",
            "employer_establishment_not_firm",
            "not_demand_measurement",
        ),
        _measurement(
            "cbp-sd-testing-lab-establishments-2023",
            "Testing laboratories and services employer establishments (NAICS 541380)",
            93,
            "employer establishments",
            "San Diego County, California",
            "2023",
            "Census County Business Patterns",
            _CENSUS_CBP_2023_COUNTY,
            (
                f"{related_caveat} NAICS 541380 is much broader than BSC field "
                "certification, so this is not a count of qualified BSC providers."
            ),
            "official_primary",
            "broad_supplier_proxy",
            "employer_establishment_not_firm",
            "not_capacity_measurement",
        ),
    )


def _bsc_operational_measurements() -> tuple[PacketMeasurement, ...]:
    return (
        _measurement(
            "bmbl-bsc-certification-cadence",
            "BMBL BSC validation and certification cadence",
            "before service; after repair or relocation; at least annually",
            "recommended certification cadence",
            "United States",
            "BMBL 6th edition; reviewed 2026-07-31",
            "CDC/NIH Biosafety in Microbiological and Biomedical Laboratories",
            _BMBL_6,
            (
                "BMBL is federal biosafety guidance, not a statute or universal "
                "regulation. The recommendation establishes a safety practice, not "
                "installed base, outsourced spend, private willingness to pay, or demand."
            ),
            "official_guidance",
            "advisory_not_mandate",
            "recurring_practice",
            "not_demand_measurement",
        ),
        _measurement(
            "ucsd-bsc-annual-certification-policy",
            "UC San Diego BSC annual certification policy",
            True,
            "public institutional policy",
            "University of California San Diego",
            "web page reviewed 2026-07-31",
            "UC San Diego Environment, Health and Safety",
            _UCSD_ENGINEERING_CONTROLS,
            (
                "The UC San Diego page states that biosafety cabinets are required to be "
                "certified annually. This is UC San Diego's first-party institutional "
                "policy, not a statewide rule, a buyer count, an outsourced-spend record, "
                "or private-market demand."
            ),
            "public_institution_first_party",
            "institution_specific",
            "recurring_practice",
            "not_demand_measurement",
        ),
        _measurement(
            "nsf-sd-bsc-current-certifiers",
            "NSF-listed BSC field certifier individuals with San Diego County addresses",
            22,
            "listed certifier individuals",
            "San Diego County, California",
            "registry retrieved 2026-07-31",
            "NSF Biosafety Cabinet Field Certifier Accreditation",
            _NSF_BSC_CERTIFIERS_CA,
            (
                "The current California listing contained 22 individual entries with "
                "San Diego County addresses: 12 associated with TSS, 7 with Occupational "
                "Services, 2 with R&D Laboratory Equipment, and 1 with AABC. NSF is a "
                "non-government credentialing organization. Listed individuals are not "
                "firms, active service capacity, availability, service territory, quality, "
                "price, or backlog; the count is counterevidence to an assumed empty market."
            ),
            "non_government_credential_registry_first_party",
            "individual_not_firm",
            "address_not_service_territory",
            "not_capacity_measurement",
            "counterevidence",
        ),
        _measurement(
            "nsf-sd-bsc-employer-address-groups",
            "Distinct normalized employers represented by NSF-listed San Diego entries",
            4,
            "employer address groups",
            "San Diego County, California",
            "registry retrieved 2026-07-31",
            "NSF Biosafety Cabinet Field Certifier Accreditation",
            _NSF_BSC_CERTIFIERS_CA,
            (
                "The four normalized employer names are TSS, Occupational Services, R&D "
                "Laboratory Equipment, and AABC. Employer labels and addresses do not "
                "establish independent ownership, local capacity, backlog, or willingness "
                "to subcontract."
            ),
            "non_government_credential_registry_first_party",
            "derived_distinct_count",
            "address_not_service_territory",
            "not_capacity_measurement",
            "counterevidence",
        ),
        _measurement(
            "omnia-bsc-nsf49-certification-price-range",
            "Published cooperative-contract BSC NSF 49 certification price range",
            "145-189",
            "USD per cabinet",
            "United States",
            "UC agreement 2025004593 pricing schedules; reviewed 2026-07-31",
            "UC / OMNIA Partners / Flagship public contract",
            _OMNIA_FLAGSHIP_PRICING,
            (
                "This is a published tiered cooperative-contract schedule from a named "
                "supplier, not an observed private transaction, San Diego quote, accepted "
                "price, margin, or willingness-to-pay measurement. Travel, parts, volume, "
                "and site conditions may change an actual invoice."
            ),
            "public_contract_first_party",
            "named_supplier",
            "price_benchmark_not_transaction",
            "not_private_wtp",
        ),
        _measurement(
            "omnia-bsc-pm-price-range",
            "Published cooperative-contract BSC preventive-maintenance price range",
            "550-715",
            "USD per cabinet",
            "United States",
            "UC agreement 2025004593 pricing schedules; reviewed 2026-07-31",
            "UC / OMNIA Partners / Flagship public contract",
            _OMNIA_FLAGSHIP_PRICING,
            (
                "This is a published tiered cooperative-contract schedule from a named "
                "supplier, not an observed private transaction, San Diego quote, accepted "
                "price, margin, or willingness-to-pay measurement. It does not establish "
                "that certification and preventive maintenance are purchased together."
            ),
            "public_contract_first_party",
            "named_supplier",
            "price_benchmark_not_transaction",
            "not_private_wtp",
        ),
        _measurement(
            "derived-bsc-certification-pm-price-range",
            "Arithmetic sum of published BSC certification and PM schedule ranges",
            "695-904",
            "USD per cabinet",
            "United States",
            "derived from UC agreement 2025004593 schedules",
            "UC / OMNIA Partners / Flagship public contract",
            _OMNIA_FLAGSHIP_PRICING,
            (
                "This simply adds the two published range endpoints. It is not a quoted "
                "bundle, transaction, San Diego price, margin, or willingness-to-pay "
                "measurement."
            ),
            "public_contract_first_party",
            "derived_sum",
            "price_benchmark_not_transaction",
            "not_private_wtp",
        ),
    )


def _bsc_unknowns() -> tuple[PacketMeasurement, ...]:
    return (
        _measurement(
            "unknown-sd-bsc-installed-base",
            "Installed BSC base among San Diego target establishments",
            None,
            "biosafety cabinets",
            "San Diego County, California",
            "unknown as of 2026-07-31",
            "Targeted evidence gap record",
            _CENSUS_CBP_2023_COUNTY,
            (
                "CBP establishment, employment, and payroll rows contain no equipment "
                "inventory. The 437 target establishments and 88 size-filtered "
                "establishments must not be converted into BSC counts."
            ),
            "research_gap_record",
            "missing_direct_measurement",
            "unknown_not_zero",
            "not_device_count",
        ),
        _measurement(
            "unknown-sd-bsc-provider-backlog",
            "Active qualified BSC provider backlog and next-available service date",
            None,
            "days to available appointment",
            "San Diego County, California",
            "unknown as of 2026-07-31",
            "Targeted evidence gap record",
            _NSF_BSC_CERTIFIERS_CA,
            (
                "The NSF registry lists credentialed individuals but reports no schedule, "
                "utilization, queue, response time, missed service, or willingness to "
                "subcontract."
            ),
            "research_gap_record",
            "missing_direct_measurement",
            "unknown_not_zero",
            "backlog_unobserved",
        ),
        _measurement(
            "unknown-sd-bsc-private-wtp",
            "Private buyer willingness to pay for a BSC compliance and uptime contract",
            None,
            "USD per cabinet per year",
            "San Diego County, California",
            "unknown as of 2026-07-31",
            "Targeted evidence gap record",
            _OMNIA_FLAGSHIP_PRICING,
            (
                "A public cooperative schedule is not a private accepted offer. No "
                "customer-level quote acceptance, deposit, renewal, invoice, or budget "
                "record for the proposed San Diego offer was observed."
            ),
            "research_gap_record",
            "missing_direct_measurement",
            "unknown_not_zero",
            "private_wtp_unobserved",
        ),
    )


def _senior_facility_measurements() -> tuple[PacketMeasurement, ...]:
    snapshot_caveat = (
        "Derived from the California open-data RCFE resource by filtering county_name "
        "SAN DIEGO and facility_status LICENSED. The resource file_date is 2025-05-25 "
        "and the retrieved CSV SHA-256 was "
        "f35887f9d6aed0c857ba6c20ae0460d2ece3146e8ee0fa19cb711c8dd9e22797. "
        "License records are not unique owners, occupied facilities, fire systems, "
        "extinguishers, service visits, outsourced spend, or demand."
    )
    return (
        _measurement(
            "cdss-sd-licensed-rcfe-ccrc-records",
            "Licensed RCFE and RCFE-CCRC records in San Diego County",
            581,
            "licensed facility records",
            "San Diego County, California",
            "resource file_date 2025-05-25; retrieved 2026-07-31",
            "California Department of Social Services Open Data",
            _CDSS_RCFE_DATA,
            (
                f"{snapshot_caveat} The total comprises 569 RESIDENTIAL CARE ELDERLY "
                "records and 12 RCFE-CONTINUING CARE RETIREMENT COMMUNITY records."
            ),
            "official_primary",
            "administrative_record_snapshot",
            "facility_record_not_owner",
            "installed_base_proxy",
            "not_demand_measurement",
            "not_fire_system_count",
        ),
        _measurement(
            "cdss-sd-licensed-rcfe-ccrc-capacity",
            "Licensed capacity on San Diego RCFE and RCFE-CCRC records",
            22_248,
            "licensed resident capacity",
            "San Diego County, California",
            "resource file_date 2025-05-25; retrieved 2026-07-31",
            "California Department of Social Services Open Data",
            _CDSS_RCFE_DATA,
            (
                f"{snapshot_caveat} Licensed capacity is a regulatory maximum, not "
                "occupancy, vacancy, live availability, serviceable fire assets, or payer "
                "demand."
            ),
            "official_primary",
            "administrative_record_snapshot",
            "licensed_capacity_not_occupancy",
            "installed_base_proxy",
            "not_demand_measurement",
            "not_fire_system_count",
        ),
        _measurement(
            "cdss-rcfe-fire-clearance-rule",
            "California RCFE fire-clearance operating requirement",
            True,
            "regulatory requirement",
            "California",
            "regulation reviewed 2026-07-31",
            "California Department of Social Services RCFE Regulations",
            _CDSS_RCFE_RULES,
            (
                "Section 87202 requires facilities to maintain an approved fire "
                "clearance, with an appropriate clearance before accepting or retaining "
                "specified nonambulatory or bedridden residents. Fire clearance is not a "
                "count of systems, a recurring service interval, outsourced spend, or "
                "demand for the proposed provider."
            ),
            "official_primary",
            "regulatory_requirement",
            "mandate_not_contestable_spend",
            "not_demand_measurement",
        ),
    )


def _fire_operational_measurements() -> tuple[PacketMeasurement, ...]:
    regulation_caveat = (
        "The rule establishes a compliance duty for covered automatic extinguishing "
        "systems. It does not count systems, measure active supplier capacity, require "
        "every owner to use the proposed provider, report prices, or establish "
        "contestable demand."
    )
    return (
        _measurement(
            "ca-osfm-fixed-system-semiannual-itm",
            "California fixed engineered and pre-engineered system ITM cadence",
            "at least semi-annually and immediately after activation",
            "minimum regulatory cadence",
            "California",
            "2026 AES laws and regulations; reviewed 2026-07-31",
            "California Office of the State Fire Marshal",
            _CA_OSFM_AES_2026,
            (
                f"{regulation_caveat} This cadence applies to engineered and "
                "pre-engineered fixed extinguishing systems; it must not be generalized "
                "to every portable extinguisher or every facility asset."
            ),
            "official_primary",
            "regulatory_requirement",
            "system_type_specific",
            "mandate_not_contestable_spend",
            "not_demand_measurement",
        ),
        _measurement(
            "ca-osfm-aes-service-license-paths",
            "Authorized California business license paths for servicing automatic "
            "extinguishing systems",
            "OSFM A Type 1 or Type 2, or CSLB C-16 as applicable",
            "license path",
            "California",
            "2026 AES laws and regulations; reviewed 2026-07-31",
            "California Office of the State Fire Marshal",
            _CA_OSFM_AES_2026,
            (
                "Section 905 requires a valid OSFM A license or C-16 license for the "
                "business of servicing automatic extinguishing systems and distinguishes "
                "Type 1 water-based from Type 2 engineered/pre-engineered fixed systems. "
                "Owner Type L and other stated exceptions must be resolved for the exact "
                "work. Licensing does not establish low supply or demand."
            ),
            "official_primary",
            "regulatory_requirement",
            "license_scope_specific",
            "not_supply_measurement",
            "not_demand_measurement",
        ),
        _measurement(
            "ca-osfm-aes-liability-insurance",
            "Minimum liability and property-damage insurance for AES service licensees",
            1_000_000,
            "USD combined single limit per occurrence",
            "California",
            "2026 AES laws and regulations; reviewed 2026-07-31",
            "California Office of the State Fire Marshal",
            _CA_OSFM_AES_2026,
            (
                "Section 905 requires the policy to be maintained for licensed automatic "
                "extinguishing-system service. This is a provider compliance cost, not "
                "proof of scarcity, price, financeability, margin, or customer demand."
            ),
            "official_primary",
            "regulatory_requirement",
            "provider_cost_boundary",
            "not_demand_measurement",
        ),
        _measurement(
            "ca-osfm-aes-records-invoice-deficiencies",
            "California AES service record, deficiency, and invoice duties",
            ("records retained; deficiencies corrected before tag; itemized invoice provided"),
            "regulatory workflow",
            "California",
            "2026 AES laws and regulations; reviewed 2026-07-31",
            "California Office of the State Fire Marshal",
            _CA_OSFM_AES_2026,
            (
                "Section 904.2 requires testing and maintenance records to be retained "
                "on premises for five years after the next required service, deficiencies "
                "to be corrected before a tag is affixed, and an itemized invoice for "
                "work and parts. Workflow burden is not software demand, outsourced "
                "spend, price, or backlog."
            ),
            "official_primary",
            "regulatory_requirement",
            "reporting_workflow",
            "mandate_not_contestable_spend",
            "not_demand_measurement",
        ),
        _measurement(
            "sdfd-compliance-engine-reporting",
            "City of San Diego electronic ITM report submission requirement",
            "all compliant and non-compliant reports; submit PDF within 14 days",
            "local reporting workflow",
            "City of San Diego, California",
            "city page reviewed 2026-07-31; process effective 2020-04-15",
            "San Diego Fire-Rescue Department",
            _SDFD_COMPLIANCE_ENGINE,
            (
                "The City requires registered service providers in its jurisdiction to "
                "submit inspection, installation, testing, and maintenance reports "
                "through The Compliance Engine. This applies to City jurisdiction, not "
                "all San Diego County facilities. Reporting burden is not demand, private "
                "willingness to pay, price, or proof of an underserved market."
            ),
            "official_primary",
            "local_requirement",
            "reporting_workflow",
            "geography_scope_limited",
            "not_demand_measurement",
        ),
        _measurement(
            "cslb-sd-clear-c16-addresses",
            "CLEAR C-16 license address-of-record rows in San Diego County",
            93,
            "CLEAR license address rows",
            "San Diego County, California",
            "CSLB export retrieved 2026-07-31T08:10:06Z",
            "California Contractors State License Board",
            _CSLB_C16_DATA,
            (
                "The CSLB classification export XLSX SHA-256 was "
                "8b5dec0ab82188be688da355812decb30040c2437871ab3fc7e4f5862cdd6e0c. "
                "CLEAR status and address of record do not establish active availability, "
                "service territory, pure-play status, price, capacity, backlog, quality, "
                "or willingness to serve. C-16 also covers installation, so this is "
                "counterevidence to an empty market, not an exact competitor count."
            ),
            "official_primary",
            "license_address_snapshot",
            "address_not_service_territory",
            "not_capacity_measurement",
            "counterevidence",
        ),
        _measurement(
            "socal-portable-extinguisher-service-call-price",
            "Posted San Diego portable-extinguisher annual-service call fee",
            39.0,
            "USD per service call",
            "San Diego metro area, California",
            "provider page retrieved 2026-07-31",
            "So-Cal State Fire Protection",
            _SOCAL_FIRE_PRICE,
            (
                "This is a single vendor's advertised first-party price, not a government "
                "record, accepted transaction, representative market quote, margin, or "
                "willingness-to-pay measure. It is for portable extinguishers, not fixed "
                "engineered or pre-engineered systems."
            ),
            "vendor_first_party",
            "non_government_source",
            "single_vendor",
            "advertised_price_not_transaction",
            "portable_not_fixed_system",
        ),
        _measurement(
            "socal-portable-extinguisher-per-unit-price",
            "Posted San Diego portable-extinguisher annual certification price",
            12.5,
            "USD per extinguisher",
            "San Diego metro area, California",
            "provider page retrieved 2026-07-31",
            "So-Cal State Fire Protection",
            _SOCAL_FIRE_PRICE,
            (
                "This is a single vendor's advertised first-party price in addition to "
                "the posted service-call fee, not a government record, accepted "
                "transaction, representative market quote, margin, or private "
                "willingness-to-pay measure. It cannot price fixed fire systems."
            ),
            "vendor_first_party",
            "non_government_source",
            "single_vendor",
            "advertised_price_not_transaction",
            "portable_not_fixed_system",
        ),
        _measurement(
            "sd-city-kitchen-system-itm-unit-price-2017",
            "City of San Diego annual kitchen-system inspection/test/maintenance price",
            119.0,
            "USD per inspection visit",
            "City of San Diego, California",
            "City purchase order dated 2017-10-17 for FY 2017-18",
            "City of San Diego Purchase Order 4500094800",
            _SAN_DIEGO_CITY_PO,
            (
                "The purchase order listed six annual kitchen "
                "inspection/test/maintenance units at $119 each. This is a historical "
                "municipal contract benchmark, not a current 2026 price, RCFE transaction, "
                "private accepted offer, representative market price, margin, or "
                "willingness-to-pay measurement."
            ),
            "official_primary",
            "historical_public_procurement_price",
            "price_benchmark_not_current",
            "price_benchmark_not_private_wtp",
            "fixed_kitchen_system_specific",
        ),
    )


def _fire_unknowns() -> tuple[PacketMeasurement, ...]:
    return (
        _measurement(
            "unknown-sd-senior-fire-installed-systems",
            "Fire-extinguisher and fixed-system installed base at San Diego senior facilities",
            None,
            "covered fire assets",
            "San Diego County, California",
            "unknown as of 2026-07-31",
            "Targeted evidence gap record",
            _CDSS_RCFE_DATA,
            (
                "CDSS facility and licensed-capacity records contain no portable "
                "extinguisher, kitchen system, sprinkler, or other fire-system inventory. "
                "Facility counts must not be converted into asset or service-visit counts."
            ),
            "research_gap_record",
            "missing_direct_measurement",
            "unknown_not_zero",
            "not_fire_system_count",
        ),
        _measurement(
            "unknown-sd-fire-provider-backlog",
            "Active qualified fire-provider backlog and next-available service date",
            None,
            "days to available appointment",
            "San Diego County, California",
            "unknown as of 2026-07-31",
            "Targeted evidence gap record",
            _CSLB_C16_DATA,
            (
                "CSLB license-address rows report no active workload, utilization, queue, "
                "response time, unfilled work, quote delay, or willingness to subcontract."
            ),
            "research_gap_record",
            "missing_direct_measurement",
            "unknown_not_zero",
            "backlog_unobserved",
        ),
        _measurement(
            "unknown-sd-fire-outsourced-share",
            "Share of covered San Diego senior-facility fire work bought from outside providers",
            None,
            "percent of covered work",
            "San Diego County, California",
            "unknown as of 2026-07-31",
            "Targeted evidence gap record",
            _CA_OSFM_AES_2026,
            (
                "The regulations establish covered work and license paths but do not "
                "report owner insourcing, bundled incumbent work, third-party purchase "
                "incidence, renewal, or addressable outsourced spend."
            ),
            "research_gap_record",
            "missing_direct_measurement",
            "unknown_not_zero",
            "outsourced_share_unobserved",
            "mandate_not_contestable_spend",
        ),
        _measurement(
            "unknown-sd-fire-private-wtp",
            "Private senior-facility willingness to pay for the proposed fire-service route",
            None,
            "USD per facility per year",
            "San Diego County, California",
            "unknown as of 2026-07-31",
            "Targeted evidence gap record",
            _SAN_DIEGO_CITY_PO,
            (
                "A vendor advertisement and historical City purchase order do not show "
                "an RCFE buyer's accepted price. No current private quote acceptance, "
                "deposit, contract, invoice, renewal, or budget record was observed."
            ),
            "research_gap_record",
            "missing_direct_measurement",
            "unknown_not_zero",
            "private_wtp_unobserved",
        ),
    )


def build_targeted_evidence_packet() -> EvidencePacket:
    """Build the frozen packet for the final narrowed San Diego run."""
    measurements = (
        *_naics_measurements(),
        *_biotech_buyer_measurements(),
        *_bsc_operational_measurements(),
        *_bsc_unknowns(),
        *_senior_facility_measurements(),
        *_fire_operational_measurements(),
        *_fire_unknowns(),
    )
    return EvidencePacket(
        packet_id="targeted-san-diego-2026-07-31-v3",
        as_of=_AS_OF,
        measurements=measurements,
        allowed_geographies=(
            "United States",
            "California",
            "San Diego County, California",
            "City of San Diego, California",
        ),
        allowed_scenarios=(Scenario.BOOTSTRAPPED, Scenario.OPERATOR_HEAVY),
        allowed_naics_codes=TARGETED_ALLOWED_NAICS_CODES,
        source_policy=(
            "Frozen targeted evidence only. Government administrative, regulatory, "
            "guidance, and classification records retain their exact scope. UC San Diego "
            "policy is labelled public-institution first-party; NSF is labelled a "
            "non-government credential registry; the UC/OMNIA/Flagship schedule is a "
            "named-supplier public-contract benchmark; and the posted portable-"
            "extinguisher price is labelled vendor first-party. A count, policy, mandate, "
            "license, credential listing, or price sheet is never called demand. "
            "Establishments are not firms, facilities are not fire systems, listed "
            "people or addresses are not capacity, and unknown is never zero. No elder-"
            "care operating thesis is included because direct buyer WTP and workforce "
            "availability were not observed. Scientific-equipment maintenance and "
            "repair are compared with 811210 rather than the fire-specific 811310 "
            "boundary. No weighted master score."
        ),
    )


def targeted_evidence_packet_json(packet: EvidencePacket | None = None) -> str:
    """Return stable compact JSON for hashing and run artifacts."""
    selected = packet or build_targeted_evidence_packet()
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
    "TARGETED_ALLOWED_NAICS_CODES",
    "build_targeted_evidence_packet",
    "targeted_evidence_packet_json",
]
