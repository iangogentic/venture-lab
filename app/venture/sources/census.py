"""Fixed-file U.S. Census adapters for employer and county population signals.

The Census Bureau files used here are complete, versioned releases.  Callers
select rows after downloading one audited file; they cannot supply a URL,
redirect target, or credential.  CBP employment and payroll are noise-infused,
and source zeroes carrying ``D`` or ``S`` are withheld values rather than
observed zeroes.
"""

from __future__ import annotations

import csv
import io
import re
import stat
import zipfile
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from datetime import date
from typing import Final

from app.venture.sources.errors import SourceParseError
from app.venture.sources.http import (
    CENSUS_2025_COUNTY_POPULATION,
    CENSUS_CA_2024_COUNTY_AGE_SEX,
    CENSUS_CBP_2022_COUNTY,
    CENSUS_CBP_2022_US,
    CENSUS_CBP_2023_COUNTY,
    CENSUS_CBP_2023_US,
    FixedEndpoint,
    RawCapture,
    SafeHttpClient,
)

CBP_NAICS_VINTAGE: Final = 2017
CBP_MEASUREMENT_NOTE: Final = (
    "Census County Business Patterns employer establishments, mid-March employment, "
    "and annual payroll in $1,000s. Employment and payroll are noise-infused; D/S "
    "flags are withheld values, not observed zeroes."
)
CALIFORNIA_AGE65_MEASUREMENT_NOTE: Final = (
    "Vintage 2024 California county resident population age 65 and over, "
    "estimated for July 1, 2024 (source YEAR code 6)."
)
COUNTY_POPULATION_2025_MEASUREMENT_NOTE: Final = (
    "Vintage 2025 county resident population estimate for July 1, 2025."
)

_CBP_US_MAX_ZIP_BYTES: Final = 2 * 1024 * 1024
_CBP_COUNTY_MAX_ZIP_BYTES: Final = 16 * 1024 * 1024
_CBP_US_MAX_UNCOMPRESSED_BYTES: Final = 8 * 1024 * 1024
_CBP_COUNTY_MAX_UNCOMPRESSED_BYTES: Final = 112 * 1024 * 1024
_READ_CHUNK_BYTES: Final = 64 * 1024
_CBP_US_MAX_ROWS: Final = 20_000
_CBP_COUNTY_MAX_ROWS: Final = 1_200_000
_POPULATION_MAX_ROWS: Final = 5_000
_CA_AGE_MAX_ROWS: Final = 500

_CBP_SIZE_SUFFIXES: Final = (
    "<5",
    "5_9",
    "10_19",
    "20_49",
    "50_99",
    "100_249",
    "250_499",
    "500_999",
    "1000",
)
_CBP_SIZE_BUCKET_SPECS: Final = (
    ("<5", "1-4", 1, 4),
    ("5_9", "5-9", 5, 9),
    ("10_19", "10-19", 10, 19),
    ("20_49", "20-49", 20, 49),
    ("50_99", "50-99", 50, 99),
    ("100_249", "100-249", 100, 249),
    ("250_499", "250-499", 250, 499),
    ("500_999", "500-999", 500, 999),
    ("1000", "1,000+", 1000, None),
)
_CBP_US_BUCKET_COLUMNS: Final = tuple(
    column
    for suffix in _CBP_SIZE_SUFFIXES
    for column in (
        f"e{suffix}nf",
        f"e{suffix}",
        f"q{suffix}nf",
        f"q{suffix}",
        f"a{suffix}nf",
        f"a{suffix}",
        f"n{suffix}",
    )
)
CBP_US_COLUMNS: Final = (
    "uscode",
    "naics",
    "lfo",
    "emp_nf",
    "emp",
    "qp1_nf",
    "qp1",
    "ap_nf",
    "ap",
    "est",
    *_CBP_US_BUCKET_COLUMNS,
)
CBP_COUNTY_COLUMNS: Final = (
    "fipstate",
    "fipscty",
    "naics",
    "emp_nf",
    "emp",
    "qp1_nf",
    "qp1",
    "ap_nf",
    "ap",
    "est",
    "n<5",
    "n5_9",
    "n10_19",
    "n20_49",
    "n50_99",
    "n100_249",
    "n250_499",
    "n500_999",
    "n1000",
    "n1000_1",
    "n1000_2",
    "n1000_3",
    "n1000_4",
)

_CBP_US_FILES: Final[dict[int, tuple[FixedEndpoint, str]]] = {
    2022: (CENSUS_CBP_2022_US, "cbp22us.txt"),
    2023: (CENSUS_CBP_2023_US, "cbp23us.txt"),
}
_CBP_COUNTY_FILES: Final[dict[int, tuple[FixedEndpoint, str]]] = {
    2022: (CENSUS_CBP_2022_COUNTY, "cbp22co.txt"),
    2023: (CENSUS_CBP_2023_COUNTY, "cbp23co.txt"),
}
_CA_2024_YEAR_DATES: Final[dict[int, date]] = {
    1: date(2020, 4, 1),
    2: date(2020, 7, 1),
    3: date(2021, 7, 1),
    4: date(2022, 7, 1),
    5: date(2023, 7, 1),
    6: date(2024, 7, 1),
}
_CA_2024_LATEST_YEAR_CODE: Final = 6
_NOISE_FLAGS: Final = frozenset({"G", "H", "D", "S"})
_LFO_CODES: Final = frozenset({"-", "C", "Z", "S", "P", "N", "G", "O"})
_SOURCE_NAICS: Final = re.compile(r"^(?:------|\d{2}----|\d{3}///|\d{4}//|\d{5}/|\d{6})$")
_REQUESTED_NAICS: Final = re.compile(r"^\d{2,6}$")
_STATE_FIPS: Final = re.compile(r"^\d{2}$")
_COUNTY_FIPS: Final = re.compile(r"^\d{3}$")
_FULL_COUNTY_FIPS: Final = re.compile(r"^\d{5}$")
_NONNEGATIVE_INTEGER: Final = re.compile(r"^\d+$")


@dataclass(frozen=True, slots=True)
class CbpSizeBucket:
    """One mutually exclusive employer-establishment employment-size bucket."""

    label: str
    minimum_employees: int
    maximum_employees: int | None
    establishment_count: int | None
    source_flag: str | None


@dataclass(frozen=True, slots=True)
class CbpUsRecord:
    """One U.S. CBP all-legal-forms row for a requested NAICS code."""

    year: int
    naics_code: str
    source_naics_code: str
    naics_vintage: int
    legal_form_code: str
    establishment_count: int | None
    establishment_count_flag: str | None
    employment: int | None
    employment_flag: str | None
    annual_payroll_thousands: int | None
    annual_payroll_flag: str | None
    size_buckets: tuple[CbpSizeBucket, ...]
    measurement_note: str = CBP_MEASUREMENT_NOTE


@dataclass(frozen=True, slots=True)
class CbpCountyRecord:
    """One county CBP row for a requested county and NAICS code."""

    year: int
    county_fips: str
    naics_code: str
    source_naics_code: str
    naics_vintage: int
    establishment_count: int | None
    establishment_count_flag: str | None
    employment: int | None
    employment_flag: str | None
    annual_payroll_thousands: int | None
    annual_payroll_flag: str | None
    size_buckets: tuple[CbpSizeBucket, ...]
    measurement_note: str = CBP_MEASUREMENT_NOTE


@dataclass(frozen=True, slots=True)
class CbpUsResult:
    """Frozen national CBP ZIP response and selected all-legal-forms rows."""

    capture: RawCapture
    year: int
    naics_vintage: int
    records: tuple[CbpUsRecord, ...]


@dataclass(frozen=True, slots=True)
class CbpCountyResult:
    """Frozen county CBP ZIP response and selected county/NAICS rows."""

    capture: RawCapture
    year: int
    naics_vintage: int
    records: tuple[CbpCountyRecord, ...]


@dataclass(frozen=True, slots=True)
class CaliforniaCountyAge65:
    """One California county's latest Vintage 2024 age-65-plus estimate."""

    county_fips: str
    county_name: str
    year_code: int
    estimate_date: date
    total_population: int
    age_65_plus_population: int
    vintage_year: int = 2024
    measurement_note: str = CALIFORNIA_AGE65_MEASUREMENT_NOTE


@dataclass(frozen=True, slots=True)
class CaliforniaCountyAge65Result:
    """Frozen California age/sex CSV and selected latest county rows."""

    capture: RawCapture
    records: tuple[CaliforniaCountyAge65, ...]


@dataclass(frozen=True, slots=True)
class CountyPopulation2025:
    """One county's July 1, 2025 resident population estimate."""

    county_fips: str
    state_name: str
    county_name: str
    population: int
    estimate_date: date = date(2025, 7, 1)
    vintage_year: int = 2025
    measurement_note: str = COUNTY_POPULATION_2025_MEASUREMENT_NOTE


@dataclass(frozen=True, slots=True)
class CountyPopulation2025Result:
    """Frozen Vintage 2025 totals CSV and selected county rows."""

    capture: RawCapture
    records: tuple[CountyPopulation2025, ...]


class CensusSource:
    """Fetch only the six audited Census release files represented in this module."""

    def __init__(self, client: SafeHttpClient | None = None) -> None:
        self._client = client or SafeHttpClient(max_body_bytes=_CBP_COUNTY_MAX_ZIP_BYTES)

    def fetch_cbp_us(
        self,
        year: int,
        *,
        naics_codes: Collection[str],
    ) -> CbpUsResult:
        """Fetch a fixed national CBP ZIP and retain only all-legal-forms rows."""
        endpoint, _ = _cbp_file(_CBP_US_FILES, year)
        capture = self._client.request(endpoint)
        records = parse_cbp_us_zip(
            capture.raw_bytes,
            year=year,
            naics_codes=naics_codes,
        )
        return CbpUsResult(
            capture=capture,
            year=year,
            naics_vintage=CBP_NAICS_VINTAGE,
            records=records,
        )

    def fetch_cbp_counties(
        self,
        year: int,
        *,
        county_fips: Collection[str],
        naics_codes: Collection[str],
    ) -> CbpCountyResult:
        """Fetch a fixed county CBP ZIP and retain requested county/NAICS rows."""
        endpoint, _ = _cbp_file(_CBP_COUNTY_FILES, year)
        capture = self._client.request(endpoint)
        records = parse_cbp_county_zip(
            capture.raw_bytes,
            year=year,
            county_fips=county_fips,
            naics_codes=naics_codes,
        )
        return CbpCountyResult(
            capture=capture,
            year=year,
            naics_vintage=CBP_NAICS_VINTAGE,
            records=records,
        )

    def fetch_california_county_age65(
        self,
        *,
        county_fips: Collection[str] | None = None,
    ) -> CaliforniaCountyAge65Result:
        """Fetch California Vintage 2024 selected-age rows and select YEAR code 6."""
        capture = self._client.request(CENSUS_CA_2024_COUNTY_AGE_SEX)
        records = parse_california_county_age65(
            capture.raw_bytes,
            county_fips=county_fips,
        )
        return CaliforniaCountyAge65Result(capture=capture, records=records)

    def fetch_county_population_2025(
        self,
        *,
        county_fips: Collection[str] | None = None,
    ) -> CountyPopulation2025Result:
        """Fetch Vintage 2025 totals and retain county (SUMLEV 050) rows."""
        capture = self._client.request(CENSUS_2025_COUNTY_POPULATION)
        records = parse_county_population_2025(
            capture.raw_bytes,
            county_fips=county_fips,
        )
        return CountyPopulation2025Result(capture=capture, records=records)


def extract_single_zip_member(
    raw_bytes: bytes,
    *,
    expected_member: str,
    max_compressed_bytes: int,
    max_uncompressed_bytes: int,
) -> bytes:
    """Extract one exact flat regular-file member under two independent caps."""
    if (
        not expected_member
        or expected_member in {".", ".."}
        or "/" in expected_member
        or "\\" in expected_member
    ):
        raise ValueError("expected_member must be one flat filename")
    if max_compressed_bytes < 1 or max_uncompressed_bytes < 1:
        raise ValueError("ZIP size caps must be positive")
    if len(raw_bytes) > max_compressed_bytes:
        raise SourceParseError(f"Census ZIP exceeds the {max_compressed_bytes}-byte compressed cap")

    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            members = archive.infolist()
            if len(members) != 1:
                raise SourceParseError("Census ZIP must contain exactly one member")
            member = members[0]
            if (
                member.filename.startswith("/")
                or "/" in member.filename
                or "\\" in member.filename
                or member.filename in {".", ".."}
            ):
                raise SourceParseError("Census ZIP member has an unsafe path")
            if member.filename != expected_member:
                raise SourceParseError(f"Census ZIP member must be exactly {expected_member!r}")
            mode = (member.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            if member.is_dir() or stat.S_ISLNK(mode):
                raise SourceParseError("Census ZIP member must be a regular file, not a link")
            if file_type not in (0, stat.S_IFREG):
                raise SourceParseError("Census ZIP member must be a regular file")
            if member.flag_bits & 0x1:
                raise SourceParseError("Census ZIP member cannot be encrypted")
            if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise SourceParseError("Census ZIP uses an unsupported compression method")
            if (
                member.compress_size < 0
                or member.compress_size > max_compressed_bytes
                or member.file_size < 0
                or member.file_size > max_uncompressed_bytes
            ):
                raise SourceParseError("Census ZIP member exceeds a compressed/uncompressed cap")

            chunks: list[bytes] = []
            extracted_bytes = 0
            with archive.open(member, "r") as source:
                while chunk := source.read(_READ_CHUNK_BYTES):
                    extracted_bytes += len(chunk)
                    if extracted_bytes > max_uncompressed_bytes:
                        raise SourceParseError(
                            "Census ZIP member exceeds the uncompressed byte cap"
                        )
                    chunks.append(chunk)
            if extracted_bytes != member.file_size:
                raise SourceParseError("Census ZIP member size does not match its directory entry")
            return b"".join(chunks)
    except SourceParseError:
        raise
    except (
        EOFError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise SourceParseError("Census response is not a valid readable ZIP") from exc


def parse_cbp_us_zip(
    raw_bytes: bytes,
    *,
    year: int,
    naics_codes: Collection[str],
) -> tuple[CbpUsRecord, ...]:
    """Parse requested national CBP rows, selecting only ``LFO='-'`` totals."""
    _, expected_member = _cbp_file(_CBP_US_FILES, year)
    selected_naics = set(_normalize_naics_codes(naics_codes))
    csv_bytes = extract_single_zip_member(
        raw_bytes,
        expected_member=expected_member,
        max_compressed_bytes=_CBP_US_MAX_ZIP_BYTES,
        max_uncompressed_bytes=_CBP_US_MAX_UNCOMPRESSED_BYTES,
    )
    records: list[CbpUsRecord] = []
    seen: set[str] = set()
    for line_number, row in _csv_rows(
        csv_bytes,
        source=f"CBP {year} U.S.",
        required_columns=frozenset(CBP_US_COLUMNS),
        expected_columns=CBP_US_COLUMNS,
        max_rows=_CBP_US_MAX_ROWS,
    ):
        if row["uscode"].strip() != "98":
            raise SourceParseError(f"CBP U.S. row {line_number} has invalid uscode")
        source_naics, naics_code = _parse_source_naics(
            row["naics"],
            field=f"CBP U.S. row {line_number} naics",
        )
        legal_form = row["lfo"].strip().upper()
        if legal_form not in _LFO_CODES:
            raise SourceParseError(f"CBP U.S. row {line_number} has invalid lfo")
        if legal_form != "-" or naics_code not in selected_naics:
            continue
        if naics_code in seen:
            raise SourceParseError(
                f"CBP U.S. contains duplicate all-legal-forms row for NAICS {naics_code}"
            )
        seen.add(naics_code)
        establishment_count, establishment_flag = _parse_count_with_marker(
            row["est"],
            field=f"CBP U.S. row {line_number} est",
        )
        employment, employment_flag = _parse_noise_value(
            row["emp"],
            row["emp_nf"],
            field=f"CBP U.S. row {line_number} emp",
        )
        annual_payroll, annual_payroll_flag = _parse_noise_value(
            row["ap"],
            row["ap_nf"],
            field=f"CBP U.S. row {line_number} ap",
        )
        records.append(
            CbpUsRecord(
                year=year,
                naics_code=naics_code,
                source_naics_code=source_naics,
                naics_vintage=CBP_NAICS_VINTAGE,
                legal_form_code=legal_form,
                establishment_count=establishment_count,
                establishment_count_flag=establishment_flag,
                employment=employment,
                employment_flag=employment_flag,
                annual_payroll_thousands=annual_payroll,
                annual_payroll_flag=annual_payroll_flag,
                size_buckets=_parse_size_buckets(
                    row,
                    field_prefix=f"CBP U.S. row {line_number}",
                ),
            )
        )
    return tuple(records)


def parse_cbp_county_zip(
    raw_bytes: bytes,
    *,
    year: int,
    county_fips: Collection[str],
    naics_codes: Collection[str],
) -> tuple[CbpCountyRecord, ...]:
    """Parse county CBP rows for exact requested five-digit FIPS and NAICS codes."""
    _, expected_member = _cbp_file(_CBP_COUNTY_FILES, year)
    selected_counties = set(_normalize_county_fips(county_fips))
    selected_naics = set(_normalize_naics_codes(naics_codes))
    csv_bytes = extract_single_zip_member(
        raw_bytes,
        expected_member=expected_member,
        max_compressed_bytes=_CBP_COUNTY_MAX_ZIP_BYTES,
        max_uncompressed_bytes=_CBP_COUNTY_MAX_UNCOMPRESSED_BYTES,
    )
    records: list[CbpCountyRecord] = []
    seen: set[tuple[str, str]] = set()
    for line_number, row in _csv_rows(
        csv_bytes,
        source=f"CBP {year} county",
        required_columns=frozenset(CBP_COUNTY_COLUMNS),
        expected_columns=CBP_COUNTY_COLUMNS,
        max_rows=_CBP_COUNTY_MAX_ROWS,
    ):
        full_fips = _parse_fips_parts(
            row["fipstate"],
            row["fipscty"],
            field=f"CBP county row {line_number}",
        )
        source_naics, naics_code = _parse_source_naics(
            row["naics"],
            field=f"CBP county row {line_number} naics",
        )
        if full_fips not in selected_counties or naics_code not in selected_naics:
            continue
        key = (full_fips, naics_code)
        if key in seen:
            raise SourceParseError(
                f"CBP county contains duplicate row for {full_fips} NAICS {naics_code}"
            )
        seen.add(key)
        establishment_count, establishment_flag = _parse_count_with_marker(
            row["est"],
            field=f"CBP county row {line_number} est",
        )
        employment, employment_flag = _parse_noise_value(
            row["emp"],
            row["emp_nf"],
            field=f"CBP county row {line_number} emp",
        )
        annual_payroll, annual_payroll_flag = _parse_noise_value(
            row["ap"],
            row["ap_nf"],
            field=f"CBP county row {line_number} ap",
        )
        records.append(
            CbpCountyRecord(
                year=year,
                county_fips=full_fips,
                naics_code=naics_code,
                source_naics_code=source_naics,
                naics_vintage=CBP_NAICS_VINTAGE,
                establishment_count=establishment_count,
                establishment_count_flag=establishment_flag,
                employment=employment,
                employment_flag=employment_flag,
                annual_payroll_thousands=annual_payroll,
                annual_payroll_flag=annual_payroll_flag,
                size_buckets=_parse_size_buckets(
                    row,
                    field_prefix=f"CBP county row {line_number}",
                ),
            )
        )
    return tuple(records)


def parse_california_county_age65(
    raw_bytes: bytes,
    *,
    county_fips: Collection[str] | None = None,
) -> tuple[CaliforniaCountyAge65, ...]:
    """Parse Vintage 2024 California rows and select YEAR 6 (July 1, 2024)."""
    selected = (
        None
        if county_fips is None
        else set(_normalize_county_fips(county_fips, required_state="06"))
    )
    required = frozenset(
        {
            "SUMLEV",
            "STATE",
            "COUNTY",
            "STNAME",
            "CTYNAME",
            "YEAR",
            "POPESTIMATE",
            "AGE65PLUS_TOT",
        }
    )
    records: list[CaliforniaCountyAge65] = []
    seen: set[str] = set()
    for line_number, row in _csv_rows(
        raw_bytes,
        source="Census California Vintage 2024 age/sex",
        required_columns=required,
        max_rows=_CA_AGE_MAX_ROWS,
    ):
        if row["SUMLEV"].strip() != "050":
            raise SourceParseError(f"California age row {line_number} must have SUMLEV 050")
        full_fips = _parse_fips_parts(
            row["STATE"],
            row["COUNTY"],
            field=f"California age row {line_number}",
        )
        if not full_fips.startswith("06"):
            raise SourceParseError(f"California age row {line_number} is outside California")
        year_code = _required_nonnegative_int(
            row["YEAR"],
            field=f"California age row {line_number} YEAR",
        )
        try:
            estimate_date = _CA_2024_YEAR_DATES[year_code]
        except KeyError as exc:
            raise SourceParseError(
                f"California age row {line_number} has unknown YEAR code {year_code}"
            ) from exc
        if year_code != _CA_2024_LATEST_YEAR_CODE:
            continue
        if selected is not None and full_fips not in selected:
            continue
        if full_fips in seen:
            raise SourceParseError(f"California age file has duplicate latest row for {full_fips}")
        seen.add(full_fips)
        records.append(
            CaliforniaCountyAge65(
                county_fips=full_fips,
                county_name=_required_text(
                    row["CTYNAME"],
                    field=f"California age row {line_number} CTYNAME",
                ),
                year_code=year_code,
                estimate_date=estimate_date,
                total_population=_required_nonnegative_int(
                    row["POPESTIMATE"],
                    field=f"California age row {line_number} POPESTIMATE",
                ),
                age_65_plus_population=_required_nonnegative_int(
                    row["AGE65PLUS_TOT"],
                    field=f"California age row {line_number} AGE65PLUS_TOT",
                ),
            )
        )
    return tuple(records)


def parse_county_population_2025(
    raw_bytes: bytes,
    *,
    county_fips: Collection[str] | None = None,
) -> tuple[CountyPopulation2025, ...]:
    """Parse Vintage 2025 total-population CSV county rows (SUMLEV 050)."""
    selected = None if county_fips is None else set(_normalize_county_fips(county_fips))
    required = frozenset(
        {
            "SUMLEV",
            "STATE",
            "COUNTY",
            "STNAME",
            "CTYNAME",
            "POPESTIMATE2025",
        }
    )
    records: list[CountyPopulation2025] = []
    seen: set[str] = set()
    for line_number, row in _csv_rows(
        raw_bytes,
        source="Census Vintage 2025 county population",
        required_columns=required,
        max_rows=_POPULATION_MAX_ROWS,
        encoding="latin-1",
    ):
        summary_level = row["SUMLEV"].strip()
        if summary_level == "040":
            continue
        if summary_level != "050":
            raise SourceParseError(f"County population row {line_number} has unsupported SUMLEV")
        full_fips = _parse_fips_parts(
            row["STATE"],
            row["COUNTY"],
            field=f"County population row {line_number}",
        )
        if selected is not None and full_fips not in selected:
            continue
        if full_fips in seen:
            raise SourceParseError(f"County population file has duplicate row for {full_fips}")
        seen.add(full_fips)
        records.append(
            CountyPopulation2025(
                county_fips=full_fips,
                state_name=_required_text(
                    row["STNAME"],
                    field=f"County population row {line_number} STNAME",
                ),
                county_name=_required_text(
                    row["CTYNAME"],
                    field=f"County population row {line_number} CTYNAME",
                ),
                population=_required_nonnegative_int(
                    row["POPESTIMATE2025"],
                    field=f"County population row {line_number} POPESTIMATE2025",
                ),
            )
        )
    return tuple(records)


def _cbp_file(
    files: dict[int, tuple[FixedEndpoint, str]],
    year: int,
) -> tuple[FixedEndpoint, str]:
    try:
        return files[year]
    except KeyError as exc:
        raise ValueError("CBP year must be 2022 or 2023") from exc


def _normalize_naics_codes(naics_codes: Collection[str]) -> tuple[str, ...]:
    values = (naics_codes,) if isinstance(naics_codes, str) else naics_codes
    normalized = tuple(dict.fromkeys(value.strip() for value in values))
    if not normalized:
        raise ValueError("at least one NAICS code is required")
    if any(not _REQUESTED_NAICS.fullmatch(value) for value in normalized):
        raise ValueError("NAICS codes must contain 2 to 6 digits")
    return normalized


def _normalize_county_fips(
    county_fips: Collection[str],
    *,
    required_state: str | None = None,
) -> tuple[str, ...]:
    values = (county_fips,) if isinstance(county_fips, str) else county_fips
    normalized = tuple(dict.fromkeys(value.strip() for value in values))
    if not normalized:
        raise ValueError("at least one five-digit county FIPS code is required")
    if any(not _FULL_COUNTY_FIPS.fullmatch(value) or value[2:] == "000" for value in normalized):
        raise ValueError("county FIPS codes must be five digits with a nonzero county part")
    if required_state is not None and any(
        not value.startswith(required_state) for value in normalized
    ):
        raise ValueError(f"county FIPS codes must be in state {required_state}")
    return normalized


def _parse_source_naics(value: str, *, field: str) -> tuple[str, str]:
    source_code = value.strip()
    if not _SOURCE_NAICS.fullmatch(source_code):
        raise SourceParseError(f"{field} is not a supported six-character CBP code")
    normalized = "00" if source_code == "------" else source_code.rstrip("-/")
    return source_code, normalized


def _parse_fips_parts(state: str, county: str, *, field: str) -> str:
    state_code = state.strip()
    county_code = county.strip()
    if not _STATE_FIPS.fullmatch(state_code):
        raise SourceParseError(f"{field} state FIPS must be two digits")
    if not _COUNTY_FIPS.fullmatch(county_code) or county_code == "000":
        raise SourceParseError(f"{field} county FIPS must be three nonzero digits")
    return f"{state_code}{county_code}"


def _parse_size_buckets(
    row: dict[str, str],
    *,
    field_prefix: str,
) -> tuple[CbpSizeBucket, ...]:
    buckets: list[CbpSizeBucket] = []
    for suffix, label, minimum, maximum in _CBP_SIZE_BUCKET_SPECS:
        count, marker = _parse_count_with_marker(
            row[f"n{suffix}"],
            field=f"{field_prefix} n{suffix}",
        )
        buckets.append(
            CbpSizeBucket(
                label=label,
                minimum_employees=minimum,
                maximum_employees=maximum,
                establishment_count=count,
                source_flag=marker,
            )
        )
    return tuple(buckets)


def _parse_noise_value(
    value: str,
    noise_flag: str,
    *,
    field: str,
) -> tuple[int | None, str | None]:
    flag = noise_flag.strip().upper() or None
    if flag is not None and flag not in _NOISE_FLAGS:
        raise SourceParseError(f"{field} has invalid noise/suppression flag {flag!r}")
    parsed, marker = _parse_count_with_marker(value, field=field)
    if flag in {"D", "S"}:
        if parsed not in (None, 0):
            raise SourceParseError(f"{field} is flagged withheld but is not zero/missing")
        return None, flag
    return parsed, flag or marker


def _parse_count_with_marker(
    value: str,
    *,
    field: str,
) -> tuple[int | None, str | None]:
    text = value.strip()
    if not text:
        return None, None
    if text.upper() == "N":
        return None, "N"
    if not _NONNEGATIVE_INTEGER.fullmatch(text):
        raise SourceParseError(f"{field} must be a non-negative integer or missing marker")
    return int(text), None


def _required_nonnegative_int(value: str, *, field: str) -> int:
    parsed, marker = _parse_count_with_marker(value, field=field)
    if parsed is None:
        detail = f" ({marker})" if marker is not None else ""
        raise SourceParseError(f"{field} is required{detail}")
    return parsed


def _required_text(value: str, *, field: str) -> str:
    text = value.strip()
    if not text:
        raise SourceParseError(f"{field} must be non-empty text")
    return text


def _csv_rows(
    raw_bytes: bytes,
    *,
    source: str,
    required_columns: frozenset[str],
    max_rows: int,
    expected_columns: tuple[str, ...] | None = None,
    encoding: str = "utf-8-sig",
) -> Iterator[tuple[int, dict[str, str]]]:
    try:
        text = raw_bytes.decode(encoding)
    except UnicodeDecodeError as exc:
        raise SourceParseError(f"{source} is not valid {encoding} CSV") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise SourceParseError(f"{source} CSV is empty") from exc
    header_tuple = tuple(header)
    if not header_tuple or any(not column for column in header_tuple):
        raise SourceParseError(f"{source} CSV has an empty header")
    if len(set(header_tuple)) != len(header_tuple):
        raise SourceParseError(f"{source} CSV has duplicate columns")
    missing = required_columns.difference(header_tuple)
    if missing:
        raise SourceParseError(f"{source} CSV is missing required columns: {sorted(missing)!r}")
    if expected_columns is not None and header_tuple != expected_columns:
        raise SourceParseError(f"{source} CSV columns/order do not match the audited schema")

    row_count = 0
    for line_number, values in enumerate(reader, start=2):
        if not values or all(not value.strip() for value in values):
            continue
        row_count += 1
        if row_count > max_rows:
            raise SourceParseError(f"{source} CSV exceeds the {max_rows}-row cap")
        if len(values) != len(header_tuple):
            raise SourceParseError(f"{source} CSV row {line_number} has the wrong field count")
        yield line_number, dict(zip(header_tuple, values, strict=True))


parse_cbp_us = parse_cbp_us_zip
parse_cbp_county = parse_cbp_county_zip
parse_ca_county_age65 = parse_california_county_age65


__all__ = [
    "CALIFORNIA_AGE65_MEASUREMENT_NOTE",
    "CBP_COUNTY_COLUMNS",
    "CBP_MEASUREMENT_NOTE",
    "CBP_NAICS_VINTAGE",
    "CBP_US_COLUMNS",
    "COUNTY_POPULATION_2025_MEASUREMENT_NOTE",
    "CaliforniaCountyAge65",
    "CaliforniaCountyAge65Result",
    "CbpCountyRecord",
    "CbpCountyResult",
    "CbpSizeBucket",
    "CbpUsRecord",
    "CbpUsResult",
    "CensusSource",
    "CountyPopulation2025",
    "CountyPopulation2025Result",
    "extract_single_zip_member",
    "parse_ca_county_age65",
    "parse_california_county_age65",
    "parse_cbp_county",
    "parse_cbp_county_zip",
    "parse_cbp_us",
    "parse_cbp_us_zip",
    "parse_county_population_2025",
]
