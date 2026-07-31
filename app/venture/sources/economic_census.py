"""2022 Economic Census Basic Data from fixed anonymous sector bulk files.

The Census data API now requires credentials for data queries.  This adapter
therefore uses four audited HTTPS bulk ZIPs that cover the requested service
sectors.  It selects U.S. rows for exact six-digit 2022 NAICS codes and, where
the file includes tax-status breakouts, uses only ``TAXSTAT=00`` (all
establishments).
"""

from __future__ import annotations

import csv
import io
import re
import stat
import zipfile
from collections.abc import Collection
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from app.venture.sources.errors import SourceParseError
from app.venture.sources.http import (
    ECONOMIC_CENSUS_2022_BASIC_54,
    ECONOMIC_CENSUS_2022_BASIC_56,
    ECONOMIC_CENSUS_2022_BASIC_62,
    ECONOMIC_CENSUS_2022_BASIC_81,
    FixedEndpoint,
    RawCapture,
    SafeHttpClient,
)

ECONOMIC_CENSUS_YEAR: Final = 2022
ECONOMIC_CENSUS_NAICS_VINTAGE: Final = 2022
PILOT_NAICS_CODES: Final = (
    "811210",
    "541380",
    "621610",
    "624120",
    "561611",
    "561621",
    "541350",
    "541614",
    "541620",
)

FIRM_LABEL: Final = "Number of firms"
ESTABLISHMENT_LABEL: Final = "Number of establishments"
RECEIPTS_LABEL: Final = "Sales, value of shipments, or revenue ($1,000)"
ANNUAL_PAYROLL_LABEL: Final = "Annual payroll ($1,000)"
EMPLOYEE_LABEL: Final = "Number of employees"
ECONOMIC_CENSUS_MEASUREMENT_NOTE: Final = (
    "2022 Economic Census Basic Data for U.S. employer establishments. Receipts "
    "and payroll are reported in $1,000s. Derived payroll/receipts and per-unit "
    "values are operating-scale proxies, not profit or margin measures."
)

_MAX_REQUESTED_CODES: Final = 100
_NAICS_CODE: Final = re.compile(r"^\d{6}$")
_NONNEGATIVE_INTEGER: Final = re.compile(r"^\d+$")
_SUPPRESSION_OR_MISSING_FLAGS: Final = frozenset({"D", "S", "N", "X"})
_MAX_METADATA_MEMBER_BYTES: Final = 16 * 1024


@dataclass(frozen=True, slots=True)
class _SectorSpec:
    sector: str
    endpoint: FixedEndpoint
    archive_name: str
    data_member: str
    fields_member: str
    readme_member: str
    max_zip_bytes: int
    max_data_bytes: int
    max_rows: int
    uses_tax_status: bool


_SECTOR_SPECS: Final[dict[str, _SectorSpec]] = {
    "54": _SectorSpec(
        sector="54",
        endpoint=ECONOMIC_CENSUS_2022_BASIC_54,
        archive_name="EC2254BASIC.zip",
        data_member="EC2254BASIC.dat",
        fields_member="EC2254BASIC_FIELDS.txt",
        readme_member="EC2254BASIC_README.txt",
        max_zip_bytes=10 * 1024 * 1024,
        max_data_bytes=148 * 1024 * 1024,
        max_rows=750_000,
        uses_tax_status=True,
    ),
    "56": _SectorSpec(
        sector="56",
        endpoint=ECONOMIC_CENSUS_2022_BASIC_56,
        archive_name="EC2256BASIC.zip",
        data_member="EC2256BASIC.dat",
        fields_member="EC2256BASIC_FIELDS.txt",
        readme_member="EC2256BASIC_README.txt",
        max_zip_bytes=5 * 1024 * 1024,
        max_data_bytes=40 * 1024 * 1024,
        max_rows=300_000,
        uses_tax_status=False,
    ),
    "62": _SectorSpec(
        sector="62",
        endpoint=ECONOMIC_CENSUS_2022_BASIC_62,
        archive_name="EC2262BASIC.zip",
        data_member="EC2262BASIC.dat",
        fields_member="EC2262BASIC_FIELDS.txt",
        readme_member="EC2262BASIC_README.txt",
        max_zip_bytes=13 * 1024 * 1024,
        max_data_bytes=178 * 1024 * 1024,
        max_rows=900_000,
        uses_tax_status=True,
    ),
    "81": _SectorSpec(
        sector="81",
        endpoint=ECONOMIC_CENSUS_2022_BASIC_81,
        archive_name="EC2281BASIC.zip",
        data_member="EC2281BASIC.dat",
        fields_member="EC2281BASIC_FIELDS.txt",
        readme_member="EC2281BASIC_README.txt",
        max_zip_bytes=10 * 1024 * 1024,
        max_data_bytes=144 * 1024 * 1024,
        max_rows=700_000,
        uses_tax_status=True,
    ),
}

_BASE_COLUMNS: Final = frozenset(
    {
        "GEOTYPE",
        "GEO_ID",
        "GEO_LABEL",
        "SECTOR",
        "INDLEVEL",
        "NAICS2022",
        "NAICS2022_LABEL",
        "NAICS2022_F",
        "YEAR",
        "FIRM",
        "FIRM_F",
        "ESTAB",
        "ESTAB_F",
        "RCPTOT",
        "RCPTOT_F",
        "PAYANN",
        "PAYANN_F",
        "EMP",
        "EMP_F",
    }
)


@dataclass(frozen=True, slots=True)
class EconomicCensusBasicRecord:
    """One national all-establishment row for an exact six-digit NAICS code."""

    year: int
    naics_code: str
    naics_label: str
    naics_footnote: str | None
    naics_vintage: int
    sector: str
    geography_id: str
    geography_label: str
    tax_status_code: str | None
    tax_status_label: str | None
    firms: int | None
    firms_flag: str | None
    establishments: int | None
    establishments_flag: str | None
    receipts_thousands: int | None
    receipts_flag: str | None
    annual_payroll_thousands: int | None
    annual_payroll_flag: str | None
    employees: int | None
    employees_flag: str | None
    firm_label: str = FIRM_LABEL
    establishment_label: str = ESTABLISHMENT_LABEL
    receipts_label: str = RECEIPTS_LABEL
    annual_payroll_label: str = ANNUAL_PAYROLL_LABEL
    employee_label: str = EMPLOYEE_LABEL
    measurement_note: str = ECONOMIC_CENSUS_MEASUREMENT_NOTE

    @property
    def payroll_to_receipts_proxy(self) -> Decimal | None:
        """Annual payroll divided by receipts; this is not a profit margin."""
        return payroll_to_receipts_proxy(self)

    @property
    def receipts_per_establishment_usd_proxy(self) -> Decimal | None:
        """Receipts per establishment in dollars as an operating-scale proxy."""
        return receipts_per_establishment_usd_proxy(self)

    @property
    def receipts_per_employee_usd_proxy(self) -> Decimal | None:
        """Receipts per employee in dollars as an operating-scale proxy."""
        return receipts_per_employee_usd_proxy(self)


@dataclass(frozen=True, slots=True)
class EconomicCensusBasicResult:
    """Frozen sector archives and requested national Basic Data records."""

    captures: tuple[RawCapture, ...]
    requested_naics_codes: tuple[str, ...]
    records: tuple[EconomicCensusBasicRecord, ...]


class EconomicCensusSource:
    """Fetch fixed anonymous 2022 Basic Data sector archives."""

    def __init__(self, client: SafeHttpClient | None = None) -> None:
        self._client = client or SafeHttpClient(max_body_bytes=13 * 1024 * 1024)

    def fetch_basic_data(
        self,
        naics_codes: Collection[str],
    ) -> EconomicCensusBasicResult:
        """Fetch exact U.S. rows, making one fixed bulk request per needed sector."""
        normalized = _normalize_naics_codes(naics_codes)
        codes_by_sector: dict[str, list[str]] = {}
        sector_order: list[str] = []
        for code in normalized:
            sector = code[:2]
            if sector not in codes_by_sector:
                codes_by_sector[sector] = []
                sector_order.append(sector)
            codes_by_sector[sector].append(code)

        captures: list[RawCapture] = []
        records_by_code: dict[str, EconomicCensusBasicRecord] = {}
        for sector in sector_order:
            spec = _SECTOR_SPECS[sector]
            capture = self._client.request(spec.endpoint)
            captures.append(capture)
            for record in parse_economic_census_basic_zip(
                capture.raw_bytes,
                sector=sector,
                naics_codes=codes_by_sector[sector],
            ):
                records_by_code[record.naics_code] = record

        return EconomicCensusBasicResult(
            captures=tuple(captures),
            requested_naics_codes=normalized,
            records=tuple(records_by_code[code] for code in normalized),
        )


def parse_economic_census_basic_zip(
    raw_bytes: bytes,
    *,
    sector: str,
    naics_codes: Collection[str],
) -> tuple[EconomicCensusBasicRecord, ...]:
    """Parse one fixed sector ZIP and return requested national six-digit rows."""
    try:
        spec = _SECTOR_SPECS[sector]
    except KeyError as exc:
        supported = ", ".join(sorted(_SECTOR_SPECS))
        raise ValueError(f"Economic Census sector must be one of: {supported}") from exc
    normalized = _normalize_naics_codes(naics_codes)
    if any(code[:2] != sector for code in normalized):
        raise ValueError(f"all requested NAICS codes must be in sector {sector}")
    if len(raw_bytes) > spec.max_zip_bytes:
        raise SourceParseError(f"{spec.archive_name} exceeds the {spec.max_zip_bytes}-byte ZIP cap")

    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            _validate_archive(archive, spec)
            archive.read(spec.fields_member)
            archive.read(spec.readme_member)
            records = _parse_data_member(archive, spec, normalized)
    except SourceParseError:
        raise
    except (
        EOFError,
        OSError,
        RuntimeError,
        UnicodeError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise SourceParseError(f"{spec.archive_name} is not a valid readable archive") from exc

    missing = set(normalized).difference(record.naics_code for record in records)
    if missing:
        raise SourceParseError(
            f"{spec.archive_name} is missing requested U.S. rows: {sorted(missing)!r}"
        )
    return records


def payroll_to_receipts_proxy(
    record: EconomicCensusBasicRecord,
) -> Decimal | None:
    """Return payroll/receipts as a scale proxy, never labeled as margin."""
    payroll = record.annual_payroll_thousands
    receipts = record.receipts_thousands
    if payroll is None or receipts is None or receipts == 0:
        return None
    return Decimal(payroll) / Decimal(receipts)


def receipts_per_establishment_usd_proxy(
    record: EconomicCensusBasicRecord,
) -> Decimal | None:
    """Return receipts per establishment in dollars as a scale proxy."""
    receipts = record.receipts_thousands
    establishments = record.establishments
    if receipts is None or establishments is None or establishments == 0:
        return None
    return Decimal(receipts * 1000) / Decimal(establishments)


def receipts_per_employee_usd_proxy(
    record: EconomicCensusBasicRecord,
) -> Decimal | None:
    """Return receipts per employee in dollars as a scale proxy."""
    receipts = record.receipts_thousands
    employees = record.employees
    if receipts is None or employees is None or employees == 0:
        return None
    return Decimal(receipts * 1000) / Decimal(employees)


def _normalize_naics_codes(naics_codes: Collection[str]) -> tuple[str, ...]:
    values = (naics_codes,) if isinstance(naics_codes, str) else naics_codes
    normalized = tuple(dict.fromkeys(value.strip() for value in values))
    if not normalized:
        raise ValueError("at least one exact six-digit NAICS code is required")
    if len(normalized) > _MAX_REQUESTED_CODES:
        raise ValueError(f"at most {_MAX_REQUESTED_CODES} NAICS codes may be requested")
    if any(not _NAICS_CODE.fullmatch(code) for code in normalized):
        raise ValueError("Economic Census NAICS codes must be exactly six digits")
    unsupported = sorted({code[:2] for code in normalized}.difference(_SECTOR_SPECS))
    if unsupported:
        supported = ", ".join(sorted(_SECTOR_SPECS))
        raise ValueError(
            f"NAICS sectors {unsupported!r} are not covered; supported sectors: {supported}"
        )
    return normalized


def _validate_archive(archive: zipfile.ZipFile, spec: _SectorSpec) -> None:
    expected = {spec.data_member, spec.fields_member, spec.readme_member}
    members = archive.infolist()
    if len(members) != len(expected) or {member.filename for member in members} != expected:
        raise SourceParseError(
            f"{spec.archive_name} must contain exactly its audited data/fields/readme members"
        )
    for member in members:
        if (
            member.filename.startswith("/")
            or "/" in member.filename
            or "\\" in member.filename
            or member.filename in {".", ".."}
        ):
            raise SourceParseError(f"{spec.archive_name} contains an unsafe member path")
        mode = (member.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if member.is_dir() or stat.S_ISLNK(mode):
            raise SourceParseError(f"{spec.archive_name} members must be regular files")
        if file_type not in (0, stat.S_IFREG):
            raise SourceParseError(f"{spec.archive_name} contains a non-regular member")
        if member.flag_bits & 0x1:
            raise SourceParseError(f"{spec.archive_name} members cannot be encrypted")
        if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise SourceParseError(f"{spec.archive_name} uses unsupported compression")
        if member.compress_size < 0 or member.file_size < 0:
            raise SourceParseError(f"{spec.archive_name} has invalid member sizes")
        member_cap = (
            spec.max_data_bytes
            if member.filename == spec.data_member
            else _MAX_METADATA_MEMBER_BYTES
        )
        if member.file_size > member_cap or member.compress_size > spec.max_zip_bytes:
            raise SourceParseError(f"{spec.archive_name} member exceeds its size cap")


def _parse_data_member(
    archive: zipfile.ZipFile,
    spec: _SectorSpec,
    naics_codes: tuple[str, ...],
) -> tuple[EconomicCensusBasicRecord, ...]:
    selected = set(naics_codes)
    records: list[EconomicCensusBasicRecord] = []
    seen: set[str] = set()
    with (
        archive.open(spec.data_member, "r") as binary_stream,
        io.TextIOWrapper(binary_stream, encoding="latin-1", newline="") as text_stream,
    ):
        reader = csv.reader(text_stream, delimiter="|")
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise SourceParseError(f"{spec.data_member} is empty") from exc
        header = tuple(
            column.removeprefix("#") if index == 0 else column
            for index, column in enumerate(raw_header)
        )
        required = _BASE_COLUMNS | (
            frozenset({"TAXSTAT", "TAXSTAT_LABEL"}) if spec.uses_tax_status else frozenset()
        )
        if len(set(header)) != len(header):
            raise SourceParseError(f"{spec.data_member} has duplicate columns")
        missing_columns = required.difference(header)
        if missing_columns:
            raise SourceParseError(
                f"{spec.data_member} is missing required columns: {sorted(missing_columns)!r}"
            )

        row_count = 0
        for line_number, values in enumerate(reader, start=2):
            if not values or all(not value.strip() for value in values):
                continue
            row_count += 1
            if row_count > spec.max_rows:
                raise SourceParseError(f"{spec.data_member} exceeds the {spec.max_rows}-row cap")
            if len(values) != len(header):
                raise SourceParseError(
                    f"{spec.data_member} row {line_number} has the wrong field count"
                )
            row = dict(zip(header, values, strict=True))
            if row["SECTOR"].strip() != spec.sector:
                raise SourceParseError(f"{spec.data_member} row {line_number} has the wrong sector")
            if row["YEAR"].strip() != str(ECONOMIC_CENSUS_YEAR):
                raise SourceParseError(f"{spec.data_member} row {line_number} has the wrong year")
            naics_code = row["NAICS2022"].strip()
            if (
                naics_code not in selected
                or row["GEOTYPE"].strip() != "01"
                or row["GEO_ID"].strip() != "0100000US"
            ):
                continue
            if row["INDLEVEL"].strip() != "6":
                raise SourceParseError(
                    f"{spec.data_member} row {line_number} is not six-digit detail"
                )
            if spec.uses_tax_status and row["TAXSTAT"].strip() != "00":
                continue
            if naics_code in seen:
                raise SourceParseError(
                    f"{spec.data_member} repeats the U.S. all-establishment row for {naics_code}"
                )
            seen.add(naics_code)
            records.append(_parse_record(row, spec=spec, line_number=line_number))
    return tuple(records)


def _parse_record(
    row: dict[str, str],
    *,
    spec: _SectorSpec,
    line_number: int,
) -> EconomicCensusBasicRecord:
    field_prefix = f"{spec.data_member} row {line_number}"
    firms, firms_flag = _parse_measure(
        row["FIRM"],
        row["FIRM_F"],
        field=f"{field_prefix} FIRM",
    )
    establishments, establishments_flag = _parse_measure(
        row["ESTAB"],
        row["ESTAB_F"],
        field=f"{field_prefix} ESTAB",
    )
    receipts, receipts_flag = _parse_measure(
        row["RCPTOT"],
        row["RCPTOT_F"],
        field=f"{field_prefix} RCPTOT",
    )
    payroll, payroll_flag = _parse_measure(
        row["PAYANN"],
        row["PAYANN_F"],
        field=f"{field_prefix} PAYANN",
    )
    employees, employees_flag = _parse_measure(
        row["EMP"],
        row["EMP_F"],
        field=f"{field_prefix} EMP",
    )
    return EconomicCensusBasicRecord(
        year=ECONOMIC_CENSUS_YEAR,
        naics_code=_required_text(row["NAICS2022"], field=f"{field_prefix} NAICS2022"),
        naics_label=_required_text(
            row["NAICS2022_LABEL"],
            field=f"{field_prefix} NAICS2022_LABEL",
        ),
        naics_footnote=row["NAICS2022_F"].strip() or None,
        naics_vintage=ECONOMIC_CENSUS_NAICS_VINTAGE,
        sector=spec.sector,
        geography_id="0100000US",
        geography_label=_required_text(
            row["GEO_LABEL"],
            field=f"{field_prefix} GEO_LABEL",
        ),
        tax_status_code=row["TAXSTAT"].strip() if spec.uses_tax_status else None,
        tax_status_label=(
            _required_text(
                row["TAXSTAT_LABEL"],
                field=f"{field_prefix} TAXSTAT_LABEL",
            )
            if spec.uses_tax_status
            else None
        ),
        firms=firms,
        firms_flag=firms_flag,
        establishments=establishments,
        establishments_flag=establishments_flag,
        receipts_thousands=receipts,
        receipts_flag=receipts_flag,
        annual_payroll_thousands=payroll,
        annual_payroll_flag=payroll_flag,
        employees=employees,
        employees_flag=employees_flag,
    )


def _parse_measure(
    value: str,
    source_flag: str,
    *,
    field: str,
) -> tuple[int | None, str | None]:
    text = value.strip()
    flag = source_flag.strip() or None
    if flag is not None and len(flag) > 4:
        raise SourceParseError(f"{field} flag is too long")
    parsed: int | None
    if not text:
        parsed = None
    elif not _NONNEGATIVE_INTEGER.fullmatch(text):
        raise SourceParseError(f"{field} must be a non-negative integer or null")
    else:
        parsed = int(text)
    if flag is not None and flag.upper() in _SUPPRESSION_OR_MISSING_FLAGS:
        if parsed not in (None, 0):
            raise SourceParseError(f"{field} is suppressed/missing but is not zero/null")
        return None, flag
    if flag is not None and parsed in (None, 0):
        return None, flag
    return parsed, flag


def _required_text(value: str, *, field: str) -> str:
    text = value.strip()
    if not text:
        raise SourceParseError(f"{field} must be non-empty text")
    return text


parse_economic_census_basic = parse_economic_census_basic_zip


__all__ = [
    "ANNUAL_PAYROLL_LABEL",
    "ECONOMIC_CENSUS_MEASUREMENT_NOTE",
    "ECONOMIC_CENSUS_NAICS_VINTAGE",
    "ECONOMIC_CENSUS_YEAR",
    "EMPLOYEE_LABEL",
    "ESTABLISHMENT_LABEL",
    "FIRM_LABEL",
    "PILOT_NAICS_CODES",
    "RECEIPTS_LABEL",
    "EconomicCensusBasicRecord",
    "EconomicCensusBasicResult",
    "EconomicCensusSource",
    "parse_economic_census_basic",
    "parse_economic_census_basic_zip",
    "payroll_to_receipts_proxy",
    "receipts_per_employee_usd_proxy",
    "receipts_per_establishment_usd_proxy",
]
