"""2022 Economic Census Basic Data fixed-bulk source tests."""

import csv
import hashlib
import io
import stat
import zipfile

import httpx
import pytest

from app.venture.sources.economic_census import (
    ANNUAL_PAYROLL_LABEL,
    ECONOMIC_CENSUS_NAICS_VINTAGE,
    EMPLOYEE_LABEL,
    ESTABLISHMENT_LABEL,
    FIRM_LABEL,
    PILOT_NAICS_CODES,
    RECEIPTS_LABEL,
    EconomicCensusSource,
    parse_economic_census_basic_zip,
)
from app.venture.sources.errors import SourceParseError
from app.venture.sources.http import (
    ECONOMIC_CENSUS_2022_BASIC_54,
    ECONOMIC_CENSUS_2022_BASIC_56,
    ECONOMIC_CENSUS_2022_BASIC_62,
    ECONOMIC_CENSUS_2022_BASIC_81,
    SafeHttpClient,
)

_BASE_HEADER = (
    "#GEOTYPE",
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
)

_PILOT_VALUES = {
    "811210": (
        "Electronic and precision equipment repair and maintenance",
        10_441,
        11_702,
        16_664_385,
        5_351_063,
        96_345,
        32.1,
        1_424_000,
        172_966,
    ),
    "541380": (
        "Testing laboratories and services",
        5_690,
        7_970,
        27_932_032,
        11_193_145,
        164_755,
        40.1,
        3_505_000,
        169_537,
    ),
    "621610": (
        "Home health care services",
        27_774,
        39_218,
        114_187_602,
        55_548_138,
        1_560_134,
        48.6,
        2_912_000,
        73_191,
    ),
    "624120": (
        "Services for the elderly and persons with disabilities",
        30_629,
        38_947,
        77_281_380,
        40_170_394,
        1_545_401,
        52.0,
        1_984_000,
        50_007,
    ),
    "561611": (
        "Investigation and personal background check services",
        3_799,
        4_060,
        6_154_657,
        2_049_772,
        45_643,
        33.3,
        1_516_000,
        134_843,
    ),
    "561621": (
        "Security systems services (except locksmiths)",
        6_161,
        7_462,
        31_313_513,
        8_708_745,
        140_876,
        27.8,
        4_196_000,
        222_277,
    ),
    "541350": (
        "Building inspection services",
        7_459,
        7_624,
        4_464_207,
        1_692_885,
        31_488,
        37.9,
        585_547,
        141_775,
    ),
    "541614": (
        "Process, physical distribution, and logistics consulting services",
        8_668,
        9_637,
        26_038_041,
        6_801_960,
        103_192,
        26.1,
        2_702_000,
        252_326,
    ),
    "541620": (
        "Environmental consulting services",
        8_498,
        9_932,
        21_384_227,
        7_635_831,
        99_085,
        35.7,
        2_153_000,
        215_817,
    ),
}

_ARCHIVES = {
    "54": ("EC2254BASIC", True),
    "56": ("EC2256BASIC", False),
    "62": ("EC2262BASIC", True),
    "81": ("EC2281BASIC", True),
}

_ENDPOINTS = {
    "54": ECONOMIC_CENSUS_2022_BASIC_54,
    "56": ECONOMIC_CENSUS_2022_BASIC_56,
    "62": ECONOMIC_CENSUS_2022_BASIC_62,
    "81": ECONOMIC_CENSUS_2022_BASIC_81,
}


def _row(
    code: str,
    *,
    label: str,
    firms: str,
    establishments: str,
    receipts: str,
    payroll: str,
    employees: str,
    tax_status: str = "00",
    flags: dict[str, str] | None = None,
) -> dict[str, str]:
    source_flags = flags or {}
    return {
        "GEOTYPE": "01",
        "GEO_ID": "0100000US",
        "GEO_LABEL": "United States",
        "SECTOR": code[:2],
        "INDLEVEL": "6",
        "NAICS2022": code,
        "NAICS2022_LABEL": label,
        "NAICS2022_F": "",
        "TAXSTAT": tax_status,
        "TAXSTAT_LABEL": (
            "All establishments"
            if tax_status == "00"
            else "Establishments subject to federal income tax"
        ),
        "YEAR": "2022",
        "FIRM": firms,
        "FIRM_F": source_flags.get("FIRM", ""),
        "ESTAB": establishments,
        "ESTAB_F": source_flags.get("ESTAB", ""),
        "RCPTOT": receipts,
        "RCPTOT_F": source_flags.get("RCPTOT", ""),
        "PAYANN": payroll,
        "PAYANN_F": source_flags.get("PAYANN", ""),
        "EMP": employees,
        "EMP_F": source_flags.get("EMP", ""),
    }


def _archive(
    sector: str,
    rows: list[dict[str, str]],
    *,
    omit_column: str | None = None,
    extra_member: bool = False,
    symlink_data: bool = False,
) -> bytes:
    prefix, uses_tax_status = _ARCHIVES[sector]
    normalized_header = tuple(column.removeprefix("#") for column in _BASE_HEADER)
    if uses_tax_status:
        normalized_header += ("TAXSTAT", "TAXSTAT_LABEL")
    if omit_column is not None:
        normalized_header = tuple(column for column in normalized_header if column != omit_column)
    written_header = tuple(
        f"#{column}" if index == 0 else column for index, column in enumerate(normalized_header)
    )
    text = io.StringIO(newline="")
    writer = csv.writer(text, delimiter="|", lineterminator="\n")
    writer.writerow(written_header)
    for row in rows:
        writer.writerow([row.get(column, "") for column in normalized_header])
    data = text.getvalue().encode("latin-1")

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        data_name = f"{prefix}.dat"
        if symlink_data:
            member = zipfile.ZipInfo(data_name)
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            zip_file.writestr(member, data)
        else:
            zip_file.writestr(data_name, data)
        zip_file.writestr(f"{prefix}_FIELDS.txt", b"field metadata")
        zip_file.writestr(f"{prefix}_README.txt", b"official readme")
        if extra_member:
            zip_file.writestr("extra.txt", b"unexpected")
    return stream.getvalue()


def _pilot_archives() -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for sector in _ARCHIVES:
        rows: list[dict[str, str]] = []
        for code, values in _PILOT_VALUES.items():
            if code[:2] != sector:
                continue
            label, firms, establishments, receipts, payroll, employees, _, _, _ = values
            rows.append(
                _row(
                    code,
                    label=str(label),
                    firms=str(firms),
                    establishments=str(establishments),
                    receipts=str(receipts),
                    payroll=str(payroll),
                    employees=str(employees),
                )
            )
            if _ARCHIVES[sector][1]:
                rows.append(
                    _row(
                        code,
                        label=str(label),
                        firms="1",
                        establishments="1",
                        receipts="1",
                        payroll="1",
                        employees="1",
                        tax_status="T",
                    )
                )
        result[sector] = _archive(sector, rows)
    return result


def test_all_pilot_codes_match_exact_official_raw_values_and_labels() -> None:
    archives = _pilot_archives()
    observed_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert not request.url.query
        assert request.headers["accept"] == "application/zip"
        assert "authorization" not in request.headers
        sector = next(
            sector for sector, endpoint in _ENDPOINTS.items() if request.url == endpoint.url
        )
        observed_paths.append(request.url.path)
        return httpx.Response(
            200,
            content=archives[sector],
            headers={"Content-Type": "application/zip"},
        )

    source = EconomicCensusSource(SafeHttpClient(transport=httpx.MockTransport(handler)))
    result = source.fetch_basic_data(PILOT_NAICS_CODES)

    assert [record.naics_code for record in result.records] == list(PILOT_NAICS_CODES)
    assert len(result.captures) == 4
    assert set(observed_paths) == {
        "/programs-surveys/economic-census/data/2022/sector54/EC2254BASIC.zip",
        "/programs-surveys/economic-census/data/2022/sector56/EC2256BASIC.zip",
        "/programs-surveys/economic-census/data/2022/sector62/EC2262BASIC.zip",
        "/programs-surveys/economic-census/data/2022/sector81/EC2281BASIC.zip",
    }

    for capture in result.captures:
        sector = capture.endpoint_name.rsplit("-", 1)[1]
        assert capture.request_url == _ENDPOINTS[sector].url
        assert capture.request_body is None
        assert capture.body_sha256 == hashlib.sha256(archives[sector]).hexdigest()

    for record in result.records:
        label, firms, establishments, receipts, payroll, employees, _, _, _ = _PILOT_VALUES[
            record.naics_code
        ]
        assert record.naics_label == label
        assert record.firms == firms
        assert record.establishments == establishments
        assert record.receipts_thousands == receipts
        assert record.annual_payroll_thousands == payroll
        assert record.employees == employees
        assert record.naics_vintage == ECONOMIC_CENSUS_NAICS_VINTAGE
        assert record.firm_label == FIRM_LABEL
        assert record.establishment_label == ESTABLISHMENT_LABEL
        assert record.receipts_label == RECEIPTS_LABEL
        assert record.annual_payroll_label == ANNUAL_PAYROLL_LABEL
        assert record.employee_label == EMPLOYEE_LABEL
        assert not hasattr(record, "margin")


def test_explicit_proxy_transforms_match_audited_rounded_sanity_values() -> None:
    archives = _pilot_archives()

    def handler(request: httpx.Request) -> httpx.Response:
        sector = next(
            sector for sector, endpoint in _ENDPOINTS.items() if request.url == endpoint.url
        )
        return httpx.Response(
            200,
            content=archives[sector],
            headers={"Content-Type": "application/zip"},
        )

    records = (
        EconomicCensusSource(SafeHttpClient(transport=httpx.MockTransport(handler)))
        .fetch_basic_data(PILOT_NAICS_CODES)
        .records
    )

    for record in records:
        *_, payroll_percent, per_establishment, per_employee = _PILOT_VALUES[record.naics_code]
        payroll_proxy = record.payroll_to_receipts_proxy
        establishment_proxy = record.receipts_per_establishment_usd_proxy
        employee_proxy = record.receipts_per_employee_usd_proxy
        assert payroll_proxy is not None
        assert establishment_proxy is not None
        assert employee_proxy is not None
        assert float(payroll_proxy * 100) == pytest.approx(
            payroll_percent,
            abs=0.05,
        )
        assert float(establishment_proxy) == pytest.approx(
            per_establishment,
            rel=0.001,
        )
        assert float(employee_proxy) == pytest.approx(
            per_employee,
            rel=0.0001,
        )


def test_suppressed_missing_and_observed_zero_remain_distinct() -> None:
    row = _row(
        "561611",
        label="Investigation services",
        firms="",
        establishments="0",
        receipts="0",
        payroll="0",
        employees="0",
        flags={"RCPTOT": "D", "EMP": "a"},
    )
    record = parse_economic_census_basic_zip(
        _archive("56", [row]),
        sector="56",
        naics_codes=["561611"],
    )[0]

    assert record.firms is None
    assert record.firms_flag is None
    assert record.establishments == 0
    assert record.establishments_flag is None
    assert record.receipts_thousands is None
    assert record.receipts_flag == "D"
    assert record.annual_payroll_thousands == 0
    assert record.annual_payroll_flag is None
    assert record.employees is None
    assert record.employees_flag == "a"
    assert record.payroll_to_receipts_proxy is None
    assert record.receipts_per_establishment_usd_proxy is None
    assert record.receipts_per_employee_usd_proxy is None


def test_tax_status_breakdowns_are_not_double_counted() -> None:
    total = _row(
        "541380",
        label="Testing laboratories and services",
        firms="10",
        establishments="12",
        receipts="1000",
        payroll="400",
        employees="20",
    )
    taxable = _row(
        "541380",
        label="Testing laboratories and services",
        firms="9",
        establishments="11",
        receipts="900",
        payroll="350",
        employees="18",
        tax_status="T",
    )

    records = parse_economic_census_basic_zip(
        _archive("54", [total, taxable]),
        sector="54",
        naics_codes=["541380"],
    )

    assert len(records) == 1
    assert records[0].tax_status_code == "00"
    assert records[0].tax_status_label == "All establishments"
    assert records[0].receipts_thousands == 1000


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"omit_column": "RCPTOT"}, "missing required columns"),
        ({"extra_member": True}, "audited data/fields/readme"),
        ({"symlink_data": True}, "regular files"),
    ],
)
def test_malformed_or_unsafe_archives_are_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    row = _row(
        "811210",
        label="Repair",
        firms="1",
        establishments="1",
        receipts="1",
        payroll="1",
        employees="1",
    )

    with pytest.raises(SourceParseError, match=message):
        parse_economic_census_basic_zip(
            _archive("81", [row], **kwargs),  # type: ignore[arg-type]
            sector="81",
            naics_codes=["811210"],
        )


def test_missing_requested_national_row_is_rejected() -> None:
    with pytest.raises(SourceParseError, match=r"missing requested U\.S\. rows"):
        parse_economic_census_basic_zip(
            _archive("56", []),
            sector="56",
            naics_codes=["561611"],
        )


@pytest.mark.parametrize("code", ["5413", "5413800", "abcdef", "111111"])
def test_non_exact_or_unaudited_naics_is_rejected_before_transport(code: str) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, content=b"")

    source = EconomicCensusSource(SafeHttpClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(ValueError):
        source.fetch_basic_data([code])

    assert called is False
