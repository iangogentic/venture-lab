"""Official Census fixed-file adapters and hostile ZIP parsing tests."""

import csv
import io
import stat
import zipfile
from datetime import date

import httpx
import pytest

from app.venture.sources.census import (
    CBP_COUNTY_COLUMNS,
    CBP_US_COLUMNS,
    CensusSource,
    extract_single_zip_member,
    parse_california_county_age65,
    parse_cbp_county_zip,
    parse_cbp_us_zip,
    parse_county_population_2025,
)
from app.venture.sources.errors import EgressPolicyError, SourceParseError
from app.venture.sources.http import CENSUS_CBP_2023_COUNTY, SafeHttpClient


def _csv_bytes(
    columns: tuple[str, ...],
    rows: list[dict[str, str]],
    *,
    encoding: str = "utf-8",
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode(encoding)


def _zip_bytes(
    member_name: str,
    content: bytes,
    *,
    symlink: bool = False,
    second_member: bool = False,
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if symlink:
            member = zipfile.ZipInfo(member_name)
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, content)
        else:
            archive.writestr(member_name, content)
        if second_member:
            archive.writestr("extra.txt", b"extra")
    return stream.getvalue()


def _us_row(
    *,
    naics: str = "62----",
    lfo: str = "-",
    establishments: str = "10",
    employment: str = "100",
    employment_flag: str = "G",
    annual_payroll: str = "1",
    annual_payroll_flag: str = "G",
) -> dict[str, str]:
    row = dict.fromkeys(CBP_US_COLUMNS, "")
    row.update(
        {
            "uscode": "98",
            "naics": naics,
            "lfo": lfo,
            "emp_nf": employment_flag,
            "emp": employment,
            "qp1_nf": "G",
            "qp1": "1",
            "ap_nf": annual_payroll_flag,
            "ap": annual_payroll,
            "est": establishments,
        }
    )
    for column in CBP_US_COLUMNS:
        if column.startswith("n") and column != "naics":
            row[column] = "0"
    return row


def _county_row(
    *,
    state: str = "06",
    county: str = "001",
    naics: str = "62----",
    establishments: str = "10",
    employment: str = "100",
    employment_flag: str = "G",
    annual_payroll: str = "1",
    annual_payroll_flag: str = "G",
) -> dict[str, str]:
    row = dict.fromkeys(CBP_COUNTY_COLUMNS, "")
    row.update(
        {
            "fipstate": state,
            "fipscty": county,
            "naics": naics,
            "emp_nf": employment_flag,
            "emp": employment,
            "qp1_nf": "G",
            "qp1": "1",
            "ap_nf": annual_payroll_flag,
            "ap": annual_payroll,
            "est": establishments,
        }
    )
    for column in CBP_COUNTY_COLUMNS:
        if column.startswith("n") and column != "naics":
            row[column] = "0"
    return row


def test_us_parser_uses_only_total_legal_form_row_without_double_counting() -> None:
    rows = [
        _us_row(lfo="-", establishments="10", employment="100"),
        _us_row(lfo="C", establishments="6", employment="60"),
        _us_row(lfo="S", establishments="4", employment="40"),
    ]
    archive = _zip_bytes("cbp23us.txt", _csv_bytes(CBP_US_COLUMNS, rows))

    records = parse_cbp_us_zip(archive, year=2023, naics_codes=["62"])

    assert len(records) == 1
    record = records[0]
    assert record.legal_form_code == "-"
    assert record.establishment_count == 10
    assert record.employment == 100
    assert record.naics_code == "62"
    assert record.source_naics_code == "62----"
    assert record.naics_vintage == 2017


def test_suppressed_and_missing_values_remain_unknown_while_zero_remains_zero() -> None:
    row = _county_row(
        establishments="0",
        employment="0",
        employment_flag="D",
        annual_payroll="0",
        annual_payroll_flag="G",
    )
    row["n<5"] = "N"
    row["n5_9"] = "0"
    archive = _zip_bytes("cbp23co.txt", _csv_bytes(CBP_COUNTY_COLUMNS, [row]))

    record = parse_cbp_county_zip(
        archive,
        year=2023,
        county_fips=["06001"],
        naics_codes=["62"],
    )[0]

    assert record.establishment_count == 0
    assert record.establishment_count_flag is None
    assert record.employment is None
    assert record.employment_flag == "D"
    assert record.annual_payroll_thousands == 0
    assert record.annual_payroll_flag == "G"
    assert record.size_buckets[0].establishment_count is None
    assert record.size_buckets[0].source_flag == "N"
    assert record.size_buckets[1].establishment_count == 0
    assert record.size_buckets[1].source_flag is None


def test_county_parser_filters_exact_fips_and_naics() -> None:
    rows = [
        _county_row(county="001", naics="5415//", establishments="12"),
        _county_row(county="013", naics="5415//", establishments="99"),
        _county_row(county="001", naics="62----", establishments="50"),
    ]
    archive = _zip_bytes("cbp22co.txt", _csv_bytes(CBP_COUNTY_COLUMNS, rows))

    records = parse_cbp_county_zip(
        archive,
        year=2022,
        county_fips=["06001"],
        naics_codes=["5415"],
    )

    assert [(record.county_fips, record.naics_code) for record in records] == [("06001", "5415")]
    assert records[0].establishment_count == 12
    assert records[0].naics_vintage == 2017


@pytest.mark.parametrize(
    ("archive", "message"),
    [
        (_zip_bytes("../cbp23us.txt", b"x"), "unsafe path"),
        (_zip_bytes("cbp23us.txt", b"x", symlink=True), "not a link"),
        (
            _zip_bytes("cbp23us.txt", b"x", second_member=True),
            "exactly one member",
        ),
    ],
)
def test_zip_paths_links_and_extra_members_are_rejected(
    archive: bytes,
    message: str,
) -> None:
    with pytest.raises(SourceParseError, match=message):
        extract_single_zip_member(
            archive,
            expected_member="cbp23us.txt",
            max_compressed_bytes=10_000,
            max_uncompressed_bytes=10_000,
        )


def test_zip_bomb_is_rejected_by_declared_uncompressed_cap() -> None:
    archive = _zip_bytes("cbp23us.txt", b"A" * 10_000)

    with pytest.raises(SourceParseError, match="compressed/uncompressed cap"):
        extract_single_zip_member(
            archive,
            expected_member="cbp23us.txt",
            max_compressed_bytes=10_000,
            max_uncompressed_bytes=100,
        )


def test_entire_zip_is_rejected_by_compressed_body_cap() -> None:
    archive = _zip_bytes("cbp23us.txt", b"fixture")

    with pytest.raises(SourceParseError, match="compressed cap"):
        extract_single_zip_member(
            archive,
            expected_member="cbp23us.txt",
            max_compressed_bytes=len(archive) - 1,
            max_uncompressed_bytes=10_000,
        )


def test_cbp_schema_mismatch_is_rejected() -> None:
    columns = tuple(column for column in CBP_US_COLUMNS if column != "ap")
    archive = _zip_bytes("cbp23us.txt", _csv_bytes(columns, []))

    with pytest.raises(SourceParseError, match="missing required columns"):
        parse_cbp_us_zip(archive, year=2023, naics_codes=["62"])


def test_california_age65_parser_selects_year_code_6_as_july_2024() -> None:
    columns = (
        "SUMLEV",
        "STATE",
        "COUNTY",
        "STNAME",
        "CTYNAME",
        "YEAR",
        "POPESTIMATE",
        "AGE65PLUS_TOT",
    )
    rows = [
        {
            "SUMLEV": "050",
            "STATE": "06",
            "COUNTY": "001",
            "STNAME": "California",
            "CTYNAME": "Alameda County",
            "YEAR": "5",
            "POPESTIMATE": "1600000",
            "AGE65PLUS_TOT": "230000",
        },
        {
            "SUMLEV": "050",
            "STATE": "06",
            "COUNTY": "001",
            "STNAME": "California",
            "CTYNAME": "Alameda County",
            "YEAR": "6",
            "POPESTIMATE": "1610000",
            "AGE65PLUS_TOT": "240000",
        },
    ]

    record = parse_california_county_age65(
        _csv_bytes(columns, rows),
        county_fips=["06001"],
    )[0]

    assert record.year_code == 6
    assert record.estimate_date == date(2024, 7, 1)
    assert record.total_population == 1_610_000
    assert record.age_65_plus_population == 240_000


def test_california_age65_rejects_literal_year_instead_of_documented_code() -> None:
    columns = (
        "SUMLEV",
        "STATE",
        "COUNTY",
        "STNAME",
        "CTYNAME",
        "YEAR",
        "POPESTIMATE",
        "AGE65PLUS_TOT",
    )
    row = {
        "SUMLEV": "050",
        "STATE": "06",
        "COUNTY": "001",
        "STNAME": "California",
        "CTYNAME": "Alameda County",
        "YEAR": "2024",
        "POPESTIMATE": "1",
        "AGE65PLUS_TOT": "1",
    }

    with pytest.raises(SourceParseError, match="unknown YEAR code 2024"):
        parse_california_county_age65(_csv_bytes(columns, [row]))


def test_population_2025_parser_uses_county_rows_and_exact_latest_column() -> None:
    columns = (
        "SUMLEV",
        "STATE",
        "COUNTY",
        "STNAME",
        "CTYNAME",
        "POPESTIMATE2025",
    )
    rows = [
        {
            "SUMLEV": "040",
            "STATE": "06",
            "COUNTY": "000",
            "STNAME": "California",
            "CTYNAME": "California",
            "POPESTIMATE2025": "40000000",
        },
        {
            "SUMLEV": "050",
            "STATE": "06",
            "COUNTY": "001",
            "STNAME": "California",
            "CTYNAME": "Doña Alameda County",
            "POPESTIMATE2025": "1678340",
        },
        {
            "SUMLEV": "050",
            "STATE": "06",
            "COUNTY": "013",
            "STNAME": "California",
            "CTYNAME": "Contra Costa County",
            "POPESTIMATE2025": "1170000",
        },
    ]

    records = parse_county_population_2025(
        _csv_bytes(columns, rows, encoding="latin-1"),
        county_fips=["06001"],
    )

    assert len(records) == 1
    assert records[0].county_fips == "06001"
    assert records[0].county_name == "Doña Alameda County"
    assert records[0].population == 1_678_340
    assert records[0].estimate_date == date(2025, 7, 1)
    assert records[0].vintage_year == 2025


@pytest.mark.parametrize(
    "raw",
    [
        b"SUMLEV,STATE,COUNTY,CTYNAME,YEAR,POPESTIMATE\n",
        b"SUMLEV,STATE,COUNTY,STNAME,CTYNAME\n",
        b"SUMLEV,STATE,STATE,COUNTY,STNAME,CTYNAME,POPESTIMATE2025\n",
    ],
)
def test_population_csv_malformed_schemas_are_rejected(raw: bytes) -> None:
    with pytest.raises(SourceParseError):
        parse_county_population_2025(raw)


def test_source_fetches_only_fixed_2023_county_zip() -> None:
    archive = _zip_bytes(
        "cbp23co.txt",
        _csv_bytes(CBP_COUNTY_COLUMNS, [_county_row()]),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == CENSUS_CBP_2023_COUNTY.url
        assert request.headers["accept"] == "application/zip"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            content=archive,
            headers={"Content-Type": "application/zip"},
        )

    source = CensusSource(SafeHttpClient(transport=httpx.MockTransport(handler)))
    result = source.fetch_cbp_counties(
        2023,
        county_fips=["06001"],
        naics_codes=["62"],
    )

    assert result.capture.endpoint_name == "census-cbp-2023-county"
    assert result.records[0].county_fips == "06001"


def test_census_endpoint_content_type_and_redirect_remain_fail_closed() -> None:
    wrong_type = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"not a zip",
            headers={"Content-Type": "application/octet-stream"},
        )
    )
    with pytest.raises(EgressPolicyError, match="content type"):
        SafeHttpClient(transport=wrong_type).request(CENSUS_CBP_2023_COUNTY)

    redirect = httpx.MockTransport(
        lambda request: httpx.Response(
            302,
            headers={"Location": "https://example.com/not-approved"},
        )
    )
    with pytest.raises(EgressPolicyError, match="redirect"):
        SafeHttpClient(transport=redirect).request(CENSUS_CBP_2023_COUNTY)


def test_census_endpoint_cap_overrides_a_more_permissive_client_cap() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"x",
            headers={
                "Content-Type": "application/zip",
                "Content-Length": str(16 * 1024 * 1024 + 1),
            },
        )
    )
    client = SafeHttpClient(transport=transport, max_body_bytes=64 * 1024 * 1024)

    with pytest.raises(EgressPolicyError, match="16777216-byte cap"):
        client.request(CENSUS_CBP_2023_COUNTY)


def test_unsupported_cbp_year_is_rejected_before_transport() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, content=b"")

    source = CensusSource(SafeHttpClient(transport=httpx.MockTransport(handler)))

    with pytest.raises(ValueError, match="2022 or 2023"):
        source.fetch_cbp_us(2024, naics_codes=["62"])

    assert called is False
