"""BLS anonymous API and bulk parser tests."""

import json
from decimal import Decimal

import httpx
import pytest

from app.venture.sources.bls import (
    BED_ESTABLISHMENT_SURVIVAL_NOTE,
    ECI_PRIVATE_INDUSTRY_COMPENSATION_12_MONTH,
    JOLTS_TOTAL_NONFARM_JOB_OPENINGS,
    BlsSource,
    parse_bed_establishment_survival,
    parse_bls_bulk,
)
from app.venture.sources.errors import SourceParseError
from app.venture.sources.http import SafeHttpClient


def _api_response(
    *,
    series_id: str = JOLTS_TOTAL_NONFARM_JOB_OPENINGS,
    value: str = "7594",
) -> dict[str, object]:
    return {
        "status": "REQUEST_SUCCEEDED",
        "responseTime": 10,
        "message": [],
        "Results": {
            "series": [
                {
                    "seriesID": series_id,
                    "data": [
                        {
                            "year": "2026",
                            "period": "M05",
                            "periodName": "May",
                            "latest": "true",
                            "value": value,
                            "footnotes": [{"code": "P", "text": "preliminary"}],
                        }
                    ],
                }
            ]
        },
    }


def _source(handler: httpx.MockTransport) -> BlsSource:
    return BlsSource(SafeHttpClient(transport=handler))


def _bed_table(*, five_year_rate: str = "50.0") -> bytes:
    return (
        "Table 7. Survival of private sector establishments by opening year\n"
        "\n"
        "Health Care and Social Assistance\n"
        "\n"
        "Annual openings\n"
        "Year ended: March 2000\n"
        "\n"
        " March 2000 1,000 10,000 100.0 _ 10.0\n"
        " March 2001 800 9,000 80.0 80.0 11.2\n"
        " March 2002 700 8,000 70.0 87.5 11.4\n"
        " March 2003 650 7,500 65.0 92.9 11.5\n"
        " March 2004 550 6,500 55.0 84.6 11.8\n"
        f" March 2005 500 6,000 {five_year_rate} 90.9 12.0\n"
        "\n"
        "Annual openings\n"
        "Year ended: March 2003\n"
        "\n"
        " March 2003 900 9,500 100.0 _ 10.6\n"
        " March 2004 720 8,900 80.0 80.0 12.4\n"
        " March 2005 650 8,100 72.2 90.3 12.5\n"
    ).encode()


def test_job_openings_helper_records_exact_anonymous_query() -> None:
    seen_query: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_query.update(json.loads(request.content))
        assert "registrationkey" not in seen_query
        return httpx.Response(200, json=_api_response())

    result = _source(httpx.MockTransport(handler)).fetch_job_openings(
        start_year=2025,
        end_year=2026,
    )

    assert seen_query == {
        "endyear": "2026",
        "seriesid": [JOLTS_TOTAL_NONFARM_JOB_OPENINGS],
        "startyear": "2025",
    }
    point = result.series[0].points[0]
    assert point.series_id == JOLTS_TOTAL_NONFARM_JOB_OPENINGS
    assert point.value == Decimal("7594")
    assert point.latest is True
    assert point.footnotes[0].code == "P"
    assert result.capture.request_body is not None
    assert json.loads(result.capture.request_body) == seen_query


def test_labor_cost_helper_uses_verified_eci_series() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        query = json.loads(request.content)
        assert query["seriesid"] == [ECI_PRIVATE_INDUSTRY_COMPENSATION_12_MONTH]
        return httpx.Response(
            200,
            json=_api_response(
                series_id=ECI_PRIVATE_INDUSTRY_COMPENSATION_12_MONTH,
                value="3.4",
            ),
        )

    result = _source(httpx.MockTransport(handler)).fetch_labor_cost_growth(
        start_year=2025,
        end_year=2026,
    )

    assert result.series[0].points[0].value == Decimal("3.4")


def test_missing_api_value_is_preserved_as_missing() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=_api_response(value=""))
    )

    point = (
        _source(transport)
        .fetch_job_openings(
            start_year=2026,
            end_year=2026,
        )
        .series[0]
        .points[0]
    )

    assert point.value is None


def test_failed_bls_status_is_rejected() -> None:
    payload = {
        "status": "REQUEST_FAILED",
        "message": ["Invalid Series for Series JUNK"],
        "Results": {},
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))

    with pytest.raises(SourceParseError, match="REQUEST_SUCCEEDED"):
        _source(transport).fetch_series(["JUNK"], start_year=2026, end_year=2026)


def test_anonymous_request_limits_are_enforced_before_transport() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=_api_response())

    with pytest.raises(ValueError, match="at most 10 years"):
        _source(httpx.MockTransport(handler)).fetch_job_openings(
            start_year=2016,
            end_year=2026,
        )

    assert called is False


def test_bulk_parser_filters_series_and_preserves_missing_values() -> None:
    raw = (
        "series_id\tyear\tperiod\tvalue\tfootnote_codes\n"
        f"{JOLTS_TOTAL_NONFARM_JOB_OPENINGS}\t2026\tM05\t7594\tP\n"
        f"{JOLTS_TOTAL_NONFARM_JOB_OPENINGS}\t2026\tM06\t\t\n"
        "JTS100000000000000JOL\t2026\tM05\t100\t\n"
    ).encode()

    points = parse_bls_bulk(raw, series_ids=[JOLTS_TOTAL_NONFARM_JOB_OPENINGS])

    assert len(points) == 2
    assert points[0].value == Decimal("7594")
    assert points[0].footnotes[0].code == "P"
    assert points[1].value is None


def test_fixed_bulk_fetch_returns_capture_and_selected_rows() -> None:
    raw = (
        "series_id\tyear\tperiod\tvalue\tfootnote_codes\n"
        f"{JOLTS_TOTAL_NONFARM_JOB_OPENINGS}\t2026\tM05\t7594\tP\n"
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/pub/time.series/jt/jt.data.1.AllItems"
        return httpx.Response(200, content=raw, headers={"Content-Type": "text/plain"})

    result = _source(httpx.MockTransport(handler)).fetch_jolts_bulk()

    assert result.capture.request_url.endswith("/jt.data.1.AllItems")
    assert result.capture.raw_bytes == raw
    assert result.points[0].value == Decimal("7594")


def test_bed_parser_returns_one_and_five_year_establishment_survival() -> None:
    cohorts = parse_bed_establishment_survival(_bed_table(), naics_code="62")

    assert len(cohorts) == 1
    cohort = cohorts[0]
    assert cohort.naics_code == "62"
    assert cohort.cohort_year == 2000
    assert cohort.one_year_establishment_survival_percent == Decimal("80.0")
    # This proves the parser selected "survival since birth" (50.0), not the
    # adjacent previous-year survival column (90.9).
    assert cohort.five_year_establishment_survival_percent == Decimal("50.0")
    assert cohort.measurement_note == BED_ESTABLISHMENT_SURVIVAL_NOTE
    assert "establishment survival, not firm survival" in cohort.measurement_note


def test_bed_source_uses_only_the_six_approved_naics_endpoints() -> None:
    observed_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_paths.append(request.url.path)
        return httpx.Response(
            200,
            content=_bed_table(),
            headers={"Content-Type": "text/plain"},
        )

    source = _source(httpx.MockTransport(handler))
    for naics_code in ("62", "54", "56", "23", "51", "81"):
        result = source.fetch_establishment_survival(naics_code)
        assert result.naics_code == naics_code
        assert result.capture.request_url.endswith(f"/bdm/us_age_naics_{naics_code}_table7.txt")

    assert set(observed_paths) == {
        "/bdm/us_age_naics_62_table7.txt",
        "/bdm/us_age_naics_54_table7.txt",
        "/bdm/us_age_naics_56_table7.txt",
        "/bdm/us_age_naics_23_table7.txt",
        "/bdm/us_age_naics_51_table7.txt",
        "/bdm/us_age_naics_81_table7.txt",
    }


def test_unsupported_bed_naics_is_rejected_before_transport() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, content=_bed_table())

    with pytest.raises(ValueError, match="23, 51, 54, 56, 62, 81"):
        _source(httpx.MockTransport(handler)).fetch_establishment_survival("99")

    assert called is False


def test_mature_bed_cohort_missing_five_year_observation_is_rejected() -> None:
    malformed = _bed_table().replace(
        b" March 2005 500 6,000 50.0 90.9 12.0\n",
        b" March 2006 450 5,500 45.0 90.0 12.2\n",
    )

    with pytest.raises(SourceParseError, match="mature cohort 2000"):
        parse_bed_establishment_survival(malformed, naics_code="62")


def test_bed_rate_outside_percent_range_is_rejected() -> None:
    with pytest.raises(SourceParseError, match="outside 0-100"):
        parse_bed_establishment_survival(
            _bed_table(five_year_rate="101.0"),
            naics_code="62",
        )


def test_non_table_bed_response_is_rejected() -> None:
    with pytest.raises(SourceParseError, match="not establishment-survival Table 7"):
        parse_bed_establishment_survival(b"<html>Access Denied</html>", naics_code="62")


@pytest.mark.parametrize(
    "raw",
    [
        b"not\tthe\tright\tcolumns\n",
        (b"series_id\tyear\tperiod\tvalue\tfootnote_codes\nJTS000000000000000JOL\t2026\n"),
        (
            b"series_id\tyear\tperiod\tvalue\tfootnote_codes\n"
            b"JTS000000000000000JOL\ttwenty\tM01\t1\t\n"
        ),
    ],
)
def test_malformed_bulk_files_are_rejected(raw: bytes) -> None:
    with pytest.raises(SourceParseError):
        parse_bls_bulk(raw)
