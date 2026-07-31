"""USAspending NAICS contract-obligation adapter tests."""

import json
from datetime import date
from decimal import Decimal

import httpx
import pytest

from app.venture.sources.errors import SourceParseError
from app.venture.sources.http import SafeHttpClient
from app.venture.sources.usaspending import CONTRACT_AWARD_TYPE_CODES, UsaSpendingSource


def _response(
    *,
    results: list[dict[str, object]] | None = None,
    has_next: bool = True,
) -> dict[str, object]:
    return {
        "category": "naics",
        "spending_level": "transactions",
        "limit": 2,
        "page_metadata": {
            "page": 1,
            "next": 2 if has_next else None,
            "previous": None,
            "hasNext": has_next,
            "hasPrevious": False,
        },
        "results": results
        if results is not None
        else [
            {
                "amount": 39189707496.79,
                "code": "236220",
                "id": None,
                "name": "Commercial and Institutional Building Construction",
                "total_outlays": None,
                "year_retired": None,
            },
            {
                "amount": 35020094066.6,
                "code": "336411",
                "id": None,
                "name": "Aircraft Manufacturing",
                "total_outlays": 100.25,
                "year_retired": 2017,
            },
        ],
        "messages": ["Official API warning."],
    }


def _source(handler: httpx.MockTransport) -> UsaSpendingSource:
    return UsaSpendingSource(SafeHttpClient(transport=handler))


def test_top_naics_uses_exact_contract_action_date_query() -> None:
    seen_query: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v2/search/spending_by_category/naics/"
        seen_query.update(json.loads(request.content))
        return httpx.Response(200, json=_response())

    source = _source(httpx.MockTransport(handler))
    result = source.fetch_top_naics(
        start_date=date(2025, 10, 1),
        end_date=date(2026, 6, 30),
        limit=2,
    )

    filters = seen_query["filters"]
    assert isinstance(filters, dict)
    assert filters["award_type_codes"] == list(CONTRACT_AWARD_TYPE_CODES)
    assert filters["time_period"] == [{"end_date": "2026-06-30", "start_date": "2025-10-01"}]
    assert seen_query["spending_level"] == "transactions"
    assert result.capture.request_body is not None
    assert json.loads(result.capture.request_body) == seen_query


def test_obligation_records_preserve_decimal_values_and_paging() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=_response()))
    result = _source(transport).fetch_top_naics(
        start_date=date(2025, 10, 1),
        end_date=date(2026, 6, 30),
        limit=2,
    )

    assert result.next_page == 2
    assert result.records[0].obligations_usd == Decimal("39189707496.79")
    assert result.records[0].total_outlays_usd is None
    assert result.records[1].total_outlays_usd == Decimal("100.25")
    assert result.records[1].year_retired == 2017
    assert result.messages == ("Official API warning.",)


def test_missing_optional_values_remain_missing() -> None:
    row: dict[str, object] = {
        "amount": "12.00",
        "code": "541511",
        "name": "Custom Computer Programming Services",
        "total_outlays": None,
        "year_retired": None,
    }
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=_response(results=[row], has_next=False))
    )

    record = (
        _source(transport)
        .fetch_top_naics(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 31),
        )
        .records[0]
    )

    assert record.total_outlays_usd is None
    assert record.year_retired is None


def test_missing_required_obligation_is_rejected() -> None:
    row: dict[str, object] = {
        "amount": None,
        "code": "541511",
        "name": "Programming",
    }
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=_response(results=[row], has_next=False))
    )

    with pytest.raises(SourceParseError, match="amount is required"):
        _source(transport).fetch_top_naics(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 31),
        )


def test_unsorted_top_results_are_rejected() -> None:
    rows = [
        {"amount": 1, "code": "111111", "name": "First"},
        {"amount": 2, "code": "222222", "name": "Second"},
    ]
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=_response(results=rows, has_next=False))
    )

    with pytest.raises(SourceParseError, match="not ordered"):
        _source(transport).fetch_top_naics(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 31),
        )


def test_invalid_date_period_is_rejected_before_transport() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=_response())

    with pytest.raises(ValueError, match="must not precede"):
        _source(httpx.MockTransport(handler)).fetch_top_naics(
            start_date=date(2026, 2, 1),
            end_date=date(2026, 1, 1),
        )

    assert called is False
