"""CMS Provider Data metadata and non-live capacity observations."""

from decimal import Decimal

import httpx
import pytest

from app.venture.sources.cms import CAPACITY_MEASUREMENT_NOTE, CmsProviderSource
from app.venture.sources.errors import SourceParseError
from app.venture.sources.http import SafeHttpClient


def _metadata_payload() -> dict[str, object]:
    return {
        "identifier": "4pq5-n9py",
        "title": "Provider Information",
        "description": "General information on currently active nursing homes.",
        "issued": "2026-02-01",
        "modified": "2026-07-01",
        "released": "2026-07-29",
        "nextUpdateDate": "2026-08-26",
        "distribution": [
            {
                "downloadURL": "https://data.cms.gov/provider-data/provider.csv",
                "mediaType": "text/csv",
                "describedBy": "https://data.cms.gov/provider-data/dictionary.pdf",
            }
        ],
    }


def _provider_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "cms_certification_number_ccn": "055555",
        "provider_name": "Example Nursing Home",
        "state": "CA",
        "citytown": "Sacramento",
        "zip_code": "95814",
        "countyparish": "Sacramento",
        "number_of_certified_beds": "57",
        "average_number_of_residents_per_day": "51.6",
        "average_number_of_residents_per_day_footnote": "",
        "processing_date": "2026-07-01",
    }
    row.update(changes)
    return row


def _source(handler: httpx.MockTransport) -> CmsProviderSource:
    return CmsProviderSource(SafeHttpClient(transport=handler))


def test_metadata_includes_version_dates_and_distribution() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=_metadata_payload()))

    result = _source(transport).fetch_metadata()

    assert result.metadata.identifier == "4pq5-n9py"
    assert result.metadata.released is not None
    assert result.metadata.released.isoformat() == "2026-07-29"
    assert result.metadata.next_update_date is not None
    assert result.metadata.distributions[0].media_type == "text/csv"
    assert result.capture.raw_bytes


def test_state_page_records_exact_query_and_non_live_capacity_semantics() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        assert params["limit"] == "2"
        assert params["offset"] == "0"
        assert params["schema"] == "false"
        assert params["conditions[0][property]"] == "state"
        assert params["conditions[0][value]"] == "CA"
        assert params["conditions[0][operator]"] == "="
        return httpx.Response(
            200,
            json={"count": 3, "results": [_provider_row(), _provider_row()]},
        )

    page = _source(httpx.MockTransport(handler)).fetch_page(state="ca", limit=2)

    assert page.state == "CA"
    assert page.total_count == 3
    assert page.next_offset == 2
    record = page.records[0]
    assert record.certified_beds == 57
    assert record.average_residents_per_day == Decimal("51.6")
    assert record.average_residents_per_certified_bed == Decimal("51.6") / Decimal(57)
    assert record.measurement_note == CAPACITY_MEASUREMENT_NOTE
    assert "not licensed-bed inventory" in record.measurement_note
    assert "live availability" in record.measurement_note
    assert "conditions%5B0%5D%5Bvalue%5D=CA" in page.capture.request_url


def test_national_page_has_no_state_condition_and_stops_at_end() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert not any(key.startswith("conditions") for key in request.url.params)
        return httpx.Response(200, json={"count": 1, "results": [_provider_row()]})

    page = _source(httpx.MockTransport(handler)).fetch_page(limit=10)

    assert page.state is None
    assert page.next_offset is None
    assert len(page.records) == 1


def test_missing_capacity_values_are_preserved_as_missing_not_zero() -> None:
    row = _provider_row(
        number_of_certified_beds="",
        average_number_of_residents_per_day=None,
        processing_date="",
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"count": 1, "results": [row]})
    )

    record = _source(transport).fetch_page().records[0]

    assert record.certified_beds is None
    assert record.average_residents_per_day is None
    assert record.average_residents_per_certified_bed is None
    assert record.processing_date is None


def test_malformed_capacity_value_is_rejected() -> None:
    row = _provider_row(number_of_certified_beds="fifty-seven")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"count": 1, "results": [row]})
    )

    with pytest.raises(SourceParseError, match="number_of_certified_beds"):
        _source(transport).fetch_page()


def test_state_filtered_response_cannot_smuggle_another_state() -> None:
    row = _provider_row(state="NV")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"count": 1, "results": [row]})
    )

    with pytest.raises(SourceParseError, match="expected 'CA'"):
        _source(transport).fetch_page(state="CA")


def test_malformed_json_is_a_source_parse_error() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"{broken",
            headers={"Content-Type": "application/json"},
        )
    )

    with pytest.raises(SourceParseError, match="malformed JSON"):
        _source(transport).fetch_page()
