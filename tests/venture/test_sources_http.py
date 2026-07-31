"""The public-data HTTP boundary is fixed, anonymous, and bounded."""

import hashlib

import httpx
import pytest

from app.venture.sources.errors import EgressPolicyError
from app.venture.sources.http import BLS_TIMESERIES, CMS_METADATA, FixedEndpoint, SafeHttpClient


def test_capture_freezes_exact_query_response_and_hash() -> None:
    raw = b'{"identifier":"4pq5-n9py"}'

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["probe"] == "one"
        assert request.headers["accept-encoding"] == "identity"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            content=raw,
            headers={"Content-Type": "application/json; charset=utf-8", "ETag": '"v1"'},
        )

    client = SafeHttpClient(transport=httpx.MockTransport(handler))
    capture = client.request(CMS_METADATA, params=[("probe", "one")])

    assert capture.request_url.endswith("?probe=one")
    assert capture.request_body is None
    assert capture.raw_bytes == raw
    assert capture.body_sha256 == hashlib.sha256(raw).hexdigest()
    assert ("etag", '"v1"') in capture.response_headers
    assert capture.fetched_at.tzinfo is not None


def test_private_host_is_rejected_before_transport() -> None:
    private = FixedEndpoint(
        name="private",
        method="GET",
        url="https://127.0.0.1/latest/meta-data",
        accepted_content_types=frozenset({"application/json"}),
    )
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = SafeHttpClient(transport=httpx.MockTransport(handler))

    with pytest.raises(EgressPolicyError, match="not approved"):
        client.request(private)

    assert called is False


def test_url_credentials_are_rejected() -> None:
    credentialed = FixedEndpoint(
        name="credentialed",
        method="GET",
        url=(
            "https://user:secret@data.cms.gov/provider-data/api/1/metastore/"
            "schemas/dataset/items/4pq5-n9py"
        ),
        accepted_content_types=frozenset({"application/json"}),
    )

    with pytest.raises(EgressPolicyError, match="credentials"):
        SafeHttpClient().request(credentialed)


@pytest.mark.parametrize(
    ("params", "json_body"),
    [
        ([("api_key", "secret")], None),
        ([], {"registrationkey": "secret", "seriesid": ["SERIES"]}),
        ([], {"filters": {"access_token": "secret"}}),
    ],
)
def test_credential_fields_are_rejected(
    params: list[tuple[str, str]],
    json_body: dict[str, object] | None,
) -> None:
    endpoint = CMS_METADATA if json_body is None else BLS_TIMESERIES

    with pytest.raises(EgressPolicyError, match="credential"):
        SafeHttpClient().request(endpoint, params=params, json_body=json_body)


def test_redirect_to_private_address_is_not_followed() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            302,
            headers={"Location": "http://169.254.169.254/latest/meta-data"},
        )

    client = SafeHttpClient(transport=httpx.MockTransport(handler))

    with pytest.raises(EgressPolicyError, match="redirect"):
        client.request(CMS_METADATA)

    assert requests == [CMS_METADATA.url]


def test_declared_oversize_body_is_rejected() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b'{"too":"large"}',
            headers={"Content-Type": "application/json", "Content-Length": "15"},
        )
    )
    client = SafeHttpClient(transport=transport, max_body_bytes=10)

    with pytest.raises(EgressPolicyError, match="10-byte cap"):
        client.request(CMS_METADATA)


def test_actual_oversize_body_is_rejected_without_content_length() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"01234567890",
            headers={"Content-Type": "application/json"},
        )
    )
    client = SafeHttpClient(transport=transport, max_body_bytes=10)

    with pytest.raises(EgressPolicyError, match="10-byte cap"):
        client.request(CMS_METADATA)


@pytest.mark.parametrize("content_type", ["text/html", "", "application/problem+json"])
def test_unapproved_or_missing_content_type_is_rejected(content_type: str) -> None:
    headers = {"Content-Type": content_type} if content_type else {}
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"{}", headers=headers)
    )

    with pytest.raises(EgressPolicyError, match="content type"):
        SafeHttpClient(transport=transport).request(CMS_METADATA)
