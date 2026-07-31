"""A narrow HTTP client for approved public-data endpoints.

The adapters in this package cannot accept arbitrary URLs or headers.  Each
request is made to one audited endpoint, without credentials or redirects, and
the exact request plus the bounded response entity is frozen for provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx

from app.venture.sources.errors import EgressPolicyError, SourceTransportError

HttpMethod = Literal["GET", "POST"]
QueryParam = tuple[str, str | int]

CMS_METADATA_ENDPOINT: Final = (
    "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/4pq5-n9py"
)
CMS_DATA_ENDPOINT: Final = "https://data.cms.gov/provider-data/api/1/datastore/query/4pq5-n9py/0"
USASPENDING_NAICS_ENDPOINT: Final = (
    "https://api.usaspending.gov/api/v2/search/spending_by_category/naics/"
)
BLS_TIMESERIES_ENDPOINT: Final = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_JOLTS_BULK_ENDPOINT: Final = "https://download.bls.gov/pub/time.series/jt/jt.data.1.AllItems"
BLS_ECI_BULK_ENDPOINT: Final = "https://download.bls.gov/pub/time.series/ci/ci.data.1.AllData"
BLS_BED_NAICS_62_ENDPOINT: Final = "https://www.bls.gov/bdm/us_age_naics_62_table7.txt"
BLS_BED_NAICS_54_ENDPOINT: Final = "https://www.bls.gov/bdm/us_age_naics_54_table7.txt"
BLS_BED_NAICS_56_ENDPOINT: Final = "https://www.bls.gov/bdm/us_age_naics_56_table7.txt"
BLS_BED_NAICS_23_ENDPOINT: Final = "https://www.bls.gov/bdm/us_age_naics_23_table7.txt"
BLS_BED_NAICS_51_ENDPOINT: Final = "https://www.bls.gov/bdm/us_age_naics_51_table7.txt"
BLS_BED_NAICS_81_ENDPOINT: Final = "https://www.bls.gov/bdm/us_age_naics_81_table7.txt"
CENSUS_CBP_2022_US_ENDPOINT: Final = (
    "https://www2.census.gov/programs-surveys/cbp/datasets/2022/cbp22us.zip"
)
CENSUS_CBP_2022_COUNTY_ENDPOINT: Final = (
    "https://www2.census.gov/programs-surveys/cbp/datasets/2022/cbp22co.zip"
)
CENSUS_CBP_2023_US_ENDPOINT: Final = (
    "https://www2.census.gov/programs-surveys/cbp/datasets/2023/cbp23us.zip"
)
CENSUS_CBP_2023_COUNTY_ENDPOINT: Final = (
    "https://www2.census.gov/programs-surveys/cbp/datasets/2023/cbp23co.zip"
)
CENSUS_CA_2024_COUNTY_AGE_SEX_ENDPOINT: Final = (
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/"
    "counties/asrh/cc-est2024-agesex-06.csv"
)
CENSUS_2025_COUNTY_POPULATION_ENDPOINT: Final = (
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/"
    "counties/totals/co-est2025-alldata.csv"
)
ECONOMIC_CENSUS_2022_BASIC_54_ENDPOINT: Final = (
    "https://www2.census.gov/programs-surveys/economic-census/data/2022/sector54/EC2254BASIC.zip"
)
ECONOMIC_CENSUS_2022_BASIC_56_ENDPOINT: Final = (
    "https://www2.census.gov/programs-surveys/economic-census/data/2022/sector56/EC2256BASIC.zip"
)
ECONOMIC_CENSUS_2022_BASIC_62_ENDPOINT: Final = (
    "https://www2.census.gov/programs-surveys/economic-census/data/2022/sector62/EC2262BASIC.zip"
)
ECONOMIC_CENSUS_2022_BASIC_81_ENDPOINT: Final = (
    "https://www2.census.gov/programs-surveys/economic-census/data/2022/sector81/EC2281BASIC.zip"
)

_JSON_TYPES: Final = frozenset({"application/json"})
_BULK_TYPES: Final = frozenset({"text/plain", "application/octet-stream"})
_ZIP_TYPES: Final = frozenset({"application/zip"})
_CSV_TYPES: Final = frozenset({"text/csv"})
_MAX_TIMEOUT_SECONDS: Final = 30.0
_MAX_BODY_BYTES: Final = 64 * 1024 * 1024
_CREDENTIAL_KEY: Final = re.compile(
    r"(?:authorization|credential|password|secret|token|api[_-]?key|registrationkey)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class FixedEndpoint:
    """One method/URL/content-type combination approved by code review."""

    name: str
    method: HttpMethod
    url: str
    accepted_content_types: frozenset[str]
    max_body_bytes: int | None = None


CMS_METADATA = FixedEndpoint("cms-provider-metadata", "GET", CMS_METADATA_ENDPOINT, _JSON_TYPES)
CMS_DATA = FixedEndpoint("cms-provider-data", "GET", CMS_DATA_ENDPOINT, _JSON_TYPES)
USASPENDING_NAICS = FixedEndpoint(
    "usaspending-naics-obligations",
    "POST",
    USASPENDING_NAICS_ENDPOINT,
    _JSON_TYPES,
)
BLS_TIMESERIES = FixedEndpoint(
    "bls-public-timeseries",
    "POST",
    BLS_TIMESERIES_ENDPOINT,
    _JSON_TYPES,
)
BLS_JOLTS_BULK = FixedEndpoint(
    "bls-jolts-bulk",
    "GET",
    BLS_JOLTS_BULK_ENDPOINT,
    _BULK_TYPES,
)
BLS_ECI_BULK = FixedEndpoint(
    "bls-eci-bulk",
    "GET",
    BLS_ECI_BULK_ENDPOINT,
    _BULK_TYPES,
)
BLS_BED_NAICS_62 = FixedEndpoint(
    "bls-bed-establishment-survival-naics-62",
    "GET",
    BLS_BED_NAICS_62_ENDPOINT,
    _BULK_TYPES,
)
BLS_BED_NAICS_54 = FixedEndpoint(
    "bls-bed-establishment-survival-naics-54",
    "GET",
    BLS_BED_NAICS_54_ENDPOINT,
    _BULK_TYPES,
)
BLS_BED_NAICS_56 = FixedEndpoint(
    "bls-bed-establishment-survival-naics-56",
    "GET",
    BLS_BED_NAICS_56_ENDPOINT,
    _BULK_TYPES,
)
BLS_BED_NAICS_23 = FixedEndpoint(
    "bls-bed-establishment-survival-naics-23",
    "GET",
    BLS_BED_NAICS_23_ENDPOINT,
    _BULK_TYPES,
)
BLS_BED_NAICS_51 = FixedEndpoint(
    "bls-bed-establishment-survival-naics-51",
    "GET",
    BLS_BED_NAICS_51_ENDPOINT,
    _BULK_TYPES,
)
BLS_BED_NAICS_81 = FixedEndpoint(
    "bls-bed-establishment-survival-naics-81",
    "GET",
    BLS_BED_NAICS_81_ENDPOINT,
    _BULK_TYPES,
    512 * 1024,
)
CENSUS_CBP_2022_US = FixedEndpoint(
    "census-cbp-2022-us",
    "GET",
    CENSUS_CBP_2022_US_ENDPOINT,
    _ZIP_TYPES,
    2 * 1024 * 1024,
)
CENSUS_CBP_2022_COUNTY = FixedEndpoint(
    "census-cbp-2022-county",
    "GET",
    CENSUS_CBP_2022_COUNTY_ENDPOINT,
    _ZIP_TYPES,
    16 * 1024 * 1024,
)
CENSUS_CBP_2023_US = FixedEndpoint(
    "census-cbp-2023-us",
    "GET",
    CENSUS_CBP_2023_US_ENDPOINT,
    _ZIP_TYPES,
    2 * 1024 * 1024,
)
CENSUS_CBP_2023_COUNTY = FixedEndpoint(
    "census-cbp-2023-county",
    "GET",
    CENSUS_CBP_2023_COUNTY_ENDPOINT,
    _ZIP_TYPES,
    16 * 1024 * 1024,
)
CENSUS_CA_2024_COUNTY_AGE_SEX = FixedEndpoint(
    "census-california-2024-county-age-sex",
    "GET",
    CENSUS_CA_2024_COUNTY_AGE_SEX_ENDPOINT,
    _CSV_TYPES,
    512 * 1024,
)
CENSUS_2025_COUNTY_POPULATION = FixedEndpoint(
    "census-2025-county-population",
    "GET",
    CENSUS_2025_COUNTY_POPULATION_ENDPOINT,
    _CSV_TYPES,
    3 * 1024 * 1024,
)
ECONOMIC_CENSUS_2022_BASIC_54 = FixedEndpoint(
    "economic-census-2022-basic-sector-54",
    "GET",
    ECONOMIC_CENSUS_2022_BASIC_54_ENDPOINT,
    _ZIP_TYPES,
    10 * 1024 * 1024,
)
ECONOMIC_CENSUS_2022_BASIC_56 = FixedEndpoint(
    "economic-census-2022-basic-sector-56",
    "GET",
    ECONOMIC_CENSUS_2022_BASIC_56_ENDPOINT,
    _ZIP_TYPES,
    5 * 1024 * 1024,
)
ECONOMIC_CENSUS_2022_BASIC_62 = FixedEndpoint(
    "economic-census-2022-basic-sector-62",
    "GET",
    ECONOMIC_CENSUS_2022_BASIC_62_ENDPOINT,
    _ZIP_TYPES,
    13 * 1024 * 1024,
)
ECONOMIC_CENSUS_2022_BASIC_81 = FixedEndpoint(
    "economic-census-2022-basic-sector-81",
    "GET",
    ECONOMIC_CENSUS_2022_BASIC_81_ENDPOINT,
    _ZIP_TYPES,
    10 * 1024 * 1024,
)

APPROVED_ENDPOINTS: Final = frozenset(
    {
        BLS_BED_NAICS_23,
        BLS_BED_NAICS_51,
        BLS_BED_NAICS_54,
        BLS_BED_NAICS_56,
        BLS_BED_NAICS_62,
        BLS_BED_NAICS_81,
        CMS_METADATA,
        CMS_DATA,
        USASPENDING_NAICS,
        BLS_TIMESERIES,
        BLS_JOLTS_BULK,
        BLS_ECI_BULK,
        CENSUS_CBP_2022_US,
        CENSUS_CBP_2022_COUNTY,
        CENSUS_CBP_2023_US,
        CENSUS_CBP_2023_COUNTY,
        CENSUS_CA_2024_COUNTY_AGE_SEX,
        CENSUS_2025_COUNTY_POPULATION,
        ECONOMIC_CENSUS_2022_BASIC_54,
        ECONOMIC_CENSUS_2022_BASIC_56,
        ECONOMIC_CENSUS_2022_BASIC_62,
        ECONOMIC_CENSUS_2022_BASIC_81,
    }
)
APPROVED_HOSTS: Final = frozenset(
    {
        "data.cms.gov",
        "api.usaspending.gov",
        "api.bls.gov",
        "download.bls.gov",
        "www.bls.gov",
        "www2.census.gov",
    }
)


@dataclass(frozen=True, slots=True)
class RawCapture:
    """The exact request and bounded response bytes consumed by a parser."""

    endpoint_name: str
    request_method: HttpMethod
    request_url: str
    request_body: bytes | None
    response_status: int
    response_headers: tuple[tuple[str, str], ...]
    fetched_at: datetime
    body_sha256: str
    raw_bytes: bytes


def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    """Serialize a POST query deterministically so it can be reproduced exactly."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class SafeHttpClient:
    """Execute only fixed, anonymous, bounded public-data requests."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 20.0,
        max_body_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if not 0 < timeout_seconds <= _MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeout_seconds must be in (0, {_MAX_TIMEOUT_SECONDS:g}]")
        if not 0 < max_body_bytes <= _MAX_BODY_BYTES:
            raise ValueError(f"max_body_bytes must be in [1, {_MAX_BODY_BYTES}]")
        self._transport = transport
        self._timeout_seconds = timeout_seconds
        self._max_body_bytes = max_body_bytes

    def request(
        self,
        endpoint: FixedEndpoint,
        *,
        params: Sequence[QueryParam] = (),
        json_body: Mapping[str, object] | None = None,
    ) -> RawCapture:
        """Fetch an approved endpoint and freeze the exact request/response."""
        self._validate_endpoint(endpoint)
        self._reject_credentials(params=params, json_body=json_body)
        if endpoint.method == "GET" and json_body is not None:
            raise EgressPolicyError("GET endpoints cannot send a request body")
        if endpoint.method == "POST" and json_body is None:
            raise EgressPolicyError("POST endpoints require an explicit JSON query")

        body = canonical_json_bytes(json_body) if json_body is not None else None
        headers = {
            "Accept": ", ".join(sorted(endpoint.accepted_content_types)),
            "Accept-Encoding": "identity",
            "User-Agent": "opportunity-lab-public-data/0.1",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"

        try:
            with (
                httpx.Client(
                    transport=self._transport,
                    timeout=httpx.Timeout(self._timeout_seconds),
                    follow_redirects=False,
                    headers=headers,
                ) as client,
                client.stream(
                    endpoint.method,
                    endpoint.url,
                    params=list(params),
                    content=body,
                ) as response,
            ):
                return self._capture_response(endpoint, response, body)
        except EgressPolicyError:
            raise
        except httpx.HTTPError as exc:
            raise SourceTransportError(
                f"{endpoint.name} request failed: {type(exc).__name__}"
            ) from exc

    def _capture_response(
        self,
        endpoint: FixedEndpoint,
        response: httpx.Response,
        request_body: bytes | None,
    ) -> RawCapture:
        body_cap = self._max_body_bytes
        if endpoint.max_body_bytes is not None:
            if not 0 < endpoint.max_body_bytes <= _MAX_BODY_BYTES:
                raise EgressPolicyError(f"{endpoint.name} has an invalid response body cap")
            body_cap = min(body_cap, endpoint.max_body_bytes)
        if 300 <= response.status_code < 400:
            raise EgressPolicyError(
                f"{endpoint.name} returned a redirect; redirects are never followed"
            )
        if not 200 <= response.status_code < 300:
            raise SourceTransportError(f"{endpoint.name} returned HTTP {response.status_code}")

        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in endpoint.accepted_content_types:
            expected = ", ".join(sorted(endpoint.accepted_content_types))
            raise EgressPolicyError(
                f"{endpoint.name} returned content type {content_type or '<missing>'}; "
                f"expected {expected}"
            )

        declared_length = response.headers.get("content-length")
        if declared_length is not None:
            try:
                length = int(declared_length)
            except ValueError as exc:
                raise EgressPolicyError(
                    f"{endpoint.name} returned an invalid Content-Length"
                ) from exc
            if length < 0 or length > body_cap:
                raise EgressPolicyError(f"{endpoint.name} response exceeds the {body_cap}-byte cap")

        chunks: list[bytes] = []
        received = 0
        for chunk in response.iter_bytes():
            received += len(chunk)
            if received > body_cap:
                raise EgressPolicyError(f"{endpoint.name} response exceeds the {body_cap}-byte cap")
            chunks.append(chunk)
        raw_bytes = b"".join(chunks)
        return RawCapture(
            endpoint_name=endpoint.name,
            request_method=endpoint.method,
            request_url=str(response.request.url),
            request_body=request_body,
            response_status=response.status_code,
            response_headers=tuple(response.headers.multi_items()),
            fetched_at=datetime.now(UTC),
            body_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            raw_bytes=raw_bytes,
        )

    @staticmethod
    def _validate_endpoint(endpoint: FixedEndpoint) -> None:
        parsed = urlsplit(endpoint.url)
        if parsed.scheme != "https":
            raise EgressPolicyError("source endpoints must use HTTPS")
        if parsed.username is not None or parsed.password is not None:
            raise EgressPolicyError("source endpoint URLs cannot contain credentials")
        if parsed.hostname not in APPROVED_HOSTS:
            raise EgressPolicyError(f"source host {parsed.hostname!r} is not approved")
        if parsed.port not in (None, 443):
            raise EgressPolicyError("source endpoints may use only the default HTTPS port")
        if parsed.query or parsed.fragment:
            raise EgressPolicyError(
                "fixed endpoint constants cannot contain query text or fragments"
            )
        if endpoint not in APPROVED_ENDPOINTS:
            raise EgressPolicyError("source method/path/content types are not an approved endpoint")

    @classmethod
    def _reject_credentials(
        cls,
        *,
        params: Sequence[QueryParam],
        json_body: Mapping[str, object] | None,
    ) -> None:
        if any(_CREDENTIAL_KEY.search(name) for name, _ in params):
            raise EgressPolicyError("source requests cannot contain credential parameters")
        if json_body is not None and cls._contains_credential_key(json_body):
            raise EgressPolicyError("source requests cannot contain credential fields")

    @classmethod
    def _contains_credential_key(cls, value: object) -> bool:
        if isinstance(value, Mapping):
            return any(
                _CREDENTIAL_KEY.search(str(key)) is not None or cls._contains_credential_key(item)
                for key, item in value.items()
            )
        if isinstance(value, list | tuple):
            return any(cls._contains_credential_key(item) for item in value)
        return False


__all__ = [
    "APPROVED_ENDPOINTS",
    "APPROVED_HOSTS",
    "BLS_BED_NAICS_23",
    "BLS_BED_NAICS_23_ENDPOINT",
    "BLS_BED_NAICS_51",
    "BLS_BED_NAICS_51_ENDPOINT",
    "BLS_BED_NAICS_54",
    "BLS_BED_NAICS_54_ENDPOINT",
    "BLS_BED_NAICS_56",
    "BLS_BED_NAICS_56_ENDPOINT",
    "BLS_BED_NAICS_62",
    "BLS_BED_NAICS_62_ENDPOINT",
    "BLS_BED_NAICS_81",
    "BLS_BED_NAICS_81_ENDPOINT",
    "BLS_ECI_BULK",
    "BLS_ECI_BULK_ENDPOINT",
    "BLS_JOLTS_BULK",
    "BLS_JOLTS_BULK_ENDPOINT",
    "BLS_TIMESERIES",
    "BLS_TIMESERIES_ENDPOINT",
    "CENSUS_2025_COUNTY_POPULATION",
    "CENSUS_2025_COUNTY_POPULATION_ENDPOINT",
    "CENSUS_CA_2024_COUNTY_AGE_SEX",
    "CENSUS_CA_2024_COUNTY_AGE_SEX_ENDPOINT",
    "CENSUS_CBP_2022_COUNTY",
    "CENSUS_CBP_2022_COUNTY_ENDPOINT",
    "CENSUS_CBP_2022_US",
    "CENSUS_CBP_2022_US_ENDPOINT",
    "CENSUS_CBP_2023_COUNTY",
    "CENSUS_CBP_2023_COUNTY_ENDPOINT",
    "CENSUS_CBP_2023_US",
    "CENSUS_CBP_2023_US_ENDPOINT",
    "CMS_DATA",
    "CMS_DATA_ENDPOINT",
    "CMS_METADATA",
    "CMS_METADATA_ENDPOINT",
    "ECONOMIC_CENSUS_2022_BASIC_54",
    "ECONOMIC_CENSUS_2022_BASIC_54_ENDPOINT",
    "ECONOMIC_CENSUS_2022_BASIC_56",
    "ECONOMIC_CENSUS_2022_BASIC_56_ENDPOINT",
    "ECONOMIC_CENSUS_2022_BASIC_62",
    "ECONOMIC_CENSUS_2022_BASIC_62_ENDPOINT",
    "ECONOMIC_CENSUS_2022_BASIC_81",
    "ECONOMIC_CENSUS_2022_BASIC_81_ENDPOINT",
    "USASPENDING_NAICS",
    "USASPENDING_NAICS_ENDPOINT",
    "FixedEndpoint",
    "QueryParam",
    "RawCapture",
    "SafeHttpClient",
    "canonical_json_bytes",
]
