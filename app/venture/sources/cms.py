"""CMS Provider Data nursing-home capacity observations.

Dataset ``4pq5-n9py`` reports CMS-certified beds and an average resident census.
Neither measure is licensed-bed inventory or current availability.  The ratio
below is therefore an average historical load signal, never a vacancy claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Final

from app.venture.sources.errors import SourceParseError
from app.venture.sources.http import CMS_DATA, CMS_METADATA, RawCapture, SafeHttpClient
from app.venture.sources.parsing import (
    JsonObject,
    json_object,
    object_list,
    optional_date,
    optional_decimal,
    optional_int,
    optional_text,
    required_text,
)

CMS_DATASET_ID: Final = "4pq5-n9py"
CAPACITY_MEASUREMENT_NOTE: Final = (
    "CMS-certified beds and average residents per day; this is not licensed-bed "
    "inventory, a vacancy count, or live availability."
)
_STATE: Final = re.compile(r"^[A-Z]{2}$")
_MAX_PAGE_SIZE: Final = 500


@dataclass(frozen=True, slots=True)
class CmsDistribution:
    """One downloadable distribution advertised by CMS metadata."""

    download_url: str
    media_type: str | None
    data_dictionary_url: str | None


@dataclass(frozen=True, slots=True)
class CmsDatasetMetadata:
    """Version and distribution metadata for nursing-home Provider Information."""

    identifier: str
    title: str
    description: str
    issued: date | None
    modified: date | None
    released: date | None
    next_update_date: date | None
    distributions: tuple[CmsDistribution, ...]


@dataclass(frozen=True, slots=True)
class CmsMetadataResult:
    """Frozen metadata response and its normalized fields."""

    capture: RawCapture
    metadata: CmsDatasetMetadata


@dataclass(frozen=True, slots=True)
class CmsNursingHomeRecord:
    """One facility's non-live CMS capacity observations."""

    cms_certification_number: str
    provider_name: str
    state: str
    city: str | None
    zip_code: str | None
    county: str | None
    certified_beds: int | None
    average_residents_per_day: Decimal | None
    average_residents_per_certified_bed: Decimal | None
    average_residents_footnote: str | None
    processing_date: date | None
    measurement_note: str = CAPACITY_MEASUREMENT_NOTE


@dataclass(frozen=True, slots=True)
class CmsProviderPage:
    """One state or national page plus its frozen source response."""

    capture: RawCapture
    state: str | None
    offset: int
    limit: int
    total_count: int
    next_offset: int | None
    records: tuple[CmsNursingHomeRecord, ...]


class CmsProviderSource:
    """Fetch metadata and deterministic pages from CMS dataset ``4pq5-n9py``."""

    def __init__(self, client: SafeHttpClient | None = None) -> None:
        self._client = client or SafeHttpClient()

    def fetch_metadata(self) -> CmsMetadataResult:
        """Fetch the dataset's official version/distribution metadata."""
        capture = self._client.request(CMS_METADATA)
        payload = json_object(capture.raw_bytes, source=CMS_METADATA.name)
        identifier = required_text(payload.get("identifier"), field="identifier")
        if identifier != CMS_DATASET_ID:
            raise SourceParseError(
                f"CMS metadata identifier {identifier!r} does not match {CMS_DATASET_ID}"
            )
        distributions = tuple(
            self._parse_distribution(row)
            for row in object_list(payload.get("distribution"), field="distribution")
        )
        metadata = CmsDatasetMetadata(
            identifier=identifier,
            title=required_text(payload.get("title"), field="title"),
            description=required_text(payload.get("description"), field="description"),
            issued=optional_date(payload.get("issued"), field="issued"),
            modified=optional_date(payload.get("modified"), field="modified"),
            released=optional_date(payload.get("released"), field="released"),
            next_update_date=optional_date(
                payload.get("nextUpdateDate"),
                field="nextUpdateDate",
            ),
            distributions=distributions,
        )
        return CmsMetadataResult(capture=capture, metadata=metadata)

    def fetch_page(
        self,
        *,
        state: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> CmsProviderPage:
        """Fetch one national page or one page filtered by two-letter state code."""
        normalized_state = self._normalize_state(state)
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")

        params: list[tuple[str, str | int]] = [
            ("limit", limit),
            ("offset", offset),
            ("count", "true"),
            ("results", "true"),
            ("schema", "false"),
            ("keys", "false"),
        ]
        if normalized_state is not None:
            params.extend(
                [
                    ("conditions[0][property]", "state"),
                    ("conditions[0][value]", normalized_state),
                    ("conditions[0][operator]", "="),
                ]
            )
        capture = self._client.request(CMS_DATA, params=params)
        payload = json_object(capture.raw_bytes, source=CMS_DATA.name)
        total_count = optional_int(payload.get("count"), field="count")
        if total_count is None or total_count < 0:
            raise SourceParseError("count must be a non-negative integer")
        records = tuple(
            self._parse_provider(row, requested_state=normalized_state)
            for row in object_list(payload.get("results"), field="results")
        )
        consumed = offset + len(records)
        next_offset = consumed if records and consumed < total_count else None
        return CmsProviderPage(
            capture=capture,
            state=normalized_state,
            offset=offset,
            limit=limit,
            total_count=total_count,
            next_offset=next_offset,
            records=records,
        )

    @staticmethod
    def _normalize_state(state: str | None) -> str | None:
        if state is None:
            return None
        normalized = state.strip().upper()
        if not _STATE.fullmatch(normalized):
            raise ValueError("state must be a two-letter postal abbreviation")
        return normalized

    @staticmethod
    def _parse_distribution(row: JsonObject) -> CmsDistribution:
        return CmsDistribution(
            download_url=required_text(row.get("downloadURL"), field="distribution.downloadURL"),
            media_type=optional_text(row.get("mediaType"), field="distribution.mediaType"),
            data_dictionary_url=optional_text(
                row.get("describedBy"),
                field="distribution.describedBy",
            ),
        )

    @staticmethod
    def _parse_provider(
        row: JsonObject,
        *,
        requested_state: str | None,
    ) -> CmsNursingHomeRecord:
        state = required_text(row.get("state"), field="results.state").upper()
        if not _STATE.fullmatch(state):
            raise SourceParseError(f"results.state contains invalid state code {state!r}")
        if requested_state is not None and state != requested_state:
            raise SourceParseError(
                f"CMS state-filtered response contained {state!r}, expected {requested_state!r}"
            )
        beds = optional_int(
            row.get("number_of_certified_beds"),
            field="number_of_certified_beds",
        )
        residents = optional_decimal(
            row.get("average_number_of_residents_per_day"),
            field="average_number_of_residents_per_day",
        )
        if beds is not None and beds < 0:
            raise SourceParseError("number_of_certified_beds cannot be negative")
        if residents is not None and residents < 0:
            raise SourceParseError("average_number_of_residents_per_day cannot be negative")
        load_ratio = residents / Decimal(beds) if beds and residents is not None else None
        return CmsNursingHomeRecord(
            cms_certification_number=required_text(
                row.get("cms_certification_number_ccn"),
                field="cms_certification_number_ccn",
            ),
            provider_name=required_text(row.get("provider_name"), field="provider_name"),
            state=state,
            city=optional_text(row.get("citytown"), field="citytown"),
            zip_code=optional_text(row.get("zip_code"), field="zip_code"),
            county=optional_text(row.get("countyparish"), field="countyparish"),
            certified_beds=beds,
            average_residents_per_day=residents,
            average_residents_per_certified_bed=load_ratio,
            average_residents_footnote=optional_text(
                row.get("average_number_of_residents_per_day_footnote"),
                field="average_number_of_residents_per_day_footnote",
            ),
            processing_date=optional_date(
                row.get("processing_date"),
                field="processing_date",
            ),
        )


__all__ = [
    "CAPACITY_MEASUREMENT_NOTE",
    "CMS_DATASET_ID",
    "CmsDatasetMetadata",
    "CmsDistribution",
    "CmsMetadataResult",
    "CmsNursingHomeRecord",
    "CmsProviderPage",
    "CmsProviderSource",
]
