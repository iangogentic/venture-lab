"""USAspending transaction obligations grouped by NAICS."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from itertools import pairwise
from typing import Final

from app.venture.sources.errors import SourceParseError
from app.venture.sources.http import USASPENDING_NAICS, RawCapture, SafeHttpClient
from app.venture.sources.parsing import (
    JsonObject,
    json_object,
    object_list,
    object_value,
    optional_decimal,
    optional_int,
    required_text,
)

CONTRACT_AWARD_TYPE_CODES: Final = ("A", "B", "C", "D")
OBLIGATION_MEASUREMENT_NOTE: Final = (
    "USAspending transaction-level prime-contract obligations grouped by NAICS "
    "for the exact requested action-date period."
)
_EARLIEST_SEARCH_DATE: Final = date(2007, 10, 1)
_MAX_PAGE_SIZE: Final = 100


@dataclass(frozen=True, slots=True)
class NaicsContractObligation:
    """One NAICS category and its summed federal contract obligations."""

    naics_code: str
    name: str
    obligations_usd: Decimal
    total_outlays_usd: Decimal | None
    year_retired: int | None
    measurement_note: str = OBLIGATION_MEASUREMENT_NOTE


@dataclass(frozen=True, slots=True)
class UsaSpendingNaicsPage:
    """One ranked page plus the exact frozen POST request and response."""

    capture: RawCapture
    start_date: date
    end_date: date
    page: int
    limit: int
    next_page: int | None
    records: tuple[NaicsContractObligation, ...]
    messages: tuple[str, ...]


class UsaSpendingSource:
    """Fetch top NAICS contract obligations from the official v2 API."""

    def __init__(self, client: SafeHttpClient | None = None) -> None:
        self._client = client or SafeHttpClient()

    def fetch_top_naics(
        self,
        *,
        start_date: date,
        end_date: date,
        page: int = 1,
        limit: int = 25,
    ) -> UsaSpendingNaicsPage:
        """Fetch descending prime-contract obligations for an action-date period."""
        if start_date < _EARLIEST_SEARCH_DATE:
            raise ValueError(f"start_date must be on or after {_EARLIEST_SEARCH_DATE.isoformat()}")
        if end_date < start_date:
            raise ValueError("end_date must not precede start_date")
        if page < 1:
            raise ValueError("page must be at least 1")
        if not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")

        query: dict[str, object] = {
            "category": "naics",
            "filters": {
                "award_type_codes": list(CONTRACT_AWARD_TYPE_CODES),
                "time_period": [
                    {
                        "end_date": end_date.isoformat(),
                        "start_date": start_date.isoformat(),
                    }
                ],
            },
            "limit": limit,
            "page": page,
            "spending_level": "transactions",
        }
        capture = self._client.request(USASPENDING_NAICS, json_body=query)
        payload = json_object(capture.raw_bytes, source=USASPENDING_NAICS.name)
        if payload.get("category") != "naics":
            raise SourceParseError("USAspending category must be 'naics'")
        if payload.get("spending_level") != "transactions":
            raise SourceParseError("USAspending spending_level must be 'transactions'")
        records = tuple(
            self._parse_record(row) for row in object_list(payload.get("results"), field="results")
        )
        self._validate_descending(records)
        metadata = object_value(payload.get("page_metadata"), field="page_metadata")
        has_next = metadata.get("hasNext")
        if not isinstance(has_next, bool):
            raise SourceParseError("page_metadata.hasNext must be boolean")
        next_page_value = optional_int(metadata.get("next"), field="page_metadata.next")
        next_page = next_page_value if has_next else None
        if has_next and (next_page is None or next_page <= page):
            raise SourceParseError("page_metadata.next must advance when hasNext is true")
        messages_value = payload.get("messages", [])
        if not isinstance(messages_value, list) or not all(
            isinstance(message, str) for message in messages_value
        ):
            raise SourceParseError("messages must be a list of strings")
        return UsaSpendingNaicsPage(
            capture=capture,
            start_date=start_date,
            end_date=end_date,
            page=page,
            limit=limit,
            next_page=next_page,
            records=records,
            messages=tuple(messages_value),
        )

    @staticmethod
    def _parse_record(row: JsonObject) -> NaicsContractObligation:
        obligations = optional_decimal(row.get("amount"), field="results.amount")
        if obligations is None:
            raise SourceParseError("results.amount is required")
        if obligations < 0:
            raise SourceParseError("results.amount cannot be negative")
        return NaicsContractObligation(
            naics_code=required_text(row.get("code"), field="results.code"),
            name=required_text(row.get("name"), field="results.name"),
            obligations_usd=obligations,
            total_outlays_usd=optional_decimal(
                row.get("total_outlays"),
                field="results.total_outlays",
            ),
            year_retired=optional_int(row.get("year_retired"), field="results.year_retired"),
        )

    @staticmethod
    def _validate_descending(records: tuple[NaicsContractObligation, ...]) -> None:
        if any(
            previous.obligations_usd < current.obligations_usd
            for previous, current in pairwise(records)
        ):
            raise SourceParseError("USAspending NAICS results are not ordered by obligation")


__all__ = [
    "CONTRACT_AWARD_TYPE_CODES",
    "OBLIGATION_MEASUREMENT_NOTE",
    "NaicsContractObligation",
    "UsaSpendingNaicsPage",
    "UsaSpendingSource",
]
