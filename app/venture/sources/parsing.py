"""Small, dependency-free response parsing helpers."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import cast

from app.venture.sources.errors import SourceParseError

JsonObject = dict[str, object]


def json_object(raw_bytes: bytes, *, source: str) -> JsonObject:
    """Decode a UTF-8 JSON object, preserving decimal values exactly."""
    try:
        parsed = json.loads(raw_bytes, parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceParseError(f"{source} returned malformed JSON") from exc
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise SourceParseError(f"{source} response must be a JSON object")
    return cast(JsonObject, parsed)


def object_value(value: object, *, field: str) -> JsonObject:
    """Require a JSON object at a named field."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SourceParseError(f"{field} must be an object")
    return cast(JsonObject, value)


def object_list(value: object, *, field: str) -> list[JsonObject]:
    """Require a list of JSON objects at a named field."""
    if not isinstance(value, list):
        raise SourceParseError(f"{field} must be a list")
    records: list[JsonObject] = []
    for index, item in enumerate(value):
        records.append(object_value(item, field=f"{field}[{index}]"))
    return records


def required_text(value: object, *, field: str) -> str:
    """Require a non-empty textual field."""
    if not isinstance(value, str) or not value.strip():
        raise SourceParseError(f"{field} must be non-empty text")
    return value.strip()


def optional_text(value: object, *, field: str) -> str | None:
    """Normalize a blank or null source string to missing."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise SourceParseError(f"{field} must be text or null")
    return value.strip() or None


def optional_decimal(value: object, *, field: str) -> Decimal | None:
    """Normalize a source decimal string/number, preserving missing values."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise SourceParseError(f"{field} must be numeric or missing")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise SourceParseError(f"{field} must be numeric or missing") from exc
    if not result.is_finite():
        raise SourceParseError(f"{field} must be finite")
    return result


def optional_int(value: object, *, field: str) -> int | None:
    """Normalize an integral source string/number, preserving missing values."""
    parsed = optional_decimal(value, field=field)
    if parsed is None:
        return None
    integral = parsed.to_integral_value()
    if parsed != integral:
        raise SourceParseError(f"{field} must be an integer")
    return int(integral)


def optional_date(value: object, *, field: str) -> date | None:
    """Parse an ISO date, preserving missing values."""
    text = optional_text(value, field=field)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise SourceParseError(f"{field} must be an ISO date") from exc


__all__ = [
    "JsonObject",
    "json_object",
    "object_list",
    "object_value",
    "optional_date",
    "optional_decimal",
    "optional_int",
    "optional_text",
    "required_text",
]
