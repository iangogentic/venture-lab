"""Deterministic identifiers, canonical JSON, and safe relative paths.

Identifiers in the venture ledger become filenames, lookup keys, and hash
inputs.  This module gives all three uses one deliberately narrow grammar:
ASCII letters and digits, with internal ``_`` and ``-`` separators.  In
particular, path separators and dot segments are never valid identifiers.
"""

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Final

from pydantic import BaseModel

_SAFE_IDENTIFIER: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,126}[A-Za-z0-9])?$"
)
_SHA256: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


def validate_identifier(value: str, *, field: str = "identifier") -> str:
    """Return a safe opaque identifier or raise ``ValueError``.

    The explicit path checks make the security property obvious even if the
    regular expression is changed later.
    """
    if not value:
        raise ValueError(f"{field} must not be blank")
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or _SAFE_IDENTIFIER.fullmatch(value) is None
    ):
        raise ValueError(
            f"{field} must contain only ASCII letters, digits, '_' and '-' "
            "and must not contain path segments"
        )
    return value


def validate_sha256(value: str, *, field: str = "sha256") -> str:
    """Require a lowercase, full-length SHA-256 hexadecimal digest."""
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be exactly 64 lowercase hexadecimal characters")
    return value


def validate_relative_path(value: str) -> str:
    """Validate a normalized POSIX relative path without traversal."""
    if not value or "\\" in value or "\x00" in value:
        raise ValueError("relative path is blank or contains a forbidden character")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("path must be a normalized relative path without dot segments")
    if path.as_posix() != value:
        raise ValueError("path must use normalized POSIX separators")
    return value


def safe_join(root: Path, relative_path: str) -> Path:
    """Resolve ``relative_path`` under ``root`` and fail closed on escape."""
    normalized = validate_relative_path(relative_path)
    resolved_root = root.expanduser().resolve()
    candidate = (resolved_root / normalized).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"path {relative_path!r} escapes root {resolved_root}")
    return candidate


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``content``."""
    return hashlib.sha256(content).hexdigest()


def canonical_json(value: object) -> bytes:
    """Serialize supported data to stable UTF-8 JSON bytes.

    Hashes must not depend on insertion order, whitespace, timezone formatting
    chosen by a caller, or Pydantic implementation details.  Non-finite numbers
    are rejected because JSON has no portable representation for them.
    """
    normalized = _normalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def make_content_id(prefix: str, content: object, *, digest_length: int = 64) -> str:
    """Build a deterministic identifier from canonicalized content."""
    safe_prefix = validate_identifier(prefix, field="prefix")
    if not 16 <= digest_length <= 64:
        raise ValueError("digest_length must be between 16 and 64")
    digest = sha256_bytes(canonical_json(content))
    return f"{safe_prefix}_{digest[:digest_length]}"


def _normalize(value: object) -> object:
    """Convert supported rich values into canonical JSON primitives."""
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetimes cannot be canonicalized")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers cannot be canonicalized")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            normalized[key] = _normalize(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    raise TypeError(f"{type(value).__name__} is not canonical JSON data")


__all__ = [
    "canonical_json",
    "make_content_id",
    "safe_join",
    "sha256_bytes",
    "validate_identifier",
    "validate_relative_path",
    "validate_sha256",
]
