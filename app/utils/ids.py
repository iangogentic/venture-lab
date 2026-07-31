"""Identifier helpers.

Artifact ids are prefixed and sortable: the prefix says what the thing is when
it shows up in a filename or a log line, and the ULID-ish body keeps files in
creation order when the workspace is listed.
"""

import re
import time
import unicodedata
from uuid import uuid4

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def new_id(prefix: str) -> str:
    """Return a new opaque, prefixed identifier, e.g. `new_id("ev")` -> `ev_0197f3a1...`.

    The body opens with the creation time in hex milliseconds, so ids — and the
    workspace filenames built from them — sort in creation order; the random
    tail keeps ids minted in the same millisecond distinct.
    """
    stamp = f"{time.time_ns() // 1_000_000:012x}"
    return f"{prefix}_{stamp}{uuid4().hex[:20]}"


def slugify(value: str, *, max_length: int = 60) -> str:
    """Reduce arbitrary text to a filesystem- and URL-safe slug."""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = _NON_SLUG.sub("-", normalized.lower()).strip("-")
    return slug[:max_length].rstrip("-")


__all__ = ["new_id", "slugify"]
