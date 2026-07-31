"""Time helpers.

One source of "now" for the whole app: artifacts and ledger rows must agree on
the clock, and everything is timezone-aware UTC so timestamps stay comparable
across machines.
"""

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(UTC)


__all__ = ["utcnow"]
