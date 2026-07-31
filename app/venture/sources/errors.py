"""Errors raised by the venture-lab source adapters."""


class SourceError(RuntimeError):
    """Base class for a source request or response failure."""


class EgressPolicyError(SourceError):
    """A request or response violated the fixed-endpoint egress policy."""


class SourceTransportError(SourceError):
    """The approved remote source could not be reached successfully."""


class SourceParseError(SourceError):
    """An approved source returned a response that did not match its contract."""


__all__ = [
    "EgressPolicyError",
    "SourceError",
    "SourceParseError",
    "SourceTransportError",
]
