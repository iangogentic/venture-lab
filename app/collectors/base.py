"""The collector contract: real sources in, candidate observations out.

Collectors exist to stop the pipeline running on model recall. Without them the
first stage degrades to `Question -> what the model remembers -> Evidence`, which
is exactly the failure the rest of the system is built to avoid: every downstream
guarantee about provenance is worthless if the provenance points at nothing.

A collector does one narrow thing — run a query against one source and return what
it actually found, verbatim. It does not judge relevance, summarise, or interpret;
that is the `collect-evidence` stage's job, and keeping the two apart is what makes
an excerpt checkable against the text it came from.

`SourceItem` is deliberately not an `Evidence` artifact. A collector has no run to
attach to and no view on whether a hit is worth keeping, so it returns raw
candidates and lets the stage decide which of them become artifacts.
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings
from app.utils.errors import CollectorError

COLLECTORS: dict[str, type["Collector"]] = {}
"""Every registered collector, keyed by `Collector.name`."""

_BODY_LIMIT: Final[int] = 200
_LOOKS_LIKE_HTML: Final[re.Pattern[str]] = re.compile(r"^\s*(?:<!doctype html|<html\b)", re.I)
_TITLE: Final[re.Pattern[str]] = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


def short_body(value: object) -> str:
    """A one-line, bounded rendering of something unexpected, for an error message.

    HTML is summarised rather than quoted. When a source refuses at the edge —
    a 429 from a CDN, a captcha, a maintenance page — the body is a whole web
    page, and the first two hundred characters of one are the doctype, a
    charset meta tag and the opening of a stylesheet. That is the least
    informative slice available: it is identical for every such refusal, it says
    nothing about which refusal this was, and it costs several wrapped lines of
    log each time. The `<title>` is the one part written for a human to read.
    """
    if isinstance(value, str) and _LOOKS_LIKE_HTML.match(value):
        found = _TITLE.search(value)
        title = " ".join(found.group(1).split()) if found else ""
        return f"an HTML page titled {title!r}" if title else "an HTML page, not the API's JSON"
    rendered = str(value).replace("\n", " ").strip()
    return rendered[:_BODY_LIMIT] if len(rendered) > _BODY_LIMIT else rendered


class SourceItem(BaseModel):
    """One thing a collector found, as the source stated it.

    `text` is verbatim on purpose: the `collect-evidence` stage checks that the
    excerpt it keeps actually appears here, which is what makes a fabricated quote
    impossible rather than merely discouraged.
    """

    model_config = ConfigDict(extra="forbid")

    collector: str = Field(description="Which collector found this.")
    external_id: str = Field(description="The source's own id, for dedup and citation.")
    text: str = Field(description="The body, verbatim. Never paraphrased.")
    title: str | None = None
    url: str | None = Field(default=None, description="Where a reader can verify it.")
    author: str | None = None
    published_at: datetime | None = None

    @field_validator("text")
    @classmethod
    def _require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("a source item with no text is not evidence of anything")
        return cleaned

    @field_validator("title", "author", "url")
    @classmethod
    def _blank_is_absent(cls, value: str | None) -> str | None:
        return (value.strip() or None) if value else None


class CollectorConfig(BaseModel):
    """Per-collector settings. Concrete collectors extend this."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    limit: int = Field(default=25, ge=1, le=200, description="Cap on items per query.")
    timeout: float = Field(default=20.0, gt=0, description="Seconds per HTTP request.")


class Collector(ABC):
    """One source the pipeline can search."""

    name: ClassVar[str]
    """Registry key, e.g. `github-issues`."""

    description: ClassVar[str]
    """One line, shown by `op collectors`."""

    requires_credentials: ClassVar[bool] = False
    """Whether this source refuses anonymous access and must be configured first."""

    def __init__(self, config: CollectorConfig | None = None) -> None:
        self.config = config if config is not None else CollectorConfig()

    @abstractmethod
    def search(self, query: str, *, limit: int | None = None) -> list[SourceItem]:
        """Run `query` against this source and return what it found.

        An empty list is a legitimate result and must not raise: a source with
        nothing to say about a question is a finding, not a failure.

        Raises:
            CollectorError: On transport failure, a refused request, or a response
                that cannot be understood. Never on an empty result.
        """

    def available(self) -> bool:
        """Whether this collector can run right now.

        Credentialled sources override this to report missing configuration, so a
        run can skip them with a clear reason instead of failing partway through.
        """
        return self.config.enabled


def config_from_settings() -> CollectorConfig:
    """Build the collector config from settings.

    The single place this is assembled. `collect-evidence` and `op doctor` must
    construct collectors identically: given a richer config than the run uses, the
    preflight would report a source available that the run then skips — and given a
    poorer one, it would warn about a source that actually works. Both have happened.
    """
    settings = get_settings()
    return CollectorConfig(
        limit=settings.collector_limit,
        feeds=list(settings.rss_feeds),
        paths=list(settings.corpus_paths),
        searxng_url=settings.searxng_url,
        tavily_api_key=settings.tavily_api_key,
        discourse_forums=list(settings.discourse_forums),
        stackexchange_sites=list(settings.stackexchange_sites),
        stackexchange_key=settings.stackexchange_key,
        app_store_countries=list(settings.app_store_countries),
    )


def register[CollectorT: Collector](cls: type[CollectorT]) -> type[CollectorT]:
    """Register a collector under its `name`; usable as a class decorator."""
    COLLECTORS[cls.name] = cls
    return cls


def get_collector(name: str) -> type[Collector]:
    """Look up a registered collector.

    Raises:
        CollectorError: If no collector is registered under `name`.
    """
    try:
        return COLLECTORS[name]
    except KeyError as exc:
        known = ", ".join(available()) or "<none>"
        raise CollectorError(f"Unknown collector {name!r}; available: {known}") from exc


def available() -> tuple[str, ...]:
    """Every registered collector name, sorted."""
    return tuple(sorted(COLLECTORS))


__all__ = [
    "COLLECTORS",
    "Collector",
    "CollectorConfig",
    "SourceItem",
    "available",
    "config_from_settings",
    "get_collector",
    "register",
    "short_body",
]
