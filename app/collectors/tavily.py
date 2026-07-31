"""Collector for the open web, via the Tavily search API.

Tavily is the hosted counterpart to the `web` collector: the same reach into
the long tail, with no instance to run, paid for with an API key and a metered
quota. The key arrives through the `tavily_api_key` config value; without one
the collector reports itself unavailable, so a run skips it with a reason
instead of dying on a source it was never configured to use.

One decision in this module matters more than the rest. Tavily returns two
texts per hit and they are not equal in kind. `raw_content` is the literal
text Tavily fetched from the page — asked for explicitly, preferred without
exception — so an excerpt kept from it is checked downstream against what the
page actually said. `content` is a snippet Tavily's own models compose, which
may compress or rephrase: words the source never wrote. It is accepted only
when `raw_content` is absent, and the trade-off is stated plainly so nobody
mistakes it — an item built from `content` grounds its excerpts in Tavily's
rendering of the page, not the page itself. Dropping such hits entirely was
the alternative, rejected because a hit with a real URL is still a lead: the
citation lets a reader verify the source even when the snippet is second-hand.
"""

import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar, Final

import httpx

from app.collectors.base import Collector, CollectorConfig, SourceItem, register
from app.utils.errors import CollectorError, RateLimitedError
from app.utils.logging import get_logger

logger = get_logger(__name__)

_API_URL: Final = "https://api.tavily.com/search"
_USER_AGENT: Final = "opportunity-engine/0.1 (+https://github.com/)"

# Cap on results per request. Tavily accepts more, but every extra result is a
# page it fetches and bills for, and evidence past the first ten hits of a web
# search is rarely worth the credits it costs.
_MAX_API_RESULTS: Final = 10

# Generous ceiling on one result's text. Cutting is a prefix cut: what survives
# is still exactly what the source served, so an excerpt taken from it still
# checks out. Only material past the cap is lost.
_MAX_TEXT_CHARS: Final = 20_000

_BLANK_RUN_RE: Final = re.compile(r"\n\s*\n\s*")


@register
class TavilyCollector(Collector):
    """Search the web through Tavily, preferring the literal page text."""

    name: ClassVar[str] = "tavily"
    description: ClassVar[str] = "General web search via the Tavily API."
    requires_credentials: ClassVar[bool] = True

    def __init__(
        self,
        config: CollectorConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        # The test seam: a MockTransport here exercises the whole request and
        # parse path without a network. None means httpx builds its usual
        # transport, so production behaviour is unchanged.
        self._transport = transport

    def available(self) -> bool:
        """False without an API key — Tavily refuses anonymous requests."""
        return self.config.enabled and _api_key(self.config) is not None

    def search(self, query: str, *, limit: int | None = None) -> list[SourceItem]:
        """Search Tavily for `query` and return what it found, verbatim where possible.

        Returns at most `limit or self.config.limit` items, in Tavily's own
        ranking order. Unusable results are skipped, so fewer items than asked
        for is normal and an empty list is a legitimate answer.

        Raises:
            CollectorError: If the collector is unconfigured, Tavily refuses
                the request (rejected key, rate limit), the transport fails, or
                the reply cannot be read. Never for an empty result.
        """
        wanted = self.config.limit if limit is None else limit
        if wanted <= 0:
            return []
        text = query.strip()
        if not text:
            # Tavily rejects an empty query outright. Nothing was asked, so
            # nothing was found.
            return []
        key = _api_key(self.config)
        if key is None:
            raise CollectorError(
                "The tavily collector is not configured: set TAVILY_API_KEY (create "
                "one at https://app.tavily.com), or disable the tavily collector for "
                "this run.",
            )

        payload = self._fetch(text, wanted=wanted, key=key)
        results = payload.get("results")
        if not isinstance(results, list):
            raise CollectorError("Tavily's reply carried no 'results' array.")

        found: list[SourceItem] = []
        for index, row in enumerate(results):
            if len(found) >= wanted:
                break
            item = self._to_item(row)
            if item is None:
                # One unusable result must never lose the rest of the page.
                logger.debug("tavily: skipping unusable result %d", index)
                continue
            found.append(item)
        return found

    # --------------------------------------------------------------- internals

    def _fetch(self, query: str, *, wanted: int, key: str) -> dict[str, Any]:
        """One search request, decoded. The key goes into a header and nowhere else.

        Raises:
            CollectorError: On transport failure, a refused request, or a body
                that is not JSON. The messages name TAVILY_API_KEY so the
                operator knows what to fix; the key itself is never echoed.
        """
        body = {
            "query": query,
            "max_results": min(wanted, _MAX_API_RESULTS),
            # Without this Tavily sends only its composed snippets, and a
            # snippet its models wrote is not the page's own text — see the
            # module docstring for why that matters.
            "include_raw_content": True,
        }
        try:
            with httpx.Client(
                timeout=self.config.timeout,
                headers={"User-Agent": _USER_AGENT, "Authorization": f"Bearer {key}"},
                transport=self._transport,
            ) as client:
                response = client.post(_API_URL, json=body)
        except httpx.HTTPError as exc:
            raise CollectorError(f"Tavily could not be reached: {exc}") from exc

        status = response.status_code
        if status in (401, 403):
            raise CollectorError(
                f"Tavily rejected the API key (HTTP {status}). Check TAVILY_API_KEY "
                "against the dashboard at https://app.tavily.com.",
            )
        if status == 429:
            raise RateLimitedError(
                "Tavily rate limit hit (HTTP 429). The free tier is 1,000 credits a "
                "month; wait for the quota to reset, lower the collector's limit, or "
                "upgrade the plan.",
            )
        if response.is_error:
            raise CollectorError(f"Tavily answered HTTP {status} to the search.")

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise CollectorError("Tavily's reply was not JSON.") from exc
        if not isinstance(payload, dict):
            raise CollectorError(
                f"Tavily's reply was {type(payload).__name__}, expected an object.",
            )
        return payload

    def _to_item(self, row: Any) -> SourceItem | None:
        """One result as a `SourceItem`, or None if it is unusable.

        None rather than an exception: one malformed record must not cost the
        run the other nine.
        """
        if not isinstance(row, dict):
            return None
        url = _string(row.get("url"))
        if url is None:
            # A result with no URL cannot be cited, and an uncitable result is
            # not evidence.
            return None

        # The page's own words first, Tavily's rendering of them only as a
        # fallback — the module docstring carries the full argument.
        text = _normalised(row.get("raw_content")) or _normalised(row.get("content"))
        if text is None:
            logger.debug("tavily: skipping %s, no text in either field", url)
            return None

        try:
            return SourceItem(
                collector=self.name,
                external_id=url,
                text=text,
                title=_string(row.get("title")),
                url=url,
                published_at=_lenient_date(row.get("published_date")),
            )
        except ValueError as exc:
            logger.debug("tavily: skipping unusable result %s: %s", url, exc)
            return None


def _api_key(config: CollectorConfig) -> str | None:
    """The configured Tavily API key, or None.

    Read defensively rather than off a typed subclass: `CollectorConfig` allows
    extra keys, so the shared factory hands over a plain config whose extras
    are untyped. Settings deliver secrets as `SecretStr` so they never repr
    into a log line; a config built from a plain dict delivers a bare string.
    Both are accepted, and anything else reads as unconfigured.
    """
    raw: object = getattr(config, "tavily_api_key", None)
    unwrap = getattr(raw, "get_secret_value", None)
    if callable(unwrap):
        raw = unwrap()
    if not isinstance(raw, str):
        return None
    return raw.strip() or None


def _normalised(value: object) -> str | None:
    """A text field with its whitespace regularised and its length bounded.

    Whitespace only: line endings unified and blank runs collapsed, not one
    word changed — this is the text an excerpt will later be checked against,
    so regularising is the only transformation it may undergo. The cut is a
    prefix cut, so everything kept is exactly what the source served.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()[:_MAX_TEXT_CHARS]


def _string(value: object) -> str | None:
    """A non-blank string field, stripped, or None when the API sent something else."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _lenient_date(value: object) -> datetime | None:
    """Read `published_date` in whichever format Tavily used, or give up quietly.

    An unparseable date is never a reason to drop a result — the words are the
    evidence, the timestamp only metadata about them.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        return _as_utc(datetime.fromisoformat(raw))
    except ValueError:
        pass
    try:
        return _as_utc(parsedate_to_datetime(raw))
    except (TypeError, ValueError):
        return None


def _as_utc(value: datetime) -> datetime:
    """Normalise to timezone-aware UTC.

    Naive timestamps do occur, and the Evidence model downstream rejects naive
    datetimes outright. Assuming UTC keeps an otherwise good date rather than
    discarding it.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = ["TavilyCollector"]
