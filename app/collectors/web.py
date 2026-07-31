"""Collector for the open web, via a SearXNG metasearch instance.

The other collectors each read one venue; this one is the reach into everywhere
else — blog posts, documentation, small forums, the long tail where a pain is
described once and never again. It needs no API key, but it does need
infrastructure: a SearXNG instance the operator runs or trusts, named by
`searxng_url`. Left unset, the collector reports itself unavailable rather than
failing — an unconfigured metasearch is not a broken one.

A search engine's snippet is thin evidence, so each hit's page is fetched and
its main text extracted with trafilatura. Extraction is a selection, not a
rewording: boilerplate — navigation, ads, footers — is dropped, and every
sentence kept is a sentence the page served, which is what makes an excerpt
checkable against `SourceItem.text` downstream. When a page will not yield text
(JS-only rendering, a refused fetch, a body too large to be an article) the
engine's own snippet stands in, because a snippet the engine served is still
verbatim text from a real response — less of the page, but not less true.

Page fetches are capped well below the collector's limit. Every hit is a
different stranger's server, and one query fanning out into dozens of fetches
is how an operator's IP ends up on blocklists.
"""

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar, Final

import httpx
import trafilatura

from app.collectors.base import Collector, CollectorConfig, SourceItem, register
from app.utils.errors import CollectorError
from app.utils.logging import get_logger

logger = get_logger(__name__)

_USER_AGENT: Final = "opportunity-engine/0.1 (+https://github.com/)"

# Politeness cap on page fetches per query, whatever the collector's limit says.
# Each hit lives on a different host, so this bounds how many strangers' servers
# one question may knock on; eight full pages is already a lot of evidence.
_MAX_PAGE_FETCHES: Final = 8

# A body over this is not an article, it is a download; skipping it costs one
# page (the snippet still stands in) and protects the run from a hostile host.
_MAX_PAGE_BYTES: Final = 4_000_000

# Generous ceiling on one page's extracted text. Cutting is a prefix cut: what
# survives is still exactly what the page said, so an excerpt taken from it
# still checks out. Only material past the cap is lost.
_MAX_TEXT_CHARS: Final = 20_000


@register
class WebCollector(Collector):
    """Search the web through a SearXNG instance and read the pages it finds.

    With no `searxng_url` in its config there is no instance to ask, so
    `available()` is False; a run skips it with a reason instead of failing on
    a source nobody pointed anywhere.
    """

    name: ClassVar[str] = "web"
    description: ClassVar[str] = "General web search via a self-hosted SearXNG instance."
    requires_credentials: ClassVar[bool] = False

    def __init__(
        self,
        config: CollectorConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        # The test seam: a MockTransport here exercises the whole search, fetch
        # and extract path without a network. None means httpx builds its usual
        # transport, so production behaviour is unchanged.
        self._transport = transport

    def available(self) -> bool:
        """False without a `searxng_url` — there is no instance to ask."""
        return self.config.enabled and _searxng_url(self.config) is not None

    def search(self, query: str, *, limit: int | None = None) -> list[SourceItem]:
        """Search the instance, fetch the result pages, and return their text.

        Returns at most `min(limit, _MAX_PAGE_FETCHES)` items, in the order the
        engine ranked them. A hit whose page yields nothing falls back to the
        engine's snippet, and one bad page never loses the others.

        Raises:
            CollectorError: If the collector is unconfigured, the instance
                refuses or cannot be reached, or its reply is not a result
                page. Never for an empty result.
        """
        wanted = self.config.limit if limit is None else limit
        if wanted <= 0:
            return []
        text = query.strip()
        if not text:
            # SearXNG answers a blank `q` with its front page, not a result
            # list. Nothing was asked, so nothing was found.
            return []
        base = _searxng_url(self.config)
        if base is None:
            raise CollectorError(
                "The web collector is not configured: set SEARXNG_URL to the base URL "
                "of a SearXNG instance (e.g. http://localhost:8888), or disable the "
                "web collector for this run.",
            )

        cap = min(wanted, _MAX_PAGE_FETCHES)
        found: list[SourceItem] = []
        seen: set[str] = set()
        with httpx.Client(
            timeout=self.config.timeout,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            transport=self._transport,
        ) as client:
            for index, hit in enumerate(self._results(client, base, text)):
                if len(found) >= cap:
                    break
                try:
                    item = self._to_item(client, hit)
                except Exception as exc:
                    # Deliberately broad: the guarantee is that one malformed hit
                    # or one strange page never loses the rest, and the ways a
                    # stranger's HTML can go wrong are not enumerable.
                    logger.debug("web: skipping result %d: %s", index, exc)
                    continue
                if item is not None and item.external_id not in seen:
                    # Metasearch merges engines, and two engines can still hand
                    # back the same URL; the same page cited twice is one source
                    # masquerading as corroboration.
                    seen.add(item.external_id)
                    found.append(item)
        return found

    # --------------------------------------------------------------- internals

    def _results(self, client: httpx.Client, base: str, query: str) -> list[Any]:
        """One search against the instance, as SearXNG's raw result rows.

        Raises:
            CollectorError: On transport failure, a refused request, or a body
                that is not a SearXNG JSON result page.
        """
        params = {"q": query, "format": "json", "safesearch": "0"}
        try:
            response = client.get(f"{base}/search", params=params)
        except httpx.HTTPError as exc:
            raise CollectorError(f"SearXNG at {base} could not be reached: {exc}") from exc

        if response.status_code == 403:
            # The stock configuration serves HTML only and answers 403 to
            # `format=json`; the fix is one line on the instance, so say so.
            raise CollectorError(
                f"SearXNG at {base} refused the JSON request (HTTP 403). Enable JSON "
                "output on the instance: add `json` under `search: formats:` in its "
                "settings.yml, then restart it.",
            )
        if response.is_error:
            raise CollectorError(
                f"SearXNG at {base} answered HTTP {response.status_code} to the search.",
            )

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise CollectorError(
                f"SearXNG at {base} answered with a body that is not JSON; check that "
                "the URL points at a SearXNG instance.",
            ) from exc
        if not isinstance(payload, dict):
            raise CollectorError(
                f"SearXNG at {base} answered {type(payload).__name__}, expected an object.",
            )
        results = payload.get("results")
        if not isinstance(results, list):
            # An empty list is a legitimate answer; a missing one means this is
            # not the search endpoint we think it is.
            raise CollectorError(f"SearXNG at {base} returned no 'results' array.")
        return results

    def _to_item(self, client: httpx.Client, hit: Any) -> SourceItem | None:
        """One search hit as a `SourceItem`, or None if nothing quotable came of it."""
        if not isinstance(hit, dict):
            return None
        url = _string(hit.get("url"))
        if url is None:
            # A hit with no URL cannot be cited, and an uncitable hit is not evidence.
            return None

        text: str | None = None
        author: str | None = None
        published: datetime | None = None
        page = self._fetch_page(client, url)
        if page is not None:
            text, author, published = _extracted(page, url)
        if not text:
            # JS-heavy pages render to nothing on the wire, and some hosts
            # refuse bots outright. The engine's snippet is still verbatim
            # engine-served text — less of the page, but not less true.
            logger.debug("web: falling back to the engine snippet for %s", url)
            text = _string(hit.get("content"))
        if not text:
            logger.debug("web: skipping %s, no text from the page or the engine", url)
            return None

        return SourceItem(
            collector=self.name,
            # The URL is the source's own identity for a web page: stable across
            # engines, and exactly what a reader needs to verify the text.
            external_id=url,
            text=text,
            title=_string(hit.get("title")),
            url=url,
            author=author,
            published_at=published or _lenient_date(hit.get("publishedDate")),
        )

    def _fetch_page(self, client: httpx.Client, url: str) -> str | None:
        """One result page's HTML, or None when it will not usefully arrive.

        Failures are None, not errors: the caller has the engine's snippet to
        fall back on, and one dead page must not cost the query its other hits.
        """
        try:
            response = client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.debug("web: could not fetch %s: %s", url, exc)
            return None
        if len(response.content) > _MAX_PAGE_BYTES:
            logger.debug(
                "web: %s returned %d bytes, over the %d byte cap",
                url,
                len(response.content),
                _MAX_PAGE_BYTES,
            )
            return None
        return response.text


def _searxng_url(config: CollectorConfig) -> str | None:
    """The configured SearXNG base URL, or None.

    Read defensively rather than off a typed subclass: `CollectorConfig` allows
    extra keys, so the shared factory hands over a plain config whose extras are
    untyped, and a non-string here should read as unconfigured rather than crash
    the run. The trailing slash is dropped so `{base}/search` never doubles it.
    """
    raw: object = getattr(config, "searxng_url", None)
    if not isinstance(raw, str):
        return None
    return raw.strip().rstrip("/") or None


def _extracted(page: str, url: str) -> tuple[str | None, str | None, datetime | None]:
    """A page's main text, author and date, as trafilatura reads them.

    Extraction drops boilerplate and returns the article body as plain text.
    That is a selection, not a rewording: every sentence kept is one the page
    served, so an excerpt taken from the result still verifies against reality.
    A page trafilatura cannot read yields no text and the caller falls back.
    """
    text = trafilatura.extract(page, url=url, include_comments=False, output_format="txt")
    author: str | None = None
    published: datetime | None = None
    try:
        metadata = trafilatura.extract_metadata(page, default_url=url)
    except Exception as exc:
        # Metadata is decoration on the evidence; a crash reading it must not
        # cost the text it decorates.
        logger.debug("web: metadata extraction failed for %s: %s", url, exc)
        metadata = None
    if metadata is not None:
        author = _string(metadata.author)
        published = _lenient_date(metadata.date)
    if not text or not text.strip():
        return None, author, published
    return text.strip()[:_MAX_TEXT_CHARS], author, published


def _string(value: object) -> str | None:
    """A non-blank string field, stripped, or None when the source sent something else."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _lenient_date(value: object) -> datetime | None:
    """Read a date field in whichever format the source used, or give up quietly.

    Engines and pages disagree: ISO 8601 with or without an offset, RFC 822, a
    bare `YYYY-MM-DD` from trafilatura. An unparseable date is never a reason to
    drop an item — the words are the evidence, the timestamp only metadata.
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

    Naive timestamps do occur — trafilatura reports bare dates, engines drop
    offsets — and the Evidence model downstream rejects naive datetimes
    outright. Assuming UTC keeps an otherwise good date rather than discarding
    it, at the cost of a few hours' error on a sloppily published page.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = ["WebCollector"]
