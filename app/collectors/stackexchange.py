"""Collector for the Stack Exchange network, via the public 2.3 API.

A Stack Exchange question is a pain statement by construction: nobody writes
one until something has already resisted their own effort, and the site's rules
push askers to state what they tried, what happened, and what it cost them.
That is why only questions are collected here — the question *is* the pain, and
an answer is somebody else's fix, which is a different kind of evidence.

The API allows keyless anonymous access at a reduced quota, which is why
`stackoverflow` is a sensible zero-config default and why this collector never
demands credentials. An optional application key (`stackexchange_key`) raises
the quota and nothing else — it is not an account credential, but it is still
never written to a log.

Licensing: everything on the network is CC BY-SA 4.0. Attribution is satisfied
because every `SourceItem` carries the question's author and permalink, which
the pipeline preserves into citations — a reader of any downstream artifact can
always see who wrote the words and where they live.

Every response is gzip-compressed regardless of request headers; httpx
decompresses transparently, so no special handling appears below.
"""

import html
import re
from datetime import UTC, datetime
from typing import Any, ClassVar, Final

import httpx

from app.collectors.base import Collector, CollectorConfig, SourceItem, register, short_body
from app.utils.errors import CollectorError, RateLimitedError
from app.utils.logging import get_logger

logger = get_logger(__name__)


_API_URL: Final = "https://api.stackexchange.com/2.3/search/advanced"
_USER_AGENT: Final = "opportunity-engine/0.1 (+https://github.com/)"

# Stack Overflow is the network's largest site by two orders of magnitude, and
# the one a software-pain query is most likely to hit. Defaulting to it keeps
# the collector useful with zero configuration.
_DEFAULT_SITES: Final = ("stackoverflow",)

# The default response filter omits question bodies entirely; `withbody` is the
# API's built-in filter that adds them, and the body is the whole point here.
_FILTER: Final = "withbody"

# One page per site, sized to the caller's limit. The API allows up to 100 per
# page, but the anonymous quota is 300 requests a day shared across every run
# on this IP, so restraint is the difference between a working collector and a
# throttled one.
_MAX_PAGESIZE: Final = 30

# Generous ceiling on one question's text. Cutting is a prefix cut and never
# happens mid-word-choice: what survives is still exactly what the asker wrote,
# so an excerpt taken from it still checks out.
_MAX_TEXT_CHARS: Final = 20_000

# Tags that end a block of prose. Turning them into newlines before every other
# tag is deleted keeps paragraphs apart, so a question does not arrive as one
# unbroken wall of words that no excerpt can be located in.
_BLOCK_TAG_RE: Final = re.compile(
    r"</?(?:p|div|br|hr|li|ul|ol|tr|h[1-6]|blockquote|pre|section|article)\b[^>]*>",
    re.IGNORECASE,
)
_TAG_RE: Final = re.compile(r"<[^>]+>")
_INLINE_SPACE_RE: Final = re.compile(r"[^\S\n]+")
_BLANK_RUN_RE: Final = re.compile(r"\n\s*\n\s*")


@register
class StackExchangeCollector(Collector):
    """Search Stack Exchange questions for problems stated by the people who hit them."""

    name: ClassVar[str] = "stack-exchange"
    description: ClassVar[str] = "Search Stack Exchange questions via the public API."

    # Keyless anonymous access works; a key only raises the quota, so this
    # source must never be skipped for want of credentials.
    requires_credentials: ClassVar[bool] = False

    def __init__(
        self,
        config: CollectorConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Optionally accept an httpx transport, so tests can serve canned payloads.

        `None` — the production default — leaves httpx to build its usual
        network transport; the seam changes nothing unless a caller injects a
        `MockTransport`.
        """
        super().__init__(config)
        self._transport = transport

    def search(self, query: str, *, limit: int | None = None) -> list[SourceItem]:
        """Search every configured site for `query` and return question bodies verbatim.

        Raises:
            RateLimitedError: If the network refused on quota grounds and no
                site answered. Stack Exchange counts requests per IP (and per
                key) across the *whole network*, not per site, so this is the
                whole source saying "not now" — and the caller must stop asking
                rather than move on to its next query.
            CollectorError: If `query` is blank, or if every configured site
                failed for some other reason. One site failing among several is
                logged and skipped instead: a run grounded in the sites that
                answered is worth more than no run at all.
        """
        cleaned = query.strip()
        # The API answers a blank `q` with an in-band error; failing here says
        # why more usefully than relaying that refusal per site would.
        if not cleaned:
            raise CollectorError("Stack Exchange search needs a non-empty query")

        cap = limit or self.config.limit
        sites = _site_names(self.config)

        per_site: list[list[SourceItem]] = []
        searched = 0
        first_failure: CollectorError | None = None
        headers = {"User-Agent": _USER_AGENT}
        with httpx.Client(
            timeout=self.config.timeout,
            headers=headers,
            transport=self._transport,
        ) as client:
            for site in sites:
                try:
                    items, backoff = self._search_site(client, cleaned, site=site, wanted=cap)
                except RateLimitedError as exc:
                    # Network-wide, so the remaining sites would refuse too.
                    # Asking them anyway spends quota to collect refusals — one
                    # real run logged ninety-odd of them per attempt. Logged at
                    # debug, not warning: the refusal is re-raised below with the
                    # same facts, and the caller announces it once for the stage
                    # rather than once per site per query.
                    first_failure = first_failure or exc
                    logger.debug(
                        "stack-exchange: quota or rate limit reached at %s; "
                        "not querying the remaining %d site(s) this call: %s",
                        site,
                        len(sites) - sites.index(site) - 1,
                        exc,
                    )
                    break
                except CollectorError as exc:
                    first_failure = first_failure or exc
                    logger.warning("stack-exchange: skipping site %s: %s", site, exc)
                    continue
                searched += 1
                per_site.append(items)
                if backoff is not None:
                    # A `backoff` in an otherwise healthy response is the API
                    # saying "slow down or be throttled". Sleeping it off here
                    # would stall the whole collect-evidence stage — every other
                    # collector waits behind this one — so it is honoured the
                    # only way a synchronous collector can: keep what this site
                    # gave and stop asking for more this call.
                    logger.warning(
                        "stack-exchange: API asked for a %ss backoff after %s; "
                        "not querying the remaining sites this call",
                        backoff,
                        site,
                    )
                    break

        if first_failure is not None and searched == 0:
            # The *type* survives, not just the message. A quota refusal that
            # arrives as a plain CollectorError tells the caller nothing it can
            # act on, and it goes on to ask the same exhausted network another
            # seven questions.
            failed = f"every configured site failed ({len(sites)} tried); first error: "
            if isinstance(first_failure, RateLimitedError):
                raise RateLimitedError(f"{failed}{first_failure}") from first_failure
            raise CollectorError(f"{failed}{first_failure}") from first_failure

        # Round-robin across sites rather than a merged sort: `sort=relevance`
        # ranks results *within* one site, those ranks are not comparable across
        # sites, and re-sorting by date would discard the relevance the query
        # asked for. Alternating preserves each site's own ordering and stops
        # the largest site from crowding the rest out of the cap.
        return _interleave(per_site)[:cap]

    def _search_site(
        self,
        client: httpx.Client,
        query: str,
        *,
        site: str,
        wanted: int,
    ) -> tuple[list[SourceItem], float | None]:
        """One site's answer to `query`, plus any backoff the API attached to it.

        Raises:
            CollectorError: On transport failure, a refusal (in-band or HTTP),
                or a payload that cannot be understood.
        """
        payload = self._fetch(client, query, site=site, wanted=wanted)
        entries = payload.get("items")
        if entries is None:
            raise CollectorError(
                f"Stack Exchange search on {site} returned no 'items' array: {short_body(payload)}"
            )
        if not isinstance(entries, list):
            raise CollectorError(f"Stack Exchange search on {site} returned a non-list 'items'")

        found: list[SourceItem] = []
        for entry in entries:
            if len(found) >= wanted:
                break
            item = self._to_item(entry, site=site)
            if item is not None:
                found.append(item)

        backoff = payload.get("backoff")
        if isinstance(backoff, int | float) and not isinstance(backoff, bool):
            return found, float(backoff)
        return found, None

    def _fetch(
        self,
        client: httpx.Client,
        query: str,
        *,
        site: str,
        wanted: int,
    ) -> dict[str, Any]:
        """Run one search request and return the decoded payload.

        Raises:
            CollectorError: On transport failure, a refused request, or a body
                that is not the JSON object the API documents. The API signals
                refusals in-band — `error_id`/`error_message` in the JSON body,
                usually paired with an HTTP 400 — and the message is the only
                part a reader can act on, so it is what the error carries.
        """
        params: dict[str, str | int] = {
            "q": query,
            "site": site,
            "order": "desc",
            "sort": "relevance",
            "filter": _FILTER,
            "pagesize": min(wanted, _MAX_PAGESIZE),
        }
        key = _api_key(self.config)
        if key is not None:
            params["key"] = key  # a quota raiser, deliberately kept out of every log line
        try:
            response = client.get(_API_URL, params=params)
        except httpx.HTTPError as exc:
            raise CollectorError(f"Stack Exchange search could not be reached: {exc}") from exc

        try:
            payload: Any = response.json()
        except ValueError as exc:
            if response.is_error:
                raise _refusal(response.status_code, short_body(response.text)) from exc
            raise CollectorError("Stack Exchange search returned a body that is not JSON") from exc
        if not isinstance(payload, dict):
            raise CollectorError(
                f"Stack Exchange search returned {type(payload).__name__}, expected an object"
            )

        error = _in_band_error(payload)
        if error is not None:
            raise CollectorError(f"Stack Exchange refused the {site} search: {error}")
        if response.is_error:
            raise _refusal(response.status_code, short_body(payload))
        return payload

    def _to_item(self, entry: Any, *, site: str) -> SourceItem | None:
        """Convert one question, or return None if it is not usable evidence."""
        if not isinstance(entry, dict):
            return None
        question_id = entry.get("question_id")
        if not isinstance(question_id, int) or isinstance(question_id, bool):
            logger.debug("stack-exchange: skipping a %s hit with no question_id", site)
            return None

        body = entry.get("body")
        text = _to_text(body) if isinstance(body, str) else ""
        # No body, no quote: without the asker's own words there is nothing an
        # excerpt could later be checked against.
        if not text:
            logger.debug("stack-exchange: skipping %s:%s: no quotable body", site, question_id)
            return None

        title = entry.get("title")
        try:
            return SourceItem(
                collector=self.name,
                # Site-qualified because question ids are only unique per site,
                # and a multi-site run must not collapse two questions into one.
                external_id=f"{site}:{question_id}",
                text=text,
                # The API entity-encodes titles even inside a JSON body, so
                # "&amp;" and "&#39;" arrive escaped and must be unescaped to
                # read as the asker wrote them.
                title=html.unescape(title) if isinstance(title, str) else None,
                url=_string(entry.get("link")),
                author=_display_name(entry.get("owner")),
                published_at=_timestamp(entry.get("creation_date")),
            )
        except ValueError:
            # One unusable record must not cost us the rest of the page.
            return None


def _refusal(status: int, body: str) -> CollectorError:
    """The right error for a refused request: throttled, or merely failed."""
    message = f"Stack Exchange search refused the request (HTTP {status}): {body}"
    return RateLimitedError(message) if status in (429, 502) else CollectorError(message)


def _site_names(config: CollectorConfig) -> list[str]:
    """The configured site slugs, defaulting to Stack Overflow.

    Read defensively rather than off a config subclass: `CollectorConfig`
    allows extra keys, so a config loaded from a file can carry
    `stackexchange_sites` without ever being a dedicated subtype, and a stray
    non-string in the list should cost that one entry rather than the
    collector's whole run. The default is non-empty on purpose — keyless
    anonymous access is allowed, so an unconfigured collector can still do
    useful work against the network's largest site.
    """
    raw: object = getattr(config, "stackexchange_sites", None)
    if isinstance(raw, str):
        raw = [raw]  # one site written as a bare string is a natural mistake to make
    if not isinstance(raw, list):
        return list(_DEFAULT_SITES)
    cleaned = [entry.strip() for entry in raw if isinstance(entry, str) and entry.strip()]
    # De-duplicated, first mention winning, because the quota is counted in
    # requests and a site listed twice buys nothing the first listing did not.
    # Hand-maintained lists grow duplicates: one real config had 16 entries and
    # 11 distinct sites, quietly wasting a third of its daily allowance.
    return list(dict.fromkeys(cleaned)) or list(_DEFAULT_SITES)


def _api_key(config: CollectorConfig) -> str | None:
    """The configured quota key, or None — its absence is never an error.

    Not a secret in the credential sense (it identifies the application, not a
    user), but it still never appears in a log line or an error message.
    """
    raw: object = getattr(config, "stackexchange_key", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _in_band_error(payload: dict[str, Any]) -> str | None:
    """The API's own explanation of a refusal, when the payload carries one."""
    if payload.get("error_id") is None:
        return None
    message = payload.get("error_message")
    name = payload.get("error_name")
    if isinstance(message, str) and message.strip():
        if isinstance(name, str) and name.strip():
            return f"{message.strip()} ({name.strip()})"
        return message.strip()
    return short_body(payload)


def _interleave(ranked: list[list[SourceItem]]) -> list[SourceItem]:
    """Merge per-site result lists round-robin, best of each first.

    Ids are deduplicated even though site-qualified ids cannot collide today:
    the cost is a set, and the cost of being wrong — a site listed twice in the
    config, say — is the same question cited twice as two sources.
    """
    merged: list[SourceItem] = []
    seen: set[str] = set()
    deepest = max((len(row) for row in ranked), default=0)
    for rank in range(deepest):
        for row in ranked:
            if rank >= len(row):
                continue
            item = row[rank]
            if item.external_id not in seen:
                seen.add(item.external_id)
                merged.append(item)
    return merged


def _to_text(raw: str) -> str:
    """A question's rendered-HTML body as readable words.

    Tags are dropped and whitespace regularised, but not one word is changed:
    this is the text an excerpt will later be checked against, so the
    conversion has to be the only transformation the body ever undergoes.
    Entities are unescaped *after* tags are stripped — doing it first would
    turn an escaped `&lt;div&gt;` in someone's code sample into a tag and eat
    the rest of the line.
    """
    if not raw.strip():
        return ""
    text = _BLOCK_TAG_RE.sub("\n", raw)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _INLINE_SPACE_RE.sub(" ", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()[:_MAX_TEXT_CHARS]


def _display_name(owner: Any) -> str | None:
    """The asker's display name, if the hit carries one.

    Part of the CC BY-SA bargain: this is the attribution a citation shows.
    """
    if isinstance(owner, dict):
        name = owner.get("display_name")
        if isinstance(name, str) and name.strip():
            # Display names arrive entity-encoded like titles do.
            return html.unescape(name)
    return None


def _string(value: Any) -> str | None:
    """A non-blank string field, or None when the API sent something else."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def _timestamp(value: Any) -> datetime | None:
    """Read `creation_date` (unix seconds) as timezone-aware UTC.

    Aware because the Evidence model downstream rejects naive datetimes; an
    unparseable stamp costs the metadata, never the item.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None


__all__ = ["StackExchangeCollector"]
