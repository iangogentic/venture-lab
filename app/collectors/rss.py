"""Collector for RSS 2.0 and Atom feeds.

Feeds are the cheapest grounded source there is: engineering blogs, changelogs,
release notes, status pages and forum searches (Hacker News publishes one) all
expose them, and none of them needs a key. What a feed cannot do is answer a
question. It carries whatever its publisher last put in it, so `query` is applied
client-side to the entries a fetch happened to return, and a feed that has moved
on since the pain was discussed simply has nothing to say — which it says by
returning nothing, not by failing.

Entry bodies are HTML. They are turned into words once, here, and never touched
again: `collect-evidence` checks its excerpt against `SourceItem.text`, so what
this module returns *is* the ground truth an excerpt is measured against. Markup
is removed and whitespace regularised; wording never is.

Parsing uses `xml.etree.ElementTree`, deliberately — a feed reader is not worth a
new dependency, and the two dialects differ in about a dozen tag names. Feed
bodies are size-capped before they reach the parser, which is the one real
defence available against a hostile feed.
"""

import html
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import ClassVar, Final, NamedTuple
from xml.etree import ElementTree

import httpx
from pydantic import Field

from app.collectors.base import Collector, CollectorConfig, SourceItem, register
from app.utils.errors import CollectorError
from app.utils.logging import get_logger

logger = get_logger(__name__)

_USER_AGENT: Final = "opportunity-engine/0.1 (+https://github.com/)"
_ACCEPT: Final = "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8"

# Atom qualifies every tag; RSS 2.0 qualifies none of its own, and borrows Dublin
# Core for the author and the content module for the full post body.
_ATOM_NS: Final = "http://www.w3.org/2005/Atom"
_ATOM_ENTRY: Final = f"{{{_ATOM_NS}}}entry"
_ATOM_TITLE: Final = f"{{{_ATOM_NS}}}title"
_ATOM_CONTENT: Final = f"{{{_ATOM_NS}}}content"
_ATOM_SUMMARY: Final = f"{{{_ATOM_NS}}}summary"
_ATOM_ID: Final = f"{{{_ATOM_NS}}}id"
_ATOM_LINK: Final = f"{{{_ATOM_NS}}}link"
_ATOM_PUBLISHED: Final = f"{{{_ATOM_NS}}}published"
_ATOM_UPDATED: Final = f"{{{_ATOM_NS}}}updated"
_ATOM_AUTHOR_NAME: Final = f"{{{_ATOM_NS}}}author/{{{_ATOM_NS}}}name"
_DC_CREATOR: Final = "{http://purl.org/dc/elements/1.1/}creator"
_CONTENT_ENCODED: Final = "{http://purl.org/rss/1.0/modules/content/}encoded"

# Tags that end a block of prose. Turning them into newlines before every other
# tag is deleted keeps paragraphs apart, so a post does not arrive as one
# unbroken wall of words that no excerpt can be located in.
_BLOCK_TAG_RE: Final = re.compile(
    r"</?(?:p|div|br|hr|li|ul|ol|tr|h[1-6]|blockquote|pre|section|article)\b[^>]*>",
    re.IGNORECASE,
)
_TAG_RE: Final = re.compile(r"<[^>]+>")
_INLINE_SPACE_RE: Final = re.compile(r"[^\S\n]+")
_BLANK_RUN_RE: Final = re.compile(r"\n\s*\n\s*")

# A feed body over this is not a feed, it is an accident or an attack; refusing to
# parse it costs one source and protects the run.
_MAX_FEED_BYTES: Final = 4_000_000
# Generous ceiling on one entry's text. Cutting is a prefix cut and never happens
# mid-word-choice: what survives is still exactly what the publisher wrote, so an
# excerpt taken from it still checks out. Only material past the cap is lost.
_MAX_TEXT_CHARS: Final = 20_000
# Ordering floor for entries whose date could not be parsed, so they sort last.
_UNDATED: Final = datetime(1970, 1, 1, tzinfo=UTC)


class RssConfig(CollectorConfig):
    """Settings for `RssCollector`."""

    feeds: list[str] = Field(
        default_factory=list,
        description="Feed URLs to fetch. RSS 2.0 or Atom; the dialect is detected.",
    )


class _Fields(NamedTuple):
    """What one entry offers, once the dialect's tag names have been resolved."""

    external_id: str | None
    title: str | None
    body: str
    url: str | None
    author: str | None
    published_at: datetime | None


@register
class RssCollector(Collector):
    """Search the entries currently carried by a set of configured feeds.

    With no `feeds` in its config there is nothing to fetch, so `available()` is
    False and `search` returns `[]`. An unconfigured feed reader is not a broken
    one — it is one nobody has pointed at anything yet, and a run that treated
    that as an error would fail for a reason the operator cannot act on mid-run.
    """

    name: ClassVar[str] = "rss"
    description: ClassVar[str] = "Search entries from configured RSS and Atom feeds."
    requires_credentials: ClassVar[bool] = False

    def search(self, query: str, *, limit: int | None = None) -> list[SourceItem]:
        """Fetch every configured feed and return the entries mentioning `query`.

        Raises:
            CollectorError: If every configured feed failed. One feed failing
                among several is logged and skipped instead: a run grounded in
                the two feeds that answered is worth more than no run at all.
        """
        cap = limit or self.config.limit
        feeds = _feed_urls(self.config)
        if not feeds:
            return []

        found: list[SourceItem] = []
        fetched = 0
        first_failure: CollectorError | None = None
        headers = {"User-Agent": _USER_AGENT, "Accept": _ACCEPT}
        with httpx.Client(
            timeout=self.config.timeout,
            headers=headers,
            follow_redirects=True,
        ) as client:
            for url in feeds:
                try:
                    entries = self._entries(self._fetch(client, url), url)
                except CollectorError as exc:
                    first_failure = first_failure or exc
                    logger.warning("rss: skipping feed %s: %s", url, exc)
                    continue
                fetched += 1
                # Capped per feed as well as overall, so one prolific feed cannot
                # make us hold thousands of items we are about to throw away.
                matched = [item for item in entries if _mentions(query, item)]
                found.extend(matched[:cap])

        if first_failure is not None and fetched == 0:
            raise CollectorError(
                f"every configured feed failed ({len(feeds)} tried); first error: {first_failure}"
            ) from first_failure

        # Newest first across all feeds, undated entries last: when the cap bites,
        # it should bite the stalest entries rather than whichever feed was slowest.
        found.sort(key=lambda item: item.published_at or _UNDATED, reverse=True)
        return found[:cap]

    def available(self) -> bool:
        """False when no feeds are configured — there would be nothing to search."""
        return self.config.enabled and bool(_feed_urls(self.config))

    # --------------------------------------------------------------- internals

    def _fetch(self, client: httpx.Client, url: str) -> bytes:
        """Fetch one feed's raw bytes.

        Raises:
            CollectorError: On transport failure, a refused request, or a body too
                large to plausibly be a feed.
        """
        try:
            response = client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CollectorError(f"could not fetch feed {url}: {exc}") from exc

        body = response.content
        if len(body) > _MAX_FEED_BYTES:
            raise CollectorError(
                f"feed {url} returned {len(body)} bytes, over the {_MAX_FEED_BYTES} byte cap"
            )
        return body

    def _entries(self, body: bytes, feed_url: str) -> list[SourceItem]:
        """Every entry in one feed, in the order the feed listed them.

        Raises:
            CollectorError: If the body is not parseable XML.
        """
        try:
            # Parsed as bytes, not str: the XML declaration names the encoding, and
            # only the byte-level parser is allowed to honour it.
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as exc:
            raise CollectorError(f"feed {feed_url} is not parseable XML: {exc}") from exc

        # A feed is one dialect or the other, so one of these is always empty;
        # asking for both costs nothing and copes with the hybrids in the wild.
        elements = [*root.iterfind(".//item"), *root.iterfind(f".//{_ATOM_ENTRY}")]

        items: list[SourceItem] = []
        for index, element in enumerate(elements):
            try:
                item = self._to_item(element, feed_url, index)
            except Exception as exc:
                # Deliberately broad: the guarantee is that one malformed entry
                # never loses the other twenty-four, and the set of ways a
                # stranger's XML can be malformed is not something we can enumerate.
                logger.debug("rss: skipping entry %d of %s: %s", index, feed_url, exc)
                continue
            if item is not None:
                items.append(item)
        return items

    def _to_item(
        self,
        element: ElementTree.Element,
        feed_url: str,
        index: int,
    ) -> SourceItem | None:
        """One entry as a `SourceItem`, or None if it carries nothing quotable."""
        fields = _atom_fields(element) if element.tag == _ATOM_ENTRY else _rss_fields(element)
        title = _to_text(fields.title or "") or None
        # A link-only entry still states something in its headline, and a headline
        # is the publisher's own words, so it stands in for a body rather than
        # costing us the item.
        text = _to_text(fields.body) or title
        if not text:
            return None

        return SourceItem(
            collector=self.name,
            # `guid`/`id` is the source's own identifier and the right dedup key;
            # the link is the usual stand-in, and the last resort at least stays
            # stable for as long as the feed's ordering does.
            external_id=fields.external_id or fields.url or f"{feed_url}#{index}",
            text=text,
            title=title,
            url=fields.url,
            author=fields.author,
            published_at=fields.published_at,
        )


def _feed_urls(config: CollectorConfig) -> list[str]:
    """The configured feed URLs, defaulting to none.

    Read defensively rather than off `RssConfig`: `CollectorConfig` allows extra
    keys, so a config loaded from a file can carry `feeds` without ever being an
    `RssConfig`, and a stray non-string in the list should cost that one entry
    rather than the collector's whole run.
    """
    raw: object = getattr(config, "feeds", None)
    if isinstance(raw, str):
        raw = [raw]  # one feed written as a bare string is a natural mistake to make
    if not isinstance(raw, list):
        return []
    return [entry.strip() for entry in raw if isinstance(entry, str) and entry.strip()]


def _mentions(query: str, item: SourceItem) -> bool:
    """Whether `query` appears in the entry's title or body, case-insensitively.

    RSS has no search endpoint, so this filtering is client-side by necessity. The
    consequence is worth stating plainly: a feed can only ever surface what it is
    carrying right now, so an absence here is evidence about the feed's window,
    not about the world. An empty query means "everything it currently carries".
    """
    needle = query.strip().lower()
    if not needle:
        return True
    return any(needle in field.lower() for field in (item.title, item.text) if field)


def _rss_fields(element: ElementTree.Element) -> _Fields:
    """The fields of one RSS 2.0 `<item>`."""
    # `content:encoded` carries the full post when a feed offers both; `description`
    # is often only the opening paragraph, and an excerpt can only be checked
    # against text we actually hold.
    body = _child_text(element, _CONTENT_ENCODED) or _child_text(element, "description") or ""
    return _Fields(
        external_id=_clean(_child_text(element, "guid")),
        title=_clean(_child_text(element, "title")),
        body=body,
        url=_clean(_child_text(element, "link")),
        # `author` is defined as an email address and is usually absent; nearly every
        # real feed puts the human name in Dublin Core instead.
        author=_clean(_child_text(element, _DC_CREATOR)) or _clean(_child_text(element, "author")),
        published_at=_rfc822_date(_child_text(element, "pubDate")),
    )


def _atom_fields(element: ElementTree.Element) -> _Fields:
    """The fields of one Atom `<entry>`."""
    # `content` is the body; `summary` is what is left when the publisher keeps the
    # full text behind the link. Both may be escaped HTML or inline XHTML, and
    # `_child_text` flattens either into the same words.
    body = _child_text(element, _ATOM_CONTENT) or _child_text(element, _ATOM_SUMMARY) or ""
    return _Fields(
        external_id=_clean(_child_text(element, _ATOM_ID)),
        title=_clean(_child_text(element, _ATOM_TITLE)),
        body=body,
        url=_atom_link(element),
        author=_clean(_child_text(element, _ATOM_AUTHOR_NAME)),
        # `published` is when it was written and `updated` when it last changed;
        # prefer the former, but many feeds only ever set the latter.
        published_at=(
            _iso_date(_child_text(element, _ATOM_PUBLISHED))
            or _iso_date(_child_text(element, _ATOM_UPDATED))
        ),
    )


def _atom_link(element: ElementTree.Element) -> str | None:
    """The entry's human-readable link.

    Atom puts the URL in an attribute rather than the element's text, and an entry
    may carry several: `alternate` is the page a reader should be sent to, while
    the others point at replies, enclosures or the entry's own feed.
    """
    fallback: str | None = None
    for link in element.iterfind(_ATOM_LINK):
        href = _clean(link.get("href"))
        if href is None:
            continue
        if link.get("rel", "alternate") == "alternate":
            return href
        fallback = fallback or href
    return fallback


def _child_text(element: ElementTree.Element, tag: str) -> str | None:
    """All text under the first `tag` child, nested markup included.

    `findtext` would return only the element's own leading text, which silently
    empties an Atom `content` written as inline XHTML.
    """
    child = element.find(tag)
    if child is None:
        return None
    return "".join(child.itertext())


def _to_text(raw: str) -> str:
    """Feed markup as readable words.

    Tags are dropped and whitespace regularised, but not one word is changed: this
    is the text an excerpt will later be checked against, so the conversion has to
    be the only transformation the body ever undergoes.
    """
    if not raw.strip():
        return ""
    text = _BLOCK_TAG_RE.sub("\n", raw)
    text = _TAG_RE.sub("", text)
    # Unescaped *after* tags are stripped: doing it first would turn an escaped
    # `&lt;div&gt;` in someone's code sample into a tag and eat the rest of the line.
    text = html.unescape(text)
    text = _INLINE_SPACE_RE.sub(" ", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()[:_MAX_TEXT_CHARS]


def _clean(value: str | None) -> str | None:
    """Strip a field, treating blank as absent."""
    if value is None:
        return None
    return value.strip() or None


def _rfc822_date(raw: str | None) -> datetime | None:
    """An RSS `pubDate` as an aware UTC datetime, or None if it is unusable.

    An unparseable date is never a reason to drop an entry: the words are the
    evidence and the timestamp is only metadata about them.
    """
    if raw is None or not raw.strip():
        return None
    try:
        return _as_utc(parsedate_to_datetime(raw.strip()))
    except (TypeError, ValueError):
        return None


def _iso_date(raw: str | None) -> datetime | None:
    """An Atom ISO 8601 date as an aware UTC datetime, or None if it is unusable."""
    if raw is None or not raw.strip():
        return None
    try:
        return _as_utc(datetime.fromisoformat(raw.strip()))
    except ValueError:
        return None


def _as_utc(value: datetime) -> datetime:
    """Normalise to timezone-aware UTC.

    Naive timestamps do occur — RFC 822 dates ending in `-0000`, Atom dates with
    the offset left off — and the Evidence model downstream rejects naive
    datetimes outright. Assuming UTC keeps an otherwise good date rather than
    discarding it, at the cost of a few hours' error on a badly published feed.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = ["RssCollector", "RssConfig"]
