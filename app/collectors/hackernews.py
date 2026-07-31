"""Collector for Hacker News, via the public Algolia search API.

HN is where practitioners say out loud what they will not put in a bug report:
what they tried, what it cost them, and what they switched to. That makes it a
good counterweight to an issue tracker, which only ever sees users who stayed.

Two shapes come back from the same endpoint and both are wanted. A comment
(`tags=comment`) carries `comment_text`; a story (`tags=story`) carries a title,
sometimes `story_text`, and often only an outbound link. They are fetched
separately because the tag is a filter, not a facet, then interleaved so neither
shape crowds the other out of the caller's limit.
"""

import html
import re
from datetime import UTC, datetime
from itertools import zip_longest
from typing import Any, ClassVar

import httpx

from app.collectors.base import Collector, SourceItem, register, short_body
from app.utils.errors import CollectorError, RateLimitedError

_API_URL = "https://hn.algolia.com/api/v1/search"
_USER_AGENT = "opportunity-engine/0.1 (+https://github.com/)"

# The HN permalink, not the story's outbound link: the permalink is where a reader
# can check that this text was posted, by this author, at this time.
_PERMALINK = "https://news.ycombinator.com/item?id={object_id}"

# Algolia allows up to 1000 per page; this collector's own limit tops out at 200,
# so one request per tag is always enough.
_MAX_HITS_PER_PAGE = 200

# Comments are short, but an Ask HN body is not always. Cut generously, and only at
# a line or sentence boundary — see `_bounded`.
_MAX_TEXT_CHARS = 20_000

# HN's markup vocabulary is tiny: paragraphs, line breaks, links, italics, and
# `<pre><code>` blocks. Paragraph and break tags carry structure worth keeping as
# whitespace; everything else is presentation and goes.
_PARAGRAPH_RE = re.compile(r"</?p\s*/?>", re.IGNORECASE)
_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


@register
class HackerNewsCollector(Collector):
    """Search Hacker News comments and stories for first-hand accounts."""

    name: ClassVar[str] = "hacker-news"
    description: ClassVar[str] = "Search Hacker News comments and stories via Algolia."

    def search(self, query: str, *, limit: int | None = None) -> list[SourceItem]:
        """Search comments and stories for `query`, with HN's markup stripped off.

        Raises:
            CollectorError: On transport failure, a refused request, or a response
                that cannot be understood.
        """
        cleaned = query.strip()
        # The API would happily answer a blank query with the site firehose, which
        # is not a search result and would poison the run with unrelated evidence.
        if not cleaned:
            raise CollectorError("Hacker News search needs a non-empty query")

        wanted = limit if limit is not None else self.config.limit
        headers = {"User-Agent": _USER_AGENT}
        with httpx.Client(timeout=self.config.timeout, headers=headers) as client:
            # Each tag is asked for the full limit, not half of it: hits whose text
            # is empty get dropped, and one shape being thin should not shrink the
            # result when the other has plenty to give.
            comments = self._search_tag(client, cleaned, tag="comment", wanted=wanted)
            stories = self._search_tag(client, cleaned, tag="story", wanted=wanted)
        return _interleave(comments, stories)[:wanted]

    def _search_tag(
        self,
        client: httpx.Client,
        query: str,
        *,
        tag: str,
        wanted: int,
    ) -> list[SourceItem]:
        """Run one tag-filtered search and parse what it returned."""
        payload = self._fetch(client, query, tag=tag, wanted=wanted)
        hits = payload.get("hits")
        if hits is None:
            raise CollectorError(
                f"Hacker News search returned no 'hits' array: {short_body(payload)}",
            )
        if not isinstance(hits, list):
            raise CollectorError("Hacker News search returned a non-list 'hits' field")

        found: list[SourceItem] = []
        for hit in hits:
            if len(found) >= wanted:
                break
            item = self._to_item(hit)
            if item is not None:
                found.append(item)
        return found

    def _fetch(
        self,
        client: httpx.Client,
        query: str,
        *,
        tag: str,
        wanted: int,
    ) -> dict[str, Any]:
        """Run one search request and return the decoded payload."""
        params: dict[str, str | int] = {
            "query": query,
            "tags": tag,
            "hitsPerPage": min(wanted, _MAX_HITS_PER_PAGE),
        }
        try:
            response = client.get(_API_URL, params=params)
        except httpx.HTTPError as exc:
            raise CollectorError(f"Hacker News search could not be reached: {exc}") from exc

        if response.status_code == 429:
            raise RateLimitedError(
                "Hacker News search rate limit reached (Algolia allows roughly "
                "10,000 requests an hour); retry later or lower the collector limit.",
            )
        if response.is_error:
            raise CollectorError(
                f"Hacker News search refused the request "
                f"(HTTP {response.status_code}): {short_body(response.text)}",
            )

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise CollectorError("Hacker News search returned a body that is not JSON") from exc
        if not isinstance(payload, dict):
            raise CollectorError(
                f"Hacker News search returned {type(payload).__name__}, expected an object",
            )
        return payload

    def _to_item(self, hit: Any) -> SourceItem | None:
        """Convert one hit of either shape, or return None if it is not quotable."""
        if not isinstance(hit, dict):
            return None

        object_id = _identifier(hit.get("objectID"))
        if object_id is None:
            return None

        own_title = _string(hit.get("title"))
        text = _readable(hit.get("comment_text")) or _readable(hit.get("story_text"))
        if not text:
            # A link submission has no body, so its title *is* the whole of what the
            # submitter wrote — still their words, still checkable at the permalink.
            # (A comment never has a title, so this never rescues a bodyless comment.)
            text = _readable(own_title)
        if not text:
            return None

        try:
            return SourceItem(
                collector=self.name,
                external_id=object_id,
                text=text,
                # A comment has no title of its own; the thread it sits in is the
                # nearest thing to one and tells a reader what it is answering.
                title=own_title or _string(hit.get("story_title")),
                url=_PERMALINK.format(object_id=object_id),
                author=_string(hit.get("author")),
                published_at=_timestamp(hit.get("created_at_i")),
            )
        except ValueError:
            # One unusable record must not cost us the rest of the page.
            return None


def _interleave(comments: list[SourceItem], stories: list[SourceItem]) -> list[SourceItem]:
    """Merge two ranked lists round-robin, best of each first.

    Relevance scores from two separate queries are not comparable, and sorting the
    union by date or points would hand the whole limit to whichever shape happens
    to score higher. Alternating preserves each list's own ranking and guarantees
    both shapes survive the caller's cut.

    Ids are deduplicated even though the two tags are disjoint today: the cost is a
    set, and the cost of being wrong is the same post cited twice as two sources.
    """
    merged: list[SourceItem] = []
    seen: set[str] = set()
    for comment, story in zip_longest(comments, stories):
        for item in (comment, story):
            if item is not None and item.external_id not in seen:
                seen.add(item.external_id)
                merged.append(item)
    return merged


def _readable(value: Any) -> str:
    """Strip HN's markup down to the plain text a human would quote.

    Tags are removed *before* entities are unescaped, deliberately: a comment about
    HTML contains `&lt;div&gt;`, and unescaping first would turn the author's own
    words into a tag and then delete them. The only other change is that paragraph
    and break tags become the whitespace they render as, because HN uses them as
    its sole structure. Nothing is reordered, summarised or rewritten, so what is
    left is the author's words verbatim — which is what the excerpt check needs.
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    text = _PARAGRAPH_RE.sub("\n\n", value)
    text = _BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    return _bounded(html.unescape(text).strip())


def _identifier(value: Any) -> str | None:
    """The hit's own id, which is also its permalink id."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _string(value: Any) -> str | None:
    """A non-blank string field, or None when the API sent something else."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def _timestamp(value: Any) -> datetime | None:
    """Read `created_at_i` (unix seconds) as timezone-aware UTC.

    Aware because the Evidence model downstream rejects naive datetimes.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _bounded(text: str) -> str:
    """Cap a runaway body at a whole line or sentence.

    A prefix is still verbatim, so an excerpt taken from it still verifies; cutting
    mid-sentence would later be quoted back as though the author stopped there.
    """
    if len(text) <= _MAX_TEXT_CHARS:
        return text
    head = text[:_MAX_TEXT_CHARS]
    cut = max(head.rfind("\n"), head.rfind(". "))
    # Only honour a boundary in the back half; a cut near the start would throw away
    # most of what the author wrote.
    return head[: cut + 1] if cut > _MAX_TEXT_CHARS // 2 else head


__all__ = ["HackerNewsCollector"]
