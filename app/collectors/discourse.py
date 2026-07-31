"""Collector for Discourse forums, via each forum's public JSON interface.

Nearly every developer tool and SaaS product of any size runs its support forum
on Discourse — OpenAI, Cloudflare, Rust, Docker, Grafana, Stripe's old forum —
which makes the platform a single integration that unlocks hundreds of
communities where users describe, in their own words, what a product does to
them. That is precisely the material this pipeline exists to ground itself in.

The endpoints used here are not scraping. Appending `.json` to a Discourse URL
is the platform's designed anonymous read interface — `/search.json` and
`/t/{id}.json` are the very calls the Discourse frontend itself makes — so this
collector consumes each forum exactly the way its own pages do, minus the
rendering.

A search hit names a topic; the quotable words live in the topic's opening
post, which costs a second request to fetch. Those follow-up requests are
capped hard, because every one of them lands on someone else's server. When a
topic fetch fails, the search response's own `blurb` — the engine-served
snippet built around the match — stands in for the post, so a fetch hiccup
costs fidelity, not the hit.
"""

import html
import re
from datetime import UTC, datetime
from typing import Any, ClassVar, Final, NamedTuple
from urllib.parse import urlsplit

import httpx

from app.collectors.base import Collector, CollectorConfig, SourceItem, register
from app.utils.errors import CollectorError
from app.utils.logging import get_logger

logger = get_logger(__name__)

_USER_AGENT: Final = "opportunity-engine/0.1 (+https://github.com/)"

# How many topics per forum, per query, get their opening post fetched. Each one
# is another request to someone else's server, so the ceiling is deliberately
# modest: six well-grounded posts per forum beat thirty rude requests, and the
# newest-first merge across forums still fills the caller's limit.
_MAX_TOPIC_FETCHES_PER_FORUM: Final = 6

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

# Generous ceiling on one post's text. Cutting is a prefix cut and never happens
# mid-word-choice: what survives is still exactly what the author wrote, so an
# excerpt taken from it still checks out. Only material past the cap is lost.
_MAX_TEXT_CHARS: Final = 20_000
# Ordering floor for topics whose date could not be parsed, so they sort last.
_UNDATED: Final = datetime(1970, 1, 1, tzinfo=UTC)


class _Blurb(NamedTuple):
    """The search engine's own snippet for one matched topic."""

    text: str
    username: str | None


class _Post(NamedTuple):
    """A topic's opening post, once fetched and stripped."""

    text: str
    username: str | None
    created_at: datetime | None


@register
class DiscourseCollector(Collector):
    """Search configured Discourse forums for topics and quote their opening posts.

    With no `discourse_forums` in its config there is nothing to search, so
    `available()` is False and `search` returns `[]`. An unconfigured forum
    list is not an error — it is a collector nobody has pointed at anything
    yet, and a run that treated that as a failure would fail for a reason the
    operator cannot act on mid-run.
    """

    name: ClassVar[str] = "discourse"
    description: ClassVar[str] = "Search configured Discourse forums for first-hand reports."

    # Discourse's JSON interface is anonymous by design; a key exists only for
    # admin write access, which this collector will never want.
    requires_credentials: ClassVar[bool] = False

    def __init__(
        self,
        config: CollectorConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Optionally accept an httpx transport, so tests can serve canned forums.

        `None` — the production default — leaves httpx to build its usual
        network transport; the seam changes nothing unless a caller injects a
        `MockTransport`.
        """
        super().__init__(config)
        self._transport = transport

    def search(self, query: str, *, limit: int | None = None) -> list[SourceItem]:
        """Search every configured forum for `query` and quote the topics found.

        Raises:
            CollectorError: If `query` is blank, or if every configured forum
                failed. One forum failing among several is logged and skipped
                instead: a run grounded in the forums that answered is worth
                more than no run at all.
        """
        cleaned = query.strip()
        # Discourse refuses a blank search with an HTTP 400; failing here says
        # why more usefully than a per-forum transport error would.
        if not cleaned:
            raise CollectorError("Discourse search needs a non-empty query")

        cap = limit or self.config.limit
        forums = _forum_urls(self.config)
        if not forums:
            return []

        found: list[SourceItem] = []
        searched = 0
        first_failure: CollectorError | None = None
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
        with httpx.Client(
            timeout=self.config.timeout,
            headers=headers,
            follow_redirects=True,
            transport=self._transport,
        ) as client:
            for base in forums:
                try:
                    found.extend(self._search_forum(client, base, cleaned, cap))
                except CollectorError as exc:
                    first_failure = first_failure or exc
                    logger.warning("discourse: skipping forum %s: %s", base, exc)
                    continue
                searched += 1

        if first_failure is not None and searched == 0:
            raise CollectorError(
                f"every configured forum failed ({len(forums)} tried); first error: {first_failure}"
            ) from first_failure

        # Newest first across all forums, undated topics last: when the cap
        # bites, it should bite the stalest topics rather than whichever forum
        # happened to be listed first.
        found.sort(key=lambda item: item.published_at or _UNDATED, reverse=True)
        return found[:cap]

    def available(self) -> bool:
        """False when no forums are configured — there would be nothing to search."""
        return self.config.enabled and bool(_forum_urls(self.config))

    # --------------------------------------------------------------- internals

    def _search_forum(
        self,
        client: httpx.Client,
        base: str,
        query: str,
        cap: int,
    ) -> list[SourceItem]:
        """One forum's answer to `query`, opening posts fetched and stripped.

        Raises:
            CollectorError: On transport failure, a refused request, or a search
                payload that cannot be understood. A topic-detail failure is
                absorbed instead — see `_opening_post`.
        """
        payload = self._fetch_json(client, f"{base}/search.json", params={"q": query})
        # A search that matched nothing may omit `topics` entirely rather than
        # send an empty array; both mean the same finding, not a failure.
        topics = payload.get("topics") or []
        if not isinstance(topics, list):
            raise CollectorError(f"search on {base} returned a non-list 'topics' field")
        blurbs = _blurbs_by_topic(payload.get("posts"))

        items: list[SourceItem] = []
        for topic in topics[: min(_MAX_TOPIC_FETCHES_PER_FORUM, cap)]:
            item = self._to_item(client, base, topic, blurbs)
            if item is not None:
                items.append(item)
        return items

    def _to_item(
        self,
        client: httpx.Client,
        base: str,
        topic: Any,
        blurbs: dict[int, _Blurb],
    ) -> SourceItem | None:
        """One search hit as a `SourceItem`, or None if it carries nothing quotable."""
        if not isinstance(topic, dict):
            return None
        topic_id = topic.get("id")
        if not isinstance(topic_id, int) or isinstance(topic_id, bool):
            logger.debug("discourse: skipping a %s hit with no topic id", base)
            return None

        host = urlsplit(base).netloc or base
        blurb = blurbs.get(topic_id)
        post = self._opening_post(client, base, topic_id)
        if post is not None:
            text: str | None = post.text
            author = post.username
        elif blurb is not None:
            # The blurb is the engine-served snippet built around the match —
            # the forum's own rendering of the matched words — so quoting it is
            # still quoting the source, just less of it than the full post.
            text = _to_text(blurb.text) or None
            author = blurb.username
        else:
            text = None
            author = None
        if not text:
            logger.debug("discourse: dropping topic %s/t/%s: no post and no blurb", host, topic_id)
            return None

        slug = _string(topic.get("slug"))
        try:
            return SourceItem(
                collector=self.name,
                # Host plus topic id: stable for as long as the topic exists,
                # and unambiguous across forums in a multi-forum run.
                external_id=f"{host}/t/{topic_id}",
                text=text,
                title=_string(topic.get("title")),
                # `/t/{id}` alone also resolves — Discourse redirects to the
                # slugged form — so a hit without a slug still gets a URL.
                url=f"{base}/t/{slug}/{topic_id}" if slug else f"{base}/t/{topic_id}",
                author=author,
                published_at=_iso_date(topic.get("created_at"))
                or (post.created_at if post else None),
            )
        except ValueError:
            # One unusable record must not cost us the rest of the forum.
            return None

    def _opening_post(self, client: httpx.Client, base: str, topic_id: int) -> _Post | None:
        """The topic's first post, stripped, or None when it could not be had.

        Failure here is absorbed rather than raised on purpose: the search
        already proved the topic exists and left us its blurb, so a fetch
        hiccup should downgrade the hit to the snippet, never lose it.
        """
        try:
            payload = self._fetch_json(client, f"{base}/t/{topic_id}.json", params=None)
        except CollectorError as exc:
            logger.debug("discourse: could not fetch topic %s on %s: %s", topic_id, base, exc)
            return None

        stream = payload.get("post_stream")
        posts = stream.get("posts") if isinstance(stream, dict) else None
        # The stream is served in topic order, so the first entry is the post
        # that opened the topic — the pain statement the search matched.
        first = posts[0] if isinstance(posts, list) and posts else None
        if not isinstance(first, dict):
            logger.debug("discourse: topic %s on %s has no readable post stream", topic_id, base)
            return None

        cooked = first.get("cooked")
        text = _to_text(cooked) if isinstance(cooked, str) else ""
        if not text:
            return None
        return _Post(
            text=text,
            username=_string(first.get("username")),
            created_at=_iso_date(first.get("created_at")),
        )

    def _fetch_json(
        self,
        client: httpx.Client,
        url: str,
        *,
        params: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Fetch one endpoint and return the decoded payload.

        Raises:
            CollectorError: On transport failure, a refused request, or a body
                that is not the JSON object Discourse documents.
        """
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CollectorError(f"could not fetch {url}: {exc}") from exc

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise CollectorError(f"{url} returned a body that is not JSON") from exc
        if not isinstance(payload, dict):
            raise CollectorError(f"{url} returned {type(payload).__name__}, expected an object")
        return payload


def _forum_urls(config: CollectorConfig) -> list[str]:
    """The configured forum base URLs, defaulting to none.

    Read defensively rather than off a config subclass: `CollectorConfig`
    allows extra keys, so a config loaded from a file can carry
    `discourse_forums` without ever being a dedicated subtype, and a stray
    non-string in the list should cost that one entry rather than the
    collector's whole run. Trailing slashes are shed so path joins stay clean.
    """
    raw: object = getattr(config, "discourse_forums", None)
    if isinstance(raw, str):
        raw = [raw]  # one forum written as a bare string is a natural mistake to make
    if not isinstance(raw, list):
        return []
    return [entry.strip().rstrip("/") for entry in raw if isinstance(entry, str) and entry.strip()]


def _blurbs_by_topic(raw: object) -> dict[int, _Blurb]:
    """Each matched topic's engine-served snippet, keyed by topic id.

    A `/search.json` response pairs its `topics` array with a `posts` array,
    one row per matching post, each carrying the `blurb` the engine built
    around the match. Held before any topic is fetched, so the fallback is
    already in hand when a fetch fails.
    """
    if not isinstance(raw, list):
        return {}
    blurbs: dict[int, _Blurb] = {}
    for post in raw:
        if not isinstance(post, dict):
            continue
        topic_id = post.get("topic_id")
        blurb = post.get("blurb")
        if not isinstance(topic_id, int) or isinstance(topic_id, bool):
            continue
        if not isinstance(blurb, str) or not blurb.strip():
            continue
        # The first row per topic wins: search ranks the best-matching post
        # first, and that is the snippet worth quoting.
        blurbs.setdefault(topic_id, _Blurb(text=blurb, username=_string(post.get("username"))))
    return blurbs


def _to_text(raw: str) -> str:
    """Discourse's `cooked` HTML as readable words.

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


def _string(value: Any) -> str | None:
    """A non-blank string field, or None when the API sent something else."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def _iso_date(raw: Any) -> datetime | None:
    """A Discourse ISO 8601 stamp as an aware UTC datetime, or None if unusable.

    An unparseable date is never a reason to drop a topic: the words are the
    evidence and the timestamp is only metadata about them.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return _as_utc(datetime.fromisoformat(raw.strip()))
    except ValueError:
        return None


def _as_utc(value: datetime) -> datetime:
    """Normalise to timezone-aware UTC.

    Discourse stamps end in `Z`, but a self-hosted forum can be configured
    oddly, and the Evidence model downstream rejects naive datetimes outright.
    Assuming UTC keeps an otherwise good date rather than discarding it.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = ["DiscourseCollector"]
