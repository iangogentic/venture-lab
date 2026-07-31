"""Collector for Apple App Store customer reviews.

The 1- and 2-star reviews of incumbent apps are among the cheapest pre-MVP
demand evidence there is: each one is a real customer who paid, or at least
installed, and then took the time to describe in their own words what the
incumbent fails to do. Nobody writes an angry review about a category they do
not care about. Both endpoints used here are Apple's own public interface — the
iTunes Search API to find the incumbents, and the per-app customer-reviews feed
(roughly the 500 most recent reviews per country) to read what their users say —
official, stable, and keyless.

Google Play is deliberately not collected. Google publishes no equivalent
public reviews feed, the community scraper library is dormant, and scraping the
store is ToS-gray; a source that can silently break or answer with a block page
is worth less than no source at all.

A query costs one search request plus one feed request per candidate app per
country, so the candidate list is capped tightly (`_MAX_APPS_PER_QUERY`). Review
bodies arrive as plain text; their whitespace is regularised once, here, and the
wording never touched, because `collect-evidence` later checks its excerpts
against `SourceItem.text` as literal substrings.
"""

import re
from datetime import UTC, datetime
from typing import Any, ClassVar, Final

import httpx

from app.collectors.base import Collector, CollectorConfig, SourceItem, register, short_body
from app.utils.errors import CollectorError
from app.utils.logging import get_logger

logger = get_logger(__name__)

_USER_AGENT: Final = "opportunity-engine/0.1 (+https://github.com/)"

_SEARCH_URL: Final = "https://itunes.apple.com/search"
_FEED_URL: Final = (
    "https://itunes.apple.com/{country}/rss/customerreviews/id={track_id}/sortBy=mostRecent/json"
)

# The app's store page, not the feed: it is where a human reader can see the
# app and scroll to its reviews, which is what a citation is for.
_APP_PAGE_URL: Final = "https://apps.apple.com/{country}/app/id{track_id}"

# Each candidate app costs another request (its review feed, once per country),
# and for a product-shaped query the incumbents whose reviews matter are the top
# few hits anyway; asking for more would spend the budget on long-tail clones.
_MAX_APPS_PER_QUERY: Final = 5

_DEFAULT_COUNTRIES: Final = ("us",)

# ISO 3166's two-letter shape. Matching it also keeps arbitrary text out of the
# URL path a country code is spliced into.
_COUNTRY_RE: Final = re.compile(r"^[a-z]{2}$")

# Apple's own review composer caps a body far below this, so the bound exists
# only so a malformed feed cannot blow the downstream context budget. Cutting is
# a prefix cut: what survives is still exactly what the customer wrote, so an
# excerpt taken from it still checks out.
_MAX_TEXT_CHARS: Final = 10_000

# Ordering floor for reviews whose date could not be parsed, so they sort last.
_UNDATED: Final = datetime(1970, 1, 1, tzinfo=UTC)

_INLINE_SPACE_RE: Final = re.compile(r"[^\S\n]+")
_BLANK_RUN_RE: Final = re.compile(r"\n\s*\n\s*")


@register
class AppStoreReviewsCollector(Collector):
    """Find apps matching a query and collect their customers' recent reviews."""

    name: ClassVar[str] = "app-reviews"
    description: ClassVar[str] = "Customer reviews of the App Store apps matching the query."
    requires_credentials: ClassVar[bool] = False

    def __init__(
        self,
        config: CollectorConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        # A test seam, nothing more: `httpx.MockTransport` slots in here so the
        # tests can serve canned Apple payloads without touching the network.
        # None means httpx's default transport, so production is unchanged.
        self._transport = transport

    def search(self, query: str, *, limit: int | None = None) -> list[SourceItem]:
        """Search the App Store for `query` and return the matching apps' reviews.

        Raises:
            CollectorError: If the App Store search itself failed, or every
                review feed did. One feed failing among several is logged and
                skipped instead: the reviews of the apps that answered are worth
                more than no run at all. Never on an empty result.
        """
        cap = limit or self.config.limit
        cleaned = query.strip()
        if not cleaned:
            # An empty term is not a question about any product. Nothing was
            # asked, so nothing was found.
            return []

        countries = _countries(self.config)
        headers = {"User-Agent": _USER_AGENT}
        with httpx.Client(
            timeout=self.config.timeout,
            headers=headers,
            transport=self._transport,
        ) as client:
            # One search, against the primary storefront: the search index is
            # broadly shared across countries, but reviews are strictly
            # per-country, so each candidate found here fans out into one feed
            # fetch per configured country.
            apps = self._search_apps(client, cleaned, countries[0])

            found: list[SourceItem] = []
            fetched = 0
            first_failure: CollectorError | None = None
            for track_id, app_name in apps:
                for country in countries:
                    try:
                        reviews = self._reviews(client, track_id, app_name, country)
                    except CollectorError as exc:
                        first_failure = first_failure or exc
                        logger.warning(
                            "app-reviews: skipping reviews of %r (id%d, %s): %s",
                            app_name,
                            track_id,
                            country,
                            exc,
                        )
                        continue
                    fetched += 1
                    # Capped per feed as well as overall, so one prolific app
                    # cannot make us hold hundreds of reviews we are about to
                    # throw away.
                    found.extend(reviews[:cap])

        if first_failure is not None and fetched == 0:
            raise CollectorError(
                f"every review feed failed ({len(apps) * len(countries)} tried); "
                f"first error: {first_failure}"
            ) from first_failure

        # Newest first across apps, undated reviews last: when the cap bites, it
        # should bite the stalest complaints rather than whichever app happened
        # to be listed first.
        found.sort(key=lambda item: item.published_at or _UNDATED, reverse=True)
        return found[:cap]

    # --------------------------------------------------------------- internals

    def _search_apps(
        self,
        client: httpx.Client,
        query: str,
        country: str,
    ) -> list[tuple[int, str]]:
        """The top App Store hits for `query`, as `(trackId, trackName)` pairs.

        An empty list is a legitimate answer — no incumbent sells software here,
        which is itself a finding about the market.

        Raises:
            CollectorError: On transport failure, a refused request, or a
                response that cannot be understood.
        """
        params: dict[str, str | int] = {
            "term": query,
            "entity": "software",
            "limit": _MAX_APPS_PER_QUERY,
            "country": country,
        }
        payload = self._get_json(client, _SEARCH_URL, params=params, what="searching for apps")
        results = payload.get("results")
        if not isinstance(results, list):
            raise CollectorError(
                f"App Store search returned no 'results' array: {short_body(payload)}"
            )

        apps: list[tuple[int, str]] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            track_id = result.get("trackId")
            if not isinstance(track_id, int) or isinstance(track_id, bool):
                # Without an id there is no feed to fetch and no page to cite.
                continue
            name = result.get("trackName")
            app_name = name.strip() if isinstance(name, str) and name.strip() else f"app {track_id}"
            apps.append((track_id, app_name))
        # The `limit` param already asks for this many; slicing again costs
        # nothing and keeps the request budget honest if the API ignores it.
        return apps[:_MAX_APPS_PER_QUERY]

    def _reviews(
        self,
        client: httpx.Client,
        track_id: int,
        app_name: str,
        country: str,
    ) -> list[SourceItem]:
        """One app's recent reviews in one country's storefront.

        Raises:
            CollectorError: On transport failure, a refused request, or a feed
                that cannot be understood. Never for an app nobody has reviewed.
        """
        url = _FEED_URL.format(country=country, track_id=track_id)
        payload = self._get_json(client, url, what=f"reading reviews of {app_name!r} ({country})")
        feed = payload.get("feed")
        if not isinstance(feed, dict):
            raise CollectorError(f"review feed for {app_name!r} ({country}) has no 'feed' object")

        entries = feed.get("entry")
        if entries is None:
            # An app nobody has reviewed in this storefront is a finding, not a failure.
            return []
        if isinstance(entries, dict):
            # Apple collapses a single-element `entry` array to a bare object.
            entries = [entries]
        if not isinstance(entries, list):
            raise CollectorError(
                f"review feed for {app_name!r} ({country}) has an unreadable 'entry' field"
            )

        app_url = _APP_PAGE_URL.format(country=country, track_id=track_id)
        items: list[SourceItem] = []
        for index, entry in enumerate(entries):
            item = self._to_item(
                entry,
                track_id=track_id,
                app_name=app_name,
                app_url=app_url,
                index=index,
            )
            if item is not None:
                items.append(item)
        return items

    def _to_item(
        self,
        entry: Any,
        *,
        track_id: int,
        app_name: str,
        app_url: str,
        index: int,
    ) -> SourceItem | None:
        """One feed entry as a `SourceItem`, or None if it is not a quotable review.

        Returns None rather than raising: one malformed entry must not cost the
        run the other forty-nine.
        """
        if not isinstance(entry, dict):
            return None
        # The feed's first entry is the app's own metadata, not a review; the
        # rating is what tells the two shapes apart, so anything unrated is
        # skipped rather than quoted as if a customer had said it.
        rating = _label(entry.get("im:rating"))
        if rating is None:
            return None

        text = _normalise(_label(entry.get("content")) or "")
        if not text:
            logger.debug("app-reviews: skipping bodyless review %d of %r", index, app_name)
            return None

        review_title = _label(entry.get("title"))
        author = entry.get("author")
        author_name = _label(author.get("name")) if isinstance(author, dict) else None

        try:
            return SourceItem(
                collector=self.name,
                external_id=_label(entry.get("id")) or f"{track_id}:{index}",
                text=text,
                # The stage only ever reads items, not feeds: naming the app and
                # the rating in the title tells it which incumbent this is and
                # how angry the customer was, without touching the verbatim body
                # the excerpt check depends on.
                title=(
                    f"{review_title} — {app_name} ({rating}/5)"
                    if review_title
                    else f"{app_name} ({rating}/5)"
                ),
                url=app_url,
                author=author_name,
                published_at=_updated_at(_label(entry.get("updated"))),
            )
        except ValueError as exc:
            logger.debug("app-reviews: skipping unusable review %d of %r: %s", index, app_name, exc)
            return None

    def _get_json(
        self,
        client: httpx.Client,
        url: str,
        *,
        what: str,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any]:
        """Run one GET and return the decoded object.

        Raises:
            CollectorError: On transport failure, an error status, or a body
                that is not a JSON object.
        """
        try:
            response = client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise CollectorError(f"could not reach the App Store while {what}: {exc}") from exc

        if response.is_error:
            raise CollectorError(
                f"the App Store refused the request while {what} "
                f"(HTTP {response.status_code}): {short_body(response.text)}"
            )

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise CollectorError(f"the App Store answered {what} with a non-JSON body") from exc
        if not isinstance(payload, dict):
            raise CollectorError(
                f"the App Store answered {what} with {type(payload).__name__}, expected an object"
            )
        return payload


def _countries(config: CollectorConfig) -> list[str]:
    """The configured storefront country codes, defaulting to the US store.

    Read defensively rather than off a typed subclass: `CollectorConfig` allows
    extra keys, so a config loaded from a file can carry `app_store_countries`
    without ever being anything more specific, and a stray non-string in the
    list should cost that one entry rather than the collector's whole run.
    """
    raw: object = getattr(config, "app_store_countries", None)
    if isinstance(raw, str):
        raw = [raw]  # one country written as a bare string is a natural mistake to make
    if not isinstance(raw, list):
        return list(_DEFAULT_COUNTRIES)
    cleaned: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        code = entry.strip().lower()
        if not _COUNTRY_RE.match(code):
            logger.debug("app-reviews: ignoring unusable country code %r", entry)
            continue
        if code not in cleaned:
            cleaned.append(code)
    return cleaned or list(_DEFAULT_COUNTRIES)


def _label(value: object) -> str | None:
    """The `label` string Apple wraps every scalar in, or None if it is absent.

    The feed spells `"five stars"` as `{"im:rating": {"label": "5"}}`; every
    field this collector reads is one of these wrappers, so unwrapping lives in
    one place and a missing or malformed wrapper reads as an absent field.
    """
    if isinstance(value, dict):
        label = value.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    return None


def _normalise(raw: str) -> str:
    """A review body with its whitespace regularised and nothing else changed.

    Reviews arrive as plain text, so unlike the feed collectors there is no
    markup to strip. Runs of spaces and blank lines are collapsed and the length
    bounded — a prefix cut, so what survives is still exactly what the customer
    wrote — and not one word is altered, because `collect-evidence` later checks
    its excerpts against this text as literal substrings.
    """
    text = _INLINE_SPACE_RE.sub(" ", raw)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()[:_MAX_TEXT_CHARS]


def _updated_at(raw: str | None) -> datetime | None:
    """A review's `updated` label as an aware UTC datetime, or None if unusable.

    An unparseable date is never a reason to drop a review: the words are the
    evidence and the timestamp is only metadata about them.
    """
    if raw is None or not raw.strip():
        return None
    try:
        return _as_utc(datetime.fromisoformat(raw.strip()))
    except ValueError:
        return None


def _as_utc(value: datetime) -> datetime:
    """Normalise to timezone-aware UTC.

    Apple stamps reviews with a Pacific-time offset today, but that is an
    observation, not a contract — and the Evidence model downstream rejects
    naive datetimes outright, so a date with the offset left off is read as UTC
    rather than discarded.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = ["AppStoreReviewsCollector"]
