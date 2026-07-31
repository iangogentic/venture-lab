"""App Store reviews collector: canned Apple payloads through the transport seam.

Nothing here touches the network. `httpx.MockTransport` slots into the
collector's transport seam and serves the two payload shapes Apple actually
returns — an iTunes Search result and per-app customer-review feeds — so every
test is deterministic and polite to the source.
"""

from datetime import UTC, datetime
from typing import Any, Final

import httpx
import pytest

from app.collectors.app_reviews import AppStoreReviewsCollector
from app.collectors.base import CollectorConfig
from app.utils.errors import CollectorError

# ---------------------------------------------------------------- canned data

_SEARCH_TWO_APPS: Final[dict[str, Any]] = {
    "resultCount": 2,
    "results": [
        {"kind": "software", "trackId": 111, "trackName": "TaskPal"},
        {"kind": "software", "trackId": 222, "trackName": "FocusFlow"},
    ],
}

_NO_RESULTS: Final[dict[str, Any]] = {"resultCount": 0, "results": []}

_CRASH_BODY: Final = (
    "The app crashes every time I sync with my calendar. Support never replied to three tickets."
)
_TRIAL_BODY: Final = (
    "Signed up for the free trial and was charged for a full year anyway. "
    "There is no way to cancel from inside the app."
)
_SYNC_BODY: Final = "Sync quietly stopped working after the last update."
_TIMER_BODY: Final = "Timer resets itself whenever the phone locks. Useless for actual work."
_PRAISE_BODY: Final = "Does exactly what it promises."


def _review(
    *,
    review_id: str | None,
    title: str,
    body: str,
    rating: str,
    author: str = "someone",
    updated: str | None,
) -> dict[str, Any]:
    """One review entry, in the wrapped-`label` shape the feed uses."""
    entry: dict[str, Any] = {
        "author": {"name": {"label": author}},
        "title": {"label": title},
        "content": {"label": body, "attributes": {"type": "text"}},
        "im:rating": {"label": rating},
    }
    if review_id is not None:
        entry["id"] = {"label": review_id}
    if updated is not None:
        entry["updated"] = {"label": updated}
    return entry


def _app_entry(name: str, track_id: int) -> dict[str, Any]:
    """The feed's first entry: the app's own metadata. It has text but no rating."""
    return {
        "im:name": {"label": name},
        "id": {"label": f"https://apps.apple.com/us/app/id{track_id}?uo=2"},
        "title": {"label": name},
        "content": {"label": f"{name} is the friendly productivity companion."},
    }


def _feed(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"feed": {"entry": list(entries)}}


# TaskPal's feed, newest first as `sortBy=mostRecent` returns it; the undated
# review sits last, as anything Apple could not date would.
_TASKPAL_FEED: Final[dict[str, Any]] = _feed(
    _app_entry("TaskPal", 111),
    _review(
        review_id="9001",
        title="Crashes on every sync",
        body=_CRASH_BODY,
        rating="1",
        author="angry-in-austin",
        updated="2026-07-20T10:00:00-07:00",
    ),
    _review(
        review_id="9002",
        title="Subscription trap",
        body=_TRIAL_BODY,
        rating="2",
        updated="2026-07-18T08:00:00-07:00",
    ),
    _review(
        review_id="9003",
        title="Broken again",
        body=_SYNC_BODY,
        rating="2",
        updated="not a date",
    ),
)

_FOCUSFLOW_FEED: Final[dict[str, Any]] = _feed(
    _app_entry("FocusFlow", 222),
    _review(
        review_id="9101",
        title="Timer keeps resetting",
        body=_TIMER_BODY,
        rating="1",
        updated="2026-07-21T09:30:00-07:00",
    ),
    _review(
        review_id="9102",
        title="Great little app",
        body=_PRAISE_BODY,
        rating="5",
        updated="2026-07-10T00:00:00-07:00",
    ),
)

# ------------------------------------------------------------------- plumbing


def _transport(
    *,
    search: httpx.Response | None = None,
    feeds: dict[int, httpx.Response] | None = None,
    requests: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    """Route `/search` and each app's review feed to a canned response.

    Any request with no canned answer fails the test outright: a collector that
    talks to endpoints its test did not anticipate is a bug, not a flake.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        path = request.url.path
        if path == "/search":
            assert search is not None, f"unexpected search request: {request.url}"
            return search
        for track_id, response in (feeds or {}).items():
            if f"/id={track_id}/" in path:
                return response
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.MockTransport(handler)


def _collector(
    transport: httpx.MockTransport,
    *,
    limit: int = 25,
    countries: list[str] | None = None,
) -> AppStoreReviewsCollector:
    config = CollectorConfig(
        limit=limit,
        app_store_countries=countries if countries is not None else ["us"],
    )
    return AppStoreReviewsCollector(config, transport=transport)


def _happy_transport(requests: list[httpx.Request] | None = None) -> httpx.MockTransport:
    return _transport(
        search=httpx.Response(200, json=_SEARCH_TWO_APPS),
        feeds={
            111: httpx.Response(200, json=_TASKPAL_FEED),
            222: httpx.Response(200, json=_FOCUSFLOW_FEED),
        },
        requests=requests,
    )


# ----------------------------------------------------------------------- tests


def test_reviews_are_parsed_verbatim() -> None:
    """Bodies survive word-for-word: the excerpt check downstream depends on it."""
    items = _collector(_happy_transport()).search("task manager")

    by_id = {item.external_id: item for item in items}
    crash = by_id["9001"]
    assert crash.text == _CRASH_BODY
    assert crash.collector == "app-reviews"
    assert crash.author == "angry-in-austin"
    assert crash.url == "https://apps.apple.com/us/app/id111"
    assert crash.published_at == datetime(2026, 7, 20, 17, 0, tzinfo=UTC)
    assert by_id["9101"].text == _TIMER_BODY
    assert by_id["9101"].url == "https://apps.apple.com/us/app/id222"


def test_app_metadata_entry_is_not_a_review() -> None:
    """The feed's first entry describes the app; only rated entries are quoted."""
    items = _collector(_happy_transport()).search("task manager")

    assert len(items) == 5
    assert not any("friendly productivity companion" in item.text for item in items)


def test_title_names_the_app_and_the_rating() -> None:
    """The stage sees which app and how angry without touching the body text."""
    items = _collector(_happy_transport()).search("task manager")

    titles = {item.external_id: item.title for item in items}
    assert titles["9001"] == "Crashes on every sync — TaskPal (1/5)"
    assert titles["9102"] == "Great little app — FocusFlow (5/5)"


def test_newest_first_across_apps() -> None:
    """Ordering merges the apps' feeds by date, with undated reviews last."""
    items = _collector(_happy_transport()).search("task manager")

    assert [item.external_id for item in items] == ["9101", "9001", "9002", "9102", "9003"]


def test_limit_caps_the_merged_result() -> None:
    """The cap keeps the newest reviews overall, not one whole app's feed."""
    items = _collector(_happy_transport(), limit=2).search("task manager")

    assert [item.external_id for item in items] == ["9101", "9001"]


def test_unparseable_date_keeps_the_review() -> None:
    """A bad timestamp costs the metadata, never the customer's words."""
    items = _collector(_happy_transport()).search("task manager")

    undated = next(item for item in items if item.external_id == "9003")
    assert undated.text == _SYNC_BODY
    assert undated.published_at is None
    assert items[-1] is undated


def test_one_failing_feed_loses_only_that_app() -> None:
    transport = _transport(
        search=httpx.Response(200, json=_SEARCH_TWO_APPS),
        feeds={
            111: httpx.Response(200, json=_TASKPAL_FEED),
            222: httpx.Response(500, text="upstream error"),
        },
    )

    items = _collector(transport).search("task manager")

    assert {item.external_id for item in items} == {"9001", "9002", "9003"}


def test_every_feed_failing_raises() -> None:
    transport = _transport(
        search=httpx.Response(200, json=_SEARCH_TWO_APPS),
        feeds={
            111: httpx.Response(500, text="upstream error"),
            222: httpx.Response(503, text="unavailable"),
        },
    )

    with pytest.raises(CollectorError, match="every review feed failed"):
        _collector(transport).search("task manager")


def test_search_failure_raises() -> None:
    transport = _transport(search=httpx.Response(500, text="upstream error"))

    with pytest.raises(CollectorError, match="refused the request"):
        _collector(transport).search("task manager")


def test_no_matching_apps_is_a_finding_not_an_error() -> None:
    """No incumbents selling software here is an answer; no feeds are fetched."""
    transport = _transport(search=httpx.Response(200, json=_NO_RESULTS))

    assert _collector(transport).search("obscure niche nothing sells") == []


def test_blank_query_asks_nothing() -> None:
    """A blank query makes no request at all — the strict transport would fail one."""
    transport = _transport()

    assert _collector(transport).search("   ") == []


def test_missing_review_id_falls_back_to_track_and_index() -> None:
    search = httpx.Response(
        200,
        json={"resultCount": 1, "results": [{"trackId": 333, "trackName": "NoteKeep"}]},
    )
    feed = httpx.Response(
        200,
        json=_feed(
            _app_entry("NoteKeep", 333),
            _review(
                review_id=None,
                title="Lost my notes",
                body="An update wiped every notebook I had.",
                rating="1",
                updated="2026-07-01T12:00:00-07:00",
            ),
        ),
    )

    items = _collector(_transport(search=search, feeds={333: feed})).search("notes")

    assert [item.external_id for item in items] == ["333:1"]


def test_whitespace_is_normalised_but_words_untouched() -> None:
    search = httpx.Response(
        200,
        json={"resultCount": 1, "results": [{"trackId": 444, "trackName": "SlowApp"}]},
    )
    feed = httpx.Response(
        200,
        json=_feed(
            _app_entry("SlowApp", 444),
            _review(
                review_id="9201",
                title="Sluggish",
                body="Way   too   slow.\n\n\nRefund please.",
                rating="1",
                updated="2026-07-02T12:00:00-07:00",
            ),
        ),
    )

    items = _collector(_transport(search=search, feeds={444: feed})).search("slow app")

    assert items[0].text == "Way too slow.\n\nRefund please."


def test_request_shape() -> None:
    """The search asks Apple's endpoint for software, capped, in the configured country."""
    requests: list[httpx.Request] = []
    _collector(_happy_transport(requests)).search("pomodoro timer")

    search_request = requests[0]
    assert search_request.url.host == "itunes.apple.com"
    assert search_request.url.params["term"] == "pomodoro timer"
    assert search_request.url.params["entity"] == "software"
    assert search_request.url.params["limit"] == "5"
    assert search_request.url.params["country"] == "us"

    feed_paths = {request.url.path for request in requests[1:]}
    assert feed_paths == {
        "/us/rss/customerreviews/id=111/sortBy=mostRecent/json",
        "/us/rss/customerreviews/id=222/sortBy=mostRecent/json",
    }
