"""DiscourseCollector against canned forum responses; nothing touches the network.

Every test injects an `httpx.MockTransport` through the collector's transport
seam, so the full request path — URL shaping, JSON decoding, HTML stripping,
fallbacks — is exercised without a socket.
"""

from datetime import UTC, datetime

import httpx
import pytest

from app.collectors.base import CollectorConfig
from app.collectors.discourse import DiscourseCollector
from app.utils.errors import CollectorError

_FORUM = "https://forum.example"

_SEARCH = {
    "topics": [
        {
            "id": 101,
            "slug": "widget-breaks-on-save",
            "title": "Widget breaks on save",
            "created_at": "2026-05-01T10:00:00.000Z",
        }
    ],
    "posts": [
        {
            "topic_id": 101,
            "username": "alice",
            "blurb": "It crashes every time I hit save",
        }
    ],
}

_TOPIC_DETAIL = {
    "post_stream": {
        "posts": [
            {
                "username": "alice",
                "created_at": "2026-05-01T10:00:00.000Z",
                "cooked": (
                    "<p>It crashes &amp; burns every time I hit <b>save</b>.</p>"
                    "<p>Reproduced on 2.4.</p>"
                ),
            }
        ]
    }
}


def _config(**extras: object) -> CollectorConfig:
    """A config carrying extra keys the way a file-loaded config would."""
    return CollectorConfig.model_validate(extras)


def _forum_transport() -> httpx.MockTransport:
    """One healthy forum: a search hit whose topic detail resolves."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search.json":
            return httpx.Response(200, json=_SEARCH)
        if request.url.path == "/t/101.json":
            return httpx.Response(200, json=_TOPIC_DETAIL)
        return httpx.Response(404, json={"errors": ["not found"]})

    return httpx.MockTransport(handler)


def _forbidden_transport() -> httpx.MockTransport:
    """A transport that fails the test if any request reaches it."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no request expected, got {request.url}")

    return httpx.MockTransport(handler)


def test_search_quotes_the_opening_post_stripped_verbatim() -> None:
    collector = DiscourseCollector(_config(discourse_forums=[_FORUM]), transport=_forum_transport())

    items = collector.search("widget")

    assert len(items) == 1
    item = items[0]
    # Markup gone, entities unescaped, paragraphs kept apart — words untouched.
    assert item.text == "It crashes & burns every time I hit save.\n\nReproduced on 2.4."
    assert item.collector == "discourse"
    assert item.external_id == "forum.example/t/101"
    assert item.title == "Widget breaks on save"
    assert item.url == f"{_FORUM}/t/widget-breaks-on-save/101"
    assert item.author == "alice"
    published = item.published_at
    assert published is not None
    assert published == datetime(2026, 5, 1, 10, 0, tzinfo=UTC)
    assert published.tzinfo is not None


def test_a_failed_topic_fetch_falls_back_to_the_search_blurb() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search.json":
            return httpx.Response(200, json=_SEARCH)
        return httpx.Response(404, json={"errors": ["not found"]})

    collector = DiscourseCollector(
        _config(discourse_forums=[_FORUM]), transport=httpx.MockTransport(handler)
    )

    items = collector.search("widget")

    assert len(items) == 1, "a topic-detail hiccup must downgrade the hit, not lose it"
    item = items[0]
    assert item.text == "It crashes every time I hit save"
    assert item.author == "alice"
    assert item.external_id == "forum.example/t/101"


def test_unconfigured_forums_mean_unavailable_not_an_error() -> None:
    collector = DiscourseCollector(CollectorConfig(), transport=_forbidden_transport())

    assert collector.available() is False
    assert collector.search("widget") == []


def test_configured_forums_make_the_collector_available() -> None:
    collector = DiscourseCollector(
        _config(discourse_forums=[_FORUM]), transport=_forbidden_transport()
    )

    assert collector.available() is True


def test_one_dead_forum_among_two_loses_only_that_forum() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "dead.example":
            return httpx.Response(500, text="down for maintenance")
        if request.url.path == "/search.json":
            return httpx.Response(200, json=_SEARCH)
        return httpx.Response(200, json=_TOPIC_DETAIL)

    collector = DiscourseCollector(
        _config(discourse_forums=["https://dead.example", _FORUM]),
        transport=httpx.MockTransport(handler),
    )

    items = collector.search("widget")

    assert [item.external_id for item in items] == ["forum.example/t/101"]


def test_every_forum_failing_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down for maintenance")

    collector = DiscourseCollector(
        _config(discourse_forums=["https://a.example", "https://b.example"]),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(CollectorError, match="every configured forum failed"):
        collector.search("widget")


def test_a_search_with_no_topics_is_a_finding_not_a_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"topics": [], "posts": []})

    collector = DiscourseCollector(
        _config(discourse_forums=[_FORUM]), transport=httpx.MockTransport(handler)
    )

    assert collector.search("widget") == []


def test_a_blank_query_is_refused_before_any_request() -> None:
    collector = DiscourseCollector(
        _config(discourse_forums=[_FORUM]), transport=_forbidden_transport()
    )

    with pytest.raises(CollectorError, match="non-empty query"):
        collector.search("   ")
