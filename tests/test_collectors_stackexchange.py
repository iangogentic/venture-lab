"""StackExchangeCollector against canned API payloads; nothing touches the network.

Every test injects an `httpx.MockTransport` through the collector's transport
seam, so the full request path — parameter shaping, in-band error handling,
backoff, HTML stripping — is exercised without a socket.
"""

from datetime import UTC, datetime

import httpx
import pytest

from app.collectors.base import CollectorConfig
from app.collectors.stackexchange import StackExchangeCollector
from app.utils.errors import CollectorError, RateLimitedError

_ASKED_AT = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

_QUESTION = {
    "question_id": 42,
    "title": "Why doesn&#39;t the widget save &amp; close?",
    "body": "<p>The widget hangs &amp; loses my draft.</p><p>Nothing in the logs.</p>",
    "link": "https://stackoverflow.com/questions/42/why-doesnt-the-widget-save",
    "owner": {"display_name": "Ada Q."},
    "creation_date": int(_ASKED_AT.timestamp()),
    "score": 7,
    "is_answered": False,
}


def _config(**extras: object) -> CollectorConfig:
    """A config carrying extra keys the way a file-loaded config would."""
    return CollectorConfig.model_validate(extras)


def _recording_transport(
    seen: list[httpx.Request], payload: dict[str, object]
) -> httpx.MockTransport:
    """Serve `payload` for every request, recording each request made."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def test_search_returns_the_question_body_stripped_verbatim() -> None:
    seen: list[httpx.Request] = []
    collector = StackExchangeCollector(
        CollectorConfig(), transport=_recording_transport(seen, {"items": [_QUESTION]})
    )

    items = collector.search("widget hangs")

    assert len(items) == 1
    item = items[0]
    # Markup gone, entities unescaped, paragraphs kept apart — words untouched.
    assert item.text == "The widget hangs & loses my draft.\n\nNothing in the logs."
    # Titles arrive entity-encoded even inside a JSON body.
    assert item.title == "Why doesn't the widget save & close?"
    assert item.collector == "stack-exchange"
    assert item.external_id == "stackoverflow:42"
    assert item.url == "https://stackoverflow.com/questions/42/why-doesnt-the-widget-save"
    assert item.author == "Ada Q."
    published = item.published_at
    assert published is not None
    assert published == _ASKED_AT
    assert published.tzinfo is not None


def test_the_default_site_is_stackoverflow_when_unconfigured() -> None:
    seen: list[httpx.Request] = []
    collector = StackExchangeCollector(
        CollectorConfig(), transport=_recording_transport(seen, {"items": []})
    )

    collector.search("widget")

    assert len(seen) == 1
    params = seen[0].url.params
    assert params["site"] == "stackoverflow"
    assert params["filter"] == "withbody"
    assert params["sort"] == "relevance"
    assert "key" not in params


def test_a_configured_key_is_sent_with_every_request() -> None:
    seen: list[httpx.Request] = []
    collector = StackExchangeCollector(
        _config(stackexchange_key="hunter2"),
        transport=_recording_transport(seen, {"items": []}),
    )

    collector.search("widget")

    assert seen[0].url.params["key"] == "hunter2"


def test_an_in_band_error_payload_raises_with_the_api_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error_id": 502,
                "error_message": "simulated throttle violation",
                "error_name": "throttle_violation",
            },
        )

    collector = StackExchangeCollector(CollectorConfig(), transport=httpx.MockTransport(handler))

    with pytest.raises(CollectorError, match="simulated throttle violation"):
        collector.search("widget")


def test_a_backoff_keeps_the_results_but_stops_further_sites() -> None:
    seen: list[httpx.Request] = []
    collector = StackExchangeCollector(
        _config(stackexchange_sites=["stackoverflow", "serverfault"]),
        transport=_recording_transport(seen, {"items": [_QUESTION], "backoff": 10}),
    )

    items = collector.search("widget hangs")

    # What the first site returned is kept; the second site is never asked.
    assert [item.external_id for item in items] == ["stackoverflow:42"]
    assert [request.url.params["site"] for request in seen] == ["stackoverflow"]


def test_one_failing_site_among_two_loses_only_that_site() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["site"] == "stackoverflow":
            return httpx.Response(500, text="oops")
        return httpx.Response(200, json={"items": [_QUESTION]})

    collector = StackExchangeCollector(
        _config(stackexchange_sites=["stackoverflow", "serverfault"]),
        transport=httpx.MockTransport(handler),
    )

    items = collector.search("widget hangs")

    assert [item.external_id for item in items] == ["serverfault:42"]


def test_a_search_with_no_items_is_a_finding_not_a_failure() -> None:
    seen: list[httpx.Request] = []
    collector = StackExchangeCollector(
        CollectorConfig(), transport=_recording_transport(seen, {"items": [], "has_more": False})
    )

    assert collector.search("widget") == []


def test_a_blank_query_is_refused_before_any_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no request expected, got {request.url}")

    collector = StackExchangeCollector(CollectorConfig(), transport=httpx.MockTransport(handler))

    with pytest.raises(CollectorError, match="non-empty query"):
        collector.search("   ")


# ------------------------------------------------------- quota is network-wide


def test_a_duplicated_site_is_only_asked_once() -> None:
    """The quota is counted in requests, so a site listed twice buys nothing.

    Hand-maintained lists grow duplicates: one real config had 16 entries and
    11 distinct sites, quietly spending a third of its daily allowance twice.
    """
    seen: list[httpx.Request] = []
    collector = StackExchangeCollector(
        _config(stackexchange_sites=["stackoverflow", "askubuntu", "stackoverflow", "askubuntu"]),
        transport=_recording_transport(seen, {"items": [_QUESTION]}),
    )

    collector.search("widget hangs")

    asked = [httpx.QueryParams(request.url.query.decode()).get("site") for request in seen]
    assert asked == ["stackoverflow", "askubuntu"]


def test_a_throttled_response_stops_the_remaining_sites() -> None:
    """Stack Exchange counts requests per IP across the whole network, so a 429
    from one site is a 429 from all of them. Asking the rest anyway spends quota
    to collect refusals — one real run logged ninety-odd per attempt."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(429, text="<!DOCTYPE html><html>throttled</html>")

    collector = StackExchangeCollector(
        _config(stackexchange_sites=["stackoverflow", "askubuntu", "serverfault", "superuser"]),
        transport=httpx.MockTransport(handler),
    )

    # The *type* survives the summary, not just the message: the caller stops
    # asking this source only if it can tell a quota refusal from bad luck.
    with pytest.raises(RateLimitedError, match="every configured site failed"):
        collector.search("widget hangs")

    assert len(seen) == 1, "the other sites were asked after a network-wide refusal"


def test_a_refusal_served_as_html_is_summarised_not_quoted() -> None:
    """A 429 from the edge is a whole web page, and its first 200 characters are
    a doctype and the opening of a stylesheet — identical for every such refusal,
    several wrapped log lines each, and silent about which refusal it was."""
    page = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n'
        "    <title>Too Many Requests - Stack Exchange</title>\n"
        '    <style type="text/css">\n        body { margin: 0; }\n'
    )
    collector = StackExchangeCollector(
        CollectorConfig(),
        transport=httpx.MockTransport(lambda request: httpx.Response(429, text=page)),
    )

    with pytest.raises(RateLimitedError) as caught:
        collector.search("widget hangs")

    message = str(caught.value)
    assert "Too Many Requests - Stack Exchange" in message
    assert "DOCTYPE" not in message
    assert "text/css" not in message


def test_one_broken_site_does_not_stop_the_others() -> None:
    """A 404 is about that site. Only a throttle is about the network."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        site = httpx.QueryParams(request.url.query.decode()).get("site")
        if site == "stackoverflow":
            return httpx.Response(404, json={"error_message": "no such site"})
        return httpx.Response(200, json={"items": [_QUESTION]})

    collector = StackExchangeCollector(
        _config(stackexchange_sites=["stackoverflow", "askubuntu"]),
        transport=httpx.MockTransport(handler),
    )

    assert collector.search("widget hangs"), "the healthy site was skipped too"
    assert len(seen) == 2
