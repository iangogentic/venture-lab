"""The tavily collector: the API's results in, the page's own words out.

Every test runs through `httpx.MockTransport` — the constructor's transport
seam exists so the request shape, the raw-content preference and every refusal
path are exercised without a network or a real key.
"""

import json
from datetime import UTC

import httpx
import pytest
from pydantic import SecretStr

from app.collectors.base import CollectorConfig
from app.collectors.tavily import TavilyCollector
from app.utils.errors import CollectorError

_KEY = "tvly-test-key"

_RAW_SENTENCE = "We spent three weekends chasing a cache that kept returning stale sessions."
_AI_SNIPPET = "An AI-written summary of the postmortem."
_FALLBACK_SNIPPET = "Users complain the API rate limits break their batch jobs."


def _payload() -> dict[str, object]:
    return {
        "results": [
            {
                "title": "Postmortem: the cache that lied",
                "url": "https://blog.test/postmortem",
                "content": _AI_SNIPPET,
                "raw_content": _RAW_SENTENCE,
                "published_date": "2026-06-10T08:00:00Z",
            },
            {
                # No raw_content: the composed snippet is the only text on offer.
                "title": "Thread on rate limits",
                "url": "https://forum.test/thread/9",
                "content": _FALLBACK_SNIPPET,
                "raw_content": None,
            },
        ]
    }


def _collector(
    transport: httpx.MockTransport | None = None,
    *,
    key: str | SecretStr | None = _KEY,
    limit: int = 5,
) -> TavilyCollector:
    config = CollectorConfig(limit=limit, tavily_api_key=key)
    handler = transport or httpx.MockTransport(lambda request: httpx.Response(200, json=_payload()))
    return TavilyCollector(config, transport=handler)


# ------------------------------------------------------------------ happy path


def test_raw_content_is_preferred_over_the_composed_snippet() -> None:
    items = _collector().search("stale cache")

    first = items[0]
    assert first.text == _RAW_SENTENCE
    assert _AI_SNIPPET not in first.text
    assert first.collector == "tavily"
    assert first.external_id == "https://blog.test/postmortem"
    assert first.url == "https://blog.test/postmortem"
    assert first.title == "Postmortem: the cache that lied"


def test_missing_raw_content_falls_back_to_the_snippet() -> None:
    second = _collector().search("stale cache")[1]

    assert second.text == _FALLBACK_SNIPPET
    assert second.external_id == "https://forum.test/thread/9"


def test_published_date_becomes_an_aware_utc_datetime() -> None:
    first = _collector().search("stale cache")[0]

    assert first.published_at is not None
    assert first.published_at.tzinfo is UTC
    assert first.published_at.isoformat() == "2026-06-10T08:00:00+00:00"


def test_the_request_carries_the_key_the_cap_and_raw_content() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": []})

    # limit=25 must be capped to the API's ten: extra results cost credits.
    _collector(httpx.MockTransport(handler), limit=25).search("stale cache")

    assert len(seen) == 1
    request = seen[0]
    assert request.headers["Authorization"] == f"Bearer {_KEY}"
    body = json.loads(request.content)
    assert body["query"] == "stale cache"
    assert body["max_results"] == 10
    assert body["include_raw_content"] is True


def test_a_secretstr_key_is_unwrapped_and_never_leaks_from_the_config() -> None:
    collector = _collector(key=SecretStr(_KEY))

    assert collector.available() is True
    assert _KEY not in repr(collector.config)
    assert collector.search("stale cache")[0].text == _RAW_SENTENCE


# ------------------------------------------------------------- configuration


def test_without_a_key_the_collector_is_unavailable_not_broken() -> None:
    collector = _collector(key=None)

    assert collector.available() is False
    with pytest.raises(CollectorError, match="TAVILY_API_KEY"):
        collector.search("anything")


# ------------------------------------------------------------------- refusals


def test_a_rejected_key_names_the_env_var_and_never_the_key() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(401))

    with pytest.raises(CollectorError, match="TAVILY_API_KEY") as excinfo:
        _collector(transport).search("anything")
    assert _KEY not in str(excinfo.value)


def test_the_rate_limit_error_states_the_free_tier_quota() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(429))

    with pytest.raises(CollectorError, match="1,000 credits"):
        _collector(transport).search("anything")


def test_a_non_json_body_is_a_collector_error() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>oops</html>"))

    with pytest.raises(CollectorError, match="not JSON"):
        _collector(transport).search("anything")


# ------------------------------------------------------------- empty and bad


def test_an_empty_result_list_is_a_finding_not_a_failure() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"results": []}))

    assert _collector(transport).search("anything") == []


def test_one_unusable_result_never_loses_the_others() -> None:
    rows = {
        "results": [
            {"title": "no url, not citable", "content": "some words"},
            {"url": "https://ok.test/post", "content": "The one usable result."},
        ]
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=rows))

    items = _collector(transport).search("anything")

    assert [item.external_id for item in items] == ["https://ok.test/post"]
