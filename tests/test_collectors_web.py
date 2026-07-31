"""The web collector: SearXNG results in, extracted page text out.

Every test runs through `httpx.MockTransport` — the constructor's transport
seam exists so the whole search, fetch and extract path is exercised without a
network. The canned page is deliberately realistic (an article with real
paragraphs around real boilerplate) because trafilatura ignores pages too thin
to plausibly be articles, and a test page it refuses would test nothing.
"""

from datetime import UTC

import httpx
import pytest

from app.collectors.base import CollectorConfig
from app.collectors.web import WebCollector
from app.utils.errors import CollectorError

_SEARX_URL = "http://searx.test"
_ALPHA_URL = "http://pages.test/alpha"
_BETA_URL = "http://pages.test/beta"

_BETA_SNIPPET = "Engineers report the beta rollout stalled for a month."

# Must survive extraction verbatim: the downstream stage literal-substring-checks
# its excerpts against `SourceItem.text`, so this sentence changing at all would
# mean the collector broke the one guarantee it exists to keep.
_DISTINCTIVE = (
    "The deploy pipeline failed every Tuesday because the cache invalidation "
    "job raced the build artifact upload."
)

_ALPHA_HTML = f"""<html>
  <head>
    <title>What broke our release train</title>
    <meta property="article:published_time" content="2026-04-02T09:30:00+00:00">
    <meta name="author" content="Priya Narayan">
  </head>
  <body>
    <nav><a href="/">Home</a> <a href="/about">About</a></nav>
    <article>
      <h1>What broke our release train</h1>
      <p>For the past six months our team has shipped a monolith to production twice a
      week, and for the past six months the release has been late more often than it
      has been on time. The failures were never dramatic. A queue backed up, a test
      suite flaked, someone forgot to bump a schema version, and the train quietly
      left without half its carriages.</p>
      <p>{_DISTINCTIVE}
      Nobody noticed for weeks because the retry masked it, and the retry itself
      doubled the length of the deploy window.</p>
      <p>We eventually bought a scheduling tool, then abandoned it within a quarter.
      The tool assumed a clean dependency graph, and our graph was a lie: half the
      edges existed only in the heads of two engineers who had since changed teams.</p>
      <p>What we actually needed was not orchestration but evidence: a record of what
      failed, when, and what the fix cost us in hours. Once we started writing that
      down, the argument for changing the process made itself.</p>
    </article>
    <footer>Copyright 2026 Example Industries</footer>
  </body>
</html>
"""


def _searx_payload() -> dict[str, object]:
    return {
        "results": [
            # A hit with no URL cannot be cited; it must be skipped, never fatal.
            {"title": "malformed hit", "content": "no url here"},
            {
                "url": _ALPHA_URL,
                "title": "What broke our release train",
                "content": "Snippet about the release train.",
                "publishedDate": "2026-05-01T10:00:00Z",
            },
            {
                "url": _BETA_URL,
                "title": "Beta thread",
                "content": _BETA_SNIPPET,
            },
        ]
    }


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.host == "searx.test":
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json=_searx_payload())
    if request.url.path == "/alpha":
        return httpx.Response(200, text=_ALPHA_HTML)
    if request.url.path == "/beta":
        return httpx.Response(404)
    raise AssertionError(f"unexpected request: {request.url}")


def _collector(transport: httpx.MockTransport | None = None) -> WebCollector:
    config = CollectorConfig(limit=5, searxng_url=_SEARX_URL)
    return WebCollector(config, transport=transport or httpx.MockTransport(_handler))


# ------------------------------------------------------------------ happy path


def test_items_carry_the_extracted_page_text_verbatim() -> None:
    items = _collector().search("release train")

    assert [item.external_id for item in items] == [_ALPHA_URL, _BETA_URL]
    alpha = items[0]
    assert _DISTINCTIVE in alpha.text
    assert alpha.collector == "web"
    assert alpha.url == _ALPHA_URL
    assert alpha.title == "What broke our release train"


def test_extraction_drops_the_boilerplate_but_not_the_words() -> None:
    """The nav and footer are not the page's statement; the article is."""
    alpha = _collector().search("release train")[0]

    assert "Copyright 2026" not in alpha.text
    assert "<p>" not in alpha.text


def test_page_metadata_becomes_aware_utc_fields() -> None:
    alpha = _collector().search("release train")[0]

    assert alpha.author == "Priya Narayan"
    assert alpha.published_at is not None
    assert alpha.published_at.tzinfo is UTC
    assert alpha.published_at.date().isoformat() == "2026-04-02"


# ------------------------------------------------------------ snippet fallback


def test_a_dead_page_falls_back_to_the_engine_snippet() -> None:
    """A 404 costs the page text, not the hit: the snippet is still engine-served
    verbatim text, and the citation still points a reader at the source."""
    beta = _collector().search("beta")[1]

    assert beta.text == _BETA_SNIPPET
    assert beta.url == _BETA_URL
    assert beta.published_at is None


# ------------------------------------------------------------- configuration


def test_unset_searxng_url_reads_as_unavailable_not_broken() -> None:
    collector = WebCollector(CollectorConfig(), transport=httpx.MockTransport(_handler))

    assert collector.available() is False
    with pytest.raises(CollectorError, match="SEARXNG_URL"):
        collector.search("anything")


def test_configured_collector_reports_available() -> None:
    assert _collector().available() is True


# ------------------------------------------------------------------- failures


def test_403_says_exactly_what_to_enable_in_settings_yml() -> None:
    """The stock SearXNG config serves HTML only; the error must name the fix."""
    transport = httpx.MockTransport(lambda request: httpx.Response(403))

    with pytest.raises(CollectorError, match=r"settings\.yml"):
        _collector(transport).search("anything")


def test_a_non_json_body_is_a_collector_error() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text="<html>a search page</html>")
    )

    with pytest.raises(CollectorError, match="not JSON"):
        _collector(transport).search("anything")


def test_an_empty_result_list_is_a_finding_not_a_failure() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"results": []}))

    assert _collector(transport).search("anything") == []


# ------------------------------------------------------------- politeness cap


def test_page_fetches_are_capped_however_large_the_limit() -> None:
    """Every hit is a different stranger's server; eight fetches is the ceiling."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "searx.test":
            rows = [{"url": f"http://pages.test/page{n}", "title": f"Page {n}"} for n in range(20)]
            return httpx.Response(200, json={"results": rows})
        return httpx.Response(200, text=_ALPHA_HTML)

    config = CollectorConfig(limit=100, searxng_url=_SEARX_URL)
    collector = WebCollector(config, transport=httpx.MockTransport(handler))

    assert len(collector.search("anything")) == 8
