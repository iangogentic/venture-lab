"""What `collect-evidence` searches for, and when it stops asking.

Two failures from one real run are pinned here. The run asked a question
phrased as an instruction, searched every source eight ways for `research top
25 manual`, dragged back 485 irrelevant candidates and kept none of them; and
one source that was out of quota was asked the same eight questions anyway,
answering each with the same wall of HTML.
"""

import logging
from typing import ClassVar

import pytest

from app.artifacts import ArtifactKind
from app.collectors import CollectorConfig, SourceItem
from app.config import get_settings
from app.skills import collect_evidence
from app.skills.base import SkillRequest
from app.skills.collect_evidence import CollectEvidenceSkill, derive_queries
from app.utils.errors import CollectorError, RateLimitedError, SkillError
from tests.factories import make

INSTRUCTION_SHAPED = (
    "Research the top 25 manual, repetitive, or error-prone tasks engineers perform "
    "before every production release, including evidence from GitHub issues, Reddit, "
    "Hacker News, engineering blogs, incident postmortems, and vendor documentation"
)
"""The question from the run this module exists because of."""


# ------------------------------------------------------------------- queries


def test_an_instruction_shaped_question_searches_for_its_subject() -> None:
    """Not for the instruction. `research top 25 manual` matches nothing that matters."""
    queries = derive_queries(INSTRUCTION_SHAPED)

    assert queries, "a long question must still derive something"
    for query in queries:
        assert "research" not in query, query
        assert "top" not in query, query
        assert "25" not in query, query


def test_the_places_to_look_are_not_searched_for() -> None:
    """ "including evidence from GitHub issues, Reddit…" names collectors, not a topic."""
    queries = derive_queries(INSTRUCTION_SHAPED)

    for query in queries:
        assert "github" not in query, query
        assert "reddit" not in query, query
        assert "postmortems" not in query, query


def test_a_long_question_is_searched_from_both_ends() -> None:
    """English puts the head of a noun phrase last: the domain is at the end.

    "manual, repetitive, or error-prone tasks engineers perform before every
    production release" is as much about production releases as about tasks,
    and taking the first four content words drops the half that says where.
    """
    queries = derive_queries(INSTRUCTION_SHAPED)

    assert any("production release" in query for query in queries)
    assert any("manual" in query for query in queries)


def test_a_short_question_derives_exactly_what_it_always_did() -> None:
    """The regression guard. Both ends of a short question are the same words."""
    assert derive_queries("Where do teams lose time in code review?") == [
        "code review",
        "code review slow",
        "code review frustrating",
        "code review wastes time",
        "code review alternatives",
        "switching from code review",
    ]


def test_a_question_with_no_content_words_is_searched_verbatim() -> None:
    assert derive_queries("Why?") == ["Why?"]


def test_the_query_budget_is_still_honoured() -> None:
    """Two subjects buy three shapes each, not six each: the limit is requests."""
    assert len(derive_queries(INSTRUCTION_SHAPED, limit=6)) == 6


# -------------------------------------------------------------- rate limits


class _Flaky:
    """A collector whose answers are scripted, and which records what it was asked."""

    name = "flaky"
    requires_credentials = False

    asked: ClassVar[list[str]] = []
    error: ClassVar[Exception | None] = None
    answers_first: ClassVar[int] = 0
    """How many queries are answered before `error` starts being raised."""

    def __init__(self, config: CollectorConfig | None = None) -> None:
        self.config = config or CollectorConfig()

    def available(self) -> bool:
        return True

    def search(self, query: str, *, limit: int | None = None) -> list[SourceItem]:
        _Flaky.asked.append(query)
        if _Flaky.error is not None and len(_Flaky.asked) > _Flaky.answers_first:
            raise _Flaky.error
        return [SourceItem(collector=self.name, external_id=query, text=f"about {query}")]


@pytest.fixture
def flaky(monkeypatch: pytest.MonkeyPatch) -> type[_Flaky]:
    """Route `collect-evidence` at one scripted collector and nothing else."""
    monkeypatch.setenv("COLLECTORS", '["flaky"]')
    get_settings.cache_clear()
    monkeypatch.setattr(collect_evidence, "get_collector", lambda name: _Flaky)
    monkeypatch.setattr(_Flaky, "asked", [])
    monkeypatch.setattr(_Flaky, "error", None)
    monkeypatch.setattr(_Flaky, "answers_first", 0)
    return _Flaky


def test_a_rate_limited_collector_is_asked_once(
    flaky: type[_Flaky], caplog: pytest.LogCaptureFixture
) -> None:
    """Out of quota is a fact about the source, not about the query that hit it."""
    flaky.error = RateLimitedError("HTTP 429: slow down")

    with (
        caplog.at_level(logging.WARNING, logger="app.skills.collect_evidence"),
        pytest.raises(SkillError, match="No collector could run"),
    ):
        CollectEvidenceSkill()._fetch(["one", "two", "three", "four"])

    assert flaky.asked == ["one"], "the exhausted source was asked again"
    assert caplog.text.count("rate limited") == 1, "said once, not once per query"


def test_an_all_throttled_stage_does_not_blame_the_configuration(
    flaky: type[_Flaky],
) -> None:
    """ "Configure a source" and "wait for the quota" are opposite instructions.

    The source is set up correctly and working; it just cannot answer now.
    """
    flaky.error = RateLimitedError("HTTP 429")

    with pytest.raises(SkillError) as caught:
        CollectEvidenceSkill()._fetch(["one", "two"])

    message = str(caught.value)
    assert "rate limited" in message
    assert "Configure at least one source" not in message


def test_an_ordinary_failure_still_tries_every_query(flaky: type[_Flaky]) -> None:
    """One query failing is not the source failing — that behaviour must survive."""
    flaky.error = CollectorError("that one query upset it")
    queries = ["one", "two", "three"]

    with pytest.raises(SkillError, match="No collector could run"):
        CollectEvidenceSkill()._fetch(queries)

    assert flaky.asked == queries


def test_a_rate_limit_keeps_what_was_already_fetched(flaky: type[_Flaky]) -> None:
    """Stopping early must not throw away the answers that arrived first."""
    flaky.error = RateLimitedError("HTTP 429")
    flaky.answers_first = 2

    items = CollectEvidenceSkill()._fetch(["one", "two", "three", "four"])

    assert [item.external_id for item in items] == ["one", "two"]
    assert flaky.asked == ["one", "two", "three"], "asked once past the limit, then stopped"


# ---------------------------------------------------------------- diagnosis


def test_keeping_nothing_reports_what_was_searched_for(
    flaky: type[_Flaky], caplog: pytest.LogCaptureFixture
) -> None:
    """The one line that tells "sources are silent" from "we asked the wrong thing"."""
    skill = CollectEvidenceSkill()
    skill._searched = ["research top 25 manual"]
    skill._candidates = {
        "x": SourceItem(collector="flaky", external_id="x", text="something unrelated")
    }
    request = SkillRequest(run_id="r1", question=make(ArtifactKind.QUESTION, run_id="r1"))

    with caplog.at_level(logging.WARNING, logger="app.skills.collect_evidence"):
        kept = skill.assemble(collect_evidence.CollectEvidenceOutput(selected=[]), request)

    assert kept == []
    assert "research top 25 manual" in caplog.text
    assert "search_queries" in caplog.text, "the override worth knowing about"
