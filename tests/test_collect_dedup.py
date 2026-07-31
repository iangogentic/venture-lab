"""Semantic dedup in collect-evidence: one complaint, two sources, one candidate.

Driven through the `_build_embedder` seam with `FakeEmbedder`, so the suite
never loads (or downloads) the real model. The behaviour under test is exactly
the degradation contract: with vectors, cross-posts collapse; without them, the
stage behaves as it did before memory existed.
"""

import logging

import pytest

from app.artifacts import ArtifactKind, Evidence
from app.collectors import SourceItem
from app.config import get_settings
from app.memory import Embedder
from app.skills import collect_evidence
from app.skills.base import SkillRequest
from app.skills.collect_evidence import CollectEvidenceSkill
from app.utils.errors import MemoryUnavailableError
from app.utils.paths import WorkspacePaths
from tests.factories import make
from tests.test_memory import FakeEmbedder, _open_store, _prepared

CROSS_POST = "Our CI takes 40 minutes and blocks every merge."
"""The complaint, as pasted to two different sources."""


def _item(collector: str, external_id: str, text: str, title: str | None = None) -> SourceItem:
    return SourceItem(collector=collector, external_id=external_id, text=text, title=title)


def _use(embedder: Embedder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_evidence, "_build_embedder", lambda: embedder)


def test_cross_posted_complaint_is_kept_once_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    hn = _item("hacker-news", "hn-1", CROSS_POST, title="CI is slow")
    reddit = _item("reddit", "rd-1", CROSS_POST + " Same story here.", title="40 minute CI")
    _use(FakeEmbedder({hn.text: [1.0, 0.0], reddit.text: [0.99, 0.01]}), monkeypatch)

    skill = CollectEvidenceSkill()
    with caplog.at_level(logging.INFO, logger="app.skills.collect_evidence"):
        kept = skill._semantically_deduplicated([hn, reddit])

    assert [item.external_id for item in kept] == ["hn-1"]
    # The drop is reported, not silent: collector and title of what went, and why.
    assert "near-duplicate" in caplog.text
    assert "reddit" in caplog.text
    assert "40 minute CI" in caplog.text
    # The kept item's vector is retained for indexing after assemble.
    assert "hn-1" in skill._vectors


def test_distinct_complaints_are_both_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    ci = _item("hacker-news", "hn-1", "CI is slow and flaky.")
    docs = _item("reddit", "rd-1", "The docs are outdated and wrong.")
    _use(FakeEmbedder({ci.text: [1.0, 0.0], docs.text: [0.0, 1.0]}), monkeypatch)

    kept = CollectEvidenceSkill()._semantically_deduplicated([ci, docs])

    assert {item.external_id for item in kept} == {"hn-1", "rd-1"}


def test_without_the_model_every_candidate_survives(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No embedder means no semantic dedup — and nothing else may change."""

    def _unavailable() -> Embedder:
        raise MemoryUnavailableError("no model here")

    monkeypatch.setattr(collect_evidence, "_build_embedder", _unavailable)
    items = [_item("hacker-news", "hn-1", CROSS_POST), _item("reddit", "rd-1", CROSS_POST)]

    with caplog.at_level(logging.DEBUG, logger="app.skills.collect_evidence"):
        kept = CollectEvidenceSkill()._semantically_deduplicated(items)

    assert kept == items
    assert "semantic dedup skipped" in caplog.text


def test_memory_disabled_never_touches_the_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    get_settings.cache_clear()
    fake = FakeEmbedder()
    _use(fake, monkeypatch)
    items = [_item("hacker-news", "hn-1", CROSS_POST), _item("reddit", "rd-1", CROSS_POST)]

    kept = CollectEvidenceSkill()._semantically_deduplicated(items)

    assert kept == items
    assert fake.calls == []


# ------------------------------------------------- indexing after assemble


def test_kept_evidence_is_remembered_with_reused_vectors(
    workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vectors computed during dedup carry through to the memory index."""
    hn = _item("hacker-news", "hn-1", CROSS_POST, title="CI is slow")
    _use(FakeEmbedder({hn.text: [1.0, 0.0]}), monkeypatch)

    skill = CollectEvidenceSkill()
    kept = skill._semantically_deduplicated([hn, _item("reddit", "rd-1", "Docs are wrong.")])
    assert len(kept) == 2

    evidence = make(
        ArtifactKind.EVIDENCE, collector="hacker-news", excerpt=CROSS_POST, source_id="hn-1"
    )
    assert isinstance(evidence, Evidence)
    question = make(ArtifactKind.QUESTION)
    skill._remember([evidence], SkillRequest(run_id="r1", question=question))

    store = _open_store(workspace)
    try:
        assert store.count() == 1
        hits = store.recall(_prepared([1.0, 0.0]), limit=1)
        assert hits[0].evidence_id == evidence.id
        assert hits[0].question_text == getattr(question, "text", None)
    finally:
        store.close()


def test_memory_failure_never_fails_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Indexing is best-effort: an unavailable memory is a debug line, not an error."""

    def _unavailable() -> Embedder:
        raise MemoryUnavailableError("no model here")

    monkeypatch.setattr(collect_evidence, "_build_embedder", _unavailable)
    evidence = make(ArtifactKind.EVIDENCE, source_id="hn-1")

    # Must not raise, whatever state memory is in.
    CollectEvidenceSkill()._remember([evidence], SkillRequest(run_id="r1"))
