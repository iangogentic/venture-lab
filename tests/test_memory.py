"""The local semantic memory: embedder contract, cosine, the store, `op recall`.

Nothing here touches the network or the real embedding model — that would make
the suite download ~30MB once per cache and fail offline. Every vector comes
from `FakeEmbedder`, which is also what the dedup tests inject.
"""

import hashlib
import json
import math

import pytest
from typer.testing import CliRunner

from app.artifacts import ArtifactKind, Evidence
from app.cli.commands import recall as recall_cmd
from app.cli.main import app
from app.collectors.base import SourceItem
from app.memory import EMBEDDING_DIMENSIONS, Embedder, MemoryStore, cosine
from app.memory.embedder import StaticModelEmbedder
from app.utils.errors import MemoryUnavailableError
from app.utils.paths import WorkspacePaths
from tests.factories import make

runner = CliRunner()


class FakeEmbedder:
    """Deterministic vectors, no model: mapped texts get their assigned vector.

    Unmapped texts get a vector derived from their hash, so any text embeds to
    *something* stable. Every vector is padded (or trimmed) to the store's
    width and L2-normalised, keeping distances comparable across tests.
    """

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self.mapping = mapping or {}
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [
            _prepared(self.mapping[text]) if text in self.mapping else _hashed(text)
            for text in texts
        ]


def _prepared(vector: list[float]) -> list[float]:
    padded = (vector + [0.0] * EMBEDDING_DIMENSIONS)[:EMBEDDING_DIMENSIONS]
    norm = math.sqrt(sum(x * x for x in padded))
    return [x / norm for x in padded] if norm else padded


def _hashed(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return _prepared([digest[i % len(digest)] / 255.0 for i in range(EMBEDDING_DIMENSIONS)])


def _open_store(paths: WorkspacePaths) -> MemoryStore:
    """A store on the test workspace, or a skip where sqlite-vec cannot load."""
    try:
        return MemoryStore(paths.memory_db)
    except MemoryUnavailableError:  # pragma: no cover - system-Python fallback
        pytest.skip("sqlite-vec cannot load into this interpreter")


def _evidence(excerpt: str, source_id: str) -> Evidence:
    built = make(ArtifactKind.EVIDENCE, excerpt=excerpt, source_id=source_id)
    assert isinstance(built, Evidence)
    return built


# ------------------------------------------------------------------ cosine


def test_cosine_of_identical_vectors_is_one() -> None:
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero() -> None:
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_of_opposed_vectors_is_minus_one() -> None:
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_with_a_zero_vector_is_zero_not_an_error() -> None:
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# ------------------------------------------------------------------- store


def test_store_failure_mode_is_the_typed_error(workspace: WorkspacePaths) -> None:
    """Construction either works or raises the one error callers degrade on."""
    try:
        store = MemoryStore(workspace.memory_db)
    except MemoryUnavailableError:  # pragma: no cover - system-Python fallback
        return
    store.close()


def test_index_then_recall_returns_nearest_first(workspace: WorkspacePaths) -> None:
    store = _open_store(workspace)
    try:
        near = _evidence("CI takes forever on every merge.", "src-near")
        far = _evidence("Nobody reads the release notes.", "src-far")
        vectors = [_prepared([1.0, 0.0]), _prepared([0.0, 1.0])]

        inserted = store.index(
            [near, far], vectors, run_id="r1", question_text="Where is time lost?", model="fake"
        )
        assert inserted == 2

        hits = store.recall(_prepared([0.9, 0.1]), limit=5)
        assert [hit.evidence_id for hit in hits] == [near.id, far.id]
        assert hits[0].distance <= hits[1].distance
        assert hits[0].run_id == "r1"
        assert hits[0].question_text == "Where is time lost?"
        assert hits[0].excerpt == "CI takes forever on every merge."
    finally:
        store.close()


def test_indexing_the_same_evidence_twice_stores_it_once(workspace: WorkspacePaths) -> None:
    """Re-running a stage re-assembles the same artifacts; memory must not double."""
    store = _open_store(workspace)
    try:
        item = _evidence("Deploys fail on Fridays.", "src-1")
        vectors = [_prepared([1.0, 0.0])]

        first = store.index([item], vectors, run_id="r1", question_text="q", model="fake")
        second = store.index([item], vectors, run_id="r1", question_text="q", model="fake")

        assert (first, second) == (1, 0)
        assert store.count() == 1
        assert len(store.recall(_prepared([1.0, 0.0]), limit=10)) == 1
    finally:
        store.close()


def test_count_starts_at_zero(workspace: WorkspacePaths) -> None:
    store = _open_store(workspace)
    try:
        assert store.count() == 0
    finally:
        store.close()


def test_index_rejects_a_wrong_width_vector(workspace: WorkspacePaths) -> None:
    """A vector of another width would poison every later KNN query."""
    store = _open_store(workspace)
    try:
        with pytest.raises(ValueError, match="dimensions"):
            store.index(
                [_evidence("text", "src-1")], [[1.0, 2.0]], run_id="r", question_text="q", model="m"
            )
    finally:
        store.close()


# --------------------------------------------------------------- op recall


def test_recall_json_emits_only_the_payload(
    workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`op recall --json` stdout must parse as JSON — nothing else may leak in."""
    store = _open_store(workspace)
    try:
        item = _evidence("CI takes forever on every merge.", "src-1")
        store.index([item], [_prepared([1.0, 0.0])], run_id="r1", question_text="q", model="fake")
    finally:
        store.close()

    fake: Embedder = FakeEmbedder({"slow builds": [1.0, 0.0]})
    monkeypatch.setattr(recall_cmd, "_build_embedder", lambda: fake)

    result = runner.invoke(app, ["recall", "slow builds", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["evidence_id"] == item.id
    assert payload[0]["run_id"] == "r1"


def test_recall_on_an_empty_memory_exits_zero(workspace: WorkspacePaths) -> None:
    """Nothing remembered yet is a state, not an error — and no model is loaded."""
    result = runner.invoke(app, ["recall", "anything", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == []
    assert "memory" in result.stderr.lower()


# ------------------------------------------------------------------ embedder


class _ExplodingModel:
    """A loaded model whose `encode` fails the way a parallel tokeniser does."""

    def encode(self, texts: list[str]) -> object:
        # Seen in the wild from model2vec's multiprocessing layer. Not an
        # ImportError, not an OSError — just whatever the pool felt like raising.
        raise ValueError("bad value(s) in fds_to_keep")


def test_an_encode_failure_is_the_typed_error_not_a_dead_run() -> None:
    """Regression: only *loading* the model was wrapped, so a crash inside
    `encode` escaped as a raw ValueError and failed a four-minute
    `collect-evidence` attempt that had already fetched everything it needed.
    Memory is a convenience layered on a run; it must never cost you the run.
    """
    embedder = StaticModelEmbedder("test/model")
    embedder._model = _ExplodingModel()

    with pytest.raises(MemoryUnavailableError, match="could not encode"):
        embedder.encode(["one", "two"])


def test_collect_evidence_carries_on_when_the_embedder_dies(
    workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The typed error is only worth having if the caller actually degrades."""
    from app.skills.collect_evidence import CollectEvidenceSkill

    broken = StaticModelEmbedder("test/model")
    broken._model = _ExplodingModel()
    monkeypatch.setattr("app.skills.collect_evidence._build_embedder", lambda: broken)

    skill = CollectEvidenceSkill()
    items = [
        SourceItem(collector="rss", external_id=f"{index}", text=f"CI takes {index} minutes")
        for index in range(3)
    ]

    assert skill._semantically_deduplicated(items) == items, (
        "a dead embedder dropped or raised instead of passing through"
    )
