"""The artifact registry: JSON files under workspace/, no database."""

import json

import pytest

from app.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactRegistry,
    ArtifactStatus,
    EvidenceLevel,
    Question,
    kind_for_id,
)
from app.artifacts.registry import HISTORY_DIRNAME
from app.utils.errors import ArtifactError
from app.utils.paths import WorkspacePaths
from tests.factories import make


@pytest.fixture
def registry(workspace: WorkspacePaths) -> ArtifactRegistry:
    return ArtifactRegistry(workspace)


# ----------------------------------------------------------------- save/load


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_save_writes_json_into_the_matching_directory(
    registry: ArtifactRegistry, workspace: WorkspacePaths, kind: ArtifactKind
) -> None:
    artifact = make(kind)
    path = registry.save(artifact)

    assert path.parent == workspace.for_directory(kind.directory)
    assert path.name == f"{artifact.id}.json"
    assert json.loads(path.read_text())["id"] == artifact.id


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_round_trip_preserves_the_artifact(registry: ArtifactRegistry, kind: ArtifactKind) -> None:
    original = make(kind, confidence=0.42, evidence_level=EvidenceLevel.MEASURED)
    registry.save(original)

    assert registry.load(kind, original.id) == original


def test_saved_json_is_self_describing(registry: ArtifactRegistry) -> None:
    """A file copied out of its directory must still say what it is."""
    artifact = make(ArtifactKind.EVIDENCE)
    data = json.loads(registry.save(artifact).read_text())

    assert data["kind"] == ArtifactKind.EVIDENCE.value


def test_load_rejects_a_file_whose_kind_contradicts_its_directory(
    registry: ArtifactRegistry,
) -> None:
    artifact = make(ArtifactKind.EVIDENCE)
    path = registry.save(artifact)

    data = json.loads(path.read_text())
    data["kind"] = ArtifactKind.REPORT.value
    path.write_text(json.dumps(data))

    with pytest.raises(ArtifactError, match="declares kind"):
        registry.load(ArtifactKind.EVIDENCE, artifact.id)


def test_load_missing_artifact_raises(registry: ArtifactRegistry) -> None:
    with pytest.raises(ArtifactError, match="No question artifact"):
        registry.load(ArtifactKind.QUESTION, "q_missing")


def test_load_invalid_json_raises(registry: ArtifactRegistry, workspace: WorkspacePaths) -> None:
    path = workspace.questions / "q_broken.json"
    path.write_text("{not json")

    with pytest.raises(ArtifactError, match="not valid JSON"):
        registry.load(ArtifactKind.QUESTION, "q_broken")


def test_save_can_refuse_to_overwrite(registry: ArtifactRegistry) -> None:
    artifact = make(ArtifactKind.QUESTION)
    registry.save(artifact)

    with pytest.raises(ArtifactError, match="already exists"):
        registry.save(artifact, overwrite=False)


def test_save_leaves_no_temp_files_behind(
    registry: ArtifactRegistry, workspace: WorkspacePaths
) -> None:
    """Writes go through a temp file plus rename; none of them should survive."""
    registry.save(make(ArtifactKind.QUESTION))
    assert list(workspace.questions.glob("*.tmp")) == []


def test_delete_removes_the_current_version(registry: ArtifactRegistry) -> None:
    artifact = make(ArtifactKind.QUESTION)
    registry.save(artifact)

    registry.delete(ArtifactKind.QUESTION, artifact.id)
    assert not registry.exists(ArtifactKind.QUESTION, artifact.id)

    registry.delete(ArtifactKind.QUESTION, artifact.id, missing_ok=True)
    with pytest.raises(ArtifactError):
        registry.delete(ArtifactKind.QUESTION, artifact.id)


# --------------------------------------------------------------------- update


def test_update_changes_content_without_bumping_the_version(
    registry: ArtifactRegistry,
) -> None:
    artifact = make(ArtifactKind.QUESTION)
    registry.save(artifact)

    updated = registry.update(artifact, status=ArtifactStatus.READY, confidence=0.8)

    assert updated.status is ArtifactStatus.READY
    assert updated.confidence == 0.8
    assert updated.version == artifact.version
    assert updated.updated_at >= artifact.updated_at
    assert registry.load(ArtifactKind.QUESTION, artifact.id) == updated


def test_update_validates_the_change(registry: ArtifactRegistry) -> None:
    """A bad value must be rejected here, not written out and discovered later."""
    artifact = make(ArtifactKind.QUESTION)
    registry.save(artifact)

    with pytest.raises(ArtifactError):
        registry.update(artifact, confidence=7.0)


def test_update_leaves_no_history(registry: ArtifactRegistry) -> None:
    artifact = make(ArtifactKind.QUESTION)
    registry.save(artifact)
    registry.update(artifact, status=ArtifactStatus.READY)

    assert registry.versions(ArtifactKind.QUESTION, artifact.id) == []


# -------------------------------------------------------------------- version


def test_version_bumps_and_archives(registry: ArtifactRegistry) -> None:
    artifact = make(ArtifactKind.QUESTION, text="First phrasing")
    registry.save(artifact)

    revised = registry.version(artifact, text="Second phrasing")

    assert revised.version == 2
    assert revised.id == artifact.id, "versioning keeps a stable identity"
    assert registry.versions(ArtifactKind.QUESTION, artifact.id) == [1]

    archived = registry.load_version(ArtifactKind.QUESTION, artifact.id, 1)
    assert isinstance(archived, Question)
    assert archived.text == "First phrasing"

    current = registry.load_as(ArtifactKind.QUESTION, artifact.id, Question)
    assert current.text == "Second phrasing"


def test_repeated_versioning_accumulates_history(registry: ArtifactRegistry) -> None:
    artifact = make(ArtifactKind.QUESTION, text="v1")
    registry.save(artifact)

    second = registry.version(artifact, text="v2")
    third = registry.version(second, text="v3")

    assert third.version == 3
    assert registry.versions(ArtifactKind.QUESTION, artifact.id) == [1, 2]

    second_revision = registry.load_version(ArtifactKind.QUESTION, artifact.id, 2)
    assert isinstance(second_revision, Question)
    assert second_revision.text == "v2"


def test_history_is_hidden_from_current_listings(
    registry: ArtifactRegistry, workspace: WorkspacePaths
) -> None:
    artifact = make(ArtifactKind.QUESTION)
    registry.save(artifact)
    registry.version(artifact, text="revised")

    assert (workspace.questions / HISTORY_DIRNAME).is_dir()
    assert registry.list_ids(ArtifactKind.QUESTION) == [artifact.id]


def test_load_missing_version_raises(registry: ArtifactRegistry) -> None:
    artifact = make(ArtifactKind.QUESTION)
    registry.save(artifact)

    with pytest.raises(ArtifactError, match="No version 9"):
        registry.load_version(ArtifactKind.QUESTION, artifact.id, 9)


# --------------------------------------------------------------- search by id


def test_find_by_id_searches_across_kinds(registry: ArtifactRegistry) -> None:
    for kind in ArtifactKind:
        artifact = make(kind)
        registry.save(artifact)

        found = registry.find_by_id(artifact.id)
        assert found is not None
        assert found.id == artifact.id
        assert type(found).kind is kind


def test_find_by_id_returns_none_when_absent(registry: ArtifactRegistry) -> None:
    assert registry.find_by_id("q_nope") is None
    assert registry.locate("q_nope") is None


def test_locate_reports_the_kind(registry: ArtifactRegistry) -> None:
    artifact = make(ArtifactKind.OPPORTUNITY)
    registry.save(artifact)

    assert registry.locate(artifact.id) == ArtifactRef(
        kind=ArtifactKind.OPPORTUNITY, id=artifact.id
    )


def test_find_by_id_falls_back_when_the_prefix_is_unhelpful(
    registry: ArtifactRegistry,
) -> None:
    """Hand-minted ids carry no usable prefix, so the scan must still find them."""
    artifact = make(ArtifactKind.REPORT, id="handwritten-id")
    registry.save(artifact)

    assert kind_for_id("handwritten-id") is None
    found = registry.find_by_id("handwritten-id")
    assert found is not None and found.id == "handwritten-id"


def test_resolve_follows_a_ref(registry: ArtifactRegistry) -> None:
    artifact = make(ArtifactKind.PAIN_CLUSTER)
    registry.save(artifact)

    assert registry.resolve(artifact.ref) == artifact


# ------------------------------------------------------------- search by type


def test_find_by_type_returns_only_that_kind(registry: ArtifactRegistry) -> None:
    for kind in ArtifactKind:
        registry.save(make(kind))
    registry.save(make(ArtifactKind.EVIDENCE))

    evidence = registry.find_by_type(ArtifactKind.EVIDENCE)

    assert len(evidence) == 2
    assert {type(a).kind for a in evidence} == {ArtifactKind.EVIDENCE}
    assert registry.count(ArtifactKind.EVIDENCE) == 2


def test_find_by_type_on_an_empty_kind(registry: ArtifactRegistry) -> None:
    assert registry.find_by_type(ArtifactKind.REPORT) == []
    assert registry.list_ids(ArtifactKind.REPORT) == []
    assert registry.count(ArtifactKind.REPORT) == 0


def test_find_by_type_filters_are_conjunctive(registry: ArtifactRegistry) -> None:
    registry.save(make(ArtifactKind.EVIDENCE, status=ArtifactStatus.READY, run_id="run_a"))
    registry.save(make(ArtifactKind.EVIDENCE, status=ArtifactStatus.DRAFT, run_id="run_a"))
    registry.save(make(ArtifactKind.EVIDENCE, status=ArtifactStatus.READY, run_id="run_b"))

    both = registry.find_by_type(ArtifactKind.EVIDENCE, status=ArtifactStatus.READY, run_id="run_a")
    assert len(both) == 1


def test_find_by_type_filters_on_evidence_level(registry: ArtifactRegistry) -> None:
    registry.save(make(ArtifactKind.EVIDENCE, evidence_level=EvidenceLevel.ANECDOTAL))
    registry.save(make(ArtifactKind.EVIDENCE, evidence_level=EvidenceLevel.VERIFIED))

    strong = registry.find_by_type(ArtifactKind.EVIDENCE, min_evidence_level=EvidenceLevel.MEASURED)
    assert len(strong) == 1
    assert strong[0].evidence_level is EvidenceLevel.VERIFIED


def test_min_confidence_excludes_unassessed(registry: ArtifactRegistry) -> None:
    """An unknown confidence cannot satisfy a threshold — it must not pass."""
    registry.save(make(ArtifactKind.EVIDENCE, confidence=None))
    registry.save(make(ArtifactKind.EVIDENCE, confidence=0.9))

    assert len(registry.find_by_type(ArtifactKind.EVIDENCE, min_confidence=0.5)) == 1


def test_find_by_type_respects_limit(registry: ArtifactRegistry) -> None:
    for _ in range(5):
        registry.save(make(ArtifactKind.EVIDENCE))

    assert len(registry.find_by_type(ArtifactKind.EVIDENCE, limit=2)) == 2


def test_iter_by_type_yields_every_saved_artifact(registry: ArtifactRegistry) -> None:
    expected = [make(ArtifactKind.RESEARCH_BRIEF) for _ in range(3)]
    for artifact in expected:
        registry.save(artifact)

    streamed = list(registry.iter_by_type(ArtifactKind.RESEARCH_BRIEF))

    assert {a.id for a in streamed} == {a.id for a in expected}


def test_iter_by_type_on_a_missing_directory(workspace: WorkspacePaths) -> None:
    """A workspace that was never initialised must read as empty, not explode."""
    registry = ArtifactRegistry(workspace)
    workspace.reports.rmdir()

    assert list(registry.iter_by_type(ArtifactKind.REPORT)) == []
    assert registry.list_ids(ArtifactKind.REPORT) == []
