"""The artifact envelope: every kind must carry the same required fields.

The point of declaring `version`/`id`/timestamps/`status`/`confidence`/
`evidence_level` once on the base is that they cannot drift per kind — so these
tests assert against every kind rather than a sample.
"""

import pytest
from pydantic import ValidationError

from app.artifacts import (
    MODELS,
    Artifact,
    ArtifactKind,
    ArtifactRef,
    ArtifactStatus,
    Evidence,
    EvidenceLevel,
    Question,
)
from app.utils.paths import STAGE_DIRECTORIES, WorkspacePaths
from tests.factories import make

ENVELOPE_FIELDS = (
    "id",
    "run_id",
    "version",
    "schema_version",
    "created_at",
    "updated_at",
    "status",
    "confidence",
    "evidence_level",
    "parents",
    "supersedes",
)


# --------------------------------------------------------------- kind ↔ layout


def test_every_kind_maps_to_a_workspace_directory() -> None:
    """`ArtifactKind.directory` and `STAGE_DIRECTORIES` are two spellings of one contract."""
    assert {kind.directory for kind in ArtifactKind} == set(STAGE_DIRECTORIES)


def test_kind_directories_are_unique() -> None:
    directories = [kind.directory for kind in ArtifactKind]
    assert len(directories) == len(set(directories))


def test_artifact_directories_exist_in_a_real_workspace(workspace: WorkspacePaths) -> None:
    for kind in ArtifactKind:
        assert workspace.for_directory(kind.directory).is_dir()


def test_every_kind_has_exactly_one_model() -> None:
    assert set(MODELS) == set(ArtifactKind)
    assert len({model.__name__ for model in MODELS.values()}) == len(ArtifactKind)


def test_id_prefixes_are_unique() -> None:
    """`kind_for_id` resolves a kind from an id prefix, so collisions would misroute lookups."""
    prefixes = [model.id_prefix for model in MODELS.values()]
    assert len(prefixes) == len(set(prefixes))


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_model_declares_its_own_kind(kind: ArtifactKind) -> None:
    assert MODELS[kind].kind is kind


# ------------------------------------------------------------------- envelope


@pytest.mark.parametrize("kind", list(ArtifactKind))
@pytest.mark.parametrize("field", ENVELOPE_FIELDS)
def test_every_model_carries_every_envelope_field(kind: ArtifactKind, field: str) -> None:
    assert field in MODELS[kind].model_fields


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_envelope_defaults(kind: ArtifactKind) -> None:
    artifact = make(kind)

    assert artifact.version == 1
    assert artifact.schema_version >= 1
    assert artifact.status is ArtifactStatus.DRAFT
    assert artifact.confidence is None, "confidence must start unassessed, not zero"
    assert artifact.evidence_level is EvidenceLevel.NONE
    assert artifact.parents == []
    assert artifact.supersedes is None


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_timestamps_are_timezone_aware(kind: ArtifactKind) -> None:
    artifact = make(kind)
    assert artifact.created_at.tzinfo is not None
    assert artifact.updated_at.tzinfo is not None


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_generated_id_carries_the_kind_prefix(kind: ArtifactKind) -> None:
    artifact = make(kind)
    assert artifact.id.startswith(f"{MODELS[kind].id_prefix}_")


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_ref_points_back_at_the_artifact(kind: ArtifactKind) -> None:
    artifact = make(kind)
    assert artifact.ref == ArtifactRef(kind=kind, id=artifact.id)


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_unknown_fields_are_rejected(kind: ArtifactKind) -> None:
    """`extra="forbid"` is what stops a typo'd field silently vanishing on write."""
    with pytest.raises(ValidationError):
        make(kind, nonsense=True)


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_version_must_be_positive(kind: ArtifactKind) -> None:
    with pytest.raises(ValidationError):
        make(kind, version=0)


@pytest.mark.parametrize("kind", list(ArtifactKind))
@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_confidence_is_bounded(kind: ArtifactKind, bad: float) -> None:
    with pytest.raises(ValidationError):
        make(kind, confidence=bad)


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_confidence_accepts_the_closed_unit_interval(kind: ArtifactKind) -> None:
    assert make(kind, confidence=0.0).confidence == 0.0
    assert make(kind, confidence=1.0).confidence == 1.0


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_validate_assignment_is_on(kind: ArtifactKind) -> None:
    """Mutating an artifact must be validated too, not just construction."""
    artifact = make(kind)
    with pytest.raises(ValidationError):
        artifact.confidence = 5.0


# ------------------------------------------------------------ evidence levels


def test_evidence_levels_are_ordinal() -> None:
    ordered = [
        EvidenceLevel.NONE,
        EvidenceLevel.ANECDOTAL,
        EvidenceLevel.CORROBORATED,
        EvidenceLevel.MEASURED,
        EvidenceLevel.VERIFIED,
    ]
    ranks = [level.rank for level in ordered]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


def test_status_and_verdict_are_separate_concerns() -> None:
    """A rejected opportunity still has a complete decision record."""
    decision = make(ArtifactKind.DECISION, status=ArtifactStatus.READY)
    assert decision.status is ArtifactStatus.READY
    assert hasattr(decision, "verdict")


# ------------------------------------------------------------------ provenance


def test_parents_form_a_provenance_chain() -> None:
    question = make(ArtifactKind.QUESTION)
    assert isinstance(question, Question)

    evidence = make(ArtifactKind.EVIDENCE, parents=[question.ref])
    assert isinstance(evidence, Evidence)

    brief = make(ArtifactKind.RESEARCH_BRIEF, parents=[evidence.ref])

    assert brief.parents[0].kind is ArtifactKind.EVIDENCE
    assert evidence.parents[0].kind is ArtifactKind.QUESTION
    assert evidence.parents[0].id == question.id


def test_artifact_refs_are_hashable() -> None:
    """Refs are used as dedup keys, so they must be usable in a set."""
    artifact: Artifact = make(ArtifactKind.EVIDENCE)
    assert len({artifact.ref, artifact.ref}) == 1
