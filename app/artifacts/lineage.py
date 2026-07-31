"""Working out which artifacts belong in one call's context.

The naive approach — hand a stage every artifact of the kinds it consumes — costs
more with every run and gets worse exactly when a project is going well. By the
time there are two hundred pieces of evidence, `decision` is being asked to read
all of them to rule on one opportunity it has already been given a summary of.

So context is assembled by **lineage** instead: the artifact being worked on, the
things that were derived from it, and the question that started the run. Nothing
else. A decision about one opportunity sees that opportunity's market analysis,
competition analysis and contradiction analysis — not another opportunity's, and
not the evidence three stages upstream.

Relatedness is read off the artifacts themselves. Every cross-artifact link in
this system is an `ArtifactRef`, whether in `parents` or in a named field like
`MarketAnalysis.opportunity`, so "does A point at B" is answerable by walking A's
serialised form. That means a new artifact type with a new ref field is picked up
here without anyone remembering to update a table.
"""

from collections.abc import Iterable, Sequence
from typing import Any

from app.artifacts.base import Artifact, ArtifactKind


def referenced_ids(artifact: Artifact) -> set[str]:
    """Every artifact id this one points at, however deeply nested.

    Walks the serialised form rather than the model fields so refs inside
    sub-models — a `Signal.supported_by`, a `Conflict.sources` — are found too.
    """
    found: set[str] = set()
    _collect_refs(artifact.model_dump(mode="json"), found)
    return found


def _collect_refs(node: Any, found: set[str]) -> None:
    """Recursively gather anything shaped like a serialised `ArtifactRef`."""
    if isinstance(node, dict):
        kind = node.get("kind")
        ref_id = node.get("id")
        # An ArtifactRef serialises to exactly {"kind": ..., "id": ...}; a full
        # artifact has many more keys, so the length check is what tells them apart.
        if isinstance(ref_id, str) and isinstance(kind, str) and len(node) == 2:
            found.add(ref_id)
        for value in node.values():
            _collect_refs(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, found)


def is_related(candidate: Artifact, subject: Artifact) -> bool:
    """Whether two artifacts are linked in either direction."""
    if candidate.id == subject.id:
        return False
    return subject.id in referenced_ids(candidate) or candidate.id in referenced_ids(subject)


def related_to(
    subject: Artifact,
    pool: Iterable[Artifact],
    *,
    kinds: Sequence[ArtifactKind] | None = None,
) -> list[Artifact]:
    """Artifacts from `pool` that are linked to `subject`, optionally filtered by kind."""
    wanted = set(kinds) if kinds is not None else None
    return [
        artifact
        for artifact in pool
        if (wanted is None or type(artifact).kind in wanted) and is_related(artifact, subject)
    ]


def supporting(
    subject: Artifact,
    pool: Iterable[Artifact],
) -> list[Artifact]:
    """Only the artifacts `subject` explicitly cites.

    Narrower than `related_to`: this is the "include the evidence ids this artifact
    actually references, not every collected record" case.
    """
    cited = referenced_ids(subject)
    return [artifact for artifact in pool if artifact.id in cited]


__all__ = ["is_related", "referenced_ids", "related_to", "supporting"]
