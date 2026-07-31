"""Lineage as a graph: every artifact wired to what it came from and what came of it.

`parents` is written into an artifact when it is created, so walking *up* has always
been possible. Walking *down* has not, and the interesting questions are all
downward ones: which opportunities rest on this one Reddit thread, what breaks if
this evidence turns out to be wrong, what gets orphaned if it is deleted.

**Children are derived, not stored.** An artifact is written once, before any of its
children exist. Recording a child would mean reopening a finished parent and
rewriting it every time a later stage ran — which would forfeit the immutability the
audit trail depends on. So the inverse edges are computed instead: scan the
workspace, read each artifact once, and invert the refs it carries. A `GraphNode`
exposes `.children`; the JSON file on disk never grows the field.

Edges come from `referenced_ids`, not from `parents` alone, so a link expressed as a
named field — `Opportunity.pain_cluster`, `Signal.supported_by` — is an edge too. A
new artifact type with a new ref field joins the graph without anyone updating a
table here.

Two properties worth knowing before trusting an answer:

* The graph is a snapshot, exactly as fresh as the scan that built it.
* A ref pointing outside the scanned set is dropped, never resurrected as a node.
  Deleted evidence disappears from the graph rather than lingering as a phantom, and
  `--run` scoping cannot drag another run's artifacts in through a stray ref.
"""

from collections import deque
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final, Self

from pydantic import BaseModel, ConfigDict

from app.artifacts.base import ArtifactKind, ArtifactRef
from app.artifacts.lineage import referenced_ids
from app.artifacts.registry import ArtifactRegistry

_KIND_ORDER: Final[Mapping[ArtifactKind, int]] = MappingProxyType(
    {kind: position for position, kind in enumerate(ArtifactKind)}
)
"""Declaration order of `ArtifactKind` is pipeline order, which is how output reads best."""


class GraphNode(BaseModel):
    """One artifact and its immediate neighbours in both directions.

    Serialises to `{"id": …, "kind": …, "parents": [...], "children": [...]}`, where
    each entry is an `ArtifactRef`. Frozen, because a node describes a scan that has
    already happened — mutating one would only ever make it lie about the workspace.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    kind: ArtifactKind
    parents: tuple[ArtifactRef, ...] = ()
    children: tuple[ArtifactRef, ...] = ()

    @property
    def is_root(self) -> bool:
        """Nothing in the scanned set produced this — typically the run's `Question`."""
        return not self.parents

    @property
    def is_leaf(self) -> bool:
        """Nothing was derived from this yet."""
        return not self.children


class ArtifactGraph:
    """Every artifact in a workspace, or in one run, wired up in both directions.

    Pure and read-only: build it from a registry, then ask it questions. It never
    writes, and it holds ids and kinds rather than artifact bodies, so keeping one
    around for the length of a command is cheap.
    """

    def __init__(self, nodes: Iterable[GraphNode], *, run_id: str | None = None) -> None:
        self._nodes: dict[str, GraphNode] = {node.id: node for node in sorted(nodes, key=_node_key)}
        self.run_id = run_id

    @classmethod
    def build(cls, registry: ArtifactRegistry, *, run_id: str | None = None) -> Self:
        """Scan the workspace and invert every ref into a child edge.

        Each artifact is read exactly once. `run_id` narrows the scan to one run;
        omit it for the whole workspace.
        """
        kinds: dict[str, ArtifactKind] = {}
        outgoing: dict[str, set[str]] = {}

        for kind in ArtifactKind:
            for artifact in registry.iter_by_type(kind):
                if run_id is not None and artifact.run_id != run_id:
                    continue
                kinds[artifact.id] = kind
                # A self-reference would be malformed data; drop it rather than let it
                # become a one-node cycle every traversal has to defend against.
                outgoing[artifact.id] = referenced_ids(artifact) - {artifact.id}

        parent_ids: dict[str, set[str]] = {artifact_id: set() for artifact_id in kinds}
        child_ids: dict[str, set[str]] = {artifact_id: set() for artifact_id in kinds}

        for artifact_id, referenced in outgoing.items():
            for target in referenced:
                # Both ends must be present. A dangling ref describes something that is
                # no longer on disk, and inventing a node for it would report a lineage
                # the workspace cannot show you.
                if target not in kinds:
                    continue
                parent_ids[artifact_id].add(target)
                child_ids[target].add(artifact_id)

        return cls(
            (
                GraphNode(
                    id=artifact_id,
                    kind=kind,
                    parents=_refs(parent_ids[artifact_id], kinds),
                    children=_refs(child_ids[artifact_id], kinds),
                )
                for artifact_id, kind in kinds.items()
            ),
            run_id=run_id,
        )

    # ------------------------------------------------------------------ lookup

    def node(self, artifact_id: str) -> GraphNode | None:
        """The node for this id, or `None` if it was not in the scan."""
        return self._nodes.get(artifact_id)

    def nodes(self) -> list[GraphNode]:
        """Every node, in pipeline order then by id."""
        return list(self._nodes.values())

    def __contains__(self, artifact_id: object) -> bool:
        return artifact_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        scope = "workspace" if self.run_id is None else f"run {self.run_id!r}"
        return f"ArtifactGraph({len(self._nodes)} nodes, {scope})"

    # --------------------------------------------------------------- one step

    def parents(self, artifact_id: str) -> list[GraphNode]:
        """What this artifact was derived from. Empty for an unknown id."""
        return self._resolve(artifact_id, upward=True)

    def children(self, artifact_id: str) -> list[GraphNode]:
        """What was derived from this artifact. Empty for an unknown id."""
        return self._resolve(artifact_id, upward=False)

    # ----------------------------------------------------------- many steps

    def ancestors(self, artifact_id: str) -> list[GraphNode]:
        """Everything upstream, breadth-first, nearest first, each node once."""
        return self._walk(artifact_id, upward=True)

    def descendants(self, artifact_id: str) -> list[GraphNode]:
        """Everything downstream, breadth-first, nearest first, each node once."""
        return self._walk(artifact_id, upward=False)

    def ancestors_of_kind(self, artifact_id: str, kind: ArtifactKind) -> list[GraphNode]:
        """Upstream artifacts of one kind, in the same order `ancestors` gives them."""
        return [node for node in self.ancestors(artifact_id) if node.kind is kind]

    def descendants_of_kind(self, artifact_id: str, kind: ArtifactKind) -> list[GraphNode]:
        """Downstream artifacts of one kind, in the same order `descendants` gives them."""
        return [node for node in self.descendants(artifact_id) if node.kind is kind]

    # ------------------------------------------------------------- questions

    def evidence_for(self, artifact_id: str) -> list[GraphNode]:
        """Which evidence supports this — every `Evidence` artifact upstream of it.

        An empty result on a decision is a finding, not a bug: it means the chain
        reaches no observation at all.
        """
        return self.ancestors_of_kind(artifact_id, ArtifactKind.EVIDENCE)

    def impact_of(self, artifact_id: str) -> list[GraphNode]:
        """What deleting this would orphan — everything downstream of it."""
        return self.descendants(artifact_id)

    def path_to_root(self, artifact_id: str) -> list[GraphNode]:
        """One representative chain from this artifact back to the question.

        Ordered `[artifact, …, root]`, as the name reads. There is usually more than
        one path up — an opportunity cites several clusters, each citing several
        briefs — so this returns the shortest chain that reaches a `Question`, and
        falls back to the longest chain that reaches a parentless artifact when no
        question is above it. Enough for the "why does this exist" narrative; use
        `ancestors` when the whole upstream set is wanted.
        """
        start = self._nodes.get(artifact_id)
        if start is None:
            return []

        queue: deque[list[GraphNode]] = deque([[start]])
        seen: set[str] = {start.id}
        fallback: list[GraphNode] = [start]

        while queue:
            path = queue.popleft()
            tip = path[-1]
            if tip.kind is ArtifactKind.QUESTION:
                return path

            onward = [node for node in self.parents(tip.id) if node.id not in seen]
            if not onward:
                if len(path) > len(fallback):
                    fallback = path
                continue

            for parent in onward:
                seen.add(parent.id)
                queue.append([*path, parent])

        return fallback

    # ---------------------------------------------------------------- shape

    def roots(self) -> list[GraphNode]:
        """Nodes nothing in the scan produced — where every chain ends."""
        return [node for node in self._nodes.values() if node.is_root]

    def leaves(self) -> list[GraphNode]:
        """Nodes nothing has been derived from yet — the current frontier."""
        return [node for node in self._nodes.values() if node.is_leaf]

    def to_dict(self) -> dict[str, dict[str, list[str]]]:
        """The whole graph as an adjacency map, for export or diffing.

        Ids only: the kind of any id is available from that id's own entry, and an
        adjacency list stays readable when it is a page long.
        """
        return {
            node.id: {
                "parents": [ref.id for ref in node.parents],
                "children": [ref.id for ref in node.children],
            }
            for node in self._nodes.values()
        }

    # ------------------------------------------------------------- internals

    def _resolve(self, artifact_id: str, *, upward: bool) -> list[GraphNode]:
        node = self._nodes.get(artifact_id)
        if node is None:
            return []
        refs = node.parents if upward else node.children
        # `build` only records edges whose ends both exist, but a hand-assembled graph
        # need not, and a half-resolved neighbour list would be worse than a short one.
        return [self._nodes[ref.id] for ref in refs if ref.id in self._nodes]

    def _walk(self, artifact_id: str, *, upward: bool) -> list[GraphNode]:
        """Breadth-first traversal in one direction, cycle-safe.

        Every id is admitted to `seen` at most once, including the starting id, so a
        malformed workspace whose refs loop terminates instead of hanging.
        """
        start = self._nodes.get(artifact_id)
        if start is None:
            return []

        seen: set[str] = {start.id}
        queue: deque[GraphNode] = deque([start])
        found: list[GraphNode] = []

        while queue:
            current = queue.popleft()
            for neighbour in self._resolve(current.id, upward=upward):
                if neighbour.id in seen:
                    continue
                seen.add(neighbour.id)
                found.append(neighbour)
                queue.append(neighbour)
        return found


def _refs(ids: Iterable[str], kinds: Mapping[str, ArtifactKind]) -> tuple[ArtifactRef, ...]:
    """Turn ids into sorted refs. Sorted so two scans of one workspace render alike."""
    return tuple(sorted((ArtifactRef(kind=kinds[i], id=i) for i in ids), key=_ref_key))


def _ref_key(ref: ArtifactRef) -> tuple[int, str]:
    return (_KIND_ORDER[ref.kind], ref.id)


def _node_key(node: GraphNode) -> tuple[int, str]:
    return (_KIND_ORDER[node.kind], node.id)


__all__ = ["ArtifactGraph", "GraphNode"]
