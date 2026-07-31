"""`op graph`, `op why`, `op impact` — read lineage without re-running anything.

Three questions the workspace can already answer, if you invert its refs:

* `op graph <id>`  — what is this wired to?
* `op why <id>`    — why does it exist? The chain down from the question, then the
  evidence it rests on.
* `op impact <id>` — what depends on it? What deleting it would orphan.

All three build the same snapshot (`app.artifacts.graph`) and none of them write, so
they are safe to run mid-pipeline. `--json` keeps stdout to one JSON document —
scan chatter and errors go to stderr — so any of them can be piped into `jq`.
"""

from collections.abc import Sequence
from typing import Annotated, Any, Final

import typer
from rich.table import Table
from rich.tree import Tree

from app.artifacts import Artifact, ArtifactKind, ArtifactRegistry
from app.artifacts.graph import ArtifactGraph, GraphNode
from app.cli.render import artifact_table, print_json
from app.utils.console import console, err_console
from app.utils.errors import ArtifactError

IdArgument = Annotated[str, typer.Argument(help="Artifact id.")]
RunOption = Annotated[
    str | None,
    typer.Option("--run", "-r", help="Only artifacts from this run. Omit to scan the workspace."),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit JSON and nothing else.")]

_HEADLINE_FIELDS: Final[tuple[str, ...]] = (
    "title",
    "label",
    "text",
    "summary",
    "objective",
    "rationale",
    "excerpt",
)
"""Probed in order for a tree label. No kind is required to have any of them."""

_ANALYSIS_NOUNS: Final[dict[ArtifactKind, tuple[str, str]]] = {
    ArtifactKind.MARKET_ANALYSIS: ("market analysis", "market analyses"),
    ArtifactKind.COMPETITION_ANALYSIS: ("competition analysis", "competition analyses"),
    ArtifactKind.CONTRADICTION_ANALYSIS: ("contradiction analysis", "contradiction analyses"),
}
"""Kinds whose directory name is an adjective, not a countable noun.

Both forms are spelled out rather than derived: "analyses" does not singularise by
any rule the other kinds follow.
"""


def graph_show(
    artifact_id: IdArgument,
    run_id: RunOption = None,
    depth: Annotated[
        int,
        typer.Option("--depth", "-d", min=1, help="Levels of parents and children to expand."),
    ] = 1,
    as_json: JsonOption = False,
) -> None:
    """Show one artifact's place in the lineage graph: parents above, children below."""
    registry, graph = _load_graph(run_id)
    node = _require(registry, graph, artifact_id, run_id)

    upward = _branch(graph, node.id, upward=True, depth=depth, seen={node.id})
    downward = _branch(graph, node.id, upward=False, depth=depth, seen={node.id})

    if as_json:
        print_json(
            {
                "id": node.id,
                "kind": node.kind.value,
                "parents": upward,
                "children": downward,
            }
        )
        return

    tree = Tree(f"[stage]{node.id}[/stage] [muted]{node.kind.value}[/muted]")
    _grow(tree.add(_branch_label("parents", len(upward))), upward, "parents")
    _grow(tree.add(_branch_label("children", len(downward))), downward, "children")
    console.print(tree)


def why(
    artifact_id: IdArgument,
    run_id: RunOption = None,
    as_json: JsonOption = False,
) -> None:
    """Explain why an artifact exists: the chain from the question, and its evidence."""
    registry, graph = _load_graph(run_id)
    node = _require(registry, graph, artifact_id, run_id)

    chain = list(reversed(graph.path_to_root(node.id)))
    evidence = graph.evidence_for(node.id)

    if as_json:
        print_json(
            {
                "id": node.id,
                "kind": node.kind.value,
                "chain": [_as_ref(step) for step in chain],
                "evidence": [_as_ref(step) for step in evidence],
                "evidence_count": len(evidence),
            }
        )
        return

    console.print(f"[stage]{node.id}[/stage] [muted]{node.kind.value}[/muted] exists because:")

    if len(chain) == 1:
        console.print("[warning]nothing above it[/warning] — this artifact records no lineage")
    else:
        console.print(_chain_tree(registry, chain, target=node.id))
        if chain[0].kind is not ArtifactKind.QUESTION:
            stops_at = chain[0].kind.value
            console.print(f"[muted]the chain stops at a {stops_at}; no question above it[/muted]")

    if not evidence:
        console.print("[warning]no evidence underneath it[/warning] — nothing observed supports it")
        return

    resting = _count(len(evidence), "piece")
    console.print(f"\nresting on [info]{resting}[/info] of evidence")
    console.print(artifact_table(_artifacts(registry, evidence)))


def impact(
    artifact_id: IdArgument,
    run_id: RunOption = None,
    as_json: JsonOption = False,
) -> None:
    """Show everything that depends on an artifact — what deleting it would orphan."""
    registry, graph = _load_graph(run_id)
    node = _require(registry, graph, artifact_id, run_id)

    affected = graph.impact_of(node.id)
    grouped: dict[ArtifactKind, list[str]] = {}
    for downstream in affected:
        grouped.setdefault(downstream.kind, []).append(downstream.id)
    # Pipeline order, so the reader walks the run forwards rather than in scan order.
    ordered = [(kind, sorted(grouped[kind])) for kind in ArtifactKind if kind in grouped]

    if as_json:
        print_json(
            {
                "id": node.id,
                "kind": node.kind.value,
                "orphaned": len(affected),
                "by_kind": {kind.value: len(ids) for kind, ids in ordered},
                "descendants": [
                    {"kind": kind.value, "id": downstream_id}
                    for kind, ids in ordered
                    for downstream_id in ids
                ],
            }
        )
        return

    if not affected:
        console.print(f"[success]nothing depends on[/success] {node.id} — deleting it is safe")
        return

    breakdown = ", ".join(_plural(kind, len(ids)) for kind, ids in ordered)
    console.print(
        f"deleting [stage]{node.id}[/stage] would orphan "
        f"[danger]{_count(len(affected), 'artifact')}[/danger] — {breakdown}"
    )
    console.print(_impact_table(ordered))


# ------------------------------------------------------------------- plumbing


def _load_graph(run_id: str | None) -> tuple[ArtifactRegistry, ArtifactGraph]:
    """Scan the workspace once. The scan note goes to stderr so `--json` stays clean."""
    registry = ArtifactRegistry()
    try:
        graph = ArtifactGraph.build(registry, run_id=run_id)
    except ArtifactError as exc:
        err_console.print(f"[danger]cannot read the workspace[/danger] {exc}")
        raise typer.Exit(code=1) from exc

    scope = "workspace" if run_id is None else f"run {run_id}"
    err_console.print(f"[muted]scanned {len(graph)} artifacts ({scope})[/muted]")
    return registry, graph


def _require(
    registry: ArtifactRegistry,
    graph: ArtifactGraph,
    artifact_id: str,
    run_id: str | None,
) -> GraphNode:
    """The node, or exit 1 — distinguishing "no such artifact" from "not in this run"."""
    node = graph.node(artifact_id)
    if node is not None:
        return node

    if run_id is not None and registry.locate(artifact_id) is not None:
        err_console.print(f"[danger]not in run[/danger] {run_id}: {artifact_id}")
    else:
        err_console.print(f"[danger]no artifact[/danger] {artifact_id}")
    raise typer.Exit(code=1)


def _artifacts(registry: ArtifactRegistry, nodes: Sequence[GraphNode]) -> list[Artifact]:
    """Load the bodies behind these nodes, skipping any that vanished since the scan."""
    loaded: list[Artifact] = []
    for node in nodes:
        try:
            loaded.append(registry.load(node.kind, node.id))
        except ArtifactError:
            continue
    return loaded


def _as_ref(node: GraphNode) -> dict[str, str]:
    """A node in the shape an `ArtifactRef` serialises to, for the `--json` payloads."""
    return {"kind": node.kind.value, "id": node.id}


def _branch(
    graph: ArtifactGraph,
    artifact_id: str,
    *,
    upward: bool,
    depth: int,
    seen: set[str],
) -> list[dict[str, Any]]:
    """Neighbours in one direction, nested `depth` levels, as plain dicts.

    One structure feeds both the JSON and the tree, so they cannot disagree. A node
    already expanded elsewhere is marked `truncated` rather than repeated — an
    artifact cited by six others would otherwise be printed six times over.
    """
    key = "parents" if upward else "children"
    entries: list[dict[str, Any]] = []

    for neighbour in _step(graph, artifact_id, upward=upward):
        entry: dict[str, Any] = {"id": neighbour.id, "kind": neighbour.kind.value}
        if not _step(graph, neighbour.id, upward=upward):
            entry[key] = []
        elif depth <= 1 or neighbour.id in seen:
            entry["truncated"] = True
        else:
            seen.add(neighbour.id)
            entry[key] = _branch(graph, neighbour.id, upward=upward, depth=depth - 1, seen=seen)
        entries.append(entry)

    return entries


def _step(graph: ArtifactGraph, artifact_id: str, *, upward: bool) -> list[GraphNode]:
    return graph.parents(artifact_id) if upward else graph.children(artifact_id)


def _grow(branch: Tree, entries: Sequence[dict[str, Any]], key: str) -> None:
    """Render `_branch` output into a Rich tree."""
    for entry in entries:
        label = f"[info]{entry['id']}[/info] [muted]{entry['kind']}[/muted]"
        if entry.get("truncated"):
            label = f"{label} [muted]…[/muted]"
        node = branch.add(label)
        nested = entry.get(key)
        if isinstance(nested, list):
            _grow(node, nested, key)


def _branch_label(name: str, count: int) -> str:
    style = "muted" if count == 0 else "info"
    return f"[{style}]{name}[/{style}] [muted]({count})[/muted]"


def _chain_tree(
    registry: ArtifactRegistry,
    chain: Sequence[GraphNode],
    *,
    target: str,
) -> Tree:
    """The derivation as nesting: each step sits inside the one it came from."""
    bodies = {artifact.id: artifact for artifact in _artifacts(registry, chain)}

    root = Tree(_step_label(chain[0], bodies, target=target))
    cursor = root
    for step in chain[1:]:
        cursor = cursor.add(_step_label(step, bodies, target=target))
    return root


def _step_label(node: GraphNode, bodies: dict[str, Artifact], *, target: str) -> str:
    label = f"[info]{node.id}[/info] [muted]{node.kind.value}[/muted]"
    gist = _describe(bodies.get(node.id))
    if gist:
        label = f"{label} — {gist}"
    if node.id == target:
        label = f"{label}  [success]← this[/success]"
    return label


def _describe(artifact: Artifact | None, *, width: int = 64) -> str:
    """A one-line gist for a tree label: whitespace collapsed, then truncated.

    No two kinds name their headline the same way, so the usual fields are probed.
    A blank result is fine — the id is the part that identifies the step.
    """
    if artifact is None:
        return ""
    for field in _HEADLINE_FIELDS:
        value = getattr(artifact, field, None)
        if isinstance(value, str) and value.strip():
            gist = " ".join(value.split())
            return gist if len(gist) <= width else f"{gist[: width - 1]}…"
    return ""


def _impact_table(grouped: Sequence[tuple[ArtifactKind, list[str]]]) -> Table:
    table = Table(header_style="stage")
    table.add_column("kind")
    table.add_column("count", justify="right")
    table.add_column("ids", overflow="fold")

    for kind, ids in grouped:
        table.add_row(kind.value, str(len(ids)), ", ".join(ids))
    return table


def _plural(kind: ArtifactKind, count: int) -> str:
    """`2 briefs`, `1 market analysis` — a countable noun for each kind.

    The workspace directory name is already plural and covers most kinds, but the
    three analysis directories are adjectives on their own ("2 market"), so those
    get spelled out.
    """
    if kind in _ANALYSIS_NOUNS:
        singular, plural = _ANALYSIS_NOUNS[kind]
        return f"{count} {singular if count == 1 else plural}"

    word = kind.directory
    if count == 1:
        word = f"{word[:-3]}y" if word.endswith("ies") else word.removesuffix("s")
    return f"{count} {word}"


def _count(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


__all__ = ["graph_show", "impact", "why"]
