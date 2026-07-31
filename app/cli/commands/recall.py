"""`op recall` — search everything past runs kept, by meaning rather than words.

The workspace answers "what did run X keep"; memory answers "have my sources
ever said something like this". Recall embeds the query with the same model
that embedded the evidence and returns the nearest excerpts across every run.
"""

from typing import Annotated

import typer
from rich.table import Table

from app.cli.render import print_json
from app.config import get_settings
from app.memory import Embedder, MemoryStore, RecallHit, default_embedder
from app.utils.console import console, err_console
from app.utils.errors import MemoryUnavailableError
from app.utils.paths import get_workspace_paths

LimitOption = Annotated[int, typer.Option("--limit", "-n", min=1, help="How many hits to show.")]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit JSON and nothing else.")]


def _build_embedder() -> Embedder:
    """Seam for tests: monkeypatched so `op recall` never downloads the model."""
    return default_embedder()


def recall(
    text: Annotated[str, typer.Argument(help="What to look for, in your own words.")],
    limit: LimitOption = 10,
    as_json: JsonOption = False,
) -> None:
    """Search remembered evidence from every run, nearest first.

    An empty or unavailable memory exits 0: nothing remembered yet is a state
    the first run fixes, not an error — and in `--json` mode stdout still
    carries a well-formed (empty) payload so pipelines never parse prose.
    """
    if not get_settings().memory_enabled:
        _nothing("memory is disabled — set MEMORY_ENABLED=true to build one", as_json)
        return

    try:
        store = MemoryStore(get_workspace_paths().memory_db)
    except MemoryUnavailableError as exc:
        _nothing(f"memory is unavailable here: {exc}", as_json)
        return

    try:
        if store.count() == 0:
            _nothing("memory is empty — run `op collect` and kept evidence will land here", as_json)
            return
        try:
            vector = _build_embedder().encode([text])[0]
        except MemoryUnavailableError as exc:
            _nothing(f"memory is unavailable here: {exc}", as_json)
            return
        hits = store.recall(vector, limit=limit)
    finally:
        store.close()

    if as_json:
        print_json([hit.model_dump(mode="json") for hit in hits])
        return
    console.print(_hits_table(hits, text))


def _nothing(reason: str, as_json: bool) -> None:
    """Report an absent memory on stderr, keeping stdout pure in `--json` mode."""
    err_console.print(f"[muted]{reason}[/muted]")
    if as_json:
        print_json([])


def _hits_table(hits: list[RecallHit], query: str) -> Table:
    table = Table(title=f"nearest to {query!r}", title_style="muted", header_style="stage")
    table.add_column("run", style="muted")
    table.add_column("collector")
    table.add_column("distance", justify="right")
    table.add_column("excerpt", overflow="ellipsis", max_width=60)
    table.add_column("url", overflow="fold", style="muted")

    for hit in hits:
        table.add_row(
            hit.run_id or "-",
            hit.collector or "-",
            f"{hit.distance:.3f}",
            (hit.excerpt or "").strip(),
            hit.url or "",
        )
    return table


__all__ = ["recall"]
