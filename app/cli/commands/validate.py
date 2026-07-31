"""`op validate` — turn a decided opportunity into a runnable validation experiment.

Deterministic on purpose: no model is called. Everything on the public page is
either drawn verbatim from artifacts a human can audit — the opportunity's own
fields, the pain cluster's verified quotes — or written in
`app/validation/templates.py` where a reviewer can read it. A landing page is a
claim made in the founder's name; generating one at scaffold time would put
unreviewed words in their mouth.
"""

from pathlib import Path
from typing import Annotated

import typer

from app.artifacts import (
    ArtifactKind,
    ArtifactRegistry,
    Decision,
    Evidence,
    Lead,
    LeadEngagement,
    Opportunity,
    PainCluster,
)
from app.pipeline import PipelineEngine
from app.utils.console import console, err_console
from app.utils.errors import ArtifactError
from app.validation import attributed_quotes, build_scaffold

app = typer.Typer(
    name="validate",
    help="Scaffold validation experiments for decided opportunities.",
    no_args_is_help=True,
)


@app.command("scaffold")
def scaffold(
    opportunity_id: Annotated[str, typer.Argument(help="Opportunity id, e.g. op_3f9c…")],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Directory to write into. Default: workspace/validation/<id>/"),
    ] = None,
) -> None:
    """Write a landing page, a pre-registered validation plan, and deploy notes.

    The plan's thresholds are blank on purpose — they are filled in by hand
    before launch, because thresholds set after the numbers arrive rationalize
    any result.
    """
    registry = ArtifactRegistry()
    found = registry.find_by_id(opportunity_id)
    if found is None:
        err_console.print(f"[danger]no artifact[/danger] {opportunity_id}")
        raise typer.Exit(code=1)
    if not isinstance(found, Opportunity):
        err_console.print(
            f"[danger]not an opportunity[/danger] — {opportunity_id} is a {type(found).kind.value}"
        )
        raise typer.Exit(code=1)

    engine = PipelineEngine(registry)
    files = build_scaffold(
        found,
        quotes=attributed_quotes(_cluster_quotes(registry, found), _sources(engine, found)),
        decision=_decision_for(engine, found),
    )

    target = out if out is not None else registry.paths.root / "validation" / found.id
    target.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (target / name).write_text(content, encoding="utf-8")

    console.print(f"[success]scaffold written[/success] {target}")
    for name in files:
        console.print(f"[muted]  {name}[/muted]")
    console.print(
        "[muted]fill in validation-plan.md before the page goes live, "
        "and replace the placeholders named in its README[/muted]"
    )


def _cluster_quotes(registry: ArtifactRegistry, opportunity: Opportunity) -> list[str]:
    """The verbatim quotes of the opportunity's pain cluster, when it has one.

    A missing or unloadable cluster costs the page its quotes section, never the
    scaffold: the experiment still runs, just without social proof.
    """
    if opportunity.pain_cluster is None:
        return []
    try:
        cluster = registry.resolve(opportunity.pain_cluster)
    except ArtifactError:
        return []
    return cluster.quotes if isinstance(cluster, PainCluster) else []


def _sources(engine: PipelineEngine, opportunity: Opportunity) -> list[tuple[str, str]]:
    """`(text, platform)` pairs the quotes can be attributed against.

    Leads first — a lead's quote is exactly the passage a cluster quotes — then
    the run's evidence. Leads excluded upstream (deleted posts, opt-outs) are
    skipped: even a platform name should not be derived from content the person
    withdrew.
    """
    pairs: list[tuple[str, str]] = []
    for lead in engine.consumable_of(ArtifactKind.LEAD, opportunity.run_id):
        if isinstance(lead, Lead) and lead.engagement is not LeadEngagement.EXCLUDED:
            pairs.append((lead.quote, lead.collector))
    for item in engine.consumable_of(ArtifactKind.EVIDENCE, opportunity.run_id):
        if isinstance(item, Evidence):
            pairs.append((item.excerpt, item.collector))
    return pairs


def _decision_for(engine: PipelineEngine, opportunity: Opportunity) -> Decision | None:
    """The current decision ruling on this opportunity, if one exists."""
    return next(
        (
            artifact
            for artifact in engine.consumable_of(ArtifactKind.DECISION, opportunity.run_id)
            if isinstance(artifact, Decision) and artifact.opportunity.id == opportunity.id
        ),
        None,
    )


__all__ = ["app", "scaffold"]
