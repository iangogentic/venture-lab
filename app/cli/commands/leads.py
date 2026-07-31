"""`op leads` — harvest, browse, and track the people behind the evidence.

A separate command group rather than a tenth stage command, because harvesting is
deliberately outside the pipeline: it runs on demand when a human decides outreach
is worth their time, and a run with zero leads is a finished result rather than
unfinished work. See `app/skills/harvest_leads.py` for the full reasoning.
"""

from typing import Annotated

import typer
from rich.table import Table

from app.artifacts import (
    ArtifactKind,
    ArtifactRegistry,
    ArtifactStatus,
    Lead,
    LeadEngagement,
    LeadIntent,
)
from app.cli.render import artifact_json, print_json
from app.pipeline import PipelineEngine
from app.skills import SkillRequest
from app.skills.harvest_leads import HarvestLeadsSkill
from app.utils.console import console, err_console
from app.utils.errors import ArtifactError, OpportunityEngineError

app = typer.Typer(
    name="leads",
    help="Harvest, browse, and track leads — people who publicly expressed a pain.",
    no_args_is_help=True,
)

RunOption = Annotated[str, typer.Option("--run", "-r", help="Run identifier.")]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit the artifacts as JSON.")]


@app.command("harvest")
def harvest(
    run_id: RunOption = "default",
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Re-harvest, superseding the prior leads.")
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Harvest leads from a run's pain clusters and their cited evidence.

    Runs the `harvest-leads` skill directly — this is not a pipeline stage, so
    `op pipeline` never triggers it. Zero leads is a legitimate outcome.
    """
    registry = ArtifactRegistry()
    engine = PipelineEngine(registry)

    # The same gate every stage reads through: superseded, rejected and archived
    # artifacts must not become leads (mirrors PipelineEngine.inputs_for).
    clusters = engine.consumable_of(ArtifactKind.PAIN_CLUSTER, run_id)
    evidence = engine.consumable_of(ArtifactKind.EVIDENCE, run_id)
    if not clusters:
        err_console.print(
            f"[danger]no pain clusters for run[/danger] {run_id} — "
            "run the pipeline through [stage]cluster-pains[/stage] first"
        )
        raise typer.Exit(code=1)
    if not evidence:
        err_console.print(
            f"[danger]no evidence for run[/danger] {run_id} — "
            "there is nothing to trace the clusters back to"
        )
        raise typer.Exit(code=1)

    existing = engine.produced_of(ArtifactKind.LEAD, run_id)
    if existing and not force:
        console.print(
            f"[muted]leads already harvested[/muted] — {len(existing)} lead(s) for run {run_id}"
        )
        console.print("[muted]re-harvest with[/muted] --force")
        return

    skill = HarvestLeadsSkill(registry=registry)
    request = SkillRequest(
        run_id=run_id,
        artifacts=[*clusters, *evidence],
        question=engine.question_for(run_id),
    )
    try:
        result = skill.execute(request)
    except OpportunityEngineError as exc:
        err_console.print(f"[danger]harvest-leads failed[/danger]: {exc}")
        raise typer.Exit(code=1) from exc

    # Only after the replacements are safely on disk — engine._supersede's spirit.
    if force and existing:
        fresh_ids = {artifact.id for artifact in result.artifacts}
        for old in existing:
            if old.id not in fresh_ids:
                registry.update(old, status=ArtifactStatus.SUPERSEDED)

    leads = [artifact for artifact in result.artifacts if isinstance(artifact, Lead)]
    if as_json:
        print_json(artifact_json(leads))
        return
    if not leads:
        console.print(
            "[muted]no leads[/muted] — the clusters' cited evidence names nobody "
            "with a permalink, or nobody expressed the pain"
        )
        return
    console.print(_leads_table(leads, title=f"harvest-leads — {len(leads)} produced"))


@app.command("list")
def list_leads(
    run_id: Annotated[
        str | None,
        typer.Option("--run", "-r", help="Only leads from this run."),
    ] = None,
    cluster: Annotated[
        str | None,
        typer.Option("--cluster", help="Only leads for this pain cluster id."),
    ] = None,
    intent: Annotated[
        LeadIntent | None,
        typer.Option("--intent", help="Only leads whose words express this."),
    ] = None,
    engagement: Annotated[
        LeadEngagement | None,
        typer.Option("--engagement", help="Only leads at this outreach state."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """List leads newest-first: who said it, where, and how far outreach has got."""
    registry = ArtifactRegistry()
    leads = [
        artifact
        for artifact in registry.find_by_type(ArtifactKind.LEAD, run_id=run_id)
        if isinstance(artifact, Lead)
    ]
    if cluster is not None:
        leads = [lead for lead in leads if lead.cluster.id == cluster]
    if intent is not None:
        leads = [lead for lead in leads if lead.intent is intent]
    if engagement is not None:
        leads = [lead for lead in leads if lead.engagement is engagement]
    leads.sort(key=lambda lead: lead.published_at or lead.created_at, reverse=True)

    if as_json:
        print_json(artifact_json(leads))
        return
    if not leads:
        console.print("[muted]no leads[/muted]")
        return
    console.print(_leads_table(leads, title=f"{len(leads)} lead(s)"))


@app.command("mark")
def mark(
    lead_id: Annotated[str, typer.Argument(help="Lead id, e.g. ld_3f9c…")],
    engagement: Annotated[
        LeadEngagement,
        typer.Argument(
            help="New state. `excluded` is also how a deletion upstream is honoured: "
            "when the source post vanishes, the lead is retired, never retained.",
        ),
    ],
) -> None:
    """Record where outreach with one lead stands, printing old → new."""
    registry = ArtifactRegistry()
    try:
        lead = registry.load_as(ArtifactKind.LEAD, lead_id, Lead)
    except ArtifactError as exc:
        err_console.print(f"[danger]no lead[/danger] {lead_id}")
        raise typer.Exit(code=1) from exc

    previous = lead.engagement
    registry.update(lead, engagement=engagement)
    console.print(
        f"{lead_id} [muted]engagement[/muted] {previous.value} [muted]→[/muted] {engagement.value}"
    )


def _leads_table(leads: list[Lead], *, title: str | None = None) -> Table:
    """One row per lead: identity, reading, outreach state, and the way back."""
    table = Table(title=title, title_style="muted", header_style="stage")
    table.add_column("id", overflow="fold")
    table.add_column("author")
    table.add_column("platform")
    table.add_column("intent")
    table.add_column("engagement")
    table.add_column("date", style="muted")
    table.add_column("quote", overflow="ellipsis", max_width=40)
    table.add_column("url", overflow="fold", style="muted")

    for lead in leads:
        table.add_row(
            lead.id,
            lead.author,
            lead.collector,
            lead.intent.value,
            lead.engagement.value,
            f"{lead.published_at:%Y-%m-%d}" if lead.published_at else "—",
            lead.quote,
            lead.url,
        )
    return table


__all__ = ["app", "harvest", "list_leads", "mark"]
