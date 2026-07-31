"""`op routes` — show which model serves which stage, and what it can do."""

from typing import Annotated

import typer
from rich.table import Table

from app.cli.render import print_json
from app.llm.catalog import ModelCatalog
from app.llm.roles import TIER_SPECS, Capability, resolve_all_tiers, tier_for
from app.llm.routing import STAGE_TEMPERATURES, get_catalogue, get_router
from app.pipeline import STAGE_ORDER
from app.utils.console import console


def routes(
    show_tiers: Annotated[
        bool, typer.Option("--tiers", help="Show what each capability and tier resolves to.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the table as JSON.")] = False,
) -> None:
    """Show the routing table: role, resolved model, temperature and capabilities.

    The fastest way to answer "why did that stage cost so much", and to confirm a
    role resolved to the model you expected before spending a run finding out.
    """
    catalogue = get_catalogue()
    resolved = get_router().table(STAGE_ORDER)

    if as_json:
        print_json(
            {
                "catalogue": len(catalogue) if catalogue else 0,
                "tiers": {t.value: s for t, s in resolve_all_tiers(catalogue).items()},
                "stages": [route.model_dump(mode="json") for route in resolved],
            }
        )
        return

    if show_tiers:
        _print_tiers(catalogue)
        return

    table = Table(header_style="stage")
    table.add_column("#", justify="right", style="muted")
    table.add_column("stage")
    table.add_column("capability")
    table.add_column("tier", style="muted")
    table.add_column("model")
    table.add_column("temp", justify="right")
    table.add_column("json")
    table.add_column("cache")

    for index, route in enumerate(resolved, start=1):
        temperature = "—" if route.temperature is None else f"{route.temperature:.1f}"
        table.add_row(
            str(index),
            route.task or "",
            route.capability.value,
            route.tier.value,
            route.model,
            temperature,
            _json_support(route.supports_structured_outputs, route.supports_response_format),
            "[success]yes[/success]" if route.supports_prompt_caching else "[muted]no[/muted]",
        )

    console.print(table)
    if catalogue is None:
        console.print(
            "[warning]catalogue unavailable[/warning] [muted]— using pinned model names[/muted]"
        )
    else:
        console.print(f"[muted]resolved against {len(catalogue)} live models[/muted]")


def _json_support(structured: bool, response_format: bool) -> str:
    """How strictly this model can be held to a schema."""
    if structured:
        return "[success]schema[/success]"
    if response_format:
        return "[warning]object[/warning]"
    return "[danger]none[/danger]"


def _print_tiers(catalogue: ModelCatalog | None) -> None:
    """Capability -> tier -> slug, with how many stages depend on each."""
    router = get_router()
    usage: dict[str, int] = {}
    for stage in STAGE_TEMPERATURES:
        tier = router.resolve(stage).tier.value
        usage[tier] = usage.get(tier, 0) + 1

    capabilities = Table(header_style="stage", title="capability -> tier", title_style="muted")
    capabilities.add_column("capability")
    capabilities.add_column("tier")
    for capability in Capability:
        capabilities.add_row(capability.value, tier_for(capability).value)
    console.print(capabilities)

    tiers = Table(header_style="stage", title="tier -> model", title_style="muted")
    tiers.add_column("tier")
    tiers.add_column("resolves to")
    tiers.add_column("pinned fallback", style="muted")
    tiers.add_column("stages", justify="right")
    for tier, slug in resolve_all_tiers(catalogue).items():
        pinned = TIER_SPECS[tier].pinned
        tiers.add_row(
            tier.value,
            slug if slug == pinned else f"[success]{slug}[/success]",
            pinned,
            str(usage.get(tier.value, 0)),
        )
    console.print(tiers)
    console.print("[muted]green = newer than the pinned fallback[/muted]")


__all__ = ["routes"]
