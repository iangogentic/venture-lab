"""`op doctor` — every limit that could stop a run, checked before one starts.

Three runs in a row died on a limit that was knowable in advance: a context window,
an unaffordable `max_tokens`, an unaffordable prompt. Each was discovered by failing,
which is the most expensive way to learn it. This command asks the same questions
locally, so the next surprise is about research quality rather than configuration.

The context estimate is why this exists. It runs the real collectors and then the
stage's own preview and budget logic, so the number reported here is the number
`collect-evidence` would send — not a second implementation of it that can drift.

Nothing here prints a credential. Presence, never value.
"""

import os
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Final

import httpx
import typer
from pydantic import BaseModel, ConfigDict, Field
from rich.table import Table

from app.artifacts import ArtifactKind, ArtifactRegistry, Question
from app.cli.render import print_json
from app.collectors import COLLECTORS, SourceItem, available, config_from_settings
from app.config import Settings, get_settings
from app.llm.catalog import ModelCatalog, ModelInfo
from app.llm.provider import Provider
from app.llm.routing import ModelRouter, ResolvedRoute, get_catalogue
from app.llm.service import estimate_tokens
from app.pipeline import STAGE_ORDER
from app.prompts import load_prompt
from app.skills.collect_evidence import CollectEvidenceInput, CollectEvidenceSkill
from app.utils.console import console, err_console
from app.utils.errors import ArtifactError, CollectorError, ConfigurationError
from app.utils.paths import get_workspace_paths
from app.utils.time import utcnow

KEY_URL: Final[str] = "https://openrouter.ai/api/v1/key"
"""What the key can afford. Unlike the models endpoint, this one needs the key."""

KEY_TIMEOUT: Final[float] = 10.0

COLLECT_STAGE: Final[str] = "collect-evidence"
"""The stage whose context this command estimates: the first, and the largest."""

DEFAULT_QUERY: Final[str] = "developer workflow friction"
"""Stand-in when a run has no Question yet — the probe needs something to search."""

CREDIT_LABEL: Final[str] = "credit"
"""Marks the ceiling that comes from the balance rather than from a setting."""

LOCAL_LABEL: Final[str] = "MAX_INPUT_TOKENS"
"""Marks the ceiling this application enforces on itself, before the gateway sees it."""

EXPIRY_WARNING_HOURS: Final[float] = 48.0
"""A key dying mid-pipeline is a run lost; two days is enough notice to replace it."""

NEAR_LIMIT: Final[float] = 0.9
"""Close enough to a ceiling to be worth saying before it is crossed."""


class Status(StrEnum):
    """How a check went. Only `FAIL` stops a run, and only `FAIL` exits non-zero."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


STATUS_STYLE: dict[Status, str] = {
    Status.OK: "success",
    Status.WARN: "warning",
    Status.FAIL: "danger",
}


class Check(BaseModel):
    """One thing that could stop a run, and what to change if it would."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: Status
    detail: str = Field(description="One line on what was found.")
    remedy: str = Field(default="", description="The setting to change, when there is one.")


class KeyLimits(BaseModel):
    """What the gateway says the key can still afford. Every field is optional."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    limit: float | None = None
    usage: float | None = None
    remaining: float | None = None
    is_free_tier: bool | None = None
    rate_limit: str | None = None
    expires_at: datetime | None = None


class Probe(BaseModel):
    """What a live search would put in front of the model at `collect-evidence`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ran: tuple[str, ...] = ()
    fetched: int = 0
    kept: int = Field(default=0, description="Candidates surviving the stage's own trimming.")
    tokens: int = 0
    floor: int = Field(
        default=0,
        description="Tokens sent before a single candidate: philosophy, prompt, question.",
    )


def doctor(
    run_id: Annotated[
        str, typer.Option("--run", "-r", help="Run whose question the probe searches for.")
    ] = "default",
    probe: Annotated[
        bool, typer.Option("--probe/--no-probe", help="Search live sources to size the context.")
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the checks as JSON.")] = False,
) -> None:
    """Report everything that could stop a run, before one is paid for.

    The probe searches the real collectors with the run's own question, then measures
    what the first stage would send. It costs a few seconds and no money; `--no-probe`
    skips it and leaves the context estimate unmeasured.

    Exits 1 only for something that would actually stop a run. Warnings are for
    reading, not for failing.
    """
    settings = get_settings()
    chatter = err_console if as_json else console

    # Resolve once from pinned tiers before deciding whether OpenRouter is involved.
    # A native-only doctor must not contact an unrelated gateway just to learn that
    # every active route is GPT. Tier vendors do not change when a live catalogue
    # supplies a newer slug, so this is enough to choose the transport safely.
    routes, unroutable = _resolve_routes(None)
    openrouter_stages = _openrouter_stages(settings, routes)
    if openrouter_stages:
        catalogue = get_catalogue()
        routes, unroutable = _resolve_routes(catalogue)
        openrouter_stages = _openrouter_stages(settings, routes)
    else:
        catalogue = None

    key_check, limits = _key_limits_check(settings, openrouter_stages)
    availability = _availability(settings)

    measured: Probe | None = None
    note = "not measured (--no-probe)"
    if probe:
        chatter.print("[muted]searching live sources — a few seconds…[/muted]")
        try:
            measured = _run_probe(settings, run_id, availability)
            note = ""
        except (ConfigurationError, ArtifactError) as exc:
            note = f"could not estimate: {_short(str(exc))}"

    route = routes.get(COLLECT_STAGE)
    info = catalogue.get(route.model) if catalogue is not None and route is not None else None
    openrouter_remaining = limits.remaining if limits is not None else None
    context_remaining = (
        limits.remaining
        if limits is not None and route is not None and _uses_openrouter(settings, route)
        else None
    )
    affordable = _affordable(context_remaining, _price(info, "prompt"))

    checks = [
        _api_key_check(settings, routes),
        _optional_credentials_check(settings),
        key_check,
        _context_check(settings, route, affordable, measured, note),
        _output_check(settings, routes, catalogue, openrouter_remaining),
        _routing_check(
            routes,
            unroutable,
            catalogue,
            openrouter_catalogue_applicable=bool(openrouter_stages),
        ),
        _collectors_check(settings, availability),
        _workspace_check(),
    ]
    overall = _worst(checks)

    if as_json:
        print_json(
            {
                "status": overall.value,
                "probed": measured is not None,
                "checks": [check.model_dump(mode="json") for check in checks],
            }
        )
    else:
        console.print(_table(checks))
        console.print(_summary(checks))

    if overall is Status.FAIL:
        raise typer.Exit(code=1)


# ------------------------------------------------------------------- the checks


def _uses_native_openai(settings: Settings, route: ResolvedRoute) -> bool:
    """Whether one resolved route reaches GPT through the direct OpenAI transport."""
    if route.provider is not Provider.GPT:
        return False
    if settings.llm_transport == "openai":
        return True
    if settings.llm_transport == "openrouter":
        return False
    return settings.openai_api_key is not None


def _uses_openrouter(settings: Settings, route: ResolvedRoute) -> bool:
    """Every route except direct GPT is still served by OpenRouter."""
    return not _uses_native_openai(settings, route)


def _native_openai_stages(
    settings: Settings,
    routes: dict[str, ResolvedRoute],
) -> tuple[str, ...]:
    return tuple(stage for stage, route in routes.items() if _uses_native_openai(settings, route))


def _openrouter_stages(
    settings: Settings,
    routes: dict[str, ResolvedRoute],
) -> tuple[str, ...]:
    return tuple(stage for stage, route in routes.items() if _uses_openrouter(settings, route))


def _api_key_check(
    settings: Settings,
    routes: dict[str, ResolvedRoute] | None = None,
) -> Check:
    """Check only keys used by active routes, and reveal presence rather than values."""
    resolved = routes or {}
    native = _native_openai_stages(settings, resolved)
    gateway = _openrouter_stages(settings, resolved)

    # Preserve the helper's useful standalone behaviour if routing itself could not
    # resolve. Doctor reports that separate failure, but the key row should still say
    # which credential the configured default provider would use.
    if not resolved:
        native_default = settings.llm_provider == Provider.GPT.value and (
            settings.llm_transport == "openai"
            or (settings.llm_transport == "auto" and settings.openai_api_key is not None)
        )
        native = ("configured GPT provider",) if native_default else ()
        gateway = () if native_default else ("configured provider",)

    missing: list[str] = []
    remedies: list[str] = []
    if native and settings.openai_api_key is None:
        missing.append(f"OPENAI_API_KEY is not set (native GPT: {_listed(native)})")
        remedies.append("set OPENAI_API_KEY")
    if gateway and settings.openrouter_api_key is None:
        if native:
            missing.append(f"OPENROUTER_API_KEY is not set (OpenRouter routes: {_listed(gateway)})")
        else:
            # This was the original, OpenRouter-only result. Keep it stable for
            # Claude, Gemini, and explicitly gateway-routed GPT configurations.
            missing.append("OPENROUTER_API_KEY is not set")
        remedies.append("set OPENROUTER_API_KEY")

    if missing:
        return Check(
            name="api key",
            status=Status.FAIL,
            detail="; ".join(missing),
            remedy=(
                "copy .env.example to .env and fill it in"
                if remedies == ["set OPENROUTER_API_KEY"]
                else "; ".join(remedies)
            ),
        )

    configured: list[str] = []
    if native:
        configured.append("OPENAI_API_KEY is set")
    if gateway:
        configured.append("OPENROUTER_API_KEY is set")
    return Check(name="api key", status=Status.OK, detail="; ".join(configured))


def _optional_credentials_check(settings: Settings) -> Check:
    """Sources that degrade without credentials rather than break."""
    github = settings.github_token is not None
    reddit = settings.reddit_client_id is not None and settings.reddit_client_secret is not None

    detail = "; ".join(
        (
            f"github {'token set' if github else 'anonymous, ~10 req/min'}",
            f"reddit {'configured' if reddit else 'skipped'}",
        )
    )
    if github and reddit:
        return Check(name="optional creds", status=Status.OK, detail=detail)

    remedies: list[str] = []
    if not github:
        remedies.append("GITHUB_TOKEN lifts search to ~30 req/min")
    if not reddit:
        remedies.append("REDDIT_CLIENT_ID and SECRET add reddit")
    return Check(
        name="optional creds",
        status=Status.WARN,
        detail=detail,
        remedy="; ".join(remedies),
    )


def _key_limits_check(
    settings: Settings,
    openrouter_stages: Sequence[str] | None = None,
) -> tuple[Check, KeyLimits | None]:
    """Ask the gateway what this key can afford, and never let the asking fail a run."""
    name = "key limits"
    if openrouter_stages is not None and not openrouter_stages:
        return (
            Check(
                name=name,
                status=Status.OK,
                detail="not applicable — all active stages use native OpenAI",
            ),
            None,
        )

    key = settings.openrouter_api_key
    if key is None:
        return (
            Check(
                name=name,
                status=Status.WARN,
                detail="not read — no API key",
                remedy="set OPENROUTER_API_KEY",
            ),
            None,
        )

    try:
        limits = _fetch_key_limits(key.get_secret_value())
    except (httpx.HTTPError, ValueError) as exc:
        return (
            Check(
                name=name,
                status=Status.WARN,
                detail=f"could not read key limits: {_short(str(exc), 48)}",
                remedy="check OPENROUTER_API_KEY, or retry",
            ),
            None,
        )

    parts: list[str] = []
    if limits.is_free_tier:
        parts.append("free tier")
    if limits.limit is not None:
        parts.append(f"limit ${limits.limit:,.2f}")
    if limits.usage is not None:
        parts.append(f"used ${limits.usage:,.2f}")
    if limits.remaining is not None:
        parts.append(f"${limits.remaining:,.4f} left")
    if limits.rate_limit:
        parts.append(limits.rate_limit)
    if limits.expires_at is not None:
        parts.append(f"expires {limits.expires_at:%Y-%m-%d %H:%M} UTC")

    detail = ", ".join(parts) or "read, but no limits reported"
    left = _expires_in_hours(limits.expires_at)

    if left is not None and left <= 0:
        return (
            Check(name=name, status=Status.FAIL, detail=detail, remedy="issue a new key"),
            limits,
        )
    if limits.remaining is not None and limits.remaining <= 0:
        return (
            Check(name=name, status=Status.FAIL, detail=detail, remedy="add credit before running"),
            limits,
        )
    if left is not None and left <= EXPIRY_WARNING_HOURS:
        return (
            Check(
                name=name,
                status=Status.WARN,
                detail=detail,
                remedy=f"expires in {left:.0f}h — issue a new key before a long run",
            ),
            limits,
        )
    return Check(name=name, status=Status.OK, detail=detail), limits


def _expires_in_hours(expires_at: datetime | None) -> float | None:
    """Hours of life left in the key, or None when it does not expire."""
    if expires_at is None:
        return None
    return (expires_at - utcnow()).total_seconds() / 3600


def _context_check(
    settings: Settings,
    route: ResolvedRoute | None,
    affordable: int | None,
    measured: Probe | None,
    note: str,
) -> Check:
    """The fully computable one: what `collect-evidence` would actually send.

    Three ceilings apply and the tightest is the one that bites, so the remedy has to
    name which. They fail differently: a local ceiling is a setting, a model's context
    is a fact, and an unaffordable prompt is a balance.
    """
    name = "context budget"
    if measured is None:
        return Check(name=name, status=Status.WARN, detail=note, remedy="re-run with --probe")

    if not measured.fetched:
        return Check(
            name=name,
            status=Status.WARN,
            detail=f"no candidates from {len(measured.ran)} searched source(s)",
            remedy="broaden the question, or configure another source",
        )

    context_length = route.context_length if route is not None else None
    ceilings = [(settings.max_input_tokens, LOCAL_LABEL)]
    if route is not None and context_length:
        ceilings.append((context_length, f"{route.model} context"))
    if affordable is not None:
        ceilings.append((affordable, CREDIT_LABEL))
    ceiling, source = min(ceilings)

    detail = (
        f"≈{measured.tokens:,} tokens from {measured.kept}/{measured.fetched} "
        f"candidates, ceiling {ceiling:,} ({source})"
    )
    remedy = _direction(source, measured.floor, ceiling)
    if measured.tokens > ceiling:
        return Check(name=name, status=Status.FAIL, detail=detail, remedy=remedy)
    if measured.tokens > int(ceiling * NEAR_LIMIT):
        return Check(name=name, status=Status.WARN, detail=detail, remedy=remedy)
    return Check(name=name, status=Status.OK, detail=detail)


def _direction(source: str, floor: int, ceiling: int) -> str:
    """Which way to move the binding ceiling. Getting this right is the whole point.

    MAX_INPUT_TOKENS is deliberately not offered as something to raise against a
    credit limit: `collect-evidence` fills COLLECT_CONTEXT_FRACTION of it, so raising
    it sends *more* and makes an unaffordable prompt larger. Raising it is the answer
    in exactly one case — when the floor, everything sent before the first candidate,
    is already over the ceiling. Trimming cannot reach that, so it is said separately.
    """
    if floor >= ceiling:
        if source == LOCAL_LABEL:
            return "raise MAX_INPUT_TOKENS — the prompt alone is over it"
        if source == CREDIT_LABEL:
            return "add credit — the prompt alone is unaffordable"
        return "route this stage to a model with a larger context"
    if source == CREDIT_LABEL:
        return "add credit, or lower MAX_INPUT_TOKENS — it sizes the candidate list"
    if source == LOCAL_LABEL:
        return "lower COLLECT_PREVIEW_CHARS, COLLECTOR_LIMIT or COLLECT_CONTEXT_FRACTION"
    return "lower COLLECT_PREVIEW_CHARS or COLLECTOR_LIMIT to fit the model"


def _output_check(
    settings: Settings,
    routes: dict[str, ResolvedRoute],
    catalogue: ModelCatalog | None,
    remaining: float | None,
) -> Check:
    """Whether each stage's `max_tokens` is one the key and the model will accept.

    The gateway reserves credit against `max_tokens` before the call, so an output cap
    the balance cannot cover is refused outright — no output is generated and no
    partial answer comes back.
    """
    name = "output budget"
    unaffordable: list[str] = []
    over_model: list[str] = []
    largest = 0
    largest_stage = ""

    for stage, route in routes.items():
        cap = route.max_output_tokens or settings.max_output_tokens
        if cap > largest:
            largest, largest_stage = cap, stage

        info = catalogue.get(route.model) if catalogue is not None else None
        if info is not None and info.max_completion_tokens and cap > info.max_completion_tokens:
            over_model.append(f"{stage} {cap:,}>{info.max_completion_tokens:,}")

        affordable = (
            _affordable(remaining, _price(info, "completion"))
            if _uses_openrouter(settings, route)
            else None
        )
        if affordable is not None and cap > affordable:
            unaffordable.append(f"{stage} {cap:,}>{affordable:,}")

    if unaffordable:
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"credit cannot cover: {_listed(unaffordable)}",
            remedy="add credit, or lower MAX_OUTPUT_TOKENS — it caps every stage",
        )
    if over_model:
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"over the model's own cap: {_listed(over_model)}",
            remedy="lower MAX_OUTPUT_TOKENS, or route the stage to a larger model",
        )
    if not largest_stage:
        return Check(name=name, status=Status.WARN, detail="no stage resolved to a model")
    return Check(
        name=name,
        status=Status.OK,
        detail=f"largest cap {largest:,} ({largest_stage})",
    )


def _routing_check(
    routes: dict[str, ResolvedRoute],
    unroutable: list[str],
    catalogue: ModelCatalog | None,
    *,
    openrouter_catalogue_applicable: bool = True,
) -> Check:
    """Every stage must reach a model, and it matters which one answers."""
    name = "routing"
    if unroutable:
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"{len(unroutable)} unroutable: {_listed(unroutable)}",
            remedy="check LLM_CAPABILITIES, LLM_STAGE_CAPABILITIES and LLM_FALLBACK_TIERS",
        )
    if not openrouter_catalogue_applicable:
        return Check(
            name=name,
            status=Status.OK,
            detail=(f"{len(routes)} stages use native OpenAI; OpenRouter catalogue not applicable"),
        )
    if catalogue is None:
        return Check(
            name=name,
            status=Status.WARN,
            detail=f"{len(routes)} stages resolve to pinned slugs",
            remedy="catalogue unreachable — models may be stale, and capabilities unknown",
        )

    loose = [stage for stage, route in routes.items() if not route.supports_structured_outputs]
    if loose:
        return Check(
            name=name,
            status=Status.WARN,
            detail=f"no structured outputs: {_listed(loose)}",
            remedy="replies fall back to plain JSON; re-point with LLM_CAPABILITIES",
        )
    return Check(
        name=name,
        status=Status.OK,
        detail=f"{len(routes)} stages resolve against {len(catalogue)} live models",
    )


def _collectors_check(settings: Settings, availability: dict[str, bool]) -> Check:
    """At least one source must be usable, or the first stage has nothing to judge."""
    name = "collectors"
    selected = tuple(settings.collectors) or available()

    unknown = [source for source in selected if source not in COLLECTORS]
    if unknown:
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"unknown in COLLECTORS: {_listed(unknown)}",
            remedy=f"registered: {', '.join(available())}",
        )

    usable = [source for source in selected if availability.get(source, False)]
    skipped = [source for source in selected if source not in usable]
    detail = f"available: {', '.join(usable) or 'none'}"
    if skipped:
        detail += f"; skipped: {', '.join(skipped)}"

    if not usable:
        return Check(
            name=name,
            status=Status.FAIL,
            detail=detail,
            remedy="clear COLLECTORS to use every available source",
        )

    # Set-but-inert configuration is worse than absent: it reads as done. These two
    # settings are never handed to the collectors the stage builds.
    stranded: list[str] = []
    if settings.rss_feeds and not availability.get("rss", False):
        stranded.append("RSS_FEEDS")
    if settings.corpus_paths and not availability.get("filesystem", False):
        stranded.append("CORPUS_PATHS")
    if stranded:
        return Check(
            name=name,
            status=Status.WARN,
            detail=detail,
            remedy=f"{', '.join(stranded)} set, but does not reach its collector",
        )
    return Check(name=name, status=Status.OK, detail=detail)


def _workspace_check() -> Check:
    """Artifacts are the run's whole output; nowhere to write them stops it dead."""
    name = "workspace"
    paths = get_workspace_paths()
    if not paths.root.is_dir():
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"no workspace at {paths.root}",
            remedy="run op init",
        )
    if not os.access(paths.root, os.W_OK):
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"not writable: {paths.root}",
            remedy="fix the permissions, or point WORKSPACE_DIR elsewhere",
        )

    registry = ArtifactRegistry(paths)
    missing = [kind.value for kind in ArtifactKind if not registry.directory_for(kind).is_dir()]
    if missing:
        return Check(
            name=name,
            status=Status.WARN,
            detail=f"{len(missing)} stage directories missing",
            remedy="run op init",
        )
    return Check(
        name=name,
        status=Status.OK,
        detail=f"writable, {len(ArtifactKind)} stage directories",
    )


# ---------------------------------------------------------------------- probing


def _run_probe(settings: Settings, run_id: str, availability: dict[str, bool]) -> Probe:
    """Measure the real `collect-evidence` context: real sources, the stage's own logic.

    Every step here is the stage's, deliberately. A doctor that fetched differently, or
    trimmed differently, would report a number the run does not honour — which is the
    failure this command exists to end.
    """
    question = _question_for(run_id)
    query = str(getattr(question, "text", DEFAULT_QUERY))
    config = config_from_settings()

    fetched: list[SourceItem] = []
    ran: list[str] = []
    for source in tuple(settings.collectors) or available():
        if source not in COLLECTORS or not availability.get(source, False):
            continue
        try:
            found = COLLECTORS[source](config).search(query, limit=settings.collector_limit)
        except CollectorError:
            continue
        fetched.extend(found)
        ran.append(source)

    # `_deduplicated` and `_previews` are the stage's own; calling them rather than
    # copying them is what stops the estimate and the behaviour drifting apart.
    unique = CollectEvidenceSkill._deduplicated(fetched)
    previews = CollectEvidenceSkill()._previews(unique)

    asked = question.model_dump(mode="json")
    prompt = load_prompt(CollectEvidenceSkill.prompt_name)
    sent = CollectEvidenceInput(
        question=asked,
        candidates=[item.model_dump(mode="json") for item in previews],
    )
    # The same prompt with nothing found: what the stage costs before any candidate,
    # and therefore the floor no amount of trimming can go under.
    empty = CollectEvidenceInput(question=asked, candidates=[])

    return Probe(
        ran=tuple(ran),
        fetched=len(unique),
        kept=len(previews),
        tokens=estimate_tokens(prompt.render(sent.model_dump(mode="json"))),
        floor=estimate_tokens(prompt.render(empty.model_dump(mode="json"))),
    )


def _question_for(run_id: str) -> Question:
    """The run's own Question when it has one.

    The estimate is only as representative as the query behind it, and a real question
    fetches what a real run would. The stand-in is for a workspace not yet seeded.
    """
    try:
        found = ArtifactRegistry().find_by_type(ArtifactKind.QUESTION, run_id=run_id, limit=1)
    except ArtifactError:
        found = []
    seeded = found[0] if found else None
    if isinstance(seeded, Question):
        return seeded
    return Question(id=Question.make_id(), run_id=run_id, text=DEFAULT_QUERY)


def _availability(settings: Settings) -> dict[str, bool]:
    """Whether each collector could run, built exactly as `collect-evidence` builds it.

    Given a richer config than the stage uses, a source would report available here and
    then be skipped by the run — a doctor that lies in the reassuring direction.
    """
    config = config_from_settings()
    result: dict[str, bool] = {}
    for source, factory in sorted(COLLECTORS.items()):
        try:
            result[source] = factory(config).available()
        except CollectorError:
            result[source] = False
    return result


def _resolve_routes(
    catalogue: ModelCatalog | None = None,
) -> tuple[dict[str, ResolvedRoute], list[str]]:
    """Where each stage would go, and the stages that cannot answer that."""
    try:
        router = ModelRouter.from_settings(catalogue)
    except (ValueError, ConfigurationError) as exc:
        return {}, [_short(str(exc))]

    routes: dict[str, ResolvedRoute] = {}
    failures: list[str] = []
    for stage in STAGE_ORDER:
        try:
            routes[stage] = router.resolve(stage)
        except (ValueError, ConfigurationError) as exc:
            failures.append(f"{stage}: {_short(str(exc), 40)}")
    return routes, failures


# ------------------------------------------------------------------- key limits


def _fetch_key_limits(key: str) -> KeyLimits:
    """Read the key's own limits from the gateway.

    The shape read here was observed on a live 200 — `data` wrapping `limit`,
    `usage`, `limit_remaining`, `is_free_tier`, `expires_at` and a `rate_limit` the
    payload itself marks deprecated. It is still parsed as though none of that were
    guaranteed: this is one undocumented endpoint away from changing, and a doctor
    that raised on an unfamiliar payload would fail exactly the runs it exists to
    protect. Every field is optional, and any failure is a warning, never an error.
    """
    response = httpx.get(
        KEY_URL,
        headers={"Authorization": f"Bearer {key}"},
        timeout=KEY_TIMEOUT,
    )
    response.raise_for_status()
    return _parse_key_limits(response.json())


def _parse_key_limits(payload: Any) -> KeyLimits:
    """Pull whatever of the documented fields are present, tolerating any that are not."""
    body = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(body, dict):
        body = payload if isinstance(payload, dict) else {}

    limit = _as_float(body.get("limit"))
    usage = _as_float(body.get("usage"))
    remaining = _as_float(body.get("limit_remaining"))
    if remaining is None and limit is not None and usage is not None:
        remaining = limit - usage

    free = body.get("is_free_tier")
    return KeyLimits(
        limit=limit,
        usage=usage,
        remaining=remaining,
        is_free_tier=free if isinstance(free, bool) else None,
        rate_limit=_rate_limit(body.get("rate_limit")),
        expires_at=_as_datetime(body.get("expires_at")),
    )


def _rate_limit(raw: Any) -> str | None:
    """Render a rate limit as `requests/interval`, when there is one to render.

    A live key reports `-1` requests here alongside a note calling the field
    deprecated, so a non-positive count is reported as no limit rather than shown.
    """
    if not isinstance(raw, dict):
        return None
    requests = _as_float(raw.get("requests"))
    interval = raw.get("interval")
    if requests is None or requests <= 0 or not isinstance(interval, str):
        return None
    return f"{requests:.0f}/{interval}"


def _as_datetime(raw: Any) -> datetime | None:
    """An expiry timestamp, tolerating the trailing `Z` the gateway sends."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _as_float(raw: Any) -> float | None:
    """A number, whether the payload spelled it as one or as a decimal string."""
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _price(info: ModelInfo | None, side: str) -> float | None:
    """USD per token for one side of a call, when the catalogue states a plain number."""
    if info is None:
        return None
    # Pricing values are usually decimal strings, but some vendors nest a list of
    # tiered overrides under a key, so anything unparseable means "unknown" — and
    # unknown must mean "do not judge", never a price of zero or one.
    price = _as_float(info.pricing.get(side))
    return price if price is not None and price > 0 else None


def _affordable(remaining: float | None, price: float | None) -> int | None:
    """Tokens the balance still covers at this price.

    Optimistic: the gateway reserves credit for the reply as well, so the ceiling a
    call really meets is lower than this. An exhausted balance returns None because
    the key-limits check already reports that, and reporting it twice buries it.
    """
    if remaining is None or price is None or remaining <= 0:
        return None
    return int(remaining / price)


# --------------------------------------------------------------------- printing


def _table(checks: Sequence[Check]) -> Table:
    """One row per check, ordered by what kills a run first."""
    table = Table(header_style="stage")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold", max_width=52)
    table.add_column("remedy", style="muted", overflow="fold", max_width=42)

    for check in checks:
        style = STATUS_STYLE[check.status]
        table.add_row(
            check.name,
            f"[{style}]{check.status.value}[/{style}]",
            check.detail,
            check.remedy,
        )
    return table


def _summary(checks: Sequence[Check]) -> str:
    counts = Counter(check.status for check in checks)
    return "  ".join(
        f"[{STATUS_STYLE[status]}]{counts[status]} {status.value}[/{STATUS_STYLE[status]}]"
        for status in Status
        if counts[status]
    )


def _worst(checks: Sequence[Check]) -> Status:
    """The overall verdict: a warning never outranks an ok into failure."""
    if any(check.status is Status.FAIL for check in checks):
        return Status.FAIL
    if any(check.status is Status.WARN for check in checks):
        return Status.WARN
    return Status.OK


def _listed(items: Sequence[str], keep: int = 2) -> str:
    """The first few offenders. A detail column is a pointer, not a report."""
    shown = ", ".join(items[:keep])
    return shown if len(items) <= keep else f"{shown}, +{len(items) - keep}"


def _short(text: str, limit: int = 60) -> str:
    """One clipped line: an exception's prose must not break the table open."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[:limit]}…"


__all__ = ["doctor"]
