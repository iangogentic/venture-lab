"""`op benchmark` — re-ask a fixed question, and see what a change did to the answer.

After editing a prompt or moving a stage to another model, the output is different.
Nothing about it being different says it is better. These four commands exist to make
that difference legible:

* `op benchmark list`    — the fixed questions available, and what each expects.
* `op benchmark score`   — read one finished run: volumes, grounding, theme coverage.
* `op benchmark compare` — put two runs side by side and report what survived.
* `op benchmark run`     — seed a benchmark's question and drive the whole pipeline.

Nothing here prints an overall pass or fail, and that is deliberate. A green tick would
turn a benchmark into a gate, and a gate would be optimised against: the run that
declines to size a market, or searches for counter-evidence and honestly finds none,
would start looking like a failing run. Every number below is a reading to go and look
at. The judgement stays with the reader.

`--json` on any subcommand emits exactly one JSON document on stdout; everything else —
which benchmark was inferred, what a run is about to cost — goes to stderr, so the
output can always be piped into `jq`.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Any, Final

import typer
from rich.table import Table

from app.artifacts import ArtifactKind, ArtifactRegistry, Question
from app.benchmarks import (
    SCORED_KINDS,
    Benchmark,
    Expectation,
    RunComparison,
    RunScore,
    StageScore,
    available,
    benchmarks_root,
    compare_runs,
    load_all,
    load_benchmark,
    normalise_text,
    score_run,
    similarity,
)
from app.cli.render import as_payload, outcome_table, print_json
from app.config import Settings, get_settings
from app.llm.provider import Provider
from app.llm.routing import ModelRouter, ResolvedRoute
from app.pipeline import STAGE_ORDER, PipelineEngine
from app.storage.schema import create_all
from app.utils.console import console, err_console
from app.utils.errors import OpportunityEngineError
from app.utils.paths import get_workspace_paths
from app.utils.time import utcnow

app = typer.Typer(
    name="benchmark",
    help="Re-ask a fixed question and measure what changed — not whether it passed.",
    no_args_is_help=True,
)

JsonOption = Annotated[bool, typer.Option("--json", help="Emit JSON and nothing else.")]

_STAGE_LABELS: Final[Mapping[ArtifactKind, str]] = MappingProxyType(
    dict(zip(SCORED_KINDS, STAGE_ORDER, strict=False))
)
"""Which stage produced each scored kind, for a table a reader can follow.

Zipped without `strict`: the two tuples are declared in the same order in two modules,
and if they ever drift the right outcome is a column that falls back to the kind's own
name, not a CLI that refuses to start.
"""

_INFERENCE_THRESHOLD: Final[float] = 0.6
"""How alike a run's question and a benchmark's must be to assume they are the same ask.

Only reached when the two are not identical, which they are whenever `op benchmark run`
seeded the run. Set high because guessing wrong scores a run against someone else's
expectations, and reporting no benchmark at all is the more honest failure.
"""


# ---------------------------------------------------------------------- list


@app.command("list")
def list_available(as_json: JsonOption = False) -> None:
    """Show the benchmarks on disk: the question each asks, and what it expects."""
    try:
        benchmarks = load_all()
    except OpportunityEngineError as exc:
        err_console.print(f"[danger]{exc}[/danger]")
        raise typer.Exit(code=1) from exc

    if as_json:
        print_json(
            {
                "count": len(benchmarks),
                "benchmarks": [benchmark.model_dump(mode="json") for benchmark in benchmarks],
            }
        )
        return

    if not benchmarks:
        console.print(f"[muted]no benchmarks under[/muted] {benchmarks_root()}")
        return

    console.print(_catalogue_table(benchmarks))
    err_console.print(
        "[muted]score a run against one with[/muted] "
        "op benchmark score --run <id> --benchmark <name>"
    )


# --------------------------------------------------------------------- score


@app.command("score")
def score(
    run_id: Annotated[str, typer.Option("--run", "-r", help="Run identifier to score.")],
    benchmark_name: Annotated[
        str | None,
        typer.Option("--benchmark", "-b", help="Expectations to use. Inferred when omitted."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Read one finished run: what each stage produced, and whether it rests on anything.

    Without `--benchmark` the run's own Question is matched against the benchmarks on
    disk. When nothing matches, the counts and the grounding are still reported — only
    the expected bands and the themes go unasserted.
    """
    registry = ArtifactRegistry()
    _require_run(registry, run_id)

    benchmark = _load(benchmark_name) if benchmark_name is not None else _infer(registry, run_id)
    if benchmark_name is None:
        if benchmark is None:
            err_console.print(
                "[muted]no benchmark matches this run's question — counting only, "
                "nothing is asserted[/muted]"
            )
        else:
            err_console.print(f"[muted]inferred benchmark[/muted] [stage]{benchmark.name}[/stage]")

    scored = score_run(registry, run_id, benchmark)

    if as_json:
        print_json(_score_payload(scored))
        return

    _print_score(scored)


# ------------------------------------------------------------------- compare


@app.command("compare")
def compare(
    left: Annotated[str, typer.Option("--left", help="The run to compare from.")],
    right: Annotated[str, typer.Option("--right", help="The run to compare against.")],
    as_json: JsonOption = False,
) -> None:
    """Put two runs side by side: how much of the first survived into the second.

    The question a prompt change actually raises. Overlap near 1.0 says the change moved
    the wording; overlap near 0.0 says it moved the findings. Neither is good or bad on
    its own — but a verdict that flipped is always worth reading.
    """
    registry = ArtifactRegistry()
    _require_run(registry, left)
    _require_run(registry, right)

    comparison = compare_runs(registry, left, right)

    if as_json:
        payload: dict[str, Any] = comparison.model_dump(mode="json")
        payload["matched_verdicts"] = comparison.matched_verdicts
        print_json(payload)
        return

    console.print(_stability_table(comparison))
    _print_movement(comparison)


# ----------------------------------------------------------------------- run


@app.command("run")
def run_benchmark(
    name: Annotated[str, typer.Argument(help="Benchmark to run, by directory name.")],
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Run identifier. Defaults to the name plus a timestamp."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Seed the question and print the plan. Calls no model."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Do not ask for confirmation before spending."),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Seed a benchmark's question, run the whole pipeline against it, and score it.

    This one spends money: every one of the nine stages calls a model, so it needs an
    API key and a willingness to pay for the run. `--dry-run` seeds the question and
    prints the plan without calling anything, which is enough to check that a benchmark
    is wired up before committing to it.

    The default run id carries a timestamp, so two invocations produce two runs that
    `op benchmark compare` can be pointed at rather than one run that resumes itself.
    """
    benchmark = _load(name)
    resolved = run_id if run_id is not None else f"{name}-{utcnow():%Y%m%dT%H%M%SZ}"

    paths = get_workspace_paths()
    registry = ArtifactRegistry(paths)
    existing = _existing_question(registry, benchmark, resolved)

    # Everything expensive is gated here, before a single artifact is written.
    if not dry_run:
        _announce_cost(resolved)
        _require_api_key()
        if not yes:
            _confirm()

    paths.ensure()
    create_all()

    seeded = existing if existing is not None else _seed(registry, benchmark, resolved)

    status = err_console if as_json else console
    status.print(
        f"[success]{'seeded' if existing is None else 'reusing'}[/success] "
        f"[stage]{resolved}[/stage] [muted]{benchmark.name} · {seeded.id}[/muted]"
    )
    status.print(f"  {seeded.text}")

    if dry_run:
        if as_json:
            print_json(
                {
                    "benchmark": benchmark.model_dump(mode="json"),
                    "run_id": resolved,
                    "dry_run": True,
                    "question": as_payload(seeded),
                    "plan": list(STAGE_ORDER),
                }
            )
            return
        console.print(_plan_table(benchmark))
        console.print("[muted]dry run — no model was called and nothing was spent[/muted]")
        console.print(
            "[muted]run it for real with[/muted] "
            f"op benchmark run {benchmark.name} --run-id {resolved}"
        )
        return

    result = PipelineEngine(registry).run(resolved)
    scored = score_run(registry, resolved, benchmark)

    if as_json:
        print_json(
            {
                "benchmark": benchmark.name,
                "run_id": resolved,
                "dry_run": False,
                "pipeline": result.model_dump(mode="json"),
                "score": _score_payload(scored),
            }
        )
    else:
        console.print(outcome_table(result))
        _print_score(scored)

    if not result.ok:
        raise typer.Exit(code=1)


# ------------------------------------------------------------------ plumbing


def _load(name: str) -> Benchmark:
    """Load a benchmark, or exit 1 naming the ones that do exist."""
    try:
        return load_benchmark(name)
    except OpportunityEngineError as exc:
        err_console.print(f"[danger]{exc}[/danger]")
        known = ", ".join(available())
        err_console.print(f"[muted]available:[/muted] {known or 'none'}")
        raise typer.Exit(code=1) from exc


def _require_run(registry: ArtifactRegistry, run_id: str) -> None:
    """Exit 1 when a run id names nothing at all in the workspace.

    Scoring an id that was never run would otherwise report a flawless-looking wall of
    zeroes, which reads as "the pipeline produced nothing" rather than "you typoed it".
    """
    if any(registry.find_by_type(kind, run_id=run_id, limit=1) for kind in ArtifactKind):
        return
    err_console.print(f"[danger]no artifacts for run[/danger] {run_id}")
    raise typer.Exit(code=1)


def _infer(registry: ArtifactRegistry, run_id: str) -> Benchmark | None:
    """Work out which benchmark a run was seeded from, by comparing its Question.

    An exact match on the normalised text settles it, which is the case whenever
    `op benchmark run` seeded the run. Otherwise the closest benchmark is taken, and
    only if it is close enough to be the same ask rather than a neighbouring one.
    """
    questions = registry.find_by_type(ArtifactKind.QUESTION, run_id=run_id)
    asked = next((q.text for q in questions if isinstance(q, Question) and q.text), None)
    if asked is None:
        return None

    try:
        candidates = load_all()
    except OpportunityEngineError as exc:
        err_console.print(f"[warning]benchmarks could not be read[/warning] {exc}")
        return None

    normalised = normalise_text(asked)
    for benchmark in candidates:
        if normalise_text(benchmark.question) == normalised:
            return benchmark

    best: Benchmark | None = None
    best_score = 0.0
    for benchmark in candidates:
        value = similarity(asked, benchmark.question)
        if value > best_score:
            best, best_score = benchmark, value
    return best if best_score >= _INFERENCE_THRESHOLD else None


def _existing_question(
    registry: ArtifactRegistry,
    benchmark: Benchmark,
    run_id: str,
) -> Question | None:
    """The Question this run already carries, when it is this benchmark's.

    Pointing `--run-id` at an existing run is how a run that failed halfway gets
    resumed, so a second identical Question must not be piled on top of the first. A run
    already asking something *else* is refused: two questions under one run id would
    leave every artifact beneath them ambiguous about what it was answering.
    """
    questions = [
        artifact
        for artifact in registry.find_by_type(ArtifactKind.QUESTION, run_id=run_id)
        if isinstance(artifact, Question)
    ]
    asked = normalise_text(benchmark.question)
    for question in questions:
        if normalise_text(question.text) == asked:
            return question

    if questions:
        err_console.print(f"[danger]run {run_id} already asks[/danger] {questions[0].text}")
        err_console.print(f"[muted]not {benchmark.name}'s question — pick another[/muted] --run-id")
        raise typer.Exit(code=1)
    return None


def _seed(registry: ArtifactRegistry, benchmark: Benchmark, run_id: str) -> Question:
    """Write the benchmark's question into the workspace as the run's root artifact."""
    seeded = Question(
        id=Question.make_id(),
        run_id=run_id,
        text=benchmark.question,
        scope=benchmark.scope,
        tags=list(benchmark.tags),
    )
    registry.save(seeded)
    return seeded


def _announce_cost(run_id: str) -> None:
    """Say what this costs before it costs anything. On stderr, so `--json` stays clean."""
    native, gateway = _transport_stages()
    if native and gateway:
        billing = (
            "it spends real money through OpenRouter and directly with OpenAI; "
            "needs OPENROUTER_API_KEY and OPENAI_API_KEY"
        )
    elif native:
        billing = "it is billed directly by OpenAI to your own API key; needs OPENAI_API_KEY"
    else:
        # Preserve the existing wording for Claude, Gemini, and gateway-routed GPT.
        billing = "it spends real money and needs OPENROUTER_API_KEY"

    err_console.print(
        f"[warning]this calls a model at every one of the {len(STAGE_ORDER)} stages[/warning] "
        f"[muted]— {billing}[/muted]"
    )
    err_console.print(
        f"[muted]run[/muted] [stage]{run_id}[/stage] "
        "[muted]· --dry-run seeds the question and stops[/muted]"
    )


def _require_api_key() -> None:
    """Fail before the first stage rather than partway through it."""
    settings = get_settings()
    native, gateway = _transport_stages(settings)
    missing: list[str] = []
    if native and settings.openai_api_key is None:
        missing.append(f"OPENAI_API_KEY for {_listed_stages(native)}")
    if gateway and settings.openrouter_api_key is None:
        missing.append(f"OPENROUTER_API_KEY for {_listed_stages(gateway)}")
    if not missing:
        return

    if not native and gateway and settings.openrouter_api_key is None:
        # Keep the long-standing error concise when every route is on the gateway.
        err_console.print(
            "[danger]no OPENROUTER_API_KEY configured[/danger] — set one, or use --dry-run"
        )
    else:
        err_console.print(
            f"[danger]missing {'; '.join(missing)}[/danger] — set the required "
            "key(s), or use --dry-run"
        )
    raise typer.Exit(code=1)


def _transport_stages(
    settings: Settings | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition active stages by billing transport without contacting a catalogue."""
    configured = settings or get_settings()
    router = ModelRouter.from_settings(catalogue=None)
    native: list[str] = []
    gateway: list[str] = []
    for stage in STAGE_ORDER:
        route = router.resolve(stage)
        target = native if _uses_native_openai(configured, route) else gateway
        target.append(stage)
    return tuple(native), tuple(gateway)


def _uses_native_openai(settings: Settings, route: ResolvedRoute) -> bool:
    """Match GPT adapter transport selection without constructing a network client."""
    if route.provider is not Provider.GPT:
        return False
    if settings.llm_transport == "openai":
        return True
    if settings.llm_transport == "openrouter":
        return False
    return settings.openai_api_key is not None


def _listed_stages(stages: Sequence[str], keep: int = 2) -> str:
    """Enough route names to explain why a credential is required."""
    shown = ", ".join(stages[:keep])
    return shown if len(stages) <= keep else f"{shown}, +{len(stages) - keep}"


def _confirm() -> None:
    """Ask, but only where somebody is there to answer.

    A piped or scripted invocation proceeds: hanging a CI job on an unanswerable prompt
    would be a worse failure than the spend the prompt is guarding.
    """
    if not err_console.is_terminal:
        return
    typer.confirm("spend a full pipeline run on this?", default=False, abort=True)


# ------------------------------------------------------------------ rendering


def _print_score(scored: RunScore) -> None:
    """The whole reading of one run: volumes, then grounding, then themes."""
    console.print(_stage_table(scored))
    console.print(_grounding_table(scored))
    _print_themes(scored)
    _print_verdicts(scored)
    console.print(
        "[muted]readings, not a grade — run it twice and `op benchmark compare` "
        "to tell better from merely different[/muted]"
    )


def _score_payload(scored: RunScore) -> dict[str, Any]:
    """The score as one JSON document, with the derived ratios written out.

    `model_dump` omits properties, and the ratios are exactly what a caller diffs
    between two runs — leaving them out would push that arithmetic into every consumer.
    """
    payload: dict[str, Any] = scored.model_dump(mode="json")
    payload["artifact_total"] = scored.artifact_total
    payload["within_expectation"] = scored.within_expectation
    payload["empty_stages"] = [kind.value for kind in scored.empty_stages]

    grounding: dict[str, Any] = payload["grounding"]
    grounding["traceable_fraction"] = scored.grounding.traceable_fraction
    grounding["decision_traceable_fraction"] = scored.grounding.decision_traceable_fraction
    grounding["citation_fraction"] = scored.grounding.citation_fraction
    grounding["contradiction_searches_unrecorded"] = (
        scored.grounding.contradiction_searches_unrecorded
    )
    return payload


def _catalogue_table(benchmarks: Sequence[Benchmark]) -> Table:
    table = Table(header_style="stage")
    table.add_column("benchmark")
    table.add_column("question", overflow="fold", max_width=44)
    table.add_column("themes", overflow="fold", max_width=28)
    table.add_column("expects", overflow="fold", max_width=30)

    for benchmark in benchmarks:
        table.add_row(
            benchmark.name,
            benchmark.question,
            ", ".join(benchmark.themes) or "[muted]—[/muted]",
            _expectations_text(benchmark),
        )
    return table


def _expectations_text(benchmark: Benchmark) -> str:
    """The declared bands, named after the workspace directories.

    Run together and wrapped rather than one per line: nine kinds down the page would
    make every row eleven lines tall, and the column is a summary — the per-stage view
    is what `op benchmark score` is for.
    """
    bands = [
        f"{kind.directory} {_band(expectation.min, expectation.max)}"
        for kind in SCORED_KINDS
        if (expectation := benchmark.expectation_for(kind)) is not None and not expectation.is_empty
    ]
    return ", ".join(bands) or "[muted]—[/muted]"


def _stage_table(scored: RunScore) -> Table:
    """Per-stage volumes against the benchmark's bands. No overall verdict row."""
    title = f"run {scored.run_id}"
    if scored.benchmark is not None:
        title = f"{title}  vs  {scored.benchmark}"

    table = Table(title=title, title_style="muted", header_style="stage")
    table.add_column("#", justify="right", style="muted")
    table.add_column("stage")
    table.add_column("artifacts", justify="right")
    table.add_column("expected", justify="right")
    table.add_column("reading")

    for index, stage in enumerate(scored.stages, start=1):
        table.add_row(
            str(index),
            _STAGE_LABELS.get(stage.kind, stage.kind.value),
            str(stage.count),
            _band(stage.expected_min, stage.expected_max),
            _reading(stage),
        )
    return table


def _band(low: int | None, high: int | None) -> str:
    """An `Expectation` as a column: `3-10`, `≥1`, `≤10`, or nothing asserted."""
    if low is None and high is None:
        return "[muted]—[/muted]"
    if low is not None and high is not None:
        return f"{low}-{high}"
    return f"≥{low}" if low is not None else f"≤{high}"


def _reading(stage: StageScore) -> str:
    """How to read one stage's count — inside the band, outside it, or unasserted.

    Never a tick and never a cross. Outside a band is a prompt to go and look at the
    stage, not a failed test: the bands describe a healthy run, and a healthy run is a
    range rather than a number.
    """
    if stage.expected_min is None and stage.expected_max is None:
        if stage.produced:
            return "[muted]not asserted[/muted]"
        return "[warning]produced nothing[/warning]"
    if stage.within_expectation:
        return "[success]within[/success]"
    if stage.expected_min is not None and stage.count < stage.expected_min:
        return f"[warning]below {stage.expected_min}[/warning]"
    return f"[warning]above {stage.expected_max}[/warning]"


def _grounding_table(scored: RunScore) -> Table:
    """Whether the run's conclusions rest on anything, and how honestly it refused.

    Traceability first, because an opportunity with no evidence beneath it is this
    pipeline's central failure mode. The refusals below it — an unsized market, a
    contradiction search that came back empty — are correct outputs and are labelled as
    such, so nobody reads a low number there as something to fix.
    """
    grounding = scored.grounding
    table = Table(title="grounding", title_style="muted", header_style="stage")
    table.add_column("measure")
    table.add_column("reading", justify="right")
    table.add_column("note", style="muted", overflow="fold", max_width=44)

    table.add_row(
        "opportunities reaching evidence",
        _ratio(
            grounding.opportunities_traceable_to_evidence,
            grounding.opportunities_total,
            stark=True,
        ),
        "one with none was assembled out of nothing",
    )
    table.add_row(
        "decisions reaching evidence",
        _ratio(grounding.decisions_traceable_to_evidence, grounding.decisions_total, stark=True),
        "a verdict with none was reached about nothing",
    )
    table.add_row(
        "briefs citing evidence",
        _ratio(grounding.briefs_citing_evidence, grounding.briefs_total, stark=True),
        "cited in the brief's own signals and quotes, not in its parents",
    )
    table.add_row(
        "evidence a reader can check",
        _ratio(grounding.evidence_with_source, grounding.evidence_total, stark=True),
        "carries a source url or a source id",
    )
    table.add_row(
        "markets sized",
        _ratio(grounding.markets_sized, grounding.markets_total),
        "sam or som carries a number",
    )
    table.add_row(
        "markets unsized, with a reason",
        str(grounding.markets_unsized_with_basis),
        "a refusal to invent a figure — correct output, never a miss",
    )
    table.add_row(
        "markets silent on size",
        str(grounding.markets_without_estimate),
        "no estimate either way, so nothing was refused",
    )
    table.add_row(
        "contradiction searches recorded",
        _ratio(
            grounding.contradiction_searches_recorded,
            grounding.contradiction_analyses_total,
        ),
        "without it, finding nothing cannot be told from never looking",
    )
    table.add_row(
        "searches that found nothing",
        str(grounding.contradiction_searches_empty_result),
        "the adversarial pass ran and came back empty — a real result",
    )
    table.add_row(
        "confidence unassessed",
        str(grounding.unassessed_confidence),
        "none means not yet scored, not zero",
    )
    return table


def _ratio(part: int, whole: int, *, stark: bool = False) -> str:
    """`3/4 75%`, coloured only where zero genuinely means something went wrong.

    `stark` is reserved for traceability. Colouring a market-sizing ratio the same way
    would mark an honest "we cannot size this" in red, which is the reading this whole
    package exists to avoid.
    """
    if whole == 0:
        return "[muted]—[/muted]"
    text = f"{part}/{whole} {part / whole:.0%}"
    if not stark:
        return text
    if part == 0:
        return f"[danger]{text}[/danger]"
    if part == whole:
        return f"[success]{text}[/success]"
    return f"[warning]{text}[/warning]"


def _print_themes(scored: RunScore) -> None:
    themes = scored.themes
    if themes is None:
        console.print("\n[muted]no themes declared — coverage not measured[/muted]")
        return

    console.print(
        f"\nthemes [info]{len(themes.matched)}/{len(themes.expected)}[/info] "
        f"[muted]{themes.coverage:.0%} coverage[/muted]"
    )
    for theme in themes.matched:
        console.print(f"  [success]seen[/success] {theme}")
    for theme in themes.missed:
        console.print(f"  [warning]unseen[/warning] {theme}")
    if themes.missed:
        console.print(
            "[muted]matched on wording, not on meaning — unseen means go and look, "
            "not wrong[/muted]"
        )


def _print_verdicts(scored: RunScore) -> None:
    if not scored.verdicts:
        return
    tally = "  ".join(f"[muted]{name}[/muted] {count}" for name, count in scored.verdicts.items())
    console.print(f"\nverdicts  {tally}")


def _plan_table(benchmark: Benchmark) -> Table:
    """What a real run would do, stage by stage, and what it would be read against."""
    table = Table(title=f"plan — {benchmark.name}", title_style="muted", header_style="stage")
    table.add_column("#", justify="right", style="muted")
    table.add_column("stage")
    table.add_column("produces", style="muted")
    table.add_column("expected", justify="right")

    for index, kind in enumerate(SCORED_KINDS, start=1):
        table.add_row(
            str(index),
            _STAGE_LABELS.get(kind, kind.value),
            kind.value,
            _expected_band(benchmark.expectation_for(kind)),
        )
    return table


def _expected_band(expectation: Expectation | None) -> str:
    if expectation is None:
        return "[muted]—[/muted]"
    return _band(expectation.min, expectation.max)


def _stability_table(comparison: RunComparison) -> Table:
    table = Table(
        title=f"{comparison.left}  →  {comparison.right}",
        title_style="muted",
        header_style="stage",
    )
    table.add_column("measure")
    table.add_column("reading", justify="right")
    table.add_column("note", style="muted", overflow="fold", max_width=44)

    table.add_row(
        "pain clusters kept",
        f"{comparison.cluster_overlap:.0%}",
        "matched labels over the clusters of both runs",
    )
    table.add_row(
        "opportunities kept",
        f"{comparison.opportunity_overlap:.0%}",
        "matched titles over the opportunities of both runs",
    )
    table.add_row(
        "verdicts agreeing",
        "[muted]—[/muted]"
        if comparison.verdict_agreement is None
        else f"{comparison.verdict_agreement:.0%}",
        f"over {_opportunities(comparison.matched_verdicts)} both runs decided"
        if comparison.matched_verdicts
        else "nothing was matched and decided in both runs",
    )
    return table


def _opportunities(count: int) -> str:
    """`1 opportunity` / `3 opportunities`."""
    return f"{count} opportunity" if count == 1 else f"{count} opportunities"


def _print_movement(comparison: RunComparison) -> None:
    """List what actually moved. A changed verdict is the most consequential of them."""
    moved = bool(
        comparison.changed_verdicts
        or comparison.dropped_opportunities
        or comparison.new_opportunities
    )
    if not moved:
        console.print(
            "[muted]nothing moved: same clusters, same opportunities, same verdicts[/muted]"
        )
        return

    console.print("\nwhat moved")
    for pair in comparison.changed_verdicts:
        # The two titles are matched, not identical — showing both is what makes it
        # checkable that the match was a fair one.
        titles = pair.left_title
        if pair.right_title != pair.left_title:
            titles = f"{titles} / {pair.right_title}"
        console.print(
            f"  [warning]verdict[/warning] {titles} "
            f"[muted]{pair.left_verdict.value} → {pair.right_verdict.value}[/muted]"
        )
    for title in comparison.dropped_opportunities:
        console.print(f"  [muted]dropped[/muted] {title}")
    for title in comparison.new_opportunities:
        console.print(f"  [info]new[/info] {title}")

    console.print(
        "[muted]stability is not the goal — a prompt change is meant to change "
        "something. This says what.[/muted]"
    )


__all__ = ["app", "compare", "list_available", "run_benchmark", "score"]
