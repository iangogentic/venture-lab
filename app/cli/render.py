"""Rich rendering shared by the commands.

Every command prints through here so tables look the same everywhere, and so the
`--json` path is identical no matter which command produced the artifacts.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from rich.markup import escape
from rich.table import Table

from app.artifacts import Artifact, ArtifactKind, Decision
from app.pipeline import AutoResult, PipelineRun, StageAttempt, StageOutcome, StageStatus
from app.utils.console import console

STATUS_STYLE: dict[StageStatus, str] = {
    StageStatus.COMPLETED: "success",
    StageStatus.SKIPPED: "muted",
    StageStatus.EMPTY: "warning",
    StageStatus.BLOCKED: "warning",
    StageStatus.FAILED: "danger",
}

STATUS_MARK: dict[StageStatus, str] = {
    StageStatus.COMPLETED: "✓",
    StageStatus.SKIPPED: "·",
    StageStatus.EMPTY: "∅",
    StageStatus.BLOCKED: "!",
    StageStatus.FAILED: "✗",
}
"""A glyph per outcome, so a long run scans vertically without reading a word."""

VERDICT_STYLE: dict[str, str] = {"build": "success", "wait": "warning", "reject": "muted"}


def as_payload(artifact: Artifact) -> dict[str, Any]:
    """One artifact as the registry stores it: its fields, tagged with its kind."""
    return {"kind": type(artifact).kind.value, **artifact.model_dump(mode="json")}


def artifact_json(artifacts: Sequence[Artifact]) -> str:
    """Serialise a list of artifacts as a JSON array."""
    return json.dumps([as_payload(a) for a in artifacts], indent=2, ensure_ascii=False)


def single_artifact_json(artifact: Artifact) -> str:
    """Serialise one artifact as a JSON object, not a one-element array."""
    return json.dumps(as_payload(artifact), indent=2, ensure_ascii=False)


def print_json(data: Any) -> None:
    """Emit raw JSON on stdout, unstyled, so it can be piped into `jq`."""
    text = data if isinstance(data, str) else json.dumps(data, indent=2, ensure_ascii=False)
    console.print_json(text)


def artifact_table(artifacts: Sequence[Artifact], *, title: str | None = None) -> Table:
    """One row per artifact, with the envelope fields that matter when scanning."""
    table = Table(title=title, title_style="muted", header_style="stage")
    table.add_column("id", overflow="fold")
    table.add_column("kind")
    table.add_column("v", justify="right")
    table.add_column("status")
    table.add_column("conf", justify="right")
    table.add_column("evidence")
    table.add_column("summary", overflow="ellipsis", max_width=48)

    for artifact in artifacts:
        confidence = "—" if artifact.confidence is None else f"{artifact.confidence:.2f}"
        table.add_row(
            artifact.id,
            type(artifact).kind.value,
            str(artifact.version),
            artifact.status.value,
            confidence,
            artifact.evidence_level.value,
            escape(_headline(artifact)),
        )
    return table


def counts_table(counts: dict[ArtifactKind, int]) -> Table:
    """Artifact totals per kind, in pipeline order."""
    table = Table(header_style="stage")
    table.add_column("kind")
    table.add_column("directory", style="muted")
    table.add_column("artifacts", justify="right")

    for kind, count in counts.items():
        style = "" if count else "muted"
        table.add_row(
            f"[{style}]{kind.value}[/{style}]" if style else kind.value,
            kind.directory,
            str(count),
        )
    return table


def status_table(status: dict[str, bool], counts: dict[str, int], run_id: str) -> Table:
    """Stage-by-stage completion for one run — what `op inspect` shows."""
    table = Table(title=f"run {run_id}", title_style="muted", header_style="stage")
    table.add_column("#", justify="right", style="muted")
    table.add_column("stage")
    table.add_column("state")
    table.add_column("artifacts", justify="right")

    for index, (stage, done) in enumerate(status.items(), start=1):
        state = "[success]done[/success]" if done else "[muted]pending[/muted]"
        table.add_row(str(index), stage, state, str(counts.get(stage, 0)))
    return table


def outcome_table(run: PipelineRun) -> Table:
    """What each attempted stage did."""
    table = Table(title=f"run {run.run_id}", title_style="muted", header_style="stage")
    table.add_column("stage")
    table.add_column("status")
    table.add_column("produced", justify="right")
    table.add_column("note", overflow="fold", max_width=52)

    for outcome in run.outcomes:
        table.add_row(
            outcome.stage,
            _styled_status(outcome),
            str(len(outcome.produced)),
            escape(outcome.reason or ""),
        )
    return table


def stage_line(attempt: StageAttempt) -> str:
    """One line for one stage attempt, printed while the run is still going.

    A whole-run table would only appear once the run had ended, which is the one
    moment you no longer need it: `op auto` can spend a quarter of an hour inside
    a single stage, and silence for that long is indistinguishable from a hang.
    """
    style = STATUS_STYLE[attempt.outcome.status]
    mark = STATUS_MARK[attempt.outcome.status]
    retry = f" [warning](attempt {attempt.attempt})[/warning]" if attempt.attempt > 1 else ""
    note = _stage_note(attempt)
    return (
        f"  [{style}]{mark}[/{style}] [stage]{escape(attempt.stage):<24}[/stage]"
        f"{note}{retry} [muted]{attempt.seconds:.0f}s[/muted]"
    )


def _stage_note(attempt: StageAttempt) -> str:
    """What the attempt achieved, in as few words as carry the meaning.

    The failure reason is escaped: it comes from a provider, and a model error
    that happens to contain `[brackets]` would otherwise be eaten as styling —
    or raise, and lose the reason the run stopped.
    """
    outcome = attempt.outcome
    if outcome.status is StageStatus.COMPLETED:
        produced = f"{len(outcome.produced)} produced"
        resumed = f" [muted]+{outcome.reused} already done[/muted]" if outcome.reused else ""
        return f"{produced}{resumed}"
    if outcome.status is StageStatus.SKIPPED:
        return "[muted]already done[/muted]"
    # Styled by outcome rather than always red: a stage that found nothing is
    # not a stage that broke, and colouring the two alike is what sends someone
    # hunting for a bug in a run that worked and came back empty-handed.
    style = STATUS_STYLE.get(outcome.status, "danger")
    return f"[{style}]{escape(outcome.reason or outcome.status.value)}[/{style}]"


def decision_table(decisions: Sequence[Decision], titles: Mapping[str, str]) -> Table:
    """The verdicts a run reached — what someone asked for the run wants to see."""
    table = Table(title="decisions", title_style="muted", header_style="stage")
    table.add_column("opportunity", overflow="ellipsis", max_width=34)
    table.add_column("verdict")
    table.add_column("conf", justify="right")
    table.add_column("next validation step", overflow="ellipsis", max_width=44)

    for decision in decisions:
        verdict = decision.verdict.value
        style = VERDICT_STYLE.get(verdict, "info")
        table.add_row(
            escape(titles.get(decision.opportunity.id, decision.opportunity.id)),
            f"[{style}]{verdict}[/{style}]",
            f"{decision.decision_confidence:.2f}",
            escape(decision.next_validation_step),
        )
    return table


def spend_line(result: AutoResult) -> str:
    """The one-line footer: how much of the pipeline ran, and what it cost."""
    done = sum(1 for a in result.attempts if a.outcome.ok)
    parts = [
        f"{done}/{len(result.attempts)} stages",
        f"{result.spend.calls} calls",
        f"{result.spend.total_tokens:,} tokens",
        _money(result.spend),
        _duration(result.seconds),
    ]
    return "[muted]" + "  ·  ".join(part for part in parts if part) + "[/muted]"


def _money(spend: object) -> str:
    """Cost, said honestly: a BYOK run's zero is not a free run."""
    cost = getattr(spend, "cost", 0.0)
    byok = getattr(spend, "byok", 0)
    if byok:
        return f"${cost:.2f} (+{byok} BYOK, billed to your own key)"
    return f"${cost:.2f}"


def _duration(seconds: float | None) -> str:
    """A duration a person reads at a glance."""
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _styled_status(outcome: StageOutcome) -> str:
    style = STATUS_STYLE[outcome.status]
    return f"[{style}]{outcome.status.value}[/{style}]"


def _headline(artifact: Artifact) -> str:
    """The most human-readable one-liner an artifact offers.

    Callers escape the result before it reaches a table: every value here came
    from a model or a fetched page, and Rich would read a quote containing
    `[brackets]` as styling and silently drop it. Losing text out of an evidence
    excerpt is the one failure this project cannot tolerate.

    Artifacts do not share a title field — each kind names its headline
    differently — so this probes the usual suspects rather than forcing every
    model to carry a field it does not need.
    """
    for field in ("title", "label", "text", "summary", "objective", "rationale", "excerpt"):
        value = getattr(artifact, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


__all__ = [
    "STATUS_STYLE",
    "artifact_json",
    "artifact_table",
    "as_payload",
    "counts_table",
    "outcome_table",
    "print_json",
    "single_artifact_json",
    "status_table",
]
