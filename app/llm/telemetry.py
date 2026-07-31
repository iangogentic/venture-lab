"""A durable record of every model call the pipeline makes.

One JSONL line per call, appended and never rewritten. The point is empirical:
after a hundred runs this file answers questions no amount of reasoning about the
architecture can — which stage actually costs the money, whether a prompt edit
changed latency, whether the expensive tier is earning its place on the stages
routed to it.

It is deliberately not the artifact store. Artifacts record *what the pipeline
concluded*; this records *what it cost to conclude it*, and the two have different
lifetimes: a workspace can be cleaned without losing the cost history, and the cost
history can be deleted without touching a single finding.

Append-only JSONL rather than a table because it survives a crash mid-write, needs
no migration when a field is added, and can be read with `jq` by someone who has
never seen this codebase.
"""

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.paths import WorkspacePaths, get_workspace_paths
from app.utils.time import utcnow

logger = get_logger(__name__)


class CallRecord(BaseModel):
    """What one model call cost, and exactly what produced it.

    Carries enough to reproduce the call's routing decision — capability, tier,
    resolved model, prompt digest — because "the output changed" is only useful
    alongside "and here is what was different about the request".
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=utcnow)
    run_id: str
    skill: str

    # --- routing ---
    capability: str | None = None
    tier: str | None = None
    model: str | None = Field(default=None, description="The slug that actually answered.")

    # --- what was asked ---
    prompt_name: str | None = None
    prompt_digest: str | None = Field(
        default=None,
        description="Content hash of the prompt. This is the prompt's version: two calls "
        "with the same digest were asked the same thing.",
    )
    fingerprint: str | None = None
    input_artifacts: list[str] = Field(default_factory=list)

    # --- what came back ---
    output_artifacts: list[str] = Field(default_factory=list)
    cached: bool = Field(
        default=False,
        description="True when an identical computation was reused and no model was called. "
        "Kept in the log so reuse is measurable, not invisible.",
    )
    error: str | None = None

    # --- what it cost ---
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    is_byok: bool = Field(
        default=False,
        description="Billed against your own provider key. Cost reads zero here and is "
        "real somewhere else.",
    )

    @property
    def billable(self) -> bool:
        """Whether this call actually reached a provider."""
        return not self.cached and self.error is None


class TelemetrySink:
    """Appends call records to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, call: CallRecord) -> None:
        """Append one record.

        Never raises. Telemetry that can break a pipeline run is worse than no
        telemetry — a full disk or a read-only workspace must cost you the
        measurement, not the research.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(call.model_dump_json() + "\n")
        except OSError as exc:
            logger.debug("could not write telemetry: %s", exc)

    def read(self, *, run_id: str | None = None) -> list[CallRecord]:
        """Every record, optionally narrowed to one run."""
        return [call for call in self._iter() if run_id is None or call.run_id == run_id]

    def _iter(self) -> Iterator[CallRecord]:
        """Stream records, skipping any line that cannot be parsed.

        A partial final line is expected after a crash; losing it should not make
        the whole history unreadable.
        """
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield CallRecord.model_validate(json.loads(stripped))
                except (ValueError, TypeError):
                    continue


class CallSummary(BaseModel):
    """Aggregated cost for a group of calls."""

    model_config = ConfigDict(extra="forbid")

    label: str
    calls: int = 0
    cached: int = 0
    errors: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    byok: int = 0
    latency_ms: float = 0.0

    @property
    def mean_latency_ms(self) -> float:
        """Mean over calls that actually reached a provider."""
        billable = self.calls - self.cached - self.errors
        return self.latency_ms / billable if billable else 0.0


def default_sink(paths: WorkspacePaths | None = None) -> TelemetrySink:
    """The sink the pipeline writes to, resolved from settings.

    A relative `TELEMETRY_PATH` lands inside the workspace, so a workspace stays
    self-describing when it is copied. Resolved here rather than at each call
    site: three copies of this rule is three chances for the writer and a reader
    to disagree about which file the history is in.
    """
    path = get_settings().telemetry_path
    if path.is_absolute():
        return TelemetrySink(path)
    root = (paths if paths is not None else get_workspace_paths()).root
    return TelemetrySink(root / path)


def summarise(calls: list[CallRecord], *, by: str = "skill") -> list[CallSummary]:
    """Group records and total their cost.

    `by` is an attribute name — `skill`, `model`, `capability`, `tier`, `run_id` —
    so the same function answers "which stage costs most" and "which model costs
    most" without a second implementation.
    """
    grouped: dict[str, CallSummary] = {}
    for call in calls:
        label = str(getattr(call, by, None) or "-")
        summary = grouped.setdefault(label, CallSummary(label=label))
        summary.calls += 1
        summary.cached += int(call.cached)
        summary.errors += int(call.error is not None)
        summary.total_tokens += call.total_tokens or 0
        summary.cost += call.cost or 0.0
        summary.byok += int(call.is_byok)
        summary.latency_ms += call.latency_ms or 0.0
    return sorted(grouped.values(), key=lambda s: (-s.cost, s.label))


__all__ = ["CallRecord", "CallSummary", "TelemetrySink", "default_sink", "summarise"]
