"""Composing a run's report, for the two callers that need it.

`op report` composes on demand and `op auto` composes at the end of a run. They
must agree about what a report requires, what it supersedes, and when an existing
one is reused — so the rule lives here once and both call it, rather than in the
CLI where only one of them could see it.

Composition stays outside `STAGE_ORDER` for the reasons in
`app.skills.compose_report`: a report is a derived view that nothing downstream
consumes, so making it a stage would teach the resume machinery a dependency
that exists only to satisfy the machinery.
"""

from typing import Final

from pydantic import BaseModel, ConfigDict

from app.artifacts import Artifact, ArtifactKind, ArtifactRegistry, ArtifactStatus, Report
from app.llm import LLM
from app.pipeline.engine import PipelineEngine
from app.skills import SkillRequest
from app.skills.compose_report import ComposeReportSkill
from app.utils.errors import PipelineError

REQUIRED: Final[tuple[tuple[ArtifactKind, str], ...]] = (
    (ArtifactKind.RESEARCH_BRIEF, "research-brief"),
    (ArtifactKind.OPPORTUNITY, "discover-opportunities"),
    (ArtifactKind.DECISION, "decision"),
)
"""What a report cannot be written without, and the stage that produces each.

Only these three gate composition; market, competition, contradiction, interview
and lead artifacts are folded in by the skill when the run has them.
"""


class ReportUnavailableError(PipelineError):
    """The run has not produced enough to report on."""

    def __init__(self, run_id: str, stages: list[str]) -> None:
        super().__init__(
            f"nothing to report for run {run_id!r} — missing output from {', '.join(stages)}"
        )
        self.run_id = run_id
        self.stages = stages
        """The stages that still have to run, in pipeline order."""


class Composition(BaseModel):
    """A run's report, and whether this call is what wrote it."""

    model_config = ConfigDict(extra="forbid")

    report: Report
    composed: bool
    """False when an existing report was reused rather than regenerated."""


def compose_report(
    run_id: str,
    *,
    registry: ArtifactRegistry | None = None,
    llm: LLM | None = None,
    force: bool = False,
) -> Composition:
    """Compose the run's report, reusing an existing one unless `force`.

    Raises:
        ReportUnavailableError: If the run is missing a prerequisite stage.
        SkillError: If the model's reply could not be validated — including a
            citation naming an artifact this report does not cover.
    """
    engine = PipelineEngine(registry if registry is not None else ArtifactRegistry(), llm)
    store = engine.registry

    inputs, missing = _gather(engine, run_id)
    if missing:
        raise ReportUnavailableError(run_id, missing)

    existing = [
        artifact
        for artifact in engine.produced_of(ArtifactKind.REPORT, run_id)
        if isinstance(artifact, Report)
    ]
    if existing and not force:
        return Composition(
            report=max(existing, key=lambda report: report.created_at), composed=False
        )

    result = ComposeReportSkill(llm=llm, registry=store).execute(
        SkillRequest(run_id=run_id, artifacts=inputs, question=engine.question_for(run_id))
    )
    composed = next((a for a in result.artifacts if isinstance(a, Report)), None)
    if composed is None:
        raise PipelineError(f"compose-report produced no report for run {run_id!r}")

    # Only after the replacement is safely on disk — `PipelineEngine._supersede`'s
    # spirit: a stale report must never sit beside its successor with no way to
    # tell which one the run stands behind.
    fresh_ids = {artifact.id for artifact in result.artifacts}
    for old in existing:
        if old.id not in fresh_ids:
            store.update(old, status=ArtifactStatus.SUPERSEDED)

    return Composition(report=composed, composed=True)


def _gather(engine: PipelineEngine, run_id: str) -> tuple[list[Artifact], list[str]]:
    """The artifacts a report covers, and the stages still missing.

    Reads through the same gate every stage reads through, so superseded,
    rejected and archived artifacts are never narrated.
    """
    inputs: list[Artifact] = []
    missing: list[str] = []
    for kind, stage in REQUIRED:
        found = engine.consumable_of(kind, run_id)
        inputs.extend(found)
        if not found:
            missing.append(stage)
    return inputs, missing


__all__ = ["REQUIRED", "Composition", "ReportUnavailableError", "compose_report"]
