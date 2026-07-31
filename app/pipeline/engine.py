"""The pipeline engine: run the nine stages in order, resumably.

A stage *is* a skill — there is no separate Stage class. The order below is the
pipeline, and each entry names the skill that implements it. Composition lives
here and nowhere else, which is why skills are forbidden from calling each other:
a hidden skill-to-skill call would be work the engine cannot see, cannot skip and
cannot resume.

Resumability needs no database. A stage is complete when the workspace already
holds its output for this run — counted at the stage's own grain: any artifact
of the produced kind for a fan-in stage, one per primary item for a per-item
stage. That makes the question "what is left to do?" answerable by listing
files, and it survives a crash, a machine change, or a `git clone` of the
workspace.
"""

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactRef,
    ArtifactRegistry,
    ArtifactStatus,
    Question,
)
from app.artifacts.lineage import referenced_ids, related_to
from app.llm import LLM
from app.skills import Batching, Skill, SkillRequest, get_skill
from app.utils.errors import PipelineError, SkillError
from app.utils.logging import get_logger

logger = get_logger(__name__)

STAGE_ORDER: Final[tuple[str, ...]] = (
    "collect-evidence",
    "research-brief",
    "cluster-pains",
    "discover-opportunities",
    "analyze-market",
    "analyze-competition",
    "contradiction-analysis",
    "decision",
    "interview-plan",
)
"""The pipeline, in order. Each name is also the skill name and the prompt file stem."""

_RETIRED: Final[frozenset[ArtifactStatus]] = frozenset(
    {ArtifactStatus.SUPERSEDED, ArtifactStatus.ARCHIVED}
)
"""Statuses that no longer count as work done: replaced, or shelved."""

_NOT_CONSUMABLE: Final[frozenset[ArtifactStatus]] = _RETIRED | {ArtifactStatus.REJECTED}
"""Statuses downstream stages must not read. A rejected artifact still counts as
handled work — a reviewer looked and said no — but its content is out of play."""


class StageStatus(StrEnum):
    """What happened to one stage in one run."""

    COMPLETED = "completed"
    """The stage ran and produced artifacts."""

    SKIPPED = "skipped"
    """Already done for this run; its artifacts were left alone."""

    EMPTY = "empty"
    """The stage ran, without error, and produced nothing.

    Distinct from both of its neighbours, because it is genuinely neither. It is
    not `FAILED`: nothing raised, the answer is simply that there was none, and
    a run whose sources have nothing to say is a real finding rather than a bug.
    It is not `COMPLETED` either — the pipeline cannot continue past it, and
    reporting it green while the *next* stage blocks for want of input puts the
    ✓ on the stage that killed the run and the ! on the one that noticed. A real
    run did exactly that: `collect-evidence completed — 0 produced` above
    `research-brief blocked — no evidence artifacts`."""

    BLOCKED = "blocked"
    """An upstream stage produced nothing, so there was no input to work from."""

    FAILED = "failed"
    """At least one item raised. Whatever succeeded before or after it is on disk."""


class StageOutcome(BaseModel):
    """The result of attempting one stage."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    status: StageStatus
    produced: list[ArtifactRef] = Field(default_factory=list)
    reused: int = Field(
        default=0,
        description="Items answered by work already on disk rather than by asking a "
        "model — the primaries an earlier, interrupted attempt had finished. High "
        "here means a resume cost almost nothing.",
    )
    reason: str | None = Field(
        default=None,
        description="Why it was skipped, blocked, or how it failed.",
    )

    @property
    def ok(self) -> bool:
        """Whether the pipeline may continue past this stage."""
        return self.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)


class PipelineRun(BaseModel):
    """Every stage outcome from one invocation."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    outcomes: list[StageOutcome] = Field(default_factory=list)

    @property
    def completed(self) -> list[str]:
        """Stages that actually ran."""
        return [o.stage for o in self.outcomes if o.status is StageStatus.COMPLETED]

    @property
    def skipped(self) -> list[str]:
        """Stages that were already done."""
        return [o.stage for o in self.outcomes if o.status is StageStatus.SKIPPED]

    @property
    def failed(self) -> list[str]:
        """Stages that failed or were blocked."""
        return [o.stage for o in self.outcomes if not o.ok]

    @property
    def ok(self) -> bool:
        """Whether every attempted stage succeeded or was already done."""
        return all(o.ok for o in self.outcomes)

    @property
    def produced(self) -> list[ArtifactRef]:
        """Everything written across the run."""
        return [ref for outcome in self.outcomes for ref in outcome.produced]


class PipelineEngine:
    """Runs stages, deciding what to skip and what each stage is given."""

    def __init__(
        self,
        registry: ArtifactRegistry | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.registry = registry if registry is not None else ArtifactRegistry()
        self.llm = llm

    # ------------------------------------------------------------------ lookup

    @staticmethod
    def stages() -> tuple[str, ...]:
        """The pipeline order."""
        return STAGE_ORDER

    @staticmethod
    def skill_for(stage: str) -> type[Skill]:
        """The skill class implementing a stage.

        Raises:
            PipelineError: If `stage` is not part of the pipeline.
        """
        if stage not in STAGE_ORDER:
            known = ", ".join(STAGE_ORDER)
            raise PipelineError(f"Unknown stage {stage!r}. Pipeline stages: {known}")
        try:
            return get_skill(stage)
        except SkillError as exc:
            raise PipelineError(f"No skill registered for stage {stage!r}") from exc

    def build_skill(self, stage: str) -> Skill:
        """Instantiate the skill for a stage, wired to this engine's collaborators."""
        return self.skill_for(stage)(llm=self.llm, registry=self.registry)

    # -------------------------------------------------------------- run state

    def question_for(self, run_id: str) -> Question | None:
        """The Question seeding a run, if one has been written."""
        questions = self.registry.find_by_type(ArtifactKind.QUESTION, run_id=run_id)
        found = questions[0] if questions else None
        return found if isinstance(found, Question) else None

    def artifacts_of(self, kind: ArtifactKind, run_id: str) -> list[Artifact]:
        """Every artifact of one kind belonging to a run, whatever its status."""
        return self.registry.find_by_type(kind, run_id=run_id)

    def produced_of(self, kind: ArtifactKind, run_id: str) -> list[Artifact]:
        """Artifacts that count as completed work: everything not superseded or archived."""
        return [a for a in self.artifacts_of(kind, run_id) if a.status not in _RETIRED]

    def consumable_of(self, kind: ArtifactKind, run_id: str) -> list[Artifact]:
        """Artifacts a downstream stage may read: completed work minus the rejected."""
        return [a for a in self.artifacts_of(kind, run_id) if a.status not in _NOT_CONSUMABLE]

    def is_complete(self, stage: str, run_id: str) -> bool:
        """Whether this stage already did its work for this run.

        Presence on disk is still the record, but counted at the stage's own
        grain. A fan-in stage is done when any artifact of its kind exists; a
        per-item stage only when every primary item has been ruled on. One
        market analysis over three opportunities is a stage interrupted, not a
        stage finished — treating it as done is how items get silently dropped.
        """
        skill = self.skill_for(stage)
        produced = self.produced_of(skill.produces, run_id)
        if skill.batching is Batching.PER_ITEM and skill.primary_kind is not None:
            primaries = self.consumable_of(skill.primary_kind, run_id)
            if primaries:
                return not self._unhandled(primaries, produced)
        return bool(produced)

    @staticmethod
    def _unhandled(primaries: list[Artifact], produced: list[Artifact]) -> list[Artifact]:
        """The primary items no produced artifact points back at."""
        handled: set[str] = set()
        for artifact in produced:
            handled |= referenced_ids(artifact)
        return [primary for primary in primaries if primary.id not in handled]

    def pending(self, run_id: str) -> tuple[str, ...]:
        """Stages still to do for a run, in order."""
        return tuple(stage for stage in STAGE_ORDER if not self.is_complete(stage, run_id))

    def status(self, run_id: str) -> dict[str, bool]:
        """Completion of every stage, in pipeline order — what `op inspect` renders."""
        return {stage: self.is_complete(stage, run_id) for stage in STAGE_ORDER}

    def inputs_for(self, stage: str, run_id: str) -> list[Artifact]:
        """Load everything a stage may consume, in the order it declared.

        Superseded, rejected and archived artifacts are excluded here — this is
        the one gate through which every stage reads, so a retired artifact
        cannot leak back into a prompt or be double-counted as evidence.
        """
        skill = self.skill_for(stage)
        inputs: list[Artifact] = []
        for kind in skill.consumes:
            inputs.extend(self.consumable_of(kind, run_id))
        return inputs

    def requests_for(
        self,
        stage: str,
        run_id: str,
        *,
        pending_only: bool = False,
    ) -> list[SkillRequest]:
        """The calls a stage will make, each carrying only the context it needs.

        A `FAN_IN` stage gets one request holding everything it consumes, because
        synthesising across sources is the work. A `PER_ITEM` stage gets one
        request per primary artifact, holding that artifact and the things linked
        to it — so ruling on one opportunity does not mean reading every other
        opportunity's analysis, and the cost of a run stops growing quadratically
        with the number of candidates.

        `pending_only` narrows a `PER_ITEM` stage to the primaries not yet ruled
        on — the resume path after an interrupted attempt.
        """
        skill = self.skill_for(stage)
        question = self.question_for(run_id)
        available = self.inputs_for(stage, run_id)

        if skill.batching is Batching.FAN_IN or skill.primary_kind is None:
            return [SkillRequest(run_id=run_id, artifacts=available, question=question)]

        primaries = [a for a in available if type(a).kind is skill.primary_kind]
        others = [a for a in available if type(a).kind is not skill.primary_kind]
        if pending_only:
            primaries = self._unhandled(primaries, self.produced_of(skill.produces, run_id))

        return [
            SkillRequest(
                run_id=run_id,
                artifacts=[primary, *related_to(primary, others)],
                question=question,
            )
            for primary in primaries
        ]

    # ------------------------------------------------------------------- run

    def run_stage(self, stage: str, run_id: str, *, force: bool = False) -> StageOutcome:
        """Run one stage. Skips work already done unless `force`.

        A per-item stage resumes at the item level: primaries already ruled on
        by an earlier, interrupted attempt are left alone and only the missing
        ones are attempted. One failing item does not abandon the rest — every
        request is attempted and the failures are reported together. Under
        `force`, each artifact a re-run replaces is marked superseded once its
        replacement is safely on disk, so a stale result can never sit next to
        its successor and be double-counted downstream.
        """
        skill_cls = self.skill_for(stage)

        if not force and self.is_complete(stage, run_id):
            existing = self.produced_of(skill_cls.produces, run_id)
            return StageOutcome(
                stage=stage,
                status=StageStatus.SKIPPED,
                produced=[artifact.ref for artifact in existing],
                reason=f"{len(existing)} {skill_cls.produces.value} artifact(s) already present",
            )

        inputs = self.inputs_for(stage, run_id)
        for kind in skill_cls.consumes:
            if not any(type(a).kind is kind for a in inputs):
                return StageOutcome(
                    stage=stage,
                    status=StageStatus.BLOCKED,
                    reason=f"no {kind.value} artifacts for run {run_id!r}",
                )

        requests = self.requests_for(stage, run_id, pending_only=not force)
        if not requests:
            return StageOutcome(
                stage=stage,
                status=StageStatus.BLOCKED,
                reason=f"nothing of kind {skill_cls.primary_kind} to work on",
            )

        already = 0
        if not force and skill_cls.batching is Batching.PER_ITEM:
            total = sum(1 for a in inputs if type(a).kind is skill_cls.primary_kind)
            already = total - len(requests)

        replaceable = self.produced_of(skill_cls.produces, run_id) if force else []
        retired: set[str] = set()

        skill = self.build_skill(stage)
        produced: list[ArtifactRef] = []
        failures: list[str] = []
        for index, request in enumerate(requests, start=1):
            try:
                result = skill.execute(request)
            except Exception as exc:  # one bad item must not kill the rest
                logger.exception("stage %s failed on item %d/%d", stage, index, len(requests))
                failures.append(f"item {index} of {len(requests)}: {exc}")
                continue
            produced.extend(result.refs)
            if replaceable:
                self._supersede(
                    replaceable,
                    request,
                    result.artifacts,
                    retired,
                    per_item=skill_cls.batching is Batching.PER_ITEM,
                )

        if failures:
            return StageOutcome(
                stage=stage,
                status=StageStatus.FAILED,
                produced=produced,
                reused=already,
                reason="; ".join(failures),
            )
        if not produced and not already:
            # Said here, where it happened, rather than left for the next stage
            # to notice as missing input. Both stages are telling the truth, but
            # only this one can say which sources were asked and came back with
            # nothing — and it is the one a reader needs to go and look at.
            return StageOutcome(
                stage=stage,
                status=StageStatus.EMPTY,
                reason=(
                    f"ran without error but produced no {skill_cls.produces.value} "
                    f"artifacts, so there is nothing for the rest of the pipeline to "
                    f"work from"
                ),
            )
        note = f"resumed: {already} item(s) already done" if already else None
        return StageOutcome(
            stage=stage,
            status=StageStatus.COMPLETED,
            produced=produced,
            reused=already,
            reason=note,
        )

    def _supersede(
        self,
        replaceable: list[Artifact],
        request: SkillRequest,
        fresh: list[Artifact],
        retired: set[str],
        *,
        per_item: bool,
    ) -> None:
        """Retire what a forced re-run has just replaced.

        A fan-in stage replaces its whole prior output; a per-item stage only
        what descends from this request's artifacts. When the replacement is
        unambiguous — one old artifact for this request — the new artifacts also
        record it in `supersedes`, so the audit trail says not just that the old
        one retired but what stands in its place.
        """
        fresh_ids = {artifact.id for artifact in fresh}
        request_ids = {artifact.id for artifact in request.artifacts}
        replaced: list[Artifact] = []
        for old in replaceable:
            if old.id in retired or old.id in fresh_ids:
                continue
            if per_item and not (referenced_ids(old) & request_ids):
                continue
            self.registry.update(old, status=ArtifactStatus.SUPERSEDED)
            retired.add(old.id)
            replaced.append(old)
        if len(replaced) == 1:
            for artifact in fresh:
                self.registry.update(artifact, supersedes=replaced[0].ref)

    def run(
        self,
        run_id: str,
        *,
        only: str | None = None,
        start_at: str | None = None,
        stop_after: str | None = None,
        force: bool = False,
        stop_on_error: bool = True,
    ) -> PipelineRun:
        """Run a slice of the pipeline.

        Args:
            run_id: The run to work on. Artifacts are scoped to it.
            only: Run exactly one stage.
            start_at: First stage to attempt.
            stop_after: Last stage to attempt.
            force: Re-run stages that are already complete.
            stop_on_error: Halt at the first failed or blocked stage. Off, the
                engine attempts every remaining stage, which is useful for seeing
                how far a run *could* get.
        """
        selected = self.select(only=only, start_at=start_at, stop_after=stop_after)
        run = PipelineRun(run_id=run_id)

        for stage in selected:
            outcome = self.run_stage(stage, run_id, force=force)
            run.outcomes.append(outcome)
            if not outcome.ok and stop_on_error:
                logger.warning("halting at %s: %s", stage, outcome.reason)
                break

        return run

    @staticmethod
    def select(
        *,
        only: str | None = None,
        start_at: str | None = None,
        stop_after: str | None = None,
    ) -> tuple[str, ...]:
        """Resolve a stage selection to an ordered slice of the pipeline.

        Raises:
            PipelineError: On an unknown stage name, or a slice that runs backwards.
        """
        if only is not None:
            PipelineEngine.skill_for(only)
            return (only,)

        start = 0 if start_at is None else PipelineEngine._index_of(start_at)
        stop = len(STAGE_ORDER) - 1 if stop_after is None else PipelineEngine._index_of(stop_after)

        if start > stop:
            raise PipelineError(f"--start-at {start_at!r} comes after --stop-after {stop_after!r}")
        return STAGE_ORDER[start : stop + 1]

    @staticmethod
    def _index_of(stage: str) -> int:
        try:
            return STAGE_ORDER.index(stage)
        except ValueError as exc:
            known = ", ".join(STAGE_ORDER)
            raise PipelineError(f"Unknown stage {stage!r}. Pipeline stages: {known}") from exc


__all__ = [
    "STAGE_ORDER",
    "PipelineEngine",
    "PipelineRun",
    "StageOutcome",
    "StageStatus",
]
