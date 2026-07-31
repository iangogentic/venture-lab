"""The pipeline engine: order, resumability, and skipping completed stages."""

import pytest

from app.artifacts import Artifact, ArtifactKind, ArtifactRegistry, ArtifactStatus
from app.pipeline import STAGE_ORDER, PipelineEngine, StageStatus
from app.skills import SkillRequest, SkillResult
from app.utils.errors import PipelineError, SkillError
from app.utils.paths import WorkspacePaths
from tests.factories import make


def _seed_opportunities(
    engine: PipelineEngine,
    count: int,
    run_id: str = "r1",
) -> list[Artifact]:
    """A run populated through `discover-opportunities`, with honest lineage."""
    registry = engine.registry
    question = make(ArtifactKind.QUESTION, run_id=run_id)
    registry.save(question)
    evidence = make(ArtifactKind.EVIDENCE, run_id=run_id, parents=[question.ref])
    registry.save(evidence)
    brief = make(ArtifactKind.RESEARCH_BRIEF, run_id=run_id, parents=[evidence.ref])
    registry.save(brief)
    cluster = make(ArtifactKind.PAIN_CLUSTER, run_id=run_id, parents=[brief.ref])
    registry.save(cluster)

    opportunities = []
    for _ in range(count):
        opportunity = make(ArtifactKind.OPPORTUNITY, run_id=run_id, parents=[cluster.ref])
        registry.save(opportunity)
        opportunities.append(opportunity)
    return opportunities


def _rule_on(engine: PipelineEngine, opportunity: Artifact, run_id: str = "r1") -> None:
    """Every per-item artifact one opportunity's lineage carries when complete."""
    registry = engine.registry
    for kind in (
        ArtifactKind.MARKET_ANALYSIS,
        ArtifactKind.COMPETITION_ANALYSIS,
        ArtifactKind.CONTRADICTION_ANALYSIS,
    ):
        registry.save(
            make(kind, run_id=run_id, opportunity=opportunity.ref, parents=[opportunity.ref])
        )
    decision = make(
        ArtifactKind.DECISION,
        run_id=run_id,
        opportunity=opportunity.ref,
        parents=[opportunity.ref],
    )
    registry.save(decision)
    registry.save(
        make(
            ArtifactKind.INTERVIEW_PLAN,
            run_id=run_id,
            decision=decision.ref,
            parents=[decision.ref],
        )
    )


class _ScriptedSkill:
    """Stands in for a per-item skill: persists one canned analysis per request."""

    def __init__(self, registry: ArtifactRegistry, *, fail_first: bool = False) -> None:
        self.registry = registry
        self.fail_next = fail_first
        self.calls = 0

    def execute(self, request: SkillRequest) -> SkillResult:
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise SkillError("scripted failure")
        primary = request.artifacts[0]
        analysis = make(
            ArtifactKind.MARKET_ANALYSIS,
            run_id=request.run_id,
            opportunity=primary.ref,
            parents=[primary.ref],
        )
        self.registry.save(analysis)
        return SkillResult(skill="analyze-market", artifacts=[analysis])


class _BarrenSkill:
    """Runs cleanly and produces nothing — the shape of a run with no evidence."""

    def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult(skill="analyze-market", artifacts=[])


EXPECTED_ORDER = (
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


@pytest.fixture
def engine(workspace: WorkspacePaths) -> PipelineEngine:
    return PipelineEngine(ArtifactRegistry(workspace))


# ------------------------------------------------------------------- order


def test_stage_order_is_the_specified_pipeline() -> None:
    assert STAGE_ORDER == EXPECTED_ORDER


def test_unknown_stage_raises() -> None:
    with pytest.raises(PipelineError, match="Unknown stage"):
        PipelineEngine.skill_for("nope")


# --------------------------------------------------------------- selection


def test_select_defaults_to_the_whole_pipeline() -> None:
    assert PipelineEngine.select() == STAGE_ORDER


def test_select_only_one_stage() -> None:
    assert PipelineEngine.select(only="decision") == ("decision",)


def test_select_a_slice() -> None:
    selected = PipelineEngine.select(start_at="cluster-pains", stop_after="analyze-market")
    assert selected == (
        "cluster-pains",
        "discover-opportunities",
        "analyze-market",
    )


def test_select_refuses_a_backwards_slice() -> None:
    with pytest.raises(PipelineError, match="comes after"):
        PipelineEngine.select(start_at="decision", stop_after="collect-evidence")


def test_select_refuses_unknown_stages() -> None:
    with pytest.raises(PipelineError):
        PipelineEngine.select(start_at="nope")


# ------------------------------------------------------------ completion


def test_nothing_is_complete_in_an_empty_workspace(engine: PipelineEngine) -> None:
    assert engine.pending("r1") == STAGE_ORDER
    assert not any(engine.status("r1").values())


def test_a_stage_is_complete_once_its_artifacts_exist(engine: PipelineEngine) -> None:
    """Resumability rests on this: presence on disk is the record of what is done."""
    engine.registry.save(make(ArtifactKind.EVIDENCE, run_id="r1"))

    assert engine.is_complete("collect-evidence", "r1")
    assert "collect-evidence" not in engine.pending("r1")


def test_completion_is_scoped_to_a_run(engine: PipelineEngine) -> None:
    engine.registry.save(make(ArtifactKind.EVIDENCE, run_id="r1"))

    assert engine.is_complete("collect-evidence", "r1")
    assert not engine.is_complete("collect-evidence", "r2")


def test_inputs_are_gathered_from_the_declared_kinds(engine: PipelineEngine) -> None:
    market = make(ArtifactKind.MARKET_ANALYSIS, run_id="r1")
    competition = make(ArtifactKind.COMPETITION_ANALYSIS, run_id="r1")
    engine.registry.save(market)
    engine.registry.save(competition)
    engine.registry.save(make(ArtifactKind.EVIDENCE, run_id="r1"))

    inputs = engine.inputs_for("contradiction-analysis", "r1")

    assert {type(a).kind for a in inputs} == {
        ArtifactKind.MARKET_ANALYSIS,
        ArtifactKind.COMPETITION_ANALYSIS,
    }


# -------------------------------------------------------------- run_stage


def test_a_completed_stage_is_skipped(engine: PipelineEngine) -> None:
    engine.registry.save(make(ArtifactKind.EVIDENCE, run_id="r1"))

    outcome = engine.run_stage("collect-evidence", "r1")

    assert outcome.status is StageStatus.SKIPPED
    assert outcome.ok
    assert len(outcome.produced) == 1


def test_a_stage_without_inputs_is_blocked(engine: PipelineEngine) -> None:
    """Blocked, not failed: nothing went wrong, there is simply nothing to read yet."""
    outcome = engine.run_stage("research-brief", "r1")

    assert outcome.status is StageStatus.BLOCKED
    assert not outcome.ok
    assert "evidence" in (outcome.reason or "")


def test_a_stage_that_produces_nothing_is_empty_not_completed(
    engine: PipelineEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ✓ has to land on the stage that killed the run, not the one that noticed.

    Reported completed, the next stage blocks for want of input and the table
    shows a green collect-evidence above a yellow research-brief — which sends a
    reader to debug the stage that behaved correctly.
    """
    _seed_opportunities(engine, count=1)
    monkeypatch.setattr(engine, "build_skill", lambda stage: _BarrenSkill())

    outcome = engine.run_stage("analyze-market", "r1")

    assert outcome.status is StageStatus.EMPTY
    assert not outcome.ok, "the pipeline cannot continue past a stage with no output"
    assert not outcome.produced
    assert "market_analysis" in (outcome.reason or ""), "say which kind never arrived"


def test_an_empty_stage_is_not_a_failure(
    engine: PipelineEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing raised. Sources with nothing to say is a finding, not a bug."""
    _seed_opportunities(engine, count=1)
    monkeypatch.setattr(engine, "build_skill", lambda stage: _BarrenSkill())

    outcome = engine.run_stage("analyze-market", "r1")

    assert outcome.status is not StageStatus.FAILED


def test_a_resumed_stage_that_adds_nothing_is_still_completed(
    engine: PipelineEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emptiness is about the run's output, not this attempt's.

    One item ruled on by an earlier attempt means the stage has artifacts on
    disk and downstream has something to read — so a second attempt that adds
    nothing has completed the stage, not emptied it.
    """
    opportunities = _seed_opportunities(engine, count=2)
    scripted = _ScriptedSkill(engine.registry)
    monkeypatch.setattr(engine, "build_skill", lambda stage: scripted)
    engine.registry.save(
        make(
            ArtifactKind.MARKET_ANALYSIS,
            run_id="r1",
            opportunity=opportunities[0].ref,
            parents=[opportunities[0].ref],
        )
    )
    monkeypatch.setattr(engine, "build_skill", lambda stage: _BarrenSkill())

    outcome = engine.run_stage("analyze-market", "r1")

    assert outcome.status is StageStatus.COMPLETED
    assert outcome.reused == 1


def test_force_reruns_a_completed_stage(engine: PipelineEngine) -> None:
    """With no API key configured the forced run fails — but it is not skipped."""
    engine.registry.save(make(ArtifactKind.QUESTION, run_id="r1"))
    engine.registry.save(make(ArtifactKind.EVIDENCE, run_id="r1"))

    outcome = engine.run_stage("collect-evidence", "r1", force=True)

    assert outcome.status is not StageStatus.SKIPPED


def test_a_failing_stage_is_reported_not_raised(engine: PipelineEngine) -> None:
    """One bad stage must not take down the whole invocation."""
    engine.registry.save(make(ArtifactKind.QUESTION, run_id="r1"))

    outcome = engine.run_stage("collect-evidence", "r1")

    assert outcome.status is StageStatus.FAILED
    assert outcome.reason


# --------------------------------------------------------------------- run


def test_run_halts_at_the_first_blocked_stage(engine: PipelineEngine) -> None:
    run = engine.run("r1")

    assert [o.stage for o in run.outcomes] == ["collect-evidence"]
    assert not run.ok


def test_run_skips_completed_stages_and_stops_at_the_first_gap(
    engine: PipelineEngine,
) -> None:
    """The resume path: everything already done is skipped, work restarts at the gap."""
    engine.registry.save(make(ArtifactKind.QUESTION, run_id="r1"))
    engine.registry.save(make(ArtifactKind.EVIDENCE, run_id="r1"))
    engine.registry.save(make(ArtifactKind.RESEARCH_BRIEF, run_id="r1"))

    run = engine.run("r1")

    assert run.skipped == ["collect-evidence", "research-brief"]
    assert run.outcomes[-1].stage == "cluster-pains"


def test_keep_going_attempts_later_stages(engine: PipelineEngine) -> None:
    run = engine.run("r1", stop_on_error=False)

    assert len(run.outcomes) == len(STAGE_ORDER)
    assert not run.ok


def test_run_respects_a_slice(engine: PipelineEngine) -> None:
    run = engine.run("r1", start_at="analyze-market", stop_after="analyze-competition")

    assert [o.stage for o in run.outcomes] == ["analyze-market"]


def test_a_fully_populated_run_is_all_skips(engine: PipelineEngine) -> None:
    for opportunity in _seed_opportunities(engine, count=2):
        _rule_on(engine, opportunity)

    run = engine.run("r1")

    assert run.ok
    assert run.skipped == list(STAGE_ORDER)
    assert engine.pending("r1") == ()


# ------------------------------------------------- per-item resume and force


def test_a_partially_ruled_per_item_stage_is_not_complete(engine: PipelineEngine) -> None:
    """One analysis over three opportunities is a stage interrupted, not finished.

    This is the resume-safety property: treating it as complete is how the other
    two opportunities used to be silently dropped after a mid-stage failure.
    """
    opportunities = _seed_opportunities(engine, count=3)
    engine.registry.save(
        make(
            ArtifactKind.MARKET_ANALYSIS,
            run_id="r1",
            opportunity=opportunities[0].ref,
            parents=[opportunities[0].ref],
        )
    )

    assert not engine.is_complete("analyze-market", "r1")
    assert "analyze-market" in engine.pending("r1")


def test_resume_requests_only_the_missing_items(engine: PipelineEngine) -> None:
    opportunities = _seed_opportunities(engine, count=3)
    engine.registry.save(
        make(
            ArtifactKind.MARKET_ANALYSIS,
            run_id="r1",
            opportunity=opportunities[0].ref,
            parents=[opportunities[0].ref],
        )
    )

    requests = engine.requests_for("analyze-market", "r1", pending_only=True)

    assert {request.artifacts[0].id for request in requests} == {
        opportunities[1].id,
        opportunities[2].id,
    }


def test_one_failing_item_does_not_abandon_the_rest(
    engine: PipelineEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_opportunities(engine, count=2)
    scripted = _ScriptedSkill(engine.registry, fail_first=True)
    monkeypatch.setattr(engine, "build_skill", lambda stage: scripted)

    outcome = engine.run_stage("analyze-market", "r1")

    assert outcome.status is StageStatus.FAILED
    assert len(outcome.produced) == 1
    assert "scripted failure" in (outcome.reason or "")


def test_a_failed_per_item_stage_resumes_at_the_missing_item(
    engine: PipelineEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_opportunities(engine, count=2)
    scripted = _ScriptedSkill(engine.registry, fail_first=True)
    monkeypatch.setattr(engine, "build_skill", lambda stage: scripted)
    engine.run_stage("analyze-market", "r1")

    outcome = engine.run_stage("analyze-market", "r1")

    assert outcome.status is StageStatus.COMPLETED
    assert outcome.reused == 1  # the item that succeeded first time was not re-bought
    assert scripted.calls == 3  # two attempts in the first pass, one in the second
    assert engine.is_complete("analyze-market", "r1")


def test_force_supersedes_what_it_replaces(
    engine: PipelineEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced re-run must retire the old artifact, not double it."""
    opportunities = _seed_opportunities(engine, count=1)
    old = make(
        ArtifactKind.MARKET_ANALYSIS,
        run_id="r1",
        opportunity=opportunities[0].ref,
        parents=[opportunities[0].ref],
    )
    engine.registry.save(old)
    scripted = _ScriptedSkill(engine.registry)
    monkeypatch.setattr(engine, "build_skill", lambda stage: scripted)

    outcome = engine.run_stage("analyze-market", "r1", force=True)

    assert outcome.status is StageStatus.COMPLETED
    survivors = engine.consumable_of(ArtifactKind.MARKET_ANALYSIS, "r1")
    assert [a.id for a in survivors] == [ref.id for ref in outcome.produced]
    retired = engine.registry.load(ArtifactKind.MARKET_ANALYSIS, old.id)
    assert retired.status is ArtifactStatus.SUPERSEDED
    assert survivors[0].supersedes == old.ref
    assert engine.is_complete("analyze-market", "r1")
