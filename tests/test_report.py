"""The REPORT layer: the compose-report skill and `op report`.

The guarantees under test are the layer's whole point: the model writes only the
prose, every fact of record (title, verdict tallies, covered period, parents) is
computed from the artifacts, a citation must name a covered artifact, and the
optional kinds never block a report.
"""

import json
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from app.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactRegistry,
    ArtifactStatus,
    Report,
    ReportFormat,
    Verdict,
)
from app.cli.main import app
from app.llm import LLM, GenerationRequest, GenerationResult, Provider, ProviderAdapter, TokenUsage
from app.pipeline import STAGE_ORDER
from app.skills import SKILLS, SkillRequest, SkillResult
from app.skills.compose_report import ComposeReportInput, ComposeReportSkill
from app.utils.errors import SkillError
from app.utils.paths import WorkspacePaths
from tests.factories import make

runner = CliRunner()

RUN = "r1"


# --------------------------------------------------------------- test doubles


class _EchoAdapter(ProviderAdapter):
    """Returns a fixed JSON reply, so the real skill runs with no network."""

    provider: ClassVar[Provider] = Provider.CLAUDE
    default_model: ClassVar[str] = "fake/model"

    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply

    def generate(
        self,
        request: GenerationRequest,
        *,
        response_format: object | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            text=json.dumps(self.reply),
            model=self.default_model,
            provider=self.provider,
            usage=TokenUsage(),
            finish_reason="stop",
        )


def _skill(reply: dict[str, Any], workspace: WorkspacePaths) -> ComposeReportSkill:
    return ComposeReportSkill(
        llm=LLM(adapter=_EchoAdapter(reply)),
        registry=ArtifactRegistry(workspace),
    )


def _reply(**overrides: Any) -> dict[str, Any]:
    return {
        "executive_summary": "One opportunity, decided build on corroborated pain.",
        "highlights": ["Build-cache pain recurs across sources."],
        "body": "# Findings\n\nThe pain recurs and the decision was build.",
        **overrides,
    }


def _core(run_id: str = RUN) -> dict[str, Any]:
    """A question plus the three required kinds, wired together."""
    question = make(ArtifactKind.QUESTION, run_id=run_id)
    brief = make(ArtifactKind.RESEARCH_BRIEF, run_id=run_id)
    opportunity = make(ArtifactKind.OPPORTUNITY, run_id=run_id)
    decision = make(
        ArtifactKind.DECISION,
        run_id=run_id,
        opportunity=ArtifactRef(kind=ArtifactKind.OPPORTUNITY, id=opportunity.id),
    )
    return {
        "question": question,
        "brief": brief,
        "opportunity": opportunity,
        "decision": decision,
    }


def _request(core: dict[str, Any], *extra_decisions: Any) -> SkillRequest:
    return SkillRequest(
        run_id=RUN,
        artifacts=[core["brief"], core["opportunity"], core["decision"], *extra_decisions],
        question=core["question"],
    )


# --------------------------------------------------------------------- gather


def test_gather_optional_kinds_are_truly_optional(workspace: WorkspacePaths) -> None:
    """A run with no analyses, plans or leads still reports — empty, not refused."""
    skill = _skill(_reply(), workspace)

    payload = skill.gather(_request(_core()))

    assert isinstance(payload, ComposeReportInput)
    assert payload.clusters == []
    assert payload.market == []
    assert payload.competition == []
    assert payload.contradictions == []
    assert payload.interviews == []
    assert payload.leads == []


def test_gather_pulls_optional_kinds_from_the_registry(workspace: WorkspacePaths) -> None:
    """Optional kinds come from disk, through the same status gate every stage reads."""
    registry = ArtifactRegistry(workspace)
    kept = make(ArtifactKind.MARKET_ANALYSIS, run_id=RUN)
    retired = make(ArtifactKind.MARKET_ANALYSIS, run_id=RUN, status=ArtifactStatus.SUPERSEDED)
    lead = make(ArtifactKind.LEAD, run_id=RUN)
    cluster = make(ArtifactKind.PAIN_CLUSTER, run_id=RUN)
    for artifact in (kept, retired, lead, cluster):
        registry.save(artifact)

    skill = _skill(_reply(), workspace)
    payload = skill.gather(_request(_core()))

    assert isinstance(payload, ComposeReportInput)
    assert [entry["id"] for entry in payload.market] == [kept.id]
    assert [entry["id"] for entry in payload.leads] == [lead.id]
    assert [entry["id"] for entry in payload.clusters] == [cluster.id]


# -------------------------------------------------------- execute / assemble


def test_execute_overwrites_model_tallies_with_computed_counts(
    workspace: WorkspacePaths,
) -> None:
    """Verdicts are tallied from the Decision artifacts; the model's claim is discarded."""
    core = _core()
    others = [
        make(ArtifactKind.DECISION, run_id=RUN, verdict=verdict)
        for verdict in (Verdict.BUILD, Verdict.WAIT)
    ]
    skill = _skill(_reply(verdict_counts={"reject": 7}), workspace)

    result = skill.execute(_request(core, *others))

    (report,) = result.artifacts
    assert isinstance(report, Report)
    assert report.verdict_counts == {Verdict.BUILD: 2, Verdict.WAIT: 1}


def test_execute_rejects_a_cited_id_that_does_not_exist(workspace: WorkspacePaths) -> None:
    body = "The pain is well attested (ev_deadbeef00) and the verdict follows."
    skill = _skill(_reply(body=body), workspace)

    with pytest.raises(SkillError, match="cannot cite artifacts that do not exist"):
        skill.execute(_request(_core()))


def test_execute_accepts_a_cited_run_artifact_the_report_does_not_cover(
    workspace: WorkspacePaths,
) -> None:
    """Evidence is quoted by the briefs, never carried whole — citing it is not a crime."""
    registry = ArtifactRegistry(workspace)
    evidence = make(ArtifactKind.EVIDENCE, run_id=RUN)
    registry.save(evidence)
    body = f"The pain is attested first-hand ({evidence.id}) and the verdict follows."
    skill = _skill(_reply(body=body), workspace)

    result = skill.execute(_request(_core()))

    (report,) = result.artifacts
    assert isinstance(report, Report)
    assert report.body == body
    # Cited, but not covered: provenance still names only what the report reports on.
    assert evidence.id not in {ref.id for ref in report.parents}


def test_execute_rejects_a_cited_artifact_from_another_run(workspace: WorkspacePaths) -> None:
    """Existence is not enough — the citation has to be this run's work."""
    registry = ArtifactRegistry(workspace)
    stranger = make(ArtifactKind.EVIDENCE, run_id="r2")
    registry.save(stranger)
    skill = _skill(_reply(body=f"Attested elsewhere ({stranger.id})."), workspace)

    with pytest.raises(SkillError, match="cannot cite artifacts that do not exist"):
        skill.execute(_request(_core()))


def test_execute_accepts_the_cluster_an_opportunity_answers(workspace: WorkspacePaths) -> None:
    """The regression: a `pc_…` reached the model on the opportunity and failed the run."""
    registry = ArtifactRegistry(workspace)
    cluster = make(ArtifactKind.PAIN_CLUSTER, run_id=RUN)
    registry.save(cluster)
    core = _core()
    body = f"The opportunity ({core['opportunity'].id}) answers a recurring pain ({cluster.id})."
    skill = _skill(_reply(body=body), workspace)

    result = skill.execute(_request(core))

    (report,) = result.artifacts
    assert isinstance(report, Report)
    assert cluster.id in {ref.id for ref in report.parents}


def test_execute_sets_the_period_from_the_covered_artifacts(
    workspace: WorkspacePaths,
) -> None:
    """The window is read off the artifacts' own timestamps, not the clock."""
    stamps = {
        "question": datetime(2026, 1, 1, tzinfo=UTC),
        "brief": datetime(2026, 1, 2, tzinfo=UTC),
        "opportunity": datetime(2026, 1, 3, tzinfo=UTC),
        "decision": datetime(2026, 1, 5, tzinfo=UTC),
    }
    question = make(ArtifactKind.QUESTION, run_id=RUN, created_at=stamps["question"])
    brief = make(ArtifactKind.RESEARCH_BRIEF, run_id=RUN, created_at=stamps["brief"])
    opportunity = make(ArtifactKind.OPPORTUNITY, run_id=RUN, created_at=stamps["opportunity"])
    decision = make(ArtifactKind.DECISION, run_id=RUN, created_at=stamps["decision"])
    request = SkillRequest(
        run_id=RUN,
        artifacts=[brief, opportunity, decision],
        question=question,
    )
    skill = _skill(_reply(), workspace)

    result = skill.execute(request)

    (report,) = result.artifacts
    assert isinstance(report, Report)
    assert report.period_start == stamps["question"]
    assert report.period_end == stamps["decision"]


def test_execute_parents_cover_everything_gathered(workspace: WorkspacePaths) -> None:
    """Provenance names the whole covered set, optional kinds included."""
    registry = ArtifactRegistry(workspace)
    market = make(ArtifactKind.MARKET_ANALYSIS, run_id=RUN)
    registry.save(market)
    core = _core()
    skill = _skill(_reply(), workspace)

    result = skill.execute(_request(core))

    (report,) = result.artifacts
    expected = {
        core["question"].id,
        core["brief"].id,
        core["opportunity"].id,
        core["decision"].id,
        market.id,
    }
    assert {ref.id for ref in report.parents} == expected


def test_execute_end_to_end_with_a_canned_body_citing_real_ids(
    workspace: WorkspacePaths,
) -> None:
    """The happy path: real ids cited, prose kept, computed fields stamped, persisted."""
    core = _core()
    body = (
        f"# Findings\n\nThe opportunity ({core['opportunity'].id}) rests on the "
        f"brief ({core['brief'].id}); the verdict is build ({core['decision'].id})."
    )
    skill = _skill(_reply(body=body), workspace)

    result = skill.execute(_request(core))

    (report,) = result.artifacts
    assert isinstance(report, Report)
    assert report.body == body
    assert report.title.startswith(core["question"].text)
    assert report.format is ReportFormat.MARKDOWN
    assert report.executive_summary is not None
    assert report.highlights
    stored = ArtifactRegistry(workspace).load(ArtifactKind.REPORT, report.id)
    assert stored.id == report.id


def test_compose_report_is_registered_but_not_a_pipeline_stage() -> None:
    assert "compose-report" in SKILLS
    assert "compose-report" not in STAGE_ORDER


# ----------------------------------------------------------------- op report


def _seed_run(registry: ArtifactRegistry) -> dict[str, Any]:
    core = _core()
    for key in ("question", "brief", "opportunity", "decision"):
        registry.save(core[key])
    return core


def test_report_without_inputs_exits_non_zero(workspace: WorkspacePaths) -> None:
    result = runner.invoke(app, ["report", "--run", "empty"])
    assert result.exit_code == 1
    assert "research-brief" in result.output


def test_report_refusal_names_only_what_is_missing(workspace: WorkspacePaths) -> None:
    registry = ArtifactRegistry(workspace)
    registry.save(make(ArtifactKind.RESEARCH_BRIEF, run_id=RUN))

    result = runner.invoke(app, ["report", "--run", RUN])

    assert result.exit_code == 1
    assert "research-brief" not in result.output
    assert "discover" in result.output
    assert "decision" in result.output


def test_report_json_is_pure(workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--json` emits the Report artifact object and nothing else on stdout."""
    registry = ArtifactRegistry(workspace)
    _seed_run(registry)
    fresh = make(ArtifactKind.REPORT, run_id=RUN, body="# canned")

    def fake_execute(self: ComposeReportSkill, request: SkillRequest) -> SkillResult:
        self.registry.save(fresh)
        return SkillResult(skill="compose-report", artifacts=[fresh])

    monkeypatch.setattr(ComposeReportSkill, "execute", fake_execute)
    result = runner.invoke(app, ["report", "--run", RUN, "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["id"] == fresh.id
    assert payload["kind"] == "report"


def test_report_out_writes_the_body(
    workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ArtifactRegistry(workspace)
    _seed_run(registry)
    fresh = make(ArtifactKind.REPORT, run_id=RUN, body="# The findings\n\nAll of them.")
    target = workspace.root / "out" / "report.md"

    def fake_execute(self: ComposeReportSkill, request: SkillRequest) -> SkillResult:
        self.registry.save(fresh)
        return SkillResult(skill="compose-report", artifacts=[fresh])

    monkeypatch.setattr(ComposeReportSkill, "execute", fake_execute)
    result = runner.invoke(app, ["report", "--run", RUN, "--out", str(target)])

    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == "# The findings\n\nAll of them."
    # The console wraps long paths, so assert on parts wrapping cannot split.
    assert "report written" in result.output
    assert target.name in result.output


def test_report_skips_when_a_report_exists(
    workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ArtifactRegistry(workspace)
    _seed_run(registry)
    registry.save(make(ArtifactKind.REPORT, run_id=RUN))

    def boom(self: ComposeReportSkill, request: SkillRequest) -> SkillResult:
        raise AssertionError("the skill must not run when a report already exists")

    monkeypatch.setattr(ComposeReportSkill, "execute", boom)
    result = runner.invoke(app, ["report", "--run", RUN])

    assert result.exit_code == 0
    assert "--force" in result.output


def test_report_force_supersedes_the_prior_report(
    workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ArtifactRegistry(workspace)
    _seed_run(registry)
    stale = make(ArtifactKind.REPORT, run_id=RUN)
    registry.save(stale)
    fresh = make(ArtifactKind.REPORT, run_id=RUN, body="# fresher findings")

    def fake_execute(self: ComposeReportSkill, request: SkillRequest) -> SkillResult:
        self.registry.save(fresh)
        return SkillResult(skill="compose-report", artifacts=[fresh])

    monkeypatch.setattr(ComposeReportSkill, "execute", fake_execute)
    result = runner.invoke(app, ["report", "--run", RUN, "--force"])

    assert result.exit_code == 0
    assert "fresher findings" in result.output
    assert registry.load(ArtifactKind.REPORT, stale.id).status is ArtifactStatus.SUPERSEDED
    assert registry.load(ArtifactKind.REPORT, fresh.id).status is not ArtifactStatus.SUPERSEDED
