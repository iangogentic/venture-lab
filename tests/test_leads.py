"""The LEAD layer: the Lead model, the harvest-leads skill, and `op leads`.

The guarantees under test are the layer's whole point: a lead names a person you
can find again (author + permalink + verbatim quote, all code-verified), every
identity field comes from the evidence rather than the model, and harvesting
stays outside the pipeline.
"""

import json
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from app.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactRegistry,
    ArtifactStatus,
    Evidence,
    Lead,
    LeadEngagement,
    LeadIntent,
    PainCluster,
    Signal,
)
from app.cli.main import app
from app.llm import LLM, GenerationRequest, GenerationResult, Provider, ProviderAdapter, TokenUsage
from app.pipeline import STAGE_ORDER
from app.skills import SKILLS, SkillRequest, SkillResult
from app.skills.harvest_leads import HarvestLeadsInput, HarvestLeadsSkill
from app.utils.errors import SkillError
from app.utils.paths import WorkspacePaths
from tests.factories import make

runner = CliRunner()

RUN = "r1"


# ------------------------------------------------------------------ the model


@pytest.mark.parametrize("field", ["author", "url", "quote", "collector"])
def test_lead_rejects_blank_identity(field: str) -> None:
    """No person, no permalink, no quote — no lead."""
    with pytest.raises(ValidationError):
        make(ArtifactKind.LEAD, **{field: "   "})


def test_lead_engagement_defaults_to_new() -> None:
    lead = make(ArtifactKind.LEAD)
    assert isinstance(lead, Lead)
    assert lead.engagement is LeadEngagement.NEW


def test_lead_enums_roundtrip() -> None:
    lead = make(ArtifactKind.LEAD, intent=LeadIntent.SEEKING, engagement=LeadEngagement.CONVERTED)
    revived = Lead.model_validate(lead.model_dump(mode="json"))
    assert revived.intent is LeadIntent.SEEKING
    assert revived.engagement is LeadEngagement.CONVERTED


def test_lead_refs_must_point_at_the_right_kinds() -> None:
    with pytest.raises(ValidationError):
        make(ArtifactKind.LEAD, cluster=ArtifactRef(kind=ArtifactKind.EVIDENCE, id="ev_1"))
    with pytest.raises(ValidationError):
        make(ArtifactKind.LEAD, evidence=ArtifactRef(kind=ArtifactKind.PAIN_CLUSTER, id="pc_1"))


def test_lead_blank_rationale_becomes_none() -> None:
    lead = make(ArtifactKind.LEAD, intent_rationale="   ", external_id="")
    assert isinstance(lead, Lead)
    assert lead.intent_rationale is None
    assert lead.external_id is None


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


def _skill(reply: dict[str, Any], workspace: WorkspacePaths) -> HarvestLeadsSkill:
    return HarvestLeadsSkill(
        llm=LLM(adapter=_EchoAdapter(reply)),
        registry=ArtifactRegistry(workspace),
    )


def _viable_evidence(**overrides: Any) -> Evidence:
    fields: dict[str, Any] = {
        "run_id": RUN,
        "author": "jane_doe",
        "source_url": "https://forum.example/p/1",
        "excerpt": "Our CI takes 40 minutes and the team hates it.",
        **overrides,
    }
    built = make(ArtifactKind.EVIDENCE, **fields)
    assert isinstance(built, Evidence)
    return built


# ------------------------------------------------------------------- gather


def test_gather_keeps_only_authored_and_linked_evidence(workspace: WorkspacePaths) -> None:
    """No person, no permalink → no candidate; those items remain evidence."""
    viable = _viable_evidence()
    no_author = make(ArtifactKind.EVIDENCE, run_id=RUN, source_url="https://forum.example/p/2")
    no_url = make(ArtifactKind.EVIDENCE, run_id=RUN, author="joe")
    cluster = make(
        ArtifactKind.PAIN_CLUSTER, run_id=RUN, parents=[viable.ref, no_author.ref, no_url.ref]
    )

    skill = _skill({"leads": []}, workspace)
    payload = skill.gather(SkillRequest(run_id=RUN, artifacts=[cluster, viable, no_author, no_url]))

    assert isinstance(payload, HarvestLeadsInput)
    (entry,) = payload.candidates
    assert entry["cluster_id"] == cluster.id
    assert [c["evidence_id"] for c in entry["candidates"]] == [viable.id]


def test_gather_yields_empty_candidates_for_a_cluster_citing_nothing(
    workspace: WorkspacePaths,
) -> None:
    uncited = _viable_evidence()
    cluster = make(ArtifactKind.PAIN_CLUSTER, run_id=RUN)

    skill = _skill({"leads": []}, workspace)
    payload = skill.gather(SkillRequest(run_id=RUN, artifacts=[cluster, uncited]))

    assert isinstance(payload, HarvestLeadsInput)
    (entry,) = payload.candidates
    assert entry["candidates"] == []


def test_gather_follows_citations_through_the_briefs(workspace: WorkspacePaths) -> None:
    """Clusters cite briefs; briefs cite evidence — the hop must be walked."""
    registry = ArtifactRegistry(workspace)
    evidence = _viable_evidence()
    brief = make(
        ArtifactKind.RESEARCH_BRIEF,
        run_id=RUN,
        signals=[Signal(statement="CI waits dominate", supported_by=[evidence.ref])],
    )
    registry.save(brief)
    cluster = make(ArtifactKind.PAIN_CLUSTER, run_id=RUN, parents=[brief.ref])

    skill = HarvestLeadsSkill(llm=LLM(adapter=_EchoAdapter({"leads": []})), registry=registry)
    payload = skill.gather(SkillRequest(run_id=RUN, artifacts=[cluster, evidence]))

    assert isinstance(payload, HarvestLeadsInput)
    (entry,) = payload.candidates
    assert [c["evidence_id"] for c in entry["candidates"]] == [evidence.id]


# -------------------------------------------------------- execute / assemble


def _selection(evidence: Evidence, cluster: PainCluster, **overrides: Any) -> dict[str, Any]:
    return {
        "evidence_id": evidence.id,
        "cluster_id": cluster.id,
        "quote": "Our CI takes 40 minutes and the team hates it.",
        "intent": "complaining",
        "intent_rationale": "States the cost unprompted.",
        "confidence": 0.8,
        **overrides,
    }


def test_execute_builds_the_lead_from_the_evidence_not_the_model(
    workspace: WorkspacePaths,
) -> None:
    """Identity comes from what was collected; the model only picked and read it."""
    evidence = _viable_evidence(
        collector="reddit",
        source_id="t3_abc",
        published_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    cluster = make(ArtifactKind.PAIN_CLUSTER, run_id=RUN, parents=[evidence.ref])
    assert isinstance(cluster, PainCluster)
    skill = _skill({"leads": [_selection(evidence, cluster)]}, workspace)

    result = skill.execute(SkillRequest(run_id=RUN, artifacts=[cluster, evidence]))

    (lead,) = result.artifacts
    assert isinstance(lead, Lead)
    assert lead.author == "jane_doe"
    assert lead.url == "https://forum.example/p/1"
    assert lead.collector == "reddit"
    assert lead.external_id == "t3_abc"
    assert lead.published_at == datetime(2026, 1, 2, tzinfo=UTC)
    assert lead.intent is LeadIntent.COMPLAINING
    assert lead.parents == [cluster.ref, evidence.ref]
    stored = ArtifactRegistry(workspace).load(ArtifactKind.LEAD, lead.id)
    assert stored.id == lead.id


def test_execute_accepts_a_reflowed_quote(workspace: WorkspacePaths) -> None:
    """Whitespace-normalised matching: reflowing is faithful, rewriting is not."""
    evidence = _viable_evidence(excerpt="Our CI takes 40 minutes\nand the team   hates it.")
    cluster = make(ArtifactKind.PAIN_CLUSTER, run_id=RUN, parents=[evidence.ref])
    assert isinstance(cluster, PainCluster)
    reply = {"leads": [_selection(evidence, cluster, quote="40 minutes and the team hates it.")]}
    skill = _skill(reply, workspace)

    result = skill.execute(SkillRequest(run_id=RUN, artifacts=[cluster, evidence]))

    (lead,) = result.artifacts
    assert isinstance(lead, Lead)
    assert lead.quote == "40 minutes and the team hates it."


def test_execute_rejects_a_fabricated_quote(workspace: WorkspacePaths) -> None:
    evidence = _viable_evidence()
    cluster = make(ArtifactKind.PAIN_CLUSTER, run_id=RUN, parents=[evidence.ref])
    assert isinstance(cluster, PainCluster)
    reply = {"leads": [_selection(evidence, cluster, quote="Everything about CI is broken.")]}
    skill = _skill(reply, workspace)

    with pytest.raises(SkillError, match="does not appear"):
        skill.execute(SkillRequest(run_id=RUN, artifacts=[cluster, evidence]))


def test_execute_rejects_an_unknown_evidence_id(workspace: WorkspacePaths) -> None:
    evidence = _viable_evidence()
    cluster = make(ArtifactKind.PAIN_CLUSTER, run_id=RUN, parents=[evidence.ref])
    assert isinstance(cluster, PainCluster)
    reply = {"leads": [_selection(evidence, cluster, evidence_id="ev_invented")]}
    skill = _skill(reply, workspace)

    with pytest.raises(SkillError, match="cannot introduce people"):
        skill.execute(SkillRequest(run_id=RUN, artifacts=[cluster, evidence]))


def test_execute_with_an_empty_selection_produces_nothing(workspace: WorkspacePaths) -> None:
    """A cluster can legitimately yield zero leads — that is a result, not a failure."""
    evidence = _viable_evidence()
    cluster = make(ArtifactKind.PAIN_CLUSTER, run_id=RUN, parents=[evidence.ref])
    skill = _skill({"leads": []}, workspace)

    result = skill.execute(SkillRequest(run_id=RUN, artifacts=[cluster, evidence]))

    assert result.artifacts == []


def test_harvest_leads_is_registered_but_not_a_pipeline_stage() -> None:
    assert "harvest-leads" in SKILLS
    assert "harvest-leads" not in STAGE_ORDER


# ---------------------------------------------------------------- op leads


def test_leads_list_json_is_pure(workspace: WorkspacePaths) -> None:
    """`--json` emits the artifact list and nothing else on stdout."""
    registry = ArtifactRegistry(workspace)
    lead = make(ArtifactKind.LEAD)
    registry.save(lead)

    result = runner.invoke(app, ["leads", "list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["id"] == lead.id
    assert payload[0]["kind"] == "lead"


def test_leads_list_filters_by_engagement(workspace: WorkspacePaths) -> None:
    registry = ArtifactRegistry(workspace)
    registry.save(make(ArtifactKind.LEAD))
    kept = make(ArtifactKind.LEAD, engagement=LeadEngagement.ENGAGED)
    registry.save(kept)

    result = runner.invoke(app, ["leads", "list", "--engagement", "engaged", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [entry["id"] for entry in payload] == [kept.id]


def test_leads_mark_updates_engagement(workspace: WorkspacePaths) -> None:
    registry = ArtifactRegistry(workspace)
    lead = make(ArtifactKind.LEAD)
    registry.save(lead)

    result = runner.invoke(app, ["leads", "mark", lead.id, "engaged"])

    assert result.exit_code == 0
    assert "new" in result.output and "engaged" in result.output
    stored = registry.load_as(ArtifactKind.LEAD, lead.id, Lead)
    assert stored.engagement is LeadEngagement.ENGAGED


def test_leads_mark_unknown_id_exits_non_zero(workspace: WorkspacePaths) -> None:
    result = runner.invoke(app, ["leads", "mark", "ld_nope", "reviewed"])
    assert result.exit_code == 1


def test_harvest_without_clusters_exits_non_zero(workspace: WorkspacePaths) -> None:
    result = runner.invoke(app, ["leads", "harvest", "--run", "empty"])
    assert result.exit_code == 1
    assert "cluster" in result.output.lower()


def test_harvest_skips_when_leads_exist(
    workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ArtifactRegistry(workspace)
    registry.save(make(ArtifactKind.PAIN_CLUSTER, run_id=RUN))
    registry.save(_viable_evidence())
    registry.save(make(ArtifactKind.LEAD, run_id=RUN))

    def boom(self: HarvestLeadsSkill, request: SkillRequest) -> SkillResult:
        raise AssertionError("the skill must not run when leads already exist")

    monkeypatch.setattr(HarvestLeadsSkill, "execute", boom)
    result = runner.invoke(app, ["leads", "harvest", "--run", RUN])

    assert result.exit_code == 0
    assert "--force" in result.output


def test_harvest_force_supersedes_the_prior_leads(
    workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ArtifactRegistry(workspace)
    registry.save(make(ArtifactKind.PAIN_CLUSTER, run_id=RUN))
    registry.save(_viable_evidence())
    stale = make(ArtifactKind.LEAD, run_id=RUN)
    registry.save(stale)
    fresh = make(ArtifactKind.LEAD, run_id=RUN)

    def fake_execute(self: HarvestLeadsSkill, request: SkillRequest) -> SkillResult:
        self.registry.save(fresh)
        return SkillResult(skill="harvest-leads", artifacts=[fresh])

    monkeypatch.setattr(HarvestLeadsSkill, "execute", fake_execute)
    result = runner.invoke(app, ["leads", "harvest", "--run", RUN, "--force"])

    assert result.exit_code == 0
    assert registry.load(ArtifactKind.LEAD, stale.id).status is ArtifactStatus.SUPERSEDED
    assert registry.load(ArtifactKind.LEAD, fresh.id).status is not ArtifactStatus.SUPERSEDED
