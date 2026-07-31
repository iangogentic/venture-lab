"""Domain fields and validators, per artifact model.

`test_artifacts.py` covers the shared envelope. This covers what each model adds,
starting with a test that pins the owner's per-skill output spec directly.
"""

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactRef,
    ClaimType,
    CompetitionAnalysis,
    ContradictionAnalysis,
    CounterEvidence,
    CounterEvidenceKind,
    CounterSeverity,
    Decision,
    Evidence,
    EvidenceDensity,
    EvidenceLevel,
    InterviewPlan,
    MarketAnalysis,
    Opportunity,
    PainCluster,
    PainSeverity,
    PricingEstimate,
    Question,
    QuestionPriority,
    ResearchBrief,
    SizeEstimate,
    Verdict,
)
from app.artifacts.research_brief import Contradiction, Quote, Signal
from tests.factories import make

RUN = "run_test"


# ------------------------------------------------------------- the owner's spec

SPEC_FIELDS: dict[type[Artifact], set[str]] = {
    ResearchBrief: {
        "summary",
        "signals",
        "quotes",
        "contradictions",
        "unknowns",
        "evidence_quality",
        "evidence_density",
    },
    Opportunity: {"workflow", "icp", "buyer", "problem", "why_now", "missing_evidence"},
    MarketAnalysis: {"buyer", "pricing", "sam", "som", "budget_owner", "assumptions", "unknowns"},
    CompetitionAnalysis: {
        "competitors",
        "switching_costs",
        "moats",
        "differentiation",
    },
    Decision: {
        "verdict",
        "decision_confidence",
        "top_reasons",
        "next_validation_step",
        "biggest_unknown",
    },
    InterviewPlan: {
        "targets",
        "questions",
        "success_criteria",
        "failure_criteria",
        "questions_to_avoid",
    },
}


@pytest.mark.parametrize(
    ("model", "expected"),
    list(SPEC_FIELDS.items()),
    ids=[m.__name__ for m in SPEC_FIELDS],
)
def test_model_carries_its_specified_outputs(model: type[Artifact], expected: set[str]) -> None:
    """Each stage's declared outputs must exist as fields on the artifact it produces."""
    missing = expected - set(model.model_fields)
    assert not missing, f"{model.__name__} is missing specified fields: {sorted(missing)}"


def test_market_analysis_has_no_tam() -> None:
    """ "Prefer SAM and SOM over TAM" — TAM is the number that gets invented."""
    assert "tam" not in MarketAnalysis.model_fields


def test_contradiction_analysis_returns_evidence_only() -> None:
    """ "Return only evidence" — no verdict, score, or recommendation may live here."""
    forbidden = {"verdict", "score", "recommendation", "conclusion", "summary"}
    assert not (forbidden & set(ContradictionAnalysis.model_fields))


def test_opportunity_does_not_analyse_competitors() -> None:
    """Competitors are a later stage; leaking them here would skip the evidence step."""
    fields = set(Opportunity.model_fields)
    assert not any("competit" in name for name in fields)


# ------------------------------------------------------- required text fields


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (lambda v: Question(id="q_1", run_id=RUN, text=v), "Question.text"),
        (
            lambda v: Evidence(id="ev_1", run_id=RUN, collector="rss", excerpt=v),
            "Evidence.excerpt",
        ),
        (lambda v: ResearchBrief(id="rb_1", run_id=RUN, summary=v), "ResearchBrief.summary"),
        (lambda v: PainCluster(id="pc_1", run_id=RUN, label=v), "PainCluster.label"),
    ],
)
@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_required_text_rejects_blank(
    build: Callable[[str], Artifact], field: str, blank: str
) -> None:
    with pytest.raises(ValidationError):
        build(blank)


def test_required_text_is_stripped() -> None:
    assert Question(id="q_1", run_id=RUN, text="  Why is CI slow?  ").text == "Why is CI slow?"


def test_blank_optional_text_becomes_none() -> None:
    question = Question(id="q_1", run_id=RUN, text="t", rationale="   ", scope="")
    assert question.rationale is None
    assert question.scope is None


def test_tags_are_stripped_lowercased_and_deduplicated() -> None:
    question = Question(id="q_1", run_id=RUN, text="t", tags=[" CI ", "ci", "Build", "build"])
    assert question.tags == ["ci", "build"]


def test_question_priority_defaults_to_medium() -> None:
    assert Question(id="q_1", run_id=RUN, text="t").priority is QuestionPriority.MEDIUM


# ------------------------------------------------------------ Evidence


def test_naive_timestamps_are_read_as_utc() -> None:
    evidence = Evidence(
        id="ev_1",
        run_id=RUN,
        collector="rss",
        excerpt="x",
        published_at=datetime(2026, 1, 1, 12, 0),
    )
    assert evidence.published_at == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_source_url_must_be_a_url() -> None:
    with pytest.raises(ValidationError):
        Evidence(id="ev_1", run_id=RUN, collector="rss", excerpt="x", source_url="not a url")


# ------------------------------------------------------- ResearchBrief


def test_signals_are_tagged_observation_inference_or_hypothesis() -> None:
    """The pipeline's central discipline, enforced at the field level."""
    signal = Signal(statement="Builds dominate complaints", claim_type=ClaimType.OBSERVATION)
    assert signal.claim_type is ClaimType.OBSERVATION
    assert Signal(statement="s").claim_type is ClaimType.OBSERVATION


def test_quotes_stay_attributable() -> None:
    ref = ArtifactRef(kind=ArtifactKind.EVIDENCE, id="ev_1")
    quote = Quote(text="CI takes 40 minutes", source=ref)
    assert quote.source == ref


def test_contradiction_needs_two_distinct_positions() -> None:
    with pytest.raises(ValidationError):
        Contradiction(topic="t", positions=["only one"])
    assert len(Contradiction(topic="t", positions=["fast", "slow"]).positions) == 2


def test_evidence_density_is_ordinal() -> None:
    ranks = [density.rank for density in EvidenceDensity]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


def test_brief_defaults_are_honest_about_thin_evidence() -> None:
    """An unassessed brief must not look well-evidenced."""
    brief = ResearchBrief(id="rb_1", run_id=RUN, summary="s")
    assert brief.evidence_quality is EvidenceLevel.NONE
    assert brief.evidence_density is EvidenceDensity.SPARSE
    assert brief.source_count == 0


# ------------------------------------------------------------ Opportunity


def test_opportunity_requires_the_full_inference() -> None:
    """Every one of the specified inferences is required — none may be skipped."""
    for omitted in ("workflow", "icp", "buyer", "problem", "why_now"):
        fields = {
            "id": "op_1",
            "run_id": RUN,
            "title": "t",
            "workflow": "w",
            "icp": "i",
            "buyer": "b",
            "problem": "p",
            "why_now": "n",
        }
        del fields[omitted]
        with pytest.raises(ValidationError):
            Opportunity.model_validate(fields)


def test_opportunity_is_an_inference_by_default() -> None:
    assert make(ArtifactKind.OPPORTUNITY).claim_type is ClaimType.INFERENCE  # type: ignore[attr-defined]


# ---------------------------------------------------------- MarketAnalysis


def test_a_size_estimate_cannot_carry_an_unexplained_number() -> None:
    """ "Never fabricate market numbers" — a figure must state where it came from."""
    with pytest.raises(ValidationError):
        SizeEstimate(amount=1_000_000, currency="USD", basis="  ")


def test_a_size_estimate_may_decline_to_size() -> None:
    """ "When evidence is insufficient, say so" must be expressible, not worked around."""
    estimate = SizeEstimate(basis="No pricing data for this segment could be found.")
    assert estimate.amount is None
    assert not estimate.is_quantified


def test_an_amount_must_state_its_currency() -> None:
    with pytest.raises(ValidationError):
        SizeEstimate(amount=250_000, basis="Bottom-up from seat counts.")


def test_pricing_requires_a_basis() -> None:
    with pytest.raises(ValidationError):
        PricingEstimate(model="per-seat", basis="")


def test_market_analysis_is_unsized_until_a_number_is_justified() -> None:
    analysis = make(ArtifactKind.MARKET_ANALYSIS)
    assert isinstance(analysis, MarketAnalysis)
    assert not analysis.is_sized


# ------------------------------------------------------ ContradictionAnalysis


@pytest.mark.parametrize("kind", list(CounterEvidenceKind))
def test_every_counter_evidence_category_is_expressible(kind: CounterEvidenceKind) -> None:
    """The five things the spec says to search for must each have a home."""
    item = CounterEvidence(kind=kind, observation="something discouraging")
    assert item.kind is kind


def test_counter_severity_is_ordinal() -> None:
    ranks = [severity.rank for severity in CounterSeverity]
    assert ranks == sorted(ranks)


def test_blocking_counter_evidence_is_surfaced() -> None:
    analysis = ContradictionAnalysis(
        id="cx_1",
        run_id=RUN,
        opportunity=ArtifactRef(kind=ArtifactKind.OPPORTUNITY, id="op_1"),
        counter_evidence=[
            CounterEvidence(
                kind=CounterEvidenceKind.FAILED_STARTUP,
                observation="Three companies tried this and folded.",
                severity=CounterSeverity.BLOCKING,
            )
        ],
    )
    assert analysis.is_blocking
    assert analysis.by_kind(CounterEvidenceKind.FAILED_STARTUP)


def test_finding_nothing_is_a_recordable_result() -> None:
    """An empty search is only meaningful if what was searched for is recorded."""
    analysis = ContradictionAnalysis(
        id="cx_1",
        run_id=RUN,
        opportunity=ArtifactRef(kind=ArtifactKind.OPPORTUNITY, id="op_1"),
        searched_for=["failed startups", "negative reviews"],
    )
    assert analysis.counter_evidence == []
    assert not analysis.is_blocking
    assert len(analysis.searched_for) == 2


def test_contradiction_must_argue_against_an_opportunity() -> None:
    with pytest.raises(ValidationError):
        ContradictionAnalysis(
            id="cx_1",
            run_id=RUN,
            opportunity=ArtifactRef(kind=ArtifactKind.EVIDENCE, id="ev_1"),
        )


# ---------------------------------------------------------------- Decision


def test_verdict_is_build_reject_or_wait() -> None:
    assert {v.value for v in Verdict} == {"build", "reject", "wait"}


@pytest.mark.parametrize("verdict", list(Verdict))
def test_every_verdict_still_demands_an_unknown_and_a_next_step(verdict: Verdict) -> None:
    """A confident build that names no unknown has stopped reasoning."""
    decision = make(ArtifactKind.DECISION, verdict=verdict)
    assert isinstance(decision, Decision)
    assert decision.biggest_unknown
    assert decision.next_validation_step


def test_a_decision_needs_at_least_one_reason() -> None:
    with pytest.raises(ValidationError):
        make(ArtifactKind.DECISION, top_reasons=[])


def test_decision_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        make(ArtifactKind.DECISION, decision_confidence=1.5)


def test_decision_must_rule_on_an_opportunity() -> None:
    with pytest.raises(ValidationError):
        make(
            ArtifactKind.DECISION,
            opportunity=ArtifactRef(kind=ArtifactKind.EVIDENCE, id="ev_1"),
        )


# ------------------------------------------------------------ InterviewPlan


def test_interview_plan_separates_success_from_failure() -> None:
    """Failure criteria stated up front are what stop a result being reinterpreted."""
    plan = make(
        ArtifactKind.INTERVIEW_PLAN,
        success_criteria=["Four of five describe the wait unprompted."],
        failure_criteria=["Nobody can name a time it cost them anything."],
        questions_to_avoid=["Would you use a tool that fixed this?"],
    )
    assert isinstance(plan, InterviewPlan)
    assert plan.success_criteria and plan.failure_criteria and plan.questions_to_avoid


def test_interview_plan_must_test_a_decision() -> None:
    with pytest.raises(ValidationError):
        make(
            ArtifactKind.INTERVIEW_PLAN,
            decision=ArtifactRef(kind=ArtifactKind.OPPORTUNITY, id="op_1"),
        )


# ---------------------------------------------------------- PainCluster


def test_pain_severity_is_ordinal() -> None:
    ranks = [severity.rank for severity in PainSeverity]
    assert ranks == sorted(ranks)


def test_severity_starts_unassessed() -> None:
    assert PainCluster(id="pc_1", run_id=RUN, label="Slow CI").severity is None


def test_quotes_keep_duplicates_but_drop_blanks() -> None:
    cluster = PainCluster(
        id="pc_1", run_id=RUN, label="Slow CI", quotes=["CI is slow", "CI is slow", "   "]
    )
    assert cluster.quotes == ["CI is slow", "CI is slow"]
