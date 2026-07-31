"""Minimal valid artifacts, one per kind.

Each model has different required domain fields, so tests that care about the
*envelope* rather than the domain build through here and stay readable.
"""

from typing import Any

from app.artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactRef,
    CompetitionAnalysis,
    ContradictionAnalysis,
    Decision,
    Evidence,
    InterviewPlan,
    Lead,
    LeadIntent,
    MarketAnalysis,
    Opportunity,
    PainCluster,
    Question,
    Report,
    ResearchBrief,
    Verdict,
)

RUN_ID = "run_test"

_OPPORTUNITY_REF = ArtifactRef(kind=ArtifactKind.OPPORTUNITY, id="op_ruled_on")
_DECISION_REF = ArtifactRef(kind=ArtifactKind.DECISION, id="dec_ruled")
_CLUSTER_REF = ArtifactRef(kind=ArtifactKind.PAIN_CLUSTER, id="pc_recurring")
_EVIDENCE_REF = ArtifactRef(kind=ArtifactKind.EVIDENCE, id="ev_original")

_REQUIRED: dict[ArtifactKind, dict[str, Any]] = {
    ArtifactKind.QUESTION: {"text": "Where do platform teams lose time?"},
    ArtifactKind.EVIDENCE: {"collector": "rss", "excerpt": "Our CI takes 40 minutes."},
    ArtifactKind.RESEARCH_BRIEF: {"summary": "Build times dominate the complaints."},
    ArtifactKind.PAIN_CLUSTER: {"label": "Slow CI"},
    ArtifactKind.OPPORTUNITY: {
        "title": "Remote build cache",
        "workflow": "Running CI on every pull request.",
        "icp": "Platform teams at 50-500 engineer companies.",
        "buyer": "VP of Engineering.",
        "problem": "Every merge waits on a full rebuild.",
        "why_now": "Remote cache infrastructure became cheap to self-host.",
    },
    ArtifactKind.MARKET_ANALYSIS: {
        "opportunity": _OPPORTUNITY_REF,
        "buyer": "VP of Engineering.",
        "budget_owner": "Engineering platform budget.",
    },
    ArtifactKind.COMPETITION_ANALYSIS: {"opportunity": _OPPORTUNITY_REF},
    ArtifactKind.CONTRADICTION_ANALYSIS: {"opportunity": _OPPORTUNITY_REF},
    ArtifactKind.INTERVIEW_PLAN: {
        "decision": _DECISION_REF,
        "objective": "Test whether teams would pay to remove build waits.",
    },
    ArtifactKind.DECISION: {
        "opportunity": _OPPORTUNITY_REF,
        "verdict": Verdict.BUILD,
        "decision_confidence": 0.6,
        "top_reasons": ["Cheapest fix with the widest reach."],
        "next_validation_step": "Interview five platform leads about build waits.",
        "biggest_unknown": "Whether teams will pay rather than self-host.",
    },
    ArtifactKind.LEAD: {
        "cluster": _CLUSTER_REF,
        "evidence": _EVIDENCE_REF,
        "collector": "reddit",
        "author": "buildwatcher",
        "url": "https://reddit.com/r/devops/comments/abc123",
        "quote": "Our CI takes 40 minutes.",
        "intent": LeadIntent.COMPLAINING,
    },
    ArtifactKind.REPORT: {"title": "Weekly roll-up"},
}

_MODELS: dict[ArtifactKind, type[Artifact]] = {
    ArtifactKind.QUESTION: Question,
    ArtifactKind.EVIDENCE: Evidence,
    ArtifactKind.RESEARCH_BRIEF: ResearchBrief,
    ArtifactKind.PAIN_CLUSTER: PainCluster,
    ArtifactKind.OPPORTUNITY: Opportunity,
    ArtifactKind.MARKET_ANALYSIS: MarketAnalysis,
    ArtifactKind.COMPETITION_ANALYSIS: CompetitionAnalysis,
    ArtifactKind.CONTRADICTION_ANALYSIS: ContradictionAnalysis,
    ArtifactKind.DECISION: Decision,
    ArtifactKind.INTERVIEW_PLAN: InterviewPlan,
    ArtifactKind.LEAD: Lead,
    ArtifactKind.REPORT: Report,
}


def make(kind: ArtifactKind, **overrides: Any) -> Artifact:
    """Build a minimal valid artifact of `kind`, with a freshly generated id."""
    model = _MODELS[kind]
    fields: dict[str, Any] = {
        "id": model.make_id(),
        "run_id": RUN_ID,
        **_REQUIRED[kind],
        **overrides,
    }
    return model.model_validate(fields)


def every_kind() -> list[Artifact]:
    """One minimal artifact of every kind."""
    return [make(kind) for kind in ArtifactKind]


__all__ = ["RUN_ID", "every_kind", "make"]
