"""Artifact models and the JSON registry that persists them under `workspace/`.

Two layers:

* the models — one Pydantic class per `ArtifactKind`, all sharing the envelope in
  `base.py` (`version`, `id`, timestamps, `status`, `confidence`, `evidence_level`);
* `ArtifactRegistry` — save, load, update, version, and search, backed by files.

The chain the models form, question first::

    Question -> Evidence -> ResearchBrief -> PainCluster -> Opportunity -> Decision -> Report

Each link is recorded in the child's `parents`, so any artifact can be walked back
to the evidence it rests on.
"""

from app.artifacts.base import (
    SCHEMA_VERSION,
    Artifact,
    ArtifactKind,
    ArtifactRef,
    ArtifactStatus,
    ClaimType,
    Confidence,
    EvidenceLevel,
)
from app.artifacts.competition_analysis import CompetitionAnalysis, Competitor
from app.artifacts.contradiction_analysis import (
    ContradictionAnalysis,
    CounterEvidence,
    CounterEvidenceKind,
    CounterSeverity,
)
from app.artifacts.decision import Decision, Verdict
from app.artifacts.evidence import Evidence, EvidenceKind
from app.artifacts.graph import ArtifactGraph, GraphNode
from app.artifacts.interview_plan import InterviewPlan, InterviewQuestion, InterviewTarget
from app.artifacts.lead import Lead, LeadEngagement, LeadIntent
from app.artifacts.market_analysis import MarketAnalysis, PricingEstimate, SizeEstimate
from app.artifacts.opportunity import Opportunity
from app.artifacts.pain_cluster import PainCluster, PainSeverity
from app.artifacts.question import Question, QuestionPriority
from app.artifacts.registry import (
    HISTORY_DIRNAME,
    MODELS,
    ArtifactRegistry,
    kind_for_id,
    model_for,
)
from app.artifacts.report import Report, ReportFormat
from app.artifacts.research_brief import (
    Contradiction,
    EvidenceDensity,
    Quote,
    ResearchBrief,
    Signal,
)

__all__ = [
    "HISTORY_DIRNAME",
    "MODELS",
    "SCHEMA_VERSION",
    "Artifact",
    "ArtifactGraph",
    "ArtifactKind",
    "ArtifactRef",
    "ArtifactRegistry",
    "ArtifactStatus",
    "ClaimType",
    "CompetitionAnalysis",
    "Competitor",
    "Confidence",
    "Contradiction",
    "ContradictionAnalysis",
    "CounterEvidence",
    "CounterEvidenceKind",
    "CounterSeverity",
    "Decision",
    "Evidence",
    "EvidenceDensity",
    "EvidenceKind",
    "EvidenceLevel",
    "GraphNode",
    "InterviewPlan",
    "InterviewQuestion",
    "InterviewTarget",
    "Lead",
    "LeadEngagement",
    "LeadIntent",
    "MarketAnalysis",
    "Opportunity",
    "PainCluster",
    "PainSeverity",
    "PricingEstimate",
    "Question",
    "QuestionPriority",
    "Quote",
    "Report",
    "ReportFormat",
    "ResearchBrief",
    "Signal",
    "SizeEstimate",
    "Verdict",
    "kind_for_id",
    "model_for",
]
