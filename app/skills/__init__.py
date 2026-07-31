"""Skills — the nine cognitive steps of the pipeline, plus two run on demand.

Each skill reads artifacts, asks a model one question about them via a Markdown
prompt, validates the reply against its `output_schema`, and writes the resulting
artifacts. Skills never call each other; the pipeline engine composes them.
`harvest-leads` and `compose-report` are the deliberate exceptions to the
pipeline itself: registered like the rest, but driven by `op leads harvest` and
`op report` rather than a stage.

Importing this package registers every skill, which is what lets the engine look
one up by stage name.
"""

from app.skills.analyze_competition import AnalyzeCompetitionSkill
from app.skills.analyze_market import AnalyzeMarketSkill
from app.skills.base import (
    SKILLS,
    Batching,
    Skill,
    SkillInput,
    SkillOutput,
    SkillRequest,
    SkillResult,
    available,
    get_skill,
    register,
)
from app.skills.cluster_pains import ClusterPainSkill
from app.skills.collect_evidence import CollectEvidenceSkill
from app.skills.compose_report import ComposeReportSkill
from app.skills.contradiction_analysis import ContradictionSkill
from app.skills.decision import DecisionSkill
from app.skills.discover_opportunities import DiscoverOpportunitySkill
from app.skills.harvest_leads import HarvestLeadsSkill
from app.skills.interview_plan import InterviewPlanSkill
from app.skills.research_brief import ResearchBriefSkill

__all__ = [
    "SKILLS",
    "AnalyzeCompetitionSkill",
    "AnalyzeMarketSkill",
    "Batching",
    "ClusterPainSkill",
    "CollectEvidenceSkill",
    "ComposeReportSkill",
    "ContradictionSkill",
    "DecisionSkill",
    "DiscoverOpportunitySkill",
    "HarvestLeadsSkill",
    "InterviewPlanSkill",
    "ResearchBriefSkill",
    "Skill",
    "SkillInput",
    "SkillOutput",
    "SkillRequest",
    "SkillResult",
    "available",
    "get_skill",
    "register",
]
