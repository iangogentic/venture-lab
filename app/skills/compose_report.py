"""`compose-report`: render a run's findings as one report a human can act on.

Deliberately **not** a stage in the pipeline's `STAGE_ORDER`, for three reasons:

* A report is a derived view. Every claim in it must already exist in the
  artifacts it covers, so it is safe to regenerate at any time and adds no new
  knowledge a later stage could depend on.
* It is rendered when a human wants one — `op report`, before a review or a
  conversation — not as a side effect of every research run. A run that nobody
  reads a report of is still a finished run.
* Nothing consumes a Report. Making it a stage would teach the resume machinery
  a dependency that exists only to satisfy the machinery.

The split of labour mirrors `collect-evidence`'s select-don't-author discipline,
applied to narrative: the model writes the prose — summary, highlights, the
per-opportunity story — but every fact of record is computed by code. The title
comes from the question, the verdict tallies from the actual Decision artifacts
(anything the model volunteers is overwritten), the covered period from the
artifacts' own timestamps, and `parents` from everything gathered.

Citations are verified against the *run*, not against this skill's own inputs.
The distinction matters because the two sets are legitimately different: a brief
quotes evidence by id, and the report carries the brief rather than the hundreds
of evidence artifacts underneath it. An id naming an artifact the run produced is
a citation a reader can follow, whether or not this report covers the artifact in
full; only an id naming nothing — a fabrication, or another run's work — fails
the whole reply.
"""

import re
from typing import Any, ClassVar, Final

from pydantic import Field

from app.artifacts import (
    MODELS,
    Artifact,
    ArtifactKind,
    ArtifactStatus,
    Decision,
    Question,
    Report,
    ReportFormat,
    Verdict,
)
from app.skills.base import Batching, Skill, SkillInput, SkillOutput, SkillRequest, register
from app.utils.errors import SkillError
from app.utils.logging import get_logger
from app.utils.time import utcnow

logger = get_logger(__name__)

_ID_PREFIXES: Final[tuple[str, ...]] = tuple(
    sorted((model.id_prefix for model in MODELS.values()), key=len, reverse=True)
)
"""Every known artifact id prefix, longest first so the regex alternation is safe.

Derived from the registry's model table rather than restated, so a new artifact
kind is citable here without anyone remembering to update a list.
"""

_CITED_ID: Final[re.Pattern[str]] = re.compile(
    r"\b(?:" + "|".join(_ID_PREFIXES) + r")_[a-z0-9_]+\b"
)
"""What an inline citation looks like: a known prefix, an underscore, an id body."""

# Redeclared rather than imported from the pipeline engine: skills must not import
# app.pipeline (the engine imports skills), and the two-line rule is cheaper than
# the cycle. Mirrors `PipelineEngine.consumable_of`'s gate.
_NOT_CONSUMABLE: Final[frozenset[ArtifactStatus]] = frozenset(
    {ArtifactStatus.SUPERSEDED, ArtifactStatus.REJECTED, ArtifactStatus.ARCHIVED}
)

_OPTIONAL_KINDS: Final[tuple[ArtifactKind, ...]] = (
    ArtifactKind.PAIN_CLUSTER,
    ArtifactKind.MARKET_ANALYSIS,
    ArtifactKind.COMPETITION_ANALYSIS,
    ArtifactKind.CONTRADICTION_ANALYSIS,
    ArtifactKind.INTERVIEW_PLAN,
    ArtifactKind.LEAD,
)
"""Kinds folded in when the run has them, pulled from the registry inside
`gather` rather than declared in `consumes` — a run that stopped after the
decision stage, or harvested no leads, still deserves a report, and a declared
input would make the missing kind a refusal instead of an absence."""


class ComposeReportInput(SkillInput):
    """Placeholder values for `compose-report`'s prompt."""

    question: dict[str, Any] = Field(description="The Question seeding the run, serialised.")
    briefs: list[dict[str, Any]] = Field(description="Every research brief for the run.")
    clusters: list[dict[str, Any]] = Field(
        description="The pain clusters the opportunities answer. May be empty.",
    )
    opportunities: list[dict[str, Any]] = Field(description="Every opportunity for the run.")
    decisions: list[dict[str, Any]] = Field(description="Every decision for the run.")
    market: list[dict[str, Any]] = Field(
        description="Market analyses, when the run has them. May be empty.",
    )
    competition: list[dict[str, Any]] = Field(
        description="Competition analyses, when the run has them. May be empty.",
    )
    contradictions: list[dict[str, Any]] = Field(
        description="Contradiction analyses, when the run has them. May be empty.",
    )
    interviews: list[dict[str, Any]] = Field(
        description="Interview plans, when the run has them. May be empty.",
    )
    leads: list[dict[str, Any]] = Field(
        description="Harvested leads, when the run has them. May be empty.",
    )


class ComposeReportOutput(SkillOutput):
    """The prose half of the report — everything else is computed by code."""

    executive_summary: str = Field(
        description="The reader-who-stops-here summary: the findings without the argument.",
    )
    highlights: list[str] = Field(
        default_factory=list,
        description="The few lines worth reading if nothing else is.",
    )
    body: str = Field(
        description="The full Markdown report: an evidence-first narrative per "
        "opportunity, citing artifact ids inline.",
    )
    verdict_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Discarded. The application tallies verdicts from the Decision "
        "artifacts itself; this field only exists so a reply that volunteers one "
        "is not rejected on shape.",
    )


@register
class ComposeReportSkill(Skill):
    """Weave a run's artifacts into one report, with every claim traceable by id."""

    name: ClassVar[str] = "compose-report"
    description: ClassVar[str] = "Render a run's findings as one evidence-first report."
    prompt_name: ClassVar[str] = "compose-report"
    consumes: ClassVar[tuple[ArtifactKind, ...]] = (
        ArtifactKind.RESEARCH_BRIEF,
        ArtifactKind.OPPORTUNITY,
        ArtifactKind.DECISION,
    )
    # One call over everything: a report's value is the through-line across
    # opportunities, and splitting per opportunity would lose exactly that.
    batching: ClassVar[Batching] = Batching.FAN_IN
    produces: ClassVar[ArtifactKind] = ArtifactKind.REPORT
    input_schema: ClassVar[type[SkillInput]] = ComposeReportInput
    output_schema: ClassVar[type[SkillOutput]] = ComposeReportOutput

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._covered: dict[str, Artifact] = {}
        """Everything this report covers, keyed by id — the source of the computed
        fields (period, parents, verdict tallies), and the first thing a citation is
        checked against."""

        self._resolved: dict[str, bool] = {}
        """Citations already looked up on disk, so a repeated id costs one read."""

    def gather(self, request: SkillRequest) -> ComposeReportInput:
        """Collect the required kinds from the request and the optional ones from disk.

        The required three arrive through the request, gated by the caller the way
        every stage's inputs are. The optional kinds are looked up here so their
        absence never blocks: an empty list reaches the prompt as an honest
        "this run does not have these", not as a refusal.
        """
        question = request.question
        if not isinstance(question, Question):
            raise SkillError(f"{self.name} needs the run's question to title the report")

        briefs = request.of_kind(ArtifactKind.RESEARCH_BRIEF)
        opportunities = request.of_kind(ArtifactKind.OPPORTUNITY)
        decisions = request.of_kind(ArtifactKind.DECISION)
        optional = {kind: self._consumable(kind, request.run_id) for kind in _OPTIONAL_KINDS}

        gathered: list[Artifact] = [question, *briefs, *opportunities, *decisions]
        for kind in _OPTIONAL_KINDS:
            gathered.extend(optional[kind])
        self._covered = {artifact.id: artifact for artifact in gathered}
        self._resolved = {}

        return ComposeReportInput(
            question=question.model_dump(mode="json"),
            briefs=[artifact.model_dump(mode="json") for artifact in briefs],
            clusters=self._dumped(optional[ArtifactKind.PAIN_CLUSTER]),
            opportunities=[artifact.model_dump(mode="json") for artifact in opportunities],
            decisions=[artifact.model_dump(mode="json") for artifact in decisions],
            market=self._dumped(optional[ArtifactKind.MARKET_ANALYSIS]),
            competition=self._dumped(optional[ArtifactKind.COMPETITION_ANALYSIS]),
            contradictions=self._dumped(optional[ArtifactKind.CONTRADICTION_ANALYSIS]),
            interviews=self._dumped(optional[ArtifactKind.INTERVIEW_PLAN]),
            leads=self._dumped(optional[ArtifactKind.LEAD]),
        )

    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        """Wrap the model's prose in the facts of record, all computed here.

        The model supplies the summary, highlights and body; title, tallies,
        period and parents come from the covered artifacts, so they cannot drift
        from what is actually on disk. Any tally the model returned is discarded.
        """
        if not isinstance(output, ComposeReportOutput):
            raise SkillError(
                f"{self.name} expected {ComposeReportOutput.__name__}, got {type(output).__name__}"
            )
        question = request.question
        if not isinstance(question, Question):
            # gather() refused this request; reaching here is a bug, not a model crime.
            raise SkillError(f"{self.name} lost the run's question between gather and assemble")

        body = output.body.strip()
        if not body:
            raise SkillError(f"{self.name} returned an empty report body")
        for text in (body, output.executive_summary, *output.highlights):
            self._verify_citations(text, request.run_id)

        covered = list(self._covered.values())
        counts: dict[Verdict, int] = {}
        for decision in covered:
            if isinstance(decision, Decision):
                counts[decision.verdict] = counts.get(decision.verdict, 0) + 1

        return [
            Report(
                id=Report.make_id(),
                run_id=request.run_id,
                # Question plus date, because a report is a snapshot: the same
                # question re-reported next month is a different document.
                title=f"{question.text} — {utcnow():%Y-%m-%d}",
                executive_summary=output.executive_summary,
                highlights=output.highlights,
                body=body,
                format=ReportFormat.MARKDOWN,
                period_start=min(artifact.created_at for artifact in covered),
                period_end=max(artifact.created_at for artifact in covered),
                # Computed, never copied: `output.verdict_counts` is discarded so a
                # model cannot report a tally the decisions on disk do not support.
                verdict_counts=counts,
                parents=[artifact.ref for artifact in covered],
            )
        ]

    # ---------------------------------------------------------------- internals

    def _consumable(self, kind: ArtifactKind, run_id: str) -> list[Artifact]:
        """A run's current artifacts of one kind — the same gate every stage reads through."""
        return [
            artifact
            for artifact in self.registry.find_by_type(kind, run_id=run_id)
            if artifact.status not in _NOT_CONSUMABLE
        ]

    @staticmethod
    def _dumped(artifacts: list[Artifact]) -> list[dict[str, Any]]:
        """Serialise a gathered list for the prompt."""
        return [artifact.model_dump(mode="json") for artifact in artifacts]

    def _verify_citations(self, text: str, run_id: str) -> None:
        """Refuse any citation that names an artifact this run did not produce.

        The report is the artifact a human is most likely to read alone, so its
        citations are the one place a fabricated id would go unnoticed — every
        other stage's refs are checked by validators or by code.

        The check is existence, not membership. Narrating an opportunity means
        naming the evidence under its brief and the cluster it answers, and those
        ids reach the model inside the artifacts it was given rather than as
        artifacts of their own; refusing them would fail a run for citing exactly
        what the prompt asks for. Scoping to the run is what keeps the guard real:
        an id that resolves to nothing, or to another run's work, still fails.
        """
        for token in dict.fromkeys(_CITED_ID.findall(text)):
            if token in self._covered or self._produced_by_run(token, run_id):
                continue
            raise SkillError(
                f"{self.name} cited {token!r}, which is not an artifact of run "
                f"{run_id!r} — the model cannot cite artifacts that do not exist."
            )

    def _produced_by_run(self, artifact_id: str, run_id: str) -> bool:
        """Whether a cited id names an artifact this run actually wrote."""
        cached = self._resolved.get(artifact_id)
        if cached is not None:
            return cached
        found = self.registry.find_by_id(artifact_id)
        verdict = found is not None and found.run_id == run_id
        self._resolved[artifact_id] = verdict
        return verdict


__all__ = [
    "ComposeReportInput",
    "ComposeReportOutput",
    "ComposeReportSkill",
]
