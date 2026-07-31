"""Metrics over a finished run: is it any good, and did two runs stay stable?

These are the two questions a prompt change actually raises. The first — `score_run`
— reads one run and reports whether each stage produced anything, whether the volumes
sit inside the benchmark's bands, and above all whether the conclusions can be walked
back to an observation. The second — `compare_runs` — puts two runs side by side and
reports how much of the substance survived, so a change can be told apart from an
improvement.

Three commitments shape everything below.

**Traceability is the headline.** An opportunity that cannot be walked back to a piece
of evidence is this pipeline's central failure mode: it means a plausible story was
assembled out of nothing, which is exactly what every artifact model is built to make
awkward. `ArtifactGraph.evidence_for` answers it directly, and the answer is reported
for opportunities and for decisions, the two places where a groundless claim does the
most damage.

**An honest refusal is never a defect.** `confidence=None` means unassessed, not zero.
A `SizeEstimate` with no amount but a stated `basis` is a correct output — the market
could not be sized and the analysis said so. An empty `ContradictionAnalysis` with a
non-empty `searched_for` means the adversarial pass ran and found nothing. Scoring any
of these as a miss would train the pipeline out of the honesty it exists to produce,
so each is counted in its own right and none is subtracted from anything.

**Nothing here is a gate.** Every number is a reading. A stage outside its band, a
theme unmatched, a verdict that flipped: each is a prompt to go and look, not a
failure. Which is why no function raises on a thin run — an empty workspace scores
zero everywhere and reports it plainly.

Pure throughout: reads a registry, returns models. No printing, no writing, no network.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import (
    Artifact,
    ArtifactGraph,
    ArtifactKind,
    ArtifactRef,
    ArtifactRegistry,
    ContradictionAnalysis,
    Decision,
    Evidence,
    MarketAnalysis,
    Opportunity,
    PainCluster,
    ResearchBrief,
    Verdict,
)
from app.benchmarks.spec import Benchmark

SCORED_KINDS: Final[tuple[ArtifactKind, ...]] = (
    ArtifactKind.EVIDENCE,
    ArtifactKind.RESEARCH_BRIEF,
    ArtifactKind.PAIN_CLUSTER,
    ArtifactKind.OPPORTUNITY,
    ArtifactKind.MARKET_ANALYSIS,
    ArtifactKind.COMPETITION_ANALYSIS,
    ArtifactKind.CONTRADICTION_ANALYSIS,
    ArtifactKind.DECISION,
    ArtifactKind.INTERVIEW_PLAN,
)
"""One kind per pipeline stage, in `STAGE_ORDER`.

`QUESTION` is the run's input and `REPORT` only re-renders what is already scored, so
neither is a stage output. Spelled out rather than derived from `app.pipeline`, which
would drag the skill registry and the LLM client into a module that just reads files.
"""

OVERLAP_THRESHOLD: Final[float] = 0.5
"""Jaccard similarity at or above which two labels are treated as the same thing.

Labels are short — two to six content tokens once stopwords are dropped — so half the
union means a pair must share the majority of its distinct vocabulary. Measured on
real rewordings, that lands where it should:

* "slow CI feedback loops" / "slow feedback loops in CI" -> 1.0, matched. Word order
  and filler differ, the finding does not.
* "flaky test triage" / "triaging flaky tests" -> 0.5, matched, and exactly on the
  line. Which is why the comparison is inclusive: this is the weakest pair that is
  still obviously the same pain, so it has to be admitted.
* "CI cost" / "CI flakiness" -> 0.33, not matched. Same topic, different problem;
  counting them as one would report a pain as surviving a re-run when it did not.

A knob, and a forgiving one to set: overlap is reported as a fraction rather than as a
verdict, so a threshold slightly off moves a number instead of flipping a result.
"""

STOPWORDS: Final[frozenset[str]] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "do",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "too",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)
"""English function words only.

Domain words are deliberately left in. Dropping "tool" or "team" would make labels
from different segments collide, and the whole point of the overlap metric is to
notice when a re-run wandered into a different segment.
"""

_PUNCTUATION: Final[re.Pattern[str]] = re.compile(r"[^\w\s]+")


# --------------------------------------------------------------------- text


def normalise_text(text: str) -> str:
    """Casefold, replace punctuation with space, collapse whitespace.

    The one normalisation used on both sides of every comparison in this module, so
    a hyphen, a stray quote or a capital cannot decide whether two runs agree.
    """
    return " ".join(_PUNCTUATION.sub(" ", text.casefold()).split())


def content_tokens(text: str) -> frozenset[str]:
    """The distinct meaningful words of `text`.

    Stopwords are dropped unless that would empty the set — a label made only of
    function words is rare, but silently reducing it to nothing would make it match
    every other such label.

    A trailing plural is then folded away ("queues" -> "queue"). Crude, and knowingly
    so: number is the difference that most often stops two runs' labels for the same
    finding from overlapping, and no two genuinely different pains are told apart by a
    final "s". Folding happens after stopwords are removed, so "this" is never mangled
    into a token that no longer looks like a stopword.
    """
    words = normalise_text(text).split()
    content = frozenset(_singular(word) for word in words if word not in STOPWORDS)
    return content or frozenset(_singular(word) for word in words)


def _singular(word: str) -> str:
    """Drop a trailing plural "s". Short words and "ss" endings are left alone."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Intersection over union of two token sets.

    Two empty sets score 0.0, not 1.0: an empty label says nothing about a run, and
    treating "nothing" as agreeing with "nothing" would report stability that was
    never demonstrated.
    """
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def similarity(left: str, right: str) -> float:
    """How alike two labels are, on the token-set measure `OVERLAP_THRESHOLD` grades."""
    return jaccard(content_tokens(left), content_tokens(right))


# ------------------------------------------------------------------- models


class StageScore(BaseModel):
    """How much one stage produced, against what the benchmark expected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ArtifactKind
    count: int = Field(ge=0)
    expected_min: int | None = None
    expected_max: int | None = None
    within_expectation: bool = Field(
        description="Whether `count` sits inside the declared band. True when no band "
        "was declared — an unstated expectation is not an unmet one.",
    )

    @property
    def produced(self) -> bool:
        """Whether the stage produced anything at all."""
        return self.count > 0


class GroundingScore(BaseModel):
    """Whether the run's conclusions rest on anything.

    Read the traceability counts first. Everything else is context for them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_total: int = Field(default=0, ge=0)
    evidence_with_source: int = Field(
        default=0,
        ge=0,
        description="Evidence carrying a `source_url` or `source_id` — a reader can go "
        "and check it. Evidence with neither cannot be verified by anyone.",
    )
    grounded_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    briefs_total: int = Field(default=0, ge=0)
    briefs_citing_evidence: int = Field(
        default=0,
        ge=0,
        description="Briefs whose own signals or quotes point at an Evidence artifact "
        "present in this run. `parents` is excluded on purpose: the engine writes "
        "parents, so counting them would measure the engine, not the brief.",
    )

    opportunities_total: int = Field(default=0, ge=0)
    opportunities_traceable_to_evidence: int = Field(
        default=0,
        ge=0,
        description="Opportunities with at least one Evidence artifact upstream of them "
        "in the lineage graph. The headline metric.",
    )

    decisions_total: int = Field(default=0, ge=0)
    decisions_traceable_to_evidence: int = Field(
        default=0,
        ge=0,
        description="Verdicts that reach an observation. A decision with none is a "
        "judgement made about nothing.",
    )

    unassessed_confidence: int = Field(
        default=0,
        ge=0,
        description="Artifacts whose `confidence` is None. NOT a defect: None means the "
        "stage declined to score itself, which is honest. Reported so a change in how "
        "freely the model assigns confidence is visible.",
    )

    markets_total: int = Field(default=0, ge=0)
    markets_sized: int = Field(default=0, ge=0, description="SAM or SOM carries a number.")
    markets_unsized_with_basis: int = Field(
        default=0,
        ge=0,
        description="An estimate exists, states no amount, and explains why. A correct "
        "output — the refusal to invent a number this project asks for.",
    )
    markets_without_estimate: int = Field(
        default=0,
        ge=0,
        description="Neither SAM nor SOM present at all. Distinct from a refusal: "
        "nothing was said either way, so nothing was refused.",
    )

    contradiction_analyses_total: int = Field(default=0, ge=0)
    contradiction_searches_recorded: int = Field(
        default=0,
        ge=0,
        description="Analyses stating what they searched for. Without it, an empty "
        "result cannot be told apart from never having looked.",
    )
    contradiction_searches_empty_result: int = Field(
        default=0,
        ge=0,
        description="Searched, recorded the search, and found nothing. A real result.",
    )

    @property
    def traceable_fraction(self) -> float:
        """Share of opportunities that reach evidence. 0.0 when there are none."""
        return _fraction(self.opportunities_traceable_to_evidence, self.opportunities_total)

    @property
    def decision_traceable_fraction(self) -> float:
        """Share of decisions that reach evidence. 0.0 when there are none."""
        return _fraction(self.decisions_traceable_to_evidence, self.decisions_total)

    @property
    def citation_fraction(self) -> float:
        """Share of briefs citing evidence in their own body. 0.0 when there are none."""
        return _fraction(self.briefs_citing_evidence, self.briefs_total)

    @property
    def contradiction_searches_unrecorded(self) -> int:
        """Analyses that did not say what they looked for."""
        return self.contradiction_analyses_total - self.contradiction_searches_recorded


class ThemeScore(BaseModel):
    """Which of the benchmark's themes the run actually talked about.

    A coarse proxy, and known to be one: matching is substring-on-normalised-text over
    cluster labels and descriptions and opportunity titles and problems. It catches a
    run that wandered off the question entirely. It cannot tell a well-phrased finding
    from a badly-phrased one, and a run that says the right thing in other words scores
    as a miss. Read `missed` as "go and look", never as "wrong".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected: list[str] = Field(default_factory=list)
    matched: list[str] = Field(default_factory=list)
    missed: list[str] = Field(default_factory=list)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)


class RunScore(BaseModel):
    """Everything one run can be said about on its own."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    benchmark: str | None = Field(
        default=None,
        description="The benchmark this was scored against, if any. None means volumes "
        "and themes were not asserted — grounding still was.",
    )
    stages: list[StageScore] = Field(default_factory=list)
    grounding: GroundingScore
    themes: ThemeScore | None = Field(
        default=None,
        description="None when no benchmark was given, or when it declared no themes. "
        "Either way nothing was asked, so nothing is scored.",
    )
    verdicts: dict[str, int] = Field(
        default_factory=dict,
        description="One entry per verdict, always all of them, so two runs diff without "
        "an absent key having to be read as a zero.",
    )
    complete: bool = Field(description="Every stage produced at least one artifact.")

    @property
    def artifact_total(self) -> int:
        """How many artifacts the run produced across every scored stage."""
        return sum(stage.count for stage in self.stages)

    @property
    def within_expectation(self) -> bool:
        """Whether every stage sits inside its band. Vacuously true with no benchmark."""
        return all(stage.within_expectation for stage in self.stages)

    @property
    def empty_stages(self) -> list[ArtifactKind]:
        """Stages that produced nothing — where to start reading a bad run."""
        return [stage.kind for stage in self.stages if not stage.produced]


class VerdictPair(BaseModel):
    """One opportunity matched across two runs, with the verdict each run reached."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left_title: str
    right_title: str
    left_verdict: Verdict
    right_verdict: Verdict

    @property
    def changed(self) -> bool:
        """Whether the two runs disagreed about this opportunity."""
        return self.left_verdict != self.right_verdict


class RunComparison(BaseModel):
    """How much of one run survived into another.

    Stability is not the goal — a prompt change is *meant* to change something. This
    says how much changed and what, so the change can be looked at deliberately
    instead of discovered later.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    left: str
    right: str
    cluster_overlap: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fuzzy Jaccard over pain-cluster labels: matched pairs over the "
        "union of both runs' clusters.",
    )
    opportunity_overlap: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The same measure over opportunity titles.",
    )
    new_opportunities: list[str] = Field(
        default_factory=list,
        description="Titles in the right run that nothing in the left run matched.",
    )
    dropped_opportunities: list[str] = Field(
        default_factory=list,
        description="Titles in the left run that nothing in the right run matched.",
    )
    shared_verdicts: list[VerdictPair] = Field(
        default_factory=list,
        description="Matched opportunities both runs ruled on the same way.",
    )
    changed_verdicts: list[VerdictPair] = Field(
        default_factory=list,
        description="Matched opportunities the two runs ruled on differently. The most "
        "consequential thing a prompt change can do, so it is listed, not counted.",
    )
    verdict_agreement: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Share of matched, decided opportunities the runs agreed on. None "
        "when no opportunity could be matched and decided in both runs — nothing was "
        "compared, which is different from having compared and disagreed on everything.",
    )

    @property
    def matched_verdicts(self) -> int:
        """How many opportunities were comparable on verdict at all."""
        return len(self.shared_verdicts) + len(self.changed_verdicts)


# --------------------------------------------------------------- public API


def score_run(
    registry: ArtifactRegistry,
    run_id: str,
    benchmark: Benchmark | None = None,
) -> RunScore:
    """Read one run and report how healthy it looks.

    Args:
        registry: Where the artifacts live.
        run_id: The run to score. Artifacts from other runs are never read into it.
        benchmark: Optional expectations. Without one, volumes are still counted and
            grounding is still measured; only the bands and themes go unasserted.

    A run with nothing in it scores zero throughout and does not raise: "the pipeline
    produced nothing" is a result the caller needs to be able to print.
    """
    artifacts = {kind: registry.find_by_type(kind, run_id=run_id) for kind in SCORED_KINDS}
    graph = ArtifactGraph.build(registry, run_id=run_id)

    stages = [_stage_score(kind, artifacts[kind], benchmark) for kind in SCORED_KINDS]
    clusters = _of_kind(artifacts[ArtifactKind.PAIN_CLUSTER], PainCluster)
    opportunities = _of_kind(artifacts[ArtifactKind.OPPORTUNITY], Opportunity)

    return RunScore(
        run_id=run_id,
        benchmark=benchmark.name if benchmark is not None else None,
        stages=stages,
        grounding=_grounding(artifacts, graph),
        themes=_theme_score(benchmark, clusters, opportunities),
        verdicts=_verdict_tally(_of_kind(artifacts[ArtifactKind.DECISION], Decision)),
        complete=all(stage.produced for stage in stages),
    )


def compare_runs(registry: ArtifactRegistry, left: str, right: str) -> RunComparison:
    """Put two runs side by side and report what survived.

    Clusters and opportunities are matched fuzzily — see `OVERLAP_THRESHOLD` — because
    a re-run never reproduces a label verbatim, and demanding that it did would report
    every run as a total rewrite of the last one.
    """
    left_clusters = _cluster_labels(registry, left)
    right_clusters = _cluster_labels(registry, right)
    left_opportunities = _opportunity_titles(registry, left)
    right_opportunities = _opportunity_titles(registry, right)

    opportunity_pairs = _match_pairs(left_opportunities, right_opportunities)
    matched_left = {pair[0] for pair in opportunity_pairs}
    matched_right = {pair[1] for pair in opportunity_pairs}

    shared, changed = _verdict_pairs(
        opportunity_pairs,
        _verdicts_by_title(registry, left),
        _verdicts_by_title(registry, right),
    )
    compared = len(shared) + len(changed)

    return RunComparison(
        left=left,
        right=right,
        cluster_overlap=_overlap(left_clusters, right_clusters),
        opportunity_overlap=_overlap(left_opportunities, right_opportunities),
        new_opportunities=[title for title in right_opportunities if title not in matched_right],
        dropped_opportunities=[title for title in left_opportunities if title not in matched_left],
        shared_verdicts=shared,
        changed_verdicts=changed,
        verdict_agreement=len(shared) / compared if compared else None,
    )


# ---------------------------------------------------------------- internals


def _fraction(part: int, whole: int) -> float:
    """Divide, treating an empty denominator as 0.0.

    Every ratio in this module runs through here. A run with no evidence, no
    opportunities and no decisions is exactly the run someone most needs a score for,
    so none of them may raise.
    """
    if whole <= 0:
        return 0.0
    return part / whole


def _of_kind[ArtifactT: Artifact](
    artifacts: Iterable[Artifact],
    model: type[ArtifactT],
) -> list[ArtifactT]:
    """Narrow a registry result to one concrete model."""
    return [artifact for artifact in artifacts if isinstance(artifact, model)]


def _stage_score(
    kind: ArtifactKind,
    artifacts: Sequence[Artifact],
    benchmark: Benchmark | None,
) -> StageScore:
    """Count one stage's output and place it against the benchmark's band, if any."""
    expectation = benchmark.expectation_for(kind) if benchmark is not None else None
    count = len(artifacts)
    return StageScore(
        kind=kind,
        count=count,
        expected_min=expectation.min if expectation is not None else None,
        expected_max=expectation.max if expectation is not None else None,
        within_expectation=expectation.accepts(count) if expectation is not None else True,
    )


def _grounding(
    artifacts: Mapping[ArtifactKind, list[Artifact]],
    graph: ArtifactGraph,
) -> GroundingScore:
    """Measure what the run's conclusions rest on, reading the graph once for lineage."""
    evidence = _of_kind(artifacts[ArtifactKind.EVIDENCE], Evidence)
    briefs = _of_kind(artifacts[ArtifactKind.RESEARCH_BRIEF], ResearchBrief)
    opportunities = _of_kind(artifacts[ArtifactKind.OPPORTUNITY], Opportunity)
    decisions = _of_kind(artifacts[ArtifactKind.DECISION], Decision)
    markets = _of_kind(artifacts[ArtifactKind.MARKET_ANALYSIS], MarketAnalysis)
    contradictions = _of_kind(
        artifacts[ArtifactKind.CONTRADICTION_ANALYSIS],
        ContradictionAnalysis,
    )

    with_source = sum(1 for item in evidence if item.source_url or item.source_id)
    sized, unsized_with_basis, without_estimate = _market_counts(markets)
    traceable_opportunities = sum(1 for item in opportunities if graph.evidence_for(item.id))
    traceable_decisions = sum(1 for item in decisions if graph.evidence_for(item.id))
    unassessed = sum(
        1 for kind in SCORED_KINDS for item in artifacts[kind] if item.confidence is None
    )

    return GroundingScore(
        evidence_total=len(evidence),
        evidence_with_source=with_source,
        grounded_fraction=_fraction(with_source, len(evidence)),
        briefs_total=len(briefs),
        briefs_citing_evidence=sum(1 for brief in briefs if _cites_evidence(brief, graph)),
        opportunities_total=len(opportunities),
        opportunities_traceable_to_evidence=traceable_opportunities,
        decisions_total=len(decisions),
        decisions_traceable_to_evidence=traceable_decisions,
        unassessed_confidence=unassessed,
        markets_total=len(markets),
        markets_sized=sized,
        markets_unsized_with_basis=unsized_with_basis,
        markets_without_estimate=without_estimate,
        contradiction_analyses_total=len(contradictions),
        contradiction_searches_recorded=sum(1 for item in contradictions if item.searched_for),
        contradiction_searches_empty_result=sum(
            1 for item in contradictions if item.searched_for and not item.counter_evidence
        ),
    )


def _market_counts(markets: Sequence[MarketAnalysis]) -> tuple[int, int, int]:
    """Split market analyses into sized, honestly unsized, and silent.

    The middle case is the one worth protecting: `SizeEstimate` cannot carry an amount
    without a basis, so an estimate with no amount is a stated reason the market could
    not be sized. That is the output this project asks for and must never be counted
    against a run. Silence — no SAM and no SOM at all — is the one to look at.
    """
    sized = 0
    unsized_with_basis = 0
    without_estimate = 0
    for market in markets:
        if market.sam is None and market.som is None:
            without_estimate += 1
        elif market.is_sized:
            sized += 1
        else:
            unsized_with_basis += 1
    return sized, unsized_with_basis, without_estimate


def _cites_evidence(brief: ResearchBrief, graph: ArtifactGraph) -> bool:
    """Whether a brief's own body points at evidence that exists in this run."""
    refs: list[ArtifactRef] = [ref for signal in brief.signals for ref in signal.supported_by]
    refs.extend(quote.source for quote in brief.quotes if quote.source is not None)
    return any(_resolves_to_evidence(ref, graph) for ref in refs)


def _resolves_to_evidence(ref: ArtifactRef, graph: ArtifactGraph) -> bool:
    """A citation counts only if it lands on evidence actually present in the run.

    A ref to an id nothing on disk answers to is not a citation; it is the shape of one.
    """
    node = graph.node(ref.id)
    return node is not None and node.kind is ArtifactKind.EVIDENCE


def _theme_score(
    benchmark: Benchmark | None,
    clusters: Sequence[PainCluster],
    opportunities: Sequence[Opportunity],
) -> ThemeScore | None:
    """Look for each theme in the run's clusters and opportunities. See `ThemeScore`.

    None when nothing was asked: no benchmark, or a benchmark that declared no themes.
    """
    if benchmark is None or not benchmark.themes:
        return None

    haystack = [
        normalise_text(text)
        for text in (
            *(cluster.label for cluster in clusters),
            *(cluster.description or "" for cluster in clusters),
            *(item.title for item in opportunities),
            *(item.problem for item in opportunities),
        )
        if text
    ]

    matched: list[str] = []
    missed: list[str] = []
    for theme in benchmark.themes:
        needle = normalise_text(theme)
        found = bool(needle) and any(needle in text for text in haystack)
        (matched if found else missed).append(theme)

    return ThemeScore(
        expected=list(benchmark.themes),
        matched=matched,
        missed=missed,
        coverage=_fraction(len(matched), len(benchmark.themes)),
    )


def _newest_decisions(decisions: Iterable[Decision]) -> dict[str, Decision]:
    """One decision per opportunity, keeping the newest revision.

    Re-running the decision stage can leave more than one decision pointing at the same
    opportunity. Tallying both would report a verdict the run has since moved on from.
    """
    newest: dict[str, Decision] = {}
    for decision in decisions:
        current = newest.get(decision.opportunity.id)
        if current is None or (decision.version, decision.updated_at) > (
            current.version,
            current.updated_at,
        ):
            newest[decision.opportunity.id] = decision
    return newest


def _verdict_tally(decisions: Sequence[Decision]) -> dict[str, int]:
    """How many opportunities got each verdict, every verdict always present."""
    tally = {verdict.value: 0 for verdict in Verdict}
    for decision in _newest_decisions(decisions).values():
        tally[decision.verdict.value] += 1
    return tally


def _cluster_labels(registry: ArtifactRegistry, run_id: str) -> list[str]:
    """Every pain-cluster label in a run, sorted so matching is deterministic."""
    found = registry.find_by_type(ArtifactKind.PAIN_CLUSTER, run_id=run_id)
    return sorted(cluster.label for cluster in _of_kind(found, PainCluster))


def _opportunity_titles(registry: ArtifactRegistry, run_id: str) -> list[str]:
    """Every opportunity title in a run, sorted so matching is deterministic."""
    found = registry.find_by_type(ArtifactKind.OPPORTUNITY, run_id=run_id)
    return sorted(item.title for item in _of_kind(found, Opportunity))


def _verdicts_by_title(registry: ArtifactRegistry, run_id: str) -> dict[str, Verdict]:
    """Opportunity title -> the verdict reached on it, for the opportunities decided."""
    found = registry.find_by_type(ArtifactKind.OPPORTUNITY, run_id=run_id)
    ruled = registry.find_by_type(ArtifactKind.DECISION, run_id=run_id)
    decisions = _newest_decisions(_of_kind(ruled, Decision))
    return {
        item.title: decisions[item.id].verdict
        for item in _of_kind(found, Opportunity)
        if item.id in decisions
    }


def _match_pairs(left: Sequence[str], right: Sequence[str]) -> list[tuple[str, str]]:
    """Pair labels across two runs, greedily and one-to-one.

    Each left label takes its most similar unclaimed right label, if that similarity
    reaches `OVERLAP_THRESHOLD`. Greedy rather than an optimal assignment: with a dozen
    labels per run the two agree almost always, and an optimal matcher would add real
    machinery to move a number that is already reported as approximate. Inputs are
    sorted by the callers, so the result is at least deterministic.
    """
    unclaimed = list(right)
    pairs: list[tuple[str, str]] = []

    for label in left:
        best: str | None = None
        best_score = 0.0
        for candidate in unclaimed:
            score = similarity(label, candidate)
            if score > best_score:
                best, best_score = candidate, score
        if best is not None and best_score >= OVERLAP_THRESHOLD:
            unclaimed.remove(best)
            pairs.append((label, best))
    return pairs


def _overlap(left: Sequence[str], right: Sequence[str]) -> float:
    """Fuzzy Jaccard over two label lists: matched pairs over the union of both.

    Two empty runs score 0.0. Nothing matching nothing is not agreement, and reporting
    1.0 there would make the emptiest possible comparison look like the most stable one.
    """
    matched = len(_match_pairs(left, right))
    union = len(left) + len(right) - matched
    return _fraction(matched, union)


def _verdict_pairs(
    pairs: Sequence[tuple[str, str]],
    left_verdicts: Mapping[str, Verdict],
    right_verdicts: Mapping[str, Verdict],
) -> tuple[list[VerdictPair], list[VerdictPair]]:
    """Split matched opportunities into verdicts that held and verdicts that moved.

    An opportunity only one run decided on is in neither list: it was never compared,
    and counting it as a disagreement would blame the decision stage for something the
    discovery stage did.
    """
    shared: list[VerdictPair] = []
    changed: list[VerdictPair] = []

    for left_title, right_title in pairs:
        left_verdict = left_verdicts.get(left_title)
        right_verdict = right_verdicts.get(right_title)
        if left_verdict is None or right_verdict is None:
            continue
        pair = VerdictPair(
            left_title=left_title,
            right_title=right_title,
            left_verdict=left_verdict,
            right_verdict=right_verdict,
        )
        (changed if pair.changed else shared).append(pair)

    return shared, changed


__all__ = [
    "OVERLAP_THRESHOLD",
    "SCORED_KINDS",
    "STOPWORDS",
    "GroundingScore",
    "RunComparison",
    "RunScore",
    "StageScore",
    "ThemeScore",
    "VerdictPair",
    "compare_runs",
    "content_tokens",
    "jaccard",
    "normalise_text",
    "score_run",
    "similarity",
]
