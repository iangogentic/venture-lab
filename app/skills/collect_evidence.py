"""`collect-evidence`: fetch from real sources, then keep what is relevant.

The shape of this stage is the whole point. It does **not** ask a model what it
knows about the question — that would make the pipeline

    Question -> model recall -> Evidence

and every downstream guarantee about provenance would then be decorating a
fiction. Instead collectors fetch real material, and the model's only job is to
choose which fetched items bear on the question and which passage carries the point.

That split has a property worth stating plainly: **the model cannot invent a
quote.** Every excerpt is checked against the text the collector actually
retrieved, and a selection whose excerpt is not in that text is rejected. The
model selects; it does not author.

Three outcomes, kept distinct:

* no collector configured or reachable -> `SkillError`. Nothing ran, so a run must
  not continue as though the sources had been checked and found wanting.
* collectors ran and found nothing -> zero Evidence. A real answer: these sources
  have nothing to say about this question.
* collectors found material the model judged irrelevant -> also zero Evidence, and
  the prompt is told that keeping few is correct.
"""

import re
from collections import Counter
from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import Artifact, ArtifactKind, Evidence, EvidenceKind, EvidenceLevel
from app.collectors import (
    Collector,
    CollectorConfig,
    SourceItem,
    available,
    config_from_settings,
    get_collector,
)
from app.config import get_settings
from app.memory import Embedder, MemoryStore, cosine, default_embedder
from app.skills.base import Skill, SkillInput, SkillOutput, SkillRequest, register
from app.utils.errors import (
    CollectorError,
    MemoryUnavailableError,
    RateLimitedError,
    SkillError,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

_WHITESPACE = re.compile(r"\s+")
_CHARS_PER_TOKEN = 4
_ITEM_OVERHEAD_CHARS = 300
"""Roughly what an item's title, url, author and JSON punctuation add."""

_FINGERPRINT_CHARS: Final[int] = 1000
"""How much of an item is embedded when comparing candidates.

The opening of a post says what the complaint is; the rest is elaboration and
replies. A thousand characters fingerprints the complaint while keeping a batch
of candidates cheap to encode — and cross-posts, the duplicates this exists to
catch, share their opening verbatim.
"""


def _build_embedder() -> Embedder:
    """The embedder semantic dedup and indexing use. A seam, deliberately.

    Module-level so tests can monkeypatch it with a deterministic fake — the
    real model is a download away, and a test suite must never fetch it.

    Raises:
        MemoryUnavailableError: When no real embedder can be had (propagated
            from the model load on first encode).
    """
    return default_embedder()


_QUESTION_WORDS: Final[frozenset[str]] = frozenset(
    {
        "where",
        "what",
        "why",
        "how",
        "who",
        "when",
        "which",
        "do",
        "does",
        "did",
        "is",
        "are",
        "can",
        "should",
        "would",
        "and",
        "or",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "have",
        "has",
        "they",
        "their",
        "it",
        "its",
        "most",
        "go",
        "goes",
        "get",
        "gets",
        "that",
        "this",
        "from",
        "into",
        "about",
        "before",
        "after",
        "during",
        "every",
        "each",
        "all",
        "any",
        "other",
        "than",
        "then",
        "while",
        "across",
        "between",
        "but",
        "so",
        "as",
        "be",
        "was",
        "were",
        "will",
        "more",
        "we",
        "our",
        "you",
        "your",
    }
)
_NOISE_WORDS: Final[frozenset[str]] = frozenset(
    {
        "teams",
        "team",
        "people",
        "lose",
        "losing",
        "spend",
        "spending",
        "time",
        "tried",
        "money",
        "engineers",
        "developers",
    }
)
"""Words that describe the *asking*, not the subject. Left in, they retrieve any
document about teams losing anything — measured, and it is as bad as it sounds."""

_INSTRUCTION_WORDS: Final[frozenset[str]] = frozenset(
    {
        "research",
        "find",
        "identify",
        "list",
        "investigate",
        "explore",
        "determine",
        "discover",
        "survey",
        "compile",
        "enumerate",
        "show",
        "please",
        "top",
        "biggest",
        "common",
        "evidence",
        "including",
        "include",
    }
)
"""Words that address *me*, not the subject.

A question is not always phrased as one. "Research the top 25 manual…" opens
with four words about the task rather than the topic, and a derivation that
takes the first content words it sees turns that into the query `research top
25 manual` — which is what one real run searched every source for, eight ways,
before keeping none of the four hundred candidates it dragged back. The
imperative framing has to go before the subject can be found underneath it.

Deliberately conservative. `review`, `best`, `key`, `data` and `source` are all
plausibly meta and all excluded, because each is also the subject of a question
somebody will really ask — "where do teams lose time in code review" must keep
working, and dropping the topic is a far worse failure than keeping a filler
word."""

_SOURCING_CLAUSE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:including|such as|e\.g\.|for example|based on|drawing on|sourced from|"
    r"with evidence from|citing)\b",
    re.IGNORECASE,
)
"""Where a question stops describing the subject and starts naming places to look.

"…before every production release, including evidence from GitHub issues,
Reddit, Hacker News…" — everything after the comma is routing, addressed to a
collector list that already exists. Searching for it retrieves discussion *of
those sites*, and it crowds the real subject out of a bounded query."""

_SUBJECT_WORDS: Final[int] = 4
"""How many content words make a subject. Four is about the longest phrase a
keyword search still matches usefully; past that, recall falls off a cliff."""

_QUERY_SHAPES: Final[tuple[str, ...]] = (
    "{subject}",
    "{subject} slow",
    "{subject} frustrating",
    "{subject} wastes time",
    "{subject} alternatives",
    "switching from {subject}",
    "{subject} workaround",
    "{subject} not worth it",
)
"""Broad first, then complaint-shaped, then persona-shaped.

The first four are the original derivation: the broad term finds the topic and
almost no complaints, the suffixed ones find complaints and less topic —
measured separately. The rest borrow STORM's perspective-guided questioning:
different people phrase the same pain differently, and paraphrasing one voice
retrieves only that voice. A churned user searches "alternatives" and
"switching from"; an operator who cannot leave writes "workaround"; a skeptic
weighing the purchase writes "not worth it". None of these phrasings contains
a complaint word, so the complaint-shaped queries never find them."""


def derive_queries(question: str, *, limit: int = 6) -> list[str]:
    """Turn a research question into keyword queries a search engine can use.

    The default keeps six of the eight shapes: every query runs against every
    collector, so the tail shapes must earn their requests. Callers wanting the
    full spread can raise `limit`.

    A long question yields two subjects rather than one — see `_subjects` — and
    the shapes are then spread across both, broad terms first. So the budget
    buys three angles on two topics instead of six on one, which is the better
    trade exactly when one subject was never going to cover the question.
    """
    subjects = _subjects(question)
    if not subjects:
        return [question]
    # Shape-major: whatever `limit` cuts off, it cuts the *tail shapes* from
    # every subject alike, rather than the second subject entirely.
    queries = [
        shape.format(subject=subject).strip() for shape in _QUERY_SHAPES for subject in subjects
    ]
    return list(dict.fromkeys(queries))[:limit]


def _subjects(question: str) -> list[str]:
    """The one or two topics to search for, in the question's own words.

    Two, when the question is long enough to have them. English puts the head
    of a noun phrase at the end — "manual, repetitive, or error-prone tasks
    engineers perform before every production release" is as much about
    *production releases* as about *tasks* — so taking the first four content
    words and stopping drops the half of the subject that names the domain.
    Both ends are kept, and a question short enough that they overlap
    collapses back to the single subject it always produced.
    """
    head = _SOURCING_CLAUSE.split(question, maxsplit=1)[0]
    words = [w.strip(".,?!:;\"'()") for w in head.lower().split()]
    core = [w for w in words if _is_subject_word(w)]
    if not core:
        return []
    return list(
        dict.fromkeys(
            [" ".join(core[:_SUBJECT_WORDS]), " ".join(core[-_SUBJECT_WORDS:])],
        )
    )


def _is_subject_word(word: str) -> bool:
    """Whether a word says anything about the topic being asked about.

    Bare numbers are dropped along with the filler: "top 25" is a quantity the
    asker wants back, and no document about the subject contains it.
    """
    if not word or word.isdigit():
        return False
    return not (word in _QUESTION_WORDS or word in _NOISE_WORDS or word in _INSTRUCTION_WORDS)


def _interleave(items: list[SourceItem]) -> list[SourceItem]:
    """Round-robin across collectors, so trimming takes from every source evenly."""
    by_source: dict[str, list[SourceItem]] = {}
    for item in items:
        by_source.setdefault(item.collector, []).append(item)

    ordered: list[SourceItem] = []
    queues = list(by_source.values())
    while queues:
        for queue in list(queues):
            ordered.append(queue.pop(0))
            if not queue:
                queues.remove(queue)
    return ordered


class CollectEvidenceInput(SkillInput):
    """Placeholder values for `collect-evidence`'s prompt."""

    question: dict[str, Any] = Field(description="The Question artifact, serialised.")
    candidates: list[dict[str, Any]] = Field(
        description="What the collectors actually fetched. The model may only choose "
        "from these; it has no other source of evidence.",
    )


class EvidenceSelection(BaseModel):
    """One fetched item the model judged relevant, and the passage that carries it."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(description="`external_id` of the candidate being kept.")
    excerpt: str = Field(
        description="The passage carrying the point, copied exactly from that "
        "candidate's text. Not paraphrased.",
    )
    evidence_kind: EvidenceKind = EvidenceKind.OTHER
    relevance: str | None = Field(
        default=None,
        description="One line on what this shows about the question.",
    )
    evidence_level: EvidenceLevel = EvidenceLevel.ANECDOTAL
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class CollectEvidenceOutput(SkillOutput):
    """The subset of fetched material worth keeping."""

    selected: list[EvidenceSelection] = Field(
        default_factory=list,
        description="Empty is legitimate: nothing fetched bore on the question.",
    )


@register
class CollectEvidenceSkill(Skill):
    """Search real sources for material bearing on the run's question."""

    name: ClassVar[str] = "collect-evidence"
    description: ClassVar[str] = "Search real sources for material bearing on the question."
    prompt_name: ClassVar[str] = "collect-evidence"
    consumes: ClassVar[tuple[ArtifactKind, ...]] = (ArtifactKind.QUESTION,)
    produces: ClassVar[ArtifactKind] = ArtifactKind.EVIDENCE
    input_schema: ClassVar[type[SkillInput]] = CollectEvidenceInput
    output_schema: ClassVar[type[SkillOutput]] = CollectEvidenceOutput

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._candidates: dict[str, SourceItem] = {}
        self._searched: list[str] = []
        """What was actually searched for. Kept because it is the first thing to
        look at when a stage comes back empty, and by then the queries are two
        hundred log lines above the emptiness they explain."""
        self._vectors: dict[str, list[float]] = {}
        """Vector per kept candidate, keyed by external_id — computed during
        semantic dedup and reused when indexing into memory, so nothing is
        embedded twice in one run."""

    def gather(self, request: SkillRequest) -> CollectEvidenceInput:
        """Run every available collector and hand the results to the prompt.

        Unusually for a `gather`, this performs network I/O — the prompt's inputs
        genuinely are "whatever the sources returned", and there is nowhere else in
        the template that could fetch them.
        """
        question = request.question or next(iter(request.of_kind(ArtifactKind.QUESTION)), None)
        if question is None:
            raise SkillError(f"{self.name} needs the run's question, none supplied")

        self._searched = self._queries(question)
        items = self._semantically_deduplicated(self._fetch(self._searched))
        # Full text is kept for verification; the prompt sees previews. Truncating
        # what is *shown* cannot weaken the excerpt check, because the stored text
        # is a superset of it.
        self._candidates = {item.external_id: item for item in items}

        return CollectEvidenceInput(
            question=question.model_dump(mode="json"),
            candidates=[item.model_dump(mode="json") for item in self._previews(items)],
        )

    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        """Turn selections into Evidence, taking every fact from the fetched item.

        The model supplies the excerpt and its judgement; source, url, author and
        date come from what the collector retrieved, so they cannot be invented.
        """
        if not isinstance(output, CollectEvidenceOutput):
            raise SkillError(
                f"{self.name} expected {CollectEvidenceOutput.__name__}, "
                f"got {type(output).__name__}"
            )

        kept: list[Artifact] = []
        for choice in output.selected:
            candidate = self._candidates.get(choice.candidate_id)
            if candidate is None:
                raise SkillError(
                    f"{self.name} selected {choice.candidate_id!r}, which no collector "
                    f"returned — the model cannot introduce sources of its own"
                )

            kept.append(
                Evidence(
                    id=Evidence.make_id(),
                    run_id=request.run_id,
                    collector=candidate.collector,
                    evidence_kind=choice.evidence_kind,
                    excerpt=self._verified_excerpt(choice.excerpt, candidate),
                    title=candidate.title,
                    author=candidate.author,
                    source_url=candidate.url,
                    source_id=candidate.external_id,
                    published_at=candidate.published_at,
                    evidence_level=choice.evidence_level,
                    confidence=choice.confidence,
                )
            )

        if kept:
            self._remember(kept, request)
        else:
            self._report_nothing_kept()
        return kept

    def _report_nothing_kept(self) -> None:
        """Say what was searched for, when a stage's worth of fetching kept nothing.

        Two very different things end here, and the queries tell them apart.
        Either the sources genuinely have nothing to say — a real finding — or
        the search terms were wrong, and every candidate was about something
        else. The second is not rare: a question phrased as an instruction
        ("Research the top 25…") once derived the query `research top 25
        manual`, fetched 485 matching-but-irrelevant candidates, and kept none
        of them. Nothing in the log said what had been searched for, so the run
        read as "the internet is silent on this" rather than "we asked the
        wrong question", and those call for opposite fixes.
        """
        if not self._candidates:
            logger.warning(
                "%s: no source returned anything for %d quer%s: %s",
                self.name,
                len(self._searched),
                "y" if len(self._searched) == 1 else "ies",
                ", ".join(repr(query) for query in self._searched),
            )
            return
        logger.warning(
            "%s kept none of the %d candidate(s) fetched — nothing bore on the "
            "question. If that is a surprise, look at what was searched for: %s. "
            "Set the Question's search_queries to override the derivation.",
            self.name,
            len(self._candidates),
            ", ".join(repr(query) for query in self._searched),
        )

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _queries(question: Artifact) -> list[str]:
        """The search terms to retrieve with.

        A research question is a poor keyword query. Sources match on words, not
        intent, so "Where do teams lose time in code review?" retrieves anything
        containing "teams lose" — maintenance teams losing spare parts, in one
        measured case. Searching several angles and taking the union measured about
        four times the usable evidence of the question alone: broad terms find the
        topic, complaint-shaped terms find people actually suffering, and neither
        finds what the other does.
        """
        explicit = list(getattr(question, "search_queries", []) or [])
        if explicit:
            return explicit
        return derive_queries(str(getattr(question, "text", "")))

    def _fetch(self, queries: list[str]) -> list[SourceItem]:
        """Search every configured collector with every query, tolerating failures.

        One unreachable source must not lose the others' results — but if *nothing*
        was reachable, that is a broken setup rather than an empty finding.
        """
        settings = get_settings()
        wanted = tuple(settings.collectors) or available()
        if not wanted:
            raise SkillError("No collectors are registered; evidence cannot be gathered")

        config = config_from_settings()
        items: list[SourceItem] = []
        ran: list[str] = []
        skipped: list[str] = []
        throttled: list[str] = []

        for name in wanted:
            collector = self._build(name, config)
            if not collector.available():
                skipped.append(name)
                continue
            found = False
            for index, query in enumerate(queries):
                try:
                    items.extend(collector.search(query, limit=settings.collector_limit))
                    found = True
                except RateLimitedError as exc:
                    # Out of quota is a fact about the source, not about this
                    # query: the next one gets the same refusal. So this source
                    # is done for the stage, said once. Asking anyway is what
                    # turned one exhausted API into sixteen identical warnings
                    # and a minute of round trips spent collecting refusals.
                    remaining = len(queries) - index - 1
                    throttled.append(name)
                    logger.warning(
                        "collector %s is rate limited; skipping its remaining %d "
                        "quer%s this stage: %s",
                        name,
                        remaining,
                        "y" if remaining == 1 else "ies",
                        exc,
                    )
                    break
                except CollectorError as exc:
                    # One query failing is not the source failing; keep the others.
                    logger.warning("collector %s failed on %r: %s", name, query, exc)
            (ran if found else skipped).append(name)

        if not ran:
            # A source that is out of quota is configured correctly and working;
            # it just cannot answer *now*. Telling its operator to go and set one
            # up sends them to change something that is not wrong, and the thing
            # that would actually help — waiting, or a key — goes unsaid.
            raise SkillError(_nothing_ran(skipped, throttled))

        logger.info(
            "collected %d candidate(s) from %s using %d quer%s",
            len(items),
            ", ".join(ran),
            len(queries),
            "y" if len(queries) == 1 else "ies",
        )
        return self._deduplicated(items)

    @staticmethod
    def _build(name: str, config: CollectorConfig) -> Collector:
        try:
            return get_collector(name)(config)
        except CollectorError as exc:
            raise SkillError(str(exc)) from exc

    def _previews(self, items: list[SourceItem]) -> list[SourceItem]:
        """Shorten and, if necessary, drop candidates to fit the context budget.

        Two separate bounds, because they fail differently. Each item is cut to a
        preview: a whole GitHub thread is 20,000 characters, of which the first few
        hundred say what the problem is and the rest is back-and-forth. Then the
        *list* is capped, because forty short items still overflow.

        Interleaved by collector before trimming so one verbose source cannot crowd
        the others out — unbounded, GitHub's issues are four times the volume of
        everything else combined and would be all that survived.
        """
        settings = get_settings()
        budget = (
            int(settings.max_input_tokens * settings.collect_context_fraction) * _CHARS_PER_TOKEN
        )
        limit = settings.collect_preview_chars

        kept: list[SourceItem] = []
        spent = 0
        for item in _interleave(items):
            preview = item.text[:limit]
            # The serialised item carries more than its text; the fixed overhead
            # keeps the estimate honest for short items.
            cost = len(preview) + _ITEM_OVERHEAD_CHARS
            if spent + cost > budget:
                continue
            spent += cost
            kept.append(item.model_copy(update={"text": preview}))

        if len(kept) < len(items):
            self._report_trimming(items, kept, budget, settings.collect_context_fraction)
        return kept

    def _report_trimming(
        self,
        fetched: list[SourceItem],
        kept: list[SourceItem],
        budget: int,
        fraction: float,
    ) -> None:
        """Say what was dropped, per source, and what to change.

        Broken down by collector rather than totalled, because the failure this
        guards against is one verbose source crowding out the rest. A total would
        say the budget bound; only the breakdown shows whether diversity survived it.
        """
        before = Counter(item.collector for item in fetched)
        after = Counter(item.collector for item in kept)
        lines = "; ".join(
            f"{source} {before[source]}->{after.get(source, 0)}" for source in sorted(before)
        )
        logger.warning(
            "%s dropped %d of %d candidates to fit ~%d tokens (%.0f%% of "
            "MAX_INPUT_TOKENS): %s. Raise COLLECT_CONTEXT_FRACTION to read more, or "
            "lower COLLECTOR_LIMIT to fetch less.",
            self.name,
            len(fetched) - len(kept),
            len(fetched),
            budget // _CHARS_PER_TOKEN,
            fraction * 100,
            lines,
        )

    @staticmethod
    def _deduplicated(items: list[SourceItem]) -> list[SourceItem]:
        """Drop repeats, keeping the first sighting. Sources cross-post constantly."""
        seen: set[str] = set()
        unique: list[SourceItem] = []
        for item in items:
            key = f"{item.collector}:{item.external_id}"
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _semantically_deduplicated(self, items: list[SourceItem]) -> list[SourceItem]:
        """Drop candidates that say what an earlier candidate already said.

        Id-based dedup cannot see cross-posts: the same complaint pasted to
        Hacker News and Reddit arrives with two ids, reaches the model twice,
        and double-counts toward pain clusters — one voice reads as two. That
        is the bug this closes, by comparing what candidates *say*: each item's
        opening is embedded, and anything whose cosine similarity to an
        already-kept item reaches DEDUP_THRESHOLD is dropped.

        The first occurrence wins, in interleave order: round-robin across
        collectors has already decided the fair ordering, so "first" means
        "from the source whose turn it was", not "from whichever collector
        happened to return first".

        Without the embedding model this quietly does nothing — the id-based
        dedup already ran, and a run without memory must behave exactly as
        every run did before this feature existed.
        """
        settings = get_settings()
        if not settings.memory_enabled or len(items) < 2:
            return items

        ordered = _interleave(items)
        try:
            vectors = _build_embedder().encode([item.text[:_FINGERPRINT_CHARS] for item in ordered])
        except MemoryUnavailableError as exc:
            logger.debug("semantic dedup skipped, memory unavailable: %s", exc)
            return items

        kept: list[SourceItem] = []
        kept_vectors: list[list[float]] = []
        for item, vector in zip(ordered, vectors, strict=True):
            twin = next(
                (
                    earlier
                    for earlier, earlier_vector in zip(kept, kept_vectors, strict=True)
                    if cosine(vector, earlier_vector) >= settings.dedup_threshold
                ),
                None,
            )
            if twin is not None:
                # Reported like _report_trimming reports: a dropped candidate is
                # never silent, and the pairing shows *why* it was judged a repeat.
                logger.info(
                    "%s dropped near-duplicate from %s: %r — same complaint as %s: %r",
                    self.name,
                    item.collector,
                    item.title or item.external_id,
                    twin.collector,
                    twin.title or twin.external_id,
                )
                continue
            kept.append(item)
            kept_vectors.append(vector)
            self._vectors[item.external_id] = vector
        return kept

    def _remember(self, kept: list[Artifact], request: SkillRequest) -> None:
        """Index the kept Evidence into cross-run memory, best-effort.

        Wrapped whole: memory is a convenience layered on the run, so any
        failure here — no model, no extension, a locked file — is a debug line,
        never a run failure. The vectors computed during semantic dedup are
        reused (the excerpt came out of the candidate's text, so the candidate's
        vector is an honest stand-in); only an item that was never embedded —
        say, when dedup was skipped — is encoded now.
        """
        settings = get_settings()
        if not settings.memory_enabled:
            return
        try:
            evidence = [artifact for artifact in kept if isinstance(artifact, Evidence)]
            embedder: Embedder | None = None
            vectors: list[list[float]] = []
            for item in evidence:
                vector = self._vectors.get(item.source_id or "")
                if vector is None:
                    if embedder is None:
                        embedder = _build_embedder()
                    candidate = self._candidates.get(item.source_id or "")
                    text = candidate.text if candidate is not None else item.excerpt
                    vector = embedder.encode([text[:_FINGERPRINT_CHARS]])[0]
                vectors.append(vector)

            question = request.question or next(iter(request.of_kind(ArtifactKind.QUESTION)), None)
            store = MemoryStore(self.registry.paths.memory_db)
            try:
                remembered = store.index(
                    evidence,
                    vectors,
                    run_id=request.run_id,
                    question_text=str(getattr(question, "text", "")),
                    model=settings.embedding_model,
                )
            finally:
                store.close()
            logger.debug(
                "%s remembered %d of %d evidence item(s)", self.name, remembered, len(evidence)
            )
        except Exception as exc:  # memory must never cost you the run
            logger.debug("%s could not index evidence into memory: %s", self.name, exc)

    def _verified_excerpt(self, excerpt: str, candidate: SourceItem) -> str:
        """Return the excerpt only if it really appears in what was fetched.

        Compared with whitespace collapsed, because a model reflows a quote across
        lines without changing a word — that is still faithful, and rejecting it
        would push the model toward shorter, less useful excerpts. Anything beyond
        reflowing is a rewrite, and a rewrite is not a quote.
        """
        cleaned = excerpt.strip()
        if not cleaned:
            raise SkillError(f"{self.name} returned an empty excerpt for {candidate.external_id}")

        if _normalise(cleaned) not in _normalise(candidate.text):
            raise SkillError(
                f"{self.name} produced an excerpt that does not appear in "
                f"{candidate.external_id}: {cleaned[:80]!r}. Evidence must be quoted, "
                f"not written."
            )
        return cleaned


def _nothing_ran(skipped: list[str], throttled: list[str]) -> str:
    """Why no source answered, and which of the two remedies applies.

    "Configure a source" and "wait for a quota to reset" are opposite
    instructions, and a run that reported the first when the second was true
    would send someone editing settings that were already right.
    """
    # A throttled collector is in both lists: it was reached, and then it gave up.
    unavailable = [name for name in skipped if name not in throttled]
    if throttled and not unavailable:
        return (
            f"No collector could run: {', '.join(throttled)} "
            f"{'is' if len(throttled) == 1 else 'are'} rate limited. The setup is fine "
            f"— wait for the quota to reset, add an API key to raise it, or configure "
            f"another source."
        )
    detail = f"unavailable: {', '.join(unavailable)}" if unavailable else "none available"
    if throttled:
        detail += f"; rate limited: {', '.join(throttled)}"
    return (
        f"No collector could run ({detail}). "
        f"Configure at least one source before collecting evidence."
    )


def _normalise(text: str) -> str:
    """Collapse whitespace so a reflowed quote still matches its source."""
    return _WHITESPACE.sub(" ", text).strip().casefold()


__all__ = [
    "CollectEvidenceInput",
    "CollectEvidenceOutput",
    "CollectEvidenceSkill",
    "EvidenceSelection",
    "derive_queries",
]
