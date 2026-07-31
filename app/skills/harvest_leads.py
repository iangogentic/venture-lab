"""`harvest-leads`: turn a cluster's cited evidence into a ledger of findable people.

Deliberately **not** a stage in the pipeline's `STAGE_ORDER`, for three reasons:

* Lead harvesting is post-research and human-gated. Outreach is a human act — a
  public, human-written reply, never an automated message — so its input can be
  produced on demand by `op leads harvest` when a person decides to reach out,
  not as a side effect of every research run.
* A cluster can legitimately yield zero leads (nobody quoted had both a handle
  and a permalink, or nobody expressed a pain), and the stage-resume machinery
  would misread that empty-but-correct result as unfinished work and re-run it
  forever.
* Keeping it out preserves the nine-stage research pipeline unchanged: the
  research artifacts and their guarantees are exactly what they were before
  leads existed.

The same select-don't-author discipline as `collect-evidence` applies: the model
classifies intent and picks the quote, but every quote is verified to be a
literal passage of the evidence item's excerpt, and every identity field —
author, url, collector, published date — is copied from the Evidence artifact,
never taken from the model's reply.
"""

import re
from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import Artifact, ArtifactKind, Evidence, Lead, LeadIntent, PainCluster
from app.artifacts.lineage import supporting
from app.skills.base import Batching, Skill, SkillInput, SkillOutput, SkillRequest, register
from app.utils.errors import ArtifactError, SkillError
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Redeclared rather than imported from collect_evidence: skills may not import
# each other (enforced by test), and the two-line rule is cheaper than the coupling.
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class HarvestLeadsInput(SkillInput):
    """Placeholder values for `harvest-leads`'s prompt."""

    clusters: list[dict[str, Any]] = Field(
        description="Every pain cluster to harvest for, serialised as JSON.",
    )
    candidates: list[dict[str, Any]] = Field(
        description="Per-cluster candidate lists: the authored, linked evidence each "
        "cluster cites. The model may only pick from these; it has no other source "
        "of people.",
    )


class LeadSelection(BaseModel):
    """One candidate the model judged to be a person expressing the pain."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(description="`evidence_id` of the candidate being kept.")
    cluster_id: str = Field(description="`cluster_id` of the list the candidate came from.")
    quote: str = Field(
        description="The passage showing the pain, copied exactly from that candidate's "
        "excerpt. Not paraphrased.",
    )
    intent: LeadIntent = Field(
        description="What the quoted words express: seeking, complaining, or mentioning.",
    )
    intent_rationale: str | None = Field(
        default=None,
        description="One line on why the intent was read that way.",
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class HarvestLeadsOutput(SkillOutput):
    """The candidates worth a human's attention."""

    leads: list[LeadSelection] = Field(
        default_factory=list,
        description="Empty is legitimate: nobody in the candidates expressed the pain.",
    )


@register
class HarvestLeadsSkill(Skill):
    """Turn the people behind a cluster's evidence into a reviewable lead ledger."""

    name: ClassVar[str] = "harvest-leads"
    description: ClassVar[str] = "Turn clustered evidence into leads: people, quotes, permalinks."
    prompt_name: ClassVar[str] = "harvest-leads"
    consumes: ClassVar[tuple[ArtifactKind, ...]] = (
        ArtifactKind.PAIN_CLUSTER,
        ArtifactKind.EVIDENCE,
    )
    # One call over everything: cluster counts are small, and clusters share one
    # evidence pool — splitting per cluster would resend the same pool per call.
    batching: ClassVar[Batching] = Batching.FAN_IN
    produces: ClassVar[ArtifactKind] = ArtifactKind.LEAD
    input_schema: ClassVar[type[SkillInput]] = HarvestLeadsInput
    output_schema: ClassVar[type[SkillOutput]] = HarvestLeadsOutput

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._clusters: dict[str, PainCluster] = {}
        self._candidates: dict[str, dict[str, Evidence]] = {}
        """Viable candidates per cluster id, keyed by evidence id — what `assemble`
        verifies every selection against, exactly as `collect-evidence` keeps the
        fetched items it showed the model."""

    def gather(self, request: SkillRequest) -> HarvestLeadsInput:
        """Collect, per cluster, the evidence it cites that names a findable person.

        Only items carrying both an `author` and a `source_url` become candidates:
        no person, no permalink → no lead. The others remain perfectly good
        evidence — they are just not leads, and they are not shown to the model at
        all, so it cannot select what could never be contacted.
        """
        clusters = [
            artifact
            for artifact in request.of_kind(ArtifactKind.PAIN_CLUSTER)
            if isinstance(artifact, PainCluster)
        ]
        pool = [
            artifact
            for artifact in request.of_kind(ArtifactKind.EVIDENCE)
            if isinstance(artifact, Evidence)
        ]

        self._clusters = {cluster.id: cluster for cluster in clusters}
        self._candidates = {}
        per_cluster: list[dict[str, Any]] = []
        for cluster in clusters:
            viable = [
                item
                for item in self._cited_evidence(cluster, pool)
                if item.author is not None and item.source_url is not None
            ]
            self._candidates[cluster.id] = {item.id: item for item in viable}
            per_cluster.append(
                {
                    "cluster_id": cluster.id,
                    "label": cluster.label,
                    "candidates": [self._preview(item) for item in viable],
                }
            )

        return HarvestLeadsInput(
            clusters=[cluster.model_dump(mode="json") for cluster in clusters],
            candidates=per_cluster,
        )

    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        """Turn selections into Leads, taking every identity fact from the evidence.

        The model supplies the quote and the intent reading; author, url,
        collector, external id and date come from the Evidence artifact, so they
        cannot be invented. A quote that is not a literal passage of that
        evidence's excerpt fails the whole reply.
        """
        if not isinstance(output, HarvestLeadsOutput):
            raise SkillError(
                f"{self.name} expected {HarvestLeadsOutput.__name__}, got {type(output).__name__}"
            )

        leads: list[Artifact] = []
        for choice in output.leads:
            pool = self._candidates.get(choice.cluster_id)
            if pool is None:
                raise SkillError(
                    f"{self.name} placed a lead under {choice.cluster_id!r}, which is "
                    f"not a cluster it was shown — the model cannot introduce clusters "
                    f"of its own"
                )
            candidate = pool.get(choice.evidence_id)
            if candidate is None:
                raise SkillError(
                    f"{self.name} selected {choice.evidence_id!r}, which was not a "
                    f"candidate for {choice.cluster_id!r} — the model cannot introduce "
                    f"people of its own"
                )
            if candidate.author is None or candidate.source_url is None:
                # gather() filtered these out; reaching here is a bug, not a model crime.
                raise SkillError(f"{self.name} candidate {candidate.id!r} lost its author or url")

            cluster = self._clusters[choice.cluster_id]
            leads.append(
                Lead(
                    id=Lead.make_id(),
                    run_id=request.run_id,
                    cluster=cluster.ref,
                    evidence=candidate.ref,
                    collector=candidate.collector,
                    author=candidate.author,
                    url=str(candidate.source_url),
                    external_id=candidate.source_id,
                    published_at=candidate.published_at,
                    quote=self._verified_quote(choice.quote, candidate),
                    intent=choice.intent,
                    intent_rationale=choice.intent_rationale,
                    confidence=choice.confidence,
                    # A lead is as grounded as the observation it points at, no more.
                    evidence_level=candidate.evidence_level,
                    parents=[cluster.ref, candidate.ref],
                )
            )
        return leads

    # ---------------------------------------------------------------- internals

    def _cited_evidence(self, cluster: PainCluster, pool: list[Evidence]) -> list[Evidence]:
        """The evidence a cluster rests on, walked through the briefs that built it.

        Clusters do not cite evidence directly: their `parents` are the research
        briefs they were folded from, and the evidence refs live in each brief's
        signals and quotes. So citation is followed two hops — anything the
        cluster references itself, plus everything referenced by each parent
        brief the workspace still holds. A brief that cannot be loaded costs its
        leads, never the run.
        """
        cited: dict[str, Evidence] = {
            item.id: item for item in supporting(cluster, pool) if isinstance(item, Evidence)
        }
        for ref in cluster.parents:
            if ref.kind is not ArtifactKind.RESEARCH_BRIEF:
                continue
            try:
                brief = self.registry.resolve(ref)
            except ArtifactError as exc:
                logger.debug("%s could not load brief %s: %s", self.name, ref.id, exc)
                continue
            for item in supporting(brief, pool):
                if isinstance(item, Evidence):
                    cited.setdefault(item.id, item)
        return list(cited.values())

    @staticmethod
    def _preview(item: Evidence) -> dict[str, Any]:
        """What the prompt is shown per candidate: identity plus the checkable text."""
        return {
            "evidence_id": item.id,
            "author": item.author,
            "url": str(item.source_url),
            "collector": item.collector,
            "published_at": item.published_at.isoformat() if item.published_at else None,
            "excerpt": item.excerpt,
        }

    def _verified_quote(self, quote: str, candidate: Evidence) -> str:
        """Return the quote only if it really appears in that evidence's excerpt.

        Compared with whitespace collapsed, as `collect-evidence` compares its
        excerpts: reflowing a quote across lines is still faithful; anything
        beyond that is a rewrite, and a rewrite is not a quote.
        """
        cleaned = quote.strip()
        if not cleaned:
            raise SkillError(f"{self.name} returned an empty quote for {candidate.id}")

        if _normalise(cleaned) not in _normalise(candidate.excerpt):
            raise SkillError(
                f"{self.name} produced a quote that does not appear in "
                f"{candidate.id}: {cleaned[:80]!r}. A lead's quote must be copied, "
                f"not composed — the model cannot put words in a person's mouth."
            )
        return cleaned


def _normalise(text: str) -> str:
    """Collapse whitespace so a reflowed quote still matches its source."""
    return _WHITESPACE.sub(" ", text).strip().casefold()


__all__ = [
    "HarvestLeadsInput",
    "HarvestLeadsOutput",
    "HarvestLeadsSkill",
    "LeadSelection",
]
