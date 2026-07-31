"""Capabilities, tiers, and the resolver between them.

Three layers, each of which changes at a different rate:

    stage  ->  capability  ->  tier  ->  slug
    decision   deep_reasoning  claude-opus   anthropic/claude-opus-4.8

* **Stage -> capability** describes the *work*: the decision stage needs deep
  reasoning. This is a fact about the pipeline and almost never changes.
* **Capability -> tier** is the judgement call: "the best synthesis model today is
  Claude Sonnet". When that stops being true, one line changes and every stage
  doing synthesis follows. This is the layer the indirection exists for.
* **Tier -> slug** is churn: `claude-opus-4.5` became `4.6`, `4.7`, `4.8` inside a
  year. Resolved against the live catalogue so nobody edits config for it.

Without the middle layer, deciding that some other model is better at synthesis
means finding and editing every stage that happens to do synthesis. With it, the
stages say what they need and stay put.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict

from app.llm.catalog import ModelCatalog

# Suffixes marking a different product rather than a newer version of the same one.
_COMMON_EXCLUSIONS: Final[tuple[str, ...]] = (
    "-image",
    "-audio",
    "-search",
    "-codex",
    "-embed",
)


class Capability(StrEnum):
    """What a stage needs from a model, independent of who currently provides it."""

    FAST_EXTRACT = "fast_extract"
    """High-volume, mechanical work. Pulling structure out of text."""

    SUMMARIZE = "summarize"
    """Condensing many sources into one account. Wants a long context window."""

    SYNTHESIS = "synthesis"
    """Grouping, comparing, and organising material that is already summarised."""

    DEEP_REASONING = "deep_reasoning"
    """Novel inference and judgement. The expensive tier, used sparingly."""

    MARKET_REASONING = "market_reasoning"
    """Commercial estimation: buyers, budgets, pricing, sizing."""

    SECOND_OPINION = "second_opinion"
    """Adversarial review. Deliberately a different family from the work it checks,
    so a model's blind spots are not used to audit themselves."""


class ModelTier(StrEnum):
    """A vendor and weight class. The thing a capability is currently pointed at."""

    GEMINI_FLASH = "gemini-flash"
    GEMINI_PRO = "gemini-pro"
    CLAUDE_SONNET = "claude-sonnet"
    CLAUDE_OPUS = "claude-opus"
    GPT = "gpt"


CAPABILITY_TIERS: Final[Mapping[Capability, ModelTier]] = MappingProxyType(
    {
        Capability.FAST_EXTRACT: ModelTier.GEMINI_FLASH,
        Capability.SUMMARIZE: ModelTier.GEMINI_PRO,
        Capability.SYNTHESIS: ModelTier.CLAUDE_SONNET,
        Capability.DEEP_REASONING: ModelTier.CLAUDE_OPUS,
        Capability.MARKET_REASONING: ModelTier.GPT,
        Capability.SECOND_OPINION: ModelTier.GPT,
    },
)
"""The one mapping worth revisiting as models improve. Override with `LLM_CAPABILITIES`."""


class TierSpec(BaseModel):
    """How to find the model currently filling a tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vendor: str
    require: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    pinned: str
    """Used when the catalogue is unreachable. Expected to age; that is the point."""


TIER_SPECS: Final[Mapping[ModelTier, TierSpec]] = MappingProxyType(
    {
        ModelTier.GEMINI_FLASH: TierSpec(
            vendor="google",
            require=("gemini", "flash"),
            exclude=(*_COMMON_EXCLUSIONS, "-lite", "-preview"),
            pinned="google/gemini-3.6-flash",
        ),
        ModelTier.GEMINI_PRO: TierSpec(
            vendor="google",
            require=("gemini", "pro"),
            exclude=(*_COMMON_EXCLUSIONS, "-preview"),
            pinned="google/gemini-2.5-pro",
        ),
        ModelTier.CLAUDE_SONNET: TierSpec(
            vendor="anthropic",
            require=("claude", "sonnet"),
            exclude=(*_COMMON_EXCLUSIONS, "-fast"),
            pinned="anthropic/claude-sonnet-5",
        ),
        ModelTier.CLAUDE_OPUS: TierSpec(
            vendor="anthropic",
            require=("claude", "opus"),
            exclude=(*_COMMON_EXCLUSIONS, "-fast"),
            pinned="anthropic/claude-opus-4.8",
        ),
        ModelTier.GPT: TierSpec(
            vendor="openai",
            require=("gpt",),
            exclude=(*_COMMON_EXCLUSIONS, "-mini", "-nano", "-chat", "-pro", "-instruct"),
            pinned="openai/gpt-5.5",
        ),
    },
)


def resolve_tier(tier: ModelTier, catalogue: ModelCatalog | None) -> str:
    """The slug currently filling a tier.

    Falls back to the pinned slug when the catalogue is missing or matches nothing —
    a stale but working model beats refusing to run.
    """
    spec = TIER_SPECS[tier]
    if catalogue is None:
        return spec.pinned
    found = catalogue.latest(spec.vendor, require=spec.require, exclude=spec.exclude)
    return found.id if found is not None else spec.pinned


def tier_for(capability: Capability, overrides: Mapping[str, str] | None = None) -> ModelTier:
    """Which tier currently serves a capability, honouring config overrides."""
    if overrides and capability.value in overrides:
        return ModelTier(overrides[capability.value])
    return CAPABILITY_TIERS[capability]


def resolve_capability(
    capability: Capability,
    catalogue: ModelCatalog | None,
    overrides: Mapping[str, str] | None = None,
) -> str:
    """The slug a capability currently resolves to, all three layers applied."""
    return resolve_tier(tier_for(capability, overrides), catalogue)


def resolve_all_tiers(catalogue: ModelCatalog | None) -> dict[ModelTier, str]:
    """Every tier and the slug filling it."""
    return {tier: resolve_tier(tier, catalogue) for tier in ModelTier}


__all__ = [
    "CAPABILITY_TIERS",
    "TIER_SPECS",
    "Capability",
    "ModelTier",
    "TierSpec",
    "resolve_all_tiers",
    "resolve_capability",
    "resolve_tier",
    "tier_for",
]
