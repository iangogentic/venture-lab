"""Model routing: which role answers which stage, at what temperature.

Routes name a **logical role**, never a slug. `decision` asks for "the current
Claude Opus"; which slug that is today is resolved against the live catalogue.

Three things are decided per stage, and they are decided together because they are
one judgement about the work:

* **role** — how much reasoning the step is worth paying for;
* **temperature** — how much variation the step should tolerate. Extraction and
  adversarial search want none; inferring an opportunity wants a little;
* **output cap** — how much the step is allowed to write.

Gateway-side routing rides along: a fallback chain (`models`) and a provider
preference (`provider.sort`), so a stage can fail over without the caller writing
retry logic.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.llm.catalog import ModelCatalog, ModelInfo
from app.llm.provider import Provider
from app.llm.roles import (
    Capability,
    ModelTier,
    resolve_capability,
    resolve_tier,
    tier_for,
)
from app.utils.errors import ConfigurationError

VENDOR_PROVIDERS: Final[Mapping[str, Provider]] = MappingProxyType(
    {"anthropic": Provider.CLAUDE, "openai": Provider.GPT, "google": Provider.GEMINI},
)
"""Which adapter serves each gateway vendor prefix."""


STAGE_CAPABILITIES: Final[Mapping[str, Capability]] = MappingProxyType(
    {
        "collect-evidence": Capability.FAST_EXTRACT,
        "research-brief": Capability.SUMMARIZE,
        "cluster-pains": Capability.SYNTHESIS,
        "discover-opportunities": Capability.DEEP_REASONING,
        "analyze-market": Capability.MARKET_REASONING,
        "analyze-competition": Capability.SYNTHESIS,
        # Adversarial review of the analyses above it — the definition of a second
        # opinion, and deliberately a different family from the synthesis it checks.
        "contradiction-analysis": Capability.SECOND_OPINION,
        "decision": Capability.DEEP_REASONING,
        "interview-plan": Capability.SECOND_OPINION,
        # Off-pipeline, run by `op leads harvest` rather than the engine. Picking
        # quotes and tagging intent is mechanical selection, so the cheapest tier.
        "harvest-leads": Capability.FAST_EXTRACT,
        # Off-pipeline, run by `op report`. Weaving every artifact of a run into
        # one faithful narrative is synthesis over already-graded material — no
        # new inference, but the through-line across opportunities is the work.
        "compose-report": Capability.SYNTHESIS,
    },
)
"""What each stage needs. A fact about the work, not about any vendor."""

STAGE_TEMPERATURES: Final[Mapping[str, float]] = MappingProxyType(
    {
        "collect-evidence": 0.0,
        "research-brief": 0.1,
        "cluster-pains": 0.1,
        "discover-opportunities": 0.2,
        "analyze-market": 0.1,
        "analyze-competition": 0.1,
        "contradiction-analysis": 0.0,
        "decision": 0.1,
        "interview-plan": 0.2,
        "harvest-leads": 0.0,
        "compose-report": 0.1,
    },
)
"""Zero where the step must not invent, higher only where it must generate."""

STAGE_MAX_OUTPUT_TOKENS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "collect-evidence": 4_000,
        "research-brief": 8_000,
        "cluster-pains": 4_000,
        "discover-opportunities": 6_000,
        "analyze-market": 4_000,
        "analyze-competition": 4_000,
        "contradiction-analysis": 4_000,
        "decision": 3_000,
        "interview-plan": 4_000,
        "harvest-leads": 3_000,
        # The largest budget of any stage: a report narrates the whole run. Still
        # bounded — a body that will not fit here needs restructuring, not room.
        "compose-report": 10_000,
    },
)
"""What each stage plausibly needs to write.

Sized per stage rather than globally because gateways reserve credit against
`max_tokens` before the call: asking for the largest stage's budget on every stage
makes the cheapest one as expensive to attempt as the dearest, and can fail a run
outright on a limited key. `MAX_OUTPUT_TOKENS` still caps all of them.
"""


class RoutingSort(StrEnum):
    """How the gateway should choose among providers serving the same model."""

    PRICE = "price"
    THROUGHPUT = "throughput"
    LATENCY = "latency"


def provider_for_model(slug: str) -> Provider:
    """Derive the adapter from a model slug.

    Raises:
        ConfigurationError: If the slug has no vendor prefix, or names a vendor
            this application has no adapter for.
    """
    vendor, separator, _ = slug.partition("/")
    if not separator:
        raise ConfigurationError(
            f"Model {slug!r} has no vendor prefix; expected e.g. 'anthropic/claude-sonnet-5'"
        )
    provider = VENDOR_PROVIDERS.get(vendor)
    if provider is None:
        known = ", ".join(sorted(VENDOR_PROVIDERS))
        raise ConfigurationError(f"No adapter for vendor {vendor!r}. Supported: {known}")
    return provider


class Route(BaseModel):
    """What a stage asks for, in capability terms, before any slug is known."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: Capability
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    fallback_tiers: tuple[ModelTier, ...] = ()
    sort: RoutingSort | None = None


class ResolvedRoute(BaseModel):
    """A route with the catalogue applied: concrete slug, and what it can do."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: str | None
    capability: Capability
    tier: ModelTier
    model: str
    provider: Provider
    fallbacks: tuple[str, ...] = ()
    temperature: float | None = None
    max_output_tokens: int | None = None
    sort: RoutingSort | None = None
    supports_structured_outputs: bool = False
    supports_response_format: bool = False
    supports_prompt_caching: bool = False
    needs_explicit_cache_write: bool = False
    context_length: int | None = None

    @property
    def chain(self) -> tuple[str, ...]:
        """Primary then fallbacks, in the order the gateway will try them."""
        return (self.model, *self.fallbacks)


class ModelRouter:
    """Resolves a stage name to the model that should serve it."""

    def __init__(
        self,
        default: Route,
        routes: Mapping[str, Route] | None = None,
        catalogue: ModelCatalog | None = None,
    ) -> None:
        self.default = default
        self.routes: dict[str, Route] = dict(routes or {})
        self.catalogue = catalogue

    @classmethod
    def from_settings(cls, catalogue: ModelCatalog | None = None) -> "ModelRouter":
        """Build the router for this process from the recommended defaults plus overrides.

        `llm_stage_capabilities` overrides what a stage needs; `llm_capabilities`
        overrides which tier serves a capability. The second is the one to reach
        for when a better model appears — it moves every stage at once.
        """
        settings = get_settings()
        sort = RoutingSort(settings.llm_sort) if settings.llm_sort else None
        fallback_tiers = tuple(ModelTier(tier) for tier in settings.llm_fallback_tiers)
        cap = settings.max_output_tokens

        def build(stage: str | None) -> Route:
            capability: Capability | None = None
            if stage is not None and stage in settings.llm_stage_capabilities:
                capability = Capability(settings.llm_stage_capabilities[stage])
            elif stage is not None:
                capability = STAGE_CAPABILITIES.get(stage)
            if capability is None:
                capability = Capability(settings.llm_default_capability)

            temperature = (
                settings.llm_temperatures.get(stage)
                if stage is not None and stage in settings.llm_temperatures
                else STAGE_TEMPERATURES.get(stage or "", settings.llm_temperature)
            )
            # The global setting is a ceiling, not the value: it should be able to
            # cap a run without anyone restating all nine stages.
            wanted = (
                settings.llm_max_output_tokens.get(stage)
                if stage is not None and stage in settings.llm_max_output_tokens
                else STAGE_MAX_OUTPUT_TOKENS.get(stage or "", cap)
            )
            return Route(
                capability=capability,
                temperature=temperature,
                max_output_tokens=min(wanted or cap, cap),
                fallback_tiers=fallback_tiers,
                sort=sort,
            )

        stages = (
            set(STAGE_CAPABILITIES)
            | set(settings.llm_stage_capabilities)
            | set(settings.llm_temperatures)
        )
        return cls(
            default=build(None),
            routes={stage: build(stage) for stage in sorted(stages)},
            catalogue=catalogue,
        )

    def resolve(self, task: str | None = None) -> ResolvedRoute:
        """The concrete destination for a task, with the catalogue applied."""
        route = self.routes.get(task, self.default) if task is not None else self.default
        overrides = get_settings().llm_capabilities
        slug = resolve_capability(route.capability, self.catalogue, overrides)
        info = self.catalogue.get(slug) if self.catalogue is not None else None
        provider = provider_for_model(slug)

        return ResolvedRoute(
            task=task,
            capability=route.capability,
            tier=tier_for(route.capability, overrides),
            model=slug,
            provider=provider,
            fallbacks=tuple(
                resolved
                for tier in route.fallback_tiers
                if (resolved := resolve_tier(tier, self.catalogue)) != slug
            ),
            temperature=route.temperature,
            max_output_tokens=route.max_output_tokens,
            sort=route.sort,
            **_capabilities(info, native_openai=_uses_native_openai(provider)),
        )

    def table(self, tasks: tuple[str, ...] | None = None) -> list[ResolvedRoute]:
        """Every task and where it would go, for display."""
        names = tasks if tasks is not None else tuple(sorted(self.routes))
        return [self.resolve(name) for name in names]


def _capabilities(
    info: ModelInfo | None,
    *,
    native_openai: bool = False,
) -> dict[str, bool | int | None]:
    """What the catalogue says the chosen model can do.

    Unknown means *not* assumed: an unlisted model gets the conservative path, so
    a stage degrades to plain JSON rather than failing on an unsupported schema.
    Native OpenAI is a separate audited transport whose Responses client supports
    JSON object and JSON Schema formats directly; it does not need OpenRouter's
    catalogue to prove those transport capabilities.
    """
    if native_openai:
        return {
            "supports_structured_outputs": True,
            "supports_response_format": True,
            "supports_prompt_caching": True,
            "needs_explicit_cache_write": False,
            "context_length": info.context_length if info is not None else None,
        }
    if info is None:
        return {
            "supports_structured_outputs": False,
            "supports_response_format": False,
            "supports_prompt_caching": False,
            "needs_explicit_cache_write": False,
            "context_length": None,
        }
    return {
        "supports_structured_outputs": info.supports_structured_outputs,
        "supports_response_format": info.supports_response_format,
        "supports_prompt_caching": info.supports_prompt_caching,
        "needs_explicit_cache_write": info.needs_explicit_cache_write,
        "context_length": info.context_length,
    }


def get_catalogue() -> ModelCatalog | None:
    """The model catalogue, from disk cache or the gateway. None when unreachable."""
    settings = get_settings()
    if not settings.llm_use_catalogue or settings.llm_transport == "openai":
        return None
    return ModelCatalog.load(settings.model_cache_path)


def _uses_native_openai(provider: Provider) -> bool:
    """Whether a GPT route will use the direct Responses transport."""
    if provider is not Provider.GPT:
        return False
    settings = get_settings()
    if settings.llm_transport == "openai":
        return True
    if settings.llm_transport == "openrouter":
        return False
    return settings.openai_api_key is not None


def get_router() -> ModelRouter:
    """The router for the current settings."""
    return ModelRouter.from_settings(get_catalogue())


__all__ = [
    "STAGE_CAPABILITIES",
    "STAGE_MAX_OUTPUT_TOKENS",
    "STAGE_TEMPERATURES",
    "VENDOR_PROVIDERS",
    "ModelRouter",
    "ResolvedRoute",
    "Route",
    "RoutingSort",
    "get_catalogue",
    "get_router",
    "provider_for_model",
]
