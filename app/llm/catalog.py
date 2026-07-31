"""The gateway's model catalogue: what exists, and what each model can do.

Fetched from OpenRouter's Models API so the application never has to hold an
opinion about which slug is current. That matters twice over: model names churn
constantly, and capability — structured outputs, response formats, prompt
caching — is a property of the model, not something to assume.

The catalogue is cached on disk because it changes on the order of days and a
pipeline run should not depend on a network round trip to decide which model to
call. When the network is unavailable and no cache exists, callers fall back to
pinned slugs rather than failing: a stale model name still runs, a crash does not.
"""

import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.utils.errors import LLMError
from app.utils.logging import get_logger

logger = get_logger(__name__)

MODELS_URL: Final[str] = "https://openrouter.ai/api/v1/models"
"""Public endpoint — no API key required, which is why this can run at import time."""

CACHE_TTL_SECONDS: Final[int] = 24 * 60 * 60


class ModelInfo(BaseModel):
    """One model as the gateway describes it."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    name: str = ""
    created: int = 0
    context_length: int | None = None
    supported_parameters: frozenset[str] = frozenset()
    # Values are mostly decimal strings, but some vendors nest a list of tiered
    # `overrides` here, so the value type is deliberately open. Only the keys are
    # read — their presence is what signals a capability.
    pricing: dict[str, Any] = Field(default_factory=dict)
    max_completion_tokens: int | None = None

    @property
    def vendor(self) -> str:
        """The slug's vendor prefix, e.g. `anthropic`."""
        return self.id.partition("/")[0]

    @property
    def supports_structured_outputs(self) -> bool:
        """Whether the model can be held to a JSON Schema."""
        return "structured_outputs" in self.supported_parameters

    @property
    def supports_response_format(self) -> bool:
        """Whether the model accepts a `response_format` at all, schema or not."""
        return "response_format" in self.supported_parameters

    @property
    def supports_temperature(self) -> bool:
        """Some reasoning models reject a temperature outright."""
        return "temperature" in self.supported_parameters

    @property
    def supports_prompt_caching(self) -> bool:
        """Whether the provider prices cached input, which is how caching shows up here.

        A model that bills `input_cache_read` is one where repeating a prefix is
        cheaper than sending it fresh — exactly the property the static system and
        skill prompts are meant to exploit.
        """
        return "input_cache_read" in self.pricing

    @property
    def needs_explicit_cache_write(self) -> bool:
        """Whether caching must be asked for rather than happening automatically.

        Anthropic bills a write to prime the cache and needs a `cache_control`
        breakpoint; OpenAI caches automatically and prices only reads.
        """
        return "input_cache_write" in self.pricing


class ModelCatalog:
    """Every model the gateway offers, queryable by slug or by family."""

    def __init__(self, models: Sequence[ModelInfo]) -> None:
        self.models = tuple(models)
        self._by_id = {model.id: model for model in self.models}

    def __len__(self) -> int:
        return len(self.models)

    def get(self, slug: str) -> ModelInfo | None:
        """Look up one model, or None if the gateway does not list it."""
        return self._by_id.get(slug)

    def latest(
        self,
        vendor: str,
        *,
        require: Iterable[str] = (),
        exclude: Iterable[str] = (),
    ) -> ModelInfo | None:
        """Newest model from a vendor whose slug matches every `require` fragment.

        "Newest" is the gateway's own `created` timestamp rather than a version
        number parsed out of the name — version schemes differ per vendor and
        change without warning, but the release date does not lie.
        """
        required = tuple(require)
        excluded = tuple(exclude)
        candidates = [
            model
            for model in self.models
            if model.vendor == vendor
            and all(fragment in model.id for fragment in required)
            and not any(fragment in model.id for fragment in excluded)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda model: (model.created, model.id))

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Self:
        """Build from a raw Models API response."""
        models: list[ModelInfo] = []
        for entry in payload.get("data", []):
            top = entry.get("top_provider") or {}
            models.append(
                ModelInfo(
                    id=entry.get("id", ""),
                    name=entry.get("name", ""),
                    created=entry.get("created", 0) or 0,
                    context_length=entry.get("context_length"),
                    supported_parameters=frozenset(entry.get("supported_parameters") or ()),
                    pricing=dict(entry.get("pricing") or {}),
                    max_completion_tokens=top.get("max_completion_tokens"),
                )
            )
        return cls([model for model in models if model.id])

    @classmethod
    def fetch(cls, *, timeout: float = 10.0) -> Self:
        """Fetch the catalogue from the gateway.

        Raises:
            LLMError: If the endpoint cannot be reached or returns nonsense.
        """
        try:
            response = httpx.get(MODELS_URL, timeout=timeout)
            response.raise_for_status()
            return cls.from_payload(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"Could not fetch the model catalogue: {exc}") from exc

    @classmethod
    def load(cls, cache_path: Path, *, ttl: int = CACHE_TTL_SECONDS) -> Self | None:
        """Return the catalogue from disk cache, refreshing it when stale.

        Never raises. A run should degrade to pinned model names rather than fail
        because a metadata endpoint was briefly unreachable.
        """
        cached = cls._read_cache(cache_path, ttl=ttl)
        if cached is not None:
            return cached

        try:
            catalogue = cls.fetch()
        except LLMError as exc:
            logger.debug("model catalogue unavailable, falling back to pinned slugs: %s", exc)
            return cls._read_cache(cache_path, ttl=None)

        cls._write_cache(cache_path, catalogue)
        return catalogue

    @classmethod
    def _read_cache(cls, path: Path, *, ttl: int | None) -> Self | None:
        if not path.is_file():
            return None
        if ttl is not None and (time.time() - path.stat().st_mtime) > ttl:
            return None
        try:
            return cls.from_payload(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None

    @classmethod
    def _write_cache(cls, path: Path, catalogue: Self) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "data": [
                    {
                        "id": m.id,
                        "name": m.name,
                        "created": m.created,
                        "context_length": m.context_length,
                        "supported_parameters": sorted(m.supported_parameters),
                        "pricing": m.pricing,
                        "top_provider": {"max_completion_tokens": m.max_completion_tokens},
                    }
                    for m in catalogue.models
                ]
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:  # a read-only home should not break a run
            logger.debug("could not cache the model catalogue: %s", exc)


__all__ = ["CACHE_TTL_SECONDS", "MODELS_URL", "ModelCatalog", "ModelInfo"]
