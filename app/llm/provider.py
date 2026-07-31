"""Provider registry and the adapter contract every model call passes through.

Claude, GPT and Gemini are all reached through OpenRouter, this project's chosen gateway
and its only LLM dependency. The adapter boundary is deliberate: it is what makes a
native per-provider SDK droppable in later without touching a single call site, because
callers only ever speak :class:`GenerationRequest` / :class:`GenerationResult`.

``openrouter`` is imported here and in the rest of ``app.llm`` — nowhere else.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from time import perf_counter
from time import sleep as _sleep
from typing import Any, ClassVar, Final

import httpx
from openrouter import OpenRouter
from openrouter.components import (
    ChatAssistantMessage,
    ChatContentText,
    ChatMessages,
    ChatResult,
    ChatSystemMessage,
    ChatUserMessage,
    ProviderPreferences,
    ResponseFormat,
)
from openrouter.components.anthropiccachecontroldirective import (
    AnthropicCacheControlDirective,
)
from openrouter.types import UNSET
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.llm.client import get_client
from app.llm.messages import ChatMessage, Role
from app.utils.errors import ConfigurationError, LLMError
from app.utils.logging import get_logger

logger = get_logger(__name__)

_TRANSIENT_STATUSES: Final[frozenset[int]] = frozenset(
    {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
)
"""HTTP statuses worth retrying: rate limits, timeouts, and gateway hiccups.
Anything else — bad request, bad key, no credit — will fail identically on retry."""

_BACKOFF_BASE_SECONDS: Final[float] = 1.0
_BACKOFF_CAP_SECONDS: Final[float] = 30.0


class Provider(StrEnum):
    """The model families this application knows how to talk to."""

    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"


class GenerationRequest(BaseModel):
    """One generation call: the conversation plus per-call overrides.

    Extras are forbidden so a mistyped override fails here rather than being silently
    dropped on the way to the gateway. A ``None`` field means "let the provider decide",
    with ``model`` falling back to the adapter's ``default_model``.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    fallbacks: tuple[str, ...] = Field(
        default=(),
        description="Models the gateway should try, in order, if the primary is "
        "unavailable. Handled gateway-side, so the caller needs no retry loop.",
    )
    provider_sort: str | None = Field(
        default=None,
        description="How to pick among providers serving the model: price, throughput "
        "or latency. None leaves it to the gateway's default.",
    )
    cache_ttl: str | None = Field(
        default=None,
        description="Ask the provider to cache the static prompt prefix for this long "
        "('5m' or '1h'). None sends no directive, which is the right thing for "
        "providers that cache automatically.",
    )


class TokenUsage(BaseModel):
    """What one call consumed, as reported by the gateway.

    Every field is optional: not all providers report usage, and some omit cost.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    is_byok: bool = Field(
        default=False,
        description="Whether the call billed against your own provider key rather than "
        "gateway credit. Such a call reports zero cost here while still costing money "
        "elsewhere, so the two must not be read as the same thing.",
    )


class GenerationResult(BaseModel):
    """A completed call, flattened to the parts the application needs."""

    text: str
    model: str
    provider: Provider
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = None
    latency_ms: float | None = Field(
        default=None,
        description="Wall time for the gateway round trip. Measured at the single place "
        "the call is made, so it is comparable across providers.",
    )
    # The untouched gateway payload, kept for provenance and cost auditing.
    raw: dict[str, Any] | None = None


class ProviderAdapter(ABC):
    """One provider's slice of the gateway.

    Subclasses are thin on purpose: they declare which provider they are and which slug
    they default to, and delegate the actual call to :meth:`_send`. Everything that is
    the same for every vendor — request translation, response normalisation, error
    wrapping — lives here so there is exactly one place to change when the transport
    changes.
    """

    provider: ClassVar[Provider]
    """Which model family this adapter serves. Also its registry key."""

    default_model: ClassVar[str]
    """Gateway slug used when neither the caller nor settings name one."""

    def __init__(self, client: OpenRouter | None = None) -> None:
        """Build an adapter, optionally over an already-configured gateway client."""
        self._client = client

    @property
    def client(self) -> OpenRouter:
        """The gateway client, resolved on first use.

        Deferred so that constructing an adapter (and therefore an ``LLM``) needs no API
        key; only an actual generation does. That keeps `--help`, tests and dry runs
        working on a machine with no credentials.
        """
        if self._client is None:
            self._client = get_client()
        return self._client

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
        *,
        response_format: ResponseFormat | None = None,
    ) -> GenerationResult:
        """Run ``request`` against this provider and return the normalised reply.

        Args:
            request: Messages and per-call overrides.
            response_format: Optional output constraint (``text``, ``json_object`` or
                ``json_schema``). ``None`` leaves the reply unconstrained.

        Raises:
            LLMError: On any transport or protocol failure, or if the reply carries no
                usable text.
        """

    def _send(
        self,
        request: GenerationRequest,
        *,
        response_format: ResponseFormat | None = None,
    ) -> GenerationResult:
        """Run one non-streaming completion and normalise the reply.

        Shared by every adapter so the SDK is invoked in exactly one place.
        Transient failures — rate limits, timeouts, 5xx — are retried up to
        `LLM_RETRY_ATTEMPTS` times with exponential backoff, honouring a
        `Retry-After` when the gateway sends one. This sits *on top of* the
        gateway's own model failover: that handles "this model is down", this
        handles "the road there was briefly closed".
        """
        model = request.model or self.default_model
        messages = _to_sdk_messages(request.messages)
        # Resolved outside the try: a missing API key is a ConfigurationError the CLI
        # reports on its own terms, not a model failure.
        client = self.client
        attempts = 1 + get_settings().llm_retry_attempts
        started = perf_counter()
        for attempt in range(1, attempts + 1):
            try:
                result = client.chat.send(
                    model=model,
                    # The gateway treats `models` as the ordered fallback list *after*
                    # `model`, so the primary is deliberately not repeated here.
                    models=list(request.fallbacks) or None,
                    provider=_provider_preferences(request.provider_sort),
                    cache_control=_cache_control(request.cache_ttl),
                    messages=messages,
                    response_format=response_format,
                    # UNSET omits the field entirely; None would send an explicit null.
                    temperature=UNSET if request.temperature is None else request.temperature,
                    max_tokens=UNSET if request.max_tokens is None else request.max_tokens,
                    seed=UNSET if request.seed is None else request.seed,
                )
            except Exception as exc:
                if attempt < attempts and _is_transient(exc):
                    delay = _retry_delay(exc, attempt)
                    logger.warning(
                        "%s call to %s failed transiently (%s); retrying in %.0fs "
                        "(attempt %d of %d)",
                        self.provider,
                        model,
                        _describe(exc),
                        delay,
                        attempt,
                        attempts,
                    )
                    _sleep(delay)
                    continue
                attempted = " -> ".join((model, *request.fallbacks))
                raise LLMError(
                    f"{self.provider} call to {attempted} failed: {_describe(exc)}"
                ) from exc
            return GenerationResult(
                latency_ms=(perf_counter() - started) * 1000,
                text=_extract_text(result, provider=self.provider, model=model),
                model=result.model or model,
                provider=self.provider,
                usage=_to_usage(result),
                finish_reason=_finish_reason(result),
                raw=_dump(result),
            )
        raise LLMError(f"{self.provider} call to {model} made no attempt")  # pragma: no cover


ADAPTERS: dict[Provider, type[ProviderAdapter]] = {}
"""Registry of adapter classes, populated by ``register`` when ``app.llm.adapters`` loads."""


def register[AdapterT: ProviderAdapter](adapter: type[AdapterT]) -> type[AdapterT]:
    """Register ``adapter`` under its own ``provider`` and return it unchanged.

    Returned unchanged so it reads as a class decorator. Re-registering a provider
    overwrites it, which keeps a repeated import idempotent.
    """
    ADAPTERS[adapter.provider] = adapter
    return adapter


def get_adapter(provider: Provider) -> type[ProviderAdapter]:
    """Look up the adapter class for ``provider``.

    Raises:
        ConfigurationError: If no adapter is registered for ``provider``.
    """
    adapter = ADAPTERS.get(provider)
    if adapter is None:
        registered = ", ".join(available()) or "<none>"
        raise ConfigurationError(
            f"Unknown LLM provider {provider!r}. Registered providers: {registered}"
        )
    return adapter


def available() -> tuple[Provider, ...]:
    """Return the registered providers, sorted."""
    return tuple(sorted(ADAPTERS))


def _is_transient(exc: Exception) -> bool:
    """Whether a failure is worth retrying.

    SDK errors carry the HTTP status; anything without one is only transient if
    it is a transport-level problem (timeout, dropped connection). A schema or
    auth error would fail identically on every retry, so it fails now.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUSES
    return isinstance(exc, httpx.TimeoutException | httpx.TransportError)


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Seconds to wait before retry `attempt`, honouring a `Retry-After` when sent."""
    headers = getattr(exc, "headers", None)
    retry_after = headers.get("retry-after") if headers is not None else None
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), _BACKOFF_CAP_SECONDS)
        except ValueError:
            pass  # an HTTP-date Retry-After; the plain backoff below is close enough
    return min(_BACKOFF_BASE_SECONDS * (2.0 ** (attempt - 1)), _BACKOFF_CAP_SECONDS)


_SECRET_PATTERN = re.compile(
    r"(?:/keys/|/api/v\d+/keys/)[A-Za-z0-9_-]{16,}|\b[a-f0-9]{32,}\b|\bsk-[A-Za-z0-9_-]{8,}\b"
)


def _describe(exc: Exception) -> str:
    """The most informative account of a failure the SDK will give us.

    Gateway errors are often generic — "Provider returned error" — while the raw
    response carries the provider's actual complaint. Without the body, a 400 is
    indistinguishable from any other 400, and there is nothing to act on.
    """
    message = _redact(str(exc))
    body = getattr(exc, "body", None) or getattr(getattr(exc, "raw_response", None), "text", None)
    if not isinstance(body, str) or not body.strip():
        return message

    detail = _redact(" ".join(body.split()))[:400]
    # Only worth appending when it says more than the message already did.
    return message if detail in message else f"{message} | response: {detail}"


def _redact(text: str) -> str:
    """Strip key-shaped identifiers out of a provider's error text.

    Gateways quote your own key back at you in billing errors, and this text
    travels: into a stage outcome, out through `--json`, and from there into a bug
    report. Nothing downstream needs the identifier to act on the message.
    """
    return _SECRET_PATTERN.sub("[redacted]", text)


def _provider_preferences(sort: str | None) -> ProviderPreferences | None:
    """Gateway-side provider preferences, or None to leave them at the default."""
    if sort is None:
        return None
    return ProviderPreferences(sort=sort)


def _cache_control(ttl: str | None) -> AnthropicCacheControlDirective | None:
    """Prompt-cache directive, or None when the provider needs no asking.

    Only providers that bill a cache *write* require a breakpoint; the rest cache
    the prefix automatically and would reject or ignore the directive.
    """
    if ttl is None:
        return None
    return AnthropicCacheControlDirective(type="ephemeral", ttl=ttl)


def _to_sdk_messages(messages: Sequence[ChatMessage]) -> list[ChatMessages]:
    """Translate our narrow message type into the SDK's role-discriminated union."""
    out: list[ChatMessages] = []
    for message in messages:
        match message.role:
            case Role.SYSTEM:
                out.append(ChatSystemMessage(role="system", content=message.content))
            case Role.USER:
                out.append(ChatUserMessage(role="user", content=message.content))
            case Role.ASSISTANT:
                out.append(ChatAssistantMessage(role="assistant", content=message.content))
            case Role.TOOL:
                # A tool reply must carry the id of the call it answers, which ChatMessage
                # cannot express. Fail loudly rather than smuggle it through as a user turn.
                raise LLMError("Tool messages are not supported by this seam yet.")
    return out


def _extract_text(result: ChatResult, *, provider: Provider, model: str) -> str:
    """Flatten the assistant reply to plain text.

    ``content`` is a union of a plain string, a list of content parts, and nothing at
    all; only text parts carry something a caller can use.
    """
    if not result.choices:
        raise LLMError(f"{provider} returned no choices for {model!r}.")
    message = result.choices[0].message
    content = message.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(part.text for part in content if isinstance(part, ChatContentText))
    else:
        text = ""
    if not text.strip():
        refusal = message.refusal
        detail = f" Refusal: {refusal}" if isinstance(refusal, str) else ""
        raise LLMError(f"{provider} returned no text content for {model!r}.{detail}")
    return text


def _to_usage(result: ChatResult) -> TokenUsage:
    """Read token counts and cost off the reply, tolerating a provider that reports none."""
    usage = result.usage
    if usage is None:
        return TokenUsage()
    cost = usage.cost
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        # cost is optional-and-nullable in the SDK; anything that is not a number is unknown.
        cost=cost if isinstance(cost, float) else None,
        is_byok=bool(usage.is_byok),
    )


def _finish_reason(result: ChatResult) -> str | None:
    """Return the first choice's finish reason as a plain string."""
    if not result.choices:
        return None
    reason = result.choices[0].finish_reason
    return None if reason is None else str(reason)


def _dump(result: ChatResult) -> dict[str, Any] | None:
    """Best-effort JSON snapshot of the raw reply.

    Provenance is nice to have, not worth failing an otherwise-good call for, so an
    unserialisable payload degrades to ``None``.
    """
    try:
        return result.model_dump(mode="json")
    except Exception:
        return None


__all__ = [
    "ADAPTERS",
    "GenerationRequest",
    "GenerationResult",
    "Provider",
    "ProviderAdapter",
    "TokenUsage",
    "available",
    "get_adapter",
    "register",
]
