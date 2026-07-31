"""The single seam through which the rest of the application reaches an LLM.

Callers construct :class:`LLM`, name a provider (or let settings decide) and get back
markdown, a JSON object, or a validated Pydantic model. They never import a provider SDK
and never learn which vendor answered.

Every provider is reached through OpenRouter today; that is a fact about the adapters in
:mod:`app.llm.adapters`, not about this module, so dropping in a native per-provider SDK
later changes no call site.
"""

import json
from collections.abc import Sequence
from typing import Any, Final

from openrouter.components import (
    ChatFormatJSONObjectConfig,
    ChatFormatJSONSchemaConfig,
    ChatJSONSchemaConfig,
    ResponseFormat,
)
from pydantic import BaseModel, ValidationError

from app.config import get_settings

# Imported for the side effect: loading the adapter package is what fills ADAPTERS, so
# `LLM()` resolves a provider even when the caller reached this module directly.
from app.llm import adapters  # noqa: F401
from app.llm.messages import ChatMessage, Role
from app.llm.provider import (
    GenerationRequest,
    GenerationResult,
    Provider,
    ProviderAdapter,
    get_adapter,
)
from app.llm.roles import tier_for
from app.llm.routing import ModelRouter, ResolvedRoute, get_router, provider_for_model
from app.utils.errors import LLMError
from app.utils.ids import slugify
from app.utils.logging import get_logger

logger = get_logger(__name__)

_FENCE = "```"
_SCHEMA_NAME_LIMIT = 64
_ECHO_LIMIT = 400
# Deliberately crude: a real tokeniser differs per vendor and would make the budget
# check depend on which model was chosen. Four characters per token is the standard
# conservative approximation, and this guard only needs to catch order-of-magnitude
# mistakes before they reach the gateway.
_CHARS_PER_TOKEN = 4


class LLM:
    """Provider-agnostic entry point for every model call in the application."""

    def __init__(
        self,
        provider: Provider | None = None,
        *,
        model: str | None = None,
        adapter: ProviderAdapter | None = None,
        router: ModelRouter | None = None,
    ) -> None:
        """Resolve how this instance picks a model.

        There are two modes, and which one applies is decided here rather than per
        call. **Pinned**: an explicit `adapter`, `provider` or `model` fixes the
        destination, and routing is bypassed — this is the test seam and the
        `--model` escape hatch. **Routed**: nothing is pinned, so each call
        resolves its own model from the router by task name, which is what lets
        one stage use a cheap model and another an expensive one.

        Args:
            provider: Model family to pin to. Defaults to routed behaviour.
            model: Explicit gateway slug to pin to.
            adapter: A ready-made adapter, which wins over `provider`. Pass a fake
                and no network is touched.
            router: Routing table. Defaults to the one built from settings.
        """
        settings = get_settings()
        self._router = router if router is not None else get_router()
        self._pinned_adapter = adapter
        self._pinned = adapter is not None or provider is not None or model is not None

        # The default route only names a role; resolving it is what yields a slug.
        default_route = self._router.resolve(None)

        if adapter is not None:
            self.provider = adapter.provider
            self.model = model or adapter.default_model
        elif model is not None:
            # The vendor prefix decides the adapter, so the two cannot disagree.
            self.provider = provider or provider_for_model(model)
            self.model = model
        elif provider is not None:
            self.provider = provider
            # The default role's slug belongs to whichever vendor it resolved to;
            # handing it to another adapter would send a Claude slug to the GPT one.
            self.model = (
                default_route.model
                if default_route.provider is provider
                else get_adapter(provider).default_model
            )
        else:
            self.provider = default_route.provider
            self.model = default_route.model

        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.max_input_tokens = settings.max_input_tokens
        self.last_call: GenerationResult | None = None
        """The most recent generation, so a caller can report its cost and latency.

        Instance state rather than a return value because `generate_structured`
        returns the caller's model, and threading a usage object through every
        signature would put transport detail in every call site. The pipeline is
        sequential, so there is one call in flight per LLM at a time."""
        self._cache_prompts = settings.llm_cache_prompts
        self._cache_ttl = settings.llm_cache_ttl

    @property
    def router(self) -> ModelRouter:
        """The routing table this instance resolves against."""
        return self._router

    @property
    def is_pinned(self) -> bool:
        """Whether this instance bypasses routing because a destination was fixed."""
        return self._pinned

    def generate_markdown(
        self,
        messages: str | Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        task: str | None = None,
        seed: int | None = None,
    ) -> str:
        """Generate free-form text — typically markdown — and return it verbatim.

        No ``response_format`` is sent: omitting it is the one shape every provider on the
        gateway accepts, and asking for ``{"type": "text"}`` buys nothing.

        Args:
            messages: The conversation, or a bare prompt treated as one user turn.
            model: Per-call slug override.
            temperature: Per-call sampling override.
            max_tokens: Per-call output cap.
            task: Name of the work being done, usually a stage name. Selects the
                route; ignored when this instance is pinned to a model.
            seed: Per-call seed, for providers that support deterministic sampling.

        Raises:
            LLMError: On any transport or protocol failure, or an empty reply.
        """
        result = self._run(
            messages,
            response_format=None,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            task=task,
        )
        return result.text

    def generate_json(
        self,
        messages: str | Sequence[ChatMessage],
        *,
        schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        task: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON object and parse it.

        A ``schema`` upgrades the request from ``json_object`` to ``json_schema``, which
        every supported provider steers towards even when none of them guarantees it.
        Replies wrapped in a ```json fence are unwrapped before parsing, because that is
        the single most common way a model breaks its own JSON contract.

        Args:
            messages: The conversation, or a bare prompt treated as one user turn.
            schema: JSON Schema the reply should match. ``None`` asks only for valid JSON.
            schema_name: Name reported to the provider alongside ``schema``.
            model: Per-call slug override.
            temperature: Per-call sampling override.
            max_tokens: Per-call output cap.
            task: Name of the work being done, usually a stage name. Selects the
                route; ignored when this instance is pinned to a model.
            seed: Per-call seed, for providers that support deterministic sampling.

        A reply that stopped at the output-token cap (`finish_reason` "length")
        cannot be complete JSON, so it is not parsed at all: the call is retried
        once at double the cap, and if that is impossible or also truncated, the
        error names the real cause instead of a misleading "not valid JSON".

        Raises:
            LLMError: If the reply is not valid JSON, is JSON but not an object, or
                was truncated by the output cap with no headroom left to retry. The
                offending text is included, truncated, and the parse error is chained.
        """
        response_format = _json_response_format(schema, schema_name)
        result = self._run(
            messages,
            response_format=response_format,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            task=task,
        )
        if _truncated(result):
            result = self._retry_truncated(
                messages,
                response_format=response_format,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                task=task,
                first=result,
            )
        return _parse_json_object(result.text, provider=self.provider, model=result.model)

    def _retry_truncated(
        self,
        messages: str | Sequence[ChatMessage],
        *,
        response_format: ResponseFormat,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        seed: int | None,
        task: str | None,
        first: GenerationResult,
    ) -> GenerationResult:
        """One more attempt at the ceiling, after a truncated JSON reply.

        At the ceiling rather than at double the cap. A truncated reply says the
        answer is *larger* than the cap and nothing about how much larger, so
        doubling is a guess that can fail with headroom still unspent — and then
        report "raise MAX_OUTPUT_TOKENS" about a setting that was already high
        enough. `MAX_OUTPUT_TOKENS` is the most the operator has agreed to pay
        for one reply, so a retry that is going to happen at all should ask for
        exactly that and find out.
        """
        route = self.resolve(task, model)
        cap = _first_set(max_tokens, route.max_output_tokens, self.max_tokens)
        ceiling = get_settings().max_output_tokens
        if cap is None or ceiling <= cap:
            raise LLMError(_truncation_error(first, cap))
        bumped = ceiling
        logger.warning(
            "reply from %r was cut off at %s output tokens; retrying at the %s ceiling",
            first.model,
            f"{cap:,}",
            f"{bumped:,}",
        )
        retried = self._run(
            messages,
            response_format=response_format,
            model=model,
            temperature=temperature,
            max_tokens=bumped,
            seed=seed,
            task=task,
        )
        if _truncated(retried):
            raise LLMError(_truncation_error(retried, bumped))
        return retried

    def generate_structured[T: BaseModel](
        self,
        messages: str | Sequence[ChatMessage],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        task: str | None = None,
        seed: int | None = None,
    ) -> T:
        """Generate a reply constrained to ``schema`` and return it as that model.

        ``schema`` is the contract in both directions: its JSON Schema shapes the request
        and its validator checks the reply, so a payload model is declared exactly once
        with no hand-maintained schema alongside it.

        Args:
            messages: The conversation, or a bare prompt treated as one user turn.
            schema: Pydantic model describing the expected reply.
            model: Per-call slug override.
            temperature: Per-call sampling override.
            max_tokens: Per-call output cap.
            task: Name of the work being done, usually a stage name. Selects the
                route; ignored when this instance is pinned to a model.
            seed: Per-call seed, for providers that support deterministic sampling.

        Raises:
            LLMError: If the reply is not a JSON object, or does not validate against
                ``schema``. The underlying error is chained.
        """
        payload = self.generate_json(
            messages,
            schema=portable_schema(inline_refs(schema.model_json_schema())),
            schema_name=schema.__name__,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            task=task,
        )
        try:
            return schema.model_validate(payload)
        except ValidationError:
            # One retry against the known shape failure, before giving up.
            try:
                return schema.model_validate(_repair_stringified(payload))
            except ValidationError as exc:
                raise LLMError(
                    f"{self.provider} reply did not match {schema.__name__}: {exc}"
                ) from exc

    def _run(
        self,
        messages: str | Sequence[ChatMessage],
        *,
        response_format: ResponseFormat | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        seed: int | None,
        task: str | None = None,
    ) -> GenerationResult:
        """Resolve the destination, build the request, and run it.

        A pinned instance ignores `task` entirely; a routed one uses it to pick the
        model, its gateway fallbacks, and how providers are sorted.
        """
        route = self.resolve(task, model)
        prepared = _coerce_messages(messages)
        _check_input_budget(prepared, route, limit=self.max_input_tokens)

        request = GenerationRequest(
            messages=prepared,
            model=route.model,
            fallbacks=route.fallbacks,
            provider_sort=route.sort.value if route.sort else None,
            cache_ttl=self._cache_ttl_for(route),
            temperature=_first_set(temperature, route.temperature, self.temperature),
            max_tokens=_first_set(max_tokens, route.max_output_tokens, self.max_tokens),
            seed=seed,
        )
        result = self._adapter_for(route.model).generate(
            request,
            response_format=_downgrade_unsupported(response_format, route),
        )
        self.last_call = result
        return result

    def resolve(self, task: str | None = None, model: str | None = None) -> ResolvedRoute:
        """Where a call would go, including what the chosen model can do."""
        if self._pinned or model is not None:
            # An explicit destination is exactly that: no silent failover elsewhere.
            slug = model or self.model
            default = self._router.default
            return ResolvedRoute(
                task=task,
                capability=default.capability,
                tier=tier_for(default.capability),
                model=slug,
                provider=self.provider,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                # A pinned model is usually a fake or a one-off; assume the
                # permissive path rather than probing a catalogue for it.
                supports_structured_outputs=True,
                supports_response_format=True,
            )
        return self._router.resolve(task)

    def _cache_ttl_for(self, route: ResolvedRoute) -> str | None:
        """Whether to send a cache breakpoint for this call.

        Only sent to providers that bill a cache write and therefore need asking.
        Where caching is automatic, sending a directive buys nothing.
        """
        if not self._cache_prompts or not route.needs_explicit_cache_write:
            return None
        return self._cache_ttl

    def _adapter_for(self, slug: str) -> ProviderAdapter:
        """The adapter that serves a slug, reusing the pinned one when there is one."""
        if self._pinned_adapter is not None:
            return self._pinned_adapter
        return get_adapter(provider_for_model(slug))()


def get_llm() -> LLM:
    """Return an :class:`LLM` configured entirely from settings.

    Deliberately uncached: the object is cheap, and the expensive part — the gateway
    client and its connection pool — is already cached in :func:`app.llm.client.get_client`.
    """
    return LLM()


def _coerce_messages(messages: str | Sequence[ChatMessage]) -> list[ChatMessage]:
    """Accept a bare prompt string as shorthand for a single user turn."""
    if isinstance(messages, str):
        return [ChatMessage(role=Role.USER, content=messages)]
    if not messages:
        raise LLMError("Cannot generate from an empty message list.")
    return list(messages)


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve `$ref`/`$defs` into one self-contained schema.

    Pydantic factors nested models into `$defs` and points at them with `$ref`.
    Several providers' structured-output implementations do not resolve those, and
    fail in the worst way available: instead of rejecting the request they emit
    something schema-shaped but wrong — nested objects rendered as JSON strings,
    for instance. Inlining removes the ambiguity before it can be misread.

    Cycles are left as-is rather than expanded, since a self-referencing model
    cannot be inlined and expanding one would not terminate.
    """
    defs = schema.get("$defs", {})
    if not defs:
        return schema

    def resolve(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.removeprefix("#/$defs/")
                if name in seen or name not in defs:
                    return node
                expanded = resolve(defs[name], seen | {name})
                # Keep any sibling keys — `description` often sits next to a `$ref`.
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return {**expanded, **extra} if extra else expanded
            return {k: resolve(v, seen) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        return node

    inlined = resolve({k: v for k, v in schema.items() if k != "$defs"}, frozenset())
    return inlined if isinstance(inlined, dict) else schema


_NUMERIC_BOUNDS: Final[tuple[str, ...]] = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
)
_ARRAY_BOUNDS: Final[tuple[str, ...]] = ("minItems", "maxItems")


def portable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove schema keywords some providers reject, keeping their meaning.

    Anthropic's structured output refuses numeric bounds and array-length bounds
    outright — verified against the live API, which answers `For 'integer' type,
    property 'minimum' is not supported` with a 400 rather than ignoring it. Gemini
    accepts them. Rather than keep a per-provider table that will go stale, the
    lowest common denominator is sent to everyone.

    The constraint is not simply dropped: it is appended to the field's description,
    so the model is still told the range. Losing provider-side enforcement is
    survivable because the reply is validated against the Pydantic model regardless;
    losing the *hint* would just make invalid replies more likely.
    """

    def convert(node: Any) -> Any:
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node

        kind = node.get("type")
        unsupported = (
            _NUMERIC_BOUNDS
            if kind in ("integer", "number")
            else _ARRAY_BOUNDS
            if kind == "array"
            else ()
        )
        removed = {key: node[key] for key in unsupported if key in node}
        cleaned = {key: convert(value) for key, value in node.items() if key not in removed}

        if removed:
            note = ", ".join(f"{key} {value}" for key, value in removed.items())
            existing = cleaned.get("description")
            cleaned["description"] = f"{existing} ({note})" if existing else f"Constraint: {note}"
        return cleaned

    converted = convert(schema)
    return converted if isinstance(converted, dict) else schema


def _repair_stringified(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse list entries a model returned as JSON strings instead of objects.

    A known failure of several providers: given a schema with objects inside an
    array, they emit each object as a string. Repairing the shape is safe — the
    content is unchanged, and every guarantee that matters (the excerpt actually
    appearing in the source) is checked afterwards regardless.
    """
    repaired: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            repaired[key] = [_maybe_json(item) for item in value]
        else:
            repaired[key] = value
    return repaired


def _maybe_json(item: Any) -> Any:
    if not isinstance(item, str):
        return item
    try:
        parsed = json.loads(item)
    except ValueError:
        return item
    return parsed if isinstance(parsed, dict) else item


def _json_response_format(schema: dict[str, Any] | None, name: str) -> ResponseFormat:
    """Build the output constraint for a JSON call.

    ``strict`` is deliberately left unset. A schema derived from a Pydantic model does not
    carry the ``additionalProperties: false`` / all-required constraints that providers
    demand in strict mode, and a 400 from the gateway is worse than a reply we can parse,
    validate, and report on precisely.
    """
    if schema is None:
        return ChatFormatJSONObjectConfig(type="json_object")
    return ChatFormatJSONSchemaConfig(
        type="json_schema",
        json_schema=ChatJSONSchemaConfig(name=_schema_name(name), schema_=schema),
    )


def _schema_name(name: str) -> str:
    """Coerce ``name`` into the gateway's schema-name charset."""
    return slugify(name, max_length=_SCHEMA_NAME_LIMIT) or "response"


def _parse_json_object(text: str, *, provider: Provider, model: str) -> dict[str, Any]:
    """Parse a JSON object out of a model reply, tolerating a markdown code fence."""
    try:
        payload: object = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"{provider} reply from {model!r} was not valid JSON: {exc}. "
            f"Reply was: {_truncate(text)}"
        ) from exc
    if not isinstance(payload, dict):
        raise LLMError(
            f"{provider} reply from {model!r} was JSON but not an object "
            f"({type(payload).__name__}). Reply was: {_truncate(text)}"
        )
    return payload


def _strip_code_fence(text: str) -> str:
    """Return ``text`` without the ```json fence models like to wrap JSON in."""
    stripped = text.strip()
    if not stripped.startswith(_FENCE):
        return stripped
    body = stripped.removeprefix(_FENCE)
    # The opening fence may carry an info string ("json"); drop the rest of that line.
    newline = body.find("\n")
    if newline != -1:
        body = body[newline + 1 :]
    closing = body.rfind(_FENCE)
    if closing != -1:
        body = body[:closing]
    return body.strip()


def _truncate(text: str) -> str:
    """Collapse and clip a reply so it can be quoted safely inside an error message."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _ECHO_LIMIT:
        return repr(collapsed)
    return f"{collapsed[:_ECHO_LIMIT]!r}..."


def _truncated(result: GenerationResult) -> bool:
    """Whether the reply stopped because it hit the output-token cap."""
    return (result.finish_reason or "").lower() == "length"


def _truncation_error(result: GenerationResult, cap: int | None) -> str:
    """Say what was hit and what would actually move it.

    Both halves of the remedy matter. A reply is too long either because the cap
    is too low or because the stage was asked to say too much — for
    `collect-evidence`, thirty candidates from ten collectors is a great deal of
    selecting — and naming only the first sends the reader to raise a number
    that may already be as high as they want it.
    """
    where = f"the {cap:,}-token output cap" if cap is not None else "its output cap"
    return (
        f"{result.provider} reply from {result.model!r} was cut off at {where} "
        f"(finish_reason 'length'), so it cannot be complete JSON. Either raise "
        f"MAX_OUTPUT_TOKENS (and this stage's LLM_MAX_OUTPUT_TOKENS entry), or "
        f"give the stage less to answer about — COLLECTOR_LIMIT is the usual one."
    )


__all__ = ["LLM", "get_llm"]


def _first_set[T](*values: T | None) -> T | None:
    """The first value that was actually specified, in precedence order."""
    for value in values:
        if value is not None:
            return value
    return None


def estimate_tokens(messages: Sequence[ChatMessage]) -> int:
    """Rough token count for a conversation. Approximate by design — see `_CHARS_PER_TOKEN`."""
    return sum(len(message.content) for message in messages) // _CHARS_PER_TOKEN


def _check_input_budget(
    messages: Sequence[ChatMessage],
    route: ResolvedRoute,
    *,
    limit: int,
) -> None:
    """Refuse a call whose context exceeds the configured or model ceiling.

    Failing here rather than at the gateway keeps the error legible: it names the
    stage and the size, instead of surfacing as a truncated reply or a provider
    error hundreds of lines later.
    """
    estimated = estimate_tokens(messages)
    ceiling = min(limit, route.context_length) if route.context_length else limit
    if estimated <= ceiling:
        return

    where = f" for {route.task}" if route.task else ""
    raise LLMError(
        f"Context{where} is about {estimated:,} tokens, over the {ceiling:,} limit "
        f"for {route.model}. Pass fewer artifacts, or raise MAX_INPUT_TOKENS."
    )


def _downgrade_unsupported(
    response_format: ResponseFormat | None,
    route: ResolvedRoute,
) -> ResponseFormat | None:
    """Soften a response format the chosen model cannot honour.

    A model that takes `response_format` but not `structured_outputs` will reject
    a json_schema request outright, so it is stepped down to json_object — still
    valid JSON, just unvalidated by the provider. Our own Pydantic validation runs
    either way, so the guarantee is not lost, only the provider-side enforcement.
    """
    if response_format is None:
        return None
    if isinstance(response_format, ChatFormatJSONSchemaConfig):
        if route.supports_structured_outputs:
            return response_format
        if route.supports_response_format:
            return ChatFormatJSONObjectConfig(type="json_object")
        return None
    if not route.supports_response_format:
        return None
    return response_format
