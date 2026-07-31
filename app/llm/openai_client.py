"""Native OpenAI Responses API transport.

The rest of the application keeps OpenRouter-style, vendor-prefixed model slugs.
Only this transport removes ``openai/`` at the wire boundary, then restores the
prefix on the normalised result.
"""

import json
from functools import lru_cache
from time import perf_counter
from time import sleep as _sleep
from typing import Any, Final, cast

import httpx
from openrouter.components import ResponseFormat

from app.config import get_settings
from app.llm.messages import ChatMessage, Role
from app.llm.provider import (
    GenerationRequest,
    GenerationResult,
    Provider,
    TokenUsage,
    _describe,
    _is_transient,
    _redact,
    _retry_delay,
)
from app.utils.errors import ConfigurationError, LLMError
from app.utils.logging import get_logger

logger = get_logger(__name__)

_MISSING_API_KEY: Final = (
    "OPENAI_API_KEY is not set. Set it to use LLM_TRANSPORT=openai, or choose openrouter."
)
_PROVIDER_PREFIX: Final = "openai/"
_DEFAULT_BASE_URL: Final = "https://api.openai.com/v1"
_ERROR_DETAIL_LIMIT: Final = 400


class _OpenAIHTTPError(Exception):
    """HTTP failure carrying the fields shared retry/error helpers inspect."""

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"OpenAI Responses API returned HTTP {response.status_code}")
        self.status_code = response.status_code
        self.headers = response.headers
        self.body = response.text


class OpenAIResponsesClient:
    """Small synchronous client for ``POST /v1/responses``.

    ``http_client`` and ``transport`` are test seams. Production callers normally
    provide neither and reuse the process-wide instance returned by
    :func:`get_openai_client`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_ms: int = 120_000,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ConfigurationError(_MISSING_API_KEY)
        if http_client is not None and transport is not None:
            raise ValueError("Pass either http_client or transport, not both.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_ms / 1000
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(transport=transport)

    def close(self) -> None:
        """Close the connection pool when this instance owns it."""
        if self._owns_client:
            self._http.close()

    def generate(
        self,
        request: GenerationRequest,
        *,
        default_model: str,
        response_format: ResponseFormat | None = None,
    ) -> GenerationResult:
        """Execute one Responses API call and return the provider-neutral result."""
        internal_model = request.model or default_model
        api_model = _api_model(internal_model)
        payload = _request_payload(
            request,
            api_model=api_model,
            response_format=response_format,
        )
        attempts = 1 + get_settings().llm_retry_attempts
        started = perf_counter()

        for attempt in range(1, attempts + 1):
            try:
                raw = self._post(payload)
            except Exception as exc:
                if attempt < attempts and _is_transient(exc):
                    delay = _retry_delay(exc, attempt)
                    logger.warning(
                        "gpt call to %s via native OpenAI failed transiently (%s); "
                        "retrying in %.0fs (attempt %d of %d)",
                        internal_model,
                        _describe(exc),
                        delay,
                        attempt,
                        attempts,
                    )
                    _sleep(delay)
                    continue
                raise LLMError(
                    f"gpt call to {internal_model!r} via native OpenAI failed: {_describe(exc)}"
                ) from exc

            return _normalise_response(
                raw,
                requested_model=internal_model,
                latency_ms=(perf_counter() - started) * 1000,
            )

        raise LLMError(  # pragma: no cover - the loop always returns or raises
            f"gpt call to {internal_model!r} via native OpenAI made no attempt"
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one HTTP request and decode its JSON object response."""
        response = self._http.post(
            f"{self._base_url}/responses",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._timeout,
        )
        if response.is_error:
            raise _OpenAIHTTPError(response)
        try:
            decoded: object = response.json()
        except ValueError as exc:
            detail = _redact(" ".join(response.text.split()))[:_ERROR_DETAIL_LIMIT]
            suffix = f": {detail}" if detail else ""
            raise LLMError(f"OpenAI returned a non-JSON response{suffix}") from exc
        if not isinstance(decoded, dict):
            raise LLMError("OpenAI returned JSON that was not an object.")
        return cast(dict[str, Any], decoded)


# A concise alias for callers that do not need to distinguish this API from a
# future native OpenAI transport.
OpenAIClient = OpenAIResponsesClient


def build_openai_client(
    *,
    http_client: httpx.Client | None = None,
    transport: httpx.BaseTransport | None = None,
) -> OpenAIResponsesClient:
    """Build a native client from settings.

    Raises:
        ConfigurationError: If ``OPENAI_API_KEY`` is absent or blank.
    """
    settings = get_settings()
    if settings.openai_api_key is None:
        raise ConfigurationError(_MISSING_API_KEY)
    return OpenAIResponsesClient(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        timeout_ms=settings.openai_timeout_ms,
        http_client=http_client,
        transport=transport,
    )


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAIResponsesClient:
    """Return the process-wide native client so its connection pool is reused."""
    return build_openai_client()


def _request_payload(
    request: GenerationRequest,
    *,
    api_model: str,
    response_format: ResponseFormat | None,
) -> dict[str, Any]:
    """Translate the provider-neutral request to the Responses API shape."""
    payload: dict[str, Any] = {
        "model": api_model,
        "input": _to_responses_input(request.messages),
        # The application persists its own provenance; provider-side response
        # storage is unnecessary.
        "store": False,
    }
    if request.max_tokens is not None:
        payload["max_output_tokens"] = request.max_tokens
    if response_format is not None:
        payload["text"] = {"format": _responses_format(response_format)}

    # Responses support varies by GPT generation. In particular, reasoning
    # models reject sampling controls, so temperature and seed intentionally do
    # not cross this native boundary.
    return payload


def _to_responses_input(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Translate chat turns to Responses API input messages."""
    out: list[dict[str, str]] = []
    for message in messages:
        if message.role is Role.TOOL:
            raise LLMError("Tool messages are not supported by this seam yet.")
        out.append({"role": message.role.value, "content": message.content})
    return out


def _responses_format(response_format: ResponseFormat) -> dict[str, Any]:
    """Convert OpenRouter's response-format models to ``text.format``."""
    dumped = response_format.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )
    kind = dumped.get("type")
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        schema_config = dumped.get("json_schema")
        if not isinstance(schema_config, dict):
            raise LLMError("A json_schema response format is missing its schema configuration.")
        native: dict[str, Any] = {"type": "json_schema"}
        for field in ("name", "description", "schema", "strict"):
            if field in schema_config:
                native[field] = schema_config[field]
        if "name" not in native or "schema" not in native:
            raise LLMError("A json_schema response format must include a name and schema.")
        schema = native["schema"]
        if not isinstance(schema, dict):
            raise LLMError("A json_schema response format must contain an object schema.")
        native["schema"] = _openai_strict_schema(schema)
        native["strict"] = True
        return native
    if kind == "text":
        return {"type": "text"}
    raise LLMError(f"OpenAI does not support response format {kind!r}.")


def _openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a portable Pydantic schema valid for OpenAI Structured Outputs.

    OpenAI requires every property of every object to appear in ``required`` and
    requires ``additionalProperties: false``. Optional application fields remain
    optional in value by using Pydantic's existing nullable ``anyOf``; the key
    itself is still emitted. Pydantic defaults are validation metadata rather
    than an output-shape constraint and are unsupported by the provider, so they
    are removed only at this native transport boundary.
    """

    def convert(node: Any) -> Any:
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node

        cleaned = {key: convert(value) for key, value in node.items() if key != "default"}
        properties = cleaned.get("properties")
        if isinstance(properties, dict):
            cleaned["required"] = list(properties)
            cleaned["additionalProperties"] = False
        return cleaned

    converted = convert(schema)
    if not isinstance(converted, dict):  # pragma: no cover - root input is typed
        raise LLMError("OpenAI Structured Outputs requires an object schema.")
    return converted


def _normalise_response(
    raw: dict[str, Any],
    *,
    requested_model: str,
    latency_ms: float,
) -> GenerationResult:
    """Flatten a Responses API object to the application's generation result."""
    status = raw.get("status")
    if status in {"failed", "cancelled"}:
        detail = _response_error_detail(raw)
        suffix = f": {detail}" if detail else ""
        raise LLMError(f"OpenAI response ended with status {status!r}{suffix}")
    if status not in {None, "completed", "incomplete"}:
        raise LLMError(f"OpenAI returned unexpected response status {status!r}.")

    text, refusal = _extract_output_text(raw)
    if not text.strip():
        detail = f" Refusal: {_redact(refusal)}" if refusal else ""
        raise LLMError(
            f"gpt returned no text content for {_internal_model(requested_model)!r}.{detail}"
        )

    response_model = raw.get("model")
    model = response_model if isinstance(response_model, str) else requested_model
    return GenerationResult(
        text=text,
        model=_internal_model(model),
        provider=Provider.GPT,
        usage=_to_usage(raw.get("usage")),
        finish_reason=_finish_reason(raw),
        latency_ms=latency_ms,
        raw=raw,
    )


def _extract_output_text(raw: dict[str, Any]) -> tuple[str, str | None]:
    """Return concatenated output text and an optional refusal explanation."""
    direct = raw.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct, None

    texts: list[str] = []
    refusals: list[str] = []
    output = raw.get("output")
    if not isinstance(output, list):
        return "", None
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind in {"output_text", "text"} and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif kind == "refusal" and isinstance(part.get("refusal"), str):
                refusals.append(part["refusal"])
    return "".join(texts), " ".join(refusals) or None


def _to_usage(value: object) -> TokenUsage:
    """Map native input/output token names to the gateway-neutral names."""
    if not isinstance(value, dict):
        return TokenUsage(is_byok=True)
    return TokenUsage(
        prompt_tokens=_integer(value.get("input_tokens")),
        completion_tokens=_integer(value.get("output_tokens")),
        total_tokens=_integer(value.get("total_tokens")),
        cost=None,
        # A native call necessarily bills the operator's OpenAI account.
        is_byok=True,
    )


def _integer(value: object) -> int | None:
    """Return a JSON integer, excluding booleans."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _finish_reason(raw: dict[str, Any]) -> str | None:
    """Translate Responses completion status to the existing chat-style reason."""
    status = raw.get("status")
    if status == "completed":
        return "stop"
    if status != "incomplete":
        return None
    details = raw.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    if reason == "max_output_tokens":
        return "length"
    return reason if isinstance(reason, str) else "incomplete"


def _response_error_detail(raw: dict[str, Any]) -> str:
    """Extract a bounded, redacted explanation from a failed response."""
    error = raw.get("error")
    if isinstance(error, dict):
        selected = {
            key: value
            for key in ("message", "type", "code", "param")
            if (value := error.get(key)) is not None
        }
        detail = json.dumps(selected, ensure_ascii=False) if selected else str(error)
    elif error is None:
        detail = ""
    else:
        detail = str(error)
    return _redact(" ".join(detail.split()))[:_ERROR_DETAIL_LIMIT]


def _api_model(model: str) -> str:
    """Remove the internal OpenAI vendor prefix exactly once."""
    return model.removeprefix(_PROVIDER_PREFIX)


def _internal_model(model: str) -> str:
    """Ensure a native model name remains routable inside the application."""
    return model if model.startswith(_PROVIDER_PREFIX) else f"{_PROVIDER_PREFIX}{model}"


__all__ = [
    "OpenAIClient",
    "OpenAIResponsesClient",
    "build_openai_client",
    "get_openai_client",
]
