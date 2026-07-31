"""GPT adapter with native OpenAI and OpenRouter transports.

The application keeps one provider-neutral request/result shape whichever road
the call takes. ``LLM_TRANSPORT=auto`` prefers the native Responses API when an
OpenAI key is configured, and otherwise preserves the OpenRouter path.
"""

from typing import ClassVar

from openrouter import OpenRouter
from openrouter.components import ResponseFormat

from app.config import get_settings
from app.llm.openai_client import OpenAIResponsesClient, get_openai_client
from app.llm.provider import (
    GenerationRequest,
    GenerationResult,
    Provider,
    ProviderAdapter,
    register,
)


@register
class GptAdapter(ProviderAdapter):
    """OpenAI GPT models.

    Quirk worth knowing: OpenAI enforces ``json_schema`` server-side and rejects a
    schema that is not strict-compatible (every property required, ``additionalProperties``
    false). That is why :mod:`app.llm.service` leaves ``strict`` unset on the schema it
    derives from a Pydantic model — a hard 400 from the provider is worse than a reply we
    can validate and report on ourselves.
    """

    provider: ClassVar[Provider] = Provider.GPT
    default_model: ClassVar[str] = "openai/gpt-5.5"

    def __init__(
        self,
        client: OpenRouter | None = None,
        *,
        openai_client: OpenAIResponsesClient | None = None,
    ) -> None:
        """Build the adapter with optional transport-specific test seams."""
        if client is not None and openai_client is not None:
            raise ValueError("Pass either client or openai_client, not both.")
        super().__init__(client=client)
        self._openai_client = openai_client

    def generate(
        self,
        request: GenerationRequest,
        *,
        response_format: ResponseFormat | None = None,
    ) -> GenerationResult:
        """Run ``request`` through the selected GPT transport."""
        if self._uses_native_openai():
            client = self._openai_client or get_openai_client()
            return client.generate(
                request,
                default_model=self.default_model,
                response_format=response_format,
            )
        return self._send(request, response_format=response_format)

    def _uses_native_openai(self) -> bool:
        """Resolve GPT's transport, with an injected client taking precedence."""
        if self._openai_client is not None:
            return True
        # Preserve the long-standing `client=` seam even when the host process
        # happens to carry an OPENAI_API_KEY.
        if self._client is not None:
            return False
        settings = get_settings()
        if settings.llm_transport == "openai":
            return True
        if settings.llm_transport == "openrouter":
            return False
        return settings.openai_api_key is not None


__all__ = ["GptAdapter"]
