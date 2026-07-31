"""Gemini adapter: Google models reached through the OpenRouter gateway.

The adapter exists so nothing above ``app.llm`` knows which vendor answered. Replacing
the gateway with the native Google SDK later is a change to this file alone.
"""

from typing import ClassVar

from openrouter.components import ResponseFormat

from app.llm.provider import (
    GenerationRequest,
    GenerationResult,
    Provider,
    ProviderAdapter,
    register,
)


@register
class GeminiAdapter(ProviderAdapter):
    """Google Gemini models.

    Quirk worth knowing: of the three families this is the least reliable at honouring a
    ``strict`` ``json_schema`` — it will happily return a near-miss shape, or wrap valid
    JSON in a markdown fence. We still ask for ``json_schema`` because it measurably
    improves the odds, and let the repair path in :mod:`app.llm.service` (fence stripping,
    then validation with a chained ``LLMError``) deal with what gets through.
    """

    provider: ClassVar[Provider] = Provider.GEMINI
    default_model: ClassVar[str] = "google/gemini-3.6-flash"

    def generate(
        self,
        request: GenerationRequest,
        *,
        response_format: ResponseFormat | None = None,
    ) -> GenerationResult:
        """Run ``request`` against Gemini. No provider-specific massaging is needed."""
        return self._send(request, response_format=response_format)


__all__ = ["GeminiAdapter"]
